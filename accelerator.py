"""Hardware discovery & inference acceleration for PyTorch (CPU/CUDA) and OpenVINO (CPU/GPU/Intel NPU)."""

from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import openvino as ov

    HAS_OPENVINO = True
except Exception:
    ov = None  # type: ignore[assignment]
    HAS_OPENVINO = False


InferBackend = Literal["torch_cpu", "torch_cuda", "ov_cpu", "ov_gpu", "ov_npu"]

BACKEND_LABELS: Dict[InferBackend, str] = {
    "torch_cpu": "PyTorch CPU",
    "torch_cuda": "CUDA GPU",
    "ov_cpu": "OV CPU",
    "ov_gpu": "OV GPU",
    "ov_npu": "OV NPU",
}


@dataclass
class HardwareInfo:
    has_cuda: bool
    cuda_name: str
    has_openvino: bool
    ov_devices: List[str]
    ov_device_names: Dict[str, str]

    @property
    def has_npu(self) -> bool:
        return "NPU" in self.ov_devices

    @property
    def has_ov_gpu(self) -> bool:
        return "GPU" in self.ov_devices

    @property
    def default_train_device(self) -> str:
        return "cuda" if self.has_cuda else "cpu"

    def available_infer_backends(self) -> List[InferBackend]:
        backends: List[InferBackend] = ["torch_cpu"]
        if self.has_cuda:
            backends.append("torch_cuda")
        if self.has_openvino:
            if "CPU" in self.ov_devices:
                backends.append("ov_cpu")
            if "GPU" in self.ov_devices:
                backends.append("ov_gpu")
            if "NPU" in self.ov_devices:
                backends.append("ov_npu")
        return backends

    def summary_text(self) -> str:
        parts = []
        if self.has_cuda:
            parts.append(f"CUDA: {self.cuda_name}")
        else:
            parts.append("CUDA: 없음")
        if self.has_npu:
            parts.append(f"NPU: {self.ov_device_names.get('NPU', 'Intel NPU')}")
        else:
            parts.append("NPU: 없음")
        if self.has_ov_gpu:
            parts.append("iGPU: OV 지원")
        return "  |  ".join(parts)


_HW_CACHE: Optional[HardwareInfo] = None
_OV_CORE: Optional[object] = None


def _get_ov_core():
    global _OV_CORE
    if not HAS_OPENVINO:
        return None
    if _OV_CORE is None:
        _OV_CORE = ov.Core()
    return _OV_CORE


def detect_hardware() -> HardwareInfo:
    """Detect available CUDA GPU and OpenVINO (CPU / Intel GPU / Intel NPU) accelerators."""
    global _HW_CACHE
    if _HW_CACHE is not None:
        return _HW_CACHE

    has_cuda = bool(torch.cuda.is_available())
    cuda_name = torch.cuda.get_device_name(0) if has_cuda else ""

    ov_devices: List[str] = []
    ov_names: Dict[str, str] = {}
    if HAS_OPENVINO:
        core = _get_ov_core()
        if core is not None:
            ov_devices = list(core.available_devices)
            for dev in ov_devices:
                try:
                    ov_names[dev] = str(core.get_property(dev, "FULL_DEVICE_NAME"))
                except Exception:
                    ov_names[dev] = dev

    _HW_CACHE = HardwareInfo(
        has_cuda=has_cuda,
        cuda_name=cuda_name,
        has_openvino=HAS_OPENVINO,
        ov_devices=ov_devices,
        ov_device_names=ov_names,
    )
    return _HW_CACHE


class OpenVINORunner:
    """Compiled OpenVINO InferRequest wrapper for fast CPU / GPU / Intel NPU inference.

    One InferRequest cannot run two inferences at once ("Infer Request is busy"), so ``predict`` serializes
    callers with a per-runner lock; the UI race, a benchmark thread, and latency probes can share a runner.
    """

    def __init__(self, compiled_model, device: str, compile_ms: float):
        self.compiled_model = compiled_model
        self.device = device
        self.compile_ms = compile_ms
        self.infer_request = compiled_model.create_infer_request()
        self._lock = threading.Lock()

    def predict(self, x_np: np.ndarray) -> np.ndarray:
        """Run inference on a (1, obs_dim) float32 array; returns raw action logits of shape (1, 3)."""
        x = np.ascontiguousarray(x_np, dtype=np.float32)
        with self._lock:
            res = self.infer_request.infer({0: x})
            # Copy while holding the lock: the output tensor is reused by the next inference.
            return np.array(res[0], dtype=np.float32, copy=True)


def compile_torch_to_openvino(
    net: nn.Module,
    obs_dim: int,
    ov_device: str,
    export_xml_path: Optional[Path] = None,
) -> OpenVINORunner:
    """Convert a PyTorch nn.Module to OpenVINO IR and compile for target device (CPU / GPU / NPU)."""
    if not HAS_OPENVINO:
        raise RuntimeError("OpenVINO 패키지가 설치되어 있지 않습니다.")
    core = _get_ov_core()
    assert core is not None

    # Tracing/conversion needs the model on CPU in eval mode. Remember where it lived so that every exit path,
    # including a failed conversion / IR save / compile, moves it back (a CUDA model is never left on CPU).
    orig_device = next(net.parameters()).device
    try:
        net_cpu = net.to("cpu").eval()
        dummy = torch.zeros(1, obs_dim, dtype=torch.float32)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ov_model = ov.convert_model(net_cpu, example_input=dummy, input=([1, obs_dim], ov.Type.f32))

        if export_xml_path is not None:
            export_xml_path.parent.mkdir(parents=True, exist_ok=True)
            ov.save_model(ov_model, str(export_xml_path))

        compiled = core.compile_model(ov_model, ov_device)
        compile_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        if orig_device.type != "cpu":
            net.to(orig_device)

    return OpenVINORunner(compiled_model=compiled, device=ov_device, compile_ms=compile_ms)


def benchmark_torch_vs_accelerators(
    net: nn.Module,
    obs_dim: int,
    runs: int = 80,
) -> List[Tuple[InferBackend, str, float, float]]:
    """Benchmark a PyTorch model across all available backends.

    Returns list of (backend_key, label, infer_ms, max_abs_diff_vs_fp32).
    """
    hw = detect_hardware()
    x_np = np.random.rand(1, obs_dim).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    net_cpu = net.to("cpu").eval()
    with torch.no_grad():
        ref = net_cpu(x_t).numpy()

    results: List[Tuple[InferBackend, str, float, float]] = []

    # 1. PyTorch CPU
    with torch.no_grad():
        net_cpu(x_t)
        t0 = time.perf_counter()
        for _ in range(runs):
            out = net_cpu(x_t).numpy()
        ms = (time.perf_counter() - t0) * 1000.0 / runs
    results.append(("torch_cpu", BACKEND_LABELS["torch_cpu"], ms, 0.0))

    # 2. PyTorch CUDA (if available)
    if hw.has_cuda:
        net_cuda = net.to("cuda").eval()
        x_cuda = x_t.to("cuda")
        with torch.no_grad():
            net_cuda(x_cuda)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                out_c = net_cuda(x_cuda)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / runs
            diff = float(np.max(np.abs(out_c.cpu().numpy() - ref)))
        net.to("cpu")
        results.append(("torch_cuda", BACKEND_LABELS["torch_cuda"], ms, diff))

    # 3. OpenVINO devices (CPU / GPU / NPU)
    dev_map: List[Tuple[str, InferBackend]] = [("CPU", "ov_cpu"), ("GPU", "ov_gpu"), ("NPU", "ov_npu")]
    for ov_dev, b_key in dev_map:
        if ov_dev not in hw.ov_devices:
            continue
        try:
            runner = compile_torch_to_openvino(net_cpu, obs_dim=obs_dim, ov_device=ov_dev)
            out = runner.predict(x_np)
            t0 = time.perf_counter()
            for _ in range(runs):
                out = runner.predict(x_np)
            ms = (time.perf_counter() - t0) * 1000.0 / runs
            diff = float(np.max(np.abs(out - ref)))
            results.append((b_key, BACKEND_LABELS[b_key], ms, diff))
        except Exception:
            pass

    return results
