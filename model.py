"""Multi-model ML & DL backend (Linear, KNN, RandomForest, PyTorch MLP, PyTorch 1D CNN) + 20-Map Benchmark."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from accelerator import (
    InferBackend,
    OpenVINORunner,
    compile_torch_to_openvino,
    detect_hardware,
)
from dataset import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    DrivingDataset,
    action_to_velocity,
    mirror_observations,
)
from expert import baseline_control
from simulator import (
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    NUM_BEAMS,
    NUM_GOAL_FEATS,
    OBS_DIM,
    Simulator,
    generate_random_map,
)

# Avoid OpenMP thread-oversubscription overhead on tiny 1D LiDAR tensors
torch.set_num_threads(1)

TaskType = Literal["classification", "regression"]
ModelType = Literal["linear", "knn", "rf", "mlp", "cnn1d"]

MODEL_TYPES: Tuple[ModelType, ...] = ("linear", "knn", "rf", "mlp", "cnn1d")
MODEL_LABELS: Dict[ModelType, str] = {
    "linear": "Linear",
    "knn": "KNN",
    "rf": "RandomForest",
    "mlp": "MLP (DL)",
    "cnn1d": "1D CNN (DL)",
}
# Display label of the map-free, non-learned LiDAR avoidance baseline (``expert.baseline_control``).
BASELINE_LABEL = "라이다 회피 기준선"
# Label older CSV exports used for the previous straight-to-goal baseline; recognized only when reading them.
LEGACY_BASELINE_LABEL = "Baseline (미학습)"


# ----------------------------------------------------------------------
# PyTorch Architectures (25 LiDAR + 3 Goal Features = 28 dims)
# ----------------------------------------------------------------------
class AvoidanceMLP(nn.Module):
    """MLP mapping [25 LiDAR + 3 Goal] features to driving commands."""

    def __init__(self, obs_dim: int = OBS_DIM, num_beams: int = NUM_BEAMS, task_type: TaskType = "regression"):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_beams = num_beams
        self.task_type = task_type
        out_dim = 3 if task_type == "classification" else 2

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.task_type == "regression":
            out = torch.tanh(out)
        return out


class Avoidance1DCNN(nn.Module):
    """1D CNN extracting spatial patterns from 25 LiDAR beams, fused with 3 Goal features."""

    def __init__(self, obs_dim: int = OBS_DIM, num_beams: int = NUM_BEAMS, task_type: TaskType = "regression"):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_beams = num_beams
        self.goal_dim = obs_dim - num_beams
        self.task_type = task_type
        out_dim = 3 if task_type == "classification" else 2

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lidar_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * num_beams, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + self.goal_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lidar = x[:, : self.num_beams].unsqueeze(1)  # (B, 1, 25)
        goal = x[:, self.num_beams :]                # (B, 3)
        lidar_feat = self.lidar_fc(self.conv(lidar))
        fused = torch.cat([lidar_feat, goal], dim=1)
        out = self.head(fused)
        if self.task_type == "regression":
            out = torch.tanh(out)
        return out


# ----------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------
@dataclass
class TrainedModel:
    model_type: ModelType
    task_type: TaskType
    obs_dim: int
    num_beams: int
    backend: Literal["torch", "sklearn"]
    estimator: Any
    val_metric: float = 0.0
    train_time: float = 0.0
    infer_ms: float = 0.0
    trained_device: str = "cpu"
    active_infer_backend: InferBackend = "torch_cpu"
    ov_runners: Dict[str, OpenVINORunner] = field(default_factory=dict)


@dataclass
class TrainHistory:
    model_type: ModelType = "mlp"
    task_type: TaskType = "regression"
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_metrics: List[float] = field(default_factory=list)
    best_val_metric: float = 0.0
    total_epochs: int = 0
    current_epoch: int = 0
    num_train_samples: int = 0
    num_val_samples: int = 0
    train_time: float = 0.0
    infer_ms: float = 0.0


@dataclass
class BenchmarkEntry:
    key: str              # "baseline" or ModelType
    label: str            # Display name
    num_maps: int
    goals: int
    crashes: int
    timeouts: int
    success_rate: float   # 0..100
    avg_time: float       # seconds (for successful runs)
    val_metric: Optional[float] = None
    infer_ms: float = 0.0


def checkpoint_path_for(models_dir: Path, model_type: ModelType, task_type: TaskType) -> Path:
    ext = "pt" if model_type in ("mlp", "cnn1d") else "joblib"
    t_tag = "cls" if task_type == "classification" else "reg"
    return models_dir / f"goal_{model_type}_{t_tag}.{ext}"


def _apply_mirror_numpy(
    x: np.ndarray, y_vel: np.ndarray, y_act: np.ndarray, num_beams: int = NUM_BEAMS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_flip = mirror_observations(x, num_beams=num_beams)
    y_vel_flip = y_vel.copy()
    y_vel_flip[:, 1] = -y_vel_flip[:, 1]
    y_act_flip = np.where(
        y_act == ACTION_LEFT,
        ACTION_RIGHT,
        np.where(y_act == ACTION_RIGHT, ACTION_LEFT, ACTION_STRAIGHT),
    )
    return (
        np.concatenate([x, x_flip], axis=0),
        np.concatenate([y_vel, y_vel_flip], axis=0),
        np.concatenate([y_act, y_act_flip], axis=0),
    )


def _benchmark_inference_ms(trained: TrainedModel, sample_obs: np.ndarray, runs: int = 35) -> float:
    predict_command(trained, sample_obs)
    t0 = time.perf_counter()
    for _ in range(runs):
        predict_command(trained, sample_obs)
    return (time.perf_counter() - t0) * 1000.0 / runs


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_model(
    dataset: DrivingDataset,
    model_type: ModelType = "mlp",
    task_type: TaskType = "regression",
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    mirror_augment: bool = True,
    class_balance: bool = True,
    train_device: str = "auto",
    save_path: Optional[Path] = None,
    on_epoch: Optional[Callable[[TrainHistory], None]] = None,
) -> Tuple[TrainedModel, TrainHistory]:
    """Train any of the 5 supported ML/DL models on the 28-dim goal-directed dataset."""
    x_all, y_vel_all, y_act_all = dataset.to_arrays(mirror_augment=False)
    n = len(x_all)
    if n < 20:
        raise ValueError(f"데이터가 부족합니다 (현재 {n}개, 최소 20개 이상 필요).")

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    val_size = max(1, int(n * 0.2))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_train, y_vel_train, y_act_train = x_all[train_idx], y_vel_all[train_idx], y_act_all[train_idx]
    x_val, y_vel_val, y_act_val = x_all[val_idx], y_vel_all[val_idx], y_act_all[val_idx]

    if mirror_augment:
        x_train, y_vel_train, y_act_train = _apply_mirror_numpy(
            x_train, y_vel_train, y_act_train, num_beams=dataset.num_beams
        )

    if task_type == "classification" and len(np.unique(y_act_train)) < 2:
        raise ValueError("분류 학습에는 최소 2개 이상의 행동 클래스가 필요합니다.")

    t_start = time.perf_counter()

    if model_type in ("mlp", "cnn1d"):
        hw = detect_hardware()
        resolved_dev = hw.default_train_device if train_device == "auto" else train_device
        if resolved_dev == "cuda" and not hw.has_cuda:
            resolved_dev = "cpu"
        trained, history = _train_torch_model(
            model_type=model_type,
            task_type=task_type,
            obs_dim=dataset.obs_dim,
            num_beams=dataset.num_beams,
            x_train=x_train,
            y_train=y_act_train if task_type == "classification" else y_vel_train,
            x_val=x_val,
            y_val=y_act_val if task_type == "classification" else y_vel_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            class_balance=class_balance,
            device_str=resolved_dev,
            on_epoch=on_epoch,
        )
    else:
        trained, history = _train_sklearn_model(
            model_type=model_type,
            task_type=task_type,
            obs_dim=dataset.obs_dim,
            num_beams=dataset.num_beams,
            x_train=x_train,
            y_train=y_act_train if task_type == "classification" else y_vel_train,
            x_val=x_val,
            y_val=y_act_val if task_type == "classification" else y_vel_val,
            class_balance=class_balance,
            on_epoch=on_epoch,
        )

    train_time = time.perf_counter() - t_start
    infer_ms = _benchmark_inference_ms(trained, x_val[0])

    trained.train_time = train_time
    trained.infer_ms = infer_ms
    history.train_time = train_time
    history.infer_ms = infer_ms

    if save_path is not None:
        save_checkpoint(trained, save_path)

    return trained, history


def _train_torch_model(
    model_type: ModelType,
    task_type: TaskType,
    obs_dim: int,
    num_beams: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    class_balance: bool,
    device_str: str,
    on_epoch: Optional[Callable[[TrainHistory], None]],
) -> Tuple[TrainedModel, TrainHistory]:
    torch.manual_seed(42)
    dev = torch.device(device_str)
    net: nn.Module = (
        AvoidanceMLP(obs_dim=obs_dim, num_beams=num_beams, task_type=task_type)
        if model_type == "mlp"
        else Avoidance1DCNN(obs_dim=obs_dim, num_beams=num_beams, task_type=task_type)
    ).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if task_type == "classification":
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=dev)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=dev)
        if class_balance:
            counts = np.bincount(y_train, minlength=3).astype(np.float32)
            weights = np.where(counts > 0, len(y_train) / (3.0 * np.maximum(counts, 1.0)), 0.0)
            criterion: nn.Module = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32, device=dev)
            )
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=dev)
        criterion = nn.MSELoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=dev)
    n_train = len(x_train_t)
    bs = min(batch_size, n_train)

    history = TrainHistory(
        model_type=model_type,
        task_type=task_type,
        total_epochs=epochs,
        num_train_samples=n_train,
        num_val_samples=len(x_val_t),
    )

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n_train, device=dev)
        running_loss = 0.0
        for start in range(0, n_train, bs):
            idx = perm[start : start + bs]
            bx = x_train_t[idx]
            by = y_train_t[idx]
            optimizer.zero_grad()
            preds = net(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(bx)

        train_loss = running_loss / n_train

        net.eval()
        with torch.no_grad():
            val_preds = net(x_val_t)
            val_loss = float(criterion(val_preds, y_val_t).item())
            if task_type == "classification":
                pred_cls = torch.argmax(val_preds, dim=1)
                val_metric = float((pred_cls == y_val_t).float().mean().item()) * 100.0
            else:
                val_metric = float(torch.abs(val_preds - y_val_t).mean().item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history.best_val_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        history.current_epoch = epoch
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.val_metrics.append(val_metric)

        if on_epoch is not None:
            on_epoch(history)

    # Keep canonical weights on CPU (can switch to CUDA or compile to OpenVINO NPU/GPU anytime)
    net = net.to("cpu")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    trained = TrainedModel(
        model_type=model_type,
        task_type=task_type,
        obs_dim=obs_dim,
        num_beams=num_beams,
        backend="torch",
        estimator=net,
        val_metric=history.best_val_metric,
        trained_device=device_str,
        active_infer_backend="torch_cuda" if device_str == "cuda" else "torch_cpu",
    )
    if device_str == "cuda":
        net.to("cuda")
    return trained, history


def _train_sklearn_model(
    model_type: ModelType,
    task_type: TaskType,
    obs_dim: int,
    num_beams: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_balance: bool,
    on_epoch: Optional[Callable[[TrainHistory], None]],
) -> Tuple[TrainedModel, TrainHistory]:
    cw = "balanced" if class_balance else None

    if task_type == "classification":
        if model_type == "linear":
            est: Any = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500, class_weight=cw, random_state=42)),
                ]
            )
        elif model_type == "knn":
            est = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        KNeighborsClassifier(
                            n_neighbors=min(7, len(x_train)),
                            weights="distance",
                            algorithm="kd_tree",
                            n_jobs=1,
                        ),
                    ),
                ]
            )
        else:  # rf
            est = RandomForestClassifier(
                n_estimators=60,
                max_depth=16,
                class_weight=cw,
                random_state=42,
                n_jobs=-1,
            )
    else:
        if model_type == "linear":
            est = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=1.0)),
                ]
            )
        elif model_type == "knn":
            est = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "reg",
                        KNeighborsRegressor(
                            n_neighbors=min(7, len(x_train)),
                            weights="distance",
                            algorithm="kd_tree",
                            n_jobs=1,
                        ),
                    ),
                ]
            )
        else:  # rf
            est = RandomForestRegressor(
                n_estimators=60,
                max_depth=16,
                random_state=42,
                n_jobs=-1,
            )

    est.fit(x_train, y_train)
    if hasattr(est, "n_jobs"):
        est.n_jobs = 1

    if task_type == "classification":
        train_pred = est.predict(x_train)
        val_pred = est.predict(x_val)
        train_loss = 1.0 - float(np.mean(train_pred == y_train))
        val_loss = 1.0 - float(np.mean(val_pred == y_val))
        val_metric = (1.0 - val_loss) * 100.0
    else:
        train_pred = est.predict(x_train)
        val_pred = est.predict(x_val)
        train_loss = float(np.mean((train_pred - y_train) ** 2))
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        val_metric = float(np.mean(np.abs(val_pred - y_val)))

    history = TrainHistory(
        model_type=model_type,
        task_type=task_type,
        train_losses=[train_loss],
        val_losses=[val_loss],
        val_metrics=[val_metric],
        best_val_metric=val_metric,
        total_epochs=1,
        current_epoch=1,
        num_train_samples=len(x_train),
        num_val_samples=len(x_val),
    )
    if on_epoch is not None:
        on_epoch(history)

    trained = TrainedModel(
        model_type=model_type,
        task_type=task_type,
        obs_dim=obs_dim,
        num_beams=num_beams,
        backend="sklearn",
        estimator=est,
        val_metric=val_metric,
    )
    return trained, history


# ----------------------------------------------------------------------
# Save / Load / Predict
# ----------------------------------------------------------------------
def save_checkpoint(trained: TrainedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if trained.backend == "torch":
        cpu_state = {k: v.detach().cpu() for k, v in trained.estimator.state_dict().items()}
        torch.save(
            {
                "model_type": trained.model_type,
                "task_type": trained.task_type,
                "obs_dim": trained.obs_dim,
                "num_beams": trained.num_beams,
                "state_dict": cpu_state,
                "val_metric": trained.val_metric,
                "train_time": trained.train_time,
                "infer_ms": trained.infer_ms,
            },
            path,
        )
    else:
        joblib.dump(
            {
                "model_type": trained.model_type,
                "task_type": trained.task_type,
                "obs_dim": trained.obs_dim,
                "num_beams": trained.num_beams,
                "estimator": trained.estimator,
                "val_metric": trained.val_metric,
                "train_time": trained.train_time,
                "infer_ms": trained.infer_ms,
            },
            path,
        )


def load_checkpoint(path: Path) -> TrainedModel:
    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model_type: ModelType = ckpt["model_type"]
        task_type: TaskType = ckpt["task_type"]
        obs_dim: int = ckpt.get("obs_dim", OBS_DIM)
        num_beams: int = ckpt.get("num_beams", NUM_BEAMS)
        if obs_dim != OBS_DIM:
            raise ValueError("Incompatible obs_dim")
        net: nn.Module = (
            AvoidanceMLP(obs_dim=obs_dim, num_beams=num_beams, task_type=task_type)
            if model_type == "mlp"
            else Avoidance1DCNN(obs_dim=obs_dim, num_beams=num_beams, task_type=task_type)
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return TrainedModel(
            model_type=model_type,
            task_type=task_type,
            obs_dim=obs_dim,
            num_beams=num_beams,
            backend="torch",
            estimator=net,
            val_metric=float(ckpt.get("val_metric", 0.0)),
            train_time=float(ckpt.get("train_time", 0.0)),
            infer_ms=float(ckpt.get("infer_ms", 0.0)),
        )
    else:
        ckpt = joblib.load(path)
        if ckpt.get("obs_dim", 0) != OBS_DIM:
            raise ValueError("Incompatible obs_dim")
        est = ckpt["estimator"]
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
        return TrainedModel(
            model_type=ckpt["model_type"],
            task_type=ckpt["task_type"],
            obs_dim=ckpt["obs_dim"],
            num_beams=ckpt["num_beams"],
            backend="sklearn",
            estimator=est,
            val_metric=float(ckpt.get("val_metric", 0.0)),
            train_time=float(ckpt.get("train_time", 0.0)),
            infer_ms=float(ckpt.get("infer_ms", 0.0)),
        )


def prepare_infer_backend(
    trained: TrainedModel,
    infer_backend: InferBackend,
    export_dir: Optional[Path] = None,
) -> float:
    """Switch a PyTorch TrainedModel to target accelerator (CPU / CUDA / OpenVINO CPU, GPU, NPU).

    Returns compile/switch time in ms (0.0 if already cached).
    """
    if trained.backend != "torch":
        return 0.0

    hw = detect_hardware()
    if infer_backend == "torch_cuda":
        if not hw.has_cuda:
            raise RuntimeError("CUDA GPU가 감지되지 않았습니다.")
        trained.estimator.to("cuda").eval()
        trained.active_infer_backend = "torch_cuda"
        return 0.0

    if infer_backend == "torch_cpu":
        trained.estimator.to("cpu").eval()
        trained.active_infer_backend = "torch_cpu"
        return 0.0

    # OpenVINO backends: ov_cpu / ov_gpu / ov_npu
    ov_dev = infer_backend.split("_", 1)[1].upper()  # "CPU" | "GPU" | "NPU"
    if ov_dev not in hw.ov_devices:
        raise RuntimeError(f"OpenVINO 디바이스 '{ov_dev}'를 찾을 수 없습니다.")

    compile_ms = 0.0
    if ov_dev not in trained.ov_runners:
        xml_path = None
        if export_dir is not None:
            t_tag = "cls" if trained.task_type == "classification" else "reg"
            xml_path = export_dir / f"ov_{trained.model_type}_{t_tag}.xml"
        runner = compile_torch_to_openvino(
            trained.estimator,
            obs_dim=trained.obs_dim,
            ov_device=ov_dev,
            export_xml_path=xml_path,
        )
        trained.ov_runners[ov_dev] = runner
        compile_ms = runner.compile_ms

    trained.active_infer_backend = infer_backend
    # Refresh measured inference latency on the selected accelerator
    sample = np.zeros(trained.obs_dim, dtype=np.float32)
    trained.infer_ms = _benchmark_inference_ms(trained, sample, runs=30)
    return compile_ms


def _raw_array_to_command(
    raw_out: np.ndarray, task_type: TaskType
) -> Tuple[float, float, Dict[str, object]]:
    out = raw_out.reshape(-1)
    if task_type == "classification":
        shifted = out - float(np.max(out))
        exp = np.exp(shifted)
        probs = (exp / np.sum(exp)).astype(np.float32)
        action = int(np.argmax(probs))
        v, w = action_to_velocity(action)
        return v, w, {"task": "classification", "action": action, "probs": probs}
    else:
        v_norm = float(out[0])
        w_norm = float(out[1])
        v = float(np.clip(v_norm, 0.25, 1.0)) * MAX_LINEAR_SPEED
        w = float(np.clip(w_norm, -1.0, 1.0)) * MAX_ANGULAR_SPEED
        return v, w, {"task": "regression", "v_norm": v_norm, "w_norm": w_norm}


@torch.no_grad()
def predict_command(trained: TrainedModel, obs: np.ndarray) -> Tuple[float, float, Dict[str, object]]:
    """Run single-frame inference on CPU, CUDA, or OpenVINO (CPU / GPU / Intel NPU)."""
    if trained.backend == "torch":
        ab = trained.active_infer_backend
        if ab.startswith("ov_"):
            ov_dev = ab.split("_", 1)[1].upper()
            runner = trained.ov_runners.get(ov_dev)
            if runner is not None:
                x_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                out_np = runner.predict(x_np)
                return _raw_array_to_command(out_np, trained.task_type)

        dev = next(trained.estimator.parameters()).device
        x = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        out_np = trained.estimator(x).squeeze(0).detach().cpu().numpy()
        return _raw_array_to_command(out_np, trained.task_type)
    else:
        x_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        est = trained.estimator
        if trained.task_type == "classification":
            probs = np.zeros(3, dtype=np.float32)
            raw_probs = est.predict_proba(x_np)[0]
            classes = est.classes_ if hasattr(est, "classes_") else est[-1].classes_
            for cls_idx, p in zip(classes, raw_probs):
                if 0 <= int(cls_idx) < 3:
                    probs[int(cls_idx)] = float(p)
            action = int(np.argmax(probs))
            v, w = action_to_velocity(action)
            return v, w, {"task": "classification", "action": action, "probs": probs}
        else:
            pred = est.predict(x_np)[0]
            v_norm = float(np.clip(pred[0], -1.0, 1.0))
            w_norm = float(np.clip(pred[1], -1.0, 1.0))
            v = float(np.clip(v_norm, 0.25, 1.0)) * MAX_LINEAR_SPEED
            w = float(np.clip(w_norm, -1.0, 1.0)) * MAX_ANGULAR_SPEED
            return v, w, {"task": "regression", "v_norm": v_norm, "w_norm": w_norm}


# ----------------------------------------------------------------------
# Automated 20-Random-Map Quantitative Benchmark (LiDAR Avoidance Baseline vs All Models)
# ----------------------------------------------------------------------
def evaluate_models_on_random_maps(
    trained_models: Dict[Tuple[ModelType, TaskType], TrainedModel],
    task_type: TaskType,
    num_maps: int = 20,
    seed_start: int = 1000,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[BenchmarkEntry]:
    """Evaluate the map-free LiDAR avoidance baseline + all trained models on identical unseen random maps.

    The baseline (``expert.baseline_control``) is a hand-written reactive controller that reads the same
    28-dim observation as the learned models, so the comparison isolates what learning adds.
    """
    maps = [generate_random_map(seed=seed_start + i, difficulty=12) for i in range(num_maps)]
    sim = Simulator(noise_std=0.0)
    dt = 1.0 / 30.0
    max_steps = 450  # 15 seconds per episode

    # Build list of contenders: LiDAR avoidance baseline first, then every trained model in canonical order
    contenders: List[Tuple[str, str, Optional[TrainedModel]]] = [("baseline", BASELINE_LABEL, None)]
    for m_type in MODEL_TYPES:
        tm = trained_models.get((m_type, task_type))
        if tm is not None:
            contenders.append((m_type, MODEL_LABELS[m_type], tm))

    results: List[BenchmarkEntry] = []
    total_runs = len(contenders) * num_maps
    done_runs = 0

    for key, label, tm in contenders:
        goals = 0
        crashes = 0
        timeouts = 0
        goal_times: List[float] = []

        for cfg in maps:
            sim.set_map_config(cfg)
            ag = sim.spawn_robot_state()
            for _ in range(max_steps):
                if tm is None:
                    v, w = baseline_control(ag.last_obs)
                else:
                    v, w, _ = predict_command(tm, ag.last_obs)
                sim.step_agent(ag, v, w, dt, add_noise=False)
                if ag.reached_goal or ag.collided:
                    break

            if ag.reached_goal:
                goals += 1
                goal_times.append(ag.elapsed)
            elif ag.collided:
                crashes += 1
            else:
                timeouts += 1

            done_runs += 1
            if on_progress is not None:
                on_progress(done_runs, total_runs, label)

        avg_t = sum(goal_times) / len(goal_times) if goal_times else 0.0
        results.append(
            BenchmarkEntry(
                key=key,
                label=label,
                num_maps=num_maps,
                goals=goals,
                crashes=crashes,
                timeouts=timeouts,
                success_rate=(100.0 * goals / num_maps),
                avg_time=avg_t,
                val_metric=tm.val_metric if tm is not None else None,
                infer_ms=tm.infer_ms if tm is not None else 0.01,
            )
        )

    return results


def export_benchmark_csv(entries: List[BenchmarkEntry], csv_path: Path, task_type: TaskType) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metric_col = "val_acc_pct" if task_type == "classification" else "val_mae"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "success_rate_pct", "goals", "crashes", "timeouts", "avg_goal_time_s", metric_col, "infer_ms"])
        for e in entries:
            writer.writerow(
                [
                    e.label,
                    f"{e.success_rate:.1f}",
                    e.goals,
                    e.crashes,
                    e.timeouts,
                    f"{e.avg_time:.2f}",
                    f"{e.val_metric:.4f}" if e.val_metric is not None else "-",
                    f"{e.infer_ms:.3f}",
                ]
            )
