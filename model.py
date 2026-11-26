"""3-class action models (Linear, KNN, RandomForest, PyTorch MLP, PyTorch 1D CNN) + 20-Map Benchmark.

Every model maps the 28-dim observation (25 LiDAR beams + 3 goal features) to one of three driving
actions: 0 = Left, 1 = Straight, 2 = Right.
"""

from __future__ import annotations

import csv
import itertools
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from accelerator import (
    InferBackend,
    OpenVINORunner,
    compile_torch_to_openvino,
    detect_hardware,
)
from dataset import (
    NUM_ACTIONS,
    DrivingDataset,
    fingerprint_arrays,
    mirror_actions,
    mirror_observations,
)
from expert import baseline_action
from simulator import (
    NUM_BEAMS,
    OBS_DIM,
    Simulator,
    generate_random_map,
    validate_observation,
)

# Avoid OpenMP thread-oversubscription overhead on tiny 1D LiDAR tensors
torch.set_num_threads(1)

ModelType = Literal["linear", "knn", "rf", "mlp", "cnn1d"]

MODEL_TYPES: Tuple[ModelType, ...] = ("linear", "knn", "rf", "mlp", "cnn1d")
MODEL_LABELS: Dict[ModelType, str] = {
    "linear": "Linear",
    "knn": "KNN",
    "rf": "RandomForest",
    "mlp": "MLP (DL)",
    "cnn1d": "1D CNN (DL)",
}
# Display label of the map-free, non-learned LiDAR avoidance baseline (``expert.baseline_action``).
BASELINE_LABEL = "라이다 회피 기준선"
# Label older CSV exports used for the previous straight-to-goal baseline; recognized only when reading them.
LEGACY_BASELINE_LABEL = "Baseline (미학습)"

# Older checkpoints recorded their training objective under this key; only the classification objective
# is still supported. The key is assembled here so it is only ever read, never written.
_LEGACY_OBJECTIVE_KEY = "_".join(("task", "type"))
_SUPPORTED_LEGACY_OBJECTIVES = (None, "classification")


# ----------------------------------------------------------------------
# PyTorch Architectures (25 LiDAR + 3 Goal Features = 28 dims) -> 3 action logits
# ----------------------------------------------------------------------
class AvoidanceMLP(nn.Module):
    """MLP mapping [25 LiDAR + 3 Goal] features to raw action logits of shape (B, num_actions)."""

    def __init__(self, obs_dim: int = OBS_DIM, num_beams: int = NUM_BEAMS, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_beams = num_beams
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Avoidance1DCNN(nn.Module):
    """1D CNN over 25 LiDAR beams fused with 3 goal features; outputs raw action logits (B, num_actions)."""

    def __init__(self, obs_dim: int = OBS_DIM, num_beams: int = NUM_BEAMS, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_beams = num_beams
        self.goal_dim = obs_dim - num_beams
        self.num_actions = num_actions

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
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lidar = x[:, : self.num_beams].unsqueeze(1)  # (B, 1, 25)
        goal = x[:, self.num_beams :]                # (B, 3)
        lidar_feat = self.lidar_fc(self.conv(lidar))
        fused = torch.cat([lidar_feat, goal], dim=1)
        return self.head(fused)


# Estimator / network class each model type must load as (validated in ``load_checkpoint``).
_SKLEARN_CLASSES: Dict[str, type] = {
    "linear": LogisticRegression,
    "knn": KNeighborsClassifier,
    "rf": RandomForestClassifier,
}


def _build_net(model_type: ModelType, obs_dim: int, num_beams: int) -> nn.Module:
    if model_type == "mlp":
        return AvoidanceMLP(obs_dim=obs_dim, num_beams=num_beams)
    return Avoidance1DCNN(obs_dim=obs_dim, num_beams=num_beams)


# ----------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------
@dataclass
class TrainedModel:
    model_type: ModelType
    obs_dim: int
    num_beams: int
    backend: Literal["torch", "sklearn"]
    estimator: Any
    val_metric: float = 0.0   # validation accuracy in percent
    train_time: float = 0.0
    infer_ms: float = 0.0
    trained_device: str = "cpu"
    active_infer_backend: InferBackend = "torch_cpu"
    ov_runners: Dict[str, OpenVINORunner] = field(default_factory=dict)
    # Size and (Left, Straight, Right) counts of the dataset the model was trained on; 0 / () = unknown
    # (older checkpoints). Charts use them to tell whether they re-evaluate on the original holdout.
    num_samples: int = 0
    class_counts: Tuple[int, ...] = ()
    # Order-sensitive ``DrivingDataset.fingerprint()`` of the training dataset; None = unknown provenance
    # (older / migrated checkpoints). Only an equal fingerprint proves the chart re-evaluates the original holdout.
    dataset_fingerprint: Optional[str] = None


@dataclass
class TrainHistory:
    model_type: ModelType = "mlp"
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_metrics: List[float] = field(default_factory=list)  # validation accuracy (%) per epoch
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
    val_metric: Optional[float] = None  # validation accuracy (%), None for the baseline
    infer_ms: float = 0.0


def _checkpoint_ext(model_type: ModelType) -> str:
    return "pt" if model_type in ("mlp", "cnn1d") else "joblib"


def checkpoint_path_for(models_dir: Path, model_type: ModelType) -> Path:
    """Canonical checkpoint path: ``goal_{model_type}.pt`` (PyTorch) or ``.joblib`` (scikit-learn)."""
    return models_dir / f"goal_{model_type}.{_checkpoint_ext(model_type)}"


def legacy_checkpoint_path_for(models_dir: Path, model_type: ModelType) -> Path:
    """Checkpoint path used before the project became classification-only (``goal_{model_type}_cls.*``)."""
    return models_dir / f"goal_{model_type}_cls.{_checkpoint_ext(model_type)}"


def find_checkpoint(models_dir: Path, model_type: ModelType) -> Optional[Path]:
    """Return the canonical checkpoint if present, else the legacy ``_cls`` one, else None."""
    for path in (checkpoint_path_for(models_dir, model_type), legacy_checkpoint_path_for(models_dir, model_type)):
        if path.exists():
            return path
    return None


def _benchmark_inference_ms(trained: TrainedModel, sample_obs: np.ndarray, runs: int = 35) -> float:
    predict_action(trained, sample_obs)
    t0 = time.perf_counter()
    for _ in range(runs):
        predict_action(trained, sample_obs)
    return (time.perf_counter() - t0) * 1000.0 / runs


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_model(
    dataset: DrivingDataset,
    model_type: ModelType = "mlp",
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    mirror_augment: bool = True,
    class_balance: bool = True,
    train_device: str = "auto",
    save_path: Optional[Path] = None,
    on_epoch: Optional[Callable[[TrainHistory], None]] = None,
) -> Tuple[TrainedModel, TrainHistory]:
    """Train any of the 5 supported models as a 3-class action classifier on the 28-dim dataset."""
    x_all, y_all = dataset.to_arrays()
    n = len(x_all)
    if n < 20:
        raise ValueError(f"데이터가 부족합니다 (현재 {n}개, 최소 20개 이상 필요).")
    if x_all.shape[1] != dataset.obs_dim or not np.all(np.isfinite(x_all)):
        raise ValueError("학습 데이터에 NaN/Inf 또는 잘못된 차원의 관측값이 있습니다.")
    if len(y_all) and (y_all.min() < 0 or y_all.max() >= NUM_ACTIONS):
        raise ValueError(f"학습 데이터의 행동 라벨은 0..{NUM_ACTIONS - 1} 정수여야 합니다.")
    # Digest the exact arrays trained on now: ``on_epoch`` callers may mutate / reorder the live dataset later.
    dataset_fingerprint = fingerprint_arrays(x_all, y_all)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    val_size = max(1, int(n * 0.2))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    if mirror_augment:
        x_train = np.concatenate([x_train, mirror_observations(x_train, num_beams=dataset.num_beams)], axis=0)
        y_train = np.concatenate([y_train, mirror_actions(y_train)], axis=0)

    if len(np.unique(y_train)) < 2:
        raise ValueError("분류 학습에는 최소 2개 이상의 행동 클래스가 필요합니다.")

    t_start = time.perf_counter()

    if model_type in ("mlp", "cnn1d"):
        hw = detect_hardware()
        resolved_dev = hw.default_train_device if train_device == "auto" else train_device
        if resolved_dev == "cuda" and not hw.has_cuda:
            resolved_dev = "cpu"
        trained, history = _train_torch_model(
            model_type=model_type,
            obs_dim=dataset.obs_dim,
            num_beams=dataset.num_beams,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
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
            obs_dim=dataset.obs_dim,
            num_beams=dataset.num_beams,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            class_balance=class_balance,
            on_epoch=on_epoch,
        )

    train_time = time.perf_counter() - t_start
    infer_ms = _benchmark_inference_ms(trained, x_val[0])

    trained.train_time = train_time
    trained.infer_ms = infer_ms
    trained.num_samples = n
    trained.class_counts = tuple(int(c) for c in np.bincount(y_all, minlength=NUM_ACTIONS)[:NUM_ACTIONS])
    trained.dataset_fingerprint = dataset_fingerprint
    history.train_time = train_time
    history.infer_ms = infer_ms

    if save_path is not None:
        save_checkpoint(trained, save_path)

    return trained, history


def _balanced_class_weights(y: np.ndarray, num_classes: int = NUM_ACTIONS) -> np.ndarray:
    """``n / (num_classes * count)`` per class (sklearn "balanced"); classes absent from ``y`` get 1.0.

    Every weight is finite and strictly positive, so no loss ever divides 0 by 0.
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes)[:num_classes].astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = len(y) / (num_classes * counts[present])
    return weights.astype(np.float32)


def _train_torch_model(
    model_type: ModelType,
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
    net = _build_net(model_type, obs_dim, num_beams).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    y_train_t = torch.tensor(y_train, dtype=torch.long, device=dev)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=dev)
    if class_balance:
        weights = _balanced_class_weights(y_train)
        criterion: nn.Module = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev))
    else:
        criterion = nn.CrossEntropyLoss()
    # Validation loss is always unweighted: finite even when a validation class never appears in training,
    # and comparable across epochs and balance settings.
    val_criterion = nn.CrossEntropyLoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=dev)
    n_train = len(x_train_t)
    bs = min(batch_size, n_train)

    history = TrainHistory(
        model_type=model_type,
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
            loss = criterion(net(bx), by)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(bx)

        train_loss = running_loss / n_train

        net.eval()
        with torch.no_grad():
            val_logits = net(x_val_t)
            val_loss = float(val_criterion(val_logits, y_val_t).item())
            pred_cls = torch.argmax(val_logits, dim=1)
            val_metric = float((pred_cls == y_val_t).float().mean().item()) * 100.0

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

    est.fit(x_train, y_train)
    if hasattr(est, "n_jobs"):
        est.n_jobs = 1

    train_loss = 1.0 - float(np.mean(est.predict(x_train) == y_train))
    val_loss = 1.0 - float(np.mean(est.predict(x_val) == y_val))
    val_metric = (1.0 - val_loss) * 100.0

    history = TrainHistory(
        model_type=model_type,
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
    meta = {
        "model_type": trained.model_type,
        "obs_dim": trained.obs_dim,
        "num_beams": trained.num_beams,
        "val_metric": trained.val_metric,
        "train_time": trained.train_time,
        "infer_ms": trained.infer_ms,
        "num_samples": int(trained.num_samples),
        "class_counts": [int(c) for c in trained.class_counts],
        "dataset_fingerprint": trained.dataset_fingerprint,
    }
    if trained.backend == "torch":
        cpu_state = {k: v.detach().cpu() for k, v in trained.estimator.state_dict().items()}
        torch.save({**meta, "state_dict": cpu_state}, path)
    else:
        joblib.dump({**meta, "estimator": trained.estimator}, path)


def _reject_unsupported_objective(ckpt: Dict[str, Any], path: Path) -> None:
    objective = ckpt.get(_LEGACY_OBJECTIVE_KEY)
    if objective not in _SUPPORTED_LEGACY_OBJECTIVES:
        raise ValueError(f"{path.name}: unsupported legacy checkpoint objective {objective!r} (classification only)")


def _checkpoint_provenance(ckpt: Dict[str, Any]) -> Tuple[int, Tuple[int, ...]]:
    """(num_samples, class_counts) stored with a checkpoint; (0, ()) when absent or malformed."""
    try:
        num_samples = int(ckpt.get("num_samples", 0) or 0)
        counts = tuple(int(c) for c in (ckpt.get("class_counts") or ()))
    except (TypeError, ValueError):
        return 0, ()
    if num_samples < 0 or len(counts) not in (0, NUM_ACTIONS) or (counts and sum(counts) != num_samples):
        return 0, ()
    return num_samples, counts


def _checkpoint_fingerprint(ckpt: Dict[str, Any]) -> Optional[str]:
    """Training-dataset fingerprint stored with a checkpoint (64 hex chars); None when absent or malformed."""
    fp = ckpt.get("dataset_fingerprint")
    if isinstance(fp, str) and len(fp) == 64 and all(c in "0123456789abcdef" for c in fp):
        return fp
    return None


def _require_finite_parameters(tm: TrainedModel, path: Path) -> None:
    """Raise ValueError if any learned numeric parameter is NaN / Inf.

    The all-zero dry-run probe alone cannot see every corruption: a ``-inf`` intercept or hidden bias, or an
    infinite scaler ``scale_``, can still yield finite class scores for that one input.
    """
    def reject(name: str) -> None:
        raise ValueError(f"{path.name}: learned parameter {name!r} contains NaN / Inf")

    if tm.backend == "torch":
        net = tm.estimator
        for name, t in itertools.chain(net.named_parameters(), net.named_buffers()):
            if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                reject(name)
        return
    def check(step_name: str, step: Any) -> None:
        if hasattr(step, "steps"):  # a (possibly nested) Pipeline: check every inner step at every depth
            for inner_name, inner in step.steps:
                check(f"{step_name}.{inner_name}" if step_name else str(inner_name), inner)
            return
        if isinstance(step, LogisticRegression):
            attrs: Tuple[str, ...] = ("coef_", "intercept_")
        elif isinstance(step, StandardScaler):
            attrs = ("mean_", "scale_", "var_")
        else:
            return
        for attr in attrs:
            value = getattr(step, attr, None)
            if value is None:  # e.g. StandardScaler(with_mean=False) has mean_ = None
                continue
            arr = np.asarray(value)
            if arr.dtype.kind not in "fiu" or not np.all(np.isfinite(arr)):
                reject(f"{step_name}.{attr}")

    est = tm.estimator
    check("" if hasattr(est, "steps") else type(est).__name__, est)


def _finalize_loaded(
    tm: TrainedModel,
    path: Path,
    ckpt: Dict[str, Any],
    expected_model_type: Optional[ModelType],
) -> TrainedModel:
    """Check the requested identity, require finite learned parameters, and run a finite 28-feature dry-run
    prediction before handing a model out.

    The probe catches what static checks cannot: NaN / Inf weights or biases, and pipelines whose inner steps
    disagree on the feature count (e.g. a 28-feature scaler in front of a 2-feature classifier).
    """
    if expected_model_type is not None and tm.model_type != expected_model_type:
        raise ValueError(
            f"{path.name}: holds a {tm.model_type!r} checkpoint, expected {expected_model_type!r}"
        )
    tm.num_samples, tm.class_counts = _checkpoint_provenance(ckpt)
    tm.dataset_fingerprint = _checkpoint_fingerprint(ckpt)
    _require_finite_parameters(tm, path)
    try:
        predict_action(tm, np.zeros(OBS_DIM, dtype=np.float32))
    except Exception as exc:
        raise ValueError(f"{path.name}: dry-run prediction on a {OBS_DIM}-feature observation failed: {exc}") from exc
    return tm


def load_checkpoint(path: Path, expected_model_type: Optional[ModelType] = None) -> TrainedModel:
    """Load a 3-class action checkpoint (canonical or legacy ``_cls``); legacy non-classification ones raise.

    With ``expected_model_type`` the checkpoint must hold that model family. Every loaded model must also
    produce three finite class scores for an all-zero 28-feature observation, otherwise ValueError is raised.
    """
    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        _reject_unsupported_objective(ckpt, path)
        model_type: ModelType = ckpt["model_type"]
        obs_dim: int = ckpt.get("obs_dim", OBS_DIM)
        num_beams: int = ckpt.get("num_beams", NUM_BEAMS)
        if obs_dim != OBS_DIM:
            raise ValueError("Incompatible obs_dim")
        if model_type not in ("mlp", "cnn1d"):
            raise ValueError(f"{path.name}: unexpected PyTorch model type {model_type!r}")
        net = _build_net(model_type, obs_dim, num_beams)
        expected_net = AvoidanceMLP if model_type == "mlp" else Avoidance1DCNN
        if type(net) is not expected_net:  # pragma: no cover - guards _build_net drift
            raise ValueError(f"{path.name}: {model_type!r} built {type(net).__name__}, expected {expected_net.__name__}")
        try:
            net.load_state_dict(ckpt["state_dict"])
        except RuntimeError as exc:
            raise ValueError(f"{path.name}: weights do not match a {NUM_ACTIONS}-action network") from exc
        net.eval()
        tm = TrainedModel(
            model_type=model_type,
            obs_dim=obs_dim,
            num_beams=num_beams,
            backend="torch",
            estimator=net,
            val_metric=float(ckpt.get("val_metric", 0.0)),
            train_time=float(ckpt.get("train_time", 0.0)),
            infer_ms=float(ckpt.get("infer_ms", 0.0)),
        )
        return _finalize_loaded(tm, path, ckpt, expected_model_type)

    ckpt = joblib.load(path)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path.name}: not a checkpoint dictionary")
    _reject_unsupported_objective(ckpt, path)
    if ckpt.get("obs_dim", 0) != OBS_DIM:
        raise ValueError("Incompatible obs_dim")
    model_type = ckpt.get("model_type")
    if model_type not in _SKLEARN_CLASSES:
        raise ValueError(f"{path.name}: unexpected scikit-learn model type {model_type!r}")
    est = ckpt.get("estimator")
    if not is_classifier(est) or not hasattr(est, "predict_proba"):
        raise ValueError(f"{path.name}: estimator is not an action classifier")
    final = est.steps[-1][1] if isinstance(est, Pipeline) else est
    expected_cls = _SKLEARN_CLASSES[model_type]
    if not isinstance(final, expected_cls):
        raise ValueError(
            f"{path.name}: {model_type!r} checkpoint holds {type(final).__name__}, expected {expected_cls.__name__}"
        )
    n_features = getattr(est, "n_features_in_", None)
    if n_features != OBS_DIM:
        raise ValueError(f"{path.name}: estimator was fitted on {n_features} features, expected {OBS_DIM}")
    classes = np.asarray(getattr(final, "classes_", []))
    if (
        classes.ndim != 1
        or classes.size == 0
        or classes.dtype.kind not in "iu"
        or len(np.unique(classes)) != classes.size
        or not set(classes.tolist()) <= set(range(NUM_ACTIONS))
    ):
        raise ValueError(f"{path.name}: estimator classes {classes.tolist()} are not a subset of {{0, 1, 2}}")
    if hasattr(est, "n_jobs"):
        est.n_jobs = 1
    tm = TrainedModel(
        model_type=model_type,
        obs_dim=ckpt["obs_dim"],
        num_beams=ckpt.get("num_beams", NUM_BEAMS),
        backend="sklearn",
        estimator=est,
        val_metric=float(ckpt.get("val_metric", 0.0)),
        train_time=float(ckpt.get("train_time", 0.0)),
        infer_ms=float(ckpt.get("infer_ms", 0.0)),
    )
    return _finalize_loaded(tm, path, ckpt, expected_model_type)


def load_checkpoint_with_fallback(
    models_dir: Path,
    model_type: ModelType,
) -> Tuple[Optional[TrainedModel], List[str]]:
    """Load ``model_type`` from its canonical checkpoint, falling back to the legacy ``_cls`` sibling.

    Each candidate must hold exactly ``model_type`` and pass the dry-run probe of :func:`load_checkpoint`.
    Returns ``(model or None, errors)`` where ``errors`` lists ``"<file name>: <reason>"`` for every candidate
    that exists but failed to load (missing files are not errors).
    """
    errors: List[str] = []
    for path in (checkpoint_path_for(models_dir, model_type), legacy_checkpoint_path_for(models_dir, model_type)):
        if not path.exists():
            continue
        try:
            return load_checkpoint(path, expected_model_type=model_type), errors
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            errors.append(msg if msg.startswith(f"{path.name}:") else f"{path.name}: {msg}")
    return None, errors


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
    sample = np.zeros(trained.obs_dim, dtype=np.float32)
    if infer_backend in ("torch_cuda", "torch_cpu"):
        if infer_backend == "torch_cuda" and not hw.has_cuda:
            raise RuntimeError("CUDA GPU가 감지되지 않았습니다.")
        trained.estimator.to("cuda" if infer_backend == "torch_cuda" else "cpu").eval()
        trained.active_infer_backend = infer_backend
        # Re-measure so the reported latency never keeps a previous accelerator's timing
        trained.infer_ms = _benchmark_inference_ms(trained, sample, runs=30)
        return 0.0

    # OpenVINO backends: ov_cpu / ov_gpu / ov_npu
    ov_dev = infer_backend.split("_", 1)[1].upper()  # "CPU" | "GPU" | "NPU"
    if ov_dev not in hw.ov_devices:
        raise RuntimeError(f"OpenVINO 디바이스 '{ov_dev}'를 찾을 수 없습니다.")

    compile_ms = 0.0
    runner = trained.ov_runners.get(ov_dev)
    if runner is None:
        xml_path = export_dir / f"ov_{trained.model_type}.xml" if export_dir is not None else None
        runner = compile_torch_to_openvino(
            trained.estimator,
            obs_dim=trained.obs_dim,
            ov_device=ov_dev,
            export_xml_path=xml_path,
        )
        compile_ms = runner.compile_ms

    # Stage the runner on a throwaway view: ``trained`` only switches once a dry-run inference has produced three
    # finite logits and the latency benchmark has completed, so a failure leaves its verified backend serving.
    staged = replace(trained, active_infer_backend=infer_backend, ov_runners={ov_dev: runner})
    try:
        # Probe the runner itself: a None / malformed / non-finite result must never pass via the PyTorch path.
        _ov_logits(runner, validate_observation(sample, trained.obs_dim).reshape(1, -1), ov_dev)
        predict_action(staged, sample)
        infer_ms = _benchmark_inference_ms(staged, sample, runs=30)
    except Exception:
        if trained.ov_runners.get(ov_dev) is runner:  # a cached runner that stopped working is evicted, too
            del trained.ov_runners[ov_dev]
            if trained.active_infer_backend == infer_backend:
                trained.active_infer_backend = (
                    "torch_cuda" if next(trained.estimator.parameters()).device.type == "cuda" else "torch_cpu"
                )
                # Re-measure on the PyTorch fallback so the stale OpenVINO latency is not reported.
                trained.infer_ms = _benchmark_inference_ms(trained, sample, runs=30)
        raise

    trained.ov_runners[ov_dev] = runner
    trained.active_infer_backend = infer_backend
    trained.infer_ms = infer_ms  # measured on the selected accelerator
    return compile_ms


def _ov_logits(runner: Any, x_np: np.ndarray, ov_dev: str) -> np.ndarray:
    """Run an OpenVINO runner and require exactly three finite logits, shaped ``(1, 3)`` or ``(3,)``."""
    out = runner.predict(x_np)
    if out is None:
        raise RuntimeError(f"OpenVINO {ov_dev} runner returned no output")
    logits = np.asarray(out, dtype=np.float64)
    if logits.shape not in ((1, NUM_ACTIONS), (NUM_ACTIONS,)) or not np.all(np.isfinite(logits)):
        raise ValueError(
            f"OpenVINO {ov_dev} runner produced invalid logits (shape {logits.shape}, "
            f"expected {NUM_ACTIONS} finite values)"
        )
    return logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    out = np.asarray(logits, dtype=np.float64).reshape(-1)
    exp = np.exp(out - float(np.max(out)))
    return (exp / np.sum(exp)).astype(np.float32)


@torch.no_grad()
def predict_action(trained: TrainedModel, obs: np.ndarray) -> Tuple[int, np.ndarray]:
    """Single-frame inference on CPU, CUDA, or OpenVINO (CPU / GPU / Intel NPU).

    Returns ``(action, probs)``: the argmax action in {0, 1, 2} and float32 class probabilities of shape (3,).
    ``obs`` must be one finite observation of shape ``(obs_dim,)``; batches, wrong lengths and NaN / Inf raise
    ValueError, as does a model whose output is not three finite logits / probabilities.
    """
    x_np = validate_observation(obs, trained.obs_dim).reshape(1, -1)
    if trained.backend == "torch":
        logits: Optional[np.ndarray] = None
        ab = trained.active_infer_backend
        if ab.startswith("ov_"):
            ov_dev = ab.split("_", 1)[1].upper()
            runner = trained.ov_runners.get(ov_dev)
            if runner is not None:
                logits = _ov_logits(runner, x_np, ov_dev)  # a None result is an inference error, not a fallback
        if logits is None:
            dev = next(trained.estimator.parameters()).device
            x = torch.from_numpy(x_np).to(dev)
            logits = trained.estimator(x).detach().cpu().numpy()
        logits = np.asarray(logits, dtype=np.float64)
        if logits.size != NUM_ACTIONS or not np.all(np.isfinite(logits)):
            raise ValueError(f"model produced invalid logits (shape {logits.shape}, expected {NUM_ACTIONS} finite values)")
        probs = _softmax(logits)
    else:
        est = trained.estimator
        raw_probs = np.asarray(est.predict_proba(x_np)[0], dtype=np.float64)
        if not np.all(np.isfinite(raw_probs)):
            raise ValueError("model produced non-finite class probabilities")
        classes = est.classes_ if hasattr(est, "classes_") else est[-1].classes_
        probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for cls_idx, p in zip(classes, raw_probs):
            if 0 <= int(cls_idx) < NUM_ACTIONS:
                probs[int(cls_idx)] = float(p)
        total = float(probs.sum())
        probs = probs / total if total > 0 else np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32).reshape(NUM_ACTIONS)
    if not np.all(np.isfinite(probs)):
        raise ValueError("model produced non-finite class probabilities")
    return int(np.argmax(probs)), probs


# ----------------------------------------------------------------------
# Automated 20-Random-Map Quantitative Benchmark (LiDAR Avoidance Baseline vs All Models)
# ----------------------------------------------------------------------
def evaluate_models_on_random_maps(
    trained_models: Dict[ModelType, TrainedModel],
    num_maps: int = 20,
    seed_start: int = 1000,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[BenchmarkEntry]:
    """Evaluate the map-free LiDAR avoidance baseline + all trained models on identical unseen random maps.

    The baseline (``expert.baseline_action``) is a hand-written reactive controller that reads the same
    28-dim observation and chooses from the same three actions as the learned models, so the comparison
    isolates what learning adds.
    """
    maps = [generate_random_map(seed=seed_start + i, difficulty=12) for i in range(num_maps)]
    sim = Simulator(noise_std=0.0)
    dt = 1.0 / 30.0
    max_steps = 450  # 15 seconds per episode

    # Build list of contenders: LiDAR avoidance baseline first, then every trained model in canonical order
    contenders: List[Tuple[str, str, Optional[TrainedModel]]] = [("baseline", BASELINE_LABEL, None)]
    for m_type in MODEL_TYPES:
        tm = trained_models.get(m_type)
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
                action = baseline_action(ag.last_obs) if tm is None else predict_action(tm, ag.last_obs)[0]
                sim.step_action(ag, action, dt, add_noise=False)
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


BENCHMARK_CSV_HEADER = (
    "model", "success_rate_pct", "goals", "crashes", "timeouts", "avg_goal_time_s", "val_acc_pct", "infer_ms",
)


def export_benchmark_csv(entries: List[BenchmarkEntry], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BENCHMARK_CSV_HEADER)
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
