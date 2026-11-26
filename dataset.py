"""Dataset storage for (28-dim observation, 3-class action) samples, L/R mirror augmentation, and CSV/NPZ I/O."""

from __future__ import annotations

import csv
import hashlib
import zipfile
import zlib
from pathlib import Path
from typing import Tuple

import numpy as np

from simulator import (
    ACTION_LEFT,
    ACTION_NAMES,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    NUM_ACTIONS,
    NUM_BEAMS,
    OBS_DIM,
    validate_action,
    validate_observation,
)

__all__ = [
    "ACTION_LEFT",
    "ACTION_STRAIGHT",
    "ACTION_RIGHT",
    "NUM_ACTIONS",
    "ACTION_NAMES",
    "DrivingDataset",
    "mirror_actions",
    "mirror_observations",
]


def fingerprint_arrays(x: np.ndarray, y_actions: np.ndarray) -> str:
    """Order-sensitive SHA-256 hex digest of ``(X_obs[N, obs_dim], y_actions[N])`` as float32 / int64, in order.

    Shared by :meth:`DrivingDataset.fingerprint` and ``model.train_model`` so a model records the digest of the
    exact arrays it trained on, even if the live dataset changes while training runs.
    """
    x = np.asarray(x)
    h = hashlib.sha256()
    h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(y_actions, dtype=np.int64).tobytes())
    return h.hexdigest()


def mirror_observations(x: np.ndarray, num_beams: int = NUM_BEAMS) -> np.ndarray:
    """Mirror 28-dim obs [25 LiDAR, sin(goal_ang), cos(goal_ang), dist_norm] left<->right."""
    x_flip = x.copy()
    x_flip[:, :num_beams] = x[:, :num_beams][:, ::-1]
    if x.shape[1] > num_beams:
        # Negate sin(rel_goal_angle); cos and dist remain unchanged
        x_flip[:, num_beams] = -x[:, num_beams]
    return x_flip


def mirror_actions(y: np.ndarray) -> np.ndarray:
    """Swap left <-> right action labels (straight unchanged), preserving dtype and shape."""
    y = np.asarray(y)
    out = y.copy()
    out[y == ACTION_LEFT] = ACTION_RIGHT
    out[y == ACTION_RIGHT] = ACTION_LEFT
    return out


class DrivingDataset:
    """In-memory buffer + disk persistence for (obs[28], action) samples."""

    def __init__(self, obs_dim: int = OBS_DIM, num_beams: int = NUM_BEAMS):
        self.obs_dim = obs_dim
        self.num_beams = num_beams
        self._obs: list[np.ndarray] = []
        self._actions: list[int] = []

    def __len__(self) -> int:
        return len(self._obs)

    def add(self, obs: np.ndarray, action: int) -> None:
        """Add one (observation, action) sample. Callers decide which frames to record.

        ``obs`` must be a finite 1D array of shape ``(obs_dim,)`` and ``action`` an integer ID in {0, 1, 2};
        anything else raises ValueError and leaves the buffer unchanged.
        """
        arr = validate_observation(obs, self.obs_dim).copy()
        a = validate_action(action)
        self._obs.append(arr)
        self._actions.append(a)

    def clear(self) -> None:
        self._obs.clear()
        self._actions.clear()

    def snapshot(self) -> "DrivingDataset":
        """Independent copy of the current samples, safe to hand to a background job."""
        clone = DrivingDataset(obs_dim=self.obs_dim, num_beams=self.num_beams)
        x, y_act = self.to_arrays()
        clone._obs = [row for row in x]  # rows are views into a freshly stacked copy
        clone._actions = [int(a) for a in y_act]
        return clone

    def class_counts(self) -> Tuple[int, int, int]:
        if not self._actions:
            return (0, 0, 0)
        counts = np.bincount(np.array(self._actions, dtype=np.int64), minlength=NUM_ACTIONS)
        return (int(counts[ACTION_LEFT]), int(counts[ACTION_STRAIGHT]), int(counts[ACTION_RIGHT]))

    def fingerprint(self) -> str:
        """Order-sensitive SHA-256 hex digest of the samples (float32 observations, int64 actions, in order).

        Two datasets share a fingerprint only when they hold the same rows in the same order, so a reordering
        that keeps the size and class counts still changes it (the 20% validation split depends on row order).
        """
        return fingerprint_arrays(*self.to_arrays())

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X_obs[N, obs_dim] float32, y_actions[N] int64)."""
        if len(self) == 0:
            return np.empty((0, self.obs_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)
        x = np.stack(self._obs, axis=0).astype(np.float32)
        y_act = np.array(self._actions, dtype=np.int64)
        return x, y_act

    def save(self, npz_path: Path, export_csv: bool = True) -> int:
        """Save dataset to .npz (keys: obs, actions) and an optional .csv for easy inspection."""
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        x, y_act = self.to_arrays()
        np.savez_compressed(npz_path, obs=x, actions=y_act)

        if export_csv:
            # Always rewrite the CSV (header only when empty) so it never keeps stale rows from an older save.
            csv_path = npz_path.with_suffix(".csv")
            header = (
                [f"beam_{i}" for i in range(self.num_beams)]
                + ["goal_sin", "goal_cos", "goal_dist", "action"]
            )
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(len(x)):
                    writer.writerow([f"{val:.4f}" for val in x[i]] + [int(y_act[i])])
        return len(x)

    def load(self, npz_path: Path) -> int:
        """Load dataset from .npz (only if the feature dimension matches and action labels exist).

        Older files may also carry a ``vels`` array; it is ignored.
        """
        if not npz_path.exists():
            return 0
        # Pickled (object-dtype) arrays stay disabled; unreadable / truncated / non-NPZ files load nothing and
        # leave the in-memory buffer untouched (it is only replaced after every check below has passed).
        try:
            with npz_path.open("rb") as fh:  # our handle, so it is closed even when np.load fails midway
                data = np.load(fh, allow_pickle=False)
                if not hasattr(data, "files"):  # a bare .npy array, not an NPZ archive
                    return 0
                with data:
                    key = "obs" if "obs" in data.files else "scans"
                    if key not in data.files or "actions" not in data.files:
                        return 0
                    raw_x = np.asarray(data[key])
                    raw_y = np.asarray(data["actions"])
        except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile, zlib.error):  # zlib: bad deflate data
            return 0
        # Only real-valued observations and integer action IDs are accepted; bool, float (even integral)
        # and string action arrays are rejected instead of being truncated into different labels.
        if raw_x.dtype.kind not in "fiu" or raw_y.dtype.kind not in "iu":
            return 0
        x = raw_x.astype(np.float32)
        y_act = raw_y.reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.obs_dim or len(x) != len(y_act):
            return 0
        if not np.all(np.isfinite(x)):
            return 0
        if len(y_act) and (y_act.min() < 0 or y_act.max() >= NUM_ACTIONS):
            return 0
        y_act = y_act.astype(np.int64)

        self.clear()
        self._obs.extend(x[i] for i in range(len(x)))
        self._actions.extend(int(a) for a in y_act)
        return len(self)
