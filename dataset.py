"""Dataset storage, goal-aware L/R mirror augmentation, and CSV/NPZ I/O."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np

from simulator import MAX_ANGULAR_SPEED, MAX_LINEAR_SPEED, NUM_BEAMS, OBS_DIM


ACTION_LEFT = 0
ACTION_STRAIGHT = 1
ACTION_RIGHT = 2
ACTION_NAMES = ["좌회전 (L)", "직진 (S)", "우회전 (R)"]


def velocity_to_action(w: float, threshold: float = 0.28) -> int:
    """Map angular velocity w (screen coords: <0 left, >0 right) to 3-class label."""
    if w < -threshold:
        return ACTION_LEFT
    if w > threshold:
        return ACTION_RIGHT
    return ACTION_STRAIGHT


def action_to_velocity(action: int) -> Tuple[float, float]:
    """Map discrete class index back to (linear_v, angular_w)."""
    if action == ACTION_LEFT:
        return MAX_LINEAR_SPEED * 0.72, -MAX_ANGULAR_SPEED * 0.82
    if action == ACTION_RIGHT:
        return MAX_LINEAR_SPEED * 0.72, MAX_ANGULAR_SPEED * 0.82
    return MAX_LINEAR_SPEED * 0.92, 0.0


def mirror_observations(x: np.ndarray, num_beams: int = NUM_BEAMS) -> np.ndarray:
    """Mirror 28-dim obs [25 LiDAR, sin(goal_ang), cos(goal_ang), dist_norm] left<->right."""
    x_flip = x.copy()
    x_flip[:, :num_beams] = x[:, :num_beams][:, ::-1]
    if x.shape[1] > num_beams:
        # Negate sin(rel_goal_angle); cos and dist remain unchanged
        x_flip[:, num_beams] = -x[:, num_beams]
    return x_flip


class DrivingDataset:
    """In-memory buffer + disk persistence for (obs[28], v_norm, w_norm, action) samples."""

    def __init__(self, obs_dim: int = OBS_DIM, num_beams: int = NUM_BEAMS):
        self.obs_dim = obs_dim
        self.num_beams = num_beams
        self._obs: list[np.ndarray] = []
        self._vels: list[Tuple[float, float]] = []
        self._actions: list[int] = []

    def __len__(self) -> int:
        return len(self._obs)

    def add(self, obs: np.ndarray, v: float, w: float) -> None:
        """Add one sample (ignores stationary frames automatically)."""
        if abs(v) < 1e-3 and abs(w) < 1e-3:
            return
        arr = np.asarray(obs, dtype=np.float32).copy()
        if arr.shape != (self.obs_dim,):
            return
        self._obs.append(arr)
        self._vels.append((float(v / MAX_LINEAR_SPEED), float(w / MAX_ANGULAR_SPEED)))
        self._actions.append(velocity_to_action(w))

    def clear(self) -> None:
        self._obs.clear()
        self._vels.clear()
        self._actions.clear()

    def class_counts(self) -> Tuple[int, int, int]:
        if not self._actions:
            return (0, 0, 0)
        arr = np.array(self._actions, dtype=np.int64)
        return (
            int(np.sum(arr == ACTION_LEFT)),
            int(np.sum(arr == ACTION_STRAIGHT)),
            int(np.sum(arr == ACTION_RIGHT)),
        )

    def to_arrays(
        self, mirror_augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_obs, y_vels_norm, y_actions)."""
        if len(self) == 0:
            return (
                np.empty((0, self.obs_dim), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        x = np.stack(self._obs, axis=0).astype(np.float32)
        y_vel = np.array(self._vels, dtype=np.float32)
        y_act = np.array(self._actions, dtype=np.int64)

        if mirror_augment:
            x_flip = mirror_observations(x, num_beams=self.num_beams)
            y_vel_flip = y_vel.copy()
            y_vel_flip[:, 1] = -y_vel_flip[:, 1]
            y_act_flip = np.where(
                y_act == ACTION_LEFT,
                ACTION_RIGHT,
                np.where(y_act == ACTION_RIGHT, ACTION_LEFT, ACTION_STRAIGHT),
            )
            x = np.concatenate([x, x_flip], axis=0)
            y_vel = np.concatenate([y_vel, y_vel_flip], axis=0)
            y_act = np.concatenate([y_act, y_act_flip], axis=0)

        return x, y_vel, y_act

    def save(self, npz_path: Path, export_csv: bool = True) -> int:
        """Save dataset to .npz (and optional .csv for easy inspection)."""
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        x, y_vel, y_act = self.to_arrays(mirror_augment=False)
        np.savez_compressed(npz_path, obs=x, vels=y_vel, actions=y_act)

        if export_csv and len(x) > 0:
            csv_path = npz_path.with_suffix(".csv")
            header = (
                [f"beam_{i}" for i in range(self.num_beams)]
                + ["goal_sin", "goal_cos", "goal_dist", "v_norm", "w_norm", "action"]
            )
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(len(x)):
                    row = [f"{val:.4f}" for val in x[i]] + [
                        f"{y_vel[i, 0]:.4f}",
                        f"{y_vel[i, 1]:.4f}",
                        int(y_act[i]),
                    ]
                    writer.writerow(row)
        return len(x)

    def load(self, npz_path: Path) -> int:
        """Load dataset from .npz file (only if feature dimension matches current OBS_DIM)."""
        if not npz_path.exists():
            return 0
        data = np.load(npz_path)
        key = "obs" if "obs" in data else "scans"
        x = data[key].astype(np.float32)
        if x.ndim != 2 or x.shape[1] != self.obs_dim:
            return 0
        y_vel = data["vels"].astype(np.float32)
        y_act = data["actions"].astype(np.int64)

        self.clear()
        for i in range(len(x)):
            self._obs.append(x[i])
            self._vels.append((float(y_vel[i, 0]), float(y_vel[i, 1])))
            self._actions.append(int(y_act[i]))
        return len(self)
