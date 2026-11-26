"""2D Goal-Directed Robot & LiDAR Simulator with Guaranteed-Reachable Random Obstacle Maps."""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


WORLD_WIDTH = 1080
WORLD_HEIGHT = 960
WALL_MARGIN = 26

NUM_BEAMS = 25           # Odd count -> includes exact 0-deg forward beam
NUM_GOAL_FEATS = 3       # [sin(rel_goal_angle), cos(rel_goal_angle), norm_dist]
OBS_DIM = NUM_BEAMS + NUM_GOAL_FEATS  # 28

FOV_RAD = math.pi        # 180 degrees (-90 deg to +90 deg)
MAX_LIDAR_RANGE = 260.0  # pixels
MAX_GOAL_DIST_NORM = 700.0

ROBOT_RADIUS = 16.0
GOAL_RADIUS = 30.0
MAX_LINEAR_SPEED = 200.0  # px/s  (internal physics limit)
MAX_ANGULAR_SPEED = 2.9   # rad/s (internal physics limit)

# Discrete driving actions: the only interface between policies (baseline, models, keyboard) and physics.
ACTION_LEFT = 0
ACTION_STRAIGHT = 1
ACTION_RIGHT = 2
NUM_ACTIONS = 3
ACTION_NAMES: Tuple[str, str, str] = ("좌회전", "직진", "우회전")



def validate_action(action: object) -> int:
    """Return ``action`` as a plain int if it is an integer action ID in {0, 1, 2}, else raise ValueError.

    Only ``int`` / ``np.integer`` values are accepted: booleans, floats (even ``1.0``), strings and other
    types are rejected so that a fractional or truthy value can never silently become a driving label.
    """
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
        raise ValueError(f"invalid action {action!r}; expected an integer in 0..{NUM_ACTIONS - 1}")
    value = int(action)
    if not 0 <= value < NUM_ACTIONS:
        raise ValueError(f"invalid action {action!r}; expected an integer in 0..{NUM_ACTIONS - 1}")
    return value


def validate_observation(obs: object, dim: int = OBS_DIM) -> np.ndarray:
    """Return ``obs`` as a float32 array of exact shape ``(dim,)`` with only finite values, else raise ValueError.

    Batches such as ``(1, dim)`` / ``(N, dim)``, wrong lengths, and NaN / Inf entries are all rejected. The
    dtype is checked *before* any cast: complex (even with a zero imaginary part), boolean, string and object
    inputs raise instead of being silently coerced to float32.
    """
    try:
        raw = np.asarray(obs)
    except (TypeError, ValueError):
        raise ValueError(f"observation is not numeric: {type(obs).__name__}") from None
    if np.iscomplexobj(raw):
        raise ValueError("observation must be real-valued, got a complex array")
    if raw.dtype == np.bool_:
        raise ValueError("observation must be numeric, got a boolean array")
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"observation is not numeric: dtype {raw.dtype}")
    if raw.shape != (dim,):
        raise ValueError(f"observation shape {raw.shape} != ({dim},)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("observation contains NaN or Inf values")
    arr = raw.astype(np.float32)
    if not np.all(np.isfinite(arr)):  # e.g. float64 1e300 overflows float32
        raise ValueError("observation values overflow float32")
    return arr


# Fixed (v, w) physics command executed for each action (screen coords: w < 0 turns left).
_ACTION_COMMANDS = {
    ACTION_LEFT: (0.72 * MAX_LINEAR_SPEED, -0.82 * MAX_ANGULAR_SPEED),
    ACTION_STRAIGHT: (0.92 * MAX_LINEAR_SPEED, 0.0),
    ACTION_RIGHT: (0.72 * MAX_LINEAR_SPEED, 0.82 * MAX_ANGULAR_SPEED),
}


@dataclass
class RectObstacle:
    x: float
    y: float
    w: float
    h: float

    def to_segments(self) -> List[Tuple[float, float, float, float]]:
        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.w, self.y + self.h
        return [
            (x1, y1, x2, y1),
            (x2, y1, x2, y2),
            (x2, y2, x1, y2),
            (x1, y2, x1, y1),
        ]


@dataclass
class CircleObstacle:
    cx: float
    cy: float
    r: float


@dataclass
class MapConfig:
    name: str
    short_name: str
    spawn: Tuple[float, float, float]  # (x, y, theta)
    goal: Tuple[float, float]          # (gx, gy)
    rects: List[RectObstacle] = field(default_factory=list)
    circles: List[CircleObstacle] = field(default_factory=list)


def _point_collides_map(
    x: float,
    y: float,
    radius: float,
    rects: List[RectObstacle],
    circles: List[CircleObstacle],
) -> bool:
    if (
        x - radius < WALL_MARGIN
        or x + radius > WORLD_WIDTH - WALL_MARGIN
        or y - radius < WALL_MARGIN
        or y + radius > WORLD_HEIGHT - WALL_MARGIN
    ):
        return True
    for rect in rects:
        cx = max(rect.x, min(x, rect.x + rect.w))
        cy = max(rect.y, min(y, rect.y + rect.h))
        if (x - cx) ** 2 + (y - cy) ** 2 < radius * radius:
            return True
    for cir in circles:
        min_d = radius + cir.r
        if (x - cir.cx) ** 2 + (y - cir.cy) ** 2 < min_d * min_d:
            return True
    return False


def _is_path_reachable(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    rects: List[RectObstacle],
    circles: List[CircleObstacle],
    cell: int = 24,
) -> bool:
    """Grid BFS verifying a robot-sized path exists from start to goal."""
    cols = WORLD_WIDTH // cell
    rows = WORLD_HEIGHT // cell
    clearance = ROBOT_RADIUS + 8.0

    sx, sy = int(start[0] // cell), int(start[1] // cell)
    gx, gy = int(goal[0] // cell), int(goal[1] // cell)
    if not (0 <= sx < cols and 0 <= sy < rows and 0 <= gx < cols and 0 <= gy < rows):
        return False

    visited = [[False] * cols for _ in range(rows)]
    q: deque[Tuple[int, int]] = deque([(sx, sy)])
    visited[sy][sx] = True

    while q:
        cx, cy = q.popleft()
        if abs(cx - gx) <= 1 and abs(cy - gy) <= 1:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[ny][nx]:
                wx = (nx + 0.5) * cell
                wy = (ny + 0.5) * cell
                if not _point_collides_map(wx, wy, clearance, rects, circles):
                    visited[ny][nx] = True
                    q.append((nx, ny))
    return False


def generate_random_map(seed: Optional[int] = None, difficulty: int = 12) -> MapConfig:
    """Create a brand-new random obstacle map with guaranteed reachability and blocking obstacles."""
    rng = np.random.default_rng(seed)
    pad = WALL_MARGIN + 75.0

    for _attempt in range(40):
        # Choose start & goal on opposite-ish sides so travel distance is substantial
        side = int(rng.integers(0, 4))
        if side == 0:  # left -> right
            sx, sy = float(rng.uniform(pad, 210)), float(rng.uniform(pad, WORLD_HEIGHT - pad))
            gx, gy = float(rng.uniform(WORLD_WIDTH - 210, WORLD_WIDTH - pad)), float(rng.uniform(pad, WORLD_HEIGHT - pad))
        elif side == 1:  # right -> left
            sx, sy = float(rng.uniform(WORLD_WIDTH - 210, WORLD_WIDTH - pad)), float(rng.uniform(pad, WORLD_HEIGHT - pad))
            gx, gy = float(rng.uniform(pad, 210)), float(rng.uniform(pad, WORLD_HEIGHT - pad))
        elif side == 2:  # top -> bottom
            sx, sy = float(rng.uniform(pad, WORLD_WIDTH - pad)), float(rng.uniform(pad, 200))
            gx, gy = float(rng.uniform(pad, WORLD_WIDTH - pad)), float(rng.uniform(WORLD_HEIGHT - 200, WORLD_HEIGHT - pad))
        else:  # bottom -> top
            sx, sy = float(rng.uniform(pad, WORLD_WIDTH - pad)), float(rng.uniform(WORLD_HEIGHT - 200, WORLD_HEIGHT - pad))
            gx, gy = float(rng.uniform(pad, WORLD_WIDTH - pad)), float(rng.uniform(pad, 200))

        theta = math.atan2(gy - sy, gx - sx) + float(rng.uniform(-0.35, 0.35))
        rects: List[RectObstacle] = []
        circles: List[CircleObstacle] = []

        # 1. Try to place 2 blockers near the direct line between start and goal to encourage detours
        #    (a candidate is skipped if it would leave the arena or crowd the start/goal).
        for frac in (0.33, 0.62):
            bx = sx + (gx - sx) * frac + float(rng.uniform(-35, 35))
            by = sy + (gy - sy) * frac + float(rng.uniform(-35, 35))
            if rng.random() < 0.55:
                rw, rh = float(rng.uniform(70, 115)), float(rng.uniform(70, 115))
                rx, ry = bx - rw / 2, by - rh / 2
                if (
                    rx > WALL_MARGIN + 45
                    and rx + rw < WORLD_WIDTH - WALL_MARGIN - 45
                    and ry > WALL_MARGIN + 45
                    and ry + rh < WORLD_HEIGHT - WALL_MARGIN - 45
                ):
                    cand = [RectObstacle(rx, ry, rw, rh)]
                    if not _point_collides_map(sx, sy, ROBOT_RADIUS + 55, cand, []) and not _point_collides_map(
                        gx, gy, GOAL_RADIUS + 55, cand, []
                    ):
                        rects.append(cand[0])
            else:
                cr = float(rng.uniform(38, 56))
                cand_c = [CircleObstacle(bx, by, cr)]
                if not _point_collides_map(sx, sy, ROBOT_RADIUS + 55, [], cand_c) and not _point_collides_map(
                    gx, gy, GOAL_RADIUS + 55, [], cand_c
                ):
                    circles.append(cand_c[0])

        # 2. Scatter additional random obstacles across the arena
        target_count = max(6, difficulty)
        for _ in range(target_count * 4):
            if len(rects) + len(circles) >= target_count:
                break
            if rng.random() < 0.5:
                rw = float(rng.uniform(60, 130))
                rh = float(rng.uniform(60, 130))
                rx = float(rng.uniform(WALL_MARGIN + 40, WORLD_WIDTH - WALL_MARGIN - 40 - rw))
                ry = float(rng.uniform(WALL_MARGIN + 40, WORLD_HEIGHT - WALL_MARGIN - 40 - rh))
                r_cand = RectObstacle(rx, ry, rw, rh)
                # Keep safe clearance from start, goal, and existing obstacles
                if _point_collides_map(sx, sy, ROBOT_RADIUS + 60, [r_cand], []):
                    continue
                if _point_collides_map(gx, gy, GOAL_RADIUS + 60, [r_cand], []):
                    continue
                cx, cy = rx + rw / 2, ry + rh / 2
                if _point_collides_map(cx, cy, max(rw, rh) * 0.55 + ROBOT_RADIUS * 2.3, rects, circles):
                    continue
                rects.append(r_cand)
            else:
                cr = float(rng.uniform(30, 56))
                cx = float(rng.uniform(WALL_MARGIN + cr + 40, WORLD_WIDTH - WALL_MARGIN - cr - 40))
                cy = float(rng.uniform(WALL_MARGIN + cr + 40, WORLD_HEIGHT - WALL_MARGIN - cr - 40))
                c_cand = CircleObstacle(cx, cy, cr)
                if _point_collides_map(sx, sy, ROBOT_RADIUS + 60, [], [c_cand]):
                    continue
                if _point_collides_map(gx, gy, GOAL_RADIUS + 60, [], [c_cand]):
                    continue
                if _point_collides_map(cx, cy, cr + ROBOT_RADIUS * 2.5, rects, circles):
                    continue
                circles.append(c_cand)

        if _is_path_reachable((sx, sy), (gx, gy), rects, circles):
            tag = f"#{seed}" if seed is not None else ""
            return MapConfig(
                name=f"랜덤 장애물 맵 {tag}".strip(),
                short_name="랜덤 맵",
                spawn=(sx, sy, theta),
                goal=(gx, gy),
                rects=rects,
                circles=circles,
            )

    return build_preset_maps()[0]


def build_preset_maps() -> List[MapConfig]:
    """Fixed preset evaluation maps alongside random maps."""
    easy = MapConfig(
        name="고정 맵 A · 기본 회피 (Easy)",
        short_name="맵 A (기본 회피)",
        spawn=(365.0, 850.0, math.atan2(-700.0, 565.0)),
        goal=(930.0, 150.0),
        rects=[
            RectObstacle(645.0, 675.0, 105.0, 90.0),
            RectObstacle(230.0, 270.0, 110.0, 125.0),
        ],
        circles=[
            CircleObstacle(630.0, 570.0, 50.0),
            CircleObstacle(150.0, 125.0, 50.0),
        ],
    )

    hard = MapConfig(
        name="고정 맵 B · 복합 난코스 (Hard)",
        short_name="맵 B (복합 난코스)",
        spawn=(265.0, 165.0, 0.65),
        goal=(820.0, 805.0),
        rects=[
            RectObstacle(370.0, 315.0, 95.0, 95.0),
            RectObstacle(570.0, 525.0, 115.0, 145.0),
            RectObstacle(680.0, 115.0, 90.0, 65.0),
            RectObstacle(680.0, 360.0, 85.0, 120.0),
            RectObstacle(620.0, 745.0, 90.0, 115.0),
            RectObstacle(270.0, 470.0, 140.0, 120.0),
            RectObstacle(550.0, 220.0, 120.0, 70.0),
        ],
        circles=[
            CircleObstacle(500.0, 735.0, 30.0),
            CircleObstacle(900.0, 585.0, 55.0),
            CircleObstacle(270.0, 845.0, 30.0),
            CircleObstacle(905.0, 335.0, 40.0),
            CircleObstacle(160.0, 750.0, 40.0),
        ],
    )

    return [easy, hard]


@dataclass
class RobotState:
    """Independent robot state for single-robot or multi-model simultaneous racing."""
    x: float
    y: float
    theta: float
    v: float = 0.0
    w: float = 0.0
    collided: bool = False
    reached_goal: bool = False
    elapsed: float = 0.0
    distance: float = 0.0
    trail: List[Tuple[float, float]] = field(default_factory=list)
    last_distances: np.ndarray = field(default_factory=lambda: np.full(NUM_BEAMS, MAX_LIDAR_RANGE, dtype=np.float32))
    last_obs: np.ndarray = field(default_factory=lambda: np.ones(OBS_DIM, dtype=np.float32))


class Simulator:
    """2D world geometry, vectorized LiDAR raycasting, and multi-robot physics stepping."""

    def __init__(self, num_beams: int = NUM_BEAMS, noise_std: float = 0.004):
        if num_beams <= 0:
            raise ValueError("num_beams must be positive")
        self.preset_maps = build_preset_maps()
        self.num_beams = num_beams
        self.noise_std = noise_std
        self.beam_offsets = np.linspace(-FOV_RAD / 2.0, FOV_RAD / 2.0, num_beams, dtype=np.float32)

        self.current_map: MapConfig = generate_random_map(seed=101)
        self._seg_a: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._seg_s: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._cir_c: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._cir_r2: np.ndarray = np.empty((0,), dtype=np.float32)

        self.robot = RobotState(0.0, 0.0, 0.0)
        self.set_map_config(self.current_map)

    @property
    def goal(self) -> Tuple[float, float]:
        return self.current_map.goal

    def set_map_config(self, cfg: MapConfig) -> None:
        self.current_map = cfg
        self._rebuild_geometry()
        self.reset_robot()

    def set_preset_map(self, idx: int) -> None:
        # Load a fresh copy so spawn/goal edits never leak back into the stored preset.
        self.set_map_config(copy.deepcopy(self.preset_maps[idx % len(self.preset_maps)]))

    def randomize_map(self, seed: Optional[int] = None, difficulty: int = 12) -> None:
        self.set_map_config(generate_random_map(seed=seed, difficulty=difficulty))

    def _rebuild_geometry(self) -> None:
        m = self.current_map
        segs: List[Tuple[float, float, float, float]] = [
            (WALL_MARGIN, WALL_MARGIN, WORLD_WIDTH - WALL_MARGIN, WALL_MARGIN),
            (WORLD_WIDTH - WALL_MARGIN, WALL_MARGIN, WORLD_WIDTH - WALL_MARGIN, WORLD_HEIGHT - WALL_MARGIN),
            (WORLD_WIDTH - WALL_MARGIN, WORLD_HEIGHT - WALL_MARGIN, WALL_MARGIN, WORLD_HEIGHT - WALL_MARGIN),
            (WALL_MARGIN, WORLD_HEIGHT - WALL_MARGIN, WALL_MARGIN, WALL_MARGIN),
        ]
        for rect in m.rects:
            segs.extend(rect.to_segments())

        seg_arr = np.array(segs, dtype=np.float32)
        self._seg_a = seg_arr[:, :2]
        self._seg_s = seg_arr[:, 2:] - seg_arr[:, :2]

        if m.circles:
            self._cir_c = np.array([[c.cx, c.cy] for c in m.circles], dtype=np.float32)
            self._cir_r2 = np.array([c.r * c.r for c in m.circles], dtype=np.float32)
        else:
            self._cir_c = np.empty((0, 2), dtype=np.float32)
            self._cir_r2 = np.empty((0,), dtype=np.float32)

    def spawn_robot_state(self, pos: Optional[Tuple[float, float]] = None, theta: Optional[float] = None) -> RobotState:
        sx, sy, st = self.current_map.spawn
        if pos is not None:
            sx, sy = pos
        if theta is not None:
            st = theta
        st_obj = RobotState(x=sx, y=sy, theta=st, trail=[(sx, sy)])
        self.observe_agent(st_obj, add_noise=False)
        return st_obj

    def reset_robot(self, pos: Optional[Tuple[float, float]] = None) -> None:
        self.robot = self.spawn_robot_state(pos=pos)

    def set_goal(self, gx: float, gy: float) -> bool:
        if self.check_collision(gx, gy, extra_radius=GOAL_RADIUS - ROBOT_RADIUS):
            return False
        self.current_map.goal = (gx, gy)
        self.observe_agent(self.robot, add_noise=False)
        return True

    def check_collision(self, x: float, y: float, extra_radius: float = 0.0) -> bool:
        return _point_collides_map(
            x,
            y,
            ROBOT_RADIUS + extra_radius,
            self.current_map.rects,
            self.current_map.circles,
        )

    def check_goal_reached(self, x: float, y: float) -> bool:
        gx, gy = self.current_map.goal
        return math.hypot(x - gx, y - gy) <= (GOAL_RADIUS + ROBOT_RADIUS * 0.5)

    def step_agent(self, agent: RobotState, v: float, w: float, dt: float, add_noise: bool = True) -> None:
        """Advance a specific RobotState by dt seconds."""
        if agent.collided or agent.reached_goal:
            return

        v = float(np.clip(v, -MAX_LINEAR_SPEED * 0.5, MAX_LINEAR_SPEED))
        w = float(np.clip(w, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

        agent.v = v
        agent.w = w
        agent.elapsed += dt
        agent.theta = (agent.theta + w * dt + math.pi) % (2.0 * math.pi) - math.pi

        nx = agent.x + v * math.cos(agent.theta) * dt
        ny = agent.y + v * math.sin(agent.theta) * dt

        if self.check_collision(nx, ny):
            agent.collided = True
        else:
            moved = math.hypot(nx - agent.x, ny - agent.y)
            agent.x = nx
            agent.y = ny
            agent.distance += moved
            if not agent.trail or math.hypot(nx - agent.trail[-1][0], ny - agent.trail[-1][1]) > 5.0:
                agent.trail.append((nx, ny))
                if len(agent.trail) > 500:
                    agent.trail.pop(0)
            if self.check_goal_reached(nx, ny):
                agent.reached_goal = True

        self.observe_agent(agent, add_noise=add_noise)

    def step_action(self, agent: RobotState, action: Optional[int], dt: float, add_noise: bool = True) -> None:
        """Advance ``agent`` by dt seconds executing a discrete action (``None`` = stand still)."""
        if action is None:
            v, w = 0.0, 0.0
        else:
            v, w = _ACTION_COMMANDS[validate_action(action)]
        self.step_agent(agent, v, w, dt, add_noise=add_noise)

    def get_observation(self) -> np.ndarray:
        """Latest 28-dim observation of the primary robot."""
        return self.robot.last_obs

    def step(self, action: Optional[int], dt: float) -> Tuple[bool, bool, float]:
        """Step the primary robot with a discrete action (a wall bump is non-terminal in manual mode)."""
        prev_d = self.robot.distance
        # Manual driving may continue after bumping into a wall or passing through the goal area.
        self.robot.collided = False
        self.robot.reached_goal = False
        self.step_action(self.robot, action, dt, add_noise=True)
        if self.robot.collided:
            self.robot.reached_goal = self.check_goal_reached(self.robot.x, self.robot.y)
        return self.robot.collided, self.robot.reached_goal, self.robot.distance - prev_d

    def observe_agent(self, agent: RobotState, add_noise: bool = True) -> np.ndarray:
        """Compute 28-dim observation: [25 LiDAR norm distances, sin(rel_goal), cos(rel_goal), goal_dist_norm]."""
        angles = agent.theta + self.beam_offsets
        dx = np.cos(angles)
        dy = np.sin(angles)

        min_t = np.full(self.num_beams, MAX_LIDAR_RANGE, dtype=np.float32)

        if len(self._seg_a) > 0:
            ax_ox = self._seg_a[:, 0][None, :] - agent.x
            ay_oy = self._seg_a[:, 1][None, :] - agent.y
            sx = self._seg_s[:, 0][None, :]
            sy = self._seg_s[:, 1][None, :]

            rdx = dx[:, None]
            rdy = dy[:, None]
            denom = rdx * sy - rdy * sx
            valid_denom = np.abs(denom) > 1e-7
            safe_denom = np.where(valid_denom, denom, 1.0)

            t = (ax_ox * sy - ay_oy * sx) / safe_denom
            u = (ax_ox * rdy - ay_oy * rdx) / safe_denom
            hit = valid_denom & (t >= 0.0) & (t <= MAX_LIDAR_RANGE) & (u >= 0.0) & (u <= 1.0)
            min_t = np.minimum(min_t, np.min(np.where(hit, t, MAX_LIDAR_RANGE), axis=1))

        if len(self._cir_c) > 0:
            lx = agent.x - self._cir_c[:, 0][None, :]
            ly = agent.y - self._cir_c[:, 1][None, :]
            rdx = dx[:, None]
            rdy = dy[:, None]
            b = 2.0 * (rdx * lx + rdy * ly)
            c = (lx * lx + ly * ly) - self._cir_r2[None, :]
            disc = b * b - 4.0 * c
            hit = disc >= 0.0
            sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
            t = (-b - sqrt_disc) * 0.5
            valid_t = hit & (t >= 0.0) & (t <= MAX_LIDAR_RANGE)
            min_t = np.minimum(min_t, np.min(np.where(valid_t, t, MAX_LIDAR_RANGE), axis=1))

        agent.last_distances = min_t
        scan = min_t / MAX_LIDAR_RANGE
        if add_noise and self.noise_std > 0.0:
            scan = np.clip(scan + np.random.normal(0.0, self.noise_std, size=scan.shape).astype(np.float32), 0.0, 1.0)

        gx, gy = self.current_map.goal
        dist_g = math.hypot(gx - agent.x, gy - agent.y)
        angle_to_goal = math.atan2(gy - agent.y, gx - agent.x)
        rel_angle = (angle_to_goal - agent.theta + math.pi) % (2.0 * math.pi) - math.pi

        goal_feats = np.array(
            [
                math.sin(rel_angle),
                math.cos(rel_angle),
                min(dist_g / MAX_GOAL_DIST_NORM, 1.0),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([scan.astype(np.float32), goal_feats], axis=0)
        agent.last_obs = obs
        return obs
