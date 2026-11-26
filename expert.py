"""Privileged A* Global Map Planner (Teacher) + Map-Free LiDAR Avoidance Baseline + Auto Dataset Generator.

Teacher-Student (Privileged Imitation Learning) Paradigm:
  - Teacher (A*): Has full access to the global obstacle map & costmap inflation to compute
    optimal shortest paths from Start to Goal. Used ONLY to generate training demonstrations.
  - Student (ML models): Has ZERO map access. Uses ONLY 25 LiDAR beams + relative goal bearing/dist.
  - Baseline (hand-written, no learning): a map-free reactive Artificial Potential Field controller
    that reads the exact same 28-dim observation as the students (goal attraction + local LiDAR
    repulsion). It is the non-learned reference the trained models are compared against.

Every policy outside this module speaks the same 3-class action interface (0=Left, 1=Straight, 2=Right).
The teacher and baseline compute a continuous steering command internally; ``steer_to_action``
turns that command into the action label that is recorded or executed.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from dataset import DrivingDataset
from simulator import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    FOV_RAD,
    MAX_ANGULAR_SPEED,
    MAX_GOAL_DIST_NORM,
    MAX_LIDAR_RANGE,
    MAX_LINEAR_SPEED,
    NUM_BEAMS,
    ROBOT_RADIUS,
    WALL_MARGIN,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    MapConfig,
    RobotState,
    Simulator,
    generate_random_map,
    validate_observation,
)


def steer_to_action(w: float, threshold: float = 0.28) -> int:
    """Map a steering rate w (screen coords: <0 left, >0 right) to the 3-class action label."""
    if w < -threshold:
        return ACTION_LEFT
    if w > threshold:
        return ACTION_RIGHT
    return ACTION_STRAIGHT


def _apf_command(obs: np.ndarray, num_beams: int = NUM_BEAMS) -> Tuple[float, float]:
    """Map-free reactive baseline (Artificial Potential Field): steer toward goal and away from close LiDAR returns.

    Uses only the same observation the ML students see (``num_beams`` normalized LiDAR ranges followed by
    ``[sin, cos, norm_dist]`` of the relative goal) and never touches the global map:
      - Goal behind the robot (cos < 0): turn in place toward it.
      - Otherwise sum a unit goal attraction with squared repulsion from every beam closer than ``reach``
        (which shrinks as the goal gets close so the robot can still enter narrow goal areas), and steer
        along the resulting heading. Forward speed scales with the free space directly ahead.
    """
    if num_beams <= 0:
        raise ValueError("num_beams must be positive")
    if len(obs) < num_beams + 3 or not np.all(np.isfinite(obs[: num_beams + 3])):
        return 0.0, 0.0  # Malformed or non-finite scan: stop instead of propagating NaN/inf into the pose.
    scan = obs[:num_beams]
    goal_sin, goal_cos, goal_dist = map(float, obs[num_beams : num_beams + 3])
    if goal_dist <= 0.0:
        return 0.0, 0.0  # Already at the goal.
    if goal_cos < 0:
        return 0.0, math.copysign(MAX_ANGULAR_SPEED, goal_sin)

    angles = np.linspace(-FOV_RAD / 2, FOV_RAD / 2, num_beams)
    reach = max(0.15, min(0.65, goal_dist * MAX_GOAL_DIST_NORM / MAX_LIDAR_RANGE))
    repulsion = np.maximum(0.0, 1.0 - scan / reach) ** 2
    forward = goal_cos - float(np.dot(repulsion, np.cos(angles)))
    lateral = goal_sin - float(np.dot(repulsion, np.sin(angles)))

    half = num_beams // 8
    front = float(scan[(num_beams - 1) // 2 - half : num_beams // 2 + half + 1].min())
    if abs(lateral) < 1e-6 and front < reach:
        lateral = -0.5  # Choose left when an obstacle directly ahead gives no side preference.

    heading = math.atan2(lateral, forward)
    w = float(np.clip(2.4 * heading, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
    clear = max(0.0, min(1.0, (front - 0.07) / 0.35))
    v = 0.85 * MAX_LINEAR_SPEED * clear * max(0.25, math.cos(heading))
    return v, w


def baseline_action(obs: np.ndarray, num_beams: int = NUM_BEAMS) -> int:
    """Discrete action chosen by the map-free LiDAR avoidance baseline for one observation.

    ``obs`` must be a finite 1D array of exactly ``num_beams + 3`` values; malformed or non-finite
    observations raise ValueError instead of silently becoming a driving command.
    """
    if num_beams <= 0:
        raise ValueError("num_beams must be positive")
    arr = validate_observation(obs, num_beams + 3)
    return steer_to_action(_apf_command(arr, num_beams)[1])


class AStarTeacher:
    """Privileged Global Planner: builds an inflated 2D costmap, runs A*, and tracks via Pure Pursuit."""

    def __init__(self, cell_size: int = 16):
        self.cell = cell_size
        self.cols = WORLD_WIDTH // cell_size
        self.rows = WORLD_HEIGHT // cell_size
        self.path: List[Tuple[float, float]] = []
        self.lookahead_point: Optional[Tuple[float, float]] = None

        # Precompute world centers for every grid cell (vectorized costmap building)
        xs = (np.arange(self.cols, dtype=np.float32) + 0.5) * cell_size
        ys = (np.arange(self.rows, dtype=np.float32) + 0.5) * cell_size
        self._gx, self._gy = np.meshgrid(xs, ys)  # (rows, cols)

    def _build_costmap(self, cfg: MapConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Return (blocked_bool[rows, cols], soft_cost[rows, cols]) from full global map."""
        gx, gy = self._gx, self._gy

        # Signed clearance distance from walls
        min_dist = np.minimum(
            np.minimum(gx - WALL_MARGIN, (WORLD_WIDTH - WALL_MARGIN) - gx),
            np.minimum(gy - WALL_MARGIN, (WORLD_HEIGHT - WALL_MARGIN) - gy),
        )

        # Distance from each rectangular obstacle
        for r in cfg.rects:
            dx = np.maximum(np.maximum(r.x - gx, 0.0), gx - (r.x + r.w))
            dy = np.maximum(np.maximum(r.y - gy, 0.0), gy - (r.y + r.h))
            dist_r = np.hypot(dx, dy)
            min_dist = np.minimum(min_dist, dist_r)

        # Distance from each circular obstacle
        for c in cfg.circles:
            dist_c = np.hypot(gx - c.cx, gy - c.cy) - c.r
            min_dist = np.minimum(min_dist, dist_c)

        hard_clearance = ROBOT_RADIUS + 12.0
        soft_inflation = ROBOT_RADIUS + 54.0

        blocked = min_dist < hard_clearance
        # Nav2-style inflation layer: smooth penalty near obstacles so A* stays centered
        prox = np.clip((soft_inflation - min_dist) / max(1.0, soft_inflation - hard_clearance), 0.0, 1.0)
        soft_cost = (prox ** 2) * 3.8
        return blocked, soft_cost

    def _nearest_free_cell(self, blocked: np.ndarray, col: int, row: int) -> Tuple[int, int]:
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        if not blocked[row, col]:
            return col, row
        for rad in range(1, 8):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    nc, nr = col + dc, row + dr
                    if 0 <= nc < self.cols and 0 <= nr < self.rows and not blocked[nr, nc]:
                        return nc, nr
        return col, row

    def plan(self, cfg: MapConfig, start: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
        """Run 8-connected A* on the global inflated costmap from start to cfg.goal."""
        blocked, soft_cost = self._build_costmap(cfg)
        sx, sy = start if start is not None else (cfg.spawn[0], cfg.spawn[1])
        gx, gy = cfg.goal

        sc, sr = self._nearest_free_cell(blocked, int(sx // self.cell), int(sy // self.cell))
        gc, gr = self._nearest_free_cell(blocked, int(gx // self.cell), int(gy // self.cell))

        # 8-connected neighbor offsets
        neighbors = (
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, 1.4142),
            (1, -1, 1.4142),
            (-1, 1, 1.4142),
            (-1, -1, 1.4142),
        )

        open_heap: List[Tuple[float, int, int]] = [(0.0, sc, sr)]
        g_score = np.full((self.rows, self.cols), np.inf, dtype=np.float32)
        g_score[sr, sc] = 0.0
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        closed = np.zeros((self.rows, self.cols), dtype=bool)

        found = False
        while open_heap:
            _, cc, cr = heapq.heappop(open_heap)
            if closed[cr, cc]:
                continue
            closed[cr, cc] = True

            if cc == gc and cr == gr:
                found = True
                break

            cur_g = float(g_score[cr, cc])
            for dc, dr, step_cost in neighbors:
                nc, nr = cc + dc, cr + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows and not closed[nr, nc] and not blocked[nr, nc]:
                    # Prevent diagonal corner-cutting through obstacle edges
                    if dc != 0 and dr != 0 and (blocked[cr, nc] or blocked[nr, cc]):
                        continue
                    tentative_g = cur_g + step_cost + float(soft_cost[nr, nc])
                    if tentative_g < g_score[nr, nc]:
                        g_score[nr, nc] = tentative_g
                        came_from[(nc, nr)] = (cc, cr)
                        h = math.hypot(gc - nc, gr - nr)
                        heapq.heappush(open_heap, (tentative_g + h, nc, nr))

        if not found:
            self.path = [(sx, sy), (gx, gy)]
            return self.path

        # Reconstruct grid path -> world coordinates
        raw_pts: List[Tuple[float, float]] = [(gx, gy)]
        node = (gc, gr)
        while node in came_from:
            wx = (node[0] + 0.5) * self.cell
            wy = (node[1] + 0.5) * self.cell
            raw_pts.append((wx, wy))
            node = came_from[node]
        raw_pts.append((sx, sy))
        raw_pts.reverse()

        # Light Chaikin-like smoothing (2 passes) for natural, curved driving demonstrations
        pts = raw_pts
        for _ in range(2):
            if len(pts) <= 2:
                break
            smoothed = [pts[0]]
            for i in range(1, len(pts) - 1):
                px = 0.25 * pts[i - 1][0] + 0.50 * pts[i][0] + 0.25 * pts[i + 1][0]
                py = 0.25 * pts[i - 1][1] + 0.50 * pts[i][1] + 0.25 * pts[i + 1][1]
                smoothed.append((px, py))
            smoothed.append(pts[-1])
            pts = smoothed

        self.path = pts
        return self.path

    def _command(self, agent: RobotState, beam_offsets: np.ndarray) -> Tuple[float, float]:
        """Pure Pursuit tracking along the precomputed A* global path + close-range safety guard."""
        if not self.path:
            return _apf_command(agent.last_obs, num_beams=len(beam_offsets))

        # Find closest waypoint along path, then advance to lookahead distance
        dists = [math.hypot(px - agent.x, py - agent.y) for px, py in self.path]
        closest_idx = int(np.argmin(dists))

        lookahead = 72.0
        target = self.path[-1]
        for i in range(closest_idx, len(self.path)):
            if dists[i] >= lookahead:
                target = self.path[i]
                break
        self.lookahead_point = target

        desired_heading = math.atan2(target[1] - agent.y, target[0] - agent.x)
        heading_err = (desired_heading - agent.theta + math.pi) % (2.0 * math.pi) - math.pi

        # Mild local repulsive safety guard from very close LiDAR beams (< 0.25)
        scan = agent.last_obs[: len(beam_offsets)]
        close_mask = scan < 0.26
        repulse_w = 0.0
        if np.any(close_mask):
            strength = ((0.26 - scan[close_mask]) / 0.26) ** 2
            repulse_w = float(np.sum(strength * (-np.sign(beam_offsets[close_mask])))) * 1.4

        w = 2.65 * heading_err + repulse_w
        w = float(np.clip(w, -MAX_ANGULAR_SPEED * 0.92, MAX_ANGULAR_SPEED * 0.92))

        # Smooth forward speed modulated by turning sharpness & front clearance
        mid = len(scan) // 2
        front_min = float(np.min(scan[mid - 3 : mid + 4]))
        turn_factor = max(0.42, 1.0 - 0.55 * abs(heading_err) / math.pi)
        clear_factor = float(np.clip((front_min - 0.06) / 0.30, 0.35, 1.0))
        v = MAX_LINEAR_SPEED * 0.92 * min(turn_factor, clear_factor)
        return v, w

    def drive(self, sim: Simulator, agent: RobotState, dt: float, add_noise: bool = True) -> int:
        """Track the planned path for one physics step and return the demonstrated action label."""
        v, w = self._command(agent, sim.beam_offsets)
        sim.step_agent(agent, v, w, dt, add_noise=add_noise)
        return steer_to_action(w)


def generate_expert_dataset(
    dataset: DrivingDataset,
    num_samples: int = 3000,
    seed: Optional[int] = None,
) -> dict:
    """Auto-generate training samples using the Privileged A* Global Map Planner across random maps (seeds 0..899).

    Test benchmark maps use seeds 1000+, ensuring 100% unseen evaluation maps.
    """
    rng = np.random.default_rng(seed)
    sim = Simulator(num_beams=dataset.num_beams, noise_std=0.004)
    teacher = AStarTeacher()

    dt = 1.0 / 20.0
    added = 0
    goals_reached = 0
    collisions = 0
    episodes = 0

    def next_episode() -> None:
        nonlocal episodes
        episodes += 1
        train_seed = int(rng.integers(0, 900))
        diff = int(rng.integers(8, 15))
        cfg = generate_random_map(seed=train_seed, difficulty=diff)
        sim.set_map_config(cfg)
        teacher.plan(cfg)

    next_episode()
    ep_steps = 0
    max_ep_steps = 220
    safety_limit = num_samples * 4
    total_steps = 0

    while added < num_samples and total_steps < safety_limit:
        total_steps += 1
        obs = sim.get_observation().copy()
        action = teacher.drive(sim, sim.robot, dt)
        ep_steps += 1

        if sim.robot.collided:
            collisions += 1
            ep_steps = 0
            next_episode()
            continue

        dataset.add(obs, action)
        added += 1

        if sim.robot.reached_goal:
            goals_reached += 1
            ep_steps = 0
            next_episode()
        elif ep_steps >= max_ep_steps:
            ep_steps = 0
            next_episode()

    return {
        "added": added,
        "episodes": episodes,
        "goals_reached": goals_reached,
        "collisions": collisions,
        "class_counts": dataset.class_counts(),
    }
