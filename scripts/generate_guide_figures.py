"""Render the Korean guide's classification figures from simulator and checkpoint data."""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expert import AStarTeacher
from model import MODEL_LABELS, load_checkpoint_with_fallback, predict_action
from simulator import (
    ACTION_LEFT, ACTION_NAMES, ACTION_RIGHT, ACTION_STRAIGHT, FOV_RAD, GOAL_RADIUS,
    MAX_LIDAR_RANGE, NUM_ACTIONS, NUM_BEAMS, ROBOT_RADIUS,
    WALL_MARGIN, WORLD_HEIGHT, WORLD_WIDTH, CircleObstacle, MapConfig, RectObstacle, Simulator,
)
from ui_kit import (
    BG_WORLD, BORDER, CYAN, DANGER, GRID, MODEL_COLORS, OBSTACLE_EDGE, OBSTACLE_FILL,
    PRIMARY, SUCCESS, SURFACE, SURFACE_2, SURFACE_3, TEXT, TEXT_2, TEXT_3, WALL,
    WARNING, _first_existing, blend, tint,
)

MODELS_DIR = ROOT / "models"
FIGURES = ROOT / "docs" / "figures"
DATASET = ROOT / "data" / "dataset.npz"
MODEL_ORDER = ("cnn1d", "mlp", "rf", "knn", "linear")
LATENCY_STEPS = 6
SCENE = MapConfig(
    "가이드 장면", "가이드", spawn=(140.0, 480.0, 0.0), goal=(940.0, 480.0),
    rects=[RectObstacle(470, 250, 60, 460)],
    circles=[CircleObstacle(760, 300, 40), CircleObstacle(300, 720, 40)],
)
DT = 1.0 / 30.0
SS = 2
PANEL = 476
MAP_H = 423
SCALE = PANEL / WORLD_WIDTH
BLIND = (5, 7, 11)
LIGHT_MAP = (226, 232, 238)
LIGHT_GRID = (205, 214, 222)
LIGHT_WALL = (120, 134, 152)
LIGHT_FILL = (84, 101, 122)
LIGHT_EDGE = (60, 74, 92)
PATH = (12, 140, 104)
DARK_RIM = (20, 28, 40)
INITIAL_LINE = (45, 58, 78)
FONT_PATHS = (
    _first_existing(("NanumSquareR.ttf", "NanumBarunGothic.ttf", "NotoSansCJK-Regular.ttc")),
    _first_existing(("NanumSquareB.ttf", "NanumBarunGothicBold.ttf", "NotoSansCJK-Bold.ttc")),
)


@lru_cache(None)
def font(size, bold=False):
    path = FONT_PATHS[bold]
    if path is None:
        raise FileNotFoundError("Install NanumSquare, NanumBarunGothic or Noto Sans CJK.")
    return ImageFont.truetype(path, size * SS)


def scaled(values):
    return tuple(round(v * SS) for v in values)


class Canvas:
    """Draw in final pixel coordinates on a 2× RGB surface."""

    def __init__(self, width, height, bg=BG_WORLD):
        self.image = Image.new("RGB", scaled((width, height)), bg)
        self.draw = ImageDraw.Draw(self.image)

    def box(self, bounds, fill=None, edge=None, radius=0, width=1):
        self.draw.rounded_rectangle(scaled(bounds), radius=round(radius * SS),
                                    fill=fill, outline=edge, width=round(width * SS))

    def line(self, points, color, width=1):
        if len(points) >= 2:
            self.draw.line([scaled(p) for p in points], color, round(width * SS), joint="curve")

    def dot(self, point, radius, fill=None, edge=None, width=1):
        x, y = point
        self.draw.ellipse(scaled((x-radius, y-radius, x+radius, y+radius)),
                          fill, edge, width=round(width * SS))

    def text(self, pos, value, size, color=TEXT, bold=False, anchor="lm"):
        self.draw.text(scaled(pos), value, font=font(size, bold), fill=color, anchor=anchor)

    def dash(self, start, end, color, width=1, dash=6, gap=5):
        dx, dy = end[0]-start[0], end[1]-start[1]
        length = math.hypot(dx, dy)
        for offset in range(0, math.ceil(length), dash + gap):
            a, b = offset / length, min(offset + dash, length) / length
            self.line([(start[0]+a*dx, start[1]+a*dy), (start[0]+b*dx, start[1]+b*dy)], color, width)

    def final(self):
        w, h = self.image.size
        return self.image.resize((w // SS, h // SS), Image.Resampling.LANCZOS)


def save_gif(frames, durations, path):
    # Equal-size samples retain thin strokes. Ample swatches give short-lived accents
    # sufficient weight in the same median-cut palette as the static backgrounds.
    indices = sorted(set(range(0, len(frames), 4)) | {len(frames)-1})
    accents = [BG_WORLD, SURFACE, SURFACE_2, SURFACE_3, BORDER, TEXT, TEXT_2, TEXT_3,
               PRIMARY, SUCCESS, WARNING, DANGER, CYAN, GRID, WALL, OBSTACLE_FILL,
               OBSTACLE_EDGE, BLIND, LIGHT_MAP, LIGHT_GRID, LIGHT_WALL, LIGHT_FILL,
               LIGHT_EDGE, PATH, DARK_RIM, INITIAL_LINE, *MODEL_COLORS.values()]
    w, h = frames[0].size
    montage = Image.new("RGB", (w, len(indices)*h + 128))
    for row, i in enumerate(indices):
        montage.paste(frames[i], (0, row*h))
    draw = ImageDraw.Draw(montage)
    for i, color in enumerate(accents):
        draw.rectangle((i*w//len(accents), len(indices)*h, (i+1)*w//len(accents)-1, montage.height-1), fill=color)
    pal = montage.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    # Median cut may average adjacent accents. Reserve the closest distinct entry
    # for each swatch so arrival, impact and sensor colors survive exactly.
    colors = np.array(pal.getpalette(), dtype=np.int32).reshape(-1, 3)
    available = np.ones(len(colors), dtype=bool)
    for color in dict.fromkeys(accents):
        distances = ((colors - color)**2).sum(axis=1)
        index = int(np.argmin(np.where(available, distances, np.iinfo(np.int32).max)))
        colors[index] = color
        available[index] = False
    pal.putpalette(colors.astype(np.uint8).tobytes())
    q_frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q_frames[0].save(path, save_all=True, append_images=q_frames[1:], duration=durations,
                     loop=0, optimize=False, disposal=1, palette=pal.getpalette())
    with Image.open(path) as result:
        assert result.n_frames == len(frames)
        print(f"{path.name}: {result.width}×{result.height}, {result.n_frames} frames, "
              f"{path.stat().st_size / 1000:.3f} KB")


@dataclass
class Sample:
    xy: tuple
    theta: float
    elapsed: float
    front: float
    scan: np.ndarray
    endpoints: list
    action: int
    applied_action: Optional[int]
    lookahead: tuple | None
    goal: bool
    crash: bool


def drive(sim, teacher=None, model=None, delay=0):
    sim.reset_robot()
    robot = sim.robot
    queue: deque[Optional[int]] = deque([None] * delay)
    history = []
    for step in range(901):
        xy = (robot.x, robot.y)
        theta = robot.theta
        obs = sim.get_observation()
        front = float(obs[9:16].min())
        scan = robot.last_distances.copy()
        angles = theta + sim.beam_offsets
        ends = np.column_stack((xy[0] + np.cos(angles)*scan, xy[1] + np.sin(angles)*scan))
        goal, crash = robot.reached_goal, robot.collided
        if goal or crash or step == 900:
            # Keep displaying the last actions issued when the run stops.
            action, applied = history[-1].action, history[-1].applied_action
        elif teacher is not None:
            action = applied = teacher.drive(sim, sim.robot, DT, add_noise=False)
        else:
            action = predict_action(model, obs)[0]
            queue.append(action)
            applied = queue.popleft()
            sim.step_action(sim.robot, applied, DT, add_noise=False)
        history.append(Sample(
            xy, theta, step*DT, front, scan, ends.tolist(), action, applied,
            teacher.lookahead_point if teacher else None, goal, crash,
        ))
        if goal or crash:
            break
    return history


def load_model(key):
    """Load ``key`` like the app does: canonical ``goal_{key}.*`` first, then the legacy ``_cls`` sibling.

    Both candidates must hold exactly ``key`` and pass the 28-feature dry-run probe; raises FileNotFoundError
    (listing every rejected file) when neither loads.
    """
    model, errors = load_checkpoint_with_fallback(MODELS_DIR, key)
    if model is None:
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise FileNotFoundError(f"No loadable classification checkpoint for {key} in {MODELS_DIR}{detail}")
    for error in errors:
        print(f"Checkpoint fallback for {key}: {error}")
    return model


def candidate_runs(sim):
    for key in MODEL_ORDER:
        try:
            model = load_model(key)
        except FileNotFoundError as exc:
            print(f"Skipping {key}: {exc}")
            continue
        yield key, model, drive(sim, model=model)


def astar_scenario():
    sim = Simulator(noise_std=0.0)
    sim.set_map_config(SCENE)
    teacher = AStarTeacher()
    path = teacher.plan(SCENE)
    astar = drive(sim, teacher=teacher)
    if not astar[-1].goal:
        raise RuntimeError("A* failed to reach the guide scene goal")
    for key, _, direct in candidate_runs(sim):
        if direct[-1].goal:
            print(f"A* vs model: A* GOAL {astar[-1].elapsed:.3f}s; "
                  f"{key} GOAL {direct[-1].elapsed:.3f}s")
            return path, astar, key, direct
    raise RuntimeError("No classification checkpoint reached the A* comparison goal")


def latency_scenario():
    sim = Simulator(noise_std=0.0)
    sim.set_map_config(SCENE)
    with np.load(DATASET) as data:
        p2 = float(np.percentile(data["obs"][:, 9:16].min(axis=1), 2))
    for key, model, direct in candidate_runs(sim):
        direct_min = min(sample.front for sample in direct)
        if not direct[-1].goal or direct_min < p2:
            continue
        delayed = drive(sim, model=model, delay=LATENCY_STEPS)
        delayed_min = min(sample.front for sample in delayed)
        if delayed[-1].crash and delayed_min < p2:
            first_rare = next(sample for sample in delayed if sample.front < p2)
            print(f"Latency: {key}, {LATENCY_STEPS} steps ({LATENCY_STEPS*DT:.3f}s), P2 {p2:.6f}; "
                  f"direct GOAL {direct[-1].elapsed:.3f}s (min front {direct_min:.6f}); "
                  f"delayed CRASH {delayed[-1].elapsed:.3f}s (min front {delayed_min:.6f}, "
                  f"first below P2 at {first_rare.elapsed:.3f}s)")
            return key, direct, delayed, p2
    raise RuntimeError(
        f"No classification checkpoint reached the goal at or above P2={p2:.6f} without delay "
        f"and crashed below P2 with a {LATENCY_STEPS}-step delay"
    )


def world(point):
    return point[0]*SCALE, point[1]*SCALE


@lru_cache(None)
def map_base(style):
    bg = {"blind": BLIND, "light": LIGHT_MAP, "dark": BG_WORLD}[style]
    c = Canvas(PANEL, MAP_H, bg)
    if style != "blind":
        grid, wall, fill, edge = GRID, WALL, OBSTACLE_FILL, OBSTACLE_EDGE
        if style == "light":
            grid, wall, fill, edge = LIGHT_GRID, LIGHT_WALL, LIGHT_FILL, LIGHT_EDGE
        for x in range(0, WORLD_WIDTH, 48):
            c.line([world((x, 0)), world((x, WORLD_HEIGHT))], grid)
        for y in range(0, WORLD_HEIGHT, 48):
            c.line([world((0, y)), world((WORLD_WIDTH, y))], grid)
        c.box((*world((WALL_MARGIN, WALL_MARGIN)), *world((WORLD_WIDTH-WALL_MARGIN, WORLD_HEIGHT-WALL_MARGIN))),
              edge=wall, width=2)
        for r in SCENE.rects:
            c.box((*world((r.x, r.y)), *world((r.x+r.w, r.y+r.h))), fill, edge, width=2)
        for r in SCENE.circles:
            c.dot(world((r.cx, r.cy)), r.r*SCALE, fill, edge, 2)
    return c.image


@lru_cache(None)
def map_mask():
    mask = Image.new("L", scaled((PANEL, MAP_H)))
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, mask.width-1, mask.height-1), 10*SS, fill=255)
    return mask


def new_map(style):
    c = Canvas(PANEL, MAP_H)
    c.image.paste(map_base(style))
    return c


def two_panels(titles):
    frame = Canvas(984, 540)
    frame.line([(492, 16), (492, 524)], BORDER)
    for x, (title, subtitle, accent) in zip((0, 508), titles):
        frame.text((x+12, 24), title, 22, bold=True)
        frame.text((x+12, 49), subtitle, 15, accent)
    return frame


def chip(c, pos, label, accent, size=14):
    x, y = pos
    w = font(size, True).getlength(label)/SS + 22
    c.box((x, y, x+w, y+28), tint(accent, .11, SURFACE), blend(SURFACE, accent, .6), 7)
    c.text((x+11, y+14), label, size, accent, True)


def robot_and_goal(c, run, step, rim):
    s = run[step]
    goal = world(SCENE.goal)
    c.dot(goal, GOAL_RADIUS*SCALE, edge=SUCCESS, width=2)
    c.dot(goal, 3, WARNING)
    c.line([world(p.xy) for p in run[:step+1]], PRIMARY, 3)
    x, y = world(s.xy)
    radius = ROBOT_RADIUS*SCALE
    c.dot((x, y), radius, PRIMARY, rim, 1.5)
    c.line([(x, y), (x+radius*math.cos(s.theta), y+radius*math.sin(s.theta))], rim, 2)


def action_indicator(c, x, y, selected, accent=CYAN):
    width, height = 61, 28
    c.box((x, y, x+width*NUM_ACTIONS, y+height), SURFACE, BORDER, 7)
    for action, label in enumerate(ACTION_NAMES):
        left = x + width*action
        if action == selected:
            c.box((left, y, left+width, y+height), tint(accent, .16, SURFACE), accent, 7)
        c.text((left+width/2, y+height/2), label, 14,
               accent if action == selected else TEXT_2, action == selected, "mm")


def phase(c, x, label, t, accent):
    c.dot((x+11, 515), 4, accent)
    c.text((x+23, 515), label, 15, bold=True)
    c.text((x+PANEL-10, 515), f"t {t:.1f}초", 14, TEXT_3, anchor="rm")


def astar_gif(path, astar, key, direct):
    frames = []
    hits_history = deque(maxlen=4)
    ghosts = deque(maxlen=6)
    right_phase = 0
    right_labels = ("① 목적지까지 직선으로 출발", "② 벽 감지 → 학습한 방식으로 회피", "③ 다시 목적지 방향으로")
    wall = SCENE.rects[0]
    wall_exit = wall.x + wall.w
    last_step = max(len(astar), len(direct))-1
    steps = [0]*8 + list(range(3, last_step, 3)) + [last_step]
    for i, step in enumerate(steps):
        left_idx, right_idx = min(step, len(astar)-1), min(step, len(direct)-1)
        a, s = astar[left_idx], direct[right_idx]
        frame = two_panels([("A* 경로 계획", "사전 지도 있음 · 출발 전에 전체 경로 계산", TEXT_2),
                            (f"학습 모델 주행 · {MODEL_LABELS[key]}", "사전 지도 없음 · 라이다 빔 25개 + 목적지 방향", CYAN)])
        left = new_map("light")
        count = math.ceil(len(path)*(i+1)/8) if i < 8 else len(path)
        left.line([world(p) for p in path[:count]], PATH, 4)
        if a.lookahead and i >= 8:
            left.dot(world(a.lookahead), 5, edge=WARNING, width=2)
        robot_and_goal(left, astar, left_idx, DARK_RIM)
        chip(left, (19, 19), "경로 추종 (Pure Pursuit)", TEXT)
        if a.goal:
            chip(left, (365, 19), f"도착 {a.elapsed:.1f}초", SUCCESS)
        frame.image.paste(left.image, scaled((0, 64)), map_mask())

        right = new_map("blind")
        origin, goal = world(s.xy), world(SCENE.goal)
        ends = [world(p) for p in s.endpoints]
        right.draw.polygon([scaled(p) for p in [origin, *ends]], fill=blend(BLIND, CYAN, .10))
        right.dash(world(SCENE.spawn), goal, INITIAL_LINE)
        for start in ghosts:
            right.line([start, goal], blend(BLIND, TEXT, .16))
        for end in ends:
            right.line([origin, end], blend(BLIND, CYAN, .55))
        hits = [p for p, d in zip(ends, s.scan) if d < MAX_LIDAR_RANGE-1.5]
        for old_hits, strength in zip(hits_history, (.12, .25, .4, .6)[4-len(hits_history):]):
            for p in old_hits:
                right.dot(p, 2, blend(BLIND, WARNING, strength))
        right.dash(origin, goal, TEXT, 2, 10, 8)
        for p in hits:
            right.dot(p, 3, WARNING)
        hits_history.append(hits)
        if i % 10 == 0:
            ghosts.append(origin)
        robot_and_goal(right, direct, right_idx, TEXT)
        action_indicator(right, 19, 19, s.action)
        if s.goal:
            chip(right, (365, 19), f"도착 {s.elapsed:.1f}초", SUCCESS)
        frame.image.paste(right.image, scaled((508, 64)), map_mask())

        # Catch phase changes even when they occur between rendered samples.
        for observation in direct[max(0, right_idx-2):right_idx+1]:
            if right_phase == 0 and observation.front < .9:
                right_phase = 1
            elif (right_phase == 1 and observation.xy[0] >= wall_exit
                  and observation.front >= .9 and observation.action == ACTION_STRAIGHT):
                right_phase = 2
        left_label = "① 지도 전체로 경로 계산 (A*)" if i < 8 else "② 계산한 경로를 따라 주행"
        phase(frame, 0, f"도착 · {a.elapsed:.1f}초" if a.goal else left_label, a.elapsed, PATH)
        phase(frame, 508, f"도착 · {s.elapsed:.1f}초" if s.goal else right_labels[right_phase], s.elapsed, CYAN)
        frames.append(frame.final())
    save_gif(frames, [100]*(len(frames)-1) + [2500], FIGURES/"astar-vs-model.gif")


def latency_panel(run, idx, p2, delayed):
    c = new_map("dark")
    s = run[idx]
    origin = world(s.xy)
    for end, dist in zip(s.endpoints, s.scan):
        norm = dist / MAX_LIDAR_RANGE
        color = DANGER if norm < .32 else WARNING if norm < .62 else SUCCESS
        c.line([origin, world(end)], color)
    robot_and_goal(c, run, idx, TEXT)
    if s.front < p2:
        c.dot(origin, 14, edge=DANGER, width=6)
        chip(c, (19, MAP_H-49), "학습 데이터에 드문 거리", DANGER)
    accent = WARNING if delayed else CYAN
    chip(c, (19, 19), f"희망 행동  {ACTION_NAMES[s.action]}", accent)
    applied = "대기" if s.applied_action is None else ACTION_NAMES[s.applied_action]
    chip(c, (19, 53), f"적용 행동  {applied}", TEXT_2 if s.applied_action is None else accent)
    if s.crash:
        chip(c, (344, 19), f"충돌 · {s.elapsed:.1f}초", DANGER, 16)
    elif s.goal:
        chip(c, (344, 19), f"도착 · {s.elapsed:.1f}초", SUCCESS, 16)
    elif s.xy[0] > 560:
        chip(c, (378, 19), "벽 통과", SUCCESS, 15)
    return c


def sparkline(frame, x, run, idx, p2, last_step):
    frame.text((x+3, 496), "전방 빔 최솟값", 12, CYAN, True)
    frame.text((x+104, 496), f"학습 하위 2%  {p2:.3f}", 12, DANGER)
    frame.text((x+PANEL-4, 496), f"t {run[idx].elapsed:.1f}초", 12, TEXT_2, anchor="rm")
    top, bottom = 504, 539
    def y(value):
        return bottom - value*(bottom-top)
    frame.box((x, top, x+PANEL-1, bottom), SURFACE)
    frame.box((x, y(p2), x+PANEL-1, bottom), blend(BG_WORLD, DANGER, .18))
    frame.dash((x, y(p2)), (x+PANEL-1, y(p2)), DANGER, dash=5, gap=4)
    pts = [(x+i*(PANEL-4)/last_step+1, y(row.front)) for i, row in enumerate(run[:idx+1])]
    frame.line(pts, CYAN, 2)
    frame.dot(pts[-1], 1.5, CYAN)


def latency_gif(key, direct, delayed, p2):
    frames = []
    last_step = max(len(direct), len(delayed))-1
    for step in [*range(0, last_step, 3), last_step]:
        frame = two_panels([
            ("지연 없음", f"{MODEL_LABELS[key]} · 행동을 바로 적용", TEXT_2),
            (f"행동 {LATENCY_STEPS*DT:.1f}초 지연", f"{MODEL_LABELS[key]} · {LATENCY_STEPS}스텝 전 행동 적용", WARNING),
        ])
        for x, run in ((0, direct), (508, delayed)):
            idx = min(step, len(run)-1)
            panel = latency_panel(run, idx, p2, delayed=x != 0)
            frame.image.paste(panel.image, scaled((x, 64)), map_mask())
            sparkline(frame, x, run, idx, p2, last_step)
        frames.append(frame.final())
    save_gif(frames, [100]*(len(frames)-1)+[2500], FIGURES/"closed-loop-latency.gif")


def linear_diagram(c, coef, active):
    color = MODEL_COLORS["linear"]
    right, left = "우회전−좌회전 점수 가중치".split("−")
    c.text((11, 66), right, 13, TEXT_2, True)
    minus_x = 11 + font(13, True).getlength(right)/SS
    # The preferred Korean font can lack the mathematical minus glyph.
    c.line([(minus_x+1, 66), (minus_x+7, 66)], TEXT_2)
    c.text((minus_x+8, 66), left, 13, TEXT_2, True)
    baseline = 166
    c.line([(11, baseline), (177, baseline)], TEXT_3)
    scale = 65 / np.abs(coef).max()
    for i, weight in enumerate(coef):
        x = 13+i*5.9
        tip = baseline-scale*weight
        c.box((x, min(baseline, tip), x+4, max(baseline, tip)), color,
              CYAN if i in active else None)
    c.text((12, 88), "+", 12, TEXT_3)
    c.line([(12, 243), (19, 243)], TEXT_3)
    c.text((94, 261), "빔 25개 · 목표 입력 3개", 12, TEXT_2, anchor="mm")


def knn_diagram(c, points, shift):
    color = MODEL_COLORS["knn"]
    c.text((11, 66), "입력 간 거리", 13, TEXT_2, True)
    query = np.array([70+shift*15, 153+shift*6])
    nearest = np.argsort(np.linalg.norm(points-query, axis=1))[:7]
    for i in nearest:
        c.line([query, points[i]], tint(color, .7))
    for i, p in enumerate(points):
        c.dot(p, 2.8, color if i in nearest else TEXT_3)
    c.dot(query, 4.5, TEXT, BG_WORLD)
    c.dot((26, 261), 3.5, TEXT)
    c.text((34, 261), "현재 입력", 12, TEXT_2)
    c.dot((100, 261), 3.5, color)
    c.text((108, 261), "이웃 기록", 12, TEXT_2)


def tree_diagram(c, tree, obs):
    color = MODEL_COLORS["rf"]
    c.text((11, 66), "첫 트리의 분기", 13, TEXT_2, True)
    root = 0
    children = [tree.children_left[root], tree.children_right[root]]
    def decision(node):
        return int(obs[tree.feature[node]] > tree.threshold[node])
    branch = decision(root)
    leaf = 2*branch + decision(children[branch])
    xs = (48, 140)
    for i, x in enumerate(xs):
        c.line([(94, 116), (x, 155)], color if branch == i else SURFACE_3, 2)
        for j in range(2):
            dest = (x+(j-.5)*42, 218)
            selected = 2*i+j == leaf
            c.line([(x, 177), dest], color if selected else SURFACE_3, 2)
            c.dot(dest, 4, color if selected else SURFACE_3)
    for node, x, y, selected in [(root, 94, 104, True),
                                  (children[0], xs[0], 166, branch == 0),
                                  (children[1], xs[1], 166, branch == 1)]:
        feature = tree.feature[node]
        name = f"빔 {feature}" if feature < NUM_BEAMS else ("목표 sin", "목표 cos", "목표 거리")[feature-NUM_BEAMS]
        label = f"{name} ≤ {tree.threshold[node]:.2f}?"
        w = font(11, True).getlength(label)/SS+9
        c.box((x-w/2, y-11, x+w/2, y+11), tint(color, .13) if selected else SURFACE_2,
              color if selected else BORDER, 5)
        c.text((x, y), label, 11, TEXT if selected else TEXT_2, True, "mm")
    c.text((94, 261), "왼쪽은 예 · 오른쪽은 아니요", 12, TEXT_2, anchor="mm")


def mlp_diagram(c, active):
    color = MODEL_COLORS["mlp"]
    c.text((11, 66), "입력 → 은닉층 → 출력", 13, TEXT_2, True)
    inputs = [(22, y) for y in np.linspace(89, 238, 28)]
    hidden = [(95, y) for y in np.linspace(98, 229, 8)]
    outputs = [(164, y) for y in np.linspace(121, 205, NUM_ACTIONS)]
    for p in inputs:
        for h in hidden:
            c.line([p, h], tint(color, .13), .5)
    for i in active:
        for h in hidden:
            c.line([inputs[i], h], tint(CYAN, .56), .5)
    for h in hidden:
        for o in outputs:
            c.line([h, o], tint(color, .58))
        c.dot(h, 3.4, color)
    for i, p in enumerate(inputs):
        c.dot(p, 1.8, CYAN if i in active else TEXT_3)
    for p in outputs:
        c.dot(p, 4.5, color)
    for x, label in ((22, "28"), (95, "은닉층"), (164, str(NUM_ACTIONS))):
        c.text((x, 261), label, 12, TEXT_2, anchor="mm")


def cnn_diagram(c, scan, response, peak, center, channel):
    color = MODEL_COLORS["cnn1d"]
    c.text((11, 66), "빔 5개를 읽는 필터", 13, TEXT_2, True)
    pitch = 6.5
    for i, value in enumerate(scan):
        x = 13+i*pitch
        c.box((x, 103, x+5, 125), CYAN if value == 1 else SURFACE_3, radius=1)
    left = 13+(center-2)*pitch-1.5
    c.box((left, 100, left+5*pitch, 128), edge=color, radius=2, width=2)
    mid = 15.5+center*pitch
    c.line([(mid, 131), (mid, 147)], color)
    c.text((11, 166), f"필터 {channel+1}의 활성값", 13, TEXT_2, True)
    pts = [(15.5+i*pitch, 237-51*float(value)/peak) for i, value in enumerate(response)]
    c.draw.polygon([scaled(p) for p in [(pts[0][0], 237), *pts, (pts[-1][0], 237)]],
                   fill=tint(color, .16))
    c.line(pts, color, 2)
    c.dot(pts[center], 3, TEXT, color)
    c.text((94, 261), "첫 합성곱층 · ReLU 출력", 12, TEXT_2, anchor="mm")


CARDS = [
    ("linear", "Linear", True, ("빔마다 고정된 가중치 · 입구가", "옮겨지면 다른 가중치와 곱해짐")),
    ("knn", "KNN", False, ("가장 비슷한 기록 7개의", "행동에 거리 가중 투표")),
    ("rf", "RandomForest", True, ("60개 트리가 각자 특정 빔의", "기준값으로 질문")),
    ("mlp", "MLP", False, ("모든 입력이 모든 은닉 뉴런과", "연결 · 이웃 관계도 데이터로 학습")),
    ("cnn1d", "1D CNN", True, ("같은 필터가 모든 위치에 적용 ·", "입구가 옮겨지면 반응도 같이 이동")),
]


def linear_class_coefficients(est) -> np.ndarray:
    """Per-action weight rows (NUM_ACTIONS, n_features) of a fitted LogisticRegression, indexed by action ID.

    ``est`` may be the bare classifier or a Pipeline whose final step is one (any step names). Rows are placed by
    ``clf.classes_``. A binary model's single ``coef_`` row scores ``classes_[1]`` against ``classes_[0]``,
    so it becomes ``+coef_[0]`` / ``-coef_[0]`` for those two actions; an action the model never saw gets zeros.
    """
    clf = est.steps[-1][1] if hasattr(est, "steps") and est.steps else est
    coef = np.asarray(clf.coef_, dtype=np.float64)
    classes = [int(c) for c in clf.classes_]
    rows = np.zeros((NUM_ACTIONS, coef.shape[1]), dtype=np.float64)
    if len(classes) == 2 and coef.shape[0] == 1:
        rows[classes[0]] = -coef[0]
        rows[classes[1]] = coef[0]
    else:
        for row, cls in zip(coef, classes):
            rows[cls] = row
    return rows


def models_gif():
    scans = np.full((3, NUM_BEAMS), .30, dtype=np.float32)
    for shift, scan in enumerate(scans):
        scan[14+shift:19+shift] = 1.
    def estimator(key):
        return load_model(key).estimator

    weights = linear_class_coefficients(estimator("linear"))
    coef = weights[ACTION_RIGHT] - weights[ACTION_LEFT]
    tree = estimator("rf").estimators_[0].tree_
    with torch.inference_mode():
        responses = torch.relu(estimator("cnn1d").conv[0](torch.from_numpy(scans).unsqueeze(1))).numpy()
    channel = responses[0].max(axis=1).argmax()
    profiles = responses[:, channel, :]
    print(f"CNN first-layer peak filter: index {channel} (0-based, figure label {channel+1})")
    points = np.random.default_rng(0).uniform([15, 91], [173, 235], (40, 2))
    frames, durations = [], []
    for shift, scan in enumerate(scans):
        active = range(14+shift, 19+shift)
        obs = np.concatenate([scan, np.array([0., 1., .6], dtype=np.float32)])
        for j, center in enumerate(np.linspace(2, 22, 7).round().astype(int)):
            frame = Canvas(984, 520)
            frame.text((16, 22), "통로 입구가 빔 한 칸씩 옆으로 옮겨질 때", 20, bold=True)
            for index in range(3):
                x = 820+index*52
                selected = index == shift
                frame.box((x, 9, x+44, 35), tint(CYAN, .22, SURFACE_2) if selected else SURFACE,
                          CYAN if selected else BORDER, 7)
                frame.text((x+22, 22), f"+{index}", 14, CYAN if selected else TEXT_2, True, "mm")
            for i, value in enumerate(scan):
                x = 46+i*36
                frame.box((x, 147-100*value, x+28, 147), CYAN if i in active else SURFACE_3, radius=3)
            frame.text((46, 161), "왼쪽", 12, TEXT_2)
            frame.text((938, 161), "오른쪽", 12, TEXT_2, anchor="rm")
            for index, (key, title, real, caption) in enumerate(CARDS):
                c = Canvas(188, 334)
                c.box((.5, .5, 187, 333), SURFACE, BORDER, 12)
                c.dot((8, 24), 3, MODEL_COLORS[key])
                c.text((17, 24), title, 15, bold=True)
                c.text((181, 24), "실제 모델 값" if real else "개념도", 11, TEXT_3, anchor="rm")
                c.line([(10, 45), (177, 45)], BORDER)
                if key == "linear":
                    linear_diagram(c, coef, active)
                elif key == "knn":
                    knn_diagram(c, points, shift)
                elif key == "rf":
                    tree_diagram(c, tree, obs)
                elif key == "mlp":
                    mlp_diagram(c, active)
                else:
                    cnn_diagram(c, scan, profiles[shift], profiles.max(), center, channel)
                c.line([(10, 281), (177, 281)], BORDER)
                for k, line in enumerate(caption):
                    c.text((11, 298+18*k), line, 12, TEXT_2)
                frame.image.paste(c.image, scaled((199*index, 186)))
            frames.append(frame.final())
            durations.append(1100 if j == 6 else 140)
    save_gif(frames, durations, FIGURES/"five-models-shift.gif")


def observation_png():
    with np.load(DATASET) as data:
        observations = data["obs"]
        actions = data["actions"]
    center = NUM_BEAMS // 2
    left_medians = np.median(observations[:, :center], axis=1)
    right_minima = observations[:, center+1:NUM_BEAMS].min(axis=1)
    # A majority of left beams must be short and every right beam near full range.
    matches = np.flatnonzero((actions == ACTION_RIGHT) & (left_medians <= .40) & (right_minima >= .90))
    if not matches.size:
        raise RuntimeError("No recorded right turn has a left median ≤ 0.40 and a right minimum ≥ 0.90")
    index = matches[0]
    obs = observations[index]
    scan = obs[:NUM_BEAMS]
    action = int(actions[index])
    goal_sin, goal_cos, goal_distance = obs[NUM_BEAMS:]
    goal_angle = math.degrees(math.atan2(goal_sin, goal_cos))
    direction = "왼쪽" if goal_angle < 0 else "오른쪽"
    print(f"Recorded observation: sample {index+1}, action {ACTION_NAMES[action]}; "
          f"left min {scan[:center].min():.6f}, left median {left_medians[index]:.6f}; "
          f"right min {right_minima[index]:.6f}, right median {np.median(scan[center+1:]):.6f}")

    c = Canvas(960, 324)
    c.box((10, 10, 500, 314), SURFACE, radius=14)
    c.text((28, 36), "관측 / 로봇의 앞쪽 180°", 20, CYAN, True)
    origin = (245, 246)
    for angle, distance in zip(np.linspace(-FOV_RAD/2, FOV_RAD/2, NUM_BEAMS), scan):
        r = 181 * distance
        endpoint = (origin[0] + r*math.sin(angle), origin[1] - r*math.cos(angle))
        c.line([origin, endpoint], CYAN)
        if distance < .98:
            c.dot(endpoint, 3, WARNING)
    c.draw.polygon([scaled(p) for p in [(245, 234), (235, 252), (255, 252)]], fill=TEXT)
    c.text((28, 282), "왼쪽", 16, TEXT_2)
    c.text((470, 282), "오른쪽", 16, TEXT_2, anchor="rm")
    c.line([(514, 162), (548, 162)], TEXT_2, 2)
    c.draw.polygon([scaled(p) for p in [(541, 155), (551, 162), (541, 169)]], fill=TEXT_2)
    c.text((571, 36), "같은 순간에 기록한 행동", 20, WARNING, True)
    c.text((570, 98), ACTION_NAMES[action], 38, TEXT, True)
    action_indicator(c, 571, 139, action, WARNING)
    c.text((571, 203), f"목표 방향  {direction} {abs(goal_angle):.1f}°", 18, TEXT_2)
    c.text((571, 230), f"목표 거리 (정규화)  {goal_distance:.3f}", 16, TEXT_2)
    c.text((571, 281), f"저장 데이터의 {index+1}번째 샘플", 16, TEXT_2)
    path = FIGURES / "recorded-observation.png"
    result = c.final()
    result.save(path)
    print(f"{path.name}: {result.width}×{result.height}, {path.stat().st_size / 1000:.3f} KB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("astar", "models", "latency", "observation"))
    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    FIGURES.mkdir(exist_ok=True)
    if args.only in (None, "astar"):
        astar_gif(*astar_scenario())
    if args.only in (None, "latency"):
        latency_gif(*latency_scenario())
    if args.only in (None, "models"):
        models_gif()
    if args.only in (None, "observation"):
        observation_png()


if __name__ == "__main__":
    main()
