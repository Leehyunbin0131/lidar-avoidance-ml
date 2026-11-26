"""Large HiDPI Resizable GUI: Goal-Directed Collect -> 5-Model Train -> 6-Robot Race & 20-Map Benchmark."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pygame

import ui_kit as ui
from interpretation import CHAPTERS, GuideReader, Report, read_saved_report

from accelerator import BACKEND_LABELS, InferBackend, detect_hardware
from dataset import ACTION_LEFT, ACTION_NAMES, ACTION_RIGHT, ACTION_STRAIGHT, NUM_ACTIONS, DrivingDataset
from expert import AStarTeacher, baseline_action, generate_expert_dataset
from model import (
    BASELINE_LABEL,
    MODEL_LABELS,
    MODEL_TYPES,
    BenchmarkEntry,
    ModelType,
    TrainedModel,
    TrainHistory,
    checkpoint_path_for,
    evaluate_models_on_random_maps,
    export_benchmark_csv,
    load_checkpoint_with_fallback,
    predict_action,
    prepare_infer_backend,
    train_model,
)
from simulator import (
    GOAL_RADIUS,
    MAX_LIDAR_RANGE,
    ROBOT_RADIUS,
    WALL_MARGIN,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    RobotState,
    Simulator,
    validate_observation,
)
from visualizer import (
    CHART_CHOICES,
    ChartType,
    build_figure,
    figure_to_pygame_surface,
    save_figure_png,
)

PANEL_WIDTH = 600
LOGICAL_WIDTH = WORLD_WIDTH + PANEL_WIDTH  # 1680
LOGICAL_HEIGHT = WORLD_HEIGHT              # 960
FPS = 60
RECORD_HZ = 20.0
RACE_TIMEOUT_SEC = 18.0
BENCH_SEED_START = 1000  # Seeds 1000..1019 are reserved for the 20-map benchmark (unseen maps).
BENCH_NUM_MAPS = 20
LIVE_MAP_SEED_START = 2000  # Live "new random map" seeds start here, clear of the benchmark range.

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.npz"
MODELS_DIR = BASE_DIR / "models"
BENCH_CSV_PATH = BASE_DIR / "results" / "benchmark_20maps.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots"

BG_WINDOW = ui.BG_APP
BG_WORLD = ui.BG_WORLD
BG_PANEL = ui.BG_PANEL
BORDER_COL = ui.BORDER
GRID_COL = ui.GRID
OBSTACLE_FILL = ui.OBSTACLE_FILL
OBSTACLE_EDGE = ui.OBSTACLE_EDGE
WALL_COL = ui.WALL
TEXT_PRIMARY = ui.TEXT
TEXT_MUTED = ui.TEXT_2
ACCENT_BLUE = ui.PRIMARY
ACCENT_GREEN = ui.SUCCESS
ACCENT_RED = ui.DANGER
ACCENT_YELLOW = ui.WARNING
ACCENT_CYAN = ui.CYAN
RACE_COLORS = ui.MODEL_COLORS
RACE_TAGS: Dict[str, str] = {
    "baseline": "BASE",
    "linear": "LIN",
    "knn": "KNN",
    "rf": "RF",
    "mlp": "MLP",
    "cnn1d": "CNN",
}
SHORT_MODEL_LABELS: Dict[ModelType, str] = {
    "linear": "Linear",
    "knn": "KNN",
    "rf": "RF",
    "mlp": "MLP",
    "cnn1d": "1D CNN",
}


TABS = (
    ("collect", "1  수집", "주행 데이터 수집", "데이터셋 자동 생성을 시작하면 로봇이 달리며 학습 샘플을 추가합니다."),
    ("train", "2  학습", "모델 학습과 비교", "5종 모델 일괄 학습으로 검증 성능을 비교하세요."),
    ("test", "3  레이스", "주행 레이스와 평가", "동시 레이스를 시작해 모델별 궤적을 비교하세요."),
    ("viz", "4  시각화", "데이터와 모델 분석", "그래프를 선택하면 왼쪽에서 결과를 볼 수 있습니다."),
    ("guide", "5  해석", "이론과 결과 해석", "휠로 이어서 읽거나 목차에서 원하는 장으로 이동하세요."),
)


def columns(x: int, y: int, w: int, h: int, count: int, gap: int = 8) -> List[pygame.Rect]:
    """Split a control row evenly, including any leftover pixels."""
    usable = w - (count - 1) * gap
    rects = []
    for i in range(count):
        left = i * usable // count
        right = (i + 1) * usable // count
        rects.append(pygame.Rect(x + i * gap + left, y, right - left, h))
    return rects


def _actual_torch_backend(tm: TrainedModel) -> InferBackend:
    """Backend a PyTorch model really runs on: its OpenVINO runner if one is compiled, else its tensor device."""
    ab = tm.active_infer_backend
    if ab.startswith("ov_") and ab.split("_", 1)[1].upper() in tm.ov_runners:
        return ab
    return "torch_cuda" if next(tm.estimator.parameters()).device.type == "cuda" else "torch_cpu"


@dataclass
class RaceContender:
    key: str
    label: str
    tag: str
    color: Tuple[int, int, int]
    model: Optional[TrainedModel]  # None => map-free LiDAR avoidance baseline (expert.baseline_action)
    state: RobotState


class LidarPipelineApp:
    """Large Resizable GUI unifying Goal-Directed Collect, 5-Model Train, 6-Robot Race, and 20-Map Benchmark."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("LiDAR 목적지 장애물 회피 ML 파이프라인 (Collect → Train → Race & Benchmark)")

        # Pick generous initial window size scaled to user's monitor (great on 3200x2000 & 1080p)
        info = pygame.display.Info()
        dw = info.current_w if info.current_w > 0 else 1920
        dh = info.current_h if info.current_h > 0 else 1080
        scale = min((dw * 0.80) / LOGICAL_WIDTH, (dh * 0.82) / LOGICAL_HEIGHT)
        scale = max(0.75, min(scale, 1.45))
        init_w = int(LOGICAL_WIDTH * scale)
        init_h = int(LOGICAL_HEIGHT * scale)

        self.screen = pygame.display.set_mode((init_w, init_h), pygame.RESIZABLE)
        self.canvas = pygame.Surface((LOGICAL_WIDTH, LOGICAL_HEIGHT))
        self.clock = pygame.time.Clock()

        # Viewport transform cache for accurate mouse clicks at any window size
        self._vp_x = 0
        self._vp_y = 0
        self._vp_w = init_w
        self._vp_h = init_h

        self.fonts = ui.Fonts()
        self.ui = ui.UIKit(self.canvas, self.fonts)
        self.font_lg = self.fonts.h1
        self.font_md = self.fonts.h2
        self.font_sm = self.fonts.body
        self.font_xs = self.fonts.small
        self.font_tag = self.fonts.small_b

        self.hw = detect_hardware()
        self.train_device: str = self.hw.default_train_device  # "cuda" if available else "cpu"
        self.infer_backend: InferBackend = "torch_cuda" if self.hw.has_cuda else "torch_cpu"
        self.is_compiling_hw = False

        self.sim = Simulator()
        self.teacher = AStarTeacher()
        self.teacher.plan(self.sim.current_map)
        self.dataset = DrivingDataset()
        self.random_map_counter = LIVE_MAP_SEED_START - 1

        if DATA_PATH.exists():
            try:
                self.dataset.load(DATA_PATH)  # incompatible / corrupt files load nothing (returns 0)
            except Exception:  # pragma: no cover - defensive: a bad data file must never block startup
                self.dataset.clear()

        # Pipeline navigation
        self.tab = "collect"

        # Collect state
        self.recording = False
        self.expert_live = False
        self._record_accum = 0.0

        # Train state (every model is a 3-class Left / Straight / Right action classifier)
        self.selected_model_type: ModelType = "cnn1d"
        self.mirror_augment = True
        self.class_balance = True
        self.epochs_choices = (40, 60, 100)
        self.train_settings_open = False
        self.epochs_idx = 1  # 60 epochs default
        self.is_training = False
        self.train_lock = threading.Lock()
        self.train_history: Optional[TrainHistory] = None
        self.train_queue: List[ModelType] = []
        self.train_done_set: set[ModelType] = set()
        self.train_current_model: Optional[ModelType] = None
        self.train_overall_start: float = 0.0
        self.train_stage: str = ""
        # Finished-training results queued by the worker thread and applied only on the UI thread (_update).
        self._train_updates: Deque[Tuple[str, Dict[str, Any]]] = deque()
        self._train_updates_lock = threading.Lock()
        self._train_thread: Optional[threading.Thread] = None
        self._train_backend_warnings: List[str] = []

        self.trained_models: Dict[ModelType, TrainedModel] = {}
        self.active_model: Optional[TrainedModel] = None
        self.model_load_errors: List[str] = []

        # Test / Race state
        self.test_mode = "race"  # "race" (6 models simultaneously) | "single"
        self.race_running = False
        self.race_contenders: List[RaceContender] = []
        self.race_rebuild_pending = False  # models changed during a race; the next start rebuilds contenders
        self.single_running = False
        self.single_model_key: str = "baseline"  # "baseline" or ModelType
        # Driver bound when a single run starts; retraining never swaps it mid-run (None = baseline).
        self.single_driver_key: str = "baseline"
        self.single_driver: Optional[TrainedModel] = None
        self.last_pred_info: Dict[str, object] = {}

        # 20-Random-Map Quantitative Benchmark state
        self.is_benchmarking = False
        self.bench_progress = (0, 1, "")
        self.bench_results: List[BenchmarkEntry] = []
        self.reader = GuideReader(self.ui)
        self.guide_report = read_saved_report(BENCH_CSV_PATH)

        # Tab 4: Matplotlib 2D/3D Visualization Studio state
        self.viz_chart: ChartType = "pca3d"
        self.viz_model: ModelType = "mlp"
        self.viz_elev: float = 24.0
        self.viz_azim: float = 42.0
        self.viz_surface: Optional[pygame.Surface] = None
        self.viz_lock = threading.Lock()
        self.is_rendering_viz = False
        self._viz_dragging = False
        self._viz_drag_last: Tuple[int, int] = (0, 0)

        self.status_page = 0
        self.status_msg = TABS[0][3]
        self.status_color = TEXT_MUTED

        self._autoload_saved_models()
        self._reset_race()

    @property
    def epochs(self) -> int:
        return self.epochs_choices[self.epochs_idx]

    def _autoload_saved_models(self) -> None:
        """Load every model's canonical checkpoint, falling back to the legacy ``_cls`` file if it fails.

        Load failures are kept in ``model_load_errors`` and surfaced in the status bar instead of being dropped.
        """
        self.model_load_errors = []
        if not MODELS_DIR.exists():
            return
        for m_type in MODEL_TYPES:
            # Each candidate must hold exactly ``m_type`` and pass a finite 28-feature dry-run prediction.
            tm, errors = load_checkpoint_with_fallback(MODELS_DIR, m_type)
            self.model_load_errors.extend(errors)
            if tm is None:
                continue
            self.trained_models[m_type] = tm
            if self.active_model is None:
                self.active_model = tm
                self.selected_model_type = m_type
                self.single_model_key = m_type
        if self.model_load_errors:
            names = ", ".join(err.split(":", 1)[0] for err in self.model_load_errors)
            self.set_status(f"체크포인트 불러오기 실패 {len(self.model_load_errors)}건: {names}", ACCENT_YELLOW)

    def set_status(self, msg: str, color: Tuple[int, int, int] = TEXT_MUTED) -> None:
        self.status_page = 0
        self.status_msg = msg
        self.status_color = color

    @property
    def infer_switching(self) -> bool:
        """True while DL models are being compiled / moved to another inference backend."""
        return self.is_compiling_hw

    # ------------------------------------------------------------------
    # Viewport / Mouse Coordinate Mapping
    # ------------------------------------------------------------------
    def _update_viewport(self) -> None:
        sw, sh = self.screen.get_size()
        scale = min(sw / LOGICAL_WIDTH, sh / LOGICAL_HEIGHT)
        self._vp_w = max(1, int(LOGICAL_WIDTH * scale))
        self._vp_h = max(1, int(LOGICAL_HEIGHT * scale))
        self._vp_x = (sw - self._vp_w) // 2
        self._vp_y = (sh - self._vp_h) // 2

    def _screen_to_logical(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        lx = (pos[0] - self._vp_x) * LOGICAL_WIDTH // max(1, self._vp_w)
        ly = (pos[1] - self._vp_y) * LOGICAL_HEIGHT // max(1, self._vp_h)
        return lx, ly

    def _logical_mouse_pos(self) -> Tuple[int, int]:
        return self._screen_to_logical(pygame.mouse.get_pos())

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((max(960, event.w), max(540, event.h)), pygame.RESIZABLE)
                    self._update_viewport()
                elif event.type == pygame.MOUSEWHEEL and self.tab == "guide":
                    self.reader.scroll(-event.y * 68)
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    lpos = self._screen_to_logical(event.pos)
                    if event.button == 1:
                        on_plot = pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT).collidepoint(lpos)
                        if self.tab == "viz" and self.viz_chart == "pca3d" and on_plot and not self.is_rendering_viz:
                            self._viz_dragging = True
                            self._viz_drag_last = lpos
                        else:
                            self._handle_left_click(lpos)
                    elif event.button == 3:
                        self._handle_right_click(lpos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.reader.drag_offset = None
                    if self._viz_dragging:
                        self._viz_dragging = False
                        self._refresh_viz()
                elif event.type == pygame.MOUSEMOTION and self.tab == "guide":
                    self.reader.drag(self._screen_to_logical(event.pos)[1])
                elif event.type == pygame.MOUSEMOTION and self._viz_dragging:
                    lpos = self._screen_to_logical(event.pos)
                    dx = lpos[0] - self._viz_drag_last[0]
                    dy = lpos[1] - self._viz_drag_last[1]
                    self._viz_drag_last = lpos
                    self.viz_azim = (self.viz_azim - dx * 0.45) % 360.0
                    self.viz_elev = max(-60.0, min(80.0, self.viz_elev + dy * 0.35))

            self._update(dt)
            self._render()

            # Scale logical canvas smoothly to actual resizable window
            self._update_viewport()
            self.screen.fill(BG_WINDOW)
            if self._vp_w == LOGICAL_WIDTH and self._vp_h == LOGICAL_HEIGHT:
                self.screen.blit(self.canvas, (self._vp_x, self._vp_y))
            else:
                scaled = pygame.transform.smoothscale(self.canvas, (self._vp_w, self._vp_h))
                self.screen.blit(scaled, (self._vp_x, self._vp_y))
            pygame.display.flip()

        pygame.quit()

    # ------------------------------------------------------------------
    # Input & Actions
    # ------------------------------------------------------------------
    def _handle_keydown(self, event: pygame.event.Event) -> None:
        shortcuts = {
            pygame.K_r: "reset_robot", pygame.K_n: "new_random_map",
            pygame.K_1: "tab_collect", pygame.K_2: "tab_train",
            pygame.K_3: "tab_test", pygame.K_4: "tab_viz", pygame.K_5: "tab_guide",
        }
        if self.tab == "guide":
            increments = {pygame.K_UP: -68, pygame.K_DOWN: 68, pygame.K_PAGEUP: -860, pygame.K_PAGEDOWN: 860,
                          pygame.K_HOME: -self.reader.height, pygame.K_END: self.reader.height}
            if event.key in increments:
                self.reader.scroll(increments[event.key])
                return
            shortcuts.pop(pygame.K_n)
            shortcuts.pop(pygame.K_r)
            shortcuts.update({pygame.K_LEFT: "guide_prev", pygame.K_RIGHT: "guide_next"})
        key = shortcuts.get(event.key)
        if event.key == pygame.K_SPACE:
            key = {"collect": "toggle_record", "test": "toggle_test"}.get(self.tab)
        if key:
            self._on_button(key)

    def _handle_left_click(self, pos: Tuple[int, int]) -> None:
        mx, my = pos
        if 0 <= mx < WORLD_WIDTH and 0 <= my < WORLD_HEIGHT:
            if self.tab == "guide":
                self.reader.press(pos)
                return
            if self.tab == "viz":
                return
            if not self.sim.check_collision(float(mx), float(my)):
                self.sim.current_map.spawn = (float(mx), float(my), self.sim.robot.theta)
                self._reset_current_scene()
                self.set_status(f"출발 위치를 ({mx}, {my})로 변경했습니다.")
            return

        key = self.ui.hit_test(pos)
        if key:
            self._on_button(key)

    def _handle_right_click(self, pos: Tuple[int, int]) -> None:
        mx, my = pos
        if self.tab in ("viz", "guide"):
            return
        if 0 <= mx < WORLD_WIDTH and 0 <= my < WORLD_HEIGHT:
            if self.sim.set_goal(float(mx), float(my)):
                self.teacher.plan(self.sim.current_map, start=(self.sim.robot.x, self.sim.robot.y))
                self._reset_race()
                self.set_status(f"목적지(GOAL)를 ({mx}, {my})로 이동했습니다!", ACCENT_GREEN)
            else:
                self.set_status("장애물 위에는 목적지를 놓을 수 없습니다.", ACCENT_YELLOW)

    def _switch_tab(self, tab: str) -> None:
        if tab == self.tab:
            return
        self.reader.drag_offset = None
        if tab == "guide":
            self.reader.prepare(self.guide_report)
            self.reader.jump(3 if self.tab in ("test", "viz") else 0)
        self.tab = tab
        self._viz_dragging = False
        self.ui.hits.clear()
        self.set_status(next(hint for key, _, _, hint in TABS if key == tab))
        if tab != "collect":
            self.recording = False
            self.expert_live = False
        if tab != "test":
            self.race_running = False
            self.single_running = False
        if tab == "test":
            self._reset_race()
        elif tab == "viz":
            self._refresh_viz()

    def _spawn_new_random_map(self) -> None:
        self.random_map_counter += 1
        if BENCH_SEED_START <= self.random_map_counter < BENCH_SEED_START + BENCH_NUM_MAPS:
            self.random_map_counter = BENCH_SEED_START + BENCH_NUM_MAPS  # Never show a reserved evaluation map.
        self.sim.randomize_map(seed=self.random_map_counter, difficulty=12)
        self._reset_current_scene()
        self.set_status(f"새 랜덤 장애물 맵 #{self.random_map_counter} 생성! (목적지 도달 가능 검증 완료)", ACCENT_CYAN)

    def _reset_current_scene(self) -> None:
        self.last_pred_info = {}
        self.sim.reset_robot()
        self.teacher.plan(self.sim.current_map)
        self.race_running = False
        self.single_running = False
        self._reset_race()

    def _reset_race(self) -> None:
        """Build 6 simultaneous contenders: LiDAR avoidance baseline + every trained model."""
        contenders: List[RaceContender] = [
            RaceContender(
                key="baseline",
                label=BASELINE_LABEL,
                tag=RACE_TAGS["baseline"],
                color=RACE_COLORS["baseline"],
                model=None,
                state=self.sim.spawn_robot_state(),
            )
        ]
        for m_type in MODEL_TYPES:
            tm = self.trained_models.get(m_type)
            if tm is not None:
                contenders.append(
                    RaceContender(
                        key=m_type,
                        label=MODEL_LABELS[m_type],
                        tag=RACE_TAGS[m_type],
                        color=RACE_COLORS[m_type],
                        model=tm,
                        state=self.sim.spawn_robot_state(),
                    )
                )
        self.race_contenders = contenders
        self.race_rebuild_pending = False

    @property
    def test_start_blocked(self) -> bool:
        """A race / single run may not start while another job drives the same models (shared OpenVINO runners)."""
        return self.is_benchmarking or self.is_compiling_hw

    def _reset_race_unless_running(self) -> None:
        """Rebuild contenders now unless a race is in progress; that race keeps its robots and clocks, and the
        next start (which always calls ``_reset_race``) picks up newly trained models."""
        if self.race_running:
            self.race_rebuild_pending = True
        else:
            self._reset_race()

    def _toggle_test_run(self) -> None:
        running = self.race_running if self.test_mode == "race" else self.single_running
        if not running and self.test_start_blocked:
            job = "20맵 평가" if self.is_benchmarking else "추론 장치 전환"
            self.set_status(f"{job}이 끝난 뒤에 주행을 시작할 수 있습니다.", ACCENT_YELLOW)
            return
        if self.test_mode == "race":
            if not self.race_running:
                self._reset_race()
                self.race_running = True
                n_ml = len(self.race_contenders) - 1
                if n_ml == 0:
                    self.set_status("학습된 모델이 없어 라이다 회피 기준선만 출발합니다. [2. 학습]에서 5종 학습을 먼저 해보세요!", ACCENT_YELLOW)
                else:
                    self.set_status(f"라이다 회피 기준선과 학습 모델 {n_ml}종이 출발합니다.", ACCENT_GREEN)
            else:
                self.race_running = False
                self.set_status("레이스를 중지했습니다. 다시 시작하면 출발 위치로 돌아갑니다.")
        else:
            if not self.single_running:
                if self.single_model_key != "baseline" and self.single_model_key not in self.trained_models:
                    self.set_status("선택된 모델이 아직 학습되지 않았습니다.", ACCENT_YELLOW)
                    return
                if self.sim.robot.collided or self.sim.robot.reached_goal:
                    self.sim.reset_robot()
                self._bind_single_driver()
                self.single_running = True
                if self.single_model_key == "baseline":
                    self.set_status(f"{BASELINE_LABEL} 단일 주행 시작", ACCENT_GREEN)
                else:
                    self.set_status(f"[{self.single_model_key.upper()}] 단일 주행 시작", ACCENT_GREEN)
            else:
                self.single_running = False
                self.set_status("단일 자율주행 정지.")

    def _bind_single_driver(self) -> None:
        """Bind the selected single-run driver; it stays fixed until the run is restarted or re-picked."""
        self.single_driver_key = self.single_model_key
        self.single_driver = (
            None if self.single_model_key == "baseline" else self.trained_models.get(self.single_model_key)  # type: ignore[arg-type]
        )

    def _run_fast_auto_collect(self) -> None:
        self.recording = False
        self.expert_live = False
        res = generate_expert_dataset(dataset=self.dataset, num_samples=3000)
        self.set_status(
            f"자동 주행 완료 · 맵 {res['episodes']}개에서 샘플 {res['added']:,}개 추가 (총 {len(self.dataset):,}개)",
            ACCENT_GREEN,
        )

    def _start_training(self, train_all: bool = False) -> None:
        if self.is_training or self.is_benchmarking:
            return
        if self.infer_switching:
            self.set_status("추론 장치 전환이 끝난 뒤에 학습을 시작할 수 있습니다.", ACCENT_YELLOW)
            return
        if len(self.dataset) < 20:
            self.set_status(f"데이터가 부족합니다 (현재 {len(self.dataset)}개, 최소 20개 필요).", ACCENT_YELLOW)
            return

        self.is_training = True
        self.race_running = False
        self.single_running = False
        targets: Tuple[ModelType, ...] = MODEL_TYPES if train_all else (self.selected_model_type,)
        epochs = self.epochs
        mirror = self.mirror_augment
        balance = self.class_balance
        train_dev = self.train_device
        infer_be = self.infer_backend
        # Immutable copy: Collect-tab edits (clear / load / new samples) never reach the running job.
        dataset_snap = self.dataset.snapshot()

        with self.train_lock:
            self.train_queue = list(targets)
            self.train_done_set = set()
            self.train_current_model = targets[0]
            self.train_overall_start = time.perf_counter()
            self.train_stage = "준비 중..."
            self.train_history = None
        with self._train_updates_lock:
            self._train_updates.clear()

        post = self._post_training_update

        def worker() -> None:
            # Runs off the UI thread: it only touches its own new models and the lock-protected progress fields
            # (train_current_model / train_stage / train_history). Everything the UI, race or single run reads
            # is queued through ``post`` and applied by ``_apply_training_updates`` on the UI thread.
            def on_epoch(hist: TrainHistory) -> None:
                with self.train_lock:
                    self.train_history = hist
                    if hist.total_epochs > 1:
                        self.train_stage = f"Epoch {hist.current_epoch}/{hist.total_epochs}"
                    else:
                        self.train_stage = "검증 완료"

            error: Optional[str] = None
            try:
                for idx, m_type in enumerate(targets):
                    m_label = MODEL_LABELS[m_type]
                    prefix = f"[{idx + 1}/{len(targets)}] " if train_all else ""
                    dev_tag = f" ({train_dev.upper()})" if m_type in ("mlp", "cnn1d") else " (CPU)"
                    with self.train_lock:
                        self.train_current_model = m_type
                        self.train_stage = "Epoch 0 시작..." if m_type in ("mlp", "cnn1d") else "모델 피팅 중..."
                        # Clear previous chart curve so each sequential model draws cleanly
                        self.train_history = TrainHistory(
                            model_type=m_type,
                            total_epochs=epochs if m_type in ("mlp", "cnn1d") else 1,
                        )
                    post("status", msg=f"{prefix}{m_label}{dev_tag} 순차 학습 진행 중...", color=ACCENT_CYAN)

                    save_path = checkpoint_path_for(MODELS_DIR, m_type)
                    trained, hist = train_model(
                        dataset=dataset_snap,
                        model_type=m_type,
                        epochs=epochs,
                        mirror_augment=mirror,
                        class_balance=balance,
                        train_device=train_dev,
                        save_path=save_path,
                        on_epoch=on_epoch,
                    )
                    backend_error: Optional[str] = None
                    if trained.backend == "torch" and infer_be != trained.active_infer_backend:
                        with self.train_lock:
                            self.train_stage = f"{BACKEND_LABELS[infer_be]} 컴파일 중..."
                        try:
                            prepare_infer_backend(trained, infer_be, export_dir=MODELS_DIR)
                        except Exception as exc:
                            backend_error = str(exc) or type(exc).__name__
                            # Record what really executes now (prepare_infer_backend may have failed midway).
                            trained.active_infer_backend = _actual_torch_backend(trained)
                    post("model", model_type=m_type, trained=trained, history=hist,
                         requested_backend=infer_be, backend_error=backend_error)
                    # Short visual beat in batch mode so each sequential completion is clearly visible
                    if train_all and idx < len(targets) - 1:
                        time.sleep(0.12)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
            finally:
                post("done", train_all=train_all, error=error,
                     total_sec=time.perf_counter() - self.train_overall_start)

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()

    def _post_training_update(self, kind: str, **payload: Any) -> None:
        """Worker side: queue a training result for the UI thread."""
        with self._train_updates_lock:
            self._train_updates.append((kind, payload))

    def _apply_training_updates(self) -> None:
        """UI-thread side: apply queued training results (models, history, backend warnings, final status).

        A race in progress keeps its contenders and clocks (``race_rebuild_pending`` makes the next start pick up
        the new models) and an active single run keeps its bound driver, robot and clock.
        """
        with self._train_updates_lock:
            updates = list(self._train_updates)
            self._train_updates.clear()
        for kind, data in updates:
            if kind == "status":
                self.set_status(data["msg"], data["color"])
            elif kind == "model":
                m_type: ModelType = data["model_type"]
                trained: TrainedModel = data["trained"]
                self.trained_models[m_type] = trained
                with self.train_lock:
                    # A completion drained late (the worker has already moved on to the next model) must not
                    # revert the live chart to the finished model's curve.
                    if not self.is_training or self.train_current_model == m_type:
                        self.train_history = data["history"]
                    self.train_done_set.add(m_type)
                if data["backend_error"] is not None:
                    self._train_backend_warnings.append(self._backend_fallback_note(trained, data))
                    self._reconcile_infer_backend(trained)
            elif kind == "done":
                self._finish_training(data)

    @staticmethod
    def _backend_fallback_note(trained: TrainedModel, data: Dict[str, Any]) -> str:
        return (
            f"{SHORT_MODEL_LABELS[trained.model_type]}: {BACKEND_LABELS[data['requested_backend']]} 준비 실패"
            f"({data['backend_error']}) → [{BACKEND_LABELS[trained.active_infer_backend]}]로 실행"
        )

    def _reconcile_infer_backend(self, trained: TrainedModel) -> None:
        """Keep the selected backend label truthful: follow the fallback when no DL model runs on it any more."""
        torch_models = [tm for tm in self.trained_models.values() if tm.backend == "torch"]
        if all(tm.active_infer_backend != self.infer_backend for tm in torch_models):
            self.infer_backend = trained.active_infer_backend

    def _finish_training(self, data: Dict[str, Any]) -> None:
        warnings_note = ""
        if self._train_backend_warnings:
            warnings_note = " · 추론 장치 경고: " + " / ".join(self._train_backend_warnings)
        self._train_backend_warnings = []
        with self.train_lock:
            done = [m for m in self.train_queue if m in self.train_done_set]
            self.train_current_model = None
            self.train_stage = ""
        self.is_training = False
        self._train_thread = None

        if data["error"] is not None:
            self._reset_race_unless_running()
            self.set_status(f"학습 오류: {data['error']}{warnings_note}", ACCENT_RED)
            return
        if not done:
            return
        if data["train_all"]:
            best_tm = self._best_model()
        else:
            best_tm = self.trained_models.get(done[-1])
        if best_tm is not None:
            self.active_model = best_tm
            self.selected_model_type = best_tm.model_type
            if not self.single_running:  # never retarget an active single run; it ends or restarts first
                self.single_model_key = best_tm.model_type
        self._reset_race_unless_running()
        color = ACCENT_YELLOW if warnings_note else ACCENT_GREEN
        if data["train_all"]:
            self.set_status(
                f"5종 순차 학습 완료! (총 {data['total_sec']:.1f}s) → [3. 레이스/평가]에서 비교하세요!{warnings_note}",
                color,
            )
        elif best_tm is not None:
            self.set_status(
                f"{MODEL_LABELS[best_tm.model_type]} 학습 완료! (검증 정확도 {best_tm.val_metric:.1f}%){warnings_note}",
                color,
            )

    def _start_20map_benchmark(self) -> None:
        if self.is_benchmarking or self.is_training:
            return
        if self.infer_switching:
            self.set_status("추론 장치 전환이 끝난 뒤에 20맵 평가를 시작할 수 있습니다.", ACCENT_YELLOW)
            return
        self.bench_progress = (0, 20 * len(self.race_contenders), "")
        self.is_benchmarking = True
        self.race_running = False
        self.single_running = False
        models_snap = dict(self.trained_models)
        self.set_status(f"랜덤 맵 20개에서 {BASELINE_LABEL}과 학습 모델을 평가합니다.", ACCENT_CYAN)

        def worker() -> None:
            def on_prog(done: int, total: int, label: str) -> None:
                self.bench_progress = (done, total, label)

            try:
                results = evaluate_models_on_random_maps(
                    trained_models=models_snap,
                    num_maps=BENCH_NUM_MAPS,
                    seed_start=BENCH_SEED_START,
                    on_progress=on_prog,
                )
                self.bench_results = results
                self.guide_report = Report(tuple(results), "이번 실행의 평가")
                export_benchmark_csv(results, BENCH_CSV_PATH)
                base = results[0]
                learned = results[1:]
                if learned:
                    best = max(learned, key=lambda r: r.success_rate)
                    msg = f"20맵 평가 완료 · 기준선 {base.success_rate:.0f}% · 학습 모델 중 최고 {best.label} {best.success_rate:.0f}% (CSV 저장됨)"
                else:
                    msg = f"20맵 평가 완료 · 기준선 {base.success_rate:.0f}% · 학습 모델 없음 (CSV 저장됨)"
                self.set_status(msg, ACCENT_GREEN)
            except Exception as exc:
                self.set_status(f"벤치마크 오류: {exc}", ACCENT_RED)
            finally:
                self.is_benchmarking = False

        threading.Thread(target=worker, daemon=True).start()

    def _best_model(self) -> Optional[TrainedModel]:
        """Trained model with the highest validation accuracy."""
        candidates = [self.trained_models[model] for model in MODEL_TYPES if model in self.trained_models]
        return max(candidates, key=lambda m: m.val_metric) if candidates else None

    def _switch_infer_backend(self, backend: InferBackend) -> None:
        if self.is_compiling_hw or self.is_training or self.is_benchmarking:
            return
        prev_backend = self.infer_backend
        self.infer_backend = backend
        torch_models = [tm for tm in self.trained_models.values() if tm.backend == "torch"]
        if not torch_models:
            self.set_status(
                f"DL 추론 가속기 [{BACKEND_LABELS[backend]}] 선택됨 (MLP/1D CNN 학습 시 자동 적용)",
                ACCENT_CYAN,
            )
            return

        self.is_compiling_hw = True
        self.race_running = False
        self.single_running = False
        self.set_status(f"[{BACKEND_LABELS[backend]}] 디바이스로 DL 모델 컴파일 중...", ACCENT_YELLOW)

        previous = [(tm, tm.active_infer_backend) for tm in torch_models]

        def worker() -> None:
            try:
                lat_parts = []
                for tm, _ in previous:
                    prepare_infer_backend(tm, backend, export_dir=MODELS_DIR)
                    lat_parts.append(f"{SHORT_MODEL_LABELS[tm.model_type]} {tm.infer_ms:.2f}ms")
                extra = " (OpenVINO IR 저장됨)" if backend.startswith("ov_") else ""
                self.set_status(
                    f"[{BACKEND_LABELS[backend]}] 전환 완료! {' / '.join(lat_parts)}{extra}",
                    ACCENT_GREEN,
                )
            except Exception as exc:
                # Roll every model (including the one that failed mid-switch) back to the backend it ran on,
                # so the selected backend label always matches what actually executes.
                restored = prev_backend
                try:
                    for tm, old_backend in reversed(previous):
                        if tm.active_infer_backend != old_backend:
                            prepare_infer_backend(tm, old_backend, export_dir=MODELS_DIR)
                except Exception:  # pragma: no cover - previous accelerator vanished; PyTorch CPU always works
                    restored = "torch_cpu"
                    for tm, _ in previous:
                        prepare_infer_backend(tm, "torch_cpu")
                self.infer_backend = restored
                self.set_status(
                    f"가속기 전환 오류: {exc} → [{BACKEND_LABELS[self.infer_backend]}] 유지",
                    ACCENT_RED,
                )
            finally:
                self.is_compiling_hw = False

        threading.Thread(target=worker, daemon=True).start()

    def _viz_model_choices(self) -> Tuple[ModelType, ...]:
        if self.viz_chart == "classification":
            return tuple(model for model in MODEL_TYPES if model in self.trained_models)
        return MODEL_TYPES

    def _refresh_viz(self) -> None:
        if self.is_rendering_viz:
            return
        choices = self._viz_model_choices()
        if choices and self.viz_model not in choices:
            self.viz_model = choices[0]
        self.is_rendering_viz = True
        chart = self.viz_chart
        m_type = self.viz_model
        elev = self.viz_elev
        azim = self.viz_azim
        models_snap = dict(self.trained_models)
        bench_snap = list(self.bench_results)
        dataset_snap = self.dataset.snapshot()

        def worker() -> None:
            try:
                fig = build_figure(
                    chart_type=chart,
                    dataset=dataset_snap,
                    trained_models=models_snap,
                    selected_model=m_type,
                    bench_results=bench_snap,
                    elev=elev,
                    azim=azim,
                )
                surf = figure_to_pygame_surface(fig, size=(WORLD_WIDTH, WORLD_HEIGHT))
                with self.viz_lock:
                    self.viz_surface = surf
            except Exception as exc:
                self.set_status(f"그래프 렌더링 오류: {exc}", ACCENT_RED)
            finally:
                self.is_rendering_viz = False

        threading.Thread(target=worker, daemon=True).start()

    def _export_viz_png(self, export_all: bool = False) -> None:
        if self.is_rendering_viz:
            return
        self.is_rendering_viz = True
        charts = [c for c, _ in CHART_CHOICES] if export_all else [self.viz_chart]
        m_type = self.viz_model
        elev = self.viz_elev
        azim = self.viz_azim
        models_snap = dict(self.trained_models)
        bench_snap = list(self.bench_results)
        dataset_snap = self.dataset.snapshot()
        self.set_status("고해상도 PNG 그래프 저장 중...", ACCENT_CYAN)

        def worker() -> None:
            try:
                saved_paths = []
                for ch in charts:
                    fig = build_figure(
                        chart_type=ch,
                        dataset=dataset_snap,
                        trained_models=models_snap,
                        selected_model=m_type,
                        bench_results=bench_snap,
                        elev=elev,
                        azim=azim,
                    )
                    out_p = PLOTS_DIR / f"{ch}_{m_type}.png"
                    save_figure_png(fig, out_p, dpi=160)
                    saved_paths.append(out_p)
                if export_all:
                    self.set_status(f"전체 {len(saved_paths)}종 그래프 PNG 저장 완료! (results/plots/)", ACCENT_GREEN)
                else:
                    self.set_status(f"저장 완료: results/plots/{saved_paths[0].name}", ACCENT_GREEN)
            except Exception as exc:
                self.set_status(f"PNG 저장 오류: {exc}", ACCENT_RED)
            finally:
                self.is_rendering_viz = False

        threading.Thread(target=worker, daemon=True).start()

    def _on_button(self, key: str) -> None:
        if key.startswith("viz_") and self.is_rendering_viz:
            return
        if key == "status_next":
            self.status_page += 1
        elif key.startswith("guide_"):
            if key == "guide_prev":
                self.reader.jump(self.reader.chapter - 1)
            elif key == "guide_next":
                self.reader.jump(self.reader.chapter + 1)
            elif key == "guide_lab":
                chapter = CHAPTERS[self.reader.chapter]
                self._switch_tab(chapter.lab)
                if self.reader.chapter == 4:
                    self._on_button("mode_single")
            else:
                self.reader.jump(int(key.removeprefix("guide_")))
        elif key.startswith("tab_"):
            self._switch_tab(key[4:])
        elif key == "new_random_map":
            self._spawn_new_random_map()
        elif key == "preset_0":
            self.sim.set_preset_map(0)
            self._reset_current_scene()
            self.set_status(f"{self.sim.current_map.name} 선택됨.")
        elif key == "preset_1":
            self.sim.set_preset_map(1)
            self._reset_current_scene()
            self.set_status(f"{self.sim.current_map.name} 선택됨.")
        elif key == "reset_robot":
            self._reset_current_scene()
            self.set_status("로봇을 출발점으로 옮겼습니다.")
        elif key == "toggle_record":
            self.recording = self.expert_live or not self.recording
            self.expert_live = False
            self.set_status(
                "수동 주행 데이터를 수집하고 있습니다. W(직진)·A(좌회전)·D(우회전) 또는 방향키로 로봇을 조종하세요." if self.recording else "수동 주행 데이터 수집을 중지했습니다.",
                ACCENT_RED if self.recording else TEXT_MUTED,
            )
        elif key == "auto_fast":
            self._run_fast_auto_collect()
        elif key == "toggle_expert_live":
            self.expert_live = not self.expert_live
            self.recording = self.expert_live
            if self.expert_live:
                self.teacher.plan(self.sim.current_map, start=(self.sim.robot.x, self.sim.robot.y))
            self.set_status(
                "데이터셋 자동 생성 중입니다. A* 경로를 따라 달리며 관측과 행동(좌·직·우)을 추가합니다." if self.expert_live else "데이터셋 자동 생성을 중지했습니다.",
                ACCENT_CYAN if self.expert_live else TEXT_MUTED,
            )
        elif key == "save_data":
            count = self.dataset.save(DATA_PATH, export_csv=True)
            self.set_status(f"데이터셋 저장 완료 · 샘플 {count:,}개 (NPZ·CSV)", ACCENT_GREEN)
        elif key == "load_data":
            count = self.dataset.load(DATA_PATH)
            self.set_status(
                f"데이터셋 불러오기 완료 · 샘플 {count:,}개" if count > 0 else "호환되는 저장 데이터가 없습니다.",
                ACCENT_GREEN if count > 0 else ACCENT_YELLOW,
            )
        elif key == "clear_data":
            self.dataset.clear()
            self.set_status("현재 데이터셋의 샘플을 비웠습니다.", ACCENT_YELLOW)
        elif key.startswith("mtype_"):
            self.selected_model_type = key[6:]  # type: ignore[assignment]
        elif key == "train_settings":
            self.train_settings_open = not self.train_settings_open
        elif key == "toggle_mirror":
            self.mirror_augment = not self.mirror_augment
        elif key == "toggle_balance":
            self.class_balance = not self.class_balance
        elif key == "cycle_epochs":
            self.epochs_idx = (self.epochs_idx + 1) % len(self.epochs_choices)
        elif key == "toggle_train_dev":
            if self.hw.has_cuda:
                self.train_device = "cpu" if self.train_device == "cuda" else "cuda"
                self.set_status(f"PyTorch 학습 디바이스: {self.train_device.upper()}", ACCENT_CYAN)
        elif key.startswith("infer_"):
            be: InferBackend = key[6:]  # type: ignore[assignment]
            self._switch_infer_backend(be)
        elif key == "start_train":
            self._start_training(train_all=False)
        elif key == "train_all":
            self._start_training(train_all=True)
        elif key == "mode_race":
            if self.test_mode != "race":
                self.test_mode = "race"
                self.single_running = False
                self._reset_race()
        elif key == "mode_single" and self.test_mode != "single":
            self.last_pred_info = {}
            self.test_mode = "single"
            self.race_running = False
            self.sim.reset_robot()
        elif key == "toggle_test":
            self._toggle_test_run()
        elif key.startswith("pick_single_"):
            self.last_pred_info = {}
            self.single_model_key = key[12:]
            self.sim.reset_robot()
            self._bind_single_driver()  # explicit re-pick restarts the run from spawn with the new driver
            if self.single_model_key == "baseline":
                self.set_status(f"단일 주행에서 {BASELINE_LABEL}을 선택했습니다.", ACCENT_CYAN)
            else:
                self.set_status(f"단일 주행 모델을 [{self.single_model_key.upper()}]로 선택했습니다.", ACCENT_CYAN)
        elif key == "run_bench":
            self._start_20map_benchmark()
        elif key == "export_bench":
            if self.bench_results:
                export_benchmark_csv(self.bench_results, BENCH_CSV_PATH)
                self.set_status(f"리포트 저장 완료: results/{BENCH_CSV_PATH.name}", ACCENT_GREEN)
        elif key.startswith("viz_chart_"):
            self.viz_chart = key[10:]  # type: ignore[assignment]
            self._refresh_viz()
        elif key.startswith("viz_model_"):
            self.viz_model = key[10:]  # type: ignore[assignment]
            self._refresh_viz()
        elif key == "viz_rot_left":
            self.viz_azim = (self.viz_azim - 25.0) % 360.0
            self._refresh_viz()
        elif key == "viz_rot_right":
            self.viz_azim = (self.viz_azim + 25.0) % 360.0
            self._refresh_viz()
        elif key == "viz_rot_up":
            self.viz_elev = min(80.0, self.viz_elev + 15.0)
            self._refresh_viz()
        elif key == "viz_rot_down":
            self.viz_elev = max(-60.0, self.viz_elev - 15.0)
            self._refresh_viz()
        elif key == "viz_refresh":
            self._refresh_viz()
        elif key == "viz_save_one":
            self._export_viz_png(export_all=False)
        elif key == "viz_save_all":
            self._export_viz_png(export_all=True)

    # ------------------------------------------------------------------
    # Simulation Update
    # ------------------------------------------------------------------
    def _read_manual_action(self) -> Optional[int]:
        """Keyboard -> action: A/← left, D/→ right (exactly one pressed), else W/↑ straight, else None."""
        keys = pygame.key.get_pressed()
        left = bool(keys[pygame.K_a] or keys[pygame.K_LEFT])
        right = bool(keys[pygame.K_d] or keys[pygame.K_RIGHT])
        if left != right:
            return ACTION_LEFT if left else ACTION_RIGHT
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            return ACTION_STRAIGHT
        return None

    @staticmethod
    def _policy_action(model: Optional[TrainedModel], obs: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """(action, probs) of the baseline (``model=None``) or a trained model; (None, None) on an invalid observation.

        ``None`` makes ``step_action`` hold the robot still instead of driving on a malformed scan.
        """
        try:
            if model is None:
                action = baseline_action(obs)
                return action, np.eye(NUM_ACTIONS, dtype=np.float32)[action]
            return predict_action(model, obs)
        except ValueError:
            return None, None

    def _hold_invalid_frame(self, exc: ValueError) -> None:
        """Skip an invalid observation frame during collection: no step, no sample, a handled status message.

        The robot stays where it is, but its LiDAR scan is re-taken at that pose so a transient bad frame
        (e.g. one NaN in ``last_obs``) clears and collection resumes on the next valid frame.
        """
        self._record_accum = 0.0
        self.sim.observe_agent(self.sim.robot, add_noise=False)
        self.set_status(f"잘못된 관측값 프레임을 건너뛰었습니다 (로봇 정지, 기록 안 함): {exc}", ACCENT_YELLOW)

    def _update(self, dt: float) -> None:
        self._apply_training_updates()
        if self.tab == "guide":
            return
        if self.tab == "test":
            if self.test_mode == "race" and self.race_running:
                any_active = False
                for cont in self.race_contenders:
                    st = cont.state
                    if st.collided or st.reached_goal or st.elapsed >= RACE_TIMEOUT_SEC:
                        continue
                    any_active = True
                    action, _ = self._policy_action(cont.model, st.last_obs)
                    self.sim.step_action(st, action, dt, add_noise=False)
                if not any_active:
                    self.race_running = False
                    winners = [c for c in self.race_contenders if c.state.reached_goal]
                    winners.sort(key=lambda c: c.state.elapsed)
                    if winners:
                        w0 = winners[0]
                        self.set_status(
                            f"레이스 종료! 1등: {w0.label} ({w0.state.elapsed:.2f}s) | 총 {len(winners)}/{len(self.race_contenders)}대 완주",
                            ACCENT_GREEN,
                        )
                    else:
                        self.set_status("레이스 종료! 완주 모델 없음 — [새 랜덤 장애물 맵] 또는 재학습해 보세요.", ACCENT_YELLOW)
            elif self.test_mode == "single" and self.single_running:
                ag = self.sim.robot
                if not ag.collided and not ag.reached_goal:
                    tm = self.single_driver  # bound at start; retraining never swaps it mid-run
                    if self.single_driver_key != "baseline" and tm is None:
                        self.single_running = False
                        return
                    action, probs = self._policy_action(tm, ag.last_obs)
                    self.last_pred_info = {} if action is None else {"action": action, "probs": probs}
                    self.sim.step_action(ag, action, dt, add_noise=False)
                    if ag.reached_goal:
                        self.single_running = False
                        self.set_status(f"목적지 도달 성공!! ({ag.elapsed:.2f}초)", ACCENT_GREEN)
                    elif ag.collided:
                        self.single_running = False
                        self.set_status(f"장애물 충돌! ({ag.elapsed:.2f}초)", ACCENT_RED)
            return

        if self.tab == "collect" and self.expert_live:
            try:
                obs = validate_observation(self.sim.get_observation(), self.dataset.obs_dim)  # fresh copy
            except ValueError as exc:
                self._hold_invalid_frame(exc)
                return
            action = self.teacher.drive(self.sim, self.sim.robot, dt)
            self._record_accum += dt
            if self._record_accum >= (1.0 / RECORD_HZ):
                self._record_accum = 0.0
                if not self.sim.robot.collided:  # Same rule as the offline generator: skip crash frames.
                    self.dataset.add(obs, action)
            if self.sim.robot.reached_goal or self.sim.robot.collided or self.sim.robot.elapsed > 16.0:
                self._spawn_new_random_map()
                self.expert_live = True
                self.recording = True
        else:
            action = self._read_manual_action()
            if self.tab == "collect" and self.recording:
                try:
                    obs = validate_observation(self.sim.get_observation(), self.dataset.obs_dim)
                except ValueError as exc:
                    self._hold_invalid_frame(exc)
                    return
                self._record_accum += dt
                if self._record_accum >= (1.0 / RECORD_HZ):
                    self._record_accum = 0.0
                    if action is not None:
                        self.dataset.add(obs, action)
            _, reached, _ = self.sim.step(action, dt)
            if reached and self.tab == "collect" and self.recording:
                self._spawn_new_random_map()
                self.recording = True

    # ------------------------------------------------------------------
    # Rendering (to 1680x960 Logical Canvas)
    # ------------------------------------------------------------------
    def _render(self) -> None:
        if self.tab == "guide":
            self.reader.prepare(self.guide_report)
            self.reader.draw(self._logical_mouse_pos())
        elif self.tab == "viz":
            self._render_viz_canvas()
        else:
            self._render_world()
        self._render_panel()

    def _render_viz_canvas(self) -> None:
        surf = self.canvas
        pygame.draw.rect(surf, BG_WORLD, (0, 0, WORLD_WIDTH, WORLD_HEIGHT))
        with self.viz_lock:
            vs = self.viz_surface
        if vs is not None:
            surf.blit(vs, (0, 0))
        else:
            self._draw_text(
                "시각화 그래프를 생성하고 있습니다...",
                (WORLD_WIDTH // 2 - 160, WORLD_HEIGHT // 2 - 12),
                self.font_lg,
                ACCENT_CYAN,
            )
        if self.is_rendering_viz:
            badge = pygame.Rect(WORLD_WIDTH - 210, 16, 194, 34)
            pygame.draw.rect(surf, (20, 26, 36), badge, border_radius=8)
            pygame.draw.rect(surf, ACCENT_YELLOW, badge, 1, border_radius=8)
            self._draw_text("● 그래프 갱신 중...", (badge.x + 18, badge.y + 8), self.font_sm, ACCENT_YELLOW)

    def _render_world(self) -> None:
        surf = self.canvas
        pygame.draw.rect(surf, BG_WORLD, (0, 0, WORLD_WIDTH, WORLD_HEIGHT))

        # Grid
        for gx in range(WALL_MARGIN, WORLD_WIDTH - WALL_MARGIN, 48):
            pygame.draw.line(surf, GRID_COL, (gx, WALL_MARGIN), (gx, WORLD_HEIGHT - WALL_MARGIN), 1)
        for gy in range(WALL_MARGIN, WORLD_HEIGHT - WALL_MARGIN, 48):
            pygame.draw.line(surf, GRID_COL, (WALL_MARGIN, gy), (WORLD_WIDTH - WALL_MARGIN, gy), 1)

        # Outer walls
        pygame.draw.rect(
            surf,
            WALL_COL,
            (WALL_MARGIN, WALL_MARGIN, WORLD_WIDTH - 2 * WALL_MARGIN, WORLD_HEIGHT - 2 * WALL_MARGIN),
            4,
        )

        m = self.sim.current_map
        sx, sy, _ = m.spawn
        gx, gy = m.goal

        # Dashed direct line Start -> Goal (visually shows obstacles blocking direct straight line!)
        self.ui.dashed_line((45, 58, 78), (sx, sy), (gx, gy), dash=10, gap=8, surf=surf)

        # Obstacles
        for rect in m.rects:
            r = pygame.Rect(int(rect.x), int(rect.y), int(rect.w), int(rect.h))
            pygame.draw.rect(surf, OBSTACLE_FILL, r, border_radius=6)
            pygame.draw.rect(surf, OBSTACLE_EDGE, r, 2, border_radius=6)

        for circle in m.circles:
            center = (int(circle.cx), int(circle.cy))
            pygame.draw.circle(surf, OBSTACLE_FILL, center, int(circle.r))
            pygame.draw.circle(surf, OBSTACLE_EDGE, center, int(circle.r), 2)

        # Start marker
        pygame.draw.circle(surf, (55, 70, 95), (int(sx), int(sy)), 22, 2)
        self._draw_text("START", (int(sx) - 22, int(sy) + 25), self.font_tag, TEXT_MUTED)

        # Goal marker (glowing target)
        gcenter = (int(gx), int(gy))
        pygame.draw.circle(surf, (22, 78, 52), gcenter, int(GOAL_RADIUS) + 10)
        pygame.draw.circle(surf, ACCENT_GREEN, gcenter, int(GOAL_RADIUS), 3)
        pygame.draw.circle(surf, ACCENT_YELLOW, gcenter, 10)
        self._draw_text("GOAL", (gcenter[0] - 19, gcenter[1] - int(GOAL_RADIUS) - 20), self.font_md, ACCENT_GREEN)

        # Draw A* Global Teacher Path ONLY during [1. Collect] live Teacher demonstration
        if self.tab == "collect" and self.expert_live and len(self.teacher.path) >= 2:
            pts = [(int(px), int(py)) for px, py in self.teacher.path]
            pygame.draw.lines(surf, (16, 185, 129), False, pts, 4)
            for px, py in pts[::3]:
                pygame.draw.circle(surf, ACCENT_CYAN, (px, py), 3)
            if self.teacher.lookahead_point is not None:
                lp = (int(self.teacher.lookahead_point[0]), int(self.teacher.lookahead_point[1]))
                pygame.draw.circle(surf, ACCENT_YELLOW, lp, 7, 2)

        # Render either 6-robot Race OR Single-robot with LiDAR rays
        if self.tab == "test" and self.test_mode == "race":
            self._render_race_robots(surf)
        else:
            self._render_single_robot(surf, self.sim.robot)

        if self.tab == "test" and self.test_mode == "race":
            mode = "동시 레이스 · 주행 중" if self.race_running else "동시 레이스"
            color = ui.CYAN
        elif self.tab == "test":
            mode = f"단일 주행 · {RACE_TAGS[self.single_model_key]}"
            color = RACE_COLORS[self.single_model_key]
        elif self.expert_live:
            mode, color = "자동 주행 · 데이터 수집 중", ui.CYAN
        elif self.recording:
            mode, color = "수동 주행 · 데이터 수집 중", ui.DANGER
        else:
            mode, color = "주행 맵", TEXT_MUTED
        label_width = self.font_sm.size(mode)[0]
        map_width = min(350, self.font_sm.size(m.name)[0])
        hud = pygame.Rect(WALL_MARGIN + 12, WALL_MARGIN + 12, label_width + map_width + 54, 34)
        self.ui.rrect(hud, ui.SURFACE)
        pygame.draw.circle(surf, color, (hud.x + 14, hud.centery), 4)
        self.ui.text(mode, (hud.x + 26, hud.centery), self.font_sm, color, anchor="midleft")
        self.ui.text(m.name, (hud.right - 12, hud.centery), self.font_sm, ui.TEXT, anchor="midright", max_w=map_width)

    def _render_single_robot(self, surf: pygame.Surface, ag: RobotState) -> None:
        if len(ag.trail) >= 2:
            pts = [(int(px), int(py)) for px, py in ag.trail]
            trail_color = RACE_COLORS[self.single_model_key] if self.tab == "test" else ui.CYAN
            pygame.draw.lines(surf, trail_color, False, pts, 3)

        rx, ry = ag.x, ag.y
        for i in range(self.sim.num_beams):
            ang = ag.theta + float(self.sim.beam_offsets[i])
            dist = float(ag.last_distances[i])
            norm = dist / MAX_LIDAR_RANGE
            ex = rx + math.cos(ang) * dist
            ey = ry + math.sin(ang) * dist
            if not all(math.isfinite(v) for v in (dist, ex, ey)):
                continue  # a malformed beam (NaN / Inf distance) is not drawn rather than crashing the frame
            col = ACCENT_RED if norm < 0.32 else (ACCENT_YELLOW if norm < 0.62 else ACCENT_GREEN)
            pygame.draw.line(surf, col, (int(rx), int(ry)), (int(ex), int(ey)), 1)
            if dist < MAX_LIDAR_RANGE - 1.5:
                pygame.draw.circle(surf, col, (int(ex), int(ey)), 3)

        rcenter = (int(rx), int(ry))
        body_col = (
            ACCENT_RED
            if ag.collided
            else (ACCENT_GREEN if ag.reached_goal else (ACCENT_CYAN if self.expert_live else ACCENT_BLUE))
        )
        if self.recording and not ag.collided:
            pygame.draw.circle(surf, ACCENT_CYAN if self.expert_live else ACCENT_RED, rcenter, int(ROBOT_RADIUS) + 6, 2)
        pygame.draw.circle(surf, body_col, rcenter, int(ROBOT_RADIUS))
        pygame.draw.circle(surf, TEXT_PRIMARY, rcenter, int(ROBOT_RADIUS), 2)

        hx = rx + math.cos(ag.theta) * (ROBOT_RADIUS + 6)
        hy = ry + math.sin(ag.theta) * (ROBOT_RADIUS + 6)
        pygame.draw.line(surf, TEXT_PRIMARY, rcenter, (int(hx), int(hy)), 3)

        if ag.reached_goal:
            self._draw_banner(surf, f"목적지 도달 성공! ({ag.elapsed:.2f}초)", ACCENT_GREEN)
        elif ag.collided:
            self._draw_banner(surf, "장애물 충돌!", ACCENT_RED)

    def _render_race_robots(self, surf: pygame.Surface) -> None:
        # Draw trails first so robots always sit cleanly on top
        for cont in self.race_contenders:
            st = cont.state
            if len(st.trail) >= 2:
                pts = [(int(px), int(py)) for px, py in st.trail]
                pygame.draw.lines(surf, cont.color, False, pts, 3)

        # Draw each robot with tag & status badge
        for cont in self.race_contenders:
            st = cont.state
            rc = (int(st.x), int(st.y))
            pygame.draw.circle(surf, cont.color, rc, int(ROBOT_RADIUS))
            pygame.draw.circle(surf, TEXT_PRIMARY, rc, int(ROBOT_RADIUS), 2)

            hx = st.x + math.cos(st.theta) * (ROBOT_RADIUS + 5)
            hy = st.y + math.sin(st.theta) * (ROBOT_RADIUS + 5)
            pygame.draw.line(surf, TEXT_PRIMARY, rc, (int(hx), int(hy)), 3)

            if st.collided:
                # Draw red X over crashed robot
                d = 10
                pygame.draw.line(surf, TEXT_PRIMARY, (rc[0] - d, rc[1] - d), (rc[0] + d, rc[1] + d), 3)
                pygame.draw.line(surf, TEXT_PRIMARY, (rc[0] - d, rc[1] + d), (rc[0] + d, rc[1] - d), 3)
                badge = f"{cont.tag} 충돌"
                b_col = ACCENT_RED
            elif st.reached_goal:
                pygame.draw.circle(surf, ACCENT_YELLOW, rc, int(ROBOT_RADIUS) + 6, 3)
                badge = f"{cont.tag} 도착"
                b_col = ACCENT_GREEN
            else:
                badge = cont.tag
                b_col = cont.color

            tag_surf = self.font_tag.render(badge, True, TEXT_PRIMARY)
            tr = pygame.Rect(rc[0] - tag_surf.get_width() // 2 - 5, rc[1] - int(ROBOT_RADIUS) - 22, tag_surf.get_width() + 10, 18)
            pygame.draw.rect(surf, (18, 22, 30), tr, border_radius=4)
            pygame.draw.rect(surf, b_col, tr, 1, border_radius=4)
            surf.blit(tag_surf, (tr.x + 5, tr.y + 2))

    def _draw_banner(self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]) -> None:
        ts = self.font_lg.render(text, True, TEXT_PRIMARY)
        rect = pygame.Rect(WORLD_WIDTH // 2 - ts.get_width() // 2 - 18, WORLD_HEIGHT - WALL_MARGIN - 50, ts.get_width() + 36, 36)
        pygame.draw.rect(surf, color, rect, border_radius=8)
        surf.blit(ts, (rect.x + 18, rect.y + 7))

    def _render_panel(self) -> None:
        kit = self.ui
        f = self.fonts
        x, w = WORLD_WIDTH + 24, PANEL_WIDTH - 48
        kit.begin(self._logical_mouse_pos(), time.monotonic())
        pygame.draw.rect(self.canvas, BG_PANEL, (WORLD_WIDTH, 0, PANEL_WIDTH, LOGICAL_HEIGHT))
        pygame.draw.line(self.canvas, ui.BORDER_STRONG, (WORLD_WIDTH, 0), (WORLD_WIDTH, LOGICAL_HEIGHT))
        kit.text("LiDAR 주행 실험실", (x, 20), f.h1)
        kit.text(self.hw.summary_text(), (x + w, 24), f.small, TEXT_MUTED,
                 anchor="topright", max_w=w - 190)
        options = [(f"tab_{key}", label, True) for key, label, _, _ in TABS]
        kit.segmented(pygame.Rect(x, 60, w, 44), options, f"tab_{self.tab}", font=f.body_b)
        title = next(title for key, _, title, _ in TABS if key == self.tab)
        kit.text(title, (x, 124), f.display)
        render_tab = {"collect": self._render_tab_collect, "train": self._render_tab_train,
                      "test": self._render_tab_test, "viz": self._render_tab_viz, "guide": self._render_tab_guide}
        render_tab[self.tab](x, 170, w)

        top = LOGICAL_HEIGHT - 72
        pygame.draw.line(self.canvas, ui.BORDER_STRONG, (x, top), (x + w, top))
        lines = kit.wrap(self.status_msg, f.body, w - 22)
        page_size = max(1, (LOGICAL_HEIGHT - top - 15) // f.body.get_linesize())
        if len(lines) > page_size:
            lines = kit.wrap(self.status_msg, f.body, w - 112)
            pages = (len(lines) + page_size - 1) // page_size
            page = self.status_page % pages
            kit.button(pygame.Rect(x + w - 84, top + 15, 84, 34), f"{page + 1}/{pages}", "status_next",
                       variant="ghost", icon="chev_r")
            lines = lines[page * page_size:(page + 1) * page_size]
        pygame.draw.circle(self.canvas, self.status_color, (x + 5, top + 22), 4)
        for i, line in enumerate(lines):
            kit.text(line, (x + 18, top + 12 + i * f.body.get_linesize()), f.body)
        kit.end()

    def _render_tab_collect(self, x: int, y: int, w: int) -> None:
        kit, f = self.ui, self.fonts
        kit.button(pygame.Rect(x + w - 196, y - 48, 196, 36), "처음이라면 · 사전 이론", "tab_guide",
                   variant="ghost", icon="chev_r", font=f.small_b)
        live = self.expert_live
        manual = self.recording and not live
        kit.button(pygame.Rect(x, y, w, 52), "데이터셋 자동 생성 중지 (실시간 주행)" if live else "데이터셋 자동 생성 시작 (실시간 주행)",
                   "toggle_expert_live", variant="primary", icon="stop" if live else "play", font=f.h1)
        left, right = columns(x, y + 62, w, 40, 2)
        kit.button(left, "샘플 3,000개 일괄 생성", "auto_fast", icon="plus")
        kit.button(right, "수동 주행 데이터 수집 중지" if manual else "수동 주행 데이터 수집 시작", "toggle_record",
                   active=manual, accent=ui.DANGER, icon="record")

        y += 124
        total = len(self.dataset)
        kit.card(pygame.Rect(x, y, w, 206), "데이터셋의 주행 샘플",
                 right="자동 주행 · 데이터 수집 중" if live else ("수동 주행 · 데이터 수집 중" if manual else "수집 대기"),
                 right_color=ui.CYAN if self.recording else TEXT_MUTED)
        kit.text(f"{total:,}", (x + 16, y + 44), f.num_xl)
        kit.text("LiDAR 25개 빔 + 목적지 정보 3개", (x + w - 16, y + 62),
                 f.small, TEXT_MUTED, anchor="topright")
        counts = self.dataset.class_counts()
        colors = (ui.CYAN, ui.SUCCESS, ui.VIOLET)
        bar = pygame.Rect(x + 16, y + 96, w - 32, 10)
        kit.rrect(bar, ui.INSET, 3)
        offset = 0
        for count, color in zip(counts, colors):
            left = bar.x + bar.w * offset // max(total, 1)
            offset += count
            right = bar.x + bar.w * offset // max(total, 1)
            pygame.draw.rect(self.canvas, color, (left, bar.y, right - left, bar.h))
        for r, label, count, color in zip(columns(x + 16, y + 116, w - 32, 30, 3),
                                          ("좌회전", "직진", "우회전"), counts, colors):
            kit.text(f"{label}  {count:,}  ({count / max(total, 1):.0%})",
                     r.topleft, f.small_b, color, max_w=r.w)
        for r, (label, key, icon) in zip(columns(x + 16, y + 155, w - 32, 36, 3),
                                         (("데이터셋 저장", "save_data", "save"), ("데이터셋 불러오기", "load_data", "folder"),
                                          ("수집 샘플 비우기", "clear_data", "trash"))):
            kit.button(r, label, key, variant="ghost", icon=icon)

        y += 236
        kit.text("주행 맵", (x, y), f.h1)
        self._render_map_controls(x, y + 36, w)
        kit.text("지도 좌클릭으로 출발 위치를, 우클릭으로 목적지를 옮깁니다.",
                 (x, y + 86), f.small, TEXT_MUTED)
        y += 154
        kit.text("수동 주행", (x, y), f.h2)
        for i, (key, label) in enumerate((("W A D", "직진·좌회전·우회전 (방향키 가능)"), ("Space", "수동 데이터 수집 시작·중지"))):
            pos = x + i * (w // 2)
            width = kit.keycap(pos, y + 34, key)
            kit.text(label, (pos + width + 8, y + 36), f.small, TEXT_MUTED)

    def _render_tab_train(self, x: int, y: int, w: int) -> None:
        kit, f = self.ui, self.fonts
        busy = self.is_training or self.is_benchmarking or self.infer_switching
        with self.train_lock:
            current = self.train_current_model
            stage = self.train_stage
            queue = tuple(self.train_queue)
            done = set(self.train_done_set)
            hist = self.train_history
        label = "5종 모델 일괄 학습"
        if current:
            label = f"{SHORT_MODEL_LABELS[current]} 학습 중   {stage}"
        kit.button(pygame.Rect(x, y, w, 52), label, "train_all", variant="primary",
                   disabled=busy, icon="chart", font=f.h1)
        y += 66
        summary = f"샘플 {len(self.dataset):,}개"
        if queue and hist:
            summary += f"    {len(done)}/{len(queue)}개 학습 완료"
        kit.card(pygame.Rect(x, y, w, 248), "검증 성능 비교", right=summary, right_color=TEXT_MUTED)
        if current:
            fraction = len(done)
            if hist and current not in done and hist.model_type == current and hist.total_epochs:
                fraction += hist.current_epoch / hist.total_epochs
            kit.progress(pygame.Rect(x + 16, y + 40, w - 32, 4), fraction / max(1, len(queue)), ui.CYAN)
        xs = [x + 32, x + 190, x + 325, x + 435]
        for pos, heading in zip(xs, ("모델", "검증 정확도 ↑", "학습 시간", "추론 시간")):
            kit.text(heading, (pos, y + 51), f.small_b, TEXT_MUTED)
        best = self._best_model()
        for i, model in enumerate(MODEL_TYPES):
            ry = y + 82 + i * 32
            tm = self.trained_models.get(model)
            pygame.draw.line(self.canvas, ui.BORDER, (x + 16, ry - 7), (x + w - 16, ry - 7))
            pygame.draw.circle(self.canvas, RACE_COLORS[model], (x + 20, ry + 8), 4)
            kit.text(MODEL_LABELS[model], (xs[0], ry), f.body)
            if current == model:
                kit.text("학습 중", (xs[1], ry), f.body_b, ui.WARNING)
            elif current and model in queue and model not in done:
                kit.text("대기 중", (xs[1], ry), f.body, TEXT_MUTED)
            elif tm:
                val = f"{tm.val_metric:.1f}%"
                kit.text(val, (xs[1], ry), f.body_b, ui.SUCCESS if tm is best else ui.TEXT)
                kit.text(f"{tm.train_time:.2f} s", (xs[2], ry), f.body)
                kit.text(f"{tm.infer_ms:.2f} ms", (xs[3], ry), f.body)
            else:
                kit.text("미학습", (xs[1], ry), f.body, TEXT_MUTED)
        y += 264
        goal_row = pygame.Rect(x, y, w, 38)
        kit.rrect(goal_row, ui.SURFACE)
        kit.text("학습 목표 · 좌회전 / 직진 / 우회전 (3클래스 분류)", (goal_row.x + 16, goal_row.centery), f.body,
                 ui.TEXT_2, anchor="midleft", max_w=goal_row.w - 32)
        y += 46
        for r, model in zip(columns(x, y, w, 34, len(MODEL_TYPES)), MODEL_TYPES):
            kit.model_chip(r, SHORT_MODEL_LABELS[model], f"mtype_{model}", RACE_COLORS[model],
                           model == self.selected_model_type, disabled=busy)
        y += 42
        left, right = columns(x, y, w, 38, 2)
        kit.button(left, f"{SHORT_MODEL_LABELS[self.selected_model_type]} 개별 학습", "start_train", disabled=busy)
        kit.button(right, "학습 설정", "train_settings", active=self.train_settings_open,
                   icon="chev_u" if self.train_settings_open else "chev_d")
        y += 48
        if self.train_settings_open:
            a, b = columns(x, y, w, 32, 2)
            kit.value_chip(a, "학습 장치", self.train_device.upper(), "toggle_train_dev", busy or not self.hw.has_cuda)
            kit.value_chip(b, "에포크", str(self.epochs), "cycle_epochs", busy)
            a, b = columns(x, y + 38, w, 32, 2)
            kit.toggle(a, "좌우 반전 증강", self.mirror_augment, "toggle_mirror", busy)
            kit.toggle(b, "클래스 균형", self.class_balance, "toggle_balance", busy)
            y += 80
        self._render_loss(x, y, w, hist)

    def _render_loss(self, x: int, y: int, w: int, hist: Optional[TrainHistory]) -> None:
        kit, f = self.ui, self.fonts
        title = "손실 곡선"
        if hist:
            title += f" · {SHORT_MODEL_LABELS[hist.model_type]}"
        kit.text(title, (x, y), f.h2)
        if not hist or not hist.train_losses or not hist.val_losses:
            kit.text("학습이 진행되면 훈련·검증 손실이 표시됩니다.", (x, y + 30), f.small, TEXT_MUTED)
            return
        train, val = list(hist.train_losses), list(hist.val_losses)
        kit.text(f"훈련 {train[-1]:.4f}", (x + w - 196, y), f.small_b, ui.CYAN)
        kit.text(f"검증 {val[-1]:.4f}", (x + w - 92, y), f.small_b, ui.WARNING)
        plot = pygame.Rect(x, y + 26, w, min(98, LOGICAL_HEIGHT - 84 - y - 26))
        kit.rrect(plot, ui.SURFACE)
        if len(train) == 1:
            kit.text(f"에포크 {hist.current_epoch}/{hist.total_epochs}", plot.center, f.body, TEXT_MUTED, anchor="center")
            return
        peak = max(max(train), max(val), 1e-6)
        for values, color in ((train, ui.CYAN), (val, ui.WARNING)):
            if len(values) >= 2:
                pts = [(plot.left + 6 + i * (plot.w - 12) / max(hist.total_epochs - 1, len(values) - 1),
                        plot.bottom - 5 - value / peak * (plot.h - 10)) for i, value in enumerate(values)]
                pygame.draw.lines(self.canvas, color, False, pts, 2)

    def _render_tab_test(self, x: int, y: int, w: int) -> None:
        kit, f = self.ui, self.fonts
        race = self.test_mode == "race"
        running = self.race_running if race else self.single_running
        label = ("동시 레이스" if race else "단일 주행") + (" 중지" if running else " 시작")
        kit.button(pygame.Rect(x, y, w, 52), label, "toggle_test", variant="primary",
                   icon="stop" if running else "play", font=f.h1,
                   disabled=not running and self.test_start_blocked)
        y += 62
        kit.segmented(pygame.Rect(x, y, w, 38),
                      [("mode_race", "동시 레이스", True), ("mode_single", "단일 주행", True)],
                      f"mode_{self.test_mode}")
        y += 48
        if race:
            kit.card(pygame.Rect(x, y, w, 222), "참가 모델", right=f"지도 없는 주행   {len(self.race_contenders)}대")
            for i, cont in enumerate(self.race_contenders):
                ry = y + 46 + i * 28
                st = cont.state
                if st.reached_goal:
                    status, color = f"도착  {st.elapsed:.1f}초", ui.SUCCESS
                elif st.collided:
                    status, color = f"충돌  {st.elapsed:.1f}초", ui.DANGER
                elif st.elapsed >= RACE_TIMEOUT_SEC:
                    status, color = "시간 초과", ui.WARNING
                elif self.race_running:
                    status, color = f"주행 중  {st.elapsed:.1f}초", ui.CYAN
                else:
                    status, color = "정지" if st.elapsed else "출발 대기", TEXT_MUTED
                pygame.draw.circle(self.canvas, cont.color, (x + 20, ry + 8), 5)
                kit.text(cont.tag, (x + 35, ry), f.body_b, cont.color)
                kit.text(cont.label, (x + 108, ry), f.body)
                kit.text(status, (x + 302, ry), f.body_b, color)
                kit.text(f"{st.distance:.0f} px", (x + w - 16, ry), f.body, TEXT_MUTED, anchor="topright")
        else:
            self._render_predictions(x, y, w)
        y += 234
        self._render_map_controls(x, y, w)
        y += 48
        acc_choices = [("torch_cpu", "PyTorch CPU", True), ("torch_cuda", "CUDA GPU", self.hw.has_cuda),
                       ("ov_cpu", "OV CPU", "CPU" in self.hw.ov_devices), ("ov_npu", "OV NPU", self.hw.has_npu)]
        if self.hw.has_ov_gpu:
            acc_choices.insert(3, ("ov_gpu", "OV GPU", True))
        kit.text("추론 장치", (x, y + 10), f.small_b)
        kit.segmented(pygame.Rect(x + 76, y, w - 76, 36),
                      [(f"infer_{key}", name, available) for key, name, available in acc_choices],
                      f"infer_{self.infer_backend}", busy=self.infer_switching or self.is_training or self.is_benchmarking)
        y += 54
        kit.card(pygame.Rect(x, y, w, LOGICAL_HEIGHT - 84 - y), "20맵 평가", right="목적지 도달률")
        left, right = columns(x + 16, y + 41, w - 32, 36, 2)
        completed, total, _ = self.bench_progress
        label = f"평가 중  {completed}/{total}" if self.is_benchmarking else "20맵 평가 실행"
        kit.button(left, label, "run_bench", disabled=self.is_benchmarking or self.is_training or self.infer_switching,
                   icon="flag")
        kit.button(right, "CSV 저장", "export_bench", disabled=not self.bench_results or self.is_benchmarking, icon="save")
        if not self.bench_results:
            kit.text("같은 랜덤 맵 20개에서 각 모델의 도착·충돌 횟수를 비교합니다.",
                     (x + 16, y + 100), f.small, TEXT_MUTED)
        for i, entry in enumerate(self.bench_results):
            ry = y + 89 + 24 * i
            color = RACE_COLORS[entry.key]
            kit.text(RACE_TAGS[entry.key], (x + 16, ry), f.small_b, color)
            kit.progress(pygame.Rect(x + 84, ry + 5, w - 292, 9), entry.success_rate / 100, color)
            kit.text(f"{entry.success_rate:.0f}%   도착 {entry.goals}/{entry.num_maps}   충돌 {entry.crashes}",
                     (x + w - 16, ry), f.small, anchor="topright")

    def _render_predictions(self, x: int, y: int, w: int) -> None:
        kit, f = self.ui, self.fonts
        keys = ("baseline", *MODEL_TYPES)
        for r, key in zip(columns(x, y, w, 34, len(keys)), keys):
            kit.model_chip(r, RACE_TAGS[key], f"pick_single_{key}", RACE_COLORS[key], self.single_model_key == key,
                           disabled=key != "baseline" and key not in self.trained_models)
        y += 44
        tag = RACE_TAGS[self.single_model_key]
        caption = "규칙 출력" if self.single_model_key == "baseline" else "행동 확률"
        kit.card(pygame.Rect(x, y, w, 178), "주행 예측값", right=f"{tag} · {caption}")
        info = self.last_pred_info
        if "probs" in info:
            chosen = int(info["action"])  # type: ignore[call-overload]
            for i, (name, p, color) in enumerate(zip(ACTION_NAMES, info["probs"], (ui.CYAN, ui.SUCCESS, ui.VIOLET))):
                ry = y + 50 + 36 * i
                kit.text(name, (x + 16, ry), f.body_b if i == chosen else f.body, ui.TEXT if i == chosen else TEXT_MUTED)
                kit.progress(pygame.Rect(x + 88, ry + 4, w - 178, 12), float(p), color)
                kit.text(f"{p:.1%}", (x + w - 16, ry), f.body, anchor="topright")
        else:
            kit.text("단일 주행을 시작하면 선택한 모델의 출력이 표시됩니다.",
                     (x + 16, y + 72), f.body, TEXT_MUTED)

    def _render_tab_viz(self, x: int, y: int, w: int) -> None:
        kit, f = self.ui, self.fonts
        busy = self.is_rendering_viz
        for i, (key, label) in enumerate(CHART_CHOICES):
            row = pygame.Rect(x, y + i * 44, w, 38)
            kit.button(row, label, f"viz_chart_{key}", active=key == self.viz_chart, disabled=busy,
                       accent=ui.CYAN, align_left=True)
        y += 244
        if self.viz_chart == "pca3d":
            kit.card(pygame.Rect(x, y, w, 98), "3D 시점",
                     right=f"드래그 회전   고도 {self.viz_elev:.0f}° / 방위 {self.viz_azim:.0f}°")
            options = (("왼쪽", "left"), ("오른쪽", "right"), ("위", "up"), ("아래", "down"))
            for r, (label, key) in zip(columns(x + 16, y + 44, w - 32, 34, 4), options):
                kit.button(r, label, f"viz_rot_{key}", disabled=busy)
        elif self.viz_chart != "summary":
            choices = self._viz_model_choices()
            kit.card(pygame.Rect(x, y, w, 98), "분석 모델")
            for r, model in zip(columns(x + 16, y + 46, w - 32, 36, len(MODEL_TYPES)), MODEL_TYPES):
                kit.model_chip(r, SHORT_MODEL_LABELS[model], f"viz_model_{model}", RACE_COLORS[model],
                               model == self.viz_model and model in choices, disabled=busy or model not in choices)
        y += 192
        kit.text("그래프 내보내기", (x, y), f.h1)
        kit.text("PNG   160 DPI", (x + w, y + 3), f.small_b, TEXT_MUTED, anchor="topright")
        left, right = columns(x, y + 38, w, 44, 2)
        kit.button(left, "현재 그래프 PNG 저장", "viz_save_one", variant="primary", icon="image", disabled=busy)
        kit.button(right, f"전체 {len(CHART_CHOICES)}종 PNG 저장", "viz_save_all", icon="save", disabled=busy)
        kit.button(pygame.Rect(x, y + 94, w, 36), "최신 데이터로 새로고침", "viz_refresh",
                   variant="ghost", icon="refresh", disabled=busy)

    def _render_tab_guide(self, x: int, y: int, w: int) -> None:
        kit, f = self.ui, self.fonts
        if self.reader.hovered_link(kit.mouse):
            kit.claim_hand()
        chapter = CHAPTERS[self.reader.chapter]
        for i, item in enumerate(CHAPTERS):
            kit.button(pygame.Rect(x, y + i * 54, w, 46), f"{i + 1}    {item.menu}", f"guide_{i}",
                       active=i == self.reader.chapter, align_left=True, accent=ui.CYAN, font=f.h1)
        y += 290
        left, right = columns(x, y, w, 40, 2)
        kit.button(left, "이전 장", "guide_prev", icon="chev_l", disabled=self.reader.chapter == 0)
        kit.button(right, "다음 장", "guide_next", icon="chev_r", disabled=self.reader.chapter == len(CHAPTERS) - 1)
        y += 84
        kit.text("생각해 볼 질문", (x, y), f.h1, ui.CYAN)
        for line in kit.wrap(chapter.question, f.h1, w):
            y += 31
            kit.text(line, (x, y), f.h1)
        kit.button(pygame.Rect(x, y + 60, w, 46), chapter.action, "guide_lab", variant="primary", icon="play")

    def _render_map_controls(self, x: int, y: int, w: int) -> None:
        options = (("새 랜덤 맵 · N", "new_random_map"), ("A · 기본 회피", "preset_0"), ("B · 복합 난코스", "preset_1"),
                   ("출발점 복귀 · R", "reset_robot"))
        for rect, (label, key) in zip(columns(x, y, w, 36, len(options)), options):
            self.ui.button(rect, label, key)

    def _draw_text(self, text: str, pos: Tuple[int, int], font: pygame.font.Font, color: ui.Color) -> None:
        self.ui.text(text, pos, font, color)
