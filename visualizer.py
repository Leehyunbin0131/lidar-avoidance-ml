"""High-resolution Dark-Themed Matplotlib 2D/3D ML Visualizations rendered onto Pygame Surfaces."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ui_kit import MODEL_COLORS

from dataset import ACTION_NAMES, DrivingDataset, velocity_to_action
from model import (
    BASELINE_LABEL,
    LEGACY_BASELINE_LABEL,
    MODEL_LABELS,
    MODEL_TYPES,
    BenchmarkEntry,
    ModelType,
    TaskType,
    TrainedModel,
    predict_command,
)
from simulator import MAX_ANGULAR_SPEED, MAX_LINEAR_SPEED, NUM_BEAMS, WORLD_HEIGHT, WORLD_WIDTH

ChartType = Literal["pca3d", "scatter2d", "regression", "classification", "importance", "summary"]

CHART_CHOICES: Tuple[Tuple[ChartType, str], ...] = (
    ("pca3d", "3D PCA · 주행 샘플 분포"),
    ("scatter2d", "2D 분포와 결정 경계"),
    ("regression", "회귀 · 실제값과 예측값"),
    ("classification", "분류 · 혼동 행렬과 F1"),
    ("importance", "28차원 특징 중요도"),
    ("summary", "모델 종합 비교"),
)

# Register Korean font cleanly
_FONT_CONFIGURED = False


def _configure_matplotlib_style() -> None:
    global _FONT_CONFIGURED
    if _FONT_CONFIGURED:
        return

    candidate_paths = [
        Path("/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for extra_fp in (
        Path("/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumSquareEB.ttf"),
    ):
        if extra_fp.exists():
            try:
                fm.fontManager.addfont(str(extra_fp))
            except Exception:
                pass

    for fp in candidate_paths:
        if fp.exists():
            try:
                fm.fontManager.addfont(str(fp))
                prop = fm.FontProperties(fname=str(fp))
                plt.rcParams["font.family"] = prop.get_name()
                break
            except Exception:
                pass

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#11151c"
    plt.rcParams["axes.facecolor"] = "#181e28"
    plt.rcParams["axes.edgecolor"] = "#3a4556"
    plt.rcParams["axes.labelcolor"] = "#eef2f9"
    plt.rcParams["xtick.color"] = "#96a5bc"
    plt.rcParams["ytick.color"] = "#96a5bc"
    plt.rcParams["text.color"] = "#eef2f9"
    plt.rcParams["grid.color"] = "#283244"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.65
    _FONT_CONFIGURED = True


# Palette matching app.py
C_CYAN = "#22d3ee"
C_GREEN = "#22c55e"
C_PURPLE = "#a855f7"
C_BLUE = "#3b82f6"
C_YELLOW = "#f59e0b"
C_RED = "#ef4444"
C_ORANGE = "#f97316"
C_MUTED = "#94a3b8"

CLASS_COLORS = [C_CYAN, C_GREEN, C_PURPLE]
MODEL_HEX = {key: "#{:02x}{:02x}{:02x}".format(*rgb) for key, rgb in MODEL_COLORS.items()}


def _get_val_split(dataset: DrivingDataset, max_points: int = 1200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic 20% validation split (same RNG seed=42 as model.py)."""
    x_all, y_vel_all, y_act_all = dataset.to_arrays(mirror_augment=False)
    n = len(x_all)
    if n == 0:
        return x_all, y_vel_all, y_act_all
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    val_size = max(1, int(n * 0.2))
    val_idx = indices[:val_size]
    if len(val_idx) > max_points:
        val_idx = val_idx[:max_points]
    return x_all[val_idx], y_vel_all[val_idx], y_act_all[val_idx]


def _batch_predict_raw(trained: TrainedModel, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fast batch predictions returning (pred_vels_norm[N, 2], pred_actions[N])."""
    n = len(x)
    if n == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)

    if trained.backend == "torch":
        net = trained.estimator
        dev = next(net.parameters()).device
        with torch.no_grad():
            out = net(torch.tensor(x, dtype=torch.float32, device=dev)).detach().cpu().numpy()
        if trained.task_type == "classification":
            acts = np.argmax(out, axis=1).astype(np.int64)
            vels = np.zeros((n, 2), dtype=np.float32)
            vels[:, 0] = np.where(acts == 1, 0.92, 0.72)
            vels[:, 1] = np.where(acts == 0, -0.82, np.where(acts == 2, 0.82, 0.0))
            return vels, acts
        else:
            vels = np.clip(out, -1.0, 1.0).astype(np.float32)
            acts = np.array([velocity_to_action(float(w) * MAX_ANGULAR_SPEED) for w in vels[:, 1]], dtype=np.int64)
            return vels, acts
    else:
        est = trained.estimator
        if trained.task_type == "classification":
            acts = est.predict(x).astype(np.int64)
            vels = np.zeros((n, 2), dtype=np.float32)
            vels[:, 0] = np.where(acts == 1, 0.92, 0.72)
            vels[:, 1] = np.where(acts == 0, -0.82, np.where(acts == 2, 0.82, 0.0))
            return vels, acts
        else:
            vels = np.clip(est.predict(x), -1.0, 1.0).astype(np.float32)
            acts = np.array([velocity_to_action(float(w) * MAX_ANGULAR_SPEED) for w in vels[:, 1]], dtype=np.int64)
            return vels, acts


def _empty_figure(title: str, subtitle: str) -> plt.Figure:
    fig = plt.figure(figsize=(10.8, 9.6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#181e28")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=20, fontweight="bold", color="#eef2f9")
    ax.text(0.5, 0.45, subtitle, ha="center", va="center", fontsize=14, color="#96a5bc")
    return fig


def build_figure(
    chart_type: ChartType,
    dataset: DrivingDataset,
    trained_models: Dict[Tuple[ModelType, TaskType], TrainedModel],
    selected_model: ModelType,
    task_type: TaskType,
    bench_results: List[BenchmarkEntry],
    elev: float = 24.0,
    azim: float = 42.0,
    color_by_continuous: bool = False,
) -> plt.Figure:
    """Build a crisp 1080x960 (10.8x9.6 in @ 100 dpi) Matplotlib Figure for the chosen chart."""
    _configure_matplotlib_style()

    if chart_type == "pca3d":
        return _plot_pca3d(dataset, elev=elev, azim=azim, color_by_continuous=color_by_continuous)
    if chart_type == "scatter2d":
        return _plot_scatter2d(dataset, selected_model=selected_model)
    if chart_type == "regression":
        return _plot_regression(dataset, trained_models, selected_model=selected_model)
    if chart_type == "classification":
        return _plot_classification(dataset, trained_models, selected_model=selected_model, task_type=task_type)
    if chart_type == "importance":
        return _plot_importance(dataset, trained_models, selected_model=selected_model, task_type=task_type)
    return _plot_summary(trained_models, task_type=task_type, bench_results=bench_results)


# ----------------------------------------------------------------------
# Chart 1: 3D PCA Scatter of 28-Dim Dataset
# ----------------------------------------------------------------------
def _plot_pca3d(
    dataset: DrivingDataset,
    elev: float = 24.0,
    azim: float = 42.0,
    color_by_continuous: bool = False,
) -> plt.Figure:
    if len(dataset) < 15:
        return _empty_figure("데이터셋이 부족합니다", "[1 수집] 탭에서 먼저 데이터를 수집해 주세요 (최소 15개 이상).")

    x, y_vel, y_act = dataset.to_arrays(mirror_augment=False)
    rng = np.random.default_rng(42)
    if len(x) > 2500:
        idx = rng.choice(len(x), size=2500, replace=False)
        x, y_vel, y_act = x[idx], y_vel[idx], y_act[idx]

    x_std = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(x_std)
    var = pca.explained_variance_ratio_ * 100.0

    fig = plt.figure(figsize=(10.8, 9.6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#11151c")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#2b3648")
    ax.yaxis.pane.set_edgecolor("#2b3648")
    ax.zaxis.pane.set_edgecolor("#2b3648")

    if color_by_continuous:
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=y_vel[:, 1],
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            s=14,
            alpha=0.78,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.58, pad=0.08)
        cbar.set_label("조향 각속도 w_norm (-좌회전 / +우회전)", fontsize=12)
        subtitle = "색상: 연속 조향 각속도 (w_norm) — 회귀 타깃 분포"
    else:
        for cls_i, (name, col) in enumerate(zip(ACTION_NAMES, CLASS_COLORS)):
            mask = y_act == cls_i
            if np.any(mask):
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    coords[mask, 2],
                    c=col,
                    label=f"{name} (n={int(np.sum(mask)):,})",
                    s=15,
                    alpha=0.75,
                    edgecolors="none",
                )
        ax.legend(loc="upper right", facecolor="#181e28", edgecolor="#3a4556", fontsize=12, markerscale=1.8)
        subtitle = "색상: 행동 클래스 (좌회전 / 직진 / 우회전) — 분류 클러스터 분포"

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=12, labelpad=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=12, labelpad=10)
    ax.set_zlabel(f"PC3 ({var[2]:.1f}%)", fontsize=12, labelpad=10)
    ax.set_title(
        f"28차원 라이다+목적지 데이터셋 3D PCA 산점도 (표시 {len(x):,} / 전체 {len(dataset):,}개)\n"
        f"{subtitle}  |  누적 설명 분산: {sum(var):.1f}%  |  마우스 드래그로 3D 회전 가능",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Chart 2: 2D PCA Scatter + Physical 2D Feature Decision Boundary
# ----------------------------------------------------------------------
def _plot_scatter2d(dataset: DrivingDataset, selected_model: ModelType) -> plt.Figure:
    if len(dataset) < 30:
        return _empty_figure("데이터셋이 부족합니다", "[1 수집] 탭에서 먼저 데이터를 수집해 주세요.")

    x, y_vel, y_act = dataset.to_arrays(mirror_augment=False)
    rng = np.random.default_rng(42)
    if len(x) > 2200:
        idx = rng.choice(len(x), size=2200, replace=False)
        x, y_vel, y_act = x[idx], y_vel[idx], y_act[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 9.6), dpi=100)
    fig.suptitle(
        f"데이터셋 2D 분포 및 [{MODEL_LABELS[selected_model]}] 비선형 결정 경계 시각화",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # Left: 2D PCA Projection
    ax0 = axes[0]
    coords2 = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(x))
    for cls_i, (name, col) in enumerate(zip(ACTION_NAMES, CLASS_COLORS)):
        mask = y_act == cls_i
        if np.any(mask):
            ax0.scatter(coords2[mask, 0], coords2[mask, 1], c=col, label=name, s=14, alpha=0.68, edgecolors="none")
    ax0.set_title("2D PCA 투영 (28차원 → 2차원)", fontsize=13, fontweight="bold")
    ax0.set_xlabel("PC1", fontsize=11)
    ax0.set_ylabel("PC2", fontsize=11)
    ax0.grid(True)
    ax0.legend(facecolor="#181e28", edgecolor="#3a4556", fontsize=10, markerscale=1.6)

    # Right: Physical 2D axes + Decision Boundary of selected model family
    ax1 = axes[1]
    mid = NUM_BEAMS // 2
    # Feature A: Right vs Left open-space balance
    f_balance = np.mean(x[:, mid + 1 : NUM_BEAMS], axis=1) - np.mean(x[:, :mid], axis=1)
    # Feature B: Relative Goal sin(angle)
    f_goal_sin = x[:, NUM_BEAMS]
    x2 = np.column_stack([f_balance, f_goal_sin])

    if len(np.unique(y_act)) >= 2:
        if selected_model == "linear":
            clf2 = Pipeline([("sc", StandardScaler()), ("c", LogisticRegression(max_iter=300))])
        elif selected_model == "knn":
            clf2 = Pipeline([("sc", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=9))])
        elif selected_model == "rf":
            clf2 = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42)
        else:
            clf2 = Pipeline([("sc", StandardScaler()), ("c", MLPClassifier(hidden_layer_sizes=(48, 32), max_iter=150, random_state=42))])

        clf2.fit(x2, y_act)
        gx = np.linspace(float(np.percentile(f_balance, 1)) - 0.1, float(np.percentile(f_balance, 99)) + 0.1, 160)
        gy = np.linspace(-1.05, 1.05, 160)
        xx, yy = np.meshgrid(gx, gy)
        zz = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        from matplotlib.colors import ListedColormap
        cmap_bg = ListedColormap(["#124559", "#14452f", "#3b1c5a"])
        ax1.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_bg, alpha=0.58)
        ax1.contour(xx, yy, zz, levels=[0.5, 1.5], colors="#e2e8f0", linewidths=1.2, alpha=0.85)

    for cls_i, (name, col) in enumerate(zip(ACTION_NAMES, CLASS_COLORS)):
        mask = y_act == cls_i
        if np.any(mask):
            ax1.scatter(f_balance[mask], f_goal_sin[mask], c=col, label=name, s=12, alpha=0.72, edgecolors="#0b0e14", linewidths=0.3)

    ax1.set_title(f"물리 특징 2D 결정 경계 ({MODEL_LABELS[selected_model]})", fontsize=13, fontweight="bold")
    ax1.set_xlabel("좌우 개방도 차이 (우측 라이다 - 좌측 라이다)", fontsize=11)
    ax1.set_ylabel("목적지 상대 방향 sin(Δθ)  (-좌 / +우)", fontsize=11)
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ----------------------------------------------------------------------
# Chart 3: Regression Predicted vs Actual Scatter & Residuals
# ----------------------------------------------------------------------
def _plot_regression(
    dataset: DrivingDataset,
    trained_models: Dict[Tuple[ModelType, TaskType], TrainedModel],
    selected_model: ModelType,
) -> plt.Figure:
    tm = trained_models.get((selected_model, "regression"))
    if tm is None:
        # Fallback to any available trained regression model
        for m in MODEL_TYPES:
            if (m, "regression") in trained_models:
                tm = trained_models[(m, "regression")]
                selected_model = m
                break
    if tm is None or len(dataset) < 20:
        return _empty_figure(
            "학습된 회귀 모델이 없습니다",
            "[2 학습] 탭에서 '회귀 · 속도 예측' 모델을 먼저 학습해 주세요.",
        )

    x_val, y_vel_val, _ = _get_val_split(dataset, max_points=1200)
    pred_vel, _ = _batch_predict_raw(tm, x_val)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.6), dpi=100)
    fig.suptitle(
        f"회귀 모델 정밀 분석: [{MODEL_LABELS[selected_model]}] 실제값 vs 예측값 (Val N={len(x_val):,})",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # 1. Linear velocity v_norm scatter
    ax = axes[0, 0]
    r2_v = r2_score(y_vel_val[:, 0], pred_vel[:, 0])
    mae_v = float(np.mean(np.abs(pred_vel[:, 0] - y_vel_val[:, 0])))
    ax.scatter(y_vel_val[:, 0], pred_vel[:, 0], c=C_GREEN, s=16, alpha=0.55, edgecolors="none")
    ax.plot([0.2, 1.0], [0.2, 1.0], color=C_RED, linestyle="--", linewidth=2, label="이상적 예측 (y = x)")
    ax.set_title(f"선속도 (v_norm) 예측 산점도 | R²={r2_v:.3f}, MAE={mae_v:.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("기록된 선속도 (정규화)")
    ax.set_ylabel("예측 선속도 (학습 모델)")
    ax.legend(facecolor="#181e28", edgecolor="#3a4556", fontsize=10)
    ax.grid(True)

    # 2. Angular velocity w_norm scatter
    ax = axes[0, 1]
    r2_w = r2_score(y_vel_val[:, 1], pred_vel[:, 1])
    mae_w = float(np.mean(np.abs(pred_vel[:, 1] - y_vel_val[:, 1])))
    err_w = np.abs(pred_vel[:, 1] - y_vel_val[:, 1])
    sc = ax.scatter(y_vel_val[:, 1], pred_vel[:, 1], c=err_w, cmap="plasma", vmin=0.0, vmax=0.45, s=16, alpha=0.68, edgecolors="none")
    ax.plot([-1.0, 1.0], [-1.0, 1.0], color=C_RED, linestyle="--", linewidth=2, label="이상적 예측 (y = x)")
    fig.colorbar(sc, ax=ax, shrink=0.82, label="|오차|")
    ax.set_title(f"조향 각속도 (w_norm) 예측 산점도 | R²={r2_w:.3f}, MAE={mae_w:.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("실제 각속도 (-좌 / +우)")
    ax.set_ylabel("예측 각속도 (-좌 / +우)")
    ax.legend(facecolor="#181e28", edgecolor="#3a4556", fontsize=10)
    ax.grid(True)

    # 3. Residual error histogram
    ax = axes[1, 0]
    residuals = pred_vel[:, 1] - y_vel_val[:, 1]
    ax.hist(residuals, bins=36, color=C_CYAN, alpha=0.78, edgecolor="#0b0e14")
    ax.axvline(0.0, color=C_RED, linestyle="--", linewidth=2)
    ax.set_title(f"조향 예측 오차(잔차) 분포 | 평균={np.mean(residuals):+.4f}, 표준편차={np.std(residuals):.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("예측 오차 (pred_w - true_w)")
    ax.set_ylabel("샘플 수")
    ax.grid(True)

    # 4. All trained regression models MAE comparison
    ax = axes[1, 1]
    reg_items = [(m, trained_models[(m, "regression")]) for m in MODEL_TYPES if (m, "regression") in trained_models]
    labels = [MODEL_LABELS[m] for m, _ in reg_items]
    maes = [t.val_metric for _, t in reg_items]
    colors = [MODEL_HEX[m] for m, _ in reg_items]
    bars = ax.bar(labels, maes, color=colors, edgecolor="#eef2f9", linewidth=0.8)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(maes) * 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("학습된 회귀 모델별 Val MAE 비교 (낮을수록 우수)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Val MAE")
    ax.set_ylim(0, max(maes) * 1.18 if maes else 1.0)
    ax.grid(True, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ----------------------------------------------------------------------
# Chart 4: Classification Confusion Matrix & Per-Class F1
# ----------------------------------------------------------------------
def _plot_classification(
    dataset: DrivingDataset,
    trained_models: Dict[Tuple[ModelType, TaskType], TrainedModel],
    selected_model: ModelType,
    task_type: TaskType,
) -> plt.Figure:
    tm = trained_models.get((selected_model, task_type))
    if tm is None:
        for tt in (task_type, "classification", "regression"):
            for m in (selected_model, *MODEL_TYPES):
                if (m, tt) in trained_models:
                    tm = trained_models[(m, tt)]
                    selected_model = m
                    break
            if tm is not None:
                break
    if tm is None or len(dataset) < 20:
        return _empty_figure("학습된 모델이 없습니다", "[2 학습] 탭에서 먼저 모델을 학습해 주세요.")

    x_val, _, y_act_val = _get_val_split(dataset, max_points=1500)
    _, pred_act = _batch_predict_raw(tm, x_val)

    cm = confusion_matrix(y_act_val, pred_act, labels=[0, 1, 2])
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    prec, rec, f1, _ = precision_recall_fscore_support(y_act_val, pred_act, labels=[0, 1, 2], zero_division=0)
    acc = float(np.mean(pred_act == y_act_val)) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 9.6), dpi=100)
    mode_note = "분류 직접 예측" if tm.task_type == "classification" else "회귀 조향값 이산화 평가"
    fig.suptitle(
        f"행동 분류 성능 분석: [{MODEL_LABELS[selected_model]}] ({mode_note}) | 전체 정확도: {acc:.1f}%",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    # Left: Confusion Matrix heatmap
    ax0 = axes[0]
    im = ax0.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax0, shrink=0.55)
    ax0.set_xticks([0, 1, 2])
    ax0.set_yticks([0, 1, 2])
    ax0.set_xticklabels(ACTION_NAMES, fontsize=11)
    ax0.set_yticklabels(ACTION_NAMES, fontsize=11)
    ax0.set_xlabel("모델 예측 행동", fontsize=12)
    ax0.set_ylabel("데이터셋에 기록된 행동", fontsize=12)
    ax0.set_title("혼동 행렬 (Confusion Matrix)", fontsize=13, fontweight="bold")

    for i in range(3):
        for j in range(3):
            txt_col = "#0b0e14" if cm_norm[i, j] > 0.55 else "#eef2f9"
            ax0.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)", ha="center", va="center", fontsize=12, fontweight="bold", color=txt_col)

    # Right: Precision / Recall / F1 grouped bar chart
    ax1 = axes[1]
    x_pos = np.arange(3)
    bw = 0.25
    b1 = ax1.bar(x_pos - bw, prec * 100, width=bw, label="Precision", color=C_CYAN)
    b2 = ax1.bar(x_pos, rec * 100, width=bw, label="Recall", color=C_GREEN)
    b3 = ax1.bar(x_pos + bw, f1 * 100, width=bw, label="F1-Score", color=C_PURPLE)
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.0f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ACTION_NAMES, fontsize=11)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("점수 (%)", fontsize=12)
    ax1.set_title("클래스별 Precision / Recall / F1-Score", fontsize=13, fontweight="bold")
    ax1.legend(facecolor="#181e28", edgecolor="#3a4556", fontsize=11)
    ax1.grid(True, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ----------------------------------------------------------------------
# Chart 5: 28-Dim Feature Importance & LiDAR Angular Sensitivity
# ----------------------------------------------------------------------
def _plot_importance(
    dataset: DrivingDataset,
    trained_models: Dict[Tuple[ModelType, TaskType], TrainedModel],
    selected_model: ModelType,
    task_type: TaskType,
) -> plt.Figure:
    if len(dataset) < 20:
        return _empty_figure("데이터셋이 부족합니다", "[1 수집] 탭에서 먼저 데이터를 수집해 주세요.")

    x_val, y_vel_val, _ = _get_val_split(dataset, max_points=600)
    tm = trained_models.get((selected_model, task_type)) or trained_models.get((selected_model, "regression")) or trained_models.get((selected_model, "classification"))

    feat_names = [f"L{i}" for i in range(NUM_BEAMS)] + ["goal_sin", "goal_cos", "goal_dist"]
    imp = np.zeros(len(feat_names), dtype=np.float32)
    method_desc = ""

    if tm is not None and tm.model_type == "rf":
        imp = np.asarray(tm.estimator.feature_importances_, dtype=np.float32)
        method_desc = "RandomForest 트리 분기 중요도 (Feature Importance)"
    elif tm is not None and tm.backend == "torch":
        net = tm.estimator
        dev = next(net.parameters()).device
        xt = torch.tensor(x_val[:300], dtype=torch.float32, device=dev, requires_grad=True)
        out = net(xt)
        loss_proxy = out.abs().sum()
        loss_proxy.backward()
        imp = xt.grad.abs().mean(dim=0).detach().cpu().numpy()
        method_desc = f"{MODEL_LABELS[tm.model_type]} 입력 그래디언트 민감도 (Input Saliency)"
    elif tm is not None and tm.model_type == "linear":
        inner = tm.estimator[-1]
        coef = np.abs(inner.coef_)
        imp = np.mean(coef, axis=0) if coef.ndim == 2 else coef
        method_desc = "Linear 모델 표준화 회귀계수 |w|"
    elif tm is not None and tm.model_type == "knn":
        # Show KNN's uniform 1/28 weight directly!
        imp = np.full(len(feat_names), 1.0 / len(feat_names), dtype=np.float32)
        method_desc = "KNN 유클리드 거리 특징 가중치 (28개 차원 모두 동일한 3.57% — 목적지 3차원이 11%뿐!)"
    else:
        # Correlation with steering w_norm
        for i in range(len(feat_names)):
            c = np.corrcoef(x_val[:, i], y_vel_val[:, 1])[0, 1]
            imp[i] = abs(0.0 if np.isnan(c) else float(c))
        method_desc = "데이터셋 입력 특징 vs 조향(w_norm) 상관계수 |r|"

    imp = imp / max(float(np.sum(imp)), 1e-8) * 100.0
    lidar_sum = float(np.sum(imp[:NUM_BEAMS]))
    goal_sum = float(np.sum(imp[NUM_BEAMS:]))

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 9.6), dpi=100)
    fig.suptitle(
        f"28차원 입력 특징 중요도 분석\n{method_desc}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # Top: Bar chart of all 28 features (LiDAR vs Goal)
    ax0 = axes[0]
    cols = [C_CYAN] * NUM_BEAMS + [C_YELLOW, C_ORANGE, C_RED]
    bars = ax0.bar(np.arange(len(feat_names)), imp, color=cols, edgecolor="#0b0e14", linewidth=0.5)
    ax0.set_xticks(np.arange(len(feat_names)))
    ax0.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=9)
    ax0.set_ylabel("상대 중요도 (%)", fontsize=11)
    ax0.set_title(
        f"라이다 25개 합계: {lidar_sum:.1f}%   vs   목적지 3개(sin, cos, dist) 합계: {goal_sum:.1f}%",
        fontsize=12,
        fontweight="bold",
        color=C_YELLOW,
    )
    ax0.grid(True, axis="y")
    for idx_g in range(NUM_BEAMS, len(feat_names)):
        val = imp[idx_g]
        ax0.text(idx_g, val + max(imp) * 0.02, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_YELLOW)

    # Bottom: Angular LiDAR sensitivity curve (-90 deg Left to +90 deg Right)
    ax1 = axes[1]
    angles_deg = np.linspace(-90, 90, NUM_BEAMS)
    lidar_imp = imp[:NUM_BEAMS]
    ax1.fill_between(angles_deg, lidar_imp, color=C_BLUE, alpha=0.30)
    ax1.plot(angles_deg, lidar_imp, color=C_CYAN, marker="o", linewidth=2.4, markersize=5)
    ax1.axvline(0.0, color=C_GREEN, linestyle="--", linewidth=1.4, label="정면 (0°)")
    ax1.set_title("라이다 각도별(-90° 좌측 ~ 0° 정면 ~ +90° 우측) 판단 민감도", fontsize=12, fontweight="bold")
    ax1.set_xlabel("라이다 빔 각도 (deg)", fontsize=11)
    ax1.set_ylabel("중요도 (%)", fontsize=11)
    ax1.set_xlim(-95, 95)
    ax1.legend(facecolor="#181e28", edgecolor="#3a4556", fontsize=10)
    ax1.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ----------------------------------------------------------------------
# Chart 6: All-in-One 5-Model Report Dashboard (2x2)
# ----------------------------------------------------------------------
# Two-line x tick labels so the baseline name fits under its bar (legacy CSV label kept readable too).
_BENCH_TICK_LABELS = {
    BASELINE_LABEL: "라이다 회피\n기준선",
    LEGACY_BASELINE_LABEL: "Baseline\n(미학습)",
    "RandomForest": "Random\nForest",  # Avoid overlapping the neighbouring "MLP (DL)" tick.
}


def _plot_summary(
    trained_models: Dict[Tuple[ModelType, TaskType], TrainedModel],
    task_type: TaskType,
    bench_results: List[BenchmarkEntry],
) -> plt.Figure:
    items = [(m, trained_models[(m, task_type)]) for m in MODEL_TYPES if (m, task_type) in trained_models]
    if not items and not bench_results:
        return _empty_figure(
            "비교할 학습 모델이 없습니다",
            "[2 학습]에서 5종 모델 일괄 학습 후 [3 레이스]에서 랜덤 20맵 평가를 실행해 주세요.",
        )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.6), dpi=100)
    t_lbl = "회귀 (연속 v, w)" if task_type == "regression" else "분류 (좌/직진/우)"
    fig.suptitle(f"5종 머신러닝 모델 종합 비교 리포트 ({t_lbl})", fontsize=16, fontweight="bold", y=0.97)

    # 1. Top-Left: 20-Map Goal Success Rate
    ax = axes[0, 0]
    if bench_results:
        lbls = [_BENCH_TICK_LABELS.get(e.label, e.label) for e in bench_results]
        rates = [e.success_rate for e in bench_results]
        cols = [MODEL_HEX.get(e.key, C_BLUE) for e in bench_results]
        bars = ax.bar(lbls, rates, color=cols, edgecolor="#eef2f9", linewidth=0.7)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 2.0, f"{r:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 114)
        ax.set_ylabel("도달 성공률 (%)")
        ax.set_title("미지 랜덤 20맵 목적지 도달 성공률 (Online)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "[3 레이스] 탭에서\n'20맵 평가 실행' 실행 시\n여기에 표시됩니다.", ha="center", va="center", fontsize=12, color=C_MUTED)
        ax.set_title("미지 랜덤 20맵 목적지 도달 성공률", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # 2. Top-Right: Offline Validation Metric
    ax = axes[0, 1]
    if items:
        lbls = [MODEL_LABELS[m] for m, _ in items]
        vals = [t.val_metric for _, t in items]
        cols = [MODEL_HEX[m] for m, _ in items]
        bars = ax.bar(lbls, vals, color=cols, edgecolor="#eef2f9", linewidth=0.7)
        for b, v in zip(bars, vals):
            fmt = f"{v:.1f}%" if task_type == "classification" else f"{v:.4f}"
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03, fmt, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.18 if vals else 1.0)
        ax.set_title("오프라인 검증 성능 (" + ("Val Acc ↑" if task_type == "classification" else "Val MAE ↓") + ")", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y")

    # 3. Bottom-Left: Inference Latency (ms)
    ax = axes[1, 0]
    if items:
        lbls = [MODEL_LABELS[m] for m, _ in items]
        lat = [t.infer_ms for _, t in items]
        cols = [MODEL_HEX[m] for m, _ in items]
        bars = ax.bar(lbls, lat, color=cols, edgecolor="#eef2f9", linewidth=0.7)
        for b, v in zip(bars, lat):
            ax.text(b.get_x() + b.get_width() / 2, v + max(lat) * 0.03, f"{v:.2f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(lat) * 1.22 if lat else 1.0)
        ax.set_ylabel("추론 시간 (ms / frame)")
        ax.set_title("실시간 추론 속도 비교 (낮을수록 빠름)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y")

    # 4. Bottom-Right: Trade-off Pareto Scatter (Latency vs Success/Accuracy)
    ax = axes[1, 1]
    if bench_results:
        for e in bench_results:
            if e.key == "baseline":
                continue
            col = MODEL_HEX.get(e.key, C_BLUE)
            ax.scatter([e.infer_ms], [e.success_rate], c=col, s=160, edgecolors="#eef2f9", linewidths=1.5, zorder=5)
            ax.annotate(e.label, (e.infer_ms, e.success_rate), textcoords="offset points", xytext=(8, 4), fontsize=10, fontweight="bold", color=col)
        ax.set_xlabel("추론 시간 (ms) — 왼쪽일수록 빠름")
        ax.set_ylabel("20맵 도달 성공률 (%) — 위일수록 우수")
        ax.set_ylim(-5, 112)
        ax.set_title("속도 vs 주행 성공률 Trade-off (좌상단이 이상적)", fontsize=12, fontweight="bold")
        ax.grid(True)
    elif items:
        for m, t in items:
            ax.scatter([t.infer_ms], [t.val_metric], c=MODEL_HEX[m], s=160, edgecolors="#eef2f9", linewidths=1.5)
            ax.annotate(MODEL_LABELS[m], (t.infer_ms, t.val_metric), textcoords="offset points", xytext=(8, 4), fontsize=10, fontweight="bold", color=MODEL_HEX[m])
        ax.set_xlabel("추론 시간 (ms)")
        ax.set_ylabel("Val Acc (%)" if task_type == "classification" else "Val MAE")
        ax.set_title("속도 vs 오프라인 성능 Trade-off", fontsize=12, fontweight="bold")
        ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def figure_to_pygame_surface(fig: plt.Figure, size: Tuple[int, int] = (WORLD_WIDTH, WORLD_HEIGHT)) -> pygame.Surface:
    """Render a Matplotlib Figure cleanly into an RGB Pygame Surface."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    surf = pygame.image.load(buf, "plot.png").convert()
    if surf.get_size() != size:
        surf = pygame.transform.smoothscale(surf, size)
    return surf


def save_figure_png(fig: plt.Figure, save_path: Path, dpi: int = 160) -> Path:
    """Save a publication-ready high-DPI PNG image."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path
