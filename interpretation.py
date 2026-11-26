"""Reading companion for the five-tab LiDAR simulator."""

from __future__ import annotations

import csv
import math
import re
import threading
import webbrowser
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate, groupby
from pathlib import Path
from typing import Optional

import pygame
from PIL import GifImagePlugin, Image, ImageSequence

import ui_kit as ui
from model import BASELINE_LABEL, LEGACY_BASELINE_LABEL, BenchmarkEntry, MODEL_LABELS
from simulator import WORLD_HEIGHT, WORLD_WIDTH


@dataclass(frozen=True)
class Chapter:
    menu: str
    question: str
    lab: str
    action: str


CHAPTERS = (
    Chapter("지도와 센서", "벽 앞에서 전체 지도를 볼 때와 현재 거리만 잴 때, 각각 어디까지 알 수 있을까요?", "collect", "데이터 수집 화면 열기"),
    Chapter("주행 기록과 모방학습", "라이다가 잰 거리와 로봇이 고른 행동(좌·직·우) 중 어느 쪽이 입력이고 어느 쪽이 예측값일까요?", "collect", "데이터 수집 화면 열기"),
    Chapter("다섯 모델의 차이", "통로 입구가 센서 빔 한 칸만큼 옆으로 옮겨지면 어떤 모델이 비슷한 모양을 알아보기 쉬울까요?", "train", "학습 화면에서 모델 비교하기"),
    Chapter("실험 결과 읽기", "도착 횟수가 같다면 검증 점수와 충돌 횟수도 같을까요?", "test", "레이스와 20맵 평가 열기"),
    Chapter("실제 로봇으로 이어 보기", "벽 앞에서 늦게 돈다면 거리 측정·모델 계산·바퀴 반응 중 어디에서 시간이 걸렸을까요?", "test", "단일 주행에서 예측값 보기"),
)


# Old CSVs labelled the straight-to-goal baseline "Baseline (미학습)"; show it under its descriptive name.
LEGACY_BASELINE_NAME = "목적지 직진 기준선"
_BASELINE_LABELS = (BASELINE_LABEL, LEGACY_BASELINE_LABEL)


@dataclass(frozen=True)
class Report:
    entries: tuple[BenchmarkEntry, ...]
    source: str


def read_saved_report(path: Path) -> Optional[Report]:
    """Read the app's exported scores as one historical run, independent of current models."""
    keys = {label: key for key, label in MODEL_LABELS.items()} | dict.fromkeys(_BASELINE_LABELS, "baseline")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            metric = "val_acc_pct"
            if metric not in fields:
                return None  # Only accuracy-based (classification) exports are understood.
            rows = []
            for raw in reader:
                key = keys[raw["model"]]
                goals, crashes, timeouts = (int(raw[c]) for c in ("goals", "crashes", "timeouts"))
                n = goals + crashes + timeouts
                if n <= 0 or min(goals, crashes, timeouts) < 0 or any(r.key == key or r.num_maps != n for r in rows):
                    return None
                value = None if raw[metric] == "-" else float(raw[metric])
                avg, latency = float(raw["avg_goal_time_s"]), float(raw["infer_ms"])
                if not all(math.isfinite(v) and v >= 0 for v in (avg, latency, value or 0)):
                    return None
                rows.append(BenchmarkEntry(key, raw["model"], n, goals, crashes, timeouts, 100 * goals / n,
                                           avg, value, latency))
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return None
    if not rows or not any(r.key == "baseline" for r in rows):
        return None
    return Report(tuple(rows), "저장된 평가")


GUIDE_PATH = Path(__file__).resolve().parent / "docs" / "guide.md"
RESULT_START = "<!-- evaluation:start -->"
RESULT_END = "<!-- evaluation:end -->"


def _has_final_consonant(text: str) -> bool:
    s = text.strip().rstrip(")")
    if not s:
        return False
    ch = s[-1]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28 != 0
    if ch == "%":
        return False
    if ch in "013678":
        return True
    if ch in "2459":
        return False
    return ch.lower() in {"l", "m", "n"}


def _josa(text: str, consonant: str, vowel: str) -> str:
    return consonant if _has_final_consonant(text) else vowel


def _row_name(r: BenchmarkEntry) -> str:
    if r.key != "baseline":
        return r.label
    return LEGACY_BASELINE_NAME if r.label == LEGACY_BASELINE_LABEL else BASELINE_LABEL


def _outcome_sentence(name: str, r: BenchmarkEntry) -> str:
    """Describe goals, collisions and timeouts of one row so every map is accounted for."""
    goals, crashes, timeouts = r.goals, r.crashes, r.timeouts
    topic = f"{name}{_josa(name, '은', '는')} 맵 {r.num_maps}개"
    if goals == r.num_maps:
        return f"{topic} 모두에서 도착했습니다."
    if goals == 0 and crashes == 0:
        return f"{topic} 모두에서 충돌하지 않았으나 제한 시간 안에 도착하지 못했습니다."
    head = f"{topic} 중 {goals}개에서 도착했고" if goals else f"{topic} 중 한 곳에도 도착하지 못했고"
    rest = "나머지 " if goals else ""
    if crashes == 0:
        return f"{head} 한 번도 충돌하지 않았으나 {rest}{timeouts}개에서는 제한 시간 안에 도착하지 못했습니다."
    if timeouts == 0:
        return f"{head} {rest}{crashes}개에서 충돌했습니다."
    return f"{head} {crashes}개에서 충돌했으며 {timeouts}개에서는 제한 시간 안에 도착하지 못했습니다."


def result_markdown(report: Optional[Report]) -> str:
    if not report:
        return "아직 평가 기록이 없습니다. 레이스 탭에서 ‘20맵 평가 실행’을 누르면 이 자리에 도착·충돌·시간 초과 횟수와 해석이 표시됩니다."
    rows = report.entries
    n = rows[0].num_maps
    metric = "검증 정확도"
    lines = [f"### {report.source} · 같은 맵 {n}개", "",
             f"| 주행 방식 | {metric} | 도착 | 충돌 | 시간 초과 | 도달률 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        val = "—" if r.val_metric is None else f"{r.val_metric:.1f}%"
        lines.append(f"| {_row_name(r)} | {val} | {r.goals}/{n} | {r.crashes} | {r.timeouts} | {r.success_rate:.0f}% |")
    base = next(r for r in rows if r.key == "baseline")
    learned = [r for r in rows if r.key != "baseline"]
    lines += ["", _outcome_sentence(_row_name(base), base)]
    if learned:
        most = max(r.goals for r in learned)
        leaders = [r for r in learned if r.goals == most]
        names = ", ".join(r.label.replace(" (DL)", "") for r in leaders)
        lines[-1] += f" 학습 모델 중에서는 {names}의 도착 횟수가 {most}회로 가장 많았습니다."
        if len(leaders) > 1 and all(r.val_metric is not None for r in leaders):
            a, b = min(leaders, key=lambda r: r.val_metric), max(leaders, key=lambda r: r.val_metric)
            if a.val_metric != b.val_metric:
                a_val = f"{a.val_metric:.1f}%"
                b_val = f"{b.val_metric:.1f}%"
                pair = f"{a_val}{_josa(a_val, '과', '와')} {b_val}"
                end = _josa(b_val, "이었지만", "였지만")
                lines[-1] += f" {a.label}{_josa(a.label, '과', '와')} {b.label}의 {metric}는 각각 {pair}{end} 완주한 맵 수는 같았습니다."
    else:
        lines[-1] += " 모델을 학습한 뒤 다시 평가하면 기준선과 비교할 수 있습니다."
    lines[-1] += f" 이번 평가에서 도착 한 번은 도달률 {100 / n:g}%p에 해당합니다."
    return "\n".join(lines)


def with_report(source: str, report: Optional[Report]) -> str:
    before, _, remaining = source.partition(RESULT_START)
    _, _, after = remaining.partition(RESULT_END)
    return before + result_markdown(report) + after


_FIGURE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)(?:\s+"(?P<cap>[^"]*)")?\)')
_LINK = re.compile(r"\[([^\]]+)\]\((https://[^\s)]+)\)")
_TABLE_COLUMNS = (300, 150, 120, 120, 140, 154)
_MODEL_KEYS = {label: key for key, label in MODEL_LABELS.items()} | dict.fromkeys(
    (*_BASELINE_LABELS, LEGACY_BASELINE_NAME), "baseline")
_GIF_LOADING_LOCK = threading.Lock()
_DecodedGif = tuple[tuple[int, int], list[ui.Color], list[bytes], list[int]]


@dataclass
class _Block:
    surface: pygame.Surface
    x: int
    top: int
    overlay: Optional[pygame.Surface] = None
    animation: Optional[Path] = None
    hover: Optional[pygame.Surface] = None


def _line_kind(line: str) -> str:
    if not line:
        return "blank"
    if line.startswith(("# ", "## ", "### ")):
        return "heading"
    if line.startswith("|"):
        return "table"
    if line.startswith("> "):
        return "quote"
    if _FIGURE.fullmatch(line):
        return "figure"
    if _LINK.fullmatch(line):
        return "link"
    return "paragraph"


def _figure_size(image: Image.Image, width: int) -> tuple[int, int]:
    w = min(width, image.width)
    return w, max(1, round(w * image.height / image.width))


def _figure_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if "A" in image.getbands() or "transparency" in image.info:
        background = Image.new("RGBA", image.size, ui.BG_WORLD)
        rgb = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
    else:
        rgb = image.convert("RGB")
    return rgb if rgb.size == size else rgb.resize(size, Image.Resampling.LANCZOS)


def _palette_colors(palette: list[int]) -> list[ui.Color]:
    entries = (palette + [0] * 768)[:768]
    return list(zip(entries[::3], entries[1::3], entries[2::3]))


def _gif_palette(image: GifImagePlugin.GifImageFile, size: tuple[int, int]) -> Image.Image:
    last = image.n_frames - 1
    count = min(6, last + 1)
    samples = [round(i * last / max(1, count - 1)) for i in range(count)]
    montage = Image.new("RGB", (320 * count, 180), ui.BG_WORLD)
    for column, index in enumerate(samples):
        image.seek(index)
        sample = _figure_rgb(image, size)
        sample.thumbnail((320, 180), Image.Resampling.NEAREST)
        montage.paste(sample, (column * 320, 0))
    return montage.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def _decode_gif(path: Path, width: int) -> _DecodedGif:
    with Image.open(path, formats=("GIF",)) as image:
        size = _figure_size(image, width)
        palette = image.getpalette()
        frames, durations = [], []
        if palette and size == image.size and "transparency" not in image.info:
            for frame in ImageSequence.Iterator(image):
                if frame.mode != "P" or frame.size != size or frame.getpalette() != palette:
                    break
                frames.append(frame.tobytes())
                durations.append(max(20, int(frame.info.get("duration", 100))))
            else:
                return size, _palette_colors(palette), frames, list(accumulate(durations))

        # Sample the whole animation for its palette, then quantize one frame at a time.
        frames.clear()
        durations.clear()
        reference = _gif_palette(image, size)
        for frame in ImageSequence.Iterator(image):
            rgb = _figure_rgb(frame, size)
            indexed = rgb.quantize(palette=reference, dither=Image.Dither.NONE)
            frames.append(indexed.tobytes())
            durations.append(max(20, int(frame.info.get("duration", 100))))
        return size, _palette_colors(reference.getpalette()), frames, list(accumulate(durations))


class GuideReader:
    """Lay out the bundled Markdown and play its animations as they enter view."""

    margin = 48
    width = WORLD_WIDTH - 2 * margin
    track = pygame.Rect(WORLD_WIDTH - 17, 12, 9, WORLD_HEIGHT - 24)
    _viewport = pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT)

    def __init__(self, kit: ui.UIKit) -> None:
        self.kit = kit
        self.source = GUIDE_PATH.read_text(encoding="utf-8")
        self.offset = 0
        self.height = 0
        self.anchors: list[int] = []
        self.blocks: list[_Block] = []
        self.links: list[tuple[pygame.Rect, str]] = []
        self.report: Optional[Report] = None
        self.drag_offset: Optional[int] = None
        self._gif_lock = threading.Lock()
        self._gif_decoded: dict[Path, _DecodedGif] = {}
        self._gif_surfaces: dict[Path, tuple[list[pygame.Surface], list[int]]] = {}
        self._gif_posters: dict[Path, pygame.Surface] = {}
        self._gif_visible: set[int] = set()
        self._gif_start_tick: dict[int, int] = {}
        gif_paths = dict.fromkeys(
            (GUIDE_PATH.parent / match["path"]).resolve()
            for match in _FIGURE.finditer(self.source)
            if Path(match["path"]).suffix.lower() == ".gif"
        )
        self._decoder_thread = threading.Thread(
            target=self._decode_gifs, args=(tuple(gif_paths),), name="Guide GIF decoder", daemon=True,
        )
        self._decoder_thread.start()

    def _decode_gifs(self, paths: tuple[Path, ...]) -> None:
        if not paths:
            return
        # Pillow's strategy is process-wide. Preserve global-palette frames as indexed pixels.
        with _GIF_LOADING_LOCK:
            strategy = GifImagePlugin.LOADING_STRATEGY
            GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY
            try:
                for path in paths:
                    try:
                        decoded = _decode_gif(path, self.width)
                    except (OSError, EOFError, ValueError):
                        continue
                    with self._gif_lock:
                        self._gif_decoded[path] = decoded
            finally:
                GifImagePlugin.LOADING_STRATEGY = strategy

    def prepare(self, report: Optional[Report]) -> None:
        if self.blocks and self.report is report:
            return
        self.report = report
        self.links.clear()
        self.blocks.clear()
        self.anchors.clear()
        self._gif_visible.clear()
        self._gif_start_tick.clear()
        y = 35
        lines = (line.strip() for line in with_report(self.source, report).splitlines())
        for kind, group in groupby(lines, _line_kind):
            if kind == "paragraph":
                y = self._add_text(" ".join(group), y, self.kit.f.reading, ui.TEXT_2) + 16
            elif kind == "table":
                y = self._add_table(list(group), y)
            elif kind == "quote":
                y = self._add_quote([line[2:] for line in group], y)
            elif kind == "heading":
                for line in group:
                    y = self._add_heading(line, y)
            elif kind == "figure":
                for line in group:
                    y = self._add_figure(_FIGURE.fullmatch(line), y)
            elif kind == "link":
                for line in group:
                    y = self._add_link(*_LINK.fullmatch(line).groups(), y)
        self.height = y + 52
        if self.anchors:
            self.height = max(self.height, self.anchors[-1] + WORLD_HEIGHT)
        self.scroll(0)

    def _add_text(self, text: str, y: int, font: pygame.font.Font, color: ui.Color) -> int:
        line_height = font.get_linesize() + 6
        for line in self.kit.wrap(text, font, self.width):
            self.blocks.append(_Block(self.kit.render(line, font, color), self.margin, y))
            y += line_height
        return y - 6

    def _add_heading(self, line: str, y: int) -> int:
        f = self.kit.f
        if line.startswith("### "):
            return self._add_text(line[4:], y + 12, f.h1, ui.CYAN) + 7
        if line.startswith("# "):
            return self._add_text(line[2:], y, f.h1, ui.CYAN)

        if self.anchors:
            y += 40
            divider = pygame.Surface((self.width, 1))
            divider.fill(ui.BORDER)
            self.blocks.append(_Block(divider, self.margin, y))
            y += 1 + 32
            chapter = len(self.anchors)
            kicker_y = y
            if chapter < len(CHAPTERS):
                kicker = self.kit.render(f"{chapter + 1}장 · {CHAPTERS[chapter].menu}", f.small_b, ui.TEXT_3)
                self.blocks.append(_Block(kicker, self.margin, kicker_y))
            y += 28
            self.anchors.append(max(0, kicker_y - 20))
        else:
            self.anchors.append(0)
            y += 22
        return self._add_text(line[3:], y, f.headline, ui.TEXT) + 12

    def _add_table(self, lines: list[str], y: int) -> int:
        rows = []
        for line in lines:
            cells = [cell.strip() for cell in line.removeprefix("|").removesuffix("|").split("|")]
            if not all(not cell.strip("-:") for cell in cells):
                rows.append(cells)
        if not rows:
            return y

        kit, f = self.kit, self.kit.f
        height = 40 + 44 * (len(rows) - 1)
        table = pygame.Surface((self.width, height), pygame.SRCALPHA)
        pygame.draw.rect(table, ui.SURFACE, table.get_rect(), border_radius=12)
        bottom_radius = 12 if len(rows) == 1 else 0
        pygame.draw.rect(table, ui.SURFACE_2, (0, 0, self.width, 40), border_radius=12,
                         border_bottom_left_radius=bottom_radius, border_bottom_right_radius=bottom_radius)
        for index, row in enumerate(rows):
            top = 40 + 44 * (index - 1) if index else 0
            center = top + (22 if index else 20)
            font, color = (f.body, ui.TEXT) if index else (f.small_b, ui.TEXT_2)
            if index:
                pygame.draw.line(table, ui.BORDER, (1, top), (self.width - 2, top))
            key = _MODEL_KEYS.get(row[0])
            left = 36 if key else 16
            if key:
                pygame.draw.circle(table, ui.MODEL_COLORS[key], (20, center), 5)
            kit.text(row[0], (left, center), font, color, anchor="midleft",
                     max_w=_TABLE_COLUMNS[0] - left - 16, surf=table)
            right = _TABLE_COLUMNS[0]
            for column, cell in zip(_TABLE_COLUMNS[1:], row[1:]):
                right += column
                kit.text(cell, (right - 20, center), font, color, anchor="midright", max_w=column - 36, surf=table)
        pygame.draw.rect(table, ui.BORDER, table.get_rect(), width=1, border_radius=12)
        self.blocks.append(_Block(table, self.margin, y))
        return y + height + 28

    def _add_quote(self, paragraphs: list[str], y: int) -> int:
        kit, font = self.kit, self.kit.f.reading
        line_height = font.get_linesize() + 6
        top = 16
        text = []
        for paragraph in paragraphs:
            for line in kit.wrap(paragraph, font, 940):
                text.append((kit.render(line, font, ui.TEXT), top))
                top += line_height
            top += 10 - 6
        height = top - 10 + 16
        card = pygame.Surface((self.width, height), pygame.SRCALPHA)
        pygame.draw.rect(card, ui.SURFACE, card.get_rect(), border_radius=10)
        card.set_clip((0, 0, 4, height))
        pygame.draw.rect(card, ui.CYAN, card.get_rect(), border_top_left_radius=10, border_bottom_left_radius=10)
        card.set_clip(None)
        for surface, top in text:
            card.blit(surface, (24, top))
        self.blocks.append(_Block(card, self.margin, y))
        return y + height + 20

    def _add_figure(self, match: re.Match[str], y: int) -> int:
        path = (GUIDE_PATH.parent / match["path"]).resolve()
        animation = path if path.suffix.lower() == ".gif" else None
        picture = self._gif_posters.get(path)
        if picture is None:
            try:
                with Image.open(path) as image:
                    size = _figure_size(image, self.width)
                    rgb = _figure_rgb(image, size)
                    picture = pygame.image.frombytes(rgb.tobytes(), size, "RGB").convert()
                if animation is not None:
                    self._gif_posters[path] = picture
            except (OSError, EOFError, ValueError):
                animation = None
                picture = pygame.Surface((self.width, 112))
                picture.fill(ui.SURFACE)
                label = match["alt"] or "그림을 불러올 수 없습니다"
                self.kit.icon("image", (self.width // 2, 38), 24, ui.TEXT_3, surf=picture)
                self.kit.text(label, (self.width // 2, 72), self.kit.f.small, ui.TEXT_3,
                              anchor="center", max_w=self.width - 48, surf=picture)

        w, h = picture.get_size()
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill(ui.BG_WORLD)
        pygame.draw.rect(overlay, (0, 0, 0, 0), overlay.get_rect(), border_radius=12)
        pygame.draw.rect(overlay, ui.BORDER, overlay.get_rect(), width=1, border_radius=12)
        self.blocks.append(_Block(picture, self.margin + (self.width - w) // 2, y, overlay, animation))
        y += h
        if match["cap"]:
            font = self.kit.f.small
            y += 10
            for line in self.kit.wrap(match["cap"], font, self.width):
                caption = self.kit.render(line, font, ui.TEXT_3)
                x = self.margin + (self.width - caption.get_width()) // 2
                self.blocks.append(_Block(caption, x, y))
                y += font.get_linesize()
        return y + 28

    def _add_link(self, label: str, url: str, y: int) -> int:
        kit, font = self.kit, self.kit.f.body_b
        label_x = 44  # 20 px padding, 14 px icon, 10 px gap
        text = kit.render(kit.fit(label, font, self.width - label_x - 20), font, ui.CYAN)
        size = label_x + text.get_width() + 20, 40
        label_rect = text.get_rect(midleft=(label_x, 20))
        states = []
        for hovered in (False, True):
            pill = pygame.Surface(size, pygame.SRCALPHA)
            fill = ui.tint(ui.CYAN, 0.22 if hovered else 0.12, ui.SURFACE)
            border = ui.CYAN if hovered else ui.blend(ui.SURFACE, ui.CYAN, 0.45)
            pygame.draw.rect(pill, fill, pill.get_rect(), border_radius=20)
            pygame.draw.rect(pill, border, pill.get_rect(), width=1, border_radius=20)
            kit.icon("external", (27, 20), 14, ui.CYAN, surf=pill)
            pill.blit(text, label_rect)
            if hovered:
                underline = label_rect.bottom + 1
                pygame.draw.line(pill, ui.CYAN, (label_rect.left, underline), (label_rect.right - 1, underline))
            states.append(pill)
        normal, hover = states
        self.blocks.append(_Block(normal, self.margin, y, hover=hover))
        self.links.append((normal.get_rect(topleft=(self.margin, y)), url))
        return y + 40 + 18

    @property
    def limit(self) -> int:
        return max(0, self.height - WORLD_HEIGHT)

    @property
    def chapter(self) -> int:
        if self.offset >= self.limit and self.anchors:
            return len(self.anchors) - 1
        return max((i for i, top in enumerate(self.anchors) if top <= self.offset), default=0)

    def jump(self, index: int) -> None:
        if self.anchors:
            self.offset = self.anchors[max(0, min(index, len(self.anchors) - 1))]

    def scroll(self, dy: int) -> None:
        self.offset = max(0, min(self.limit, self.offset + dy))

    def thumb(self) -> pygame.Rect:
        height = max(32, round(self.track.h * WORLD_HEIGHT / max(WORLD_HEIGHT, self.height)))
        top = self.track.y + round((self.track.h - height) * self.offset / max(1, self.limit))
        return pygame.Rect(self.track.x, top, self.track.w, height)

    def _link_point(self, mouse: tuple[int, int]) -> Optional[tuple[int, int]]:
        if self._viewport.collidepoint(mouse) and not self.track.inflate(16, 0).collidepoint(mouse):
            return mouse[0], mouse[1] + self.offset
        return None

    def hovered_link(self, mouse: tuple[int, int]) -> Optional[str]:
        point = self._link_point(mouse)
        if point is not None:
            return next((url for rect, url in self.links if rect.collidepoint(point)), None)
        return None

    def press(self, pos: tuple[int, int]) -> None:
        if not self.track.inflate(16, 0).collidepoint(pos):
            url = self.hovered_link(pos)
            if url is not None:
                webbrowser.open(url)
            return
        thumb = self.thumb()
        if thumb.top <= pos[1] < thumb.bottom:
            self.drag_offset = pos[1] - thumb.top
        else:
            self.scroll((1 if pos[1] >= thumb.bottom else -1) * (WORLD_HEIGHT - 80))

    def drag(self, y: int) -> None:
        if self.drag_offset is not None:
            value = (y - self.track.y - self.drag_offset) / max(1, self.track.h - self.thumb().h)
            self.offset = round(self.limit * max(0, min(1, value)))

    def _gif_frame(self, path: Path, elapsed: int) -> Optional[pygame.Surface]:
        if path not in self._gif_surfaces:
            with self._gif_lock:
                decoded = self._gif_decoded.get(path)
            if decoded is None:
                return None
            size, palette, buffers, ends = decoded
            frames = []
            for buffer in buffers:
                frame = pygame.image.frombuffer(buffer, size, "P")
                frame.set_palette(palette)
                frames.append(frame)
            self._gif_surfaces[path] = frames, ends
        frames, ends = self._gif_surfaces[path]
        index = min(bisect_right(ends, elapsed % ends[-1]), len(frames) - 1)
        return frames[index]

    def draw(self, mouse: tuple[int, int] = (-1, -1)) -> None:
        canvas = self.kit.canvas
        canvas.fill(ui.BG_WORLD, self._viewport)
        point = self._link_point(mouse)
        now = pygame.time.get_ticks()
        visible = set()
        for block in self.blocks:
            y = block.top - self.offset
            if y >= WORLD_HEIGHT:
                break
            if y + block.surface.get_height() <= 0:
                continue
            surface = block.surface
            if block.animation is not None:
                visible.add(block.top)
                if block.top not in self._gif_visible:
                    self._gif_start_tick[block.top] = now
                frame = self._gif_frame(block.animation, now - self._gif_start_tick[block.top])
                if frame is not None:
                    surface = frame
            if block.hover is not None and point is not None:
                if surface.get_rect(topleft=(block.x, block.top)).collidepoint(point):
                    surface = block.hover
            canvas.blit(surface, (block.x, y))
            if block.overlay is not None:
                canvas.blit(block.overlay, (block.x, y))
        self._gif_visible = visible
        self.kit.rrect(self.track, ui.SURFACE)
        self.kit.rrect(self.thumb(), ui.TEXT_3)
