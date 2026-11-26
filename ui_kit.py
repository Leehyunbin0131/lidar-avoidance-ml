"""Design system for the Pygame GUI: theme tokens, fonts, and immediate-mode UI primitives.

All panels/tabs are built from the same small set of components (card, button, segmented
control, toggle, chip, badge, progress bar, keycap, icon) so spacing, colors, typography
and interaction feedback (hover, disabled, cursor, hint text) stay consistent everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

Color = Tuple[int, int, int]

# ----------------------------------------------------------------------
# Theme tokens
# ----------------------------------------------------------------------
BG_APP: Color = (9, 12, 18)
BG_HEADER: Color = (14, 18, 26)
BG_WORLD: Color = (13, 17, 24)
BG_PANEL: Color = (16, 20, 29)
SURFACE: Color = (23, 29, 40)
SURFACE_2: Color = (31, 38, 52)
SURFACE_3: Color = (43, 52, 70)
INSET: Color = (15, 19, 27)
BORDER: Color = (40, 50, 66)
BORDER_STRONG: Color = (66, 80, 104)

TEXT: Color = (234, 239, 247)
TEXT_2: Color = (168, 180, 199)
TEXT_3: Color = (112, 125, 147)

PRIMARY: Color = (82, 139, 255)
SUCCESS: Color = (34, 197, 94)
WARNING: Color = (245, 166, 35)
DANGER: Color = (239, 84, 84)
VIOLET: Color = (139, 92, 246)
CYAN: Color = (34, 211, 238)
ORANGE: Color = (249, 115, 22)

GRID: Color = (21, 27, 37)
OBSTACLE_FILL: Color = (43, 53, 70)
OBSTACLE_EDGE: Color = (96, 114, 142)
OBSTACLE_SHADOW: Color = (8, 11, 16)
WALL: Color = (70, 84, 106)

# Model identity colors (kept identical to visualizer.MODEL_HEX)
MODEL_COLORS: Dict[str, Color] = {
    "baseline": (239, 68, 68),
    "linear": (249, 115, 22),
    "knn": (34, 211, 238),
    "rf": (245, 158, 11),
    "mlp": (59, 130, 246),
    "cnn1d": (34, 197, 94),
}

# Filled button backgrounds (slightly deeper than accent so white text stays readable)
_FILL: Dict[str, Color] = {
    "primary": (56, 108, 228),
    "success": (22, 148, 72),
    "danger": (200, 56, 56),
    "violet": (116, 74, 220),
    "warning": (196, 128, 20),
}

# Layout tokens
RADIUS_CARD = 12
RADIUS_CTRL = 9
PAD = 16
GAP = 12


def blend(a: Color, b: Color, t: float) -> Color:
    """Linear blend a -> b by t in [0, 1]."""
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def lighten(c: Color, amt: int = 16) -> Color:
    return (min(255, c[0] + amt), min(255, c[1] + amt), min(255, c[2] + amt))


def tint(c: Color, strength: float = 0.18, base: Color = SURFACE) -> Color:
    """Subtle tinted background for selected/active states."""
    return blend(base, c, strength)


# ----------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------
_FONT_DIRS = (Path("/usr/share/fonts/truetype/nanum"), Path("/usr/share/fonts/opentype/noto"))


def _first_existing(names: Sequence[str]) -> Optional[str]:
    for d in _FONT_DIRS:
        for n in names:
            p = d / n
            if p.exists():
                return str(p)
    return None


class Fonts:
    """Typography scale. Prefers NanumSquare (R/B), falls back to system fonts."""

    def __init__(self) -> None:
        reg = _first_existing(("NanumSquareR.ttf", "NanumBarunGothic.ttf", "NotoSansCJK-Regular.ttc"))
        bold = _first_existing(("NanumSquareB.ttf", "NanumBarunGothicBold.ttf", "NotoSansCJK-Bold.ttc")) or reg
        mono = _first_existing(("NanumGothicCoding.ttf",)) or reg
        mono_b = _first_existing(("NanumGothicCodingBold.ttf",)) or mono

        def load(path: Optional[str], size: int, bold_fallback: bool = False) -> pygame.font.Font:
            if path:
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    pass
            return pygame.font.SysFont("nanumsquare,notosanscjkkr,malgungothic,applegothic,dejavusans", size, bold=bold_fallback)

        self.reading = load(reg, 22)
        self.headline = load(bold, 38, True)
        self.display = load(bold, 28, True)
        self.h1 = load(bold, 19, True)
        self.h2 = load(bold, 17, True)
        self.body = load(reg, 16)
        self.body_b = load(bold, 16, True)
        self.small = load(reg, 14)
        self.small_b = load(bold, 14, True)
        self.tiny = load(reg, 11)
        self.tiny_b = load(bold, 11, True)
        self.num_xl = load(bold, 30, True)
        self.mono = load(mono, 13)
        self.mono_b = load(mono_b, 13)


# ----------------------------------------------------------------------
# Immediate-mode UI kit
# ----------------------------------------------------------------------
@dataclass
class Hit:
    rect: pygame.Rect
    key: str
    hint: str = ""
    disabled: bool = False


class UIKit:
    """Stateless-per-frame UI primitives drawing onto a target surface and tracking click targets."""

    def __init__(self, canvas: pygame.Surface, fonts: Fonts) -> None:
        self.canvas = canvas
        self.f = fonts
        self.hits: List[Hit] = []
        self.mouse: Tuple[int, int] = (-1, -1)
        self.t: float = 0.0
        self.hover_hint: str = ""
        self._hover_clickable = False
        self._cursor_hand = False
        self._cache: Dict[Tuple[int, str, Color], pygame.Surface] = {}

    # ---- frame lifecycle -------------------------------------------------
    def begin(self, mouse: Tuple[int, int], t: float) -> None:
        self.hits = []
        self.mouse = mouse
        self.t = t
        self.hover_hint = ""
        self._hover_clickable = False

    def end(self) -> None:
        want = self._hover_clickable
        if want != self._cursor_hand:
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if want else pygame.SYSTEM_CURSOR_ARROW)
            except Exception:
                pass
            self._cursor_hand = want

    def hit_test(self, pos: Tuple[int, int]) -> Optional[str]:
        for h in reversed(self.hits):
            if h.rect.collidepoint(pos):
                return None if h.disabled else h.key
        return None

    def hovered(self, rect: pygame.Rect) -> bool:
        return rect.collidepoint(self.mouse)

    def claim_hand(self, hint: str = "") -> None:
        self._hover_clickable = True
        if hint:
            self.hover_hint = hint

    def _register(self, rect: pygame.Rect, key: str, hint: str, disabled: bool) -> bool:
        self.hits.append(Hit(rect=rect.copy(), key=key, hint=hint, disabled=disabled))
        hov = rect.collidepoint(self.mouse)
        if hov:
            if not disabled:
                self.claim_hand(hint)
            elif hint:
                self.hover_hint = hint
        return hov and not disabled

    # ---- text --------------------------------------------------------------
    def render(self, s: str, font: pygame.font.Font, color: Color) -> pygame.Surface:
        key = (id(font), s, color)
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) > 3000:
                self._cache.clear()
            surf = font.render(s, True, color)
            self._cache[key] = surf
        return surf

    def fit(self, s: str, font: pygame.font.Font, max_w: int) -> str:
        if font.size(s)[0] <= max_w:
            return s
        ell = "…"
        lo, hi = 0, len(s)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.size(s[:mid] + ell)[0] <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return s[:lo] + ell

    def wrap(self, s: str, font: pygame.font.Font, max_w: int) -> List[str]:
        lines: List[str] = []
        for para in s.split("\n"):
            cur = ""
            for word in para.split(" "):
                cand = word if not cur else f"{cur} {word}"
                if font.size(cand)[0] <= max_w:
                    cur = cand
                else:
                    if cur:
                        lines.append(cur)
                    cur = ""
                    for ch in word:
                        if cur and font.size(cur + ch)[0] > max_w:
                            lines.append(cur)
                            cur = ""
                        cur += ch
            lines.append(cur)
        return lines

    def text(
        self,
        s: str,
        pos: Tuple[int, int],
        font: pygame.font.Font,
        color: Color = TEXT,
        anchor: str = "topleft",
        max_w: Optional[int] = None,
        surf: Optional[pygame.Surface] = None,
    ) -> pygame.Rect:
        if max_w is not None:
            s = self.fit(s, font, max_w)
        ts = self.render(s, font, color)
        r = ts.get_rect(**{anchor: pos})
        (surf or self.canvas).blit(ts, r)
        return r

    # ---- shapes ------------------------------------------------------------
    def rrect(
        self,
        rect: pygame.Rect,
        color: Color,
        radius: int = RADIUS_CTRL,
        width: int = 0,
        surf: Optional[pygame.Surface] = None,
    ) -> None:
        pygame.draw.rect(surf or self.canvas, color, rect, width, border_radius=radius)

    def glass(self, rect: pygame.Rect, color: Color = (12, 16, 23), alpha: int = 215, radius: int = 10,
              border: Optional[Color] = BORDER, surf: Optional[pygame.Surface] = None) -> None:
        """Semi-transparent rounded panel (used for overlays on the world view)."""
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, (*color, alpha), layer.get_rect(), border_radius=radius)
        if border is not None:
            pygame.draw.rect(layer, (*border, 255), layer.get_rect(), 1, border_radius=radius)
        (surf or self.canvas).blit(layer, rect.topleft)

    def dashed_line(self, color: Color, p1: Tuple[float, float], p2: Tuple[float, float],
                    dash: int = 9, gap: int = 7, width: int = 2, surf: Optional[pygame.Surface] = None) -> None:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1:
            return
        ux, uy = dx / dist, dy / dist
        pos = 0.0
        target = surf or self.canvas
        while pos < dist:
            end = min(pos + dash, dist)
            pygame.draw.line(
                target, color,
                (int(p1[0] + ux * pos), int(p1[1] + uy * pos)),
                (int(p1[0] + ux * end), int(p1[1] + uy * end)),
                width,
            )
            pos += dash + gap

    # ---- icons (vector, font independent) ---------------------------------
    def icon(self, name: str, center: Tuple[int, int], size: int, color: Color,
             surf: Optional[pygame.Surface] = None) -> None:
        s = surf or self.canvas
        cx, cy = center
        h = size / 2.0
        if name == "play":
            pygame.draw.polygon(s, color, [(cx - h * 0.6, cy - h * 0.8), (cx - h * 0.6, cy + h * 0.8), (cx + h * 0.85, cy)])
        elif name == "stop":
            r = int(h * 0.72)
            pygame.draw.rect(s, color, (cx - r, cy - r, 2 * r, 2 * r), border_radius=2)
        elif name == "record":
            pygame.draw.circle(s, color, (cx, cy), int(h * 0.7))
        elif name == "check":
            pts = [(cx - h * 0.62, cy + h * 0.02), (cx - h * 0.18, cy + h * 0.48), (cx + h * 0.68, cy - h * 0.5)]
            pygame.draw.lines(s, color, False, pts, max(2, int(size / 7)))
        elif name == "plus":
            w = max(2, int(size / 7))
            pygame.draw.line(s, color, (cx - h * 0.7, cy), (cx + h * 0.7, cy), w)
            pygame.draw.line(s, color, (cx, cy - h * 0.7), (cx, cy + h * 0.7), w)
        elif name == "bolt":
            pygame.draw.polygon(s, color, [
                (cx + h * 0.15, cy - h), (cx - h * 0.62, cy + h * 0.12), (cx - h * 0.02, cy + h * 0.12),
                (cx - h * 0.2, cy + h), (cx + h * 0.62, cy - h * 0.15), (cx + h * 0.02, cy - h * 0.15),
            ])
        elif name in ("refresh", "reset"):
            r = int(h * 0.72)
            rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
            pygame.draw.arc(s, color, rect, math.radians(40), math.radians(320), max(2, int(size / 8)))
            ax = cx + r * math.cos(math.radians(40))
            ay = cy - r * math.sin(math.radians(40))
            pygame.draw.polygon(s, color, [(ax + h * 0.38, ay - h * 0.05), (ax - h * 0.15, ay - h * 0.42), (ax - h * 0.22, ay + h * 0.2)])
        elif name == "dice":
            r = int(h * 0.82)
            pygame.draw.rect(s, color, (cx - r, cy - r, 2 * r, 2 * r), 2, border_radius=3)
            d = max(1, int(size / 10))
            for ox, oy in ((-0.4, -0.4), (0.4, 0.4), (0.0, 0.0), (0.4, -0.4), (-0.4, 0.4)):
                pygame.draw.circle(s, color, (int(cx + ox * r), int(cy + oy * r)), d)
        elif name == "save":
            r = int(h * 0.8)
            pygame.draw.rect(s, color, (cx - r, cy - r, 2 * r, 2 * r), 2, border_radius=2)
            pygame.draw.rect(s, color, (cx - r * 0.5, cy - r, r, r * 0.7))
            pygame.draw.rect(s, color, (cx - r * 0.55, cy + r * 0.15, r * 1.1, r * 0.6), 1)
        elif name == "folder":
            r = h * 0.85
            pygame.draw.polygon(s, color, [
                (cx - r, cy - r * 0.6), (cx - r * 0.25, cy - r * 0.6), (cx - r * 0.05, cy - r * 0.35),
                (cx + r, cy - r * 0.35), (cx + r, cy + r * 0.7), (cx - r, cy + r * 0.7),
            ], 2)
        elif name == "trash":
            r = h * 0.75
            pygame.draw.line(s, color, (cx - r, cy - r * 0.65), (cx + r, cy - r * 0.65), 2)
            pygame.draw.rect(s, color, (cx - r * 0.7, cy - r * 0.45, r * 1.4, r * 1.5), 2, border_radius=2)
            pygame.draw.line(s, color, (cx - r * 0.3, cy - r * 0.95), (cx + r * 0.3, cy - r * 0.95), 2)
        elif name == "chart":
            r = h * 0.8
            bw = max(2, int(size / 6))
            for i, fr in enumerate((0.45, 0.95, 0.7)):
                x = cx - r + i * r * 0.8
                pygame.draw.rect(s, color, (x, cy + r - 2 * r * fr, bw, 2 * r * fr))
        elif name == "image":
            r = h * 0.85
            pygame.draw.rect(s, color, (cx - r, cy - r * 0.75, 2 * r, 1.5 * r), 2, border_radius=2)
            pygame.draw.polygon(s, color, [(cx - r * 0.7, cy + r * 0.55), (cx - r * 0.15, cy - r * 0.1), (cx + r * 0.25, cy + r * 0.3),
                                           (cx + r * 0.45, cy + r * 0.1), (cx + r * 0.75, cy + r * 0.55)])
        elif name == "external":
            edge = round(h * 0.8)
            arm = round(h * 0.4)
            stroke = max(2, size // 7)
            pygame.draw.lines(s, color, False, [
                (cx - arm, cy - arm), (cx - edge, cy - arm), (cx - edge, cy + edge),
                (cx + arm, cy + edge), (cx + arm, cy + arm),
            ], stroke)
            tip = (cx + edge, cy - edge)
            pygame.draw.lines(s, color, False, [(cx, cy - edge), tip, (cx + edge, cy)], stroke)
            pygame.draw.line(s, color, (cx - 1, cy + 1), tip, stroke)
        elif name == "chev_l":
            pygame.draw.lines(s, color, False, [(cx + h * 0.3, cy - h * 0.6), (cx - h * 0.3, cy), (cx + h * 0.3, cy + h * 0.6)], 2)
        elif name == "chev_r":
            pygame.draw.lines(s, color, False, [(cx - h * 0.3, cy - h * 0.6), (cx + h * 0.3, cy), (cx - h * 0.3, cy + h * 0.6)], 2)
        elif name == "chev_u":
            pygame.draw.lines(s, color, False, [(cx - h * 0.6, cy + h * 0.3), (cx, cy - h * 0.3), (cx + h * 0.6, cy + h * 0.3)], 2)
        elif name == "chev_d":
            pygame.draw.lines(s, color, False, [(cx - h * 0.6, cy - h * 0.3), (cx, cy + h * 0.3), (cx + h * 0.6, cy - h * 0.3)], 2)
        elif name == "flag":
            pygame.draw.line(s, color, (cx - h * 0.55, cy - h * 0.85), (cx - h * 0.55, cy + h * 0.85), 2)
            pygame.draw.polygon(s, color, [(cx - h * 0.55, cy - h * 0.85), (cx + h * 0.7, cy - h * 0.45), (cx - h * 0.55, cy - h * 0.05)])
        elif name == "cross":
            w = max(2, int(size / 7))
            pygame.draw.line(s, color, (cx - h * 0.6, cy - h * 0.6), (cx + h * 0.6, cy + h * 0.6), w)
            pygame.draw.line(s, color, (cx - h * 0.6, cy + h * 0.6), (cx + h * 0.6, cy - h * 0.6), w)
        elif name == "map":
            r = h * 0.85
            pts = [(cx - r, cy - r * 0.7), (cx - r * 0.33, cy - r), (cx + r * 0.33, cy - r * 0.7), (cx + r, cy - r),
                   (cx + r, cy + r * 0.7), (cx + r * 0.33, cy + r), (cx - r * 0.33, cy + r * 0.7), (cx - r, cy + r)]
            pygame.draw.polygon(s, color, pts, 2)
            pygame.draw.line(s, color, (cx - r * 0.33, cy - r), (cx - r * 0.33, cy + r * 0.7), 1)
            pygame.draw.line(s, color, (cx + r * 0.33, cy - r * 0.7), (cx + r * 0.33, cy + r), 1)
        elif name == "radar":
            for k in (0.95, 0.62, 0.3):
                pygame.draw.circle(s, color, (cx, cy), max(2, int(h * k)), 2)
            ang = self.t * 2.2
            pygame.draw.line(s, color, (cx, cy), (cx + h * 0.95 * math.cos(ang), cy + h * 0.95 * math.sin(ang)), 2)

    def spinner(self, center: Tuple[int, int], r: int, color: Color, width: int = 3,
                surf: Optional[pygame.Surface] = None) -> None:
        a0 = self.t * 5.0
        rect = pygame.Rect(center[0] - r, center[1] - r, 2 * r, 2 * r)
        pygame.draw.arc(surf or self.canvas, color, rect, a0, a0 + 4.2, width)

    # ---- composite components ----------------------------------------------
    def card(
        self,
        rect: pygame.Rect,
        title: Optional[str] = None,
        right: Optional[str] = None,
        right_color: Color = TEXT_3,
        accent: Optional[Color] = None,
        highlight: Optional[Color] = None,
    ) -> int:
        """Draw a card container. Returns the y coordinate where content should start."""
        self.rrect(rect, SURFACE, RADIUS_CARD)
        self.rrect(rect, highlight or BORDER, RADIUS_CARD, 1)
        if not title:
            return rect.y + PAD
        tx = rect.x + PAD
        if accent is not None:
            pygame.draw.rect(self.canvas, accent, (rect.x + PAD, rect.y + 16, 3, 14), border_radius=2)
            tx += 11
        self.text(title, (tx, rect.y + 14), self.f.h2, TEXT)
        if right:
            self.text(right, (rect.right - PAD, rect.y + 16), self.f.small, right_color, anchor="topright",
                      max_w=rect.w // 2)
        return rect.y + 44

    def label(self, s: str, x: int, y: int, color: Color = TEXT_3) -> int:
        self.text(s, (x, y), self.f.tiny_b, color)
        return y + 18

    def button(
        self,
        rect: pygame.Rect,
        label: str,
        key: str,
        variant: str = "secondary",
        active: bool = False,
        disabled: bool = False,
        icon: Optional[str] = None,
        hint: str = "",
        font: Optional[pygame.font.Font] = None,
        accent: Optional[Color] = None,
        align_left: bool = False,
    ) -> None:
        """variant: primary | success | danger | violet | warning | secondary | ghost."""
        hov = self._register(rect, key, hint, disabled)
        f = font or self.f.body_b
        if variant in _FILL:
            bg = _FILL[variant]
            fg = TEXT
            border = None
            if disabled:
                bg = blend(SURFACE_2, bg, 0.25)
                fg = TEXT_3
            elif hov:
                bg = lighten(bg, 18)
        elif variant == "ghost":
            bg = SURFACE_2 if hov else None
            fg = TEXT_3 if disabled else (accent or TEXT_2)
            border = BORDER if not hov else BORDER_STRONG
        else:  # secondary
            acc = accent or PRIMARY
            if active:
                bg = tint(acc, 0.22, SURFACE_2)
                fg = TEXT
                border = acc
            else:
                bg = SURFACE_3 if hov and not disabled else SURFACE_2
                fg = TEXT_3 if disabled else TEXT
                border = BORDER_STRONG if hov and not disabled else BORDER
        if bg is not None:
            self.rrect(rect, bg, RADIUS_CTRL)
        if border is not None:
            self.rrect(rect, border, RADIUS_CTRL, 2 if (active and variant == "secondary") else 1)

        icon_sz = 14 if rect.h >= 34 else 12
        max_w = rect.w - 16 - ((icon_sz + 8) if icon else 0)
        label = self.fit(label, f, max_w)
        total = f.size(label)[0] + ((icon_sz + 8) if icon else 0)
        x = rect.x + 16 if align_left else rect.centerx - total // 2
        if icon:
            self.icon(icon, (x + icon_sz // 2, rect.centery), icon_sz, fg)
            x += icon_sz + 8
        self.text(label, (x, rect.centery), f, fg, anchor="midleft")

    def segmented(
        self,
        rect: pygame.Rect,
        options: Sequence[Tuple[str, str, bool]],
        selected: str,
        font: Optional[pygame.font.Font] = None,
        hints: Optional[Dict[str, str]] = None,
        accent: Color = PRIMARY,
        busy: bool = False,
    ) -> None:
        """Segmented control. options = [(key, label, enabled), ...]."""
        self.rrect(rect, INSET, RADIUS_CTRL)
        self.rrect(rect, BORDER, RADIUS_CTRL, 1)
        n = max(1, len(options))
        seg_w = (rect.w - 6) / n
        f = font or self.f.small_b
        for i, (key, lbl, enabled) in enumerate(options):
            r = pygame.Rect(int(rect.x + 3 + i * seg_w), rect.y + 3, int(seg_w) - (0 if i == n - 1 else 2), rect.h - 6)
            is_sel = key == selected
            disabled = (not enabled) or busy
            hov = self._register(r, key, (hints or {}).get(key, ""), disabled)
            if is_sel:
                self.rrect(r, SURFACE_3, RADIUS_CTRL - 2)
                pygame.draw.line(self.canvas, accent, (r.x + 8, r.bottom - 2), (r.right - 8, r.bottom - 2), 2)
                fg = TEXT
            else:
                if hov:
                    self.rrect(r, SURFACE_2, RADIUS_CTRL - 2)
                fg = TEXT_3 if not enabled else TEXT_2
            self.text(lbl, r.center, f, fg, anchor="center", max_w=r.w - 10)

    def toggle(self, rect: pygame.Rect, label: str, value: bool, key: str,
               disabled: bool = False, hint: str = "") -> None:
        hov = self._register(rect, key, hint, disabled)
        self.rrect(rect, SURFACE_3 if hov else SURFACE_2, RADIUS_CTRL)
        self.rrect(rect, BORDER_STRONG if hov else BORDER, RADIUS_CTRL, 1)
        fg = TEXT_3 if disabled else TEXT
        self.text(label, (rect.x + 10, rect.centery), self.f.small_b, fg, anchor="midleft", max_w=rect.w - 52)
        tw, th = 28, 16
        track = pygame.Rect(rect.right - tw - 9, rect.centery - th // 2, tw, th)
        on_col = SUCCESS if not disabled else blend(SURFACE_3, SUCCESS, 0.3)
        self.rrect(track, on_col if value else INSET, th // 2)
        if not value:
            self.rrect(track, BORDER_STRONG, th // 2, 1)
        kx = track.right - th // 2 if value else track.x + th // 2
        pygame.draw.circle(self.canvas, TEXT if not disabled else TEXT_3, (kx, track.centery), th // 2 - 3)

    def value_chip(self, rect: pygame.Rect, label: str, value: str, key: str,
                   disabled: bool = False, hint: str = "") -> None:
        """Clickable chip that cycles a value: [label   value ▸]."""
        hov = self._register(rect, key, hint, disabled)
        self.rrect(rect, SURFACE_3 if hov else SURFACE_2, RADIUS_CTRL)
        self.rrect(rect, BORDER_STRONG if hov else BORDER, RADIUS_CTRL, 1)
        fg = TEXT_3 if disabled else TEXT
        self.text(label, (rect.x + 10, rect.centery), self.f.small_b, fg, anchor="midleft")
        self.text(value, (rect.right - 22, rect.centery), self.f.small_b, PRIMARY if not disabled else TEXT_3, anchor="midright")
        self.icon("chev_r", (rect.right - 12, rect.centery), 9, TEXT_2 if not disabled else TEXT_3)

    def model_chip(self, rect: pygame.Rect, label: str, key: str, color: Color, selected: bool,
                   disabled: bool = False, done: bool = False, hint: str = "") -> None:
        hov = self._register(rect, key, hint, disabled)
        bg = tint(color, 0.20, SURFACE_2) if selected else (SURFACE_3 if hov and not disabled else SURFACE_2)
        self.rrect(rect, bg, RADIUS_CTRL)
        self.rrect(rect, color if selected else (BORDER_STRONG if hov and not disabled else BORDER), RADIUS_CTRL,
                   2 if selected else 1)
        dot_col = color if not disabled else blend(SURFACE_2, color, 0.35)
        fg = TEXT if not disabled else TEXT_3
        tw = self.f.small_b.size(label)[0]
        x = rect.centerx - (tw + 14) // 2
        pygame.draw.circle(self.canvas, dot_col, (x + 4, rect.centery), 4)
        self.text(label, (x + 14, rect.centery), self.f.small_b, fg, anchor="midleft", max_w=rect.w - 22)
        if done:
            c = (rect.right - 9, rect.y + 9)
            pygame.draw.circle(self.canvas, SUCCESS, c, 6)
            self.icon("check", c, 8, (10, 30, 18))

    def badge(self, x: int, y: int, s: str, color: Color, filled: bool = False,
              font: Optional[pygame.font.Font] = None, anchor_right: bool = False,
              surf: Optional[pygame.Surface] = None) -> int:
        f = font or self.f.tiny_b
        tw = f.size(s)[0]
        w = tw + 16
        bx = x - w if anchor_right else x
        r = pygame.Rect(bx, y, w, 20)
        target = surf or self.canvas
        if filled:
            pygame.draw.rect(target, blend(color, (0, 0, 0), 0.15), r, border_radius=10)
            self.text(s, r.center, f, TEXT, anchor="center", surf=target)
        else:
            pygame.draw.rect(target, blend(BG_PANEL, color, 0.16), r, border_radius=10)
            pygame.draw.rect(target, blend(BG_PANEL, color, 0.55), r, 1, border_radius=10)
            self.text(s, r.center, f, color, anchor="center", surf=target)
        return w

    def progress(self, rect: pygame.Rect, frac: float, color: Color = PRIMARY, bg: Color = INSET) -> None:
        self.rrect(rect, bg, rect.h // 2)
        frac = max(0.0, min(1.0, frac))
        if frac > 0:
            fw = max(rect.h, int(rect.w * frac))
            self.rrect(pygame.Rect(rect.x, rect.y, fw, rect.h), color, rect.h // 2)

    def keycap(self, x: int, y: int, s: str) -> int:
        f = self.f.tiny_b
        w = max(24, f.size(s)[0] + 14)
        r = pygame.Rect(x, y, w, 22)
        self.rrect(r, SURFACE_3, 6)
        pygame.draw.line(self.canvas, INSET, (r.x + 4, r.bottom - 1), (r.right - 5, r.bottom - 1), 2)
        self.rrect(r, BORDER_STRONG, 6, 1)
        self.text(s, r.center, f, TEXT, anchor="center")
        return w
