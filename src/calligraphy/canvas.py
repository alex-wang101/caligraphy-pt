"""Canvas: validated create_stroke + incremental rendering. Framework-free."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from calligraphy.frame import CANVAS
from calligraphy.render.brushes import StrokeGeometry, get_brush
from calligraphy.render.raster import SCORE_SIZE, InkLayer, composite, rasterize_mask, render_geometries


class StrokeError(ValueError):
    """A create_stroke call the environment rejects (malformed, too many points, ...)."""


@dataclass(frozen=True)
class CanvasConfig:
    action_frame: int = 1024  # agent coordinates and widths are in [0, action_frame]
    obs_size: int = 224  # capture_canvas resolution (phase 2)
    supersample: int = 2
    brush: str = "disc"
    grid: bool = True
    max_points: int = 16
    max_strokes: int = 64
    min_width: float = 1.0  # canvas units
    max_width: float = 256.0

    @property
    def unit(self) -> float:
        """Canvas units per action unit."""
        return CANVAS / self.action_frame


@dataclass(frozen=True)
class Stroke:
    points: np.ndarray  # (N, 3) canvas frame, clamped
    width: float  # canvas units
    geometry: StrokeGeometry


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(v)


def validate_stroke(points, width, config: CanvasConfig) -> tuple[np.ndarray, float]:
    """Check structure, then scale to the canvas frame and clamp values. Raises StrokeError."""
    if not isinstance(points, (list, tuple, np.ndarray)) or len(points) == 0:
        raise StrokeError("points must be a non-empty list of [x, y, p]")
    if len(points) > config.max_points:
        raise StrokeError(f"too many points: {len(points)} > {config.max_points}")
    for pt in points:
        if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) != 3 or not all(_is_number(v) for v in pt):
            raise StrokeError(f"each point must be [x, y, p] with finite numbers, got {pt!r}")
    if not _is_number(width) or width <= 0:
        raise StrokeError(f"width must be a positive number, got {width!r}")
    pts = np.asarray(points, dtype=np.float64)
    pts[:, :2] = np.clip(pts[:, :2], 0, config.action_frame) * config.unit
    pts[:, 2] = np.clip(pts[:, 2], 0, 1)
    w = float(np.clip(width * config.unit, config.min_width, config.max_width))
    return pts, w


class Canvas:
    def __init__(self, config: CanvasConfig | None = None):
        self.config = config or CanvasConfig()
        self.brush = get_brush(self.config.brush)
        self.reset()

    def reset(self) -> None:
        self.strokes: list[Stroke] = []
        self._layer = InkLayer(self.config.obs_size, self.config.supersample)

    def create_stroke(self, points, width) -> Stroke:
        if len(self.strokes) >= self.config.max_strokes:
            raise StrokeError(f"stroke limit reached ({self.config.max_strokes})")
        pts, w = validate_stroke(points, width, self.config)
        stroke = Stroke(pts, w, self.brush.geometry(pts, w))
        self.strokes.append(stroke)
        self._layer.add(stroke.geometry)
        return stroke

    def capture(self) -> Image.Image:
        """Observation at obs_size; only the newest strokes are painted each call."""
        return composite(self._layer.image(), self.config.grid)

    def render(self, size: int, upto: int | None = None, grid: bool | None = None) -> Image.Image:
        """Full re-render of the first `upto` strokes at any size."""
        geoms = [s.geometry for s in self.strokes[:upto]]
        return render_geometries(geoms, size, self.config.supersample, self.config.grid if grid is None else grid)

    def ink_mask(self, size: int = SCORE_SIZE) -> np.ndarray:
        return rasterize_mask([s.geometry for s in self.strokes], size=size)
