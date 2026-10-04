"""Pillow rasterizer. The ink mask is kept separate from the display composite.

Display: supersampled grayscale mask -> box downsample -> composited over the 米字格.
Scoring: `rasterize_mask` draws agent ink and target outlines with the same
function at the same resolution, so their IoU is apples to apples.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageDraw

from calligraphy.frame import CANVAS
from calligraphy.render.brushes import StrokeGeometry
from calligraphy.render.grid import grid_background

INK_COLOR = (16, 16, 16)
SCORE_SIZE = 512


def draw_geometry(draw: ImageDraw.ImageDraw, geom: StrokeGeometry, scale: float, fill: int = 255) -> None:
    for cx, cy, r in geom.discs * scale:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
    for q in geom.quads * scale:
        draw.polygon([tuple(p) for p in q], fill=fill)
    for poly in geom.polygons:
        draw.polygon([tuple(p) for p in poly * scale], fill=fill)


def draw_polygon(draw: ImageDraw.ImageDraw, poly: np.ndarray, scale: float, fill: int = 255) -> None:
    draw.polygon([tuple(p) for p in np.asarray(poly) * scale], fill=fill)


class InkLayer:
    """Supersampled ink mask that accepts strokes incrementally."""

    def __init__(self, size: int, supersample: int = 2):
        self.size = size
        self.supersample = supersample
        self.scale = size * supersample / CANVAS
        self.mask = Image.new("L", (size * supersample, size * supersample), 0)
        self._draw = ImageDraw.Draw(self.mask)

    def add(self, geom: StrokeGeometry) -> None:
        draw_geometry(self._draw, geom, self.scale)

    def image(self) -> Image.Image:
        """Antialiased ink coverage at the output size (L mode, 255 = full ink)."""
        if self.supersample == 1:
            return self.mask.copy()
        return self.mask.resize((self.size, self.size), Image.Resampling.BOX)


def composite(ink: Image.Image, grid: bool = True) -> Image.Image:
    bg = grid_background(ink.size[0], grid)
    return Image.composite(Image.new("RGB", ink.size, INK_COLOR), bg, ink)


def render_geometries(geoms: Iterable[StrokeGeometry], size: int, supersample: int = 2, grid: bool = True) -> Image.Image:
    layer = InkLayer(size, supersample)
    for g in geoms:
        layer.add(g)
    return composite(layer.image(), grid)


def render_polygons(polys: Iterable[np.ndarray], size: int, supersample: int = 2, grid: bool = True) -> Image.Image:
    """Filled reference outlines (the dataset's ink) on the 米字格."""
    layer = InkLayer(size, supersample)
    draw = ImageDraw.Draw(layer.mask)
    for p in polys:
        draw_polygon(draw, p, layer.scale)
    return composite(layer.image(), grid)


def rasterize_mask(
    geoms: Iterable[StrokeGeometry] = (),
    polygons: Iterable[np.ndarray] = (),
    size: int = SCORE_SIZE,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Binary ink mask for scoring. Use for both agent ink and target outlines.

    `window` = (x0, y0, x1, y1) in pixels crops the raster (same pixel grid, just
    fewer pixels), which makes per-stroke IoU cheap.
    """
    x0, y0, x1, y1 = window or (0, 0, size, size)
    img = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
    draw = ImageDraw.Draw(img)
    scale = size / CANVAS
    offset = np.array([x0, y0]) / scale
    for g in geoms:
        shifted = StrokeGeometry(
            discs=g.discs - np.r_[offset, 0.0] if len(g.discs) else g.discs,
            quads=g.quads - offset if len(g.quads) else g.quads,
            polygons=tuple(p - offset for p in g.polygons),
        )
        draw_geometry(draw, shifted, scale)
    for p in polygons:
        draw_polygon(draw, np.asarray(p) - offset, scale)
    return np.asarray(img) > 127


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _pixel_window(bounds: list[tuple[float, float, float, float]], size: int) -> tuple[int, int, int, int]:
    s = size / CANVAS
    x0 = min(b[0] for b in bounds) * s
    y0 = min(b[1] for b in bounds) * s
    x1 = max(b[2] for b in bounds) * s
    y1 = max(b[3] for b in bounds) * s
    return int(np.floor(x0)) - 1, int(np.floor(y0)) - 1, int(np.ceil(x1)) + 2, int(np.ceil(y1)) + 2


def ink_moments(
    geoms: list[StrokeGeometry] = (), polygons: list[np.ndarray] = (), size: int = SCORE_SIZE
) -> tuple[np.ndarray, float, float] | None:
    """(centroid, RMS spread, area) of the ink in canvas units, or None if there is no ink."""
    bounds = [g.bounds() for g in geoms if len(g.discs) or len(g.quads) or g.polygons]
    bounds += [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polygons if len(p)]
    if not bounds:
        return None
    win = _pixel_window(bounds, size)
    ys, xs = np.nonzero(rasterize_mask(geoms, polygons, size=size, window=win))
    if len(xs) == 0:
        return None
    scale = CANVAS / size
    pts = (np.column_stack([xs, ys]) + np.array(win[:2]) + 0.5) * scale
    c = pts.mean(axis=0)
    return c, float(np.sqrt(((pts - c) ** 2).sum(axis=1).mean())), len(xs) * scale**2


def geometry_iou(geoms: list[StrokeGeometry], polygons: list[np.ndarray], size: int = SCORE_SIZE) -> float:
    """IoU of agent ink vs target outlines, rasterized identically in a shared window."""
    bounds = [g.bounds() for g in geoms if len(g.discs) or len(g.quads) or g.polygons]
    bounds += [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polygons if len(p)]
    if not bounds:
        return 0.0
    win = _pixel_window(bounds, size)
    return mask_iou(rasterize_mask(geoms, size=size, window=win), rasterize_mask(polygons=polygons, size=size, window=win))
