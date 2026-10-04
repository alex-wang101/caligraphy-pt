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
    geoms: Iterable[StrokeGeometry] = (), polygons: Iterable[np.ndarray] = (), size: int = SCORE_SIZE
) -> np.ndarray:
    """Binary ink mask for scoring. Use for both agent ink and target outlines."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    scale = size / CANVAS
    for g in geoms:
        draw_geometry(draw, g, scale)
    for p in polygons:
        draw_polygon(draw, p, scale)
    return np.asarray(img) > 127


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
