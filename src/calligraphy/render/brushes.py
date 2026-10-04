"""Brush models: stroke points [[x, y, p], ...] + width -> ink geometry (canvas units).

`width` is the brush diameter at full pressure; p in [0, 1] scales the radius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from calligraphy.frame import catmull_rom


@dataclass(frozen=True)
class StrokeGeometry:
    discs: np.ndarray  # (M, 3): cx, cy, r
    quads: np.ndarray  # (Q, 4, 2): convex hull between consecutive discs
    polygons: tuple[np.ndarray, ...] = field(default=())  # freehand outlines (may self-intersect)

    def transformed(self, s: float, t) -> "StrokeGeometry":
        """Exact image under scale s then translation t (discs and quads map to discs and quads)."""
        t = np.asarray(t, dtype=np.float64)
        discs = self.discs.copy()
        discs[:, :2] = discs[:, :2] * s + t
        discs[:, 2] *= s
        return StrokeGeometry(
            discs=discs,
            quads=self.quads * s + t if len(self.quads) else self.quads,
            polygons=tuple(p * s + t for p in self.polygons),
        )

    def bounds(self) -> tuple[float, float, float, float]:
        xs, ys = [], []
        if len(self.discs):
            d = self.discs
            xs += [d[:, 0] - d[:, 2], d[:, 0] + d[:, 2]]
            ys += [d[:, 1] - d[:, 2], d[:, 1] + d[:, 2]]
        if len(self.quads):
            xs.append(self.quads[..., 0].ravel())
            ys.append(self.quads[..., 1].ravel())
        for p in self.polygons:
            xs.append(p[:, 0])
            ys.append(p[:, 1])
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        x, y = np.concatenate(xs), np.concatenate(ys)
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def tangent_quads(discs: np.ndarray) -> np.ndarray:
    """Quads bounded by the outer tangents of each consecutive disc pair.

    Pairs where one disc contains the other need no quad (the larger disc covers it).
    """
    if len(discs) < 2:
        return np.zeros((0, 4, 2))
    c1, c2 = discs[:-1, :2], discs[1:, :2]
    r1, r2 = discs[:-1, 2:3], discs[1:, 2:3]
    d = c2 - c1
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    ok = (dist > np.abs(r1 - r2) + 1e-9)[:, 0]
    if not ok.any():
        return np.zeros((0, 4, 2))
    c1, c2, r1, r2, d, dist = c1[ok], c2[ok], r1[ok], r2[ok], d[ok], dist[ok]
    u = d / dist
    perp = np.stack([-u[:, 1], u[:, 0]], axis=1)
    cos = (r1 - r2) / dist
    sin = np.sqrt(np.clip(1 - cos**2, 0, None))
    n_a = u * cos + perp * sin
    n_b = u * cos - perp * sin
    return np.stack([c1 + r1 * n_a, c2 + r2 * n_a, c2 + r2 * n_b, c1 + r1 * n_b], axis=1)


class DiscBrush:
    """Swept disc: radius = width/2 * p along a centripetal Catmull-Rom through the points."""

    name = "disc"

    def __init__(self, spacing: float = 8.0):
        self.spacing = spacing

    def samples(self, points, width: float) -> np.ndarray:
        """Dense (M, 3) array of x, y, radius."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        dense = catmull_rom(pts, spacing=self.spacing)
        return np.column_stack([dense[:, :2], width / 2 * np.clip(dense[:, 2], 0, 1)])

    def geometry(self, points, width: float) -> StrokeGeometry:
        discs = self.samples(points, width)
        return StrokeGeometry(discs=discs[discs[:, 2] > 0], quads=tangent_quads(discs))


def _tldraw_pen_easing(t: float) -> float:
    return t * 0.65 + math.sin(t * math.pi / 2) * 0.35


class FreehandBrush:
    """perfect-freehand with tldraw's pen preset (real pressure, no simulation).

    Size is chosen so the radius at p = 1 equals width / 2, matching DiscBrush.
    Outlines can self-intersect; Pillow fills even-odd, so self-overlapping
    strokes show small holes (accepted for this secondary brush).
    """

    name = "freehand"
    thinning = 0.62

    def __init__(self, spacing: float = 8.0):
        self.spacing = spacing

    def geometry(self, points, width: float) -> StrokeGeometry:
        from perfect_freehand import get_stroke

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        dense = catmull_rom(pts, spacing=self.spacing)
        size = width / (2 * _tldraw_pen_easing(0.5 + self.thinning * 0.5))
        outline = get_stroke(
            dense.tolist(),
            size=size,
            thinning=self.thinning,
            smoothing=0.62,
            streamline=0.62,
            easing=_tldraw_pen_easing,
            simulate_pressure=False,
            last=True,
        )
        poly = np.asarray(outline, dtype=np.float64).reshape(-1, 2)
        return StrokeGeometry(discs=np.zeros((0, 3)), quads=np.zeros((0, 4, 2)), polygons=(poly,) if len(poly) >= 3 else ())


BRUSHES = {"disc": DiscBrush, "freehand": FreehandBrush}


def get_brush(name: str = "disc"):
    try:
        return BRUSHES[name]()
    except KeyError:
        raise ValueError(f"unknown brush {name!r}; choose from {sorted(BRUSHES)}") from None
