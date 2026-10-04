"""Per-stroke metrics. Path metrics use the agent's raw points; IoU uses rendered ink.

No single metric suffices: a reversed stroke has identical ink (only Fréchet
sees it), a wrong-thickness stroke has the right path (only IoU sees it).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from calligraphy.frame import resample_count
from calligraphy.geom import points_in_polygon
from calligraphy.render.brushes import get_brush
from calligraphy.render.raster import SCORE_SIZE, geometry_iou

N_RESAMPLE = 32


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Fréchet distance between two polylines."""
    d = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
    n, m = d.shape
    c = np.empty_like(d)
    c[:, 0] = np.maximum.accumulate(d[:, 0])
    c[0, :] = np.maximum.accumulate(d[0, :])
    for i in range(1, n):
        for j in range(1, m):
            c[i, j] = max(d[i, j], min(c[i - 1, j], c[i - 1, j - 1], c[i, j - 1]))
    return float(c[-1, -1])


@dataclass
class StrokeMetrics:
    inside_frac: float  # fraction of the agent path inside the outline (placement)
    iou: float  # rendered ink vs filled outline (placement, thickness, length)
    frechet: float  # agent path vs median (shape)
    frechet_reversed: float
    direction_ok: bool  # drawn in writing direction

    def to_dict(self) -> dict:
        return asdict(self)


def stroke_metrics(
    points, width: float, outline: np.ndarray, median: np.ndarray, brush: str = "disc", size: int = SCORE_SIZE
) -> StrokeMetrics:
    """points: (N, 3) canvas frame; width in canvas units."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    path = resample_count(pts[:, :2], N_RESAMPLE)
    med = resample_count(median, N_RESAMPLE)
    f, fr = frechet(path, med), frechet(path, med[::-1])
    return StrokeMetrics(
        inside_frac=float(points_in_polygon(path, outline).mean()),
        iou=geometry_iou([get_brush(brush).geometry(pts, width)], [outline], size),
        frechet=f,
        frechet_reversed=fr,
        direction_ok=bool(f <= fr),
    )
