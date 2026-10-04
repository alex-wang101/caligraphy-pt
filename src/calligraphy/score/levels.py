"""Three scoring levels, each isolating one kind of error.

- placement: bounding-box centre offset and size ratio on the raw drawing
- structure: one robust scale + translation for the whole character, then per-stroke IoU
- stroke shape: a separate scale + translation per stroke, then per-stroke IoU

Strokes are index-matched (agent stroke i vs target stroke i), so stroke order is
scored for free. Both IoU levels divide by max(n_agent, n_target): missing and
extra strokes cost score, so scribbling for coverage does not pay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from calligraphy.data.types import CharacterEntry
from calligraphy.render.brushes import get_brush
from calligraphy.render.raster import SCORE_SIZE, geometry_iou, ink_moments
from calligraphy.score.align import Moments, Similarity, fit_stroke, robust_fit

# Placement is flagged outside these bounds (logged; not part of reward v1).
PLACEMENT_MAX_OFFSET = 128.0  # two cells of a 64-grid
PLACEMENT_RATIO_RANGE = (0.8, 1.25)


@dataclass
class CharacterScore:
    placement_offset: float  # canvas units between ink bbox centres
    placement_ratio: float  # agent ink bbox size / target ink bbox size (longest side)
    structure: float
    shape: float
    structure_strokes: list[float] = field(default_factory=list)
    shape_strokes: list[float] = field(default_factory=list)
    n_agent: int = 0
    n_target: int = 0
    align_scale: float = 1.0
    align_offset: tuple[float, float] = (0.0, 0.0)
    align_inliers: list[bool] = field(default_factory=list)

    @property
    def placement_ok(self) -> bool:
        lo, hi = PLACEMENT_RATIO_RANGE
        return self.placement_offset <= PLACEMENT_MAX_OFFSET and lo <= self.placement_ratio <= hi

    def to_dict(self) -> dict:
        d = asdict(self)
        d["placement_ok"] = self.placement_ok
        return d


def _bbox(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    a = np.asarray(boxes)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())


def placement(geoms, outlines) -> tuple[float, float]:
    boxes = [g.bounds() for g in geoms if len(g.discs) or len(g.quads) or g.polygons]
    if not boxes:
        return float("inf"), 0.0
    ax0, ay0, ax1, ay1 = _bbox(boxes)
    tx0, ty0, tx1, ty1 = _bbox([(o[:, 0].min(), o[:, 1].min(), o[:, 0].max(), o[:, 1].max()) for o in outlines])
    offset = float(np.hypot((ax0 + ax1 - tx0 - tx1) / 2, (ay0 + ay1 - ty0 - ty1) / 2))
    ratio = max(ax1 - ax0, ay1 - ay0) / max(tx1 - tx0, ty1 - ty0, 1e-9)
    return offset, float(ratio)


def target_moments(entry: CharacterEntry, size: int = SCORE_SIZE) -> list[Moments]:
    """Per-stroke ink moments of the reference outlines (cached on the entry)."""
    key = ("moments", size)
    if key not in entry.cache:
        entry.cache[key] = [Moments(*ink_moments(polygons=[o], size=size)[:2]) for o in entry.outlines]
    return entry.cache[key]


def score_character(
    strokes: list[tuple[np.ndarray, float]], entry: CharacterEntry, brush: str = "disc", size: int = SCORE_SIZE
) -> CharacterScore:
    """strokes: [(points (N, 3) canvas frame, width canvas units), ...] in drawing order."""
    b = get_brush(brush)
    n_a, n_t = len(strokes), entry.n_strokes
    denom = max(n_a, n_t, 1)
    if n_a == 0:
        return CharacterScore(float("inf"), 0.0, 0.0, 0.0, n_agent=0, n_target=n_t)

    geoms = [b.geometry(p, w) for p, w in strokes]
    offset, ratio = placement(geoms, entry.outlines)

    # Index matching: agent stroke i vs target stroke i. Strokes without ink can't
    # be aligned; they score 0 and stay out of the fit.
    n = min(n_a, n_t)
    targets = target_moments(entry, size)
    agent_m, target_m, idx = [], [], []
    for i in range(n):
        a = ink_moments([geoms[i]], size=size)
        if a is None:
            continue
        agent_m.append(Moments(a[0], a[1]))
        target_m.append(targets[i])
        idx.append(i)

    def iou_under(sim: Similarity, i: int) -> float:
        return geometry_iou([geoms[i].transformed(sim.s, sim.t)], [entry.outlines[i]], size)

    sim, inlier_mask = robust_fit(agent_m, target_m) if idx else (Similarity(), np.zeros(0, dtype=bool))
    structure_strokes, shape_strokes, inliers = [0.0] * n, [0.0] * n, [False] * n
    for k, i in enumerate(idx):
        structure_strokes[i] = iou_under(sim, i)
        shape_strokes[i] = iou_under(fit_stroke(agent_m[k], target_m[k]), i)
        inliers[i] = bool(inlier_mask[k])

    return CharacterScore(
        placement_offset=offset,
        placement_ratio=ratio,
        structure=float(sum(structure_strokes) / denom),
        shape=float(sum(shape_strokes) / denom),
        structure_strokes=structure_strokes,
        shape_strokes=shape_strokes,
        n_agent=n_a,
        n_target=n_t,
        align_scale=sim.s,
        align_offset=sim.t,
        align_inliers=inliers,
    )
