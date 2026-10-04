"""Scale + translation alignment (no rotation) on index-matched per-stroke ink moments.

Each stroke is summarised by its ink centroid and RMS spread, measured on the
same masks as IoU. Moments don't depend on how a path is parametrised: path
points paired by arc length are biased when the agent's path runs to the
stroke tips but the median stops short, while ink moments are not.

Plain least squares is dragged by misplaced strokes (7 slid strokes in 觀 pull
the fit sideways and zero the 18 correct ones), so the robust fit keeps the
best-fitting 60% of strokes and refits, 3 times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SCALE_RANGE = (0.25, 4.0)


@dataclass(frozen=True)
class Moments:
    c: np.ndarray  # ink centroid (2,)
    spread: float  # RMS distance of ink from the centroid


@dataclass(frozen=True)
class Similarity:
    s: float = 1.0
    t: tuple[float, float] = (0.0, 0.0)

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        out = np.array(pts, dtype=np.float64)
        out[:, :2] = out[:, :2] * self.s + np.asarray(self.t)
        return out

    def apply_width(self, width: float) -> float:
        return width * self.s


def _clip_scale(s: float) -> float:
    return float(np.clip(s, *SCALE_RANGE))


def fit(agent: list[Moments], target: list[Moments]) -> Similarity:
    """Least squares on s*c_a + t = c_t and s*spread_a = spread_t over all strokes."""
    if not agent:
        return Similarity()
    rows, rhs = [], []
    for a, b in zip(agent, target):
        rows += [[a.c[0], 1, 0], [a.c[1], 0, 1], [a.spread, 0, 0]]
        rhs += [b.c[0], b.c[1], b.spread]
    (s, _, _), *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    s = _clip_scale(s)
    # For a fixed scale the least-squares translation is the mean centroid offset;
    # recomputing it keeps t optimal when s was clipped.
    t = np.mean([b.c - s * a.c for a, b in zip(agent, target)], axis=0)
    return Similarity(s, (float(t[0]), float(t[1])))


def fit_stroke(agent: Moments, target: Moments) -> Similarity:
    """Exact per-stroke fit: match spread, then centroid."""
    s = _clip_scale(target.spread / agent.spread) if agent.spread > 1e-9 else 1.0
    t = target.c - s * agent.c
    return Similarity(s, (float(t[0]), float(t[1])))


def residuals(agent: list[Moments], target: list[Moments], sim: Similarity) -> np.ndarray:
    t = np.asarray(sim.t)
    return np.array(
        [np.linalg.norm(sim.s * a.c + t - b.c) + abs(sim.s * a.spread - b.spread) for a, b in zip(agent, target)]
    )


def robust_fit(agent: list[Moments], target: list[Moments], keep: float = 0.6, iters: int = 3) -> tuple[Similarity, np.ndarray]:
    """Fit, keep the `keep` fraction of strokes with the lowest residual, refit; repeat."""
    sim = fit(agent, target)
    inliers = np.ones(len(agent), dtype=bool)
    if len(agent) < 3:
        return sim, inliers
    k = max(2, math.ceil(keep * len(agent)))
    for _ in range(iters):
        order = np.argsort(residuals(agent, target, sim))
        inliers = np.zeros(len(agent), dtype=bool)
        inliers[order[:k]] = True
        sim = fit([a for a, ok in zip(agent, inliers) if ok], [b for b, ok in zip(target, inliers) if ok])
    return sim, inliers
