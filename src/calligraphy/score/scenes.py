"""Synthetic drawings that each break exactly one scoring level (v2 section 3.3).

All scenes start from the converter output, in canvas frame: [(points (N, 3), width), ...].
"""

from __future__ import annotations

import numpy as np

from calligraphy.convert import convert
from calligraphy.data.types import CharacterEntry

Drawing = list[tuple[np.ndarray, float]]


def perfect(entry: CharacterEntry) -> Drawing:
    return [(np.asarray(s["points"], dtype=np.float64), float(s["width"])) for s in convert(entry)]


def small_in_corner(entry: CharacterEntry, scale: float = 0.6, offset: tuple[float, float] = (20.0, 20.0)) -> Drawing:
    """A. Correct but small and in the top-left corner: placement only."""
    out = []
    for pts, w in perfect(entry):
        p = pts.copy()
        p[:, :2] = p[:, :2] * scale + np.asarray(offset)
        out.append((p, w * scale))
    return out


def component_strokes(entry: CharacterEntry, component: int) -> list[int]:
    """Stroke indices belonging to first-level IDS component `component` (dictionary `matches`)."""
    matches = entry.meta.get("matches") or []
    return [i for i, m in enumerate(matches) if m and m[0] == component]


def slid_component(entry: CharacterEntry, component: int = 1, dx: float = -180.0) -> Drawing:
    """B. Strokes perfect, but one component (觀: 見) slid sideways: structure only."""
    moved = set(component_strokes(entry, component))
    out = []
    for i, (pts, w) in enumerate(perfect(entry)):
        p = pts.copy()
        if i in moved:
            p[:, 0] += dx
        out.append((p, w))
    return out


def straight_flat(entry: CharacterEntry, pressure: float = 0.6) -> Drawing:
    """C. Layout right, but every stroke a straight line at flat pressure: stroke shape."""
    out = []
    for (pts, w), median in zip(perfect(entry), entry.medians):
        a, b = median[0], median[-1]
        out.append((np.array([[a[0], a[1], pressure], [b[0], b[1], pressure]]), w))
    return out


SCENES = {
    "perfect": perfect,
    "A_small_corner": small_in_corner,
    "B_slid_component": slid_component,
    "C_straight_flat": straight_flat,
}
