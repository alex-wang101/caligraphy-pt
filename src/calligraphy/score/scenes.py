"""Synthetic drawings for testing the scorer.

- A, B, C each break exactly one scoring level (v2 section 3.3).
- The normalization scenarios move, resize, squash or rotate an otherwise
  perfect character, to show what the environment forgives and what it doesn't.

All scenes start from the converter output, in canvas frame: [(points (N, 3), width), ...].
`through_environment` runs a drawing through the same path as a model response
(JSON -> parser -> Canvas clamping), so scores match what training would see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from calligraphy.canvas import CanvasConfig
from calligraphy.convert import convert
from calligraphy.data.types import CharacterEntry
from calligraphy.frame import CANVAS
from calligraphy.tools import format_strokes, parse_response, render_response

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


def outline_center(entry: CharacterEntry) -> np.ndarray:
    """Centre of the reference ink's bounding box."""
    pts = np.vstack(entry.outlines)
    return (pts.min(axis=0) + pts.max(axis=0)) / 2


def transformed(
    entry: CharacterEntry,
    scale: float = 1.0,
    sx: float | None = None,
    sy: float | None = None,
    dx: float = 0.0,
    dy: float = 0.0,
    rotate_deg: float = 0.0,
) -> Drawing:
    """The perfect drawing scaled (or squashed), rotated about its own centre, then shifted.

    Widths scale with the geometric mean of the x and y factors.
    """
    ax, ay = sx if sx is not None else scale, sy if sy is not None else scale
    c = outline_center(entry)
    th = math.radians(rotate_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = []
    for pts, w in perfect(entry):
        p = pts.copy()
        p[:, :2] = ((p[:, :2] - c) * [ax, ay]) @ rot.T + c + [dx, dy]
        out.append((p, w * math.sqrt(ax * ay)))
    return out


@dataclass(frozen=True)
class Scenario:
    label: str
    category: str  # "size" | "position" | "shape"
    kwargs: dict


# Positions are in canvas units (the canvas is 1024 wide). Most characters fill
# about 80% of the box, so a full-size character has little room to move.
NORMALIZATION = {
    "small_0.5": Scenario("Too small: 0.5x", "size", {"scale": 0.5}),
    "tiny_0.2": Scenario("Tiny: 0.2x", "size", {"scale": 0.2}),
    "big_1.15": Scenario("A bit big: 1.15x", "size", {"scale": 1.15}),
    "huge_1.6": Scenario("Too big: 1.6x, off canvas", "size", {"scale": 1.6}),
    "offcenter_inside": Scenario("Off-centre: 0.6x, right + down", "position", {"scale": 0.6, "dx": 190, "dy": 170}),
    "offcenter_corner": Scenario("Corner: 0.4x, top-left", "position", {"scale": 0.4, "dx": -300, "dy": -300}),
    "offcenter_off_canvas": Scenario("Past the edge: shifted right 250", "position", {"dx": 250}),
    "squash_x_0.6": Scenario("Squashed: 0.6x wide", "shape", {"sx": 0.6, "sy": 1.0}),
    "rotate_8": Scenario("Rotated 8 degrees", "shape", {"rotate_deg": 8}),
    "rotate_20": Scenario("Rotated 20 degrees", "shape", {"rotate_deg": 20}),
}

# The normalization test set: the baseline, every scenario, and scene A (small, in a corner).
NORMALIZATION_ORDER = ["perfect", *NORMALIZATION, "A_small_corner"]
NORMALIZATION_CHARS = "觀永国心"


def scene_category(name: str) -> str:
    if name in NORMALIZATION:
        return NORMALIZATION[name].category
    return {"perfect": "baseline", "A_small_corner": "position"}.get(name, "scorer levels")


def scenario(name: str):
    spec = NORMALIZATION[name]
    return lambda entry: transformed(entry, **spec.kwargs)


SCENES = {
    "perfect": perfect,
    "A_small_corner": small_in_corner,
    "B_slid_component": slid_component,
    "C_straight_flat": straight_flat,
    **{name: scenario(name) for name in NORMALIZATION},
}

SCENE_LABELS = {
    "perfect": "Converter output (SFT label)",
    "A_small_corner": "Scene A: small, in a corner",
    "B_slid_component": "Scene B: component slid",
    "C_straight_flat": "Scene C: straight, flat pressure",
    **{name: spec.label for name, spec in NORMALIZATION.items()},
}


def count_out_of_bounds(drawing: Drawing, frame: float = CANVAS) -> int:
    """Points the canvas will clamp to its edge."""
    return int(sum(((p[:, :2] < 0) | (p[:, :2] > frame)).any(axis=1).sum() for p, _ in drawing))


def to_response(drawing: Drawing) -> str:
    """The drawing as a model would write it: phase-1 JSON with integer coordinates."""
    return format_strokes(
        [{"points": [[round(x), round(y), round(q, 2)] for x, y, q in p.tolist()], "width": round(w, 1)} for p, w in drawing]
    )


def through_environment(drawing: Drawing, config: CanvasConfig | None = None) -> Drawing:
    """What the environment actually scores: parsed, validated and clamped by the Canvas."""
    canvas = render_response(parse_response(to_response(drawing), config), config)
    return [(s.points, s.width) for s in canvas.strokes]
