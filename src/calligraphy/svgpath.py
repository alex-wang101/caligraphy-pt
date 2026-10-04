"""Parse Make Me a Hanzi stroke outlines (absolute M, L, Q, C, Z) into polygons.

The outline traces the edge of the ink, not the brush path: M is an arbitrary
point on the edge, Q/C control points lie off the outline.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from calligraphy.frame import mmah_to_canvas

N_COORDS = {"M": 1, "L": 1, "Q": 2, "C": 3, "Z": 0}
_TOKEN = re.compile(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")

Command = tuple[str, np.ndarray]  # (letter, (k, 2) coordinates)


def parse_path(d: str) -> list[Command]:
    tokens = _TOKEN.findall(d)
    cmds: list[Command] = []
    i = 0
    while i < len(tokens):
        c = tokens[i]
        if c not in N_COORDS:
            raise ValueError(f"unsupported SVG command {c!r} in {d[:60]!r}")
        i += 1
        nums = []
        while i < len(tokens) and not tokens[i].isalpha():
            nums.append(float(tokens[i]))
            i += 1
        k = N_COORDS[c]
        if len(nums) != 2 * k:
            raise ValueError(f"{c} expects {2 * k} numbers, got {len(nums)} in {d[:60]!r}")
        cmds.append((c, np.array(nums, dtype=np.float64).reshape(k, 2)))
    if not cmds or cmds[0][0] != "M":
        raise ValueError(f"path must start with M: {d[:60]!r}")
    return cmds


def transform_path(cmds: list[Command], fn: Callable[[np.ndarray], np.ndarray]) -> list[Command]:
    """Apply fn to every coordinate, control points included."""
    return [(c, fn(pts) if len(pts) else pts) for c, pts in cmds]


def path_to_polygon(cmds: list[Command], q_samples: int = 8, c_samples: int = 12) -> np.ndarray:
    """Sample the outline into an (N, 2) polygon (first point not repeated)."""
    out: list[np.ndarray] = []
    cur = start = None
    for c, pts in cmds:
        if c == "M":
            cur = start = pts[0]
            out.append(cur[None])
        elif c == "L":
            cur = pts[0]
            out.append(cur[None])
        elif c == "Q":
            t = np.linspace(0, 1, q_samples + 1)[1:, None]
            ctrl, end = pts
            out.append((1 - t) ** 2 * cur + 2 * (1 - t) * t * ctrl + t**2 * end)
            cur = end
        elif c == "C":
            t = np.linspace(0, 1, c_samples + 1)[1:, None]
            c1, c2, end = pts
            out.append((1 - t) ** 3 * cur + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * end)
            cur = end
        elif c == "Z":
            cur = start
    poly = np.vstack(out)
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    return poly


def path_to_d(cmds: list[Command], ndigits: int = 1) -> str:
    parts = []
    for c, pts in cmds:
        nums = " ".join(f"{v:.{ndigits}f}".rstrip("0").rstrip(".") for v in pts.ravel())
        parts.append(f"{c} {nums}" if nums else c)
    return " ".join(parts)


def mmah_outline(d: str) -> tuple[str, np.ndarray]:
    """MMAH outline string -> (canvas-frame path string, canvas-frame polygon)."""
    cmds = transform_path(parse_path(d), mmah_to_canvas)
    return path_to_d(cmds), path_to_polygon(cmds)
