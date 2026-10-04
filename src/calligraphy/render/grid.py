"""米字格 practice grid: solid border, dashed centre cross and diagonals."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

GRID_COLOR = (226, 150, 150)
PAPER_COLOR = (255, 255, 255)


def _dashed(draw: ImageDraw.ImageDraw, a, b, dash: float, width: int, fill) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    n = max(1, int(length // (2 * dash)))
    for i in range(n):
        t0, t1 = (2 * i) / (2 * n), (2 * i + 1) / (2 * n)
        p, q = a + (b - a) * t0, a + (b - a) * t1
        draw.line([tuple(p), tuple(q)], fill=fill, width=width)


@lru_cache(maxsize=16)
def grid_background(size: int, grid: bool = True) -> Image.Image:
    """Paper with the 米字格 drawn on it. Cached; callers must copy before drawing."""
    img = Image.new("RGB", (size, size), PAPER_COLOR)
    if not grid:
        return img
    d = ImageDraw.Draw(img)
    w = max(1, round(size / 224))
    dash = size / 64
    m = size - 1
    half = m / 2
    for a, b in [((0, half), (m, half)), ((half, 0), (half, m)), ((0, 0), (m, m)), ((m, 0), (0, m))]:
        _dashed(d, a, b, dash, w, GRID_COLOR)
    d.rectangle([0, 0, m, m], outline=GRID_COLOR, width=2 * w)
    return img
