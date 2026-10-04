"""Round-trip check: render converter output and compare with the dataset's filled outlines.

This measures label and brush quality (how well disc strokes can reproduce the
font's ink), not the agent; the agent is not capped by it.
"""

from __future__ import annotations

import numpy as np

from calligraphy.convert import DEFAULT, ConvertConfig, convert
from calligraphy.data.types import CharacterEntry
from calligraphy.frame import CANVAS
from calligraphy.render.brushes import StrokeGeometry, get_brush
from calligraphy.render.raster import SCORE_SIZE, geometry_iou


def stroke_geometries(strokes: list[dict], brush: str = "disc", action_frame: int = CANVAS) -> list[StrokeGeometry]:
    b = get_brush(brush)
    unit = CANVAS / action_frame
    geoms = []
    for s in strokes:
        pts = np.asarray(s["points"], dtype=np.float64)
        pts[:, :2] *= unit
        geoms.append(b.geometry(pts, s["width"] * unit))
    return geoms


def roundtrip(entry: CharacterEntry, cfg: ConvertConfig = DEFAULT, brush: str = "disc", size: int = SCORE_SIZE) -> dict:
    strokes = convert(entry, cfg)
    geoms = stroke_geometries(strokes, brush)
    per_stroke = [geometry_iou([g], [o], size) for g, o in zip(geoms, entry.outlines)]
    return {
        "strokes": strokes,
        "stroke_iou": per_stroke,
        "char_iou": geometry_iou(geoms, entry.outlines, size),
    }
