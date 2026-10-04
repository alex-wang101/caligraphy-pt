"""Reward v1: structure score with index matching; 0 if the response does not parse.

All levels are returned in `info` for logging even though only structure is
rewarded. Add a placement term, direction check or Hungarian matching only once
the failure they fix is observed (docs/IMPLEMENTATION.md 3.3).
"""

from __future__ import annotations

from calligraphy.canvas import CanvasConfig
from calligraphy.data.types import CharacterEntry
from calligraphy.score.levels import score_character
from calligraphy.tools import ParseError, parse_response, render_response


def compute_reward(response: str, entry: CharacterEntry, config: CanvasConfig | None = None) -> tuple[float, dict]:
    config = config or CanvasConfig()
    try:
        strokes = parse_response(response, config)
    except ParseError as e:
        return 0.0, {"parse_ok": False, "error": str(e), "n_agent": 0, "n_target": entry.n_strokes}
    canvas = render_response(strokes, config)
    score = score_character([(s.points, s.width) for s in canvas.strokes], entry, brush=config.brush)
    info = {"parse_ok": True, **score.to_dict(), "clamped_points": count_clamped(strokes, config)}
    return score.structure, info


def count_clamped(strokes: list[dict], config: CanvasConfig) -> int:
    """Points outside [0, action_frame] that the canvas pinned to its edge.

    Placement is measured after clamping, so a drawing far too big can still
    look in-bounds there; this count is the tell.
    """
    f = config.action_frame
    return sum(1 for s in strokes for x, y, _ in s["points"] if not (0 <= x <= f and 0 <= y <= f))
