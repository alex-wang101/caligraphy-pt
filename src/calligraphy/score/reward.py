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
    info = {"parse_ok": True, **score.to_dict()}
    return score.structure, info
