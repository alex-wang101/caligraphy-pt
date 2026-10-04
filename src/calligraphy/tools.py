"""Tool API and phase-1 response format. No trainer imports: adapters wrap these.

Phase 1 (single turn): the model writes every stroke in one JSON response,
    {"strokes": [{"points": [[x, y, p], ...], "width": w}, ...]}
and the environment renders and scores once. Phase 2 (multi-turn) uses the tool
schemas below; create_stroke returns text only, images come from capture_canvas.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from PIL import Image

from calligraphy.canvas import Canvas, CanvasConfig, StrokeError, validate_stroke


class ParseError(ValueError):
    """The response is not a valid phase-1 stroke list. Reward 0."""


@dataclass
class ToolResult:
    text: str
    image: Image.Image | None = None
    done: bool = False


def tool_schemas(config: CanvasConfig | None = None) -> list[dict]:
    f = (config or CanvasConfig()).action_frame
    n = (config or CanvasConfig()).max_points
    return [
        {
            "type": "function",
            "function": {
                "name": "create_stroke",
                "description": (
                    f"Draw one brush stroke on the {f}x{f} canvas (origin top-left, y down). "
                    f"Points are in writing order, at most {n}; p is pressure 0-1 and the brush "
                    "radius is width/2 * p."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "points": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                            "minItems": 1,
                            "maxItems": n,
                            "description": "[[x, y, p], ...]",
                        },
                        "width": {"type": "number", "description": "brush diameter at full pressure"},
                    },
                    "required": ["points", "width"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "capture_canvas",
                "description": "Return an image of the canvas with the 米字格 guide grid.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Finish the character.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def dispatch(canvas: Canvas, name: str, args: dict) -> ToolResult:
    """Run one tool call. Invalid calls return error text instead of raising."""
    if name == "create_stroke":
        try:
            canvas.create_stroke(args.get("points"), args.get("width"))
        except StrokeError as e:
            return ToolResult(f"error: {e}")
        return ToolResult(f"stroke {len(canvas.strokes)} created")
    if name == "capture_canvas":
        return ToolResult(f"canvas with {len(canvas.strokes)} strokes", image=canvas.capture())
    if name == "submit":
        return ToolResult(f"submitted {len(canvas.strokes)} strokes", done=True)
    return ToolResult(f"error: unknown tool {name!r}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_json(text: str):
    text = _THINK.sub("", text)  # reasoning models (e.g. Qwen3) may think out loud before answering
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for c in candidates:
        c = c.strip()
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            pass
        start, end = c.find("{"), c.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(c[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ParseError("no JSON object found")


def parse_response(text: str, config: CanvasConfig | None = None) -> list[dict]:
    """Phase-1 response -> list of {"points", "width"} in action units. Raises ParseError.

    Structure is strict (any malformed stroke fails the whole response);
    out-of-range values are left for Canvas to clamp.
    """
    config = config or CanvasConfig()
    obj = _extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("strokes"), list):
        raise ParseError('expected {"strokes": [...]}')
    strokes = obj["strokes"]
    if not strokes:
        raise ParseError("no strokes")
    if len(strokes) > config.max_strokes:
        raise ParseError(f"too many strokes: {len(strokes)} > {config.max_strokes}")
    out = []
    for i, s in enumerate(strokes):
        if not isinstance(s, dict):
            raise ParseError(f"stroke {i} is not an object")
        try:
            validate_stroke(s.get("points"), s.get("width"), config)
        except StrokeError as e:
            raise ParseError(f"stroke {i}: {e}") from None
        out.append({"points": s["points"], "width": s["width"]})
    return out


def render_response(strokes: list[dict], config: CanvasConfig | None = None) -> Canvas:
    canvas = Canvas(config)
    for s in strokes:
        canvas.create_stroke(s["points"], s["width"])
    return canvas


def format_strokes(strokes: list[dict]) -> str:
    """Serialize strokes in the phase-1 response format (compact, one stroke per line)."""
    lines = [json.dumps(s, separators=(",", ":"), ensure_ascii=False) for s in strokes]
    return '{"strokes": [\n' + ",\n".join(lines) + "\n]}"


def phase1_messages(character: str, pinyin: str | None = None, config: CanvasConfig | None = None) -> list[dict]:
    """Chat messages for the single-turn task. Only the character's name is given, never an image."""
    config = config or CanvasConfig()
    f, h, n = config.action_frame, config.action_frame // 2, config.max_points
    system = (
        "You write Chinese characters with a brush on a square canvas.\n"
        f"Canvas: {f}x{f}, origin at the top-left, x to the right, y down. A 米字格 guide grid "
        f"marks the border, the centre lines x={h} and y={h}, and both diagonals; centre the "
        "character in the square.\n"
        "Each stroke is a list of points [x, y, p] in the direction the brush moves, from where "
        f"it lands to where it lifts (at most {n} points). p is pressure from 0 to 1: the brush "
        "radius at a point is width/2 * p. width is the brush diameter at full pressure.\n"
        "Write the strokes in standard stroke order. Reply with JSON only:\n"
        '{"strokes": [{"points": [[x, y, p], ...], "width": w}, ...]}'
    )
    name = f"{character} ({pinyin})" if pinyin else character
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Write the character {name}."}]
