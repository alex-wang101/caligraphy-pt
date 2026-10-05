"""The tools the model can call, and the phase-1 answer format. No trainer imports: adapters wrap these.

Every tool is one `Tool` object in this file: its name, the description the model
reads, its argument schema and the function that runs it. `TOOLS` lists them in
the order the model sees them. The schemas sent to the model, the tool list in
the system prompt and `dispatch` are all built from `TOOLS`, so adding or
changing a tool is a change in one place.

Phase 1 (single turn): the model writes every stroke in one JSON response,
    {"strokes": [{"points": [[x, y, p], ...], "width": w}, ...]}
and the environment renders and scores once. Phase 2 (multi-turn) uses the tools:
create_stroke and delete_stroke return text only, images come from
capture_canvas, submit ends the episode. The canvas holds the drawing
(`calligraphy.canvas`); `calligraphy.episode` runs the calls and records the trace.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from calligraphy.canvas import Canvas, CanvasConfig, StrokeError, format_ids, validate_stroke


class ParseError(ValueError):
    """The response is not a valid phase-1 stroke list. Reward 0."""


@dataclass
class ToolResult:
    text: str
    image: Image.Image | None = None
    done: bool = False
    ok: bool = True  # False: the call was rejected and changed nothing
    stroke_id: int | None = None  # the stroke a create_stroke or delete_stroke acted on


@dataclass(frozen=True)
class Tool:
    """One tool: what the model is told about it and what it does to the canvas."""

    name: str
    description: str  # may use {f} (canvas side in action units) and {n} (most points per stroke)
    parameters: Callable[[CanvasConfig], dict]  # JSON schema of the arguments
    run: Callable[[Canvas, dict], ToolResult]  # rejects a bad call with ok=False instead of raising

    def describe(self, config: CanvasConfig) -> str:
        return self.description.format(f=config.action_frame, n=config.max_points)

    def schema(self, config: CanvasConfig) -> dict:
        """The function schema sent to the model (OpenAI chat-completions format)."""
        return {"type": "function", "function": {"name": self.name, "description": self.describe(config), "parameters": self.parameters(config)}}


def _no_arguments(config: CanvasConfig) -> dict:
    return {"type": "object", "properties": {}}


def _create_stroke_parameters(config: CanvasConfig) -> dict:
    return {
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "minItems": 1,
                "maxItems": config.max_points,
                "description": "[[x, y, p], ...]",
            },
            "width": {"type": "number", "description": "brush diameter at full pressure"},
        },
        "required": ["points", "width"],
    }


def _create_stroke(canvas: Canvas, args: dict) -> ToolResult:
    stroke_id = canvas.next_id()
    try:
        canvas.create_stroke(args.get("points"), args.get("width"))
    except StrokeError as e:
        return ToolResult(f"error: {e}", ok=False)
    return ToolResult(f"stroke {stroke_id} created; the canvas has {len(canvas.strokes)} strokes", stroke_id=stroke_id)


def _delete_stroke_parameters(config: CanvasConfig) -> dict:
    return {
        "type": "object",
        "properties": {"stroke_id": {"type": "integer", "description": "number of the stroke to remove"}},
        "required": ["stroke_id"],
    }


def _delete_stroke(canvas: Canvas, args: dict) -> ToolResult:
    try:
        canvas.delete_stroke(args.get("stroke_id"))
    except StrokeError as e:
        return ToolResult(f"error: {e}", ok=False)
    stroke_id = int(args["stroke_id"])
    return ToolResult(
        f"stroke {stroke_id} deleted; strokes on the canvas: {format_ids(canvas.stroke_ids)}. "
        f"The next stroke you create becomes stroke {canvas.next_id()}.",
        stroke_id=stroke_id,
    )


def _capture_canvas(canvas: Canvas, args: dict) -> ToolResult:
    return ToolResult(f"canvas image with {len(canvas.strokes)} strokes", image=canvas.capture())


def _submit(canvas: Canvas, args: dict) -> ToolResult:
    return ToolResult(f"submitted {len(canvas.strokes)} strokes", done=True)


CREATE_STROKE = Tool(
    name="create_stroke",
    description=(
        "Draw one brush stroke on the {f}x{f} canvas. points is a list of [x, y, p] in the direction the "
        "brush moves, from where it lands to where it lifts (at most {n} points). p is pressure from 0 to 1: "
        "the brush radius at a point is width/2 * p. width is the brush diameter at full pressure. "
        "Strokes are numbered in the order you create them."
    ),
    parameters=_create_stroke_parameters,
    run=_create_stroke,
)

DELETE_STROKE = Tool(
    name="delete_stroke",
    description=(
        "Remove one stroke from the canvas by its number. The other strokes keep their numbers, and the "
        "next stroke you create takes the deleted stroke's number and its place in the stroke order."
    ),
    parameters=_delete_stroke_parameters,
    run=_delete_stroke,
)

CAPTURE_CANVAS = Tool(
    name="capture_canvas",
    description="Show an image of the canvas as it is now, with the 米字格 guide grid.",
    parameters=_no_arguments,
    run=_capture_canvas,
)

SUBMIT = Tool(
    name="submit",
    description="Finish the character. Call it once, when the character is complete.",
    parameters=_no_arguments,
    run=_submit,
)

# Every tool, in the order the model sees them.
TOOLS: dict[str, Tool] = {t.name: t for t in (CREATE_STROKE, DELETE_STROKE, CAPTURE_CANVAS, SUBMIT)}


def tool_schemas(config: CanvasConfig | None = None) -> list[dict]:
    config = config or CanvasConfig()
    return [t.schema(config) for t in TOOLS.values()]


def dispatch(canvas: Canvas, name: str, args: dict) -> ToolResult:
    """Run one tool call. Invalid calls return error text instead of raising, and change nothing."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(f"error: unknown tool {name!r}; the tools are {', '.join(TOOLS)}", ok=False)
    if not isinstance(args, dict):
        return ToolResult("error: arguments must be a JSON object", ok=False)
    return tool.run(canvas, args)


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


def phase2_messages(character: str, pinyin: str | None = None, config: CanvasConfig | None = None) -> list[dict]:
    """Chat messages that open the multi-turn task. Only the character's name is given, never a reference image."""
    config = config or CanvasConfig()
    f, h = config.action_frame, config.action_frame // 2
    system = (
        "You write Chinese characters with a brush on a square canvas, one stroke at a time, by calling tools.\n"
        f"Canvas: {f}x{f}, origin at the top-left, x to the right, y down. A 米字格 guide grid "
        f"marks the border, the centre lines x={h} and y={h}, and both diagonals; centre the "
        "character in the square.\n"
        "Write the strokes in standard stroke order.\n"
        "Tools:\n" + "".join(f"- {t.name}: {t.describe(config)}\n" for t in TOOLS.values()) + "You may call several tools in one reply."
    )
    name = f"{character} ({pinyin})" if pinyin else character
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Write the character {name}."}]
