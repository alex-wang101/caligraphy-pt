"""One multi-turn episode: a canvas, the tool calls made on it, and a trace of everything that happened.

Framework-free. Whatever drives the model (the agent runner over HTTP, or a
trainer) calls `Episode.call` for each tool call and gets the same trace:

- `model` events: one per model reply (text, reasoning, stop reason, latency, usage);
- `tool` events: one per tool call (arguments, result text, whether it was accepted,
  the strokes on the canvas afterwards, the image shown for a capture, and the
  canvas score after every change);
- `nudge` and `end` events.

The step scores use the same structure score as the reward ("what this canvas
would score if submitted now"). They are logged for evaluation and reward
shaping and are never shown to the model.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from calligraphy.canvas import Canvas, CanvasConfig
from calligraphy.data.types import CharacterEntry
from calligraphy.score.levels import score_character
from calligraphy.tools import ToolResult, dispatch, format_strokes

END_REASONS = ("submit", "max_turns", "max_tool_calls", "no_tool_call", "error", "rate_limited")


@dataclass(frozen=True)
class EpisodeConfig:
    max_turns: int = 64  # model replies per episode (the driver counts them)
    max_tool_calls: int = 128
    max_nudges: int = 2  # reminders the driver may send when a reply has no tool call
    score_steps: bool = True  # score the canvas after every change

    def to_dict(self) -> dict:
        return asdict(self)


class Episode:
    def __init__(
        self,
        entry: CharacterEntry | None,
        canvas: CanvasConfig | None = None,
        config: EpisodeConfig | None = None,
        image_dir: Path | None = None,
    ):
        self.entry = entry
        self.canvas_config = canvas or CanvasConfig()
        self.config = config or EpisodeConfig()
        self.canvas = Canvas(self.canvas_config)
        self.image_dir = image_dir  # captures are saved here as step-NNN.png
        self.events: list[dict] = []
        self.end_reason: str | None = None
        self._t0 = time.perf_counter()
        self._raw: dict[int, dict] = {}  # stroke id -> the accepted call's arguments, in action units
        self._deleted: set[int] = set()
        self._last_score: dict | None = None
        self._n_calls = 0

    @property
    def done(self) -> bool:
        return self.end_reason is not None

    def _event(self, type: str, **fields) -> dict:
        event = {"i": len(self.events), "type": type, "t_s": round(time.perf_counter() - self._t0, 3), **fields}
        self.events.append(event)
        return event

    # --- what the driver reports -------------------------------------------------

    def model_turn(
        self,
        turn: int,
        text: str = "",
        reasoning: str = "",
        stop_reason: str | None = None,
        latency_s: float = 0.0,
        usage: dict | None = None,
        n_tool_calls: int = 0,
        error: str | None = None,
    ) -> dict:
        return self._event(
            "model",
            turn=turn,
            text=text,
            reasoning=reasoning,
            stop_reason=stop_reason,
            latency_s=round(latency_s, 2),
            usage=usage or {},
            n_tool_calls=n_tool_calls,
            error=error,
        )

    def nudge(self, turn: int, text: str) -> dict:
        return self._event("nudge", turn=turn, text=text)

    def end(self, reason: str) -> None:
        """Close the episode. The first reason given wins."""
        if reason not in END_REASONS:
            raise ValueError(f"unknown end reason {reason!r}; choose from {END_REASONS}")
        if self.end_reason is None:
            self.end_reason = reason
            self._event("end", reason=reason)

    # --- tool calls ---------------------------------------------------------------

    def call(self, name: str, arguments: str | dict | None, turn: int = 0, call_id: str | None = None) -> ToolResult:
        """Run one tool call and record it. `arguments` is the model's JSON string, or a dict."""
        if self.done:
            result = ToolResult("not run: the episode had already ended", ok=False)
            self._event("tool", turn=turn, call_id=call_id, name=name, arguments=None, ok=False, skipped=True, result=result.text)
            return result

        args, raw = arguments, None
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                args, raw = None, arguments
                result = ToolResult(f"error: arguments are not valid JSON ({e.msg})", ok=False)
        elif arguments is None:
            args = {}
        if raw is None:
            result = dispatch(self.canvas, name, args)

        event = self._event(
            "tool",
            turn=turn,
            call_id=call_id,
            name=name,
            arguments=args,
            ok=result.ok,
            result=result.text,
            stroke_id=result.stroke_id,
            stroke_ids=self.canvas.stroke_ids,
        )
        if raw is not None:
            event["arguments_raw"] = raw
        if result.ok and name == "create_stroke":
            self._raw[result.stroke_id] = {"points": args["points"], "width": args["width"]}
            event["redraw"] = result.stroke_id in self._deleted
            event["score"] = self._score()
        elif result.ok and name == "delete_stroke":
            self._raw.pop(result.stroke_id, None)
            self._deleted.add(result.stroke_id)
            # How good the removed stroke was, by the score just before the delete.
            event["deleted_stroke_score"] = ((self._last_score or {}).get("strokes") or {}).get(str(result.stroke_id))
            event["score"] = self._score()
        elif result.image is not None:
            event["image_size"] = result.image.size[0]
            if self.image_dir is not None:
                self.image_dir.mkdir(parents=True, exist_ok=True)
                event["image"] = f"step-{event['i']:03d}.png"
                result.image.save(self.image_dir / event["image"])

        self._n_calls += 1
        if result.done:
            self.end("submit")
        elif self._n_calls >= self.config.max_tool_calls:
            self.end("max_tool_calls")
        return result

    def _score(self) -> dict | None:
        """Structure score of the canvas as it stands, with each stroke's share keyed by stroke id."""
        if self.entry is None or not self.config.score_steps:
            return None
        strokes = [(s.points, s.width) for s in self.canvas.strokes]
        if not strokes:
            self._last_score = {"structure": 0.0, "shape": 0.0, "strokes": {}}
            return self._last_score
        score = score_character(strokes, self.entry, brush=self.canvas_config.brush)
        per_stroke = zip(self.canvas.stroke_ids, score.structure_strokes)  # strokes past the target count aren't scored
        self._last_score = {
            "structure": round(score.structure, 4),
            "shape": round(score.shape, 4),
            "strokes": {str(i): round(v, 3) for i, v in per_stroke},
        }
        return self._last_score

    # --- results --------------------------------------------------------------------

    def final_strokes(self) -> list[dict]:
        """The strokes left on the canvas, in stroke order, as the model gave them (action units)."""
        return [self._raw[i] for i in self.canvas.stroke_ids]

    def final_response(self) -> str:
        """The final canvas in the phase-1 answer format, so it is scored and viewed the same way."""
        return format_strokes(self.final_strokes())

    def summary(self) -> dict:
        tools = [e for e in self.events if e["type"] == "tool" and not e.get("skipped")]
        ok = [e for e in tools if e["ok"]]
        changes = [e for e in ok if e["name"] in ("create_stroke", "delete_stroke")]
        scores = [e["score"]["structure"] for e in changes if e.get("score")]
        first_delete = next((k for k, e in enumerate(changes) if e["name"] == "delete_stroke"), None)
        draft = changes[first_delete - 1] if first_delete else None  # the canvas just before the first delete
        last_change = changes[-1]["i"] if changes else -1
        return {
            "end_reason": self.end_reason,
            "n_turns": sum(e["type"] == "model" for e in self.events),
            "n_tool_calls": len(tools),
            "calls": dict(Counter(e["name"] for e in tools)),
            "n_failed_calls": len(tools) - len(ok),
            "n_skipped_calls": sum(e["type"] == "tool" and bool(e.get("skipped")) for e in self.events),
            "n_captures": sum(e["name"] == "capture_canvas" for e in ok),
            "n_deletes": sum(e["name"] == "delete_stroke" for e in ok),
            "n_redraws": sum(bool(e.get("redraw")) for e in ok),
            "n_nudges": sum(e["type"] == "nudge" for e in self.events),
            "n_strokes": len(self.canvas.strokes),
            # Did the model look at the canvas after its last change?
            "checked_final": any(e["name"] == "capture_canvas" and e["i"] > last_change for e in ok) if changes else False,
            "final_score": scores[-1] if scores else None,
            "peak_score": max(scores) if scores else None,
            "draft_score": draft["score"]["structure"] if draft and draft.get("score") else None,
            "duration_s": round(time.perf_counter() - self._t0, 2),
        }

    def to_dict(self) -> dict:
        return {"events": self.events, "summary": self.summary(), "final_strokes": self.final_strokes()}


def replay(events: list[dict], upto: int, canvas: CanvasConfig | None = None) -> Canvas:
    """The canvas as it stood after event `upto`, rebuilt from a trace's events (-1: before the first one).

    Only accepted create_stroke and delete_stroke calls change the canvas, so the
    state before tool call i is `replay(events, i - 1)` and the state after it is
    `replay(events, i)`.
    """
    board = Canvas(canvas)
    for e in events:
        if e["i"] > upto:
            break
        if e["type"] != "tool" or not e.get("ok"):
            continue
        if e["name"] == "create_stroke":
            board.create_stroke(e["arguments"]["points"], e["arguments"]["width"])
        elif e["name"] == "delete_stroke":
            board.delete_stroke(e["arguments"]["stroke_id"])
    return board
