"""A fake OpenAI-compatible model that answers every prompt with the SFT converter output.

Use it to check the agent pipeline end to end without a real model (a run should
score about 0.9). Listens on 127.0.0.1:8100; in the viewer pick "Scripted baseline".

    uv run python scripts/stub_model.py

When the request carries tools (--mode tools), it plays a scripted agent that
makes one mistake and fixes it, so a run exercises every tool:
  reply 1: draws every stroke, one of them out of place, and asks for the canvas;
  reply 2: deletes that stroke, redraws it correctly and asks for the canvas again;
  reply 3: submits.
"""

from __future__ import annotations

import json
import re
import time

import uvicorn
from fastapi import FastAPI, HTTPException

from calligraphy.convert import convert
from calligraphy.data.mmah import get_entry
from calligraphy.tools import format_strokes

app = FastAPI(title="Scripted baseline model")


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": "converter", "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(body: dict) -> dict:
    messages = body.get("messages") or []
    prompt = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    char = re.search(r"Write the character (\S)", prompt)
    if not char:
        content = "I only answer calligraphy prompts."
    else:
        try:
            entry = get_entry(char.group(1))
        except KeyError:
            raise HTTPException(404, f"{char.group(1)!r} is not in Make Me a Hanzi") from None
        frame = re.search(r"Canvas: (\d+)x\d+", prompt)
        frame = int(frame.group(1)) if frame else 1024
        strokes = convert(entry, action_frame=frame)
        if body.get("tools"):
            return tool_reply(body, strokes, frame, sum(m.get("role") == "assistant" for m in messages))
        content = format_strokes(strokes)
    return {
        "id": f"stub-{time.time_ns()}",
        "object": "chat.completion",
        "model": body.get("model", "converter"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def scripted_calls(strokes: list[dict], frame: int, turn: int) -> list[tuple[str, dict]]:
    """The scripted agent's tool calls for its reply number `turn` (0-based)."""
    bad = min(1, len(strokes) - 1)  # the stroke it gets wrong: the second, or the only one
    if turn == 0:
        shift = 0.15 * frame
        wrong = {"points": [[x + shift, y, p] for x, y, p in strokes[bad]["points"]], "width": strokes[bad]["width"]}
        return [("create_stroke", wrong if i == bad else s) for i, s in enumerate(strokes)] + [("capture_canvas", {})]
    if turn == 1:
        return [("delete_stroke", {"stroke_id": bad + 1}), ("create_stroke", strokes[bad]), ("capture_canvas", {})]
    return [("submit", {})]


def tool_reply(body: dict, strokes: list[dict], frame: int, turn: int) -> dict:
    calls = [
        {"id": f"call_{turn}_{k}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for k, (name, args) in enumerate(scripted_calls(strokes, frame, turn))
    ]
    return {
        "id": f"stub-{time.time_ns()}",
        "object": "chat.completion",
        "model": body.get("model", "converter"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": calls}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)
