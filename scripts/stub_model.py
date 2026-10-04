"""A fake OpenAI-compatible model that answers every phase-1 prompt with the SFT converter output.

Use it to check the agent pipeline end to end without a real model (a run should
score about 0.9). Listens on 127.0.0.1:8100; in the viewer pick "Scripted baseline".

    uv run python scripts/stub_model.py
"""

from __future__ import annotations

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
        content = format_strokes(convert(entry, action_frame=int(frame.group(1)) if frame else 1024))
    return {
        "id": f"stub-{time.time_ns()}",
        "object": "chat.completion",
        "model": body.get("model", "converter"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)
