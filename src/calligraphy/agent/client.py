"""One chat completion from an OpenAI-compatible server or from Claude.

Errors never raise: they come back in `Completion.error`, so one bad call
doesn't stop a run.

Transcripts are kept in the OpenAI chat format for both backends: assistant
messages carry `tool_calls`, tool results are `role: "tool"` messages, and an
image is a user message with an `image_url` part. `_anthropic` converts on the
way in and out. Keys starting with "_" are private to this module and are never
sent to an OpenAI-compatible server.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Claude models that accept the server-side refusal fallback (`fallbacks: "default"`).
CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
RATE_LIMIT_BACKOFF_S = (10, 30, 60)  # waits before retrying a transient 429


def load_env_file(path: Path = ENV_FILE) -> list[str]:
    """Copy KEY=VALUE lines from a .env file into os.environ; variables already set win.

    Returns the names it set (never the values). A missing file is not an error.
    """
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if not m or line.lstrip().startswith("#"):
            continue
        name, value = m.group(1), m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name not in os.environ:
            os.environ[name] = value
            names.append(name)
    return names


@dataclass
class ModelConfig:
    provider: str
    kind: str  # "openai" | "anthropic"
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)  # never written to disk
    key_env: str | None = None  # read the key from this environment variable if api_key is unset
    temperature: float | None = 1.0  # ignored for Claude (current models reject it)
    max_tokens: int = 16000
    timeout: float = 600.0
    extra_body: dict = field(default_factory=dict)  # provider options, e.g. {"reasoning": {"enabled": False}}

    def resolved_key(self) -> str | None:
        return self.api_key or (os.environ.get(self.key_env) if self.key_env else None)

    def key_source(self) -> str:
        if self.api_key:
            return "pasted (memory only)"
        if self.key_env and os.environ.get(self.key_env):
            return f"env {self.key_env}"
        return "none"


@dataclass
class Completion:
    text: str
    stop_reason: str | None = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    error: str | None = None
    reasoning: str = ""  # thinking text, when the server returns it separately from the answer
    rate_limited: bool = False  # a 429 that retrying didn't clear (e.g. a free tier's daily cap)
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments": JSON string}, ...]
    message: dict | None = None  # the assistant message to append to the transcript before the tool results


def complete(config: ModelConfig, messages: list[dict], tools: list[dict] | None = None) -> Completion:
    """One model reply. `tools` are OpenAI-format function schemas (`calligraphy.tools.tool_schemas`)."""
    t0 = time.perf_counter()
    try:
        c = _anthropic(config, messages, tools) if config.kind == "anthropic" else _openai(config, messages, tools)
    except Exception as e:  # noqa: BLE001 - surface any failure as a per-call error
        c = Completion("", error=f"{type(e).__name__}: {e}")
    c.latency_s = time.perf_counter() - t0
    return c


def tool_message(call_id: str, text: str, ok: bool = True) -> dict:
    """The transcript message that answers one tool call."""
    return {"role": "tool", "tool_call_id": call_id, "content": text, "_is_error": not ok}


def image_message(text: str, png: bytes) -> dict:
    """A user message that shows the model an image (sent after the tool results of the turn that asked for it)."""
    import base64

    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]}


def _public(message: dict) -> dict:
    return {k: v for k, v in message.items() if not k.startswith("_")}


def _openai(config: ModelConfig, messages: list[dict], tools: list[dict] | None = None) -> Completion:
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = config.resolved_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body: dict = {"model": config.model, "messages": [_public(m) for m in messages], "max_tokens": config.max_tokens, **config.extra_body}
    if tools:
        body["tools"] = tools
    if config.temperature is not None:
        body["temperature"] = config.temperature

    with httpx.Client(timeout=config.timeout) as http:
        try:
            r = _post(http, url, body, headers)
            # Some hosted reasoning models want max_completion_tokens and only the default temperature.
            if r.status_code == 400 and "max_completion_tokens" in r.text:
                body["max_completion_tokens"] = body.pop("max_tokens")
                r = _post(http, url, body, headers)
            if r.status_code == 400 and "temperature" in r.text and "temperature" in body:
                body.pop("temperature")
                r = _post(http, url, body, headers)
        except httpx.ConnectError:
            return Completion("", error=f"could not connect to {config.base_url}: is the model server running?")
        except httpx.TimeoutException:
            return Completion("", error=f"no response from {config.base_url} within {config.timeout:.0f}s")
    if r.status_code == 429:
        return Completion("", error=f"rate limited (HTTP 429): {r.text[:300]}", rate_limited=True)
    if r.status_code != 200:
        return Completion("", error=f"HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")  # a JSON string by the spec; some local servers send an object
        calls.append(
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": fn.get("name") or "",
                "arguments": args if isinstance(args, str) else json.dumps(args if args is not None else {}),
            }
        )
    assistant: dict = {"role": "assistant", "content": message.get("content") or ""}
    if calls:
        assistant["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
    if message.get("reasoning_details"):  # OpenRouter: reasoning must be passed back for the model to keep it
        assistant["reasoning_details"] = message["reasoning_details"]
    return Completion(
        text=message.get("content") or "",
        stop_reason=choice.get("finish_reason"),
        usage=data.get("usage") or {},
        reasoning=message.get("reasoning") or message.get("reasoning_content") or "",
        tool_calls=calls,
        message=assistant,
    )


def _post(http: httpx.Client, url: str, body: dict, headers: dict) -> httpx.Response:
    """POST, waiting out transient 429s. A daily cap ("per-day") is returned at once: waiting won't clear it."""
    for wait in (*RATE_LIMIT_BACKOFF_S, None):
        r = http.post(url, json=body, headers=headers)
        if r.status_code != 429 or wait is None or "per-day" in r.text:
            return r
        retry_after = r.headers.get("retry-after", "")
        time.sleep(float(retry_after) if retry_after.isdigit() and int(retry_after) <= 120 else wait)
    return r


def _anthropic_part(part: dict) -> dict:
    if part.get("type") == "image_url":
        header, data = part["image_url"]["url"].split(",", 1)  # data:image/png;base64,<data>
        return {"type": "image", "source": {"type": "base64", "media_type": header[5:].split(";")[0], "data": data}}
    return part


def _anthropic_messages(messages: list[dict]) -> list[dict]:
    """OpenAI-format transcript -> Messages API turns.

    Every result of one assistant turn's tool calls must come back in a single
    user turn, results first, so neighbouring tool and user messages are merged.
    An assistant message this module produced is sent back as the exact content
    blocks the API returned (thinking blocks included).
    """
    out: list[dict] = []

    def add(role: str, blocks: list) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant":
            if "_anthropic_content" in m:
                add("assistant", m["_anthropic_content"])
                continue
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": json.loads(fn["arguments"] or "{}")})
            add("assistant", blocks)
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if m.get("_is_error"):
                block["is_error"] = True
            add("user", [block])
        elif isinstance(m["content"], str):
            add("user", [{"type": "text", "text": m["content"]}])
        else:
            add("user", [_anthropic_part(p) for p in m["content"]])
    return out


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    return [{"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]} for t in tools]


def _anthropic(config: ModelConfig, messages: list[dict], tools: list[dict] | None = None) -> Completion:
    import anthropic

    key = config.resolved_key()
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    kwargs = dict(model=config.model, max_tokens=config.max_tokens, messages=_anthropic_messages(messages), timeout=config.timeout)
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = _anthropic_tools(tools)
    try:
        if config.model in CLAUDE_FALLBACK_MODELS:
            response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            response = client.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        return Completion("", error="authentication failed: set ANTHROPIC_API_KEY, run `ant auth login`, or paste a key")
    except anthropic.NotFoundError as e:
        return Completion("", error=f"model or endpoint not found: {e.message}")
    except anthropic.RateLimitError as e:
        return Completion("", error=f"rate limited: {e.message}")
    except anthropic.APIStatusError as e:
        return Completion("", error=f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        return Completion("", error="could not reach the Anthropic API")

    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        return Completion("", stop_reason="refusal", usage=usage, error=f"refused ({category or 'no category'})")
    text = "".join(b.text for b in response.content if b.type == "text")
    calls = [{"id": b.id, "name": b.name, "arguments": json.dumps(b.input)} for b in response.content if b.type == "tool_use"]
    # The API's own content blocks go back unchanged on the next request.
    assistant = {"role": "assistant", "content": text, "_anthropic_content": list(response.content)}
    return Completion(text=text, stop_reason=response.stop_reason, usage=usage, tool_calls=calls, message=assistant)
