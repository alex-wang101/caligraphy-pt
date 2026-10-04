"""One chat completion from an OpenAI-compatible server or from Claude.

Errors never raise: they come back in `Completion.error`, so one bad call
doesn't stop a run.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx

# Claude models that accept the server-side refusal fallback (`fallbacks: "default"`).
CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


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


def complete(config: ModelConfig, messages: list[dict]) -> Completion:
    t0 = time.perf_counter()
    try:
        c = _anthropic(config, messages) if config.kind == "anthropic" else _openai(config, messages)
    except Exception as e:  # noqa: BLE001 - surface any failure as a per-call error
        c = Completion("", error=f"{type(e).__name__}: {e}")
    c.latency_s = time.perf_counter() - t0
    return c


def _openai(config: ModelConfig, messages: list[dict]) -> Completion:
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = config.resolved_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body: dict = {"model": config.model, "messages": messages, "max_tokens": config.max_tokens}
    if config.temperature is not None:
        body["temperature"] = config.temperature

    with httpx.Client(timeout=config.timeout) as http:
        try:
            r = http.post(url, json=body, headers=headers)
        except httpx.ConnectError:
            return Completion("", error=f"could not connect to {config.base_url}: is the model server running?")
        except httpx.TimeoutException:
            return Completion("", error=f"no response from {config.base_url} within {config.timeout:.0f}s")
        # Some hosted reasoning models want max_completion_tokens and only the default temperature.
        if r.status_code == 400 and "max_completion_tokens" in r.text:
            body["max_completion_tokens"] = body.pop("max_tokens")
            r = http.post(url, json=body, headers=headers)
        if r.status_code == 400 and "temperature" in r.text and "temperature" in body:
            body.pop("temperature")
            r = http.post(url, json=body, headers=headers)
    if r.status_code != 200:
        return Completion("", error=f"HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    return Completion(
        text=(choice.get("message") or {}).get("content") or "",
        stop_reason=choice.get("finish_reason"),
        usage=data.get("usage") or {},
    )


def _anthropic(config: ModelConfig, messages: list[dict]) -> Completion:
    import anthropic

    key = config.resolved_key()
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    kwargs = dict(model=config.model, max_tokens=config.max_tokens, messages=turns, timeout=config.timeout)
    if system:
        kwargs["system"] = system
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
    return Completion(text=text, stop_reason=response.stop_reason, usage=usage)
