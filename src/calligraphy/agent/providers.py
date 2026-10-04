"""Provider presets with setup instructions (shown in the viewer and by scripts/run_agent.py).

Two backends:
- "openai": any server speaking the OpenAI chat-completions protocol (Ollama,
  LM Studio, llama.cpp, vLLM, OpenAI, OpenRouter, ...), called over HTTP.
- "anthropic": Claude through the official Anthropic SDK.

In `steps`, a line starting with "$ " is a shell command.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    kind: str  # "openai" | "anthropic"
    base_url: str
    key_env: str | None  # environment variable that holds the key; None if no key is needed
    model_hint: str
    local: bool
    steps: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key_env_set"] = bool(self.key_env and os.environ.get(self.key_env))
        return d


PROVIDERS: dict[str, Provider] = {
    p.id: p
    for p in [
        Provider(
            id="stub",
            label="Scripted baseline (no model)",
            kind="openai",
            base_url="http://127.0.0.1:8100/v1",
            key_env=None,
            model_hint="converter",
            local=True,
            steps=(
                "A fake model that answers with the SFT converter's output, for checking the pipeline end to end. "
                "A run should score about 0.9.",
                "Start it in a terminal from the repo root:",
                "$ uv run python scripts/stub_model.py",
            ),
        ),
        Provider(
            id="ollama",
            label="Local: Ollama",
            kind="openai",
            base_url="http://127.0.0.1:11434/v1",
            key_env=None,
            model_hint="qwen3:8b",
            local=True,
            steps=(
                "Ollama is installed on this Mac. Start its server (or open the Ollama app):",
                "$ ollama serve",
                "In another terminal, download a model. Phase 1 is text only, so any chat model works; "
                "a vision model such as a Qwen-VL build will matter once the canvas image is shown (phase 2).",
                "$ ollama pull qwen3:8b",
                "Use the tag you pulled as the model name. An 8B model at 4-bit needs about 6 GB of memory.",
            ),
        ),
        Provider(
            id="local_openai",
            label="Local: LM Studio, llama.cpp or another server",
            kind="openai",
            base_url="http://127.0.0.1:1234/v1",
            key_env=None,
            model_hint="the model id the server reports",
            local=True,
            steps=(
                "LM Studio: load a model, then start the server in the Developer tab (default port 1234).",
                "llama.cpp: serve a GGUF file and set the base URL to http://127.0.0.1:8080/v1",
                "$ llama-server -m model.gguf --port 8080",
                "Any server that speaks the OpenAI chat-completions protocol works; set its base URL here.",
            ),
        ),
        Provider(
            id="vllm_tunnel",
            label="Cloud GPU: vLLM over an SSH tunnel",
            kind="openai",
            base_url="http://127.0.0.1:8001/v1",
            key_env="VLLM_API_KEY",
            model_hint="Qwen/Qwen3-VL-8B-Instruct",
            local=False,
            steps=(
                "On the GPU machine, serve the model with vLLM:",
                "$ vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000",
                "On this Mac, forward a local port to it (leave this running):",
                "$ ssh -N -L 8001:localhost:8000 user@gpu-host",
                "The model name is the one vLLM is serving. A key is only needed if you started vLLM with "
                "--api-key; set VLLM_API_KEY or paste it below.",
            ),
        ),
        Provider(
            id="openai",
            label="OpenAI API",
            kind="openai",
            base_url="https://api.openai.com/v1",
            key_env="OPENAI_API_KEY",
            model_hint="a model id from your OpenAI account",
            local=False,
            steps=(
                "Set your key in the terminal that runs the viewer's API server, then restart that server:",
                "$ export OPENAI_API_KEY=...",
                "$ uv run uvicorn server.app:app --port 8000",
                "Or paste the key below: it is kept in memory for this run only and never written to disk.",
            ),
        ),
        Provider(
            id="openrouter",
            label="OpenRouter",
            kind="openai",
            base_url="https://openrouter.ai/api/v1",
            key_env="OPENROUTER_API_KEY",
            model_hint="provider/model slug from openrouter.ai/models",
            local=False,
            steps=(
                "One key reaches many hosted models, including open Qwen models. Set it before starting the API server:",
                "$ export OPENROUTER_API_KEY=...",
                "Or paste it below (kept in memory for this run only).",
            ),
        ),
        Provider(
            id="anthropic",
            label="Anthropic (Claude)",
            kind="anthropic",
            base_url="",
            key_env="ANTHROPIC_API_KEY",
            model_hint="claude-opus-5-5",
            local=False,
            steps=(
                "Uses the official Anthropic SDK. It finds credentials on its own: ANTHROPIC_API_KEY, "
                "or a profile from `ant auth login`. Set one in the terminal that runs the API server, then restart it:",
                "$ export ANTHROPIC_API_KEY=...",
                "Or paste a key below (kept in memory for this run only).",
                "Current models have no temperature setting, so that field is ignored. Refusal fallback is on "
                "for the models that support it.",
            ),
        ),
    ]
}
