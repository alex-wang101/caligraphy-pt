"""Run a model on a dataset from the command line (same runner as the viewer's Agent runs).

    uv run python scripts/run_agent.py --list-providers
    uv run python scripts/run_agent.py --provider ollama --model qwen3:8b --dataset bench --limit 10
    uv run python scripts/run_agent.py --provider vllm_tunnel --model Qwen/Qwen3-VL-8B-Instruct --dataset val
    uv run python scripts/run_agent.py --provider openrouter --model qwen/qwen3.8-27b:free --mode tools --dataset viz16
    uv run python scripts/run_agent.py --resume 20261004-230000-qwen-qwen3-8-27b-free

--mode single (default) asks for every stroke in one reply. --mode tools runs a
multi-turn episode with the create_stroke, delete_stroke, capture_canvas and
submit tools, and records every model reply and tool call under traces/.

Keys come from the provider's environment variable (e.g. OPENROUTER_API_KEY), which
may be set in the repo's .env file. Results go to out/runs/<run_id>/ and show up in
the viewer under Agent runs. A run stopped by a rate limit (status rate_limited) or
an interruption continues with --resume: unanswered items and transport errors are
asked again, answers already scored are kept.
"""

from __future__ import annotations

import argparse
import json
import time

from calligraphy.agent.client import ModelConfig, load_env_file
from calligraphy.agent.providers import PROVIDERS
from calligraphy.agent.runner import DATASETS, MODES, TOOLS_OBS_SIZE, RunManager
from calligraphy.episode import EpisodeConfig


def list_providers() -> None:
    for p in PROVIDERS.values():
        key = f"key: ${p.key_env} ({'set' if p.to_dict()['key_env_set'] else 'not set'})" if p.key_env else "no key needed"
        print(f"\n{p.id}: {p.label}  [{key}]")
        if p.base_url:
            print(f"  base URL: {p.base_url}")
        for step in p.steps:
            print(f"  {step}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-providers", action="store_true", help="show presets and setup steps, then exit")
    ap.add_argument("--provider", choices=sorted(PROVIDERS))
    ap.add_argument("--model")
    ap.add_argument("--base-url", help="override the preset's base URL")
    ap.add_argument("--dataset", default="bench", choices=sorted(DATASETS))
    ap.add_argument("--characters", default="", help="for --dataset custom")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--extra-body", type=json.loads, default={}, help='provider options as JSON, e.g. \'{"reasoning": {"enabled": false}}\'')
    ap.add_argument("--mode", default="single", choices=sorted(MODES), help="single: one reply; tools: multi-turn with tools and traces")
    ap.add_argument("--max-turns", type=int, default=EpisodeConfig.max_turns, help="tools mode: model replies per character")
    ap.add_argument("--max-tool-calls", type=int, default=EpisodeConfig.max_tool_calls, help="tools mode: tool calls per character")
    ap.add_argument("--obs-size", type=int, default=TOOLS_OBS_SIZE, help="tools mode: side of the canvas image in pixels")
    ap.add_argument("--resume", metavar="RUN_ID", help="continue a stopped run with its saved settings")
    args = ap.parse_args()
    load_env_file()

    if args.list_providers:
        return list_providers()
    runs = RunManager()
    if args.resume:
        s = runs.summary(args.resume)
        preset = PROVIDERS[s["provider"]]
        config = ModelConfig(
            provider=preset.id,
            kind=s["kind"],
            base_url=s["base_url"],
            model=s["model"],
            key_env=preset.key_env,
            temperature=s["temperature"],
            max_tokens=s["max_tokens"],
            extra_body=s.get("extra_body") or {},
        )
        run_id = args.resume
        todo = runs.resume(run_id, config)
        print(f"resuming {run_id}: {len(todo)} of {s['n_total']} items to ask (key: {config.key_source()})")
    else:
        if not args.provider or not args.model:
            ap.error("--provider and --model are required (see --list-providers)")
        preset = PROVIDERS[args.provider]
        config = ModelConfig(
            provider=preset.id,
            kind=preset.kind,
            base_url=args.base_url or preset.base_url,
            model=args.model,
            key_env=preset.key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=args.extra_body,
        )
        episode = EpisodeConfig(max_turns=args.max_turns, max_tool_calls=args.max_tool_calls)
        run_id = runs.start(
            config, args.dataset, args.characters, args.limit, args.concurrency,
            mode=args.mode, episode=episode, obs_size=args.obs_size if args.mode == "tools" else None,
        )  # fmt: skip
        print(f"run {run_id}: {args.model} on {DATASETS[args.dataset]}, mode {args.mode} (key: {config.key_source()})")
    while (s := runs.summary(run_id))["status"] == "running":
        mean = "–" if s["mean_reward"] is None else f"{s['mean_reward']:.3f}"
        print(f"\r  {s['n_done']}/{s['n_total']}  reward {mean}  errors {s['n_errors']}", end="", flush=True)
        time.sleep(1)
    print()
    for r in runs.rollouts(run_id):
        note = r["error"] or ("parsed" if r["info"].get("parse_ok") else r["info"].get("error"))
        if t := r.get("trace_summary"):
            note = (
                f"{t['end_reason']}: {t['n_turns']} turns, {t['n_tool_calls']} tool calls "
                f"({t['n_captures']} captures, {t['n_deletes']} deletes, {t['n_failed_calls']} rejected), {t['n_strokes']} strokes"
                + (f"; {r['error']}" if r["error"] else "")
            )
        print(f"  {r['character']}  {r['reward']:.3f}  {note}")
    mean = "–" if s["mean_reward"] is None else f"{s['mean_reward']:.3f}"
    print(f"{s['status']}: mean reward {mean}, parsed {s['n_parse_ok']}/{s['n_done']} of {s['n_total']}, errors {s['n_errors']}, {s['elapsed_s']}s")
    if s["status"] in ("rate_limited", "interrupted", "cancelled"):
        print(f"continue with: uv run python scripts/run_agent.py --resume {run_id}")


if __name__ == "__main__":
    main()
