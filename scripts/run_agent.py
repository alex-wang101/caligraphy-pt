"""Run a model on the phase-1 task from the command line (same runner as the viewer's Agent runs).

    uv run python scripts/run_agent.py --list-providers
    uv run python scripts/run_agent.py --provider ollama --model qwen3:8b --dataset bench --limit 10
    uv run python scripts/run_agent.py --provider vllm_tunnel --model Qwen/Qwen3-VL-8B-Instruct --dataset val

Keys come from the provider's environment variable (e.g. OPENAI_API_KEY). Results
go to out/runs/<run_id>/ and show up in the viewer under Agent runs.
"""

from __future__ import annotations

import argparse
import time

from calligraphy.agent.client import ModelConfig
from calligraphy.agent.providers import PROVIDERS
from calligraphy.agent.runner import DATASETS, RunManager


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
    args = ap.parse_args()

    if args.list_providers:
        return list_providers()
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
    )
    runs = RunManager()
    run_id = runs.start(config, args.dataset, args.characters, args.limit, args.concurrency)
    print(f"run {run_id}: {args.model} on {DATASETS[args.dataset]} (key: {config.key_source()})")
    while (s := runs.summary(run_id))["status"] == "running":
        mean = "–" if s["mean_reward"] is None else f"{s['mean_reward']:.3f}"
        print(f"\r  {s['n_done']}/{s['n_total']}  reward {mean}  errors {s['n_errors']}", end="", flush=True)
        time.sleep(1)
    print()
    for r in runs.rollouts(run_id):
        print(f"  {r['character']}  {r['reward']:.3f}  {r['error'] or ('parsed' if r['info'].get('parse_ok') else r['info'].get('error'))}")
    mean = "–" if s["mean_reward"] is None else f"{s['mean_reward']:.3f}"
    print(f"{s['status']}: mean reward {mean}, parsed {s['n_parse_ok']}/{s['n_done']}, errors {s['n_errors']}, {s['elapsed_s']}s")


if __name__ == "__main__":
    main()
