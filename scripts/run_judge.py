"""Ask a model to score drawings of known quality and compare with the reward (the viewer's Judge experiments).

    uv run python scripts/run_judge.py --provider openrouter --model qwen/qwen3.8-27b
    uv run python scripts/run_judge.py --provider openrouter --model qwen/qwen3.8-27b --tasks wxz_6420,smh_1196 --dry-run

Each drawing is shown as capture_canvas would show it (448 px with the grid).
Results go to out/judge/<id>/ and show up in the viewer under "Judge experiments".
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from calligraphy.agent.client import ModelConfig, load_env_file
from calligraphy.agent.judge import DEFAULT_TASKS, JudgeManager, build_cases
from calligraphy.agent.providers import PROVIDERS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", help="override the preset's base URL")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma-separated calligraphy-bench task ids")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = the model's single best judgement")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--extra-body", type=json.loads, default={"reasoning": {"enabled": False}}, help="provider options as JSON")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="list the drawings and stop (no model calls)")
    args = ap.parse_args()
    load_env_file()

    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    if args.dry_run:
        cases = build_cases(tasks)
        print(f"{len(cases)} drawings:", dict(Counter(c["variant"] for c in cases)))
        return
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
    judge = JudgeManager()
    exp_id, thread = judge.start(config, tasks, args.concurrency)
    print(f"experiment {exp_id}: {config.model} via {config.provider} (key: {config.key_source()})")
    while thread.is_alive():
        thread.join(5)
        s = judge.summary(exp_id)
        st = s.get("stats") or {}
        rho = st.get("spearman")
        print(f"  {s['n_done']}/{s['n_total']}  errors {s['n_errors']}  spearman {'–' if rho is None else f'{rho:.2f}'}", flush=True)
    s = judge.summary(exp_id)
    st = s.get("stats") or {}
    print(f"done ({s['status']}) in {s.get('elapsed_s', 0)} s")
    for k in ("n_scored", "spearman", "pearson", "mae", "auc_good_bad", "reads_correct", "accepts_correct", "rejects_wrong"):
        v = st.get(k)
        print(f"  {k:16s} {v:.3f}" if isinstance(v, float) else f"  {k:16s} {v}")
    for row in st.get("by_variant", []):
        print(f"  {row['label'][:36]:36s} n={row['n']:2d}  reward {row['reward_mean']:.2f}  model {row['judge_mean']:.2f}")


if __name__ == "__main__":
    main()
