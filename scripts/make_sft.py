"""Export the phase-1 SFT dataset as chat JSONL, one character per line.

    uv run python scripts/make_sft.py                      # train, val, viz16 -> out/sft/
    uv run python scripts/make_sft.py --splits train --limit 100
    uv run python scripts/make_sft.py --action-frame 256 --rdp-eps 0.5

Each line is {"character", "split", "n_strokes", "messages": [system, user, assistant]}.
The prompt is exactly what the agent harness sends at evaluation time
(`phase1_messages`: the character's name and pinyin, no image), and the
assistant turn is the converter's answer in the phase-1 JSON format, so the
label scores about 0.9 under `compute_reward`. The system prompt and the
answer frame must match at training and evaluation time; the frame is baked
into the labels (docs/IMPLEMENTATION.md 3.2).

The file is read by notebooks/sft_unsloth_together.ipynb (Unsloth on Colab, or
the Together AI fine-tuning API, which takes this format as is).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calligraphy.canvas import CanvasConfig
from calligraphy.convert import DEFAULT, convert, variant
from calligraphy.data.mmah import get_entry
from calligraphy.data.types import CharacterEntry
from calligraphy.score.reward import compute_reward
from calligraphy.splits import NAMES, load_splits
from calligraphy.tools import format_strokes, phase1_messages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "sft"


def pinyin(entry: CharacterEntry) -> str | None:
    p = entry.meta.get("pinyin")
    return ", ".join(p) if isinstance(p, list) and p else (p or None)


def sft_example(entry: CharacterEntry, canvas: CanvasConfig, cfg=DEFAULT, split: str = "") -> dict:
    """One training row: the evaluation prompt plus the converter's answer."""
    strokes = convert(entry, cfg, action_frame=canvas.action_frame)
    answer = format_strokes(strokes)
    messages = phase1_messages(entry.character, pinyin(entry), canvas) + [{"role": "assistant", "content": answer}]
    return {"character": entry.character, "split": split, "n_strokes": entry.n_strokes, "messages": messages}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "viz16"], choices=NAMES)
    ap.add_argument("--limit", type=int, help="first N characters of each split")
    ap.add_argument("--action-frame", type=int, default=CanvasConfig.action_frame, help="coordinate frame of the labels (1024 keeps the 0.9 ceiling; 64 costs 0.14)")
    ap.add_argument("--rdp-eps", type=float, default=DEFAULT.rdp_eps, help="converter point budget: 2.0 -> ~10 points/stroke, 0.5 -> ~14")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", type=int, default=20, help="score the first N labels of each split with the reward (0 to skip)")
    args = ap.parse_args()

    canvas = CanvasConfig(action_frame=args.action_frame)
    cfg = variant(rdp_eps=args.rdp_eps)
    splits = load_splits()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"action_frame": args.action_frame, "rdp_eps": args.rdp_eps, "files": {}}

    for name in args.splits:
        chars = getattr(splits, name)[: args.limit]
        path = args.out / f"{name}.jsonl"
        n_chars = n_points = 0
        rewards = []
        with path.open("w", encoding="utf-8") as f:
            for i, c in enumerate(chars):
                entry = get_entry(c)
                row = sft_example(entry, canvas, cfg, split=name)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_chars += len(row["messages"][-1]["content"])
                n_points += sum(len(s["points"]) for s in json.loads(row["messages"][-1]["content"])["strokes"])
                if i < args.check:
                    rewards.append(compute_reward(row["messages"][-1]["content"], entry, canvas)[0])
        stats = {
            "rows": len(chars),
            "answer_chars_mean": round(n_chars / max(len(chars), 1)),
            "points_mean": round(n_points / max(len(chars), 1), 1),
            "label_reward_mean": round(sum(rewards) / len(rewards), 3) if rewards else None,
        }
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        manifest["files"][name] = {"path": str(shown), **stats}
        print(f"{name:20s} {stats['rows']:6d} rows  ~{stats['answer_chars_mean']} chars/answer  {stats['points_mean']} points  label reward {stats['label_reward_mean']}  -> {shown}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
