"""Converter IoU over the full Make Me a Hanzi dataset: naive vs fixed variants.

    uv run python scripts/convert_report.py [--limit N] [--workers N]

Writes out/convert_report.json and prints a markdown table. IoU here measures
label + brush quality (disc strokes vs the font's filled outlines), not the agent.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from calligraphy.convert import DEFAULT, NAIVE, variant
from calligraphy.data.mmah import get_entry, load_graphics
from calligraphy.roundtrip import roundtrip

OUT = Path(__file__).resolve().parents[1] / "out"

VARIANTS = {
    "naive (v2 validation: 40 pts)": NAIVE,
    "fixed spacing": variant(recenter=False, extend_ends=False, max_points=None, quantize=False),
    "+ end extension": variant(recenter=False, max_points=None, quantize=False),
    "+ centerline (dense)": variant(max_points=None, quantize=False),
    "default (<=16 pts, quantized)": DEFAULT,
    "high fidelity (rdp_eps 0.5)": variant(rdp_eps=0.5),
    "default, radius x1.05": variant(radius_scale=1.05),
    "default, radius x1.1": variant(radius_scale=1.1),
}


def score_char(char: str) -> dict:
    entry = get_entry(char)
    out = {}
    for name, cfg in VARIANTS.items():
        r = roundtrip(entry, cfg)
        out[name] = {
            "stroke_iou": r["stroke_iou"],
            "char_iou": r["char_iou"],
            "points": [len(st["points"]) for st in r["strokes"]],
        }
    return {"char": char, "results": out}


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for name in VARIANTS:
        s = np.concatenate([r["results"][name]["stroke_iou"] for r in rows])
        c = np.array([r["results"][name]["char_iou"] for r in rows])
        pts = np.concatenate([r["results"][name]["points"] for r in rows])
        summary[name] = {
            "stroke_mean": float(s.mean()),
            "stroke_median": float(np.median(s)),
            "stroke_p10": float(np.percentile(s, 10)),
            "stroke_min": float(s.min()),
            "char_mean": float(c.mean()),
            "char_p10": float(np.percentile(c, 10)),
            "points_per_stroke": float(pts.mean()),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    chars = list(load_graphics())[: args.limit]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(score_char, chars, chunksize=32))
    elapsed = time.perf_counter() - t0

    summary = summarize(rows)
    naive, default = "naive (v2 validation: 40 pts)", "default (<=16 pts, quantized)"
    char_naive = np.array([r["results"][naive]["char_iou"] for r in rows])
    char_default = np.array([r["results"][default]["char_iou"] for r in rows])
    worst = sorted(rows, key=lambda r: r["results"][default]["char_iou"])[:10]

    print(f"{len(rows)} characters, {sum(len(r['results'][naive]['stroke_iou']) for r in rows)} strokes, {elapsed:.0f}s\n")
    print("| variant | stroke mean | stroke median | stroke p10 | stroke min | char mean | char p10 | pts/stroke |")
    print("|---|---|---|---|---|---|---|---|")
    for name, s in summary.items():
        print(
            f"| {name} | {s['stroke_mean']:.3f} | {s['stroke_median']:.3f} | {s['stroke_p10']:.3f} | "
            f"{s['stroke_min']:.3f} | {s['char_mean']:.3f} | {s['char_p10']:.3f} | {s['points_per_stroke']:.1f} |"
        )
    print(f"\ndefault beats naive (char IoU) on {np.mean(char_default > char_naive):.1%} of characters")
    print("worst 10 (default):", " ".join(f"{r['char']}={r['results'][default]['char_iou']:.2f}" for r in worst))

    OUT.mkdir(exist_ok=True)
    report = {
        "n_chars": len(rows),
        "summary": summary,
        "default_beats_naive_frac": float(np.mean(char_default > char_naive)),
        "worst_default": [(r["char"], r["results"][default]["char_iou"]) for r in worst],
        "per_char": {r["char"]: {k: v["char_iou"] for k, v in r["results"].items()} for r in rows},
    }
    (OUT / "convert_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\nwrote {OUT / 'convert_report.json'}")


if __name__ == "__main__":
    main()
