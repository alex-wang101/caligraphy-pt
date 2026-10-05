"""Summarize an agent run against a ceiling and two floors scored the same way.

    uv run python scripts/run_report.py <run_id>
    uv run python scripts/run_report.py <run_id> --sheet 40

The model's numbers mean little alone, so every character is also scored with:
- ceiling: the converter's answer (the SFT label), about 0.9;
- wrong character: the converter's answer for another character in the run with the
  closest stroke count, i.e. a fluent drawing of the wrong thing;
- random strokes: as many straight strokes as the target has, at random positions.

For a tools-mode run it also reports how the tools were used (turns, canvas images
asked for, deletes, rejected calls, how episodes ended) and whether deleting and
redrawing helped: the score of the draft just before the first delete against
the final score, and how good the deleted strokes were.

Writes out/runs/<run_id>/report.json and sheet.png (reference | model, per character).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calligraphy.agent.runner import RunManager, dataset_items
from calligraphy.canvas import CanvasConfig
from calligraphy.convert import convert
from calligraphy.render.raster import render_polygons
from calligraphy.score.reward import compute_reward
from calligraphy.tools import ParseError, format_strokes, parse_response, render_response

CJK_FONTS = ["/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]


def random_strokes(n: int, width: float, rng: np.random.Generator) -> list[dict]:
    out = []
    for _ in range(n):
        a, b = rng.uniform(120, 904, 2), rng.uniform(120, 904, 2)
        out.append({"points": [[*map(round, a), 0.8], [*map(round, b), 0.8]], "width": width})
    return out


def floors(entries: list, canvas: CanvasConfig) -> list[dict]:
    """Per character: ceiling, wrong-character and random-stroke rewards."""
    labels = [convert(e, action_frame=canvas.action_frame) for e in entries]
    rng = np.random.default_rng(0)
    rows = []
    for i, e in enumerate(entries):
        others = [j for j, o in enumerate(entries) if o.character != e.character]
        j = min(others, key=lambda j: (abs(entries[j].n_strokes - e.n_strokes), j)) if others else i
        width = labels[i][0]["width"] if labels[i] else 60
        rows.append(
            {
                "ceiling": compute_reward(format_strokes(labels[i]), e, canvas)[0],
                "wrong_char": compute_reward(format_strokes(labels[j]), e, canvas)[0],
                "wrong_char_used": entries[j].character,
                "random": compute_reward(format_strokes(random_strokes(e.n_strokes, width, rng)), e, canvas)[0],
            }
        )
    return rows


def stats(xs) -> str:
    a = np.asarray(xs, dtype=float)
    return f"mean {a.mean():.3f}  median {np.median(a):.3f}  p90 {np.quantile(a, 0.9):.3f}  max {a.max():.3f}" if len(a) else "–"


def model_metrics(rows: list[dict]) -> dict:
    parsed = [r for r in rows if r["info"].get("parse_ok")]
    info = [r["info"] for r in parsed]
    diff = np.array([i["n_agent"] - i["n_target"] for i in info]) if info else np.array([])
    good = sum(sum(s >= 0.5 for s in i["structure_strokes"]) for i in info)
    total = sum(max(i["n_agent"], i["n_target"]) for i in info)
    return {
        "n": len(rows),
        "parse_rate": len(parsed) / len(rows) if rows else 0.0,
        "reward_mean": float(np.mean([r["reward"] for r in rows])) if rows else 0.0,
        "reward_mean_parsed": float(np.mean([i["structure"] for i in info])) if info else 0.0,
        "shape_mean_parsed": float(np.mean([i["shape"] for i in info])) if info else 0.0,
        "stroke_count_exact": float(np.mean(diff == 0)) if len(diff) else 0.0,
        "stroke_count_within_1": float(np.mean(np.abs(diff) <= 1)) if len(diff) else 0.0,
        "stroke_count_mean_diff": float(diff.mean()) if len(diff) else 0.0,
        "strokes_iou_ge_0_5": good / total if total else 0.0,
        "placement_ok": float(np.mean([i["placement_ok"] for i in info])) if info else 0.0,
        "placement_ratio_median": float(np.median([i["placement_ratio"] for i in info])) if info else 0.0,
    }


def tool_metrics(rows: list[dict], traces: dict[int, dict]) -> dict:
    """How the tools were used in a tools-mode run, and whether editing helped."""
    ts = [r["trace_summary"] for r in rows if r.get("trace_summary")]
    if not ts:
        return {}
    n = len(ts)
    calls = sum(t["n_tool_calls"] for t in ts)
    edited = [t for t in ts if t["n_deletes"] and t["draft_score"] is not None and t["final_score"] is not None]
    deleted = [
        e["deleted_stroke_score"]
        for trace in traces.values()
        for e in trace["events"]
        if e["type"] == "tool" and e["ok"] and e["name"] == "delete_stroke" and e.get("deleted_stroke_score") is not None
    ]
    return {
        "n": n,
        "end_reasons": dict(Counter(t["end_reason"] for t in ts)),
        "turns_mean": float(np.mean([t["n_turns"] for t in ts])),
        "tool_calls_mean": calls / n,
        "rejected_call_rate": sum(t["n_failed_calls"] for t in ts) / calls if calls else 0.0,
        "captures_mean": float(np.mean([t["n_captures"] for t in ts])),
        "looked": float(np.mean([t["n_captures"] > 0 for t in ts])),
        "checked_final": float(np.mean([t["checked_final"] for t in ts])),
        "deletes_mean": float(np.mean([t["n_deletes"] for t in ts])),
        "edited": float(np.mean([t["n_deletes"] > 0 for t in ts])),
        "redraw_rate": sum(t["n_redraws"] for t in ts) / max(1, sum(t["n_deletes"] for t in ts)),
        "peak_minus_final": float(np.mean([t["peak_score"] - t["final_score"] for t in ts if t["final_score"] is not None] or [0.0])),
        # Self-correction, over the episodes that deleted something.
        "n_edited": len(edited),
        "draft_mean": float(np.mean([t["draft_score"] for t in edited])) if edited else None,
        "final_mean": float(np.mean([t["final_score"] for t in edited])) if edited else None,
        "improved": float(np.mean([t["final_score"] > t["draft_score"] for t in edited])) if edited else None,
        "n_deleted_strokes": len(deleted),
        "deleted_stroke_score_mean": float(np.mean(deleted)) if deleted else None,
        "deleted_bad_share": float(np.mean([d < 0.3 for d in deleted])) if deleted else None,
    }


def print_tool_metrics(t: dict) -> None:
    print(f"tool use ({t['n']} episodes):")
    print(f"  ended by              {', '.join(f'{k} {v}' for k, v in sorted(t['end_reasons'].items(), key=lambda kv: -kv[1]))}")
    print(f"  per episode           {t['turns_mean']:.1f} turns, {t['tool_calls_mean']:.1f} tool calls ({t['rejected_call_rate']:.1%} rejected)")
    print(f"  canvas images         {t['captures_mean']:.1f} per episode; asked for one in {t['looked']:.0%}; looked after the last change in {t['checked_final']:.0%}")
    print(f"  deletes               {t['deletes_mean']:.1f} per episode; deleted something in {t['edited']:.0%}; {t['redraw_rate']:.0%} of deleted strokes redrawn")
    print(f"  best minus final      {t['peak_minus_final']:.3f}  (above 0: the canvas scored higher at some earlier point)")
    if t["n_edited"]:
        print(f"self-correction ({t['n_edited']} episodes with a delete):")
        print(f"  draft -> final        {t['draft_mean']:.3f} -> {t['final_mean']:.3f}; final beats draft in {t['improved']:.0%}")
    if t["n_deleted_strokes"]:
        print(f"  deleted strokes       {t['n_deleted_strokes']}; mean score before deletion {t['deleted_stroke_score_mean']:.2f}; {t['deleted_bad_share']:.0%} scored under 0.3")


def font(size: int):
    for path in CJK_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_sheet(rows: list[dict], entries: dict, canvas: CanvasConfig, n: int, cell: int = 180, cols: int = 4) -> Image.Image:
    rows = rows[:n]
    label_h = 26
    pair_w = 2 * cell + 8
    nrows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * pair_w + (cols - 1) * 16, nrows * (cell + label_h + 12)), "white")
    draw, f = ImageDraw.Draw(sheet), font(16)
    for k, r in enumerate(rows):
        x, y = (k % cols) * (pair_w + 16), (k // cols) * (cell + label_h + 12)
        e = entries[r["index"]]
        sheet.paste(render_polygons(e.outlines, cell).convert("RGB"), (x, y + label_h))
        try:
            model = render_response(parse_response(r["response"], canvas), canvas).render(cell).convert("RGB")
        except ParseError:
            model = Image.new("RGB", (cell, cell), (245, 238, 238))
            ImageDraw.Draw(model).text((10, cell // 2 - 8), "no parse", fill=(150, 60, 50), font=f)
        sheet.paste(model, (x + cell + 8, y + label_h))
        i = r["info"]
        strokes = f"{i.get('n_agent', 0)}/{i.get('n_target', e.n_strokes)} strokes"
        draw.text((x, y + 4), f"{e.character}  reward {r['reward']:.2f}  {strokes}", fill=(30, 30, 30), font=f)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--sheet", type=int, default=24, help="characters in the contact sheet")
    args = ap.parse_args()

    runs = RunManager()
    s = runs.summary(args.run_id)
    rows = runs.rollouts(args.run_id)
    canvas = CanvasConfig(action_frame=s["action_frame"])
    items = dataset_items(s["dataset"], s.get("characters", ""), s["n_total"])
    entries = {i: e for i, (_, e) in enumerate(items)}
    answered = [entries[r["index"]] for r in rows]

    m = model_metrics(rows)
    ref = floors(answered, canvas)
    traces = {}
    for r in rows:
        if r.get("trace"):
            try:
                traces[r["index"]] = runs.trace(args.run_id, r["index"])
            except KeyError:
                pass
    tools = tool_metrics(rows, traces)
    mode = s.get("mode") or "single"
    print(f"{s['model']} on {s['dataset_label']} ({mode} mode): {len(rows)}/{s['n_total']} answered, status {s['status']}")
    print(f"  parse rate            {m['parse_rate']:.0%}")
    print(f"  reward (all)          {stats([r['reward'] for r in rows])}")
    print(f"  reward (parsed)       {m['reward_mean_parsed']:.3f}   shape level {m['shape_mean_parsed']:.3f}")
    print(f"  stroke count          exact {m['stroke_count_exact']:.0%}  within ±1 {m['stroke_count_within_1']:.0%}  mean diff {m['stroke_count_mean_diff']:+.1f}")
    print(f"  strokes with IoU≥0.5  {m['strokes_iou_ge_0_5']:.1%}")
    print(f"  placement ok          {m['placement_ok']:.0%}  (median size ratio {m['placement_ratio_median']:.2f})")
    if tools:
        print_tool_metrics(tools)
    print("references on the same characters:")
    for k, label in (("ceiling", "ceiling (converter)"), ("wrong_char", "wrong character"), ("random", "random strokes")):
        print(f"  {label:22}{stats([x[k] for x in ref])}")
    beats = np.mean([r["reward"] > x["wrong_char"] for r, x in zip(rows, ref)]) if rows else 0.0
    print(f"  model beats wrong-character floor on {beats:.0%} of characters")
    best = sorted(rows, key=lambda r: -r["reward"])[:5]
    print("  best: " + ", ".join(f"{r['character']} {r['reward']:.3f}" for r in best))

    run_dir = runs.root / args.run_id
    report = {"model": m, "tools": tools or None, "references": ref, "beats_wrong_char": beats}
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    ordered = sorted(rows, key=lambda r: r["index"])
    contact_sheet(ordered, entries, canvas, args.sheet).save(run_dir / "sheet.png")
    print(f"wrote {run_dir / 'report.json'} and {run_dir / 'sheet.png'}")


if __name__ == "__main__":
    main()
