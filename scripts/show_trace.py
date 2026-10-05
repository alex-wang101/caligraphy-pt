"""Print one tools-mode episode step by step, and lay the canvas images the model saw side by side.

    uv run python scripts/show_trace.py <run_id> 3        # by rollout index
    uv run python scripts/show_trace.py <run_id> 永        # by character
    uv run python scripts/show_trace.py <run_id> 永 --full # model text and reasoning uncut

Each line is one model reply or one tool call, in order. `score` is the structure
score of the canvas after that change (what it would get if submitted then); the
model never sees it. Writes traces/<index>/filmstrip.png: the reference, each
image the model was shown, and the final canvas.
"""

from __future__ import annotations

import argparse

from PIL import Image, ImageDraw
from run_report import font

from calligraphy.agent.runner import RunManager, dataset_items
from calligraphy.canvas import CanvasConfig
from calligraphy.render.raster import render_polygons
from calligraphy.tools import render_response


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def describe_call(e: dict) -> str:
    args = e.get("arguments")
    if args is None:
        return clip(e.get("arguments_raw") or "", 60)
    if e["name"] == "create_stroke" and isinstance(args.get("points"), list):
        pts = args["points"]
        try:
            ends = f"({pts[0][0]:g},{pts[0][1]:g})→({pts[-1][0]:g},{pts[-1][1]:g})"
        except (TypeError, IndexError, ValueError):
            ends = "?"
        return f"{len(pts)} pts {ends} width {args.get('width')}"
    if e["name"] == "delete_stroke":
        was = e.get("deleted_stroke_score")
        return f"stroke {args.get('stroke_id')}" + (f" (it scored {was:.2f})" if was is not None else "")
    return ""


def print_trace(trace: dict, full: bool) -> None:
    s = trace["summary"]
    print(f"{trace['character']} · {trace['model']} · reward {trace['reward']:.3f} · ended by {s['end_reason']}")
    print(f"{s['n_turns']} turns, {s['n_tool_calls']} tool calls: {s['n_captures']} canvas images, {s['n_deletes']} deletes, {s['n_failed_calls']} rejected; {s['n_strokes']} strokes left")
    if s["draft_score"] is not None:
        print(f"draft before the first delete {s['draft_score']:.3f} → final {s['final_score']:.3f}")
    cut = 100_000 if full else 160
    for e in trace["events"]:
        if e["type"] == "model":
            usage = e["usage"].get("completion_tokens") or e["usage"].get("output_tokens")
            print(f"\nturn {e['turn']} · {e['latency_s']:.1f}s · {e['n_tool_calls']} tool calls · stop {e['stop_reason']}" + (f" · {usage} tokens out" if usage else ""))
            if e["error"]:
                print(f"    error: {e['error']}")
            if e["reasoning"]:
                print(f"    thinking: {clip(e['reasoning'], cut)}")
            if e["text"]:
                print(f"    said: {clip(e['text'], cut)}")
        elif e["type"] == "tool":
            mark = "·" if e["ok"] else "✗"
            score = f"  score {e['score']['structure']:.3f}" if e.get("score") else ""
            image = f"  [{e['image']}]" if e.get("image") else ""
            print(f"  {e['i']:>3} {mark} {e['name']:<15} {describe_call(e):<46} → {clip(e['result'], 70)}{score}{image}")
        elif e["type"] == "nudge":
            print(f"  {e['i']:>3}   harness: {e['text']}")
        else:
            print(f"\nend: {e['reason']}")


def filmstrip(runs: RunManager, run_id: str, trace: dict, entry, cell: int = 240) -> Image.Image:
    cfg = CanvasConfig(action_frame=trace["canvas"]["action_frame"])
    frames = [("reference", render_polygons(entry.outlines, cell).convert("RGB"))]
    for e in trace["events"]:
        if e.get("image"):
            shown = Image.open(runs.trace_image(run_id, trace["index"], e["image"])).convert("RGB").resize((cell, cell))
            frames.append((f"step {e['i']}: shown, {len(e['stroke_ids'])} strokes", shown))
    final = render_response(trace["final_strokes"], cfg).render(cell).convert("RGB") if trace["final_strokes"] else Image.new("RGB", (cell, cell), "white")
    frames.append((f"final · reward {trace['reward']:.2f}", final))
    sheet = Image.new("RGB", (len(frames) * (cell + 10) - 10, cell + 26), "white")
    draw, f = ImageDraw.Draw(sheet), font(15)
    for k, (label, img) in enumerate(frames):
        x = k * (cell + 10)
        draw.text((x, 4), label, fill=(30, 30, 30), font=f)
        sheet.paste(img, (x, 26))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("item", help="rollout index or character")
    ap.add_argument("--full", action="store_true", help="print model text and reasoning uncut")
    args = ap.parse_args()

    runs = RunManager()
    s = runs.summary(args.run_id)
    rows = runs.rollouts(args.run_id)
    row = next((r for r in rows if str(r["index"]) == args.item or r["character"] == args.item), None)
    if row is None:
        ap.error(f"no rollout {args.item!r} in {args.run_id}; it has {', '.join(r['character'] for r in rows) or 'none yet'}")
    try:
        trace = runs.trace(args.run_id, row["index"])
    except KeyError:
        ap.error(f"rollout {row['index']} has no trace (traces are recorded in --mode tools; this run is {s.get('mode') or 'single'})")
    print_trace(trace, args.full)

    entry = dataset_items(s["dataset"], s.get("characters", ""), s["n_total"])[row["index"]][1]
    out = runs.root / args.run_id / "traces" / f"{row['index']:04d}" / "filmstrip.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    filmstrip(runs, args.run_id, trace, entry).save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
