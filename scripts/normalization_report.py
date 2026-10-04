"""How the environment treats characters that are too small, too big, off-centre, squashed or rotated.

Every scenario goes through the full reward path a model response takes
(JSON -> parser -> Canvas clamping -> scorer). Prints one table per character
and writes out/normalization_<char>.png (one tile per scenario).

    uv run python scripts/normalization_report.py [chars]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from calligraphy.data.mmah import get_entry
from calligraphy.score.reward import compute_reward
from calligraphy.score.scenes import NORMALIZATION_CHARS, NORMALIZATION_ORDER, SCENE_LABELS, SCENES, count_out_of_bounds, to_response
from calligraphy.tools import parse_response, render_response

OUT = Path(__file__).resolve().parents[1] / "out"


def run(char: str) -> list[dict]:
    entry = get_entry(char)
    rows = []
    for name in NORMALIZATION_ORDER:
        drawing = SCENES[name](entry)
        text = to_response(drawing)
        reward, info = compute_reward(text, entry)
        rows.append(
            {
                "name": name,
                "label": SCENE_LABELS[name],
                "reward": reward,
                "shape": info["shape"],
                "offset": info["placement_offset"],
                "ratio": info["placement_ratio"],
                "placement_ok": info["placement_ok"],
                "fit_scale": info["align_scale"],
                "clamped": count_out_of_bounds(drawing),
                "canvas": render_response(parse_response(text)),
            }
        )
    return rows


def print_table(char: str, rows: list[dict]) -> None:
    print(f"\n{char}")
    print(f"{'scenario':22} {'reward':>7} {'shape':>6} {'offset':>7} {'size':>6} {'placement':>10} {'fit s':>6} {'clamped pts':>12}")
    for r in rows:
        flag = "ok" if r["placement_ok"] else "FLAGGED"
        print(
            f"{r['name']:22} {r['reward']:7.3f} {r['shape']:6.3f} {r['offset']:7.0f} {r['ratio']:5.2f}x "
            f"{flag:>10} {r['fit_scale']:6.2f} {r['clamped']:12d}"
        )


def contact_sheet(char: str, rows: list[dict], tile: int = 240) -> Image.Image:
    font = ImageFont.load_default(size=13)
    small = ImageFont.load_default(size=11)
    cols = 4
    cap = 58
    sheet = Image.new("RGB", (cols * (tile + 12) + 12, ((len(rows) + cols - 1) // cols) * (tile + cap + 12) + 12), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        x = 12 + (i % cols) * (tile + 12)
        y = 12 + (i // cols) * (tile + cap + 12)
        sheet.paste(r["canvas"].render(tile), (x, y))
        d.text((x, y + tile + 4), r["label"], fill="black", font=small)
        colour = (20, 120, 60) if r["reward"] > 0.8 else (190, 120, 20) if r["reward"] > 0.5 else (180, 40, 30)
        d.text((x, y + tile + 20), f"reward {r['reward']:.2f}   shape {r['shape']:.2f}", fill=colour, font=font)
        place = f"offset {r['offset']:.0f}  size {r['ratio']:.2f}x  {'ok' if r['placement_ok'] else 'FLAGGED'}"
        if r["clamped"]:
            place += f"  clamped {r['clamped']}"
        d.text((x, y + tile + 38), place, fill=(90, 90, 90), font=small)
    return sheet


def main() -> None:
    chars = sys.argv[1] if len(sys.argv) > 1 else NORMALIZATION_CHARS
    OUT.mkdir(exist_ok=True)
    for char in chars:
        rows = run(char)
        print_table(char, rows)
        path = OUT / f"normalization_{char}.png"
        contact_sheet(char, rows).save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
