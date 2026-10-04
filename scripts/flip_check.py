"""Check the y-flip: the dataset ink, rendered flipped, should match the bench reference images.

Writes out/flip_check.png (觀: reference | raw y | flipped) and prints, over all
88 bench rows, how often the flipped render correlates better with the reference.

    uv run python scripts/flip_check.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from calligraphy.data.bench import load_bench
from calligraphy.render.raster import render_polygons

OUT = Path(__file__).resolve().parents[1] / "out"


def ink(img: Image.Image) -> np.ndarray:
    """Darkness in [0, 1] (1 = ink)."""
    return 1 - np.asarray(img.convert("L"), dtype=np.float64) / 255


def crop_to_ink(a: np.ndarray, threshold: float = 0.35, n: int = 32) -> np.ndarray:
    """Crop to the ink bounding box and resize to n x n, removing framing differences."""
    ys, xs = np.nonzero(a > threshold)
    if len(xs) == 0:
        return np.zeros((n, n))
    c = Image.fromarray((a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] * 255).astype(np.uint8))
    return np.asarray(c.resize((n, n), Image.Resampling.BILINEAR), dtype=np.float64) / 255


def flip_scores(entry, size: int = 128) -> tuple[float, float]:
    """Correlation with the reference image for the flipped render and for raw y.

    The masters' images are framed differently from the font box, so both sides
    are cropped to their ink first; what remains is mostly orientation.
    """
    ref = crop_to_ink(ink(Image.open(io.BytesIO(entry.reference_jpeg)).resize((size, size))))
    flipped = ink(render_polygons(entry.outlines, size, grid=False))
    raw = flipped[::-1]  # raw y (y-up drawn as y-down) is the vertical mirror, offset by 124
    corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return corr(ref, crop_to_ink(flipped)), corr(ref, crop_to_ink(raw))


def side_by_side(entry, size: int = 256) -> Image.Image:
    ref = Image.open(io.BytesIO(entry.reference_jpeg)).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    flipped = render_polygons(entry.outlines, size)
    raw = flipped.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    sheet = Image.new("RGB", (3 * size + 40, size + 28), "white")
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate([("bench reference", ref), ("raw y (wrong)", raw), ("flipped (canvas)", flipped)]):
        sheet.paste(im, (10 + i * (size + 10), 24))
        d.text((10 + i * (size + 10), 6), label, fill="black")
    return sheet


def main() -> None:
    bench = load_bench()
    wins = 0
    for e in bench:
        f, r = flip_scores(e)
        wins += f > r
    print(f"flipped beats raw on {wins}/{len(bench)} bench rows")
    OUT.mkdir(exist_ok=True)
    guan = next(e for e in bench if e.task_id == "wxz_6420")
    side_by_side(guan).save(OUT / "flip_check.png")
    print(f"wrote {OUT / 'flip_check.png'}")


if __name__ == "__main__":
    main()
