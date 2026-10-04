"""Renderer throughput on one core: stroke steps/s and full characters/s.

    uv run python scripts/bench_render.py
"""

from __future__ import annotations

import time

import numpy as np

from calligraphy.canvas import Canvas, CanvasConfig
from calligraphy.data.mmah import get_entry, load_graphics
from calligraphy.render.raster import rasterize_mask


def stroke_calls(entry) -> list[dict]:
    """Medians at constant pressure: a stand-in workload with realistic stroke counts."""
    return [{"points": [[x, y, 0.8] for x, y in m[:16]], "width": 60} for m in entry.medians]


def bench(chars: list[str], size: int, brush: str) -> dict:
    calls = [stroke_calls(get_entry(c)) for c in chars]
    n_strokes = sum(len(c) for c in calls)

    t0 = time.perf_counter()
    for cs in calls:
        canvas = Canvas(CanvasConfig(obs_size=size, brush=brush))
        for s in cs:
            canvas.create_stroke(s["points"], s["width"])
    t_steps = time.perf_counter() - t0

    canvas = Canvas(CanvasConfig(obs_size=size, brush=brush))
    for s in calls[0]:
        canvas.create_stroke(s["points"], s["width"])
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        canvas.capture()
    t_capture = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for cs in calls:
        canvas = Canvas(CanvasConfig(obs_size=size, brush=brush))
        for s in cs:
            canvas.create_stroke(s["points"], s["width"])
        canvas.capture()
    t_full = time.perf_counter() - t0

    return {
        "steps_per_s": n_strokes / t_steps,
        "capture_ms": 1000 * t_capture,
        "chars_per_s": len(calls) / t_full,
    }


def main() -> None:
    chars = list(load_graphics())[:300]
    mean_strokes = np.mean([len(get_entry(c).medians) for c in chars])
    print(f"{len(chars)} characters, {mean_strokes:.1f} strokes on average, single core\n")
    print(f"{'brush':9} {'size':>5} {'stroke steps/s':>15} {'capture ms':>11} {'chars/s':>8}")
    for brush in ("disc", "freehand"):
        for size in (224, 512, 1024):
            r = bench(chars, size, brush)
            print(f"{brush:9} {size:>5} {r['steps_per_s']:>15,.0f} {r['capture_ms']:>11.2f} {r['chars_per_s']:>8,.0f}")

    canvases = []
    for c in chars[:100]:
        canvas = Canvas()
        for s in stroke_calls(get_entry(c)):
            canvas.create_stroke(s["points"], s["width"])
        canvases.append(canvas)
    t0 = time.perf_counter()
    for canvas in canvases:
        canvas.ink_mask(512)
    print(f"\nscoring mask (512 px, whole character): {1000 * (time.perf_counter() - t0) / len(canvases):.2f} ms")


if __name__ == "__main__":
    main()
