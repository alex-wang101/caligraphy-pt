"""Viewer API over the calligraphy core.

    uv run uvicorn server.app:app --port 8000

Characters are addressed by bench task_id (e.g. wxz_6420) or by the character
itself (any of the ~9.5k Make Me a Hanzi characters). A "drawing" is the
converter output (default) or one of the scorer's golden scenes.
"""

from __future__ import annotations

import io
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from calligraphy.canvas import Canvas, CanvasConfig, StrokeError
from calligraphy.data.bench import bench_by_task, load_bench
from calligraphy.data.mmah import get_entry, load_dictionary, load_graphics
from calligraphy.data.types import CharacterEntry
from calligraphy.render.brushes import get_brush
from calligraphy.render.raster import render_geometries
from calligraphy.score.levels import score_character
from calligraphy.score.scenes import SCENE_LABELS, SCENES, count_out_of_bounds, through_environment
from calligraphy.score.stroke import stroke_metrics

app = FastAPI(title="Calligraphy viewer")
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
DRAWINGS = ["converter", *[k for k in SCENES if k != "perfect"]]
DRAWING_LABELS = {"converter": SCENE_LABELS["perfect"], **SCENE_LABELS}


def resolve(key: str) -> CharacterEntry:
    bench = bench_by_task()
    if key in bench:
        return bench[key]
    try:
        return get_entry(key)
    except KeyError:
        raise HTTPException(404, f"unknown task or character {key!r}") from None


def raw_drawing(key: str, name: str) -> list[tuple[np.ndarray, float]]:
    entry = resolve(key)
    if name == "converter":
        return SCENES["perfect"](entry)
    if name not in SCENES:
        raise HTTPException(400, f"unknown drawing {name!r}; choose from {DRAWINGS}")
    return SCENES[name](entry)


@lru_cache(maxsize=256)
def drawing(key: str, name: str) -> list[tuple[np.ndarray, float]]:
    """The drawing as the environment scores it (parsed, validated, clamped to the canvas)."""
    return through_environment(raw_drawing(key, name))


def png(img) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/tasks")
def tasks() -> list[dict]:
    return [{"task_id": e.task_id, "character": e.character, **e.meta} for e in load_bench()]


@app.get("/api/search")
def search(q: str = Query(..., min_length=1)) -> list[dict]:
    """Make Me a Hanzi lookup: every known character in q, in order."""
    graphics, dictionary = load_graphics(), load_dictionary()
    seen, out = set(), []
    for ch in q:
        if ch in graphics and ch not in seen:
            seen.add(ch)
            info = dictionary.get(ch, {})
            out.append({"character": ch, "pinyin": info.get("pinyin", []), "definition": info.get("definition", "")})
    return out[:50]


@app.get("/api/char/{key}")
def char(key: str, drawing_name: str = Query("converter", alias="drawing"), brush: str = "disc") -> dict:
    entry = resolve(key)
    strokes = drawing(key, drawing_name)
    t0 = time.perf_counter()
    levels = score_character(strokes, entry, brush=brush)
    per_stroke = [
        stroke_metrics(p, w, o, m, brush=brush).to_dict()
        for (p, w), o, m in zip(strokes, entry.outlines, entry.medians)
    ]
    score_ms = 1000 * (time.perf_counter() - t0)
    return {
        "key": key,
        "character": entry.character,
        "task_id": entry.task_id,
        "meta": entry.meta,
        "n_strokes": entry.n_strokes,
        "has_reference": entry.reference_jpeg is not None,
        "outlines": entry.strokes_d,
        "medians": [m.tolist() for m in entry.medians],
        "drawing": drawing_name,
        "drawings": [{"name": d, "label": DRAWING_LABELS[d]} for d in DRAWINGS],
        "clamped_points": count_out_of_bounds(raw_drawing(key, drawing_name)),
        "strokes": [{"points": p.round(2).tolist(), "width": round(w, 2)} for p, w in strokes],
        "stroke_metrics": per_stroke,
        "levels": levels.to_dict(),
        "score_ms": round(score_ms, 1),
    }


@app.get("/api/char/{key}/reference.jpg")
def reference(key: str) -> Response:
    entry = resolve(key)
    if entry.reference_jpeg is None:
        raise HTTPException(404, "no reference image (Make Me a Hanzi entries have none)")
    return Response(entry.reference_jpeg, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@lru_cache(maxsize=2048)
def _render(key: str, drawing_name: str, upto: int, size: int, brush: str, grid: bool) -> tuple[bytes, float]:
    b = get_brush(brush)
    t0 = time.perf_counter()
    img = render_geometries([b.geometry(p, w) for p, w in drawing(key, drawing_name)[:upto]], size, grid=grid)
    ms = 1000 * (time.perf_counter() - t0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), ms


@app.get("/api/char/{key}/render.png")
def render(
    key: str,
    upto: int = Query(10_000, ge=0),
    size: int = Query(512, ge=32, le=2048),
    brush: str = "disc",
    grid: bool = True,
    drawing_name: str = Query("converter", alias="drawing"),
) -> Response:
    try:
        data, ms = _render(key, drawing_name, upto, size, brush, grid)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return Response(data, media_type="image/png", headers={"Cache-Control": "max-age=3600", "X-Render-Ms": f"{ms:.2f}"})


class RenderRequest(BaseModel):
    strokes: list[dict]
    action_frame: int = 1024
    size: int = 512
    brush: str = "disc"
    grid: bool = True
    upto: int | None = None


@app.post("/api/render")
def render_calls(req: RenderRequest) -> Response:
    """Render arbitrary phase-1 strokes (hook for the rollout viewer)."""
    canvas = Canvas(CanvasConfig(action_frame=req.action_frame, brush=req.brush, grid=req.grid))
    try:
        for s in req.strokes:
            canvas.create_stroke(s.get("points"), s.get("width"))
    except StrokeError as e:
        raise HTTPException(400, str(e)) from None
    return png(canvas.render(req.size, upto=req.upto))


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")
