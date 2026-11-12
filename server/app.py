"""Viewer API over the calligraphy core.

    uv run uvicorn server.app:app --port 8000

Characters are addressed by bench task_id (e.g. wxz_6420) or by the character
itself (any of the ~9.5k Make Me a Hanzi characters). A "drawing" is the
converter output (default), one of the scorer's scenes, or a model's answer
from an agent run ("run:<run_id>:<index>").

Agent runs call the model from this server. API keys sent from the UI are held
in memory for the run only; they are never written to disk or returned.
"""

from __future__ import annotations

import io
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr

from calligraphy.agent.client import ModelConfig, complete, load_env_file
from calligraphy.agent.judge import JudgeManager
from calligraphy.agent.providers import PROVIDERS
from calligraphy.agent.runner import DATASETS, MODES, RunManager
from calligraphy.canvas import Canvas, CanvasConfig, StrokeError
from calligraphy.data.bench import bench_by_task, load_bench
from calligraphy.data.mmah import get_entry, load_dictionary, load_graphics
from calligraphy.data.types import CharacterEntry
from calligraphy.episode import EpisodeConfig, replay
from calligraphy.render.brushes import get_brush
from calligraphy.render.raster import InkLayer, composite, render_geometries
from calligraphy.score.levels import score_character
from calligraphy.score.reward import compute_reward
from calligraphy.score.scenes import (
    NORMALIZATION_CHARS,
    NORMALIZATION_ORDER,
    SCENE_LABELS,
    SCENES,
    count_out_of_bounds,
    scene_category,
    through_environment,
    to_response,
)
from calligraphy.score.stroke import stroke_metrics
from calligraphy.tools import ParseError, parse_response, render_response

app = FastAPI(title="Calligraphy viewer")
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
DRAWINGS = ["converter", *[k for k in SCENES if k != "perfect"]]
DRAWING_LABELS = {"converter": SCENE_LABELS["perfect"], **SCENE_LABELS}
RUNS = RunManager()
JUDGE = JudgeManager()
load_env_file()  # provider keys may live in the repo's .env (gitignored)


def resolve(key: str) -> CharacterEntry:
    bench = bench_by_task()
    if key in bench:
        return bench[key]
    try:
        return get_entry(key)
    except KeyError:
        raise HTTPException(404, f"unknown task or character {key!r}") from None


def run_ref(name: str) -> tuple[str, int] | None:
    """("<run_id>", index) for a drawing named run:<run_id>:<index>."""
    if not name.startswith("run:"):
        return None
    try:
        _, run_id, index = name.split(":")
        return run_id, int(index)
    except ValueError:
        raise HTTPException(400, f"bad run drawing {name!r}; expected run:<run_id>:<index>") from None


def rollout(name: str) -> tuple[dict, dict]:
    run_id, index = run_ref(name)
    try:
        return RUNS.rollout(run_id, index), RUNS.summary(run_id)
    except KeyError:
        raise HTTPException(404, f"no rollout {index} in run {run_id!r} (yet)") from None


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
    if run_ref(name):
        record, summary = rollout(name)
        cfg = CanvasConfig(action_frame=summary["action_frame"])
        try:
            canvas = render_response(parse_response(record["response"], cfg), cfg)
        except ParseError:
            return []  # unparseable answers score 0 and render as an empty canvas
        return [(s.points, s.width) for s in canvas.strokes]
    return through_environment(raw_drawing(key, name))


def _finite(d: dict) -> dict:
    """JSON has no infinity (placement of an empty drawing): send null instead."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


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
    drawings = [{"name": d, "label": DRAWING_LABELS[d]} for d in DRAWINGS]
    run_info = None
    if run_ref(drawing_name):
        record, summary = rollout(drawing_name)
        drawings.append({"name": drawing_name, "label": f"Model answer: {summary['model']}"})
        clamped = (record["info"] or {}).get("clamped_points", 0)
        run_info = {
            "run_id": summary["id"],
            "index": record["index"],
            "model": summary["model"],
            "provider": summary["provider"],
            "response": record["response"],
            "reward": record["reward"],
            "parse_ok": bool((record["info"] or {}).get("parse_ok")),
            "error": record["error"] or (record["info"] or {}).get("error"),
            "latency_s": record["latency_s"],
            "stop_reason": record["stop_reason"],
            "mode": record.get("mode") or "single",
            "trace_summary": record.get("trace_summary"),  # tools mode: the episode in numbers
        }
    else:
        clamped = count_out_of_bounds(raw_drawing(key, drawing_name))
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
        "drawings": drawings,
        "clamped_points": clamped,
        "strokes": [{"points": p.round(2).tolist(), "width": round(w, 2)} for p, w in strokes],
        "stroke_metrics": per_stroke,
        "levels": _finite(levels.to_dict()),
        "score_ms": round(score_ms, 1),
        "rollout": run_info,
    }


@lru_cache(maxsize=4)
def _normalization_cases(chars: str) -> tuple[dict, ...]:
    first_task = {}
    for e in load_bench():
        first_task.setdefault(e.character, e.task_id)
    cases = []
    for ch in chars:
        entry = get_entry(ch)
        for name in NORMALIZATION_ORDER:
            reward, info = compute_reward(to_response(SCENES[name](entry)), entry)
            cases.append(
                {
                    "key": first_task.get(ch, ch),  # bench rows show the master's reference image
                    "character": ch,
                    "drawing": "converter" if name == "perfect" else name,
                    "label": SCENE_LABELS[name],
                    "category": scene_category(name),
                    "reward": reward,
                    "shape": info["shape"],
                    "placement_ok": info["placement_ok"],
                    "clamped_points": info["clamped_points"],
                }
            )
    return tuple(cases)


@app.get("/api/normalization")
def normalization(chars: str = NORMALIZATION_CHARS) -> list[dict]:
    """The normalization test set: every scenario on each character, scored through the reward path."""
    return list(_normalization_cases(chars))


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


# --- agent runs ------------------------------------------------------------------


@app.get("/api/agent/providers")
def providers() -> list[dict]:
    """Presets with setup steps. Reports whether each key's environment variable is set, never its value."""
    return [p.to_dict() for p in PROVIDERS.values()]


@app.get("/api/agent/datasets")
def datasets() -> list[dict]:
    return [{"id": k, "label": v} for k, v in DATASETS.items()]


class ModelRequest(BaseModel):
    provider: str
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: SecretStr | None = None  # held in memory for the run only
    temperature: float | None = 1.0
    max_tokens: int = Field(16000, ge=64, le=128000)
    extra_body: dict = Field(default_factory=dict)  # provider options, saved with the run

    def config(self) -> ModelConfig:
        preset = PROVIDERS.get(self.provider)
        if preset is None:
            raise HTTPException(400, f"unknown provider {self.provider!r}")
        key = self.api_key.get_secret_value().strip() if self.api_key else None
        return ModelConfig(
            provider=preset.id,
            kind=preset.kind,
            base_url=(self.base_url or preset.base_url).strip(),
            model=self.model.strip(),
            api_key=key or None,
            key_env=preset.key_env,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )


class RunRequest(ModelRequest):
    dataset: str = "bench"
    characters: str = ""
    limit: int | None = Field(None, ge=1)
    concurrency: int = Field(2, ge=1, le=16)
    mode: str = "single"  # "tools": multi-turn episode with tools; the trace is saved with the run
    max_turns: int = Field(EpisodeConfig.max_turns, ge=1, le=512)
    max_tool_calls: int = Field(EpisodeConfig.max_tool_calls, ge=1, le=1024)
    obs_size: int | None = Field(None, ge=64, le=1024)


@app.post("/api/agent/test")
def test_connection(req: ModelRequest) -> dict:
    """One tiny request to check the endpoint, model name and key before starting a run."""
    cfg = req.config()
    cfg.max_tokens = min(cfg.max_tokens, 2048)  # leaves room for models that think before answering
    c = complete(cfg, [{"role": "user", "content": "Reply with the single word OK."}])
    return {
        "ok": c.error is None,
        "text": c.text[:300],
        "error": c.error,
        "latency_s": round(c.latency_s, 2),
        "stop_reason": c.stop_reason,
        "key_source": cfg.key_source(),
    }


@app.post("/api/agent/runs")
def start_run(req: RunRequest) -> dict:
    try:
        run_id = RUNS.start(
            req.config(), req.dataset, req.characters, req.limit, req.concurrency,
            mode=req.mode, episode=EpisodeConfig(max_turns=req.max_turns, max_tool_calls=req.max_tool_calls), obs_size=req.obs_size,
        )  # fmt: skip
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"run_id": run_id}


@app.get("/api/agent/runs")
def list_runs() -> list[dict]:
    return RUNS.list()


@app.get("/api/agent/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        summary = RUNS.summary(run_id)
    except KeyError:
        raise HTTPException(404, f"unknown run {run_id!r}") from None
    rows = [
        {
            "index": r["index"],
            "key": r["key"],
            "character": r["character"],
            "reward": r["reward"],
            "parse_ok": bool((r["info"] or {}).get("parse_ok")),
            "error": r["error"] or (r["info"] or {}).get("error"),
            "latency_s": r["latency_s"],
            "trace_summary": r.get("trace_summary"),  # tools mode only
        }
        for r in RUNS.rollouts(run_id)
    ]
    return {"summary": summary, "rollouts": rows}


@app.get("/api/agent/modes")
def modes() -> list[dict]:
    return [{"id": k, "label": v} for k, v in MODES.items()]


@app.get("/api/agent/runs/{run_id}/trace/{index}")
def get_trace(run_id: str, index: int) -> dict:
    """Every model reply and tool call of one tools-mode rollout."""
    try:
        return RUNS.trace(run_id, index)
    except KeyError:
        raise HTTPException(404, f"no trace for rollout {index} in run {run_id!r}") from None


@app.get("/api/agent/runs/{run_id}/trace/{index}/image/{name}")
def get_trace_image(run_id: str, index: int, name: str) -> FileResponse:
    """A canvas image the model was shown (the `image` field of a capture_canvas event)."""
    try:
        return FileResponse(RUNS.trace_image(run_id, index, name), media_type="image/png")
    except KeyError:
        raise HTTPException(404, f"no image {name!r} for rollout {index} in run {run_id!r}") from None


# --- judge experiments (scripts/run_judge.py) ---------------------------------------


@app.get("/api/judge")
def list_judge() -> list[dict]:
    """Experiments where a model scored drawings of known quality."""
    return JUDGE.list()


@app.get("/api/judge/{exp_id}")
def get_judge(exp_id: str) -> dict:
    try:
        summary = JUDGE.summary(exp_id)
    except KeyError:
        raise HTTPException(404, f"unknown judge experiment {exp_id!r}") from None
    return {"summary": summary, "items": JUDGE.items(exp_id)}


@app.get("/api/judge/{exp_id}/image/{index}.png")
def get_judge_image(exp_id: str, index: int) -> FileResponse:
    """The image the model was shown for one drawing."""
    try:
        return FileResponse(JUDGE.image(exp_id, index), media_type="image/png", headers={"Cache-Control": "max-age=86400"})
    except KeyError:
        raise HTTPException(404, f"no image {index} in judge experiment {exp_id!r}") from None


MARK_COLORS = {"added": (47, 111, 184), "removed": (200, 60, 45)}


@lru_cache(maxsize=32)
def _trace_at(root: str, run_id: str, index: int, mtime: float) -> dict:
    return RUNS.trace(run_id, index)


def cached_trace(run_id: str, index: int) -> dict:
    """The trace, re-read only when its file changes (a resumed run rewrites it)."""
    try:
        mtime = (RUNS.root / run_id / "traces" / f"{index:04d}.json").stat().st_mtime
        return _trace_at(str(RUNS.root), run_id, index, mtime)
    except (FileNotFoundError, KeyError):
        raise HTTPException(404, f"no trace for rollout {index} in run {run_id!r}") from None


@app.get("/api/agent/runs/{run_id}/trace/{index}/state")
def get_trace_state(
    run_id: str,
    index: int,
    step: int = Query(-1, ge=-1, description="event index; the canvas is shown as it stood after this event (-1: empty)"),
    size: int = Query(240, ge=32, le=1024),
    mark: int | None = Query(None, description="stroke number to draw in colour"),
    tone: str = Query("added", pattern="^(added|removed)$"),
) -> Response:
    """The canvas after event `step` of a tools-mode rollout, rebuilt from the trace.

    The state before tool call i is step=i-1 and the state after it is step=i.
    `mark` colours one stroke: the one a call just drew (added) or is about to delete (removed).
    """
    trace = cached_trace(run_id, index)
    board = replay(trace["events"], step, CanvasConfig(**trace["canvas"]))
    base, marked = InkLayer(size, board.config.supersample), InkLayer(size, board.config.supersample)
    for stroke_id, stroke in zip(board.stroke_ids, board.strokes):
        (marked if stroke_id == mark else base).add(stroke.geometry)
    img = composite(base.image(), grid=True)
    if mark in board.stroke_ids:
        img = Image.composite(Image.new("RGB", img.size, MARK_COLORS[tone]), img, marked.image())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=60"})


@app.post("/api/agent/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    RUNS.cancel(run_id)
    return {"ok": True}


def mount_frontend(target: FastAPI, dist: Path) -> None:
    """Serve the built page from `dist`; every non-API path returns the page (client-side routing)."""
    target.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @target.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        # Unknown API routes must fail as JSON 404s, not fall through to the page.
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, f"no API route /{path}")
        return FileResponse(dist / "index.html")


if WEB_DIST.exists():
    mount_frontend(app, WEB_DIST)
