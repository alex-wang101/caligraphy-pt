"""Can the model grade a drawing? Show it a canvas image and compare its score with the reward.

An experiment draws a few calligraphy-bench characters in ways of known quality:
the correct drawing, the scorer's scenes and normalization scenarios, deliberately
broken drawings, a different character, and real model answers from agent runs.
Each drawing is shown to the model exactly as `capture_canvas` would show it
(448 px, 米字格 grid) and the model is asked for a 0-100 score. The reward for the
same drawing comes from the evaluation path (`compute_reward`), so the two can be
compared item by item.

Each experiment lives in out/judge/<id>/:
- summary.json: settings, progress and the agreement statistics
- items.jsonl: one line per drawing with the reward, the model's reply and its parsed score
- images/<index>.png: the image the model was shown
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from calligraphy.agent.client import ModelConfig, complete, image_message
from calligraphy.agent.runner import RUNS_DIR, RunManager, _alive, _slug
from calligraphy.canvas import CanvasConfig
from calligraphy.data.bench import bench_by_task
from calligraphy.data.types import CharacterEntry
from calligraphy.score.reward import compute_reward
from calligraphy.score.scenes import SCENES, scenario, to_response
from calligraphy.tools import ParseError, parse_response, render_response

JUDGE_DIR = RUNS_DIR.parent / "judge"
OBS_SIZE = 448  # what capture_canvas shows in tools mode
# One bench task per stroke-count band, 5 to 25 strokes (觀 has the real model answers).
DEFAULT_TASKS = ("smh_1165", "mf_6340", "smh_1196", "sgt_6080", "wxz_6177", "wxz_6182", "wzm_4336", "wxz_6420")
THIN_WIDTH = 20.0  # the width the untrained model picked before the prompt said how thick strokes are
GOOD, BAD = 0.8, 0.3  # reward bands for the good-vs-bad separation (AUC)


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    kind: str  # "correct" | "flawed" | "wrong character" | "model answer"


VARIANTS = [
    Variant("perfect", "Correct (converter output)", "correct"),
    Variant("small_corner", "Correct, small, in a corner", "correct"),
    Variant("slid_component", "One component slid sideways", "flawed"),
    Variant("straight_flat", "Straight strokes, flat pressure", "flawed"),
    Variant("thin_brush", "Thin brush (width 20)", "flawed"),
    Variant("missing_half", "Second half of the strokes missing", "flawed"),
    Variant("scrambled", "Every stroke moved at random", "flawed"),
    Variant("squash_x_0.6", "Squashed to 0.6x wide", "flawed"),
    Variant("rotate_8", "Rotated 8 degrees", "flawed"),
    Variant("rotate_20", "Rotated 20 degrees", "flawed"),
    Variant("tiny_0.2", "Tiny: 0.2x", "flawed"),
    Variant("wrong_character", "A different character", "wrong character"),
]
VARIANT_BY_NAME = {v.name: v for v in VARIANTS}


def _drawing(variant: str, entry: CharacterEntry, others: list[CharacterEntry], seed: int):
    perfect = SCENES["perfect"](entry)
    if variant == "perfect":
        return perfect
    if variant == "small_corner":
        return SCENES["A_small_corner"](entry)
    if variant == "slid_component":
        return SCENES["B_slid_component"](entry)
    if variant == "straight_flat":
        return SCENES["C_straight_flat"](entry)
    if variant == "thin_brush":
        return [(p, THIN_WIDTH) for p, _ in perfect]
    if variant == "missing_half":
        return perfect[: math.ceil(len(perfect) / 2)]
    if variant == "scrambled":
        rng = np.random.default_rng(seed)
        out = []
        for p, w in perfect:
            q = p.copy()
            q[:, :2] += rng.uniform(-160, 160, size=2)
            out.append((q, w))
        return out
    if variant == "wrong_character":
        # The other character closest in stroke count, so the ink looks about as busy.
        other = min((o for o in others if o.character != entry.character), key=lambda o: abs(o.n_strokes - entry.n_strokes))
        return SCENES["perfect"](other)
    return scenario(variant)(entry)


def build_cases(task_ids: tuple[str, ...] = DEFAULT_TASKS, runs: RunManager | None = None) -> list[dict]:
    """Every drawing to judge: [{key, character, variant, label, kind, response, source}, ...]."""
    bench = bench_by_task()
    entries = [bench[t] for t in task_ids]
    cases = []
    for ci, entry in enumerate(entries):
        for vi, v in enumerate(VARIANTS):
            try:
                strokes = _drawing(v.name, entry, entries, seed=1000 * ci + vi)
            except (KeyError, ValueError, IndexError):
                continue  # e.g. no component data for the slid scene
            cases.append({"key": entry.task_id, "character": entry.character, "variant": v.name, "label": v.label, "kind": v.kind, "response": to_response(strokes), "source": None})
    # Real model answers to the same characters, from agent runs (models only, not the scripted stub).
    chars = {e.character: e.task_id for e in entries}
    runs = runs or RunManager()
    for s in runs.list():
        if s.get("provider") == "stub" or s.get("model") == "converter":
            continue
        for r in runs.rollouts(s["id"]):
            if r["character"] in chars and (r.get("info") or {}).get("parse_ok") and not r.get("error"):
                cases.append(
                    {
                        "key": chars[r["character"]],
                        "character": r["character"],
                        "variant": "model_answer",
                        "label": f"Model answer: {s['model']}",
                        "kind": "model answer",
                        "response": r["response"],
                        "source": f"run:{s['id']}:{r['index']}",
                    }
                )
    return cases


def _pinyin(entry: CharacterEntry) -> str:
    p = entry.meta.get("pinyin")
    return ", ".join(p) if isinstance(p, list) else (p or "")


def judge_messages(entry: CharacterEntry, png: bytes) -> list[dict]:
    name = f"{entry.character} ({_pinyin(entry)})" if _pinyin(entry) else entry.character
    system = "You grade Chinese characters written with a brush. Look closely at the image before you answer."
    text = (
        f"This image should show the character {name}, written stroke by stroke with a brush on a 米字格 guide grid "
        "(the pink border, centre lines and diagonals are the grid, not part of the character).\n"
        f"Score how well the ink matches a correct, standard writing of {entry.character}, from 0 to 100:\n"
        "- 100: the right character, every stroke present, well formed and in the right place;\n"
        "- about 50: recognisable, but with clear errors (a misplaced component, missing or extra strokes, distorted strokes);\n"
        f"- 0: not {entry.character} at all.\n"
        "Size and position on the grid do not matter; judge the character itself.\n"
        'Reply with JSON only: {"recognized_as": "<the character you see, or ?>", "is_correct_character": true or false, '
        '"score": <0-100>, "problems": "<one short sentence>"}'
    )
    return [{"role": "system", "content": system}, image_message(text, png)]


_JSON = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_judgement(text: str) -> dict:
    """The model's reply -> {score (0-1 or None), recognized_as, is_correct_character, problems, parse_ok}."""
    text = _THINK.sub("", text or "")
    data = None
    for m in [*_JSON.finditer(text)][::-1]:
        try:
            data = json.loads(m.group(0))
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', text)
        data = {"score": float(m.group(1))} if m else {}
    score = data.get("score")
    ok = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
    correct = data.get("is_correct_character")
    return {
        "score": min(max(float(score), 0.0), 100.0) / 100 if ok else None,
        "recognized_as": str(data.get("recognized_as") or "")[:8],
        "is_correct_character": correct if isinstance(correct, bool) else None,
        "problems": str(data.get("problems") or "")[:400],
        "parse_ok": ok,
    }


def capture(response: str, canvas: CanvasConfig) -> bytes:
    """The drawing as capture_canvas shows it: through the parser and canvas, at obs_size, with the grid."""
    try:
        board = render_response(parse_response(response, canvas), canvas)
    except ParseError:
        board = render_response([], canvas)
    buf = io.BytesIO()
    board.capture().save(buf, format="PNG")
    return buf.getvalue()


# --- statistics ------------------------------------------------------------------


def _ranks(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    for v in np.unique(a):  # ties share their mean rank
        idx = a == v
        r[idx] = r[idx].mean()
    return r


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def agreement(items: list[dict]) -> dict:
    """How well the model's scores track the reward, overall and per kind of drawing."""
    scored = [i for i in items if i.get("judge") and i["judge"]["parse_ok"]]
    x = np.array([i["judge"]["score"] for i in scored])
    y = np.array([i["reward"] for i in scored])
    good = [i["judge"]["score"] for i in scored if i["reward"] >= GOOD]
    bad = [i["judge"]["score"] for i in scored if i["reward"] < BAD]
    pairs = [(g > b) + 0.5 * (g == b) for g in good for b in bad]

    def rate(rows, pred):
        rows = [r for r in rows if r.get("judge") and r["judge"]["parse_ok"]]
        return (sum(pred(r) for r in rows) / len(rows)) if rows else None

    correct = [i for i in items if i["kind"] == "correct"]
    by_variant = []
    for name in [v.name for v in VARIANTS] + ["model_answer"]:
        rows = [i for i in scored if i["variant"] == name]
        if rows:
            by_variant.append(
                {
                    "variant": name,
                    "label": VARIANT_BY_NAME[name].label if name in VARIANT_BY_NAME else "Real model answers",
                    "kind": rows[0]["kind"],
                    "n": len(rows),
                    "reward_mean": float(np.mean([r["reward"] for r in rows])),
                    "judge_mean": float(np.mean([r["judge"]["score"] for r in rows])),
                    "says_correct": rate(rows, lambda r: r["judge"]["is_correct_character"] is True),
                }
            )
    return {
        "n": len(items),
        "n_scored": len(scored),
        "spearman": _pearson(_ranks(x), _ranks(y)) if len(scored) >= 3 else None,
        "pearson": _pearson(x, y),
        "mae": float(np.abs(x - y).mean()) if len(scored) else None,
        "auc_good_bad": (sum(pairs) / len(pairs)) if pairs else None,
        "n_good": len(good),
        "n_bad": len(bad),
        "judge_mean_good": float(np.mean(good)) if good else None,
        "judge_mean_bad": float(np.mean(bad)) if bad else None,
        # Can it read a correct drawing, and does it notice a different character?
        "reads_correct": rate(correct, lambda r: r["judge"]["recognized_as"].strip() == r["character"]),
        "accepts_correct": rate(correct, lambda r: r["judge"]["is_correct_character"] is True),
        "rejects_wrong": rate([i for i in items if i["kind"] == "wrong character"], lambda r: r["judge"]["is_correct_character"] is False),
        "by_variant": by_variant,
    }


# --- running and storage ---------------------------------------------------------


class JudgeManager:
    """Starts judge experiments in a background thread and reads them back from disk."""

    def __init__(self, root: Path = JUDGE_DIR):
        self.root = root

    def start(self, config: ModelConfig, task_ids: tuple[str, ...] = DEFAULT_TASKS, concurrency: int = 4) -> tuple[str, threading.Thread]:
        cases = build_cases(task_ids)
        exp_id = f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(config.model)}"
        exp_dir = self.root / exp_id
        (exp_dir / "images").mkdir(parents=True)
        summary = {
            "id": exp_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "provider": config.provider,
            "model": config.model,
            "api_key_source": config.key_source(),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "extra_body": config.extra_body,
            "tasks": list(task_ids),
            "characters": "".join(dict.fromkeys(c["character"] for c in cases)),
            "obs_size": OBS_SIZE,
            "n_total": len(cases),
            "n_done": 0,
            "n_errors": 0,
            "pid": os.getpid(),
            "prompt": judge_messages(bench_by_task()[task_ids[0]], b"")[1]["content"][0]["text"],
        }
        self._write(exp_dir, summary)
        t = threading.Thread(target=self._run, args=(exp_dir, summary, config, cases, concurrency), daemon=True)
        t.start()
        return exp_id, t

    def _write(self, exp_dir: Path, summary: dict) -> None:
        (exp_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    def _run(self, exp_dir: Path, summary: dict, config: ModelConfig, cases: list[dict], concurrency: int) -> None:
        bench, canvas, lock = bench_by_task(), CanvasConfig(obs_size=OBS_SIZE), threading.Lock()
        items: list[dict] = []
        t0 = time.perf_counter()

        def task(index: int, case: dict) -> None:
            entry = bench[case["key"]]
            reward, info = compute_reward(case["response"], entry)
            png = capture(case["response"], canvas)
            (exp_dir / "images" / f"{index:04d}.png").write_bytes(png)
            c = complete(config, judge_messages(entry, png))
            item = {
                "index": index,
                **{k: case[k] for k in ("key", "character", "variant", "label", "kind", "source")},
                "reward": reward,
                "shape": info.get("shape", 0.0),
                "n_strokes": info.get("n_agent", 0),
                "n_target": entry.n_strokes,
                "reply": c.text,
                "error": c.error,
                "latency_s": round(c.latency_s, 2),
                "usage": c.usage,
                "judge": None if c.error else parse_judgement(c.text),
            }
            with lock:
                items.append(item)
                with open(exp_dir / "items.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                summary["n_done"] += 1
                summary["n_errors"] += bool(c.error)
                summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
                summary["stats"] = agreement(items)
                self._write(exp_dir, summary)

        try:
            with ThreadPoolExecutor(max(1, concurrency)) as pool:
                for f in [pool.submit(task, i, c) for i, c in enumerate(cases)]:
                    f.result()
            summary["status"] = "done"
        except Exception as e:  # noqa: BLE001
            summary["status"], summary["error"] = "error", f"{type(e).__name__}: {e}"
        with lock:
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["stats"] = agreement(items)
            self._write(exp_dir, summary)

    def summary(self, exp_id: str) -> dict:
        path = self.root / exp_id / "summary.json"
        if not path.exists():
            raise KeyError(exp_id)
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("status") == "running" and not (s.get("pid") and _alive(s["pid"])):
            s["status"] = "interrupted"
        return s

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = [self.summary(p.name) for p in self.root.iterdir() if (p / "summary.json").exists()]
        return sorted(out, key=lambda s: s.get("created_at") or "", reverse=True)

    def items(self, exp_id: str) -> list[dict]:
        path = self.root / exp_id / "items.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return sorted(rows, key=lambda r: r["index"])

    def image(self, exp_id: str, index: int) -> Path:
        path = self.root / exp_id / "images" / f"{index:04d}.png"
        if not path.exists():
            raise KeyError(f"{exp_id}:{index}")
        return path
