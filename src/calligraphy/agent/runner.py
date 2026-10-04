"""Run a model on the phase-1 task over a dataset and score every answer.

Each run lives in out/runs/<run_id>/:
- summary.json: settings (never the API key), progress and aggregate scores
- rollouts.jsonl: one line per character with the raw response and the reward info
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from calligraphy.agent.client import ModelConfig, complete
from calligraphy.canvas import CanvasConfig
from calligraphy.data.bench import load_bench
from calligraphy.data.mmah import get_entry, load_graphics
from calligraphy.data.types import CharacterEntry
from calligraphy.score.reward import compute_reward
from calligraphy.splits import load_splits
from calligraphy.tools import phase1_messages

RUNS_DIR = Path(__file__).resolve().parents[3] / "out" / "runs"

DATASETS = {
    "bench": "calligraphy-bench (88, eval only)",
    "viz16": "Validation image set (16)",
    "val": "Validation split (300)",
    "test_compositional": "Compositional test split (500)",
    "custom": "Custom characters",
}


def dataset_items(dataset: str, characters: str = "", limit: int | None = None) -> list[tuple[str, CharacterEntry]]:
    """(viewer key, entry) pairs. Bench rows are keyed by task_id so the viewer shows the master's reference."""
    if dataset == "bench":
        items = [(e.task_id, e) for e in load_bench()]
    elif dataset in ("val", "viz16", "test_compositional"):
        items = [(c, get_entry(c)) for c in getattr(load_splits(), dataset)]
    elif dataset == "custom":
        graphics = load_graphics()
        chars = list(dict.fromkeys(ch for ch in characters if ch in graphics))
        if not chars:
            raise ValueError("no Make Me a Hanzi characters in the custom list")
        items = [(c, get_entry(c)) for c in chars]
    else:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    return items[:limit] if limit else items


def _pinyin(entry: CharacterEntry) -> str | None:
    p = entry.meta.get("pinyin")
    return ", ".join(p) if isinstance(p, list) else p


def run_one(config: ModelConfig, key: str, entry: CharacterEntry, canvas: CanvasConfig) -> dict:
    messages = phase1_messages(entry.character, _pinyin(entry), canvas)
    c = complete(config, messages)
    record = {
        "key": key,
        "character": entry.character,
        "response": c.text,
        "stop_reason": c.stop_reason,
        "latency_s": round(c.latency_s, 2),
        "usage": c.usage,
        "error": c.error,
    }
    if c.error:
        record.update(reward=0.0, info={"parse_ok": False, "error": c.error})
    else:
        reward, info = compute_reward(c.text, entry, canvas)
        record.update(reward=reward, info=info)
    return record


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "model"


class RunManager:
    """Starts runs in background threads and serves their state from memory and disk."""

    def __init__(self, root: Path = RUNS_DIR):
        self.root = root
        self._active: dict[str, dict] = {}  # run_id -> {"summary", "cancel", "lock"}

    def start(
        self,
        config: ModelConfig,
        dataset: str,
        characters: str = "",
        limit: int | None = None,
        concurrency: int = 2,
        action_frame: int = 1024,
    ) -> str:
        items = dataset_items(dataset, characters, limit)
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(config.model)}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        summary = {
            "id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "provider": config.provider,
            "kind": config.kind,
            "base_url": config.base_url,
            "model": config.model,
            "api_key_source": config.key_source(),
            "temperature": None if config.kind == "anthropic" else config.temperature,
            "max_tokens": config.max_tokens,
            "dataset": dataset,
            "dataset_label": DATASETS[dataset],
            "action_frame": action_frame,
            "concurrency": concurrency,
            "n_total": len(items),
            "n_done": 0,
            "n_parse_ok": 0,
            "n_errors": 0,
            "reward_sum": 0.0,
            "mean_reward": None,
            "elapsed_s": 0.0,
        }
        state = {"summary": summary, "cancel": threading.Event(), "lock": threading.Lock(), "dir": run_dir}
        self._active[run_id] = state
        self._write_summary(state)
        t = threading.Thread(target=self._run, args=(state, config, items, concurrency, action_frame), daemon=True)
        state["thread"] = t
        t.start()
        return run_id

    def _write_summary(self, state: dict) -> None:
        (state["dir"] / "summary.json").write_text(json.dumps(state["summary"], ensure_ascii=False, indent=1))

    def _run(self, state: dict, config: ModelConfig, items, concurrency: int, action_frame: int) -> None:
        summary, lock = state["summary"], state["lock"]
        canvas = CanvasConfig(action_frame=action_frame)
        t0 = time.perf_counter()

        def task(i: int, key: str, entry: CharacterEntry) -> None:
            if state["cancel"].is_set():
                return
            record = {"index": i, **run_one(config, key, entry, canvas)}
            with lock:
                with open(state["dir"] / "rollouts.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary["n_done"] += 1
                summary["n_parse_ok"] += bool(record["info"].get("parse_ok"))
                summary["n_errors"] += bool(record["error"])
                summary["reward_sum"] += record["reward"]
                summary["mean_reward"] = summary["reward_sum"] / summary["n_done"]
                summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
                self._write_summary(state)

        try:
            with ThreadPoolExecutor(max(1, concurrency)) as pool:
                for f in [pool.submit(task, i, k, e) for i, (k, e) in enumerate(items)]:
                    f.result()
            status = "cancelled" if state["cancel"].is_set() else "done"
        except Exception as e:  # noqa: BLE001
            status, summary["error"] = "error", f"{type(e).__name__}: {e}"
        with lock:
            summary["status"] = status
            summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
            self._write_summary(state)

    def wait(self, run_id: str) -> None:
        if run_id in self._active:
            self._active[run_id]["thread"].join()

    def cancel(self, run_id: str) -> None:
        if run_id in self._active:
            self._active[run_id]["cancel"].set()

    def summary(self, run_id: str) -> dict:
        if run_id in self._active:
            with self._active[run_id]["lock"]:
                return dict(self._active[run_id]["summary"])
        path = self.root / run_id / "summary.json"
        if not path.exists():
            raise KeyError(run_id)
        s = json.loads(path.read_text())
        if s.get("status") == "running":
            s["status"] = "interrupted"  # the server restarted mid-run
        return s

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        ids = sorted((p.name for p in self.root.iterdir() if (p / "summary.json").exists()), reverse=True)
        return [self.summary(i) for i in ids]

    def rollouts(self, run_id: str) -> list[dict]:
        path = self.root / run_id / "rollouts.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return sorted(rows, key=lambda r: r["index"])

    def rollout(self, run_id: str, index: int) -> dict:
        for r in self.rollouts(run_id):
            if r["index"] == index:
                return r
        raise KeyError(f"{run_id}:{index}")
