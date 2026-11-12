"""Run a model over a dataset and score every answer.

Two modes:
- "single" (phase 1): one reply holding every stroke;
- "tools" (phase 2): a multi-turn episode with the create_stroke, delete_stroke,
  capture_canvas and submit tools. The final canvas is scored exactly like a
  phase-1 answer.

Each run lives in out/runs/<run_id>/:
- summary.json: settings (never the API key), progress and aggregate scores
- rollouts.jsonl: one line per character with the answer, the reward info and,
  in tools mode, a summary of the episode
- traces/<index>.json (tools mode): every model reply and tool call of the episode
- traces/<index>/step-NNN.png (tools mode): each canvas image the model was shown
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from calligraphy.agent.client import ModelConfig, complete, image_message, tool_message
from calligraphy.canvas import CanvasConfig
from calligraphy.data.bench import load_bench
from calligraphy.data.mmah import get_entry, load_graphics
from calligraphy.data.types import CharacterEntry
from calligraphy.episode import Episode, EpisodeConfig
from calligraphy.score.reward import compute_reward
from calligraphy.splits import load_splits
from calligraphy.tools import phase1_messages, phase2_messages, tool_schemas

RUNS_DIR = Path(__file__).resolve().parents[3] / "out" / "runs"

MODES = {
    "single": "One reply with every stroke (phase 1)",
    "tools": "Multi-turn with draw, delete and canvas-image tools (phase 2)",
}
SINGLE_OBS_SIZE = 224  # unused in single mode; kept in the summary for completeness
TOOLS_OBS_SIZE = 448  # canvas image side in tools mode: about 200 image tokens on Qwen3-VL
NUDGE = "Use the tools to continue. When the character is finished, call submit."

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
        "reasoning": c.reasoning,
        "stop_reason": c.stop_reason,
        "latency_s": round(c.latency_s, 2),
        "usage": c.usage,
        "error": c.error,
        "rate_limited": c.rate_limited,
    }
    if c.error:
        record.update(reward=0.0, info={"parse_ok": False, "error": c.error})
    else:
        reward, info = compute_reward(c.text, entry, canvas)
        record.update(reward=reward, info=info)
    return record


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def run_episode(
    config: ModelConfig,
    key: str,
    entry: CharacterEntry,
    canvas: CanvasConfig,
    episode: EpisodeConfig,
    trace_dir: Path | None = None,
    index: int = 0,
) -> dict:
    """One tools-mode episode. Writes the full trace to trace_dir/<index>.json and returns the rollout row."""
    stem = f"{index:04d}"
    image_dir = trace_dir / stem if trace_dir else None
    if image_dir is not None:
        shutil.rmtree(image_dir, ignore_errors=True)  # images from an earlier, unfinished attempt at this item
    messages = phase2_messages(entry.character, _pinyin(entry), canvas)
    system, user = messages[0]["content"], messages[1]["content"]
    tools = tool_schemas(canvas)
    ep = Episode(entry, canvas, episode, image_dir=image_dir)
    usage: Counter = Counter()
    error, rate_limited, stop_reason, nudges = None, False, None, 0

    for turn in range(1, episode.max_turns + 1):
        c = complete(config, messages, tools)
        ep.model_turn(turn, c.text, c.reasoning, c.stop_reason, c.latency_s, c.usage, len(c.tool_calls), c.error)
        stop_reason = c.stop_reason
        usage.update({k: v for k, v in (c.usage or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        if c.error:
            error, rate_limited = c.error, c.rate_limited
            ep.end("rate_limited" if c.rate_limited else "error")
            break
        messages.append(c.message or {"role": "assistant", "content": c.text})
        if not c.tool_calls:
            if nudges >= episode.max_nudges:
                ep.end("no_tool_call")
                break
            nudges += 1
            ep.nudge(turn, NUDGE)
            messages.append({"role": "user", "content": NUDGE})
            continue
        shown = []
        for tc in c.tool_calls:
            result = ep.call(tc["name"], tc["arguments"], turn, tc["id"])
            messages.append(tool_message(tc["id"], result.text, result.ok))
            if result.image is not None:
                shown.append(image_message(f"Image for your capture_canvas call: {result.text}.", _png(result.image)))
        if ep.done:
            break
        messages.extend(shown)  # images go after every tool result of the turn
    else:
        ep.end("max_turns")

    response = ep.final_response()
    if error:
        reward, info = 0.0, {"parse_ok": False, "error": error}
    else:
        reward, info = compute_reward(response, entry, canvas)
    summary = ep.summary()
    record = {
        "key": key,
        "character": entry.character,
        "mode": "tools",
        "response": response,  # the final canvas in the phase-1 answer format
        "stop_reason": stop_reason,
        "latency_s": summary["duration_s"],
        "usage": dict(usage),
        "error": error,
        "rate_limited": rate_limited,
        "reward": reward,
        "info": info,
        "trace_summary": summary,
        "trace": f"traces/{stem}.json" if trace_dir else None,
    }
    if trace_dir is not None and not rate_limited:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace = {
            "index": index,
            "key": key,
            "character": entry.character,
            "provider": config.provider,
            "model": config.model,
            "system": system,
            "user": user,
            "tools": tools,
            "canvas": {"action_frame": canvas.action_frame, "obs_size": canvas.obs_size, "brush": canvas.brush, "max_points": canvas.max_points, "max_strokes": canvas.max_strokes},
            "episode": episode.to_dict(),
            **ep.to_dict(),
            "reward": reward,
            "info": info,
            "error": error,
        }
        (trace_dir / f"{stem}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists, but owned by someone else
    return True


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
        mode: str = "single",
        episode: EpisodeConfig | None = None,
        obs_size: int | None = None,
    ) -> str:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
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
            "extra_body": config.extra_body,
            "dataset": dataset,
            "dataset_label": DATASETS[dataset],
            "characters": characters,
            "mode": mode,
            "episode": (episode or EpisodeConfig()).to_dict() if mode == "tools" else None,
            "obs_size": obs_size or (TOOLS_OBS_SIZE if mode == "tools" else SINGLE_OBS_SIZE),
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
        self._launch(run_dir, summary, config, [(i, k, e) for i, (k, e) in enumerate(items)])
        return run_id

    def resume(self, run_id: str, config: ModelConfig) -> list[int]:
        """Rerun the items a stopped run never answered, plus those that failed in transport.

        Model answers that parsed or failed to parse are kept: they are results. Rows
        with an `error` (HTTP failure, timeout, rate limit) are dropped and asked again,
        appending to the same run. Returns the indices being rerun.
        """
        if run_id in self._active and self._active[run_id]["thread"].is_alive():
            raise ValueError(f"run {run_id} is still running")
        run_dir = self.root / run_id
        summary = self.summary(run_id)
        items = dataset_items(summary["dataset"], summary.get("characters", ""), summary["n_total"])
        kept = [r for r in self.rollouts(run_id) if not r["error"]]
        done = {r["index"] for r in kept}
        todo = [(i, k, e) for i, (k, e) in enumerate(items) if i not in done]
        with open(run_dir / "rollouts.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
        n_parse_ok = sum(bool(r["info"].get("parse_ok")) for r in kept)
        reward_sum = sum(r["reward"] for r in kept)
        summary.update(
            status="running",
            n_done=len(kept),
            n_parse_ok=n_parse_ok,
            n_errors=0,
            reward_sum=reward_sum,
            mean_reward=reward_sum / len(kept) if kept else None,
            resumed=summary.get("resumed", 0) + 1,
        )
        summary.pop("finished_at", None)
        summary.pop("error", None)
        self._launch(run_dir, summary, config, todo)
        return [i for i, _, _ in todo]

    def _launch(self, run_dir: Path, summary: dict, config: ModelConfig, items: list) -> None:
        state = {"summary": summary, "cancel": threading.Event(), "lock": threading.Lock(), "dir": run_dir, "rate_limited": False}
        summary["pid"] = os.getpid()  # lets another process (the viewer) tell a live run from a dead one
        self._active[summary["id"]] = state
        self._write_summary(state)
        t = threading.Thread(target=self._run, args=(state, config, items), daemon=True)
        state["thread"] = t
        t.start()

    def _write_summary(self, state: dict) -> None:
        (state["dir"] / "summary.json").write_text(json.dumps(state["summary"], ensure_ascii=False, indent=1))

    def _run(self, state: dict, config: ModelConfig, items) -> None:
        summary, lock = state["summary"], state["lock"]
        concurrency = summary["concurrency"]
        canvas = CanvasConfig(action_frame=summary["action_frame"], obs_size=summary.get("obs_size") or SINGLE_OBS_SIZE)
        tools_mode = summary.get("mode") == "tools"
        episode = EpisodeConfig(**summary["episode"]) if tools_mode else None
        t0 = time.perf_counter()
        elapsed_before = summary.get("elapsed_s", 0.0)

        def task(i: int, key: str, entry: CharacterEntry) -> None:
            if state["cancel"].is_set():
                return
            if tools_mode:
                record = {"index": i, **run_episode(config, key, entry, canvas, episode, state["dir"] / "traces", i)}
            else:
                record = {"index": i, **run_one(config, key, entry, canvas)}
            if record["rate_limited"]:
                # Not the model's answer: stop instead of scoring the rest of the quota as zeros.
                state["rate_limited"] = True
                state["cancel"].set()
                return
            with lock:
                with open(state["dir"] / "rollouts.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary["n_done"] += 1
                summary["n_parse_ok"] += bool(record["info"].get("parse_ok"))
                summary["n_errors"] += bool(record["error"])
                summary["reward_sum"] += record["reward"]
                summary["mean_reward"] = summary["reward_sum"] / summary["n_done"]
                summary["elapsed_s"] = round(elapsed_before + time.perf_counter() - t0, 1)
                self._write_summary(state)

        try:
            with ThreadPoolExecutor(max(1, concurrency)) as pool:
                for f in [pool.submit(task, i, k, e) for i, k, e in items]:
                    f.result()
            if state["rate_limited"]:
                status = "rate_limited"
                summary["error"] = "rate limited: the provider's request cap was reached; resume the run once it resets"
            else:
                status = "cancelled" if state["cancel"].is_set() else "done"
        except Exception as e:  # noqa: BLE001
            status, summary["error"] = "error", f"{type(e).__name__}: {e}"
        with lock:
            summary["status"] = status
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["elapsed_s"] = round(elapsed_before + time.perf_counter() - t0, 1)
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
            pid = s.get("pid")
            # Running in another process (e.g. started from the command line), or left behind by one that died.
            if not (pid and pid != os.getpid() and _alive(pid)):
                s["status"] = "interrupted"
        return s

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        runs = [self.summary(p.name) for p in self.root.iterdir() if (p / "summary.json").exists()]
        return sorted(runs, key=lambda s: (s.get("created_at") or "", s["id"]), reverse=True)  # newest first

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

    def trace(self, run_id: str, index: int) -> dict:
        """The full episode trace of a tools-mode rollout."""
        path = self.root / run_id / "traces" / f"{index:04d}.json"
        if not path.exists():
            raise KeyError(f"{run_id}:{index}")
        return json.loads(path.read_text(encoding="utf-8"))

    def trace_image(self, run_id: str, index: int, name: str) -> Path:
        """A canvas image the model was shown during a tools-mode rollout."""
        if not re.fullmatch(r"step-\d{3,}\.png", name):
            raise KeyError(name)
        path = self.root / run_id / "traces" / f"{index:04d}" / name
        if not path.exists():
            raise KeyError(f"{run_id}:{index}:{name}")
        return path
