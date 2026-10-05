"""Multi-turn episodes: tool calls change the canvas and every one of them is recorded."""

import json

import pytest
from conftest import needs_data
from PIL import Image

from calligraphy.canvas import CanvasConfig
from calligraphy.convert import convert
from calligraphy.data.mmah import get_entry
from calligraphy.episode import Episode, EpisodeConfig, replay
from calligraphy.score.reward import compute_reward
from calligraphy.tools import format_strokes

pytestmark = needs_data


def shifted(stroke: dict, dx: float) -> dict:
    return {"points": [[x + dx, y, p] for x, y, p in stroke["points"]], "width": stroke["width"]}


@pytest.fixture()
def yong():
    entry = get_entry("永")
    return entry, convert(entry)


def test_trace_records_every_call_and_the_score_after_each_change(yong, tmp_path):
    entry, strokes = yong
    cfg = CanvasConfig(obs_size=128)
    ep = Episode(entry, cfg, image_dir=tmp_path)

    # Draft: every stroke, the second one out of place. Then look, fix it, look again, submit.
    for i, s in enumerate(strokes):
        assert ep.call("create_stroke", json.dumps(shifted(s, 150) if i == 1 else s), turn=1, call_id=f"a{i}").ok
    look = ep.call("capture_canvas", "{}", turn=1)
    assert look.image.size == (128, 128)
    assert ep.call("delete_stroke", {"stroke_id": 2}, turn=2).ok
    assert ep.call("create_stroke", strokes[1], turn=2).stroke_id == 2
    ep.call("capture_canvas", "", turn=2)  # some servers send "" for a call without arguments
    assert ep.call("submit", None, turn=3).done and ep.done and ep.end_reason == "submit"

    tools = [e for e in ep.events if e["type"] == "tool"]
    assert [e["name"] for e in tools] == ["create_stroke"] * 5 + ["capture_canvas", "delete_stroke", "create_stroke", "capture_canvas", "submit"]
    assert [e["i"] for e in ep.events] == list(range(len(ep.events))) and ep.events[-1] == {**ep.events[-1], "type": "end", "reason": "submit"}
    assert tools[0]["call_id"] == "a0" and tools[0]["arguments"] == strokes[0] and tools[0]["stroke_ids"] == [1]
    assert all(e["turn"] in (1, 2, 3) and e["t_s"] >= 0 for e in tools)

    draft, delete, redraw = tools[4], tools[6], tools[7]
    # The score climbs as strokes go on, dips while stroke 2 is missing, and ends above the draft.
    assert draft["score"]["structure"] < 0.8
    assert delete["stroke_ids"] == [1, 3, 4, 5] and delete["score"]["structure"] < draft["score"]["structure"]
    assert delete["deleted_stroke_score"] == draft["score"]["strokes"]["2"] < 0.2  # it deleted a stroke that was wrong
    assert redraw["redraw"] is True and redraw["stroke_ids"] == [1, 2, 3, 4, 5] and not tools[0]["redraw"]
    assert redraw["score"]["structure"] > 0.85 and redraw["score"]["strokes"]["2"] > 0.8

    # Captures are saved as the model saw them, and carry no score.
    captures = [e for e in tools if e["name"] == "capture_canvas"]
    assert [e["image"] for e in captures] == [f"step-{e['i']:03d}.png" for e in captures]
    assert all((tmp_path / e["image"]).stat().st_size > 0 and "score" not in e for e in captures)

    # The final canvas is a phase-1 answer: same strokes, same reward as the last step score.
    assert ep.final_strokes() == strokes and ep.final_response() == format_strokes(strokes)
    reward, info = compute_reward(ep.final_response(), entry, cfg)
    assert info["parse_ok"] and reward == pytest.approx(redraw["score"]["structure"], abs=1e-4)

    s = ep.summary()
    assert s["end_reason"] == "submit" and s["n_tool_calls"] == 10 and s["n_failed_calls"] == 0
    assert s["calls"] == {"create_stroke": 6, "capture_canvas": 2, "delete_stroke": 1, "submit": 1}
    assert (s["n_captures"], s["n_deletes"], s["n_redraws"], s["n_strokes"]) == (2, 1, 1, 5)
    assert s["checked_final"] is True
    assert s["draft_score"] == draft["score"]["structure"] and s["final_score"] == s["peak_score"] == redraw["score"]["structure"]
    events = json.loads(json.dumps(ep.to_dict()))["events"]  # the whole trace is plain JSON

    # Any state can be rebuilt from the trace alone: the strokes on the canvas after
    # each call, and the exact image the model was shown.
    assert replay(events, -1, cfg).strokes == []
    for e in events:
        if e["type"] == "tool":
            assert replay(events, e["i"], cfg).stroke_ids == e["stroke_ids"]
        if e.get("image"):
            assert replay(events, e["i"], cfg).capture().tobytes() == Image.open(tmp_path / e["image"]).convert("RGB").tobytes()
    assert replay(events, delete["i"] - 1, cfg).stroke_ids == [1, 2, 3, 4, 5]  # before the delete


def test_rejected_calls_change_nothing_and_are_recorded(yong):
    entry, strokes = yong
    ep = Episode(entry)
    bad_json = ep.call("create_stroke", '{"points": [[1, 2, 0.5]', turn=1)
    assert not bad_json.ok and "not valid JSON" in bad_json.text
    assert not ep.call("create_stroke", {"points": [[1, 2]], "width": 5}, turn=1).ok
    assert not ep.call("delete_stroke", {"stroke_id": 1}, turn=1).ok
    assert not ep.call("erase_everything", {}, turn=1).ok
    assert not ep.call("create_stroke", "[1, 2, 3]", turn=1).ok
    assert ep.canvas.strokes == [] and not ep.done

    events = ep.events
    assert [e["ok"] for e in events] == [False] * 5 and all("score" not in e for e in events)
    assert events[0]["arguments"] is None and events[0]["arguments_raw"] == '{"points": [[1, 2, 0.5]'
    assert events[3]["result"].startswith("error: unknown tool")

    ep.call("create_stroke", strokes[0], turn=2)
    ep.call("submit", {}, turn=2)
    late = ep.call("create_stroke", strokes[1], turn=2)  # a call that arrives after submit is not run
    assert not late.ok and ep.events[-1]["skipped"] and len(ep.canvas.strokes) == 1
    s = ep.summary()
    assert (s["n_tool_calls"], s["n_failed_calls"], s["n_skipped_calls"], s["checked_final"]) == (7, 5, 1, False)
    assert s["draft_score"] is None  # nothing was deleted


def test_tool_call_cap_ends_the_episode(yong):
    entry, strokes = yong
    ep = Episode(entry, config=EpisodeConfig(max_tool_calls=3))
    for s in strokes[:3]:
        ep.call("create_stroke", s)
    assert ep.done and ep.end_reason == "max_tool_calls" and len(ep.canvas.strokes) == 3
    ep.end("submit")  # the first reason wins
    assert ep.end_reason == "max_tool_calls"
    with pytest.raises(ValueError):
        Episode(entry).end("bored")


def test_model_turns_and_nudges_are_part_of_the_trace(yong):
    entry, strokes = yong
    ep = Episode(None)  # without a target nothing is scored
    ep.model_turn(1, text="Let me think.", reasoning="dot first", stop_reason="stop", latency_s=1.234, usage={"total_tokens": 9})
    ep.nudge(1, "Use the tools.")
    ep.model_turn(2, stop_reason="tool_calls", n_tool_calls=1)
    ep.call("create_stroke", strokes[0], turn=2)
    ep.end("max_turns")
    assert [e["type"] for e in ep.events] == ["model", "nudge", "model", "tool", "end"]
    assert ep.events[0] == {**ep.events[0], "turn": 1, "text": "Let me think.", "reasoning": "dot first", "latency_s": 1.23, "usage": {"total_tokens": 9}}
    assert ep.events[3]["score"] is None
    s = ep.summary()
    assert (s["n_turns"], s["n_nudges"], s["end_reason"], s["final_score"]) == (2, 1, "max_turns", None)
