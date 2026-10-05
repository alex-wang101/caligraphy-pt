"""Agent runs: dataset selection, the runner, the HTTP client, and key handling."""

import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage
from conftest import needs_data

import calligraphy.agent.client as client_mod
import calligraphy.agent.runner as runner_mod
from calligraphy.agent.client import Completion, ModelConfig, complete
from calligraphy.agent.runner import RunManager, dataset_items
from calligraphy.convert import convert
from calligraphy.data.mmah import get_entry
from calligraphy.tools import format_strokes, parse_response

SECRET = "sk-test-DO-NOT-STORE"


def stub_config(**kw) -> ModelConfig:
    return ModelConfig(provider="stub", kind="openai", base_url="http://stub.invalid/v1", model="converter", **kw)


def converter_answer(config, messages):
    char = messages[-1]["content"].split("Write the character ")[1][0]
    return Completion(format_strokes(convert(get_entry(char))), stop_reason="stop", latency_s=0.01)


def test_parser_ignores_think_blocks():
    strokes = [{"points": [[100, 100, 1], [900, 100, 1]], "width": 40}]
    text = '<think>maybe {"strokes": []} ... no</think>\n' + format_strokes(strokes)
    assert parse_response(text) == strokes


@needs_data
def test_dataset_items():
    bench = dataset_items("bench")
    assert len(bench) == 88 and bench[0][0] == "wxz_6420"
    assert len(dataset_items("viz16")) == 16
    assert len(dataset_items("val", limit=5)) == 5
    assert [k for k, _ in dataset_items("custom", "永x永國")] == ["永", "國"]
    with pytest.raises(ValueError):
        dataset_items("custom", "abc")
    with pytest.raises(ValueError):
        dataset_items("nope")


@needs_data
def test_run_scores_answers_and_never_stores_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "complete", converter_answer)
    runs = RunManager(tmp_path)
    run_id = runs.start(stub_config(api_key=SECRET), "custom", "永國心", concurrency=2)
    runs.wait(run_id)
    s = runs.summary(run_id)
    assert s["status"] == "done" and s["n_done"] == 3 and s["n_parse_ok"] == 3
    assert s["mean_reward"] > 0.85
    assert s["api_key_source"] == "pasted (memory only)"
    assert [r["character"] for r in runs.rollouts(run_id)] == ["永", "國", "心"]
    for f in (tmp_path / run_id).iterdir():
        assert SECRET not in f.read_text()
    assert runs.list()[0]["id"] == run_id


@needs_data
def test_failures_score_zero(tmp_path, monkeypatch):
    answers = iter([Completion("I cannot draw."), Completion("", error="HTTP 500: boom")])
    monkeypatch.setattr(runner_mod, "complete", lambda c, m: next(answers))
    runs = RunManager(tmp_path)
    run_id = runs.start(stub_config(), "custom", "永國", concurrency=1)
    runs.wait(run_id)
    rows = runs.rollouts(run_id)
    assert [r["reward"] for r in rows] == [0.0, 0.0]
    assert not rows[0]["info"]["parse_ok"] and rows[0]["error"] is None
    assert rows[1]["error"] == "HTTP 500: boom"
    assert runs.summary(run_id)["n_errors"] == 1


def test_openai_client_sends_key_and_reads_answer(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}], "usage": {"total_tokens": 3}})

    real = httpx.Client
    monkeypatch.setattr(client_mod.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    c = complete(stub_config(api_key=SECRET, temperature=0.5), [{"role": "user", "content": "hi"}])
    assert c.error is None and c.text == "hello" and c.stop_reason == "stop"
    assert seen["auth"] == f"Bearer {SECRET}"
    assert seen["body"]["model"] == "converter" and seen["body"]["temperature"] == 0.5


def test_openai_client_sends_provider_options(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    mock_http(monkeypatch, handler)
    complete(stub_config(extra_body={"reasoning": {"enabled": False}}), [{"role": "user", "content": "hi"}])
    assert seen["reasoning"] == {"enabled": False}


def mock_http(monkeypatch, handler) -> None:
    real = httpx.Client
    monkeypatch.setattr(client_mod.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_openai_client_keeps_reasoning_apart_from_the_answer(monkeypatch):
    message = {"content": '{"strokes": []}', "reasoning": "first the dot, then the hook"}
    mock_http(monkeypatch, lambda r: httpx.Response(200, json={"choices": [{"message": message}]}))
    c = complete(stub_config(), [{"role": "user", "content": "hi"}])
    assert c.text == '{"strokes": []}' and c.reasoning == "first the dot, then the hook"


def test_openai_client_waits_out_transient_429(monkeypatch):
    replies = iter([httpx.Response(429, json={"error": "temporarily rate-limited upstream"}), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})])
    mock_http(monkeypatch, lambda r: next(replies))
    slept = []
    monkeypatch.setattr(client_mod.time, "sleep", slept.append)
    c = complete(stub_config(), [{"role": "user", "content": "hi"}])
    assert c.text == "ok" and c.error is None and slept == [client_mod.RATE_LIMIT_BACKOFF_S[0]]


def test_daily_cap_is_reported_at_once(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded: free-models-per-day"}})

    mock_http(monkeypatch, handler)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: pytest.fail("a daily cap must not be waited on"))
    c = complete(stub_config(), [{"role": "user", "content": "hi"}])
    assert c.rate_limited and "429" in c.error and len(calls) == 1


@needs_data
def test_rate_limit_stops_the_run_and_resume_finishes_it(tmp_path, monkeypatch):
    calls = iter(["answer", "limited", "answer", "answer"])

    def fake(config, messages):
        if next(calls) == "limited":
            return Completion("", error="rate limited (HTTP 429): per-day", rate_limited=True)
        return converter_answer(config, messages)

    monkeypatch.setattr(runner_mod, "complete", fake)
    runs = RunManager(tmp_path)
    run_id = runs.start(stub_config(), "custom", "永國心", concurrency=1)
    runs.wait(run_id)
    s = runs.summary(run_id)
    assert s["status"] == "rate_limited" and s["n_done"] == 1 and s["n_errors"] == 0
    assert [r["character"] for r in runs.rollouts(run_id)] == ["永"]  # the 429 is not scored as a zero

    assert runs.resume(run_id, stub_config()) == [1, 2]
    runs.wait(run_id)
    s = runs.summary(run_id)
    assert s["status"] == "done" and s["n_done"] == 3 and s["resumed"] == 1 and s["mean_reward"] > 0.85
    assert [r["character"] for r in runs.rollouts(run_id)] == ["永", "國", "心"]


def test_load_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# provider keys\nexport TEST_ENV_A="quoted value"\nTEST_ENV_B=plain\nTEST_ENV_C=from-file\nnot a line\n')
    monkeypatch.delenv("TEST_ENV_A", raising=False)
    monkeypatch.delenv("TEST_ENV_B", raising=False)
    monkeypatch.setenv("TEST_ENV_C", "from-shell")
    assert client_mod.load_env_file(env) == ["TEST_ENV_A", "TEST_ENV_B"]
    assert client_mod.os.environ["TEST_ENV_A"] == "quoted value" and client_mod.os.environ["TEST_ENV_B"] == "plain"
    assert client_mod.os.environ["TEST_ENV_C"] == "from-shell"  # the shell wins over the file
    assert client_mod.load_env_file(tmp_path / "missing") == []  # monkeypatch removes A and B again on teardown


def test_openai_client_retries_with_max_completion_tokens(monkeypatch):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": "use max_completion_tokens instead"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    real = httpx.Client
    monkeypatch.setattr(client_mod.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert complete(stub_config(), [{"role": "user", "content": "hi"}]).text == "ok"
    assert "max_completion_tokens" in bodies[-1]


def test_connection_errors_are_reported_not_raised():
    c = complete(ModelConfig(provider="x", kind="openai", base_url="http://127.0.0.1:9/v1", model="m", timeout=2), [{"role": "user", "content": "hi"}])
    assert c.text == "" and "could not connect" in c.error


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "from-env")
    cfg = stub_config(key_env="TEST_PROVIDER_KEY")
    assert cfg.resolved_key() == "from-env" and cfg.key_source() == "env TEST_PROVIDER_KEY"
    assert SECRET not in repr(stub_config(api_key=SECRET))


# --- tools mode: multi-turn episodes and their traces ---------------------------


def use_stub_server(monkeypatch) -> None:
    """Send the HTTP client's requests to the scripted stub model (scripts/stub_model.py) in-process."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import stub_model

    monkeypatch.setattr(client_mod.httpx, "Client", lambda **kw: TestClient(stub_model.app))


@needs_data
def test_tools_run_records_the_whole_episode(tmp_path, monkeypatch):
    """Runner -> HTTP tool-call protocol -> stub agent that draws, looks, fixes one stroke, looks, submits."""
    use_stub_server(monkeypatch)
    runs = RunManager(tmp_path)
    run_id = runs.start(stub_config(api_key=SECRET), "custom", "永國", concurrency=2, mode="tools", obs_size=160)
    runs.wait(run_id)

    s = runs.summary(run_id)
    assert s["status"] == "done" and s["mode"] == "tools" and s["obs_size"] == 160 and s["n_parse_ok"] == 2
    assert s["episode"] == {"max_turns": 64, "max_tool_calls": 128, "max_nudges": 2, "score_steps": True}
    assert s["mean_reward"] > 0.85  # the fixed drawing scores like the converter's answer

    row = runs.rollouts(run_id)[0]
    yong = get_entry("永")
    assert row["character"] == "永" and row["mode"] == "tools" and row["error"] is None and row["trace"] == "traces/0000.json"
    assert parse_response(row["response"]) == convert(yong)  # the final canvas, in the phase-1 answer format
    t = row["trace_summary"]
    assert (t["end_reason"], t["n_turns"], t["n_tool_calls"], t["n_captures"], t["n_deletes"], t["n_redraws"]) == ("submit", 3, 10, 2, 1, 1)
    assert t["draft_score"] < t["final_score"] == pytest.approx(row["reward"], abs=1e-4) and t["checked_final"]

    trace = runs.trace(run_id, 0)
    assert trace["character"] == "永" and trace["model"] == "converter" and trace["reward"] == row["reward"]
    assert "Write the character 永" in trace["user"] and "delete_stroke" in trace["system"]
    assert [x["function"]["name"] for x in trace["tools"]] == ["create_stroke", "delete_stroke", "capture_canvas", "submit"]
    assert trace["canvas"]["obs_size"] == 160 and trace["summary"] == t and trace["final_strokes"] == convert(yong)
    kinds = [(e["type"], e.get("name")) for e in trace["events"]]
    assert kinds == (
        [("model", None)] + [("tool", "create_stroke")] * 5 + [("tool", "capture_canvas")]
        + [("model", None), ("tool", "delete_stroke"), ("tool", "create_stroke"), ("tool", "capture_canvas")]
        + [("model", None), ("tool", "submit"), ("end", None)]
    )  # fmt: skip
    first_reply = trace["events"][0]
    assert first_reply["stop_reason"] == "tool_calls" and first_reply["n_tool_calls"] == 6 and first_reply["turn"] == 1
    calls = [e for e in trace["events"] if e["type"] == "tool"]
    assert calls[0]["call_id"] == "call_0_0" and calls[6]["arguments"] == {"stroke_id": 2}
    for e in calls:
        if e["name"] == "capture_canvas":
            assert runs.trace_image(run_id, 0, e["image"]).stat().st_size > 0
    for f in (tmp_path / run_id).rglob("*.json*"):
        assert SECRET not in f.read_text()
    with pytest.raises(KeyError):
        runs.trace(run_id, 7)
    with pytest.raises(KeyError):
        runs.trace_image(run_id, 0, "../summary.json")


def scripted(replies):
    """A fake `complete` that plays `replies` in order: each is a Completion or a list of (tool, arguments)."""
    it = iter(replies)

    def fake(config, messages, tools=None):
        assert tools and messages[0]["role"] == "system"
        reply = next(it)
        if isinstance(reply, Completion):
            return reply
        calls = [{"id": f"c{k}", "name": name, "arguments": json.dumps(args)} for k, (name, args) in enumerate(reply)]
        return Completion("", stop_reason="tool_calls", tool_calls=calls, message={"role": "assistant", "content": ""})

    return fake


@needs_data
def test_tools_run_nudges_then_scores_what_is_on_the_canvas(tmp_path, monkeypatch):
    stroke = convert(get_entry("永"))[0]
    talk = Completion("I will start with the dot.", stop_reason="stop", message={"role": "assistant", "content": "I will start with the dot."})
    seen = []

    def fake(config, messages, tools=None):
        seen.append([m["role"] for m in messages])
        return play(config, messages, tools)

    play = scripted([talk, [("create_stroke", stroke), ("capture_canvas", {})], talk, talk])
    monkeypatch.setattr(runner_mod, "complete", fake)
    runs = RunManager(tmp_path)
    run_id = runs.start(stub_config(), "custom", "永", mode="tools")
    runs.wait(run_id)

    row = runs.rollouts(run_id)[0]
    t = row["trace_summary"]
    assert (t["end_reason"], t["n_turns"], t["n_nudges"], t["n_strokes"]) == ("no_tool_call", 4, 2, 1)
    assert 0 < row["reward"] < 0.3 and row["info"]["parse_ok"]  # one good stroke of five
    # Transcript: tool results first, then the canvas image as a user message, then the nudges.
    assert seen[0] == ["system", "user"]
    assert seen[1] == ["system", "user", "assistant", "user"]
    assert seen[2] == ["system", "user", "assistant", "user", "assistant", "tool", "tool", "user"]
    assert seen[3][-2:] == ["assistant", "user"]
    types = [e["type"] for e in runs.trace(run_id, 0)["events"]]
    assert types == ["model", "nudge", "model", "tool", "tool", "model", "nudge", "model", "end"]


@needs_data
def test_tools_run_stops_at_the_turn_limit_and_keeps_errors_for_resume(tmp_path, monkeypatch):
    from calligraphy.episode import EpisodeConfig

    stroke = convert(get_entry("永"))[0]
    monkeypatch.setattr(runner_mod, "complete", scripted([[("create_stroke", stroke)], [("capture_canvas", {})], Completion("", error="HTTP 500: boom")]))
    runs = RunManager(tmp_path)
    run_id = runs.start(stub_config(), "custom", "永國", concurrency=1, mode="tools", episode=EpisodeConfig(max_turns=2))
    runs.wait(run_id)
    capped, failed = runs.rollouts(run_id)
    assert capped["trace_summary"]["end_reason"] == "max_turns" and capped["reward"] > 0 and capped["error"] is None
    assert failed["trace_summary"]["end_reason"] == "error" and failed["reward"] == 0.0 and failed["error"] == "HTTP 500: boom"
    assert runs.trace(run_id, 1)["events"][0]["error"] == "HTTP 500: boom"  # the failed attempt's trace is kept for debugging
    assert runs.summary(run_id)["n_errors"] == 1

    # Resume asks the failed item again, in tools mode, and replaces its trace.
    monkeypatch.setattr(runner_mod, "complete", scripted([[("create_stroke", stroke), ("submit", {})]]))
    assert runs.resume(run_id, stub_config()) == [1]
    runs.wait(run_id)
    again = runs.rollouts(run_id)[1]
    assert again["error"] is None and again["trace_summary"]["end_reason"] == "submit"
    assert runs.trace(run_id, 1)["summary"]["end_reason"] == "submit" and runs.summary(run_id)["status"] == "done"


@needs_data
def test_tools_run_rate_limit_mid_episode_writes_nothing(tmp_path, monkeypatch):
    stroke = convert(get_entry("永"))[0]
    limited = Completion("", error="rate limited (HTTP 429): per-day", rate_limited=True)
    monkeypatch.setattr(runner_mod, "complete", scripted([[("create_stroke", stroke), ("capture_canvas", {})], limited]))
    runs = RunManager(tmp_path)
    run_id = runs.start(stub_config(), "custom", "永", mode="tools")
    runs.wait(run_id)
    assert runs.summary(run_id)["status"] == "rate_limited" and runs.rollouts(run_id) == []
    assert not (tmp_path / run_id / "traces" / "0000.json").exists()

    monkeypatch.setattr(runner_mod, "complete", scripted([[("create_stroke", stroke), ("submit", {})]]))
    runs.resume(run_id, stub_config())
    runs.wait(run_id)
    assert runs.rollouts(run_id)[0]["trace_summary"]["n_captures"] == 0
    assert list((tmp_path / run_id / "traces").glob("0000/*.png")) == []  # the unfinished attempt's images are gone


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        RunManager(tmp_path).start(stub_config(), "custom", "永", mode="freehand")


def test_run_in_another_live_process_is_not_called_interrupted(tmp_path):
    import os

    run_dir = tmp_path / "20260101-000000-m"
    run_dir.mkdir()
    summary = {"id": run_dir.name, "status": "running", "pid": os.getppid()}
    (run_dir / "summary.json").write_text(json.dumps(summary))
    assert RunManager(tmp_path).summary(run_dir.name)["status"] == "running"  # e.g. started from the command line
    (run_dir / "summary.json").write_text(json.dumps({**summary, "pid": 2**22 + 12345}))
    assert RunManager(tmp_path).summary(run_dir.name)["status"] == "interrupted"
    (run_dir / "summary.json").write_text(json.dumps({"id": run_dir.name, "status": "running"}))
    assert RunManager(tmp_path).summary(run_dir.name)["status"] == "interrupted"  # runs saved before pids were recorded


def test_openai_client_reads_tool_calls_and_keeps_private_keys_local(monkeypatch):
    from calligraphy.agent.client import image_message, tool_message

    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        calls = [
            {"id": "call_a", "type": "function", "function": {"name": "create_stroke", "arguments": '{"points": [[1, 2, 0.5]], "width": 9}'}},
            {"type": "function", "function": {"name": "submit", "arguments": {}}},  # no id, arguments as an object
        ]
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": calls, "reasoning": "hm"}, "finish_reason": "tool_calls"}]})

    mock_http(monkeypatch, handler)
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "capture_canvas", "arguments": "{}"}}]},
        tool_message("c0", "error: nope", ok=False),
        image_message("the canvas", b"PNG"),
    ]
    tools = [{"type": "function", "function": {"name": "submit", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    c = complete(stub_config(), history, tools)
    assert seen["tools"] == tools
    assert seen["messages"][2] == {"role": "tool", "tool_call_id": "c0", "content": "error: nope"}  # no "_is_error" on the wire
    assert seen["messages"][3]["content"][1]["image_url"]["url"] == "data:image/png;base64,UE5H"
    assert c.error is None and c.stop_reason == "tool_calls" and c.reasoning == "hm"
    assert c.tool_calls[0] == {"id": "call_a", "name": "create_stroke", "arguments": '{"points": [[1, 2, 0.5]], "width": 9}'}
    assert c.tool_calls[1]["name"] == "submit" and c.tool_calls[1]["arguments"] == "{}" and c.tool_calls[1]["id"].startswith("call_")
    assert c.message["role"] == "assistant" and [t["id"] for t in c.message["tool_calls"]] == [t["id"] for t in c.tool_calls]


def test_transcript_converts_to_anthropic_turns():
    from calligraphy.agent.client import _anthropic_messages, _anthropic_tools, image_message, tool_message

    blocks = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "tool_use", "id": "t1", "name": "capture_canvas", "input": {}}]
    transcript = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "Write the character 永."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t0", "type": "function", "function": {"name": "create_stroke", "arguments": '{"width": 9}'}}]},
        tool_message("t0", "error: points must be a non-empty list", ok=False),
        {"role": "assistant", "content": "", "_anthropic_content": blocks},
        tool_message("t1", "canvas image with 0 strokes"),
        image_message("Image for your capture_canvas call.", b"PNG"),
    ]
    turns = _anthropic_messages(transcript)
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert turns[1]["content"] == [{"type": "tool_use", "id": "t0", "name": "create_stroke", "input": {"width": 9}}]
    assert turns[2]["content"] == [{"type": "tool_result", "tool_use_id": "t0", "content": "error: points must be a non-empty list", "is_error": True}]
    assert turns[3]["content"] == blocks  # the API's own blocks go back unchanged, thinking included
    # One user turn answers the tool call: the result first, then the image.
    assert turns[4]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "canvas image with 0 strokes"},
        {"type": "text", "text": "Image for your capture_canvas call."},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "UE5H"}},
    ]
    tools = _anthropic_tools([{"type": "function", "function": {"name": "submit", "description": "Finish.", "parameters": {"type": "object", "properties": {}}}}])
    assert tools == [{"name": "submit", "description": "Finish.", "input_schema": {"type": "object", "properties": {}}}]


# --- viewer API ----------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    import server.app as app_mod

    monkeypatch.setattr(app_mod, "RUNS", RunManager(tmp_path))
    app_mod.drawing.cache_clear()
    return TestClient(app_mod.app), app_mod


@needs_data
def test_providers_report_env_presence_not_values(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    body = c.get("/api/agent/providers").text
    assert SECRET not in body
    openai = next(p for p in json.loads(body) if p["id"] == "openai")
    assert openai["key_env_set"] is True and openai["steps"]


@needs_data
def test_run_through_api_and_open_answers(client, monkeypatch):
    c, app_mod = client
    answers = {"永": None, "國": "not json"}

    def fake(config, messages):
        char = messages[-1]["content"].split("Write the character ")[1][0]
        if answers[char]:
            return Completion(answers[char])
        return converter_answer(config, messages)

    monkeypatch.setattr(runner_mod, "complete", fake)
    run_id = c.post("/api/agent/runs", json={"provider": "stub", "model": "converter", "dataset": "custom", "characters": "永國", "api_key": SECRET}).json()["run_id"]
    app_mod.RUNS.wait(run_id)
    detail = c.get(f"/api/agent/runs/{run_id}").json()
    assert detail["summary"]["status"] == "done" and len(detail["rollouts"]) == 2
    assert SECRET not in json.dumps(detail)

    good = c.get("/api/char/永", params={"drawing": f"run:{run_id}:0"}).json()
    assert good["rollout"]["parse_ok"] and good["levels"]["structure"] > 0.85
    assert good["drawings"][-1]["name"] == f"run:{run_id}:0"

    bad = c.get("/api/char/國", params={"drawing": f"run:{run_id}:1"}).json()
    assert bad["strokes"] == [] and bad["levels"]["structure"] == 0.0
    assert bad["levels"]["placement_offset"] is None  # no ink: infinity is sent as null
    assert c.get("/api/char/國/render.png", params={"drawing": f"run:{run_id}:1"}).status_code == 200
    assert c.get("/api/char/國", params={"drawing": f"run:{run_id}:9"}).status_code == 404


@needs_data
def test_tools_run_through_api_serves_trace_and_images(client, monkeypatch):
    c, app_mod = client
    use_stub_server(monkeypatch)
    assert [m["id"] for m in c.get("/api/agent/modes").json()] == ["single", "tools"]
    body = {"provider": "stub", "model": "converter", "dataset": "custom", "characters": "永", "mode": "tools", "max_turns": 8, "obs_size": 128}
    run_id = c.post("/api/agent/runs", json=body).json()["run_id"]
    app_mod.RUNS.wait(run_id)

    detail = c.get(f"/api/agent/runs/{run_id}").json()
    assert detail["summary"]["mode"] == "tools" and detail["summary"]["episode"]["max_turns"] == 8
    assert detail["rollouts"][0]["trace_summary"]["n_deletes"] == 1

    trace = c.get(f"/api/agent/runs/{run_id}/trace/0").json()
    capture = next(e for e in trace["events"] if e.get("name") == "capture_canvas")
    image = c.get(f"/api/agent/runs/{run_id}/trace/0/image/{capture['image']}")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png" and image.content[:4] == b"\x89PNG"
    assert c.get(f"/api/agent/runs/{run_id}/trace/5").status_code == 404
    assert c.get(f"/api/agent/runs/{run_id}/trace/0/image/step-999.png").status_code == 404
    assert c.get(f"/api/agent/runs/{run_id}/trace/0/image/summary.json").status_code == 404

    # The canvas before and after any call is rebuilt from the trace.
    def state(step, **params):
        r = c.get(f"/api/agent/runs/{run_id}/trace/0/state", params={"step": step, "size": 128, **params})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        return PILImage.open(io.BytesIO(r.content)).convert("RGB")

    def ink(img):  # dark pixels
        return int((np.asarray(img).sum(axis=2) < 200).sum())

    delete = next(e for e in trace["events"] if e.get("name") == "delete_stroke")
    empty, before, after = state(-1), state(delete["i"] - 1), state(delete["i"])
    assert ink(empty) == 0 and ink(after) < ink(before)  # the deleted stroke's ink is gone
    assert np.array_equal(np.asarray(state(capture["i"])), np.asarray(state(capture["i"] - 1)))  # looking changes nothing
    shown = PILImage.open(io.BytesIO(image.content)).convert("RGB")
    assert np.array_equal(np.asarray(shown), np.asarray(state(capture["i"])))  # the rebuilt canvas is what the model saw
    marked = state(delete["i"] - 1, mark=2, tone="removed")
    reds = (np.asarray(marked) == np.array([200, 60, 45])).all(axis=2).sum()
    assert reds > 20 and ink(marked) < ink(before)  # stroke 2 is drawn in the marker colour, not in ink
    assert c.get(f"/api/agent/runs/{run_id}/trace/0/state", params={"step": 3, "tone": "sparkly"}).status_code == 422
    assert c.get(f"/api/agent/runs/{run_id}/trace/9/state").status_code == 404
    answer = c.get("/api/char/永", params={"drawing": f"run:{run_id}:0"}).json()["rollout"]
    assert answer["mode"] == "tools" and answer["trace_summary"]["n_tool_calls"] == 10

    # The final canvas opens in the viewer like any model answer.
    shown = c.get("/api/char/永", params={"drawing": f"run:{run_id}:0"}).json()
    assert shown["rollout"]["parse_ok"] and shown["levels"]["structure"] > 0.85
    assert c.post("/api/agent/runs", json={**body, "mode": "freehand"}).status_code == 400


@needs_data
def test_bad_run_requests(client):
    c, _ = client
    assert c.post("/api/agent/runs", json={"provider": "nope", "model": "m"}).status_code == 400
    assert c.post("/api/agent/runs", json={"provider": "stub", "model": ""}).status_code == 422
    assert c.post("/api/agent/runs", json={"provider": "stub", "model": "m", "dataset": "custom", "characters": "abc"}).status_code == 400


@needs_data
def test_normalization_endpoint(client):
    c, _ = client
    cases = c.get("/api/normalization").json()
    assert len(cases) == 4 * 12
    first = cases[0]
    assert first["key"] == "wxz_6420" and first["drawing"] == "converter" and first["category"] == "baseline"
    assert {x["category"] for x in cases} == {"baseline", "size", "position", "shape"}
