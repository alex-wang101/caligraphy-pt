"""Agent runs: dataset selection, the runner, the HTTP client, and key handling."""

import json

import httpx
import pytest
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
