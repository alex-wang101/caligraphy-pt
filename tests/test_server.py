import pytest
from conftest import needs_data

fastapi = pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from server.app import app

    return TestClient(app)


@needs_data
def test_tasks_and_char(client):
    tasks = client.get("/api/tasks").json()
    assert len(tasks) == 88
    d = client.get("/api/char/wxz_6420").json()
    assert d["character"] == "觀" and d["n_strokes"] == 25 and d["has_reference"]
    assert d["medians"][0][0] == [245, 110] and d["medians"][0][-1] == [308, 255]
    assert len(d["strokes"]) == len(d["stroke_metrics"]) == 25
    assert d["levels"]["structure"] > 0.85


@needs_data
def test_scene_and_mmah_lookup(client):
    b = client.get("/api/char/wxz_6420", params={"drawing": "B_slid_component"}).json()
    assert b["levels"]["structure"] < 0.75
    assert client.get("/api/char/永").json()["has_reference"] is False
    assert client.get("/api/char/nope").status_code == 404
    assert [h["character"] for h in client.get("/api/search", params={"q": "永x字"}).json()] == ["永", "字"]


@needs_data
def test_images(client):
    r = client.get("/api/char/wxz_6420/render.png", params={"upto": 3, "size": 224})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/char/wxz_6420/reference.jpg").headers["content-type"] == "image/jpeg"
    assert client.get("/api/char/永/reference.jpg").status_code == 404
    ok = client.post("/api/render", json={"strokes": [{"points": [[100, 100, 1], [900, 900, 0.5]], "width": 50}]})
    assert ok.status_code == 200
    bad = client.post("/api/render", json={"strokes": [{"points": [[1, 2]], "width": 5}]})
    assert bad.status_code == 400


def test_unknown_api_route_is_a_json_404_not_the_page(tmp_path):
    """With a built frontend, the page catch-all must not swallow unknown /api/ routes."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.app import mount_frontend

    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>viewer</title>")
    app = FastAPI()
    mount_frontend(app, tmp_path)
    c = TestClient(app)
    missing = c.get("/api/does-not-exist")
    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("application/json")
    assert c.get("/some/page").text.startswith("<!doctype html>")
