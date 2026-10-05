import hashlib
import io

import numpy as np
import pytest

from calligraphy.canvas import Canvas, CanvasConfig, StrokeError, format_ids
from calligraphy.frame import catmull_rom, resample_spacing
from calligraphy.render.brushes import DiscBrush, FreehandBrush
from calligraphy.render.raster import mask_iou, rasterize_mask
from calligraphy.tools import TOOLS, ParseError, dispatch, format_strokes, parse_response, phase2_messages, tool_schemas

STROKES = [
    {"points": [[200, 300, 0.5], [500, 320, 1.0], [800, 300, 0.4]], "width": 60},
    {"points": [[512, 150, 0.6], [515, 500, 1.0], [505, 880, 0.3]], "width": 60},
]


def png_hash(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def test_catmull_rom_passes_through_points():
    pts = np.array([[0, 0, 0.2], [100, 50, 1.0], [200, 0, 0.5], [300, 80, 0.1]], float)
    dense = catmull_rom(pts, spacing=8)
    for p in pts:
        assert np.min(np.linalg.norm(dense - p, axis=1)) < 1e-6
    assert np.all(np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1) <= 8 * 1.6)


def test_resample_spacing_is_fixed_not_fixed_count():
    short = resample_spacing([[0, 0], [16, 0]], 8)
    long = resample_spacing([[0, 0], [400, 0]], 8)
    assert len(short) == 3 and len(long) == 51


def test_straight_disc_stroke_area():
    # Full pressure line: area ~ L * w + pi * (w/2)^2.
    L, w = 600.0, 80.0
    geom = DiscBrush().geometry([[200, 512, 1], [200 + L, 512, 1]], w)
    mask = rasterize_mask([geom], size=1024)
    expected = L * w + np.pi * (w / 2) ** 2
    assert mask.sum() == pytest.approx(expected, rel=0.02)


def test_pressure_scales_radius():
    thin = rasterize_mask([DiscBrush().geometry([[200, 512, 0.25], [800, 512, 0.25]], 80)], size=1024).sum()
    thick = rasterize_mask([DiscBrush().geometry([[200, 512, 1], [800, 512, 1]], 80)], size=1024).sum()
    assert thin / thick == pytest.approx(0.25, rel=0.1)


def test_freehand_brush_renders_similar_ink():
    pts, w = STROKES[0]["points"], STROKES[0]["width"]
    disc = rasterize_mask([DiscBrush().geometry(pts, w)])
    free = rasterize_mask([FreehandBrush().geometry(pts, w)])
    assert free.sum() > 0
    assert mask_iou(disc, free) > 0.6


def test_render_is_deterministic():
    hashes = set()
    for _ in range(2):
        c = Canvas(CanvasConfig(obs_size=224))
        for s in STROKES:
            c.create_stroke(s["points"], s["width"])
        hashes.add(png_hash(c.capture()))
    assert len(hashes) == 1


def test_incremental_equals_full_render():
    cfg = CanvasConfig(obs_size=256)
    c = Canvas(cfg)
    for s in STROKES:
        c.create_stroke(s["points"], s["width"])
        assert png_hash(c.capture()) == png_hash(c.render(256))


def test_action_frame_scales_points_and_width():
    c = Canvas(CanvasConfig(action_frame=64))
    s = c.create_stroke([[16, 32, 1.0], [48, 32, 0.5]], 4)
    np.testing.assert_allclose(s.points[:, :2], [[256, 512], [768, 512]])
    assert s.width == 64


def test_clamps_out_of_range_values():
    c = Canvas()
    s = c.create_stroke([[-50, 2000, 1.7], [500, 500, -1]], 9999)
    assert s.points[:, :2].min() >= 0 and s.points[:, :2].max() <= 1024
    assert s.points[:, 2].tolist() == [1.0, 0.0]
    assert s.width == c.config.max_width


@pytest.mark.parametrize(
    "points,width",
    [([], 10), ([[1, 2]], 10), ([[1, 2, "a"]], 10), ([[1, 2, float("nan")]], 10), ([[1, 2, 1]], 0), ([[i, i, 1] for i in range(17)], 10)],
)
def test_rejects_malformed_strokes(points, width):
    with pytest.raises(StrokeError):
        Canvas().create_stroke(points, width)


def test_dispatch_returns_text_for_strokes_and_image_for_capture():
    c = Canvas()
    r = dispatch(c, "create_stroke", STROKES[0])
    assert r.image is None and "created" in r.text
    assert dispatch(c, "create_stroke", {"points": [], "width": 1}).text.startswith("error")
    cap = dispatch(c, "capture_canvas", {})
    assert cap.image is not None and cap.image.size == (224, 224)
    assert dispatch(c, "submit", {}).done
    assert not dispatch(c, "erase_all", {}).ok and not dispatch(c, "create_stroke", [1, 2]).ok


def test_every_tool_is_defined_once_and_reaches_the_model():
    """Schemas, the system prompt and dispatch are all built from TOOLS."""
    cfg = CanvasConfig(action_frame=64, max_points=12)
    assert list(TOOLS) == ["create_stroke", "delete_stroke", "capture_canvas", "submit"]
    schemas = tool_schemas(cfg)
    assert [t["function"]["name"] for t in schemas] == list(TOOLS)
    assert schemas[0]["function"]["parameters"]["properties"]["points"]["maxItems"] == 12
    assert "64x64" in schemas[0]["function"]["description"] and "at most 12 points" in schemas[0]["function"]["description"]
    system = phase2_messages("永", "yǒng", cfg)[0]["content"]
    for tool in TOOLS.values():
        assert f"- {tool.name}: {tool.describe(cfg)}" in system


def test_delete_keeps_ids_and_redraws_in_place():
    extra = {"points": [[100, 800, 1.0], [900, 820, 1.0]], "width": 40}
    c = Canvas(CanvasConfig(obs_size=128))
    for s in [*STROKES, extra]:
        c.create_stroke(s["points"], s["width"])
    first, _, third = c.strokes
    assert c.stroke_ids == [1, 2, 3] and c.next_id() == 4

    c.delete_stroke(2)
    assert c.stroke_ids == [1, 3] and c.strokes == [first, third]  # the others keep their numbers
    assert c.next_id() == 2  # the next stroke takes the deleted one's place
    assert png_hash(c.capture()) == png_hash(c.render(128))  # the deleted ink is gone from the image

    redrawn = c.create_stroke(STROKES[1]["points"], STROKES[1]["width"])
    assert c.stroke_ids == [1, 2, 3] and c.strokes == [first, redrawn, third]
    assert png_hash(c.capture()) == png_hash(c.render(128))

    c.delete_stroke(3)  # deleting the last stroke frees its number too
    assert c.stroke_ids == [1, 2] and c.next_id() == 3
    c.delete_stroke(2.0)  # models often send whole numbers as floats
    assert c.stroke_ids == [1]


@pytest.mark.parametrize("stroke_id", [0, 3, -1, "1", None, 1.5, True])
def test_delete_rejects_missing_or_malformed_ids(stroke_id):
    c = Canvas()
    for s in STROKES:
        c.create_stroke(s["points"], s["width"])
    with pytest.raises(StrokeError):
        c.delete_stroke(stroke_id)
    assert c.stroke_ids == [1, 2]


def test_dispatch_delete_reports_what_remains():
    c = Canvas()
    for s in STROKES * 2:
        dispatch(c, "create_stroke", s)
    r = dispatch(c, "delete_stroke", {"stroke_id": 2})
    assert r.ok and r.stroke_id == 2 and "strokes on the canvas: 1, 3-4" in r.text and "becomes stroke 2" in r.text
    again = dispatch(c, "delete_stroke", {"stroke_id": 2})
    assert not again.ok and again.text.startswith("error: no stroke 2")
    created = dispatch(c, "create_stroke", STROKES[1])
    assert created.ok and created.stroke_id == 2 and c.stroke_ids == [1, 2, 3, 4]


def test_format_ids():
    assert format_ids([]) == "none"
    assert format_ids([4]) == "4"
    assert format_ids([1, 2, 4, 5, 6, 9]) == "1-2, 4-6, 9"


def test_parse_response_roundtrip_and_fences():
    text = format_strokes(STROKES)
    assert parse_response(text) == STROKES
    assert parse_response(f"Here you go:\n```json\n{text}\n```") == STROKES


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"strokes": []}',
        '{"lines": [1]}',
        '{"strokes": [{"points": [[1, 2]], "width": 5}]}',
        '{"strokes": [{"points": [[1, 2, 1]], "width": -5}]}',
        '{"strokes": [{"points": ' + str([[i, i, 1] for i in range(17)]) + ', "width": 5}]}',
    ],
)
def test_parse_response_rejects(text):
    with pytest.raises(ParseError):
        parse_response(text)
