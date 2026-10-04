import numpy as np
from conftest import needs_data

from calligraphy.canvas import CanvasConfig
from calligraphy.convert import DEFAULT, NAIVE, convert, downsample, variant
from calligraphy.geom import distance_to_boundary, inscribed_radius, points_in_polygon, ray_hit_distance
from calligraphy.roundtrip import roundtrip
from calligraphy.tools import format_strokes, parse_response

SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)


def test_polygon_queries():
    pts = np.array([[50, 50], [10, 50], [150, 50]])
    assert points_in_polygon(pts, SQUARE).tolist() == [True, True, False]
    np.testing.assert_allclose(distance_to_boundary(pts, SQUARE), [50, 10, 50])
    np.testing.assert_allclose(inscribed_radius(pts, SQUARE), [50, 10, 0])
    np.testing.assert_allclose(ray_hit_distance([[50, 50]], [[1, 0]], SQUARE), [50])


def test_downsample_keeps_endpoints_and_cap():
    t = np.linspace(0, 2 * np.pi, 200)
    s = np.column_stack([t * 50, np.sin(t) * 80, 10 + 5 * np.cos(t)])
    d = downsample(s, 16)
    assert len(d) <= 16
    np.testing.assert_array_equal(d[0], s[0])
    np.testing.assert_array_equal(d[-1], s[-1])


@needs_data
def test_guan_output_is_valid_phase1_response(guan):
    strokes = convert(guan)
    assert len(strokes) == guan.n_strokes == 25
    assert all(len(s["points"]) <= 16 for s in strokes)
    assert parse_response(format_strokes(strokes), CanvasConfig()) == strokes
    # One width for the whole character: 2 * r_max (v2 reports r_max = 44 for 觀).
    assert len({s["width"] for s in strokes}) == 1
    assert 80 <= strokes[0]["width"] <= 96
    assert max(max(p[2] for p in s["points"]) for s in strokes) == 1.0


@needs_data
def test_guan_stroke1_close_to_v2_example(guan):
    # v2: lands near (245, 110), lifts near (308, 255), pressure peaks mid-stroke.
    pts = np.array(convert(guan)[0]["points"], float)
    start = pts[np.argmin(np.linalg.norm(pts[:, :2] - [245, 110], axis=1))]
    end = pts[np.argmin(np.linalg.norm(pts[:, :2] - [308, 255], axis=1))]
    assert np.linalg.norm(start[:2] - [245, 110]) < 15
    assert np.linalg.norm(end[:2] - [308, 255]) < 15
    assert pts[:, 1].argmin() < pts[:, 1].argmax()  # written top to bottom
    assert 0 < pts[:, 2].argmax() < len(pts) - 1


@needs_data
def test_fixed_beats_naive_on_guan(guan):
    naive = roundtrip(guan, NAIVE)
    fixed = roundtrip(guan, DEFAULT)
    assert np.mean(naive["stroke_iou"]) > 0.85  # reproduces v2's validation (~0.89)
    assert np.mean(fixed["stroke_iou"]) > np.mean(naive["stroke_iou"])
    assert fixed["char_iou"] > naive["char_iou"]
    # Stroke 9 (median hugs one edge) is where the centerline fix matters most.
    assert fixed["stroke_iou"][8] > naive["stroke_iou"][8]


@needs_data
def test_action_frame_64(guan):
    strokes = convert(guan, action_frame=64)
    xs = [p[0] for s in strokes for p in s["points"]]
    assert max(xs) <= 64
    assert 5 <= strokes[0]["width"] <= 6


@needs_data
def test_centerline_variant_alone_helps(guan):
    dense = variant(max_points=None, quantize=False)
    no_center = variant(recenter=False, max_points=None, quantize=False)
    assert np.mean(roundtrip(guan, dense)["stroke_iou"]) > np.mean(roundtrip(guan, no_center)["stroke_iou"])
