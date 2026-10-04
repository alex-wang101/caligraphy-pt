import numpy as np
import pytest
from conftest import needs_data

from calligraphy.frame import MMAH_Y_FLIP, mmah_to_canvas
from calligraphy.svgpath import mmah_outline, parse_path, path_to_d, path_to_polygon, transform_path


def test_parse_all_commands():
    cmds = parse_path("M 0 0 L 10 0 Q 20 0 20 10 C 20 20 10 30 0 30 Z")
    assert [c for c, _ in cmds] == ["M", "L", "Q", "C", "Z"]
    assert cmds[3][1].shape == (3, 2)


def test_rejects_unknown_command_and_bad_arity():
    with pytest.raises(ValueError):
        parse_path("M 0 0 A 1 1 0 0 1 5 5 Z")
    with pytest.raises(ValueError):
        parse_path("M 0 0 Q 1 2 3 Z")


def test_sample_counts():
    q = path_to_polygon(parse_path("M 0 0 Q 5 10 10 0 Z"), q_samples=8)
    c = path_to_polygon(parse_path("M 0 0 C 0 10 10 10 10 0 Z"), c_samples=12)
    assert len(q) == 1 + 8
    assert len(c) == 1 + 12


def test_flip_maps_every_coordinate_including_control_points():
    cmds = parse_path("M 1 900 L 2 800 Q 3 700 4 600 C 5 500 6 400 7 -124 Z")
    flipped = transform_path(cmds, mmah_to_canvas)
    for (c0, a), (c1, b) in zip(cmds, flipped):
        assert c0 == c1
        if len(a):
            np.testing.assert_allclose(b[:, 0], a[:, 0])
            np.testing.assert_allclose(b[:, 1], MMAH_Y_FLIP - a[:, 1])
    assert mmah_to_canvas([5, 900]).tolist() == [5, 0]
    assert mmah_to_canvas([5, -124]).tolist() == [5, 1024]


def test_quadratic_matches_formula():
    p0, ctrl, p1 = np.array([0.0, 0.0]), np.array([5.0, 10.0]), np.array([10.0, 0.0])
    poly = path_to_polygon(parse_path("M 0 0 Q 5 10 10 0 Z"), q_samples=4)
    t = 0.5
    expected = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1
    np.testing.assert_allclose(poly[2], expected)


def test_d_roundtrip():
    d = "M 1 2 L 3 4 Q 5 6 7 8 C 9 10 11 12 13 14 Z"
    assert path_to_d(parse_path(d)) == d


@needs_data
def test_every_graphics_entry_parses():
    from calligraphy.data.mmah import load_graphics

    for raw in load_graphics().values():
        assert len(raw["strokes"]) == len(raw["medians"]), raw["character"]
        for s in raw["strokes"]:
            _, poly = mmah_outline(s)
            assert len(poly) >= 3 and np.isfinite(poly).all(), raw["character"]
