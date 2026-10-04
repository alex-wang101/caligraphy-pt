"""Golden tests from v2 section 3.3 (relationships, not exact decimals)."""

import numpy as np
import pytest
from conftest import needs_data

from calligraphy.convert import convert
from calligraphy.score.align import Moments, robust_fit
from calligraphy.score.levels import score_character
from calligraphy.score.reward import compute_reward
from calligraphy.score.scenes import SCENES, component_strokes
from calligraphy.score.stroke import frechet, stroke_metrics
from calligraphy.tools import format_strokes

FLAG_DROP = 0.15  # a level is flagged when it drops this far below the perfect drawing


def test_frechet_basics():
    a = np.column_stack([np.linspace(0, 100, 32), np.zeros(32)])
    assert frechet(a, a) == 0
    assert frechet(a, a + [0, 5]) == pytest.approx(5)
    assert frechet(a, a[::-1]) == pytest.approx(100)


def test_robust_fit_ignores_outliers():
    rng = np.random.default_rng(0)
    target = [Moments(rng.uniform(100, 900, 2), 40.0) for _ in range(20)]
    agent = [Moments((m.c - 30) / 2, 20.0) for m in target]  # true map: s = 2, t = (30, 30)
    for k in range(6):
        agent[k] = Moments(agent[k].c + [150, -90], 20.0)
    sim, inliers = robust_fit(agent, target)
    assert sim.s == pytest.approx(2, abs=1e-6)
    np.testing.assert_allclose(sim.t, [30, 30], atol=1e-6)
    assert not inliers[:6].any()


# --- 觀 stroke 1: per-stroke metrics -----------------------------------------


@pytest.fixture(scope="module")
def stroke1(guan):
    s = convert(guan)[0]
    pts = np.asarray(s["points"], float)
    rng = np.random.default_rng(0)
    good = pts.copy()
    good[:, :2] += rng.normal(0, 2.0, good[:, :2].shape)
    return good, float(s["width"]), guan.outlines[0], guan.medians[0]


@needs_data
def test_good_stroke(stroke1):
    good, w, outline, median = stroke1
    m = stroke_metrics(good, w, outline, median)
    assert m.inside_frac >= 0.9  # converter tips sit on the edge, so noise can push one out
    assert m.iou > 0.7
    assert m.frechet < 25
    assert m.direction_ok


@needs_data
def test_reversed_stroke_invisible_to_iou(stroke1):
    good, w, outline, median = stroke1
    g = stroke_metrics(good, w, outline, median)
    r = stroke_metrics(good[::-1], w, outline, median)
    assert r.iou == pytest.approx(g.iou, abs=0.02)
    assert r.inside_frac == g.inside_frac
    assert r.frechet > 100
    assert not r.direction_ok


@needs_data
def test_shifted_stroke(stroke1):
    good, w, outline, median = stroke1
    shifted = good.copy()
    shifted[:, :2] += [120, 80]  # v2's (+120, -80) in the y-up frame
    m = stroke_metrics(shifted, w, outline, median)
    assert m.inside_frac == 0.0
    assert m.iou == 0.0
    assert m.direction_ok


@needs_data
@pytest.mark.parametrize("factor", [0.3, 3.0])
def test_wrong_width_invisible_to_frechet(stroke1, factor):
    good, w, outline, median = stroke1
    g = stroke_metrics(good, w, outline, median)
    m = stroke_metrics(good, w * factor, outline, median)
    assert m.frechet == pytest.approx(g.frechet)
    assert m.iou < 0.6 * g.iou


# --- 觀 whole character: three levels ----------------------------------------


@pytest.fixture(scope="module")
def scores(guan):
    return {name: score_character(make(guan), guan) for name, make in SCENES.items()}


@needs_data
def test_perfect_matches_converter_iou(scores, guan):
    from calligraphy.roundtrip import roundtrip

    p = scores["perfect"]
    converter_iou = np.mean(roundtrip(guan)["stroke_iou"])
    assert p.placement_ok
    assert p.placement_offset < 10 and p.placement_ratio == pytest.approx(1, abs=0.03)
    assert p.structure == pytest.approx(converter_iou, abs=0.02)
    assert p.shape == pytest.approx(converter_iou, abs=0.02)


@needs_data
def test_A_small_corner_flags_placement_only(scores):
    p, a = scores["perfect"], scores["A_small_corner"]
    assert not a.placement_ok
    assert a.placement_ratio == pytest.approx(0.6, abs=0.03)
    assert a.structure > p.structure - 0.02
    assert a.shape > p.shape - 0.02


@needs_data
def test_B_slid_component_flags_structure_only(scores, guan):
    p, b = scores["perfect"], scores["B_slid_component"]
    assert b.placement_ok
    assert b.structure < p.structure - FLAG_DROP
    assert b.shape > p.shape - 0.02
    # Robust alignment: unmoved strokes keep their score, moved strokes score ~0.
    moved = set(component_strokes(guan, 1))
    assert moved == set(range(18, 25))
    for i, (sb, sp) in enumerate(zip(b.structure_strokes, p.structure_strokes)):
        if i in moved:
            assert sb < 0.1
        else:
            assert sb > sp - 0.05
    assert not any(b.align_inliers[i] for i in moved)


@needs_data
def test_C_straight_flat_flags_shape(scores):
    p, c = scores["perfect"], scores["C_straight_flat"]
    assert c.placement_ok
    assert c.shape < p.shape - FLAG_DROP


@needs_data
def test_missing_and_extra_strokes_cost_score(guan):
    perfect = SCENES["perfect"](guan)
    full = score_character(perfect, guan).structure
    missing = score_character(perfect[:-5], guan).structure
    extra = score_character(perfect + perfect[:10], guan).structure
    assert missing < full - 0.1
    assert extra < full - 0.1


# --- reward v1 -----------------------------------------------------------------


@needs_data
def test_reward_v1(guan):
    r, info = compute_reward(format_strokes(convert(guan)), guan)
    assert info["parse_ok"]
    assert r == info["structure"] > 0.85
    for key in ("placement_offset", "placement_ratio", "structure", "shape", "structure_strokes"):
        assert key in info


@needs_data
@pytest.mark.parametrize("text", ["I can't draw that.", '{"strokes": [{"points": [[1, 2]], "width": 5}]}'])
def test_reward_zero_on_parse_failure(guan, text):
    r, info = compute_reward(text, guan)
    assert r == 0.0 and not info["parse_ok"]


@needs_data
def test_reward_action_frame_64(guan):
    from calligraphy.canvas import CanvasConfig

    cfg = CanvasConfig(action_frame=64)
    r, _ = compute_reward(format_strokes(convert(guan, action_frame=64)), guan, cfg)
    assert r > 0.6  # 16-unit quantization costs fidelity, but the drawing is still clearly right
