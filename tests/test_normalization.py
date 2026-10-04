"""How reward v1 treats mis-sized, off-centre, squashed and rotated characters.

These pin current behaviour, run through the full reward path (JSON -> parser ->
Canvas clamping -> scorer). If a scoring change moves one of them, that is a
deliberate reward change: update docs/IMPLEMENTATION.md 3.3 with it.
"""

import pytest
from conftest import needs_data

from calligraphy.data.mmah import get_entry
from calligraphy.score.reward import compute_reward
from calligraphy.score.scenes import SCENES, count_out_of_bounds, to_response

CHARS = ["觀", "永"]


def reward(char: str, scene: str) -> tuple[float, dict]:
    entry = get_entry(char)
    return compute_reward(to_response(SCENES[scene](entry)), entry)


@pytest.fixture(scope="module", params=CHARS)
def perfect(request):
    return request.param, reward(request.param, "perfect")[0]


@needs_data
@pytest.mark.parametrize("scene", ["small_0.5", "big_1.15", "offcenter_inside", "offcenter_corner", "A_small_corner"])
def test_position_and_size_are_forgiven_on_canvas(perfect, scene):
    """Structure aligns scale + translation away, so these score like the perfect drawing."""
    char, best = perfect
    r, info = reward(char, scene)
    assert r == pytest.approx(best, abs=0.02)
    assert info["clamped_points"] == count_out_of_bounds(SCENES[scene](get_entry(char)))


@needs_data
@pytest.mark.parametrize("scene", ["small_0.5", "offcenter_inside", "offcenter_corner"])
def test_placement_flags_but_does_not_cost_reward(perfect, scene):
    char, _ = perfect
    _, info = reward(char, scene)
    assert not info["placement_ok"]


@needs_data
def test_below_quarter_size_reward_collapses(perfect):
    """The fit's scale is clamped to [0.25, 4]: a 0.2x character can't be scaled back up."""
    char, best = perfect
    r, info = reward(char, "tiny_0.2")
    assert info["align_scale"] == pytest.approx(4.0)
    assert r < 0.5 * best


@needs_data
@pytest.mark.parametrize("scene", ["huge_1.6", "offcenter_off_canvas"])
def test_off_canvas_strokes_are_clamped_and_cost_reward(perfect, scene):
    char, best = perfect
    r, info = reward(char, scene)
    assert info["clamped_points"] > 0
    assert r < best - 0.03


@needs_data
@pytest.mark.parametrize("scene", ["squash_x_0.6", "rotate_8", "rotate_20"])
def test_aspect_and_rotation_are_not_normalized(perfect, scene):
    """Alignment is uniform scale + translation only, so these are scored as real errors."""
    char, best = perfect
    r, _ = reward(char, scene)
    assert r < 0.55 * best
