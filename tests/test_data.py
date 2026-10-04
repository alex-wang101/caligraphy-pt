import numpy as np
from conftest import needs_data


@needs_data
def test_guan_is_upright(guan):
    # Stroke 1 of 觀 lands top-left and lifts at the bottom tip (summary worked example).
    m = guan.medians[0]
    assert m[0].tolist() == [245, 110]
    assert m[-1].tolist() == [308, 255]
    assert m[0][1] < m[-1][1]


@needs_data
def test_bench_identical_to_mmah_and_has_references():
    from calligraphy.data.bench import load_bench
    from calligraphy.data.mmah import get_entry

    bench = load_bench()
    assert len(bench) == 88
    assert len({e.character for e in bench}) == 82
    for e in bench:
        assert e.reference_jpeg and e.task_id
        assert e.strokes_d == get_entry(e.character).strokes_d


@needs_data
def test_flipped_render_matches_reference_orientation():
    from flip_check import flip_scores

    from calligraphy.data.bench import load_bench

    bench = load_bench()
    wins = sum(f > r for f, r in map(flip_scores, bench))
    assert wins >= 0.75 * len(bench)


@needs_data
def test_writing_starts_above_where_it_ends():
    from calligraphy.data.mmah import load_graphics
    from calligraphy.frame import mmah_to_canvas

    rows = [r["medians"] for r in load_graphics().values() if len(r["medians"]) > 1]
    ok = sum(mmah_to_canvas(m[0][0])[1] < mmah_to_canvas(m[-1][-1])[1] for m in rows)
    assert ok >= 0.9 * len(rows)


@needs_data
def test_dictionary_meta_attached(guan):
    assert guan.meta["decomposition"] == "⿰雚見"
    assert len(guan.meta["matches"]) == guan.n_strokes
    assert np.all([len(o) >= 3 for o in guan.outlines])
