from collections import Counter

import pytest
from conftest import needs_data

from calligraphy.splits import ARITY, NAMES, combination, first_level, load_splits, make_splits, parse_ids


def test_parse_ids():
    assert parse_ids("⿰女子")[0] == ("⿰", ["女", "子"])
    assert first_level("⿰雚見") == ("⿰", ("雚", "見"))
    assert first_level("⿱木⿰木木") == ("⿱", ("木", "木", "木"))
    assert first_level("⿱？水") is None
    assert first_level("一") is None
    assert first_level("⿰女") is None  # truncated
    assert combination("⿱木⿰木木") == "⿱木⿰木木"


@pytest.fixture(scope="module")
def splits():
    return load_splits()


@pytest.fixture(scope="module")
def dictionary():
    from calligraphy.data.mmah import load_dictionary

    return load_dictionary()


@needs_data
def test_split_files_reproduce(splits):
    fresh = make_splits(seed=splits.meta["seed"])
    for name in NAMES:
        assert getattr(fresh, name) == getattr(splits, name), name
    assert fresh.meta["sources"] == splits.meta["sources"], "source data changed since the split was frozen"


@needs_data
def test_disjoint_and_complete(splits):
    from calligraphy.data.mmah import load_graphics

    train, val, test, bench = map(set, (splits.train, splits.val, splits.test_compositional, splits.bench_excluded))
    assert not (train & val) and not (train & test) and not (val & test)
    assert not (bench & (train | val | test))
    assert train | val | test | bench == set(load_graphics())
    assert set(splits.viz16) <= val and len(splits.viz16) == 16
    assert "永" in train


@needs_data
def test_bench_characters_excluded(splits):
    from calligraphy.data.bench import bench_characters

    assert set(splits.bench_excluded) == bench_characters()
    assert len(splits.bench_excluded) == 82


@needs_data
def test_compositional_holdout(splits, dictionary):
    train = splits.train
    decomp = {c: dictionary.get(c, {}).get("decomposition", "") for c in train + splits.test_compositional}
    seen = Counter(x for c in train if (fl := first_level(decomp[c])) for x in set(fl[1]))
    train_combos = {combination(decomp[c]) for c in train if first_level(decomp[c])}
    mentioned = {x for c in train for x in decomp[c] if x not in ARITY}
    for c in splits.test_compositional:
        op, comps = first_level(decomp[c])
        assert len(comps) >= 2
        assert all(seen[x] >= 2 for x in comps), (c, comps)
        assert combination(decomp[c]) not in train_combos, c
        assert c not in mentioned, c
