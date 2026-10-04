"""Frozen data splits (docs/IMPLEMENTATION.md 3.4).

- bench_excluded: the 82 calligraphy-bench characters, removed from everything trained on.
- val: fixed in-distribution held-out set, monitored every N steps.
- test_compositional: characters whose first-level IDS components all appear in
  training (in >= 2 training characters each) but whose combination does not,
  and which never appear inside a training character's decomposition (otherwise
  training on e.g. 清 would teach 青's strokes directly).
- viz16: 16 validation characters spread across stroke counts, for image grids.
- train: everything else in Make Me a Hanzi.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from calligraphy.data.bench import bench_characters
from calligraphy.data.mmah import load_dictionary, load_graphics
from calligraphy.data.types import data_dir

SPLITS_DIR = Path(__file__).resolve().parents[2] / "splits"
ARITY = {"⿰": 2, "⿱": 2, "⿲": 3, "⿳": 3, "⿴": 2, "⿵": 2, "⿶": 2, "⿷": 2, "⿸": 2, "⿹": 2, "⿺": 2, "⿻": 2}
UNKNOWN = "？"
KEEP_IN_TRAIN = frozenset("永")  # single-character sanity check (v2 training plan step 2)


def is_unified_ideograph(c: str) -> bool:
    """CJK Unified Ideographs block; radical forms and stroke glyphs stay in train."""
    return "\u4e00" <= c <= "\u9fff"
NAMES = ("train", "val", "test_compositional", "viz16", "bench_excluded")


def parse_ids(s: str, i: int = 0):
    """Prefix IDS -> (node, next index); node is a character or (operator, [children])."""
    if i >= len(s):
        raise ValueError(f"truncated IDS {s!r}")
    c = s[i]
    if c in ARITY:
        children, j = [], i + 1
        for _ in range(ARITY[c]):
            child, j = parse_ids(s, j)
            children.append(child)
        return (c, children), j
    return c, i + 1


def _leaves(node) -> list[str]:
    return [node] if isinstance(node, str) else [x for ch in node[1] for x in _leaves(ch)]


def _key(node) -> str:
    return node if isinstance(node, str) else node[0] + "".join(_key(ch) for ch in node[1])


def first_level(decomposition: str) -> tuple[str, tuple[str, ...]] | None:
    """(operator, components) of a compound, or None for atomic/unknown/malformed entries.

    A named child (a single character) is one component; an unnamed nested
    subtree contributes its leaves, since there is no character to "see" for it.
    """
    if not decomposition or decomposition[0] not in ARITY or UNKNOWN in decomposition:
        return None
    try:
        node, end = parse_ids(decomposition)
    except ValueError:
        return None
    if end != len(decomposition):
        return None
    comps = tuple(x for ch in node[1] for x in ([ch] if isinstance(ch, str) else _leaves(ch)))
    return node[0], comps


def combination(decomposition: str) -> str | None:
    return _key(parse_ids(decomposition)[0]) if first_level(decomposition) else None


@dataclass
class Splits:
    train: list[str]
    val: list[str]
    test_compositional: list[str]
    viz16: list[str]
    bench_excluded: list[str]
    meta: dict = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_splits(seed: int = 0, n_val: int = 300, n_test: int = 500, min_component_count: int = 2) -> Splits:
    graphics, dictionary = load_graphics(), load_dictionary()
    decomp = {c: dictionary.get(c, {}).get("decomposition", "") for c in graphics}
    bench = sorted(bench_characters())
    pool = sorted(set(graphics) - set(bench))
    rng = random.Random(seed)

    # Validation: uniform sample of the pool (in-distribution), never the sanity-check character.
    val = sorted(rng.sample([c for c in pool if is_unified_ideograph(c) and c not in KEEP_IN_TRAIN], n_val))
    remaining = [c for c in pool if c not in set(val)]

    parsed = {c: first_level(decomp[c]) for c in remaining}
    comp_count = Counter(x for c in remaining if parsed[c] for x in set(parsed[c][1]))
    combo_count = Counter(combination(decomp[c]) for c in remaining if parsed[c])
    # How many remaining characters mention each character anywhere in their decomposition.
    mentioned = Counter(x for c in remaining for x in set(decomp[c]) if x not in ARITY and x != UNKNOWN)

    candidates = [
        c for c in remaining if parsed[c] and len(parsed[c][1]) >= 2 and is_unified_ideograph(c) and c not in KEEP_IN_TRAIN
    ]
    rng.shuffle(candidates)
    test: list[str] = []
    for c in candidates:
        if len(test) >= n_test:
            break
        comps = set(parsed[c][1])
        if combo_count[combination(decomp[c])] > 1:
            continue  # another character has the exact same combination
        if mentioned[c] > 0:
            continue  # c is itself part of a training character
        if any(comp_count[x] - 1 < min_component_count for x in comps):
            continue  # a component would become too rare in training
        test.append(c)
        for x in comps:
            comp_count[x] -= 1
        for x in set(decomp[c]):
            mentioned[x] -= 1
    test = sorted(test)
    train = sorted(set(remaining) - set(test))

    by_strokes = sorted(val, key=lambda c: (len(graphics[c]["strokes"]), c))
    viz16 = [by_strokes[round(i * (len(by_strokes) - 1) / 15)] for i in range(16)]

    root = data_dir()
    meta = {
        "seed": seed,
        "sizes": {"train": len(train), "val": len(val), "test_compositional": len(test), "viz16": 16, "bench_excluded": len(bench)},
        "criteria": {
            "val": f"uniform sample of {n_val} non-bench CJK unified ideographs (excluding {''.join(sorted(KEEP_IN_TRAIN))})",
            "test_compositional": (
                f"CJK unified ideograph; first-level IDS compound without {UNKNOWN}; every component in >= {min_component_count} train "
                "characters; no train character with the same combination; not mentioned in any train decomposition"
            ),
            "viz16": "val characters evenly spaced by stroke count",
        },
        "sources": {
            "graphics.txt": _sha256(root / "mmah" / "graphics.txt"),
            "dictionary.txt": _sha256(root / "mmah" / "dictionary.txt"),
            "calligraphy-bench test.parquet": _sha256(root / "bench" / "test.parquet"),
        },
    }
    return Splits(train, val, test, viz16, bench, meta)


def write_splits(splits: Splits, out: Path = SPLITS_DIR) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        (out / f"{name}.txt").write_text("\n".join(getattr(splits, name)) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(splits.meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_splits(path: Path = SPLITS_DIR) -> Splits:
    sets = {n: (path / f"{n}.txt").read_text(encoding="utf-8").split() for n in NAMES}
    return Splits(**sets, meta=json.loads((path / "meta.json").read_text(encoding="utf-8")))
