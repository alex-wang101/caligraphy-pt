"""haizelabs/calligraphy-bench: 88 rows (82 unique characters). Eval only, never train on it.

Stroke data is byte-identical to Make Me a Hanzi (canonical Kaiti skeleton, not
the calligrapher's brushwork); reference_image is the master's 64 x 64 JPEG.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

from calligraphy.data.mmah import META_KEYS as MMAH_META_KEYS
from calligraphy.data.mmah import load_dictionary
from calligraphy.data.types import CharacterEntry, data_dir, entry_from_mmah

META_KEYS = ("pinyin", "definition", "stroke_count", "style_difficulty", "calligrapher", "calligrapher_label")


@lru_cache(maxsize=None)
def load_bench(path: Path | None = None) -> tuple[CharacterEntry, ...]:
    rows = pq.read_table(path or data_dir() / "bench" / "test.parquet").to_pylist()
    # Bench strokes are MMAH's, so MMAH's decomposition and per-stroke matches apply.
    dictionary = load_dictionary() if (data_dir() / "mmah" / "dictionary.txt").exists() else {}
    return tuple(
        entry_from_mmah(
            r["character"],
            json.loads(r["strokes_svg"]),
            json.loads(r["medians"]),
            meta={
                **{k: v for k, v in dictionary.get(r["character"], {}).items() if k in MMAH_META_KEYS},
                **{k: r[k] for k in META_KEYS},
            },
            task_id=r["task_id"],
            reference_jpeg=(r["reference_image"] or {}).get("bytes"),
        )
        for r in rows
    )


def bench_by_task() -> dict[str, CharacterEntry]:
    return {e.task_id: e for e in load_bench()}


def bench_characters() -> frozenset[str]:
    return frozenset(e.character for e in load_bench())
