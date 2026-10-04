"""Make Me a Hanzi: graphics.txt (strokes, medians) and dictionary.txt (decomposition, matches)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from calligraphy.data.types import CharacterEntry, data_dir, entry_from_mmah

META_KEYS = ("definition", "pinyin", "decomposition", "radical", "matches", "etymology")


def _read_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["character"]] = d
    return out


@lru_cache(maxsize=None)
def load_graphics(path: Path | None = None) -> dict[str, dict]:
    """Raw graphics.txt rows keyed by character (MMAH frame, unparsed)."""
    return _read_jsonl(path or data_dir() / "mmah" / "graphics.txt")


@lru_cache(maxsize=None)
def load_dictionary(path: Path | None = None) -> dict[str, dict]:
    return _read_jsonl(path or data_dir() / "mmah" / "dictionary.txt")


def entry_from_raw(raw: dict, dictionary: dict[str, dict] | None = None) -> CharacterEntry:
    char = raw["character"]
    info = (dictionary or {}).get(char, {})
    meta = {k: info[k] for k in META_KEYS if k in info}
    return entry_from_mmah(char, raw["strokes"], raw["medians"], meta=meta)


@lru_cache(maxsize=4096)
def get_entry(char: str) -> CharacterEntry:
    raw = load_graphics().get(char)
    if raw is None:
        raise KeyError(f"{char!r} not in Make Me a Hanzi")
    return entry_from_raw(raw, load_dictionary())


def iter_entries() -> Iterator[CharacterEntry]:
    dictionary = load_dictionary()
    for raw in load_graphics().values():
        yield entry_from_raw(raw, dictionary)
