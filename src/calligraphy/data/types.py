from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from calligraphy.frame import mmah_to_canvas
from calligraphy.svgpath import mmah_outline


def data_dir() -> Path:
    """Dataset root: $CALLIGRAPHY_DATA, else <repo>/data."""
    env = os.environ.get("CALLIGRAPHY_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "data"


@dataclass
class CharacterEntry:
    """One character, everything in the canvas frame (1024 box, y down)."""

    character: str
    strokes_d: list[str]  # outline path strings
    outlines: list[np.ndarray]  # outline polygons, (N, 2) each
    medians: list[np.ndarray]  # centerlines in writing order, (K, 2) each; [0] lands, [-1] lifts
    meta: dict = field(default_factory=dict)  # pinyin, definition, decomposition, matches, ...
    task_id: str | None = None
    reference_jpeg: bytes | None = None

    @property
    def n_strokes(self) -> int:
        return len(self.outlines)


def entry_from_mmah(character: str, strokes: list[str], medians: list, **kw) -> CharacterEntry:
    if len(strokes) != len(medians):
        raise ValueError(f"{character}: {len(strokes)} strokes but {len(medians)} medians")
    ds, polys = zip(*(mmah_outline(s) for s in strokes)) if strokes else ((), ())
    return CharacterEntry(
        character=character,
        strokes_d=list(ds),
        outlines=list(polys),
        medians=[mmah_to_canvas(m) for m in medians],
        **kw,
    )
