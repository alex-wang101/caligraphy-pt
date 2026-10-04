import sys
from pathlib import Path

import pytest

from calligraphy.data.types import data_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

HAVE_DATA = (data_dir() / "mmah" / "graphics.txt").exists() and (data_dir() / "bench" / "test.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run scripts/fetch_data.py first")


@pytest.fixture(scope="session")
def guan():
    from calligraphy.data.mmah import get_entry

    return get_entry("觀")
