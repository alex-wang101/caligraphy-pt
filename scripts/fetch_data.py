"""Download Make Me a Hanzi and calligraphy-bench into data/.

    uv run python scripts/fetch_data.py
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MMAH_BASE = "https://raw.githubusercontent.com/skishore/makemeahanzi/master"
MMAH_FILES = ["graphics.txt", "dictionary.txt"]
BENCH_REPO = "haizelabs/calligraphy-bench"


def fetch_mmah() -> None:
    out = DATA / "mmah"
    out.mkdir(parents=True, exist_ok=True)
    for name in MMAH_FILES:
        dst = out / name
        if dst.exists():
            print(f"have {dst.relative_to(ROOT)}")
            continue
        print(f"downloading {MMAH_BASE}/{name}")
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(f"{MMAH_BASE}/{name}") as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dst)
        print(f"  -> {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.1f} MB)")


def fetch_bench() -> None:
    out = DATA / "bench"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "test.parquet"
    if dst.exists():
        print(f"have {dst.relative_to(ROOT)}")
        return
    files = [f for f in HfApi().list_repo_files(BENCH_REPO, repo_type="dataset") if f.endswith(".parquet")]
    if len(files) != 1:
        raise RuntimeError(f"expected one parquet file in {BENCH_REPO}, found {files}")
    print(f"downloading {BENCH_REPO}/{files[0]}")
    src = hf_hub_download(BENCH_REPO, files[0], repo_type="dataset")
    shutil.copyfile(src, dst)
    print(f"  -> {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    fetch_mmah()
    fetch_bench()
