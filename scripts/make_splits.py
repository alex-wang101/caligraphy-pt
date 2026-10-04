"""Write the frozen splits to splits/ (commit the result; never regenerate casually).

    uv run python scripts/make_splits.py
"""

from calligraphy.splits import SPLITS_DIR, make_splits, write_splits

if __name__ == "__main__":
    s = make_splits()
    write_splits(s)
    print(s.meta["sizes"])
    print("viz16:", "".join(s.viz16))
    print("test sample:", "".join(s.test_compositional[:40]))
    print(f"wrote {SPLITS_DIR}")
