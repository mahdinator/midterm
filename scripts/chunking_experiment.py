#!/usr/bin/env python3
"""Compare fixed-size and content-defined chunking.

    python scripts/chunking_experiment.py            # 8 MB of random data, 50 KiB chunks
    python scripts/chunking_experiment.py --mb 32 --chunk-kb 64

Measures, for each chunker:
  * number of chunks and the average / min / max chunk size
  * chunking speed
  * how many chunks survive when the file is edited (this is what drives dedup):
      - insert 16 bytes near the start      (shifts everything after it)
      - overwrite 16 bytes in the middle    (changes content, keeps offsets)
      - append 1 KB at the end
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfs.chunking import cdc_chunks, fixed_chunks, sha256_hex  # noqa: E402


def shared_fraction(a: list[bytes], b: list[bytes]) -> float:
    """Fraction of the ORIGINAL file's chunks that also exist in the edited file."""
    ha = {sha256_hex(c) for c in a}
    hb = {sha256_hex(c) for c in b}
    return len(ha & hb) / len(ha)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mb", type=float, default=8)
    p.add_argument("--chunk-kb", type=int, default=50)
    args = p.parse_args()

    size = int(args.chunk_kb * 1024)
    rng = random.Random(42)
    base = rng.randbytes(int(args.mb * 1024 * 1024))
    mid = len(base) // 2
    edits = {
        "insert 16 B near start": base[:1000] + rng.randbytes(16) + base[1000:],
        "overwrite 16 B in middle": base[:mid] + rng.randbytes(16) + base[mid + 16:],
        "append 1 KB": base + rng.randbytes(1024),
    }
    chunkers = {"fixed": lambda d: fixed_chunks(d, size), "cdc": lambda d: cdc_chunks(d, size)}

    print(f"data: {len(base) / 1e6:.1f} MB random bytes, target chunk size {args.chunk_kb} KiB\n")
    print(f"{'chunker':8} {'chunks':>7} {'avg KB':>8} {'min KB':>8} {'max KB':>8} {'MB/s':>7}")
    results = {}
    for name, fn in chunkers.items():
        t = time.perf_counter()
        chunks = fn(base)
        dt = time.perf_counter() - t
        lens = [len(c) for c in chunks]
        results[name] = chunks
        print(f"{name:8} {len(chunks):>7} {sum(lens) / len(lens) / 1024:>8.1f} {min(lens) / 1024:>8.1f} "
              f"{max(lens) / 1024:>8.1f} {len(base) / 1e6 / dt:>7.1f}")

    print("\nchunks of the original that are still shared after an edit (higher = better dedup):")
    print(f"{'edit':28} {'fixed':>8} {'cdc':>8}")
    for label, edited in edits.items():
        row = [shared_fraction(results[n], chunkers[n](edited)) for n in ("fixed", "cdc")]
        print(f"{label:28} {row[0]:>8.1%} {row[1]:>8.1%}")


if __name__ == "__main__":
    main()
