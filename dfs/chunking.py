"""Split data into chunks. Two strategies, so they can be compared.

fixed_chunks  Cut every `size` bytes. Fast and predictable, but inserting or deleting
              one byte early in a file shifts every later boundary, so a slightly
              edited file shares almost no chunks with the original (poor dedup).

cdc_chunks    Content-defined chunking. A cut point is chosen where a rolling hash of
              the bytes just seen matches a pattern, so boundaries follow the *content*
              and survive insertions/deletions. A small edit changes only the chunks
              around it. Uses a Gear rolling hash with min/max size limits.

Both take `bytes` and return a list of `bytes`; concatenating the chunks gives back
the input exactly. An empty input gives an empty list.
"""

from __future__ import annotations

import hashlib
import math

DEFAULT_CHUNK_SIZE = 50 * 1024  # 50 KiB


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixed_chunks(data: bytes, size: int = DEFAULT_CHUNK_SIZE) -> list[bytes]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [data[i:i + size] for i in range(0, len(data), size)]


# 256 fixed pseudo-random 64-bit numbers, one per byte value (deterministic on purpose:
# every process must cut the same data at the same places).
_GEAR = [int.from_bytes(hashlib.sha256(b"gear" + bytes([i])).digest()[:8], "big") for i in range(256)]
_MASK64 = (1 << 64) - 1


def cdc_chunks(data: bytes, avg_size: int = DEFAULT_CHUNK_SIZE,
               min_size: int | None = None, max_size: int | None = None) -> list[bytes]:
    """Content-defined chunks averaging roughly `avg_size` bytes.

    min_size defaults to avg/4 and max_size to avg*4. No cut is allowed before
    min_size (those bytes are skipped without hashing, which is also faster); a cut
    is forced at max_size. Input no longer than avg_size is kept as a single chunk.
    """
    min_size = min_size if min_size is not None else max(1, avg_size // 4)
    max_size = max_size if max_size is not None else avg_size * 4
    if not (0 < min_size <= avg_size <= max_size):
        raise ValueError("need 0 < min_size <= avg_size <= max_size")

    # After skipping min_size bytes, the expected extra length until a match is 2**bits.
    bits = max(1, round(math.log2(max(2, avg_size - min_size))))
    mask = (1 << bits) - 1
    gear = _GEAR

    n = len(data)
    if n <= avg_size:
        # Splitting a small file only adds chunks and metadata for no dedup benefit.
        return [data] if n else []

    chunks: list[bytes] = []
    start = 0
    while start < n:
        remaining = n - start
        if remaining <= min_size:
            chunks.append(data[start:])
            break
        limit = start + min(max_size, remaining)
        h = 0
        cut = limit
        for i in range(start + min_size, limit):
            h = ((h << 1) + gear[data[i]]) & _MASK64
            if not (h & mask):
                cut = i + 1
                break
        chunks.append(data[start:cut])
        start = cut
    return chunks


CHUNKERS = {"fixed": fixed_chunks, "cdc": cdc_chunks}


def get_chunker(name: str, size: int = DEFAULT_CHUNK_SIZE):
    """Return a function bytes -> list[bytes] for 'fixed' or 'cdc' with the given size."""
    if name == "fixed":
        return lambda data: fixed_chunks(data, size)
    if name == "cdc":
        return lambda data: cdc_chunks(data, size)
    raise ValueError(f"unknown chunker {name!r} (use 'fixed' or 'cdc')")
