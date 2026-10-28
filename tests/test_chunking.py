import os
import random

import pytest

from dfs.chunking import cdc_chunks, fixed_chunks, get_chunker, sha256_hex


def rand_bytes(n, seed=1):
    return random.Random(seed).randbytes(n)


@pytest.mark.parametrize("fn", [fixed_chunks, cdc_chunks])
@pytest.mark.parametrize("n", [0, 1, 100, 12_800, 12_801, 51_200, 300_000])
def test_chunks_reassemble_exactly(fn, n):
    data = rand_bytes(n)
    assert b"".join(fn(data)) == data


def test_empty_input_gives_no_chunks():
    assert fixed_chunks(b"") == [] and cdc_chunks(b"") == []


def test_fixed_sizes():
    chunks = fixed_chunks(rand_bytes(130_000), 50_000)
    assert [len(c) for c in chunks] == [50_000, 50_000, 30_000]


def test_cdc_respects_min_and_max():
    data = rand_bytes(2_000_000)
    chunks = cdc_chunks(data, avg_size=50_000)
    assert all(len(c) <= 200_000 for c in chunks)
    assert all(len(c) >= 12_500 for c in chunks[:-1])      # only the last chunk may be short


def test_cdc_average_is_near_target():
    chunks = cdc_chunks(rand_bytes(5_000_000), avg_size=50_000)
    avg = 5_000_000 / len(chunks)
    assert 25_000 < avg < 80_000


def test_cdc_is_deterministic():
    data = rand_bytes(500_000)
    assert cdc_chunks(data) == cdc_chunks(data)


def hashes(chunks):
    return {sha256_hex(c) for c in chunks}


def test_cdc_survives_insertion_but_fixed_does_not():
    base = rand_bytes(3_000_000)
    edited = base[:1000] + b"INSERTED BYTES" + base[1000:]

    shared_cdc = len(hashes(cdc_chunks(base)) & hashes(cdc_chunks(edited))) / len(hashes(cdc_chunks(base)))
    shared_fixed = len(hashes(fixed_chunks(base)) & hashes(fixed_chunks(edited))) / len(hashes(fixed_chunks(base)))

    assert shared_cdc > 0.9        # only chunks around the edit change
    assert shared_fixed < 0.05     # every boundary after the edit moved


def test_small_file_is_one_chunk_either_way():
    data = rand_bytes(30_000)
    assert len(fixed_chunks(data)) == 1 and len(cdc_chunks(data)) == 1


def test_identical_content_gives_identical_chunk_hashes():
    data = rand_bytes(400_000)
    assert [sha256_hex(c) for c in cdc_chunks(data)] == [sha256_hex(c) for c in cdc_chunks(bytes(data))]


def test_get_chunker_and_bad_args():
    assert get_chunker("fixed", 10)(b"x" * 25) == [b"x" * 10, b"x" * 10, b"x" * 5]
    with pytest.raises(ValueError):
        get_chunker("nope")
    with pytest.raises(ValueError):
        fixed_chunks(b"x", 0)
    with pytest.raises(ValueError):
        cdc_chunks(b"x", avg_size=100, min_size=200)
