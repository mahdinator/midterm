#!/usr/bin/env python3
"""End-to-end demo of chunking, placement and reference-counted deduplication.

Needs the 4 storage nodes (ports 8101-8104) and Postgres running, e.g.
    docker compose -f docker-compose.nodes.yml up -d
then
    $env:DATABASE_URL = "postgresql://dfs:dfs@localhost:5432/dfs"      # PowerShell
    python scripts/demo.py

It uses its own Postgres schema ("demo"), which it recreates on every run, and deletes
everything it stored before it exits, so it leaves the nodes clean.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfs.db import MetadataDB  # noqa: E402
from dfs.nodeclient import NodeClient  # noqa: E402
from dfs.objectstore import ObjectStore  # noqa: E402

DSN = os.getenv("DATABASE_URL", "postgresql://dfs:dfs@localhost:5432/dfs")
NODES = {f"node{i}": f"http://localhost:810{i}" for i in range(1, 5)}
KB = 1024


def show_placement(store: ObjectStore, nodes: NodeClient, meta) -> None:
    print(f"    {'chunk':10} " + " ".join(f"{n:>6}" for n in NODES) + "   refs")
    for h in dict.fromkeys(meta.chunk_hashes):
        marks = " ".join(f"{'X' if nodes.has(n, h) else '.':>6}" for n in NODES)
        print(f"    {h[:8]:10} {marks}   {store.db.refcount(h)}")


def main() -> None:
    db = MetadataDB(DSN, schema="demo")
    db.drop_schema()
    db.init_schema()
    nodes = NodeClient()
    store = ObjectStore(db, nodes, replicas=3, chunker="fixed", chunk_size=50 * KB)
    for node_id, url in NODES.items():
        store.add_node(node_id, url)

    rng = random.Random(7)
    a = rng.randbytes(300 * KB)                       # 6 chunks of 50 KiB
    b = a + rng.randbytes(100 * KB)                   # same 6 chunks + 2 new ones

    print("1) put a.bin (300 KiB)")
    meta_a = store.put_object("demo", "a.bin", a)
    print(f"   {len(meta_a.chunk_hashes)} chunks, version {meta_a.version}; copies on nodes (X = stored):")
    show_placement(store, nodes, meta_a)

    print("\n2) put copy.bin (identical content): nothing should be transferred")
    sent = store.counters["bytes_sent"]
    store.put_object("demo", "copy.bin", a)
    print(f"   bytes sent to nodes: {store.counters['bytes_sent'] - sent}")
    s = store.stats()
    print(f"   logical {s['logical_bytes'] // KB} KiB, unique {s['unique_bytes'] // KB} KiB, dedup ratio {s['dedup_ratio']}")

    print("\n3) put b.bin (a.bin + 100 KiB): only the 2 new chunks should be transferred")
    sent = store.counters["bytes_sent"]
    meta_b = store.put_object("demo", "b.bin", b)
    print(f"   bytes sent to nodes: {(store.counters['bytes_sent'] - sent) // KB} KiB "
          f"(2 new chunks x 3 replicas x 50 KiB = 300 KiB)")
    s = store.stats()
    print(f"   logical {s['logical_bytes'] // KB} KiB, unique {s['unique_bytes'] // KB} KiB, dedup ratio {s['dedup_ratio']}")

    print("\n4) read back and verify")
    for key, original in (("a.bin", a), ("copy.bin", a), ("b.bin", b)):
        print(f"   {key}: {'OK' if store.get_object('demo', key)[1] == original else 'MISMATCH'}")

    print("\n5) delete copy.bin: its chunks are still used by a.bin and b.bin, so none are removed")
    store.delete_object("demo", "copy.bin")
    print(f"   chunks purged from nodes: {store.counters['chunks_purged']}; a.bin readable: "
          f"{store.get_object('demo', 'a.bin')[1] == a}")

    print("\n6) delete a.bin: still referenced by b.bin")
    store.delete_object("demo", "a.bin")
    print(f"   chunks purged from nodes: {store.counters['chunks_purged']}; b.bin readable: "
          f"{store.get_object('demo', 'b.bin')[1] == b}")

    print("\n7) delete b.bin: last reference gone, all 8 chunks removed from all nodes")
    store.delete_object("demo", "b.bin")
    left = sum(nodes.has(n, h) for n in NODES for h in set(meta_b.chunk_hashes))
    print(f"   chunks purged: {store.counters['chunks_purged']}; copies still on nodes: {left}; "
          f"unique chunks in database: {store.stats()['unique_chunks']}")

    db.drop_schema()


if __name__ == "__main__":
    main()
