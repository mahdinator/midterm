import threading
from collections import Counter

H = [f"{i:064x}" for i in range(1, 8)]          # fake but well-formed 64-char hashes
a, b, c, d, e = H[:5]


def reserve(db, *hashes):
    counts = Counter(hashes)
    return db.reserve_chunks({h: 100 for h in counts}, counts)


def test_reserve_reports_which_chunks_are_new_and_counts_references(db):
    assert reserve(db, a, b) == {a, b}
    assert reserve(db, b, c) == {c}                  # b already existed -> deduplicated
    assert (db.refcount(a), db.refcount(b), db.refcount(c)) == (1, 2, 1)
    assert db.refcount(d) is None


def test_same_chunk_twice_in_one_object_counts_twice(db):
    reserve(db, a, a, a)
    assert db.refcount(a) == 3


def test_release_frees_only_at_zero(db):
    reserve(db, a, b)
    reserve(db, a)                                   # a: 2, b: 1
    seen = []
    freed = db.release_chunks(Counter([a, b]), on_unreferenced=seen.append)
    assert freed == [b] and seen == [[b]]
    assert db.refcount(a) == 1 and db.refcount(b) is None


def test_freed_chunk_row_stays_locked_while_bytes_are_being_deleted(db):
    """A concurrent put of the same chunk must wait for the physical delete to finish."""
    import psycopg
    import pytest

    reserve(db, a)
    outcome = []

    def callback(hashes):
        with db._tx() as other:                      # a second, independent connection
            try:
                other.execute("SELECT 1 FROM chunks WHERE hash = %s FOR UPDATE NOWAIT", (hashes[0],))
                outcome.append("not locked")
            except psycopg.errors.LockNotAvailable:
                outcome.append("locked")

    db.release_chunks(Counter([a]), on_unreferenced=callback)
    assert outcome == ["locked"]


def test_commit_object_assigns_increasing_versions_and_keeps_chunk_order(db):
    reserve(db, a, b, c)
    m1 = db.commit_object("bkt", "k", 300, "fixed", [c, a, b])
    m2 = db.commit_object("bkt", "k", 300, "fixed", [a, b, c])
    assert (m1.version, m2.version) == (1, 2)
    assert m1.chunk_hashes == [c, a, b]              # order preserved
    assert m1.bucket == "bkt" and m1.size == 300 and m1.chunker == "fixed" and m1.created_at is not None
    assert db.get_object("bkt", "k").version == 2    # latest by default
    assert db.get_object("bkt", "k", 1).chunk_hashes == [c, a, b]
    assert db.get_object("bkt", "nope") is None and db.get_object("bkt", "k", 9) is None
    assert [m.version for m in db.list_versions("bkt", "k")] == [1, 2]


def test_concurrent_commits_get_distinct_versions(db):
    reserve(db, a)
    versions = []

    def work():
        versions.append(db.commit_object("bkt", "same", 1, "fixed", [a]).version)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(versions) == list(range(1, 9))


def test_list_objects_prefix_latest_only_and_bucket_isolation(db):
    reserve(db, a)
    for bucket, key in [("b1", "logs/1"), ("b1", "logs/2"), ("b1", "img/1"), ("b2", "logs/9")]:
        db.commit_object(bucket, key, 1, "fixed", [a])
    db.commit_object("b1", "logs/1", 1, "fixed", [a])             # version 2 of logs/1
    listing = db.list_objects("b1", "logs/")
    assert [(m.key, m.version) for m in listing] == [("logs/1", 2), ("logs/2", 1)]
    assert [m.key for m in db.list_objects("b1")] == ["img/1", "logs/1", "logs/2"]
    assert [m.key for m in db.list_objects("b2")] == ["logs/9"]
    assert len(db.list_objects("b1", limit=2)) == 2


def put(db, bucket, key, hashes):
    counts = Counter(hashes)
    db.reserve_chunks({h: 100 for h in counts}, counts)
    return db.commit_object(bucket, key, 100 * len(hashes), "fixed", hashes)


def test_shared_chunk_survives_until_last_reference_is_deleted(db):
    put(db, "bkt", "one", [a, b])
    put(db, "bkt", "two", [b, c])                     # b is shared
    assert db.refcount(b) == 2
    freed_cb = []
    n, freed = db.delete_object("bkt", "one", on_unreferenced=freed_cb.extend)
    assert n == 1 and freed == [a] == freed_cb         # only a became unreferenced
    assert db.refcount(b) == 1 and db.refcount(a) is None
    n, freed = db.delete_object("bkt", "two", on_unreferenced=freed_cb.extend)
    assert sorted(freed) == sorted([b, c])
    assert db.refcount(b) is None and db.get_object("bkt", "two") is None


def test_delete_single_version_vs_all_versions(db):
    put(db, "bkt", "k", [a])
    put(db, "bkt", "k", [a, b])
    assert db.refcount(a) == 2
    n, freed = db.delete_object("bkt", "k", version=1)
    assert (n, freed) == (1, []) and db.refcount(a) == 1
    assert db.get_object("bkt", "k").version == 2
    n, freed = db.delete_object("bkt", "k")            # all remaining versions
    assert n == 1 and sorted(freed) == sorted([a, b])
    assert db.delete_object("bkt", "k") == (0, [])    # deleting again is a no-op


def test_duplicate_chunk_within_one_object_released_fully(db):
    put(db, "bkt", "dup", [a, a, b])
    assert db.refcount(a) == 2
    _, freed = db.delete_object("bkt", "dup")
    assert sorted(freed) == sorted([a, b])


def test_stats_show_dedup(db):
    put(db, "bkt", "one", [a, b])
    put(db, "bkt", "two", [a, b])                     # identical content
    s = db.stats()
    assert s["object_versions"] == 2 and s["unique_chunks"] == 2 and s["chunk_references"] == 4
    assert s["logical_bytes"] == 400 and s["unique_bytes"] == 200
    assert s["saved_bytes"] == 200 and s["dedup_ratio"] == 2.0


def test_ring_nodes_crud(db):
    db.add_ring_node("node1", "http://n1")
    db.add_ring_node("node2", "http://n2")
    db.add_ring_node("node1", "http://n1-new")         # update url
    assert db.ring_nodes() == {"node1": "http://n1-new", "node2": "http://n2"}
    assert db.remove_ring_node("node2") is True and db.remove_ring_node("node2") is False
    assert list(db.ring_nodes()) == ["node1"]


def test_concurrent_reserve_counts_every_reference(db):
    def work():
        reserve(db, a)

    threads = [threading.Thread(target=work) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert db.refcount(a) == 10
