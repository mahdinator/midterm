import random

import pytest

from dfs.chunking import sha256_hex
from dfs.nodeclient import NodeError
from dfs.objectstore import (ChunkUnavailable, InsufficientReplicas, ObjectNotFound, ObjectStore)


class FakeNodes:
    """In-memory stand-in for the storage nodes, with failure injection."""

    def __init__(self):
        self.disk: dict[str, dict[str, bytes]] = {}
        self.down: set[str] = set()
        self.put_calls = 0

    def set_urls(self, urls):
        for n in urls:
            self.disk.setdefault(n, {})

    def _check(self, node):
        if node in self.down:
            raise NodeError(f"{node} is down")

    def put(self, node, h, data):
        self._check(node)
        self.put_calls += 1
        self.disk[node][h] = data

    def get(self, node, h):
        self._check(node)
        return self.disk[node].get(h)

    def has(self, node, h):
        self._check(node)
        return h in self.disk[node]

    def delete(self, node, h):
        self._check(node)
        return self.disk[node].pop(h, None) is not None

    def holders(self, h):
        return sorted(n for n, d in self.disk.items() if h in d)


def rand(n, seed=1):
    return random.Random(seed).randbytes(n)


@pytest.fixture
def fake():
    return FakeNodes()


@pytest.fixture
def store(db, fake):
    s = ObjectStore(db, fake, replicas=3, chunker="fixed", chunk_size=1000)
    for i in range(1, 5):
        s.add_node(f"node{i}", f"http://node{i}:8000")
    return s


def test_roundtrip_multi_chunk(store):
    data = rand(4500)
    meta = store.put_object("bkt", "file.bin", data)
    assert meta.size == 4500 and meta.version == 1 and len(meta.chunk_hashes) == 5
    got_meta, got = store.get_object("bkt", "file.bin")
    assert got == data and got_meta.chunk_hashes == meta.chunk_hashes
    assert store.head_object("bkt", "file.bin").size == 4500


def test_empty_object_and_missing_object(store):
    store.put_object("bkt", "empty", b"")
    assert store.get_object("bkt", "empty")[1] == b""
    with pytest.raises(ObjectNotFound):
        store.get_object("bkt", "nope")
    assert store.head_object("bkt", "nope") is None


def test_each_chunk_is_on_exactly_the_ring_nodes(store, fake):
    meta = store.put_object("bkt", "f", rand(5000))
    for h in meta.chunk_hashes:
        assert fake.holders(h) == sorted(store.placement(h)) and len(fake.holders(h)) == 3


def test_identical_content_is_stored_once(store, fake, db):
    data = rand(5000)
    store.put_object("bkt", "a", data)
    sent_before = fake.put_calls
    store.put_object("bkt", "copy-of-a", data)
    assert fake.put_calls == sent_before                       # nothing was transferred
    s = store.stats()
    assert s["dedup_ratio"] == 2.0 and s["unique_chunks"] == 5 and s["chunk_references"] == 10
    assert s["chunks_deduplicated"] == 5


def test_partial_overlap_only_sends_new_chunks(store, fake):
    a = rand(3000)
    store.put_object("bkt", "a", a)
    fake.put_calls = 0
    store.put_object("bkt", "b", a + rand(1000, seed=2))      # same 3 chunks + 1 new
    assert fake.put_calls == 3                                 # 1 new chunk x 3 replicas


def test_delete_keeps_shared_chunks_until_last_reference(store, fake):
    a = rand(3000)
    ma = store.put_object("bkt", "a", a)
    mb = store.put_object("bkt", "b", a + rand(1000, seed=2))
    only_b = mb.chunk_hashes[-1]
    shared = ma.chunk_hashes[0]

    assert store.delete_object("bkt", "b") == 1
    assert fake.holders(only_b) == []                          # b's private chunk is gone from every node
    assert len(fake.holders(shared)) == 3                      # shared chunk still stored
    assert store.get_object("bkt", "a")[1] == a                # and a is intact

    store.delete_object("bkt", "a")
    assert all(fake.holders(h) == [] for h in ma.chunk_hashes)
    assert store.stats()["unique_chunks"] == 0
    assert store.delete_object("bkt", "a") == 0                # idempotent


def test_versions(store):
    store.put_object("bkt", "k", b"first")
    store.put_object("bkt", "k", b"second")
    assert store.get_object("bkt", "k")[1] == b"second"
    assert store.get_object("bkt", "k", version=1)[1] == b"first"
    store.delete_object("bkt", "k", version=2)
    assert store.get_object("bkt", "k")[1] == b"first"


def test_read_survives_down_and_corrupted_replicas(store, fake):
    data = rand(2500)
    meta = store.put_object("bkt", "f", data)
    h = meta.chunk_hashes[0]
    first, second, _third = store.placement(h)

    fake.down.add(first)                                       # primary unreachable
    assert store.get_object("bkt", "f")[1] == data

    fake.down.clear()
    fake.disk[first][h] = b"bit rot"                           # primary returns garbage
    assert store.get_object("bkt", "f")[1] == data             # hash check rejects it, next replica used

    for n in (first, second, _third):
        fake.disk[n][h] = b"all rotten"
    with pytest.raises(ChunkUnavailable):
        store.get_object("bkt", "f")


def test_write_succeeds_with_one_node_down(store, fake):
    fake.down.add("node1")
    meta = store.put_object("bkt", "f", rand(3000))            # needs 2 of 3 replicas
    assert store.get_object("bkt", "f")[1] == rand(3000)
    assert all(len(fake.holders(h)) >= 2 for h in meta.chunk_hashes)


def test_failed_write_is_rolled_back_completely(store, fake, db):
    store.put_object("bkt", "keep", rand(2000, seed=9))        # unrelated data must be untouched
    before = store.stats()["unique_chunks"]
    fake.down.update({"node1", "node2", "node3"})              # only node4 left: cannot reach 2 replicas
    with pytest.raises(InsufficientReplicas):
        store.put_object("bkt", "doomed", rand(3000, seed=5))
    fake.down.clear()
    assert store.head_object("bkt", "doomed") is None
    assert store.stats()["unique_chunks"] == before
    assert store.get_object("bkt", "keep")[1] == rand(2000, seed=9)
    # no orphan chunks left behind: the nodes hold exactly the surviving object's chunks
    on_disk = {h for disk in fake.disk.values() for h in disk}
    assert on_disk == set(store.head_object("bkt", "keep").chunk_hashes)


def test_chunk_known_to_db_but_lost_from_a_node_is_restored_by_next_put(store, fake):
    data = rand(900)
    meta = store.put_object("bkt", "a", data)
    h = meta.chunk_hashes[0]
    victim = store.placement(h)[0]
    del fake.disk[victim][h]                                   # node lost the file
    store.put_object("bkt", "b", data)                         # dedup path notices and re-sends
    assert victim in fake.holders(h)


def test_objects_remain_readable_after_ring_membership_changes(store, fake):
    data = rand(6000)
    store.put_object("bkt", "f", data)
    store.add_node("node5", "http://node5:8000")               # placement changes, data not moved yet
    assert store.get_object("bkt", "f")[1] == data             # fallback finds the old copies
    store.remove_node("node1")
    assert store.get_object("bkt", "f")[1] == data


def test_cdc_chunker_roundtrip_and_dedup_after_edit(db, fake):
    s = ObjectStore(db, fake, replicas=2, chunker="cdc", chunk_size=20_000)
    for i in range(1, 5):
        s.add_node(f"node{i}", f"http://node{i}")
    base = rand(400_000)
    edited = base[:500] + b"XX" + base[500:]
    m1 = s.put_object("bkt", "v1", base)
    m2 = s.put_object("bkt", "v2", edited)
    assert m1.chunker == "cdc" and s.get_object("bkt", "v2")[1] == edited
    shared = set(m1.chunk_hashes) & set(m2.chunk_hashes)
    assert len(shared) / len(set(m1.chunk_hashes)) > 0.8       # small edit -> most chunks reused
    assert s.stats()["dedup_ratio"] > 1.6


def test_put_with_empty_ring_fails_cleanly(db, fake):
    with pytest.raises(Exception, match="no storage nodes"):
        ObjectStore(db, fake).put_object("bkt", "k", b"x")


def test_list_objects(store):
    for k in ("logs/a", "logs/b", "img/c"):
        store.put_object("bkt", k, b"x" + k.encode())
    assert [m.key for m in store.list_objects("bkt", "logs/")] == ["logs/a", "logs/b"]
