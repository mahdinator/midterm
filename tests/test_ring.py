import hashlib

import pytest

from dfs.ring import HashRing

HASHES = [hashlib.sha256(f"c{i}".encode()).hexdigest() for i in range(10_000)]
FOUR = ["node1", "node2", "node3", "node4"]


def test_empty_ring_returns_nothing():
    assert HashRing().get_nodes(HASHES[0], 3) == []


def test_placement_is_deterministic_and_order_independent():
    a, b = HashRing(FOUR), HashRing(list(reversed(FOUR)))
    assert all(a.get_nodes(h, 3) == b.get_nodes(h, 3) for h in HASHES[:500])


def test_returns_distinct_nodes_primary_first():
    ring = HashRing(FOUR)
    for h in HASHES[:500]:
        nodes = ring.get_nodes(h, 3)
        assert len(nodes) == len(set(nodes)) == 3
        assert nodes[0] == ring.get_nodes(h, 1)[0]
        assert nodes[:2] == ring.get_nodes(h, 2)        # smaller n is a prefix


def test_n_larger_than_cluster_returns_all_nodes():
    assert sorted(HashRing(["a", "b"]).get_nodes(HASHES[0], 5)) == ["a", "b"]


def test_load_is_reasonably_balanced():
    ring = HashRing(FOUR, vnodes=128)
    counts = {n: 0 for n in FOUR}
    for h in HASHES:
        counts[ring.get_nodes(h)[0]] += 1
    assert all(0.20 < c / len(HASHES) < 0.30 for c in counts.values())


def test_adding_a_node_moves_about_one_over_n_and_only_to_the_new_node():
    old, new = HashRing(FOUR), HashRing(FOUR + ["node5"])
    moved = [h for h in HASHES if old.get_nodes(h)[0] != new.get_nodes(h)[0]]
    assert 0.14 < len(moved) / len(HASHES) < 0.26            # ideal 20%
    assert all(new.get_nodes(h)[0] == "node5" for h in moved)


def test_removing_a_node_moves_only_its_chunks():
    old = HashRing(FOUR)
    new = old.copy()
    new.remove_node("node4")
    moved = [h for h in HASHES if old.get_nodes(h)[0] != new.get_nodes(h)[0]]
    assert 0.19 < len(moved) / len(HASHES) < 0.31            # ideal 25%
    assert all(old.get_nodes(h)[0] == "node4" for h in moved)
    assert "node4" not in {n for h in HASHES[:500] for n in new.get_nodes(h, 3)}


def test_add_then_remove_restores_original_placement():
    ring = HashRing(FOUR)
    before = {h: ring.get_nodes(h, 3) for h in HASHES[:1000]}
    ring.add_node("node5")
    ring.remove_node("node5")
    assert before == {h: ring.get_nodes(h, 3) for h in HASHES[:1000]}


def test_modulo_hashing_moves_most_data_for_contrast():
    def place(h, n):
        return int(h[:16], 16) % n
    moved = sum(place(h, 4) != place(h, 5) for h in HASHES)
    assert moved / len(HASHES) > 0.7


def test_add_is_idempotent_and_remove_unknown_is_noop():
    ring = HashRing(FOUR)
    ring.add_node("node1")
    ring.remove_node("ghost")
    assert ring.nodes == FOUR
    assert len(ring._positions) == 4 * ring.vnodes


def test_more_vnodes_means_better_balance():
    def spread(v):
        ring = HashRing(FOUR, vnodes=v)
        c = {n: 0 for n in FOUR}
        for h in HASHES:
            c[ring.get_nodes(h)[0]] += 1
        return max(c.values()) - min(c.values())
    assert spread(128) < spread(1)


def test_invalid_vnodes():
    with pytest.raises(ValueError):
        HashRing(vnodes=0)
