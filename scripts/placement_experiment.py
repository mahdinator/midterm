#!/usr/bin/env python3
"""Measure how much data moves when a node is added to / removed from the cluster.

    python scripts/placement_experiment.py                 # 20,000 chunks, 128 vnodes, 3 replicas
    python scripts/placement_experiment.py --chunks 100000 --vnodes 256 --replicas 2

Reports, for the consistent hash ring and for naive `hash % N` placement:
  * load balance across the nodes (what share of chunks each node is primary for)
  * adding a 5th node to 4:   % of chunks whose primary copy moves; % of replica
                              copies that have to be created somewhere new
  * removing one of 4 nodes:  the same two numbers
The ideal for a ring is 1/N of the data: 20% when going 4 -> 5, 25% when going 4 -> 3.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfs.ring import HashRing  # noqa: E402


def make_hashes(n: int) -> list[str]:
    return [hashlib.sha256(f"chunk-{i}".encode()).hexdigest() for i in range(n)]


def mod_place(h: str, nodes: list[str], n: int) -> list[str]:
    """Naive placement: primary = hash mod N, replicas = the next nodes in list order."""
    start = int(h[:16], 16) % len(nodes)
    return [nodes[(start + k) % len(nodes)] for k in range(min(n, len(nodes)))]


def moved(hashes, place_old, place_new, replicas: int) -> tuple[float, float]:
    """(% of chunks whose PRIMARY changed, % of replica copies that must be created anew)."""
    primary = sum(place_old(h, 1)[0] != place_new(h, 1)[0] for h in hashes)
    new_copies = sum(len(set(place_new(h, replicas)) - set(place_old(h, replicas))) for h in hashes)
    return primary / len(hashes), new_copies / (len(hashes) * replicas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=20_000)
    ap.add_argument("--vnodes", type=int, default=128)
    ap.add_argument("--replicas", type=int, default=3)
    a = ap.parse_args()
    hashes = make_hashes(a.chunks)
    four = [f"node{i}" for i in range(1, 5)]
    five = four + ["node5"]
    three = four[:-1]                      # node4 removed

    print(f"{a.chunks} chunks, {a.vnodes} virtual nodes per node, {a.replicas} replicas\n")

    print("LOAD BALANCE on 4 nodes (share of chunks each node is primary for; ideal 25% each)")
    print(f"{'vnodes':>7} {'min':>7} {'max':>7} {'max/min':>8} {'stdev':>7}")
    for v in (1, 8, 32, 128, 256):
        ring = HashRing(four, vnodes=v)
        counts = {n: 0 for n in four}
        for h in hashes:
            counts[ring.get_nodes(h)[0]] += 1
        shares = [c / len(hashes) for c in counts.values()]
        print(f"{v:>7} {min(shares):>7.1%} {max(shares):>7.1%} {max(shares) / min(shares):>8.2f} "
              f"{statistics.pstdev(shares):>7.2%}")

    ring4, ring5, ring3 = HashRing(four, a.vnodes), HashRing(five, a.vnodes), HashRing(three, a.vnodes)
    ring_place = lambda r: (lambda h, n: r.get_nodes(h, n))            # noqa: E731
    mod_place_for = lambda nodes: (lambda h, n: mod_place(h, nodes, n))  # noqa: E731

    print("\nDATA MOVEMENT")
    print(f"{'scenario':26} {'placement':12} {'primary moved':>14} {'replica copies moved':>21} {'ideal':>7}")
    rows = [
        ("add node5 (4 -> 5 nodes)", ring_place(ring4), ring_place(ring5), mod_place_for(four), mod_place_for(five), 1 / 5),
        ("remove node4 (4 -> 3)", ring_place(ring4), ring_place(ring3), mod_place_for(four), mod_place_for(three), 1 / 4),
    ]
    for label, r_old, r_new, m_old, m_new, ideal in rows:
        for name, old, new in (("hash ring", r_old, r_new), ("hash mod N", m_old, m_new)):
            p, c = moved(hashes, old, new, a.replicas)
            print(f"{label:26} {name:12} {p:>14.1%} {c:>21.1%} {ideal:>7.0%}")

    only_new = all(ring5.get_nodes(h)[0] == "node5" for h in hashes if ring5.get_nodes(h)[0] != ring4.get_nodes(h)[0])
    only_lost = all(ring4.get_nodes(h)[0] == "node4" for h in hashes if ring4.get_nodes(h)[0] != ring3.get_nodes(h)[0])
    print(f"\nring: when node5 joins, every chunk that moves goes TO node5:      {only_new}")
    print(f"ring: when node4 leaves, every chunk that moves was ON node4:       {only_lost}")


if __name__ == "__main__":
    main()
