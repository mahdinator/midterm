"""Consistent hash ring with virtual nodes.

Every node is placed on a 64-bit ring many times ("virtual nodes", `vnodes` per node).
A chunk is placed at the position given by the first 8 bytes of its SHA-256 and is
stored on the first `n` DISTINCT nodes found walking clockwise from there.

Why this instead of `hash % number_of_nodes`:
  * Adding or removing a node only moves the chunks that fall in the ring segments
    that node gains or gives up: about 1/N of the data. With modulo hashing,
    changing N reshuffles almost everything.
  * Many virtual nodes per physical node smooth out the load, because a single
    position per node gives very uneven segments.
"""

from __future__ import annotations

import bisect
import hashlib

DEFAULT_VNODES = 128


def _pos(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def chunk_position(chunk_hash: str) -> int:
    """Ring position of a chunk. Its id is already a SHA-256, so reuse its first 8 bytes."""
    return int(chunk_hash[:16], 16)


class HashRing:
    def __init__(self, nodes: list[str] | None = None, vnodes: int = DEFAULT_VNODES):
        if vnodes < 1:
            raise ValueError("vnodes must be >= 1")
        self.vnodes = vnodes
        self._positions: list[int] = []        # sorted ring positions
        self._owner: dict[int, str] = {}       # position -> node id
        self._nodes: set[str] = set()
        for n in nodes or []:
            self.add_node(n)

    @property
    def nodes(self) -> list[str]:
        return sorted(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def add_node(self, node_id: str) -> None:
        if node_id in self._nodes:
            return
        self._nodes.add(node_id)
        for i in range(self.vnodes):
            p = _pos(f"{node_id}#{i}".encode())
            if p in self._owner:               # astronomically unlikely 64-bit collision
                continue
            self._owner[p] = node_id
            bisect.insort(self._positions, p)

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        self._nodes.discard(node_id)
        self._owner = {p: n for p, n in self._owner.items() if n != node_id}
        self._positions = sorted(self._owner)

    def get_nodes(self, chunk_hash: str, n: int = 1) -> list[str]:
        """The first n distinct nodes clockwise from the chunk's position (primary first)."""
        if not self._positions:
            return []
        n = min(n, len(self._nodes))
        start = bisect.bisect_right(self._positions, chunk_position(chunk_hash))
        found: list[str] = []
        for k in range(len(self._positions)):
            node = self._owner[self._positions[(start + k) % len(self._positions)]]
            if node not in found:
                found.append(node)
                if len(found) == n:
                    break
        return found

    def copy(self) -> "HashRing":
        r = HashRing(vnodes=self.vnodes)
        r._positions = list(self._positions)
        r._owner = dict(self._owner)
        r._nodes = set(self._nodes)
        return r
