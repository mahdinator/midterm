"""ObjectStore: the core that the S3-like API will wrap.

put    split the data into chunks -> hash each chunk -> reserve references in Postgres
       (dedup: chunks that already exist are not sent again) -> write new chunks to the
       `replicas` nodes the hash ring picks -> record the object version
get    look up the ordered chunk list -> fetch each chunk from its ring nodes (falling back
       to the other nodes), verifying its SHA-256 -> reassemble
delete drop the version(s), decrement chunk references, and delete from the nodes only the
       chunks that no object references any more

Not done here (later tasks): moving existing chunks after the ring changes (rebalancing),
re-replicating after a node failure (repair), erasure coding, periodic scrubbing.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Protocol

from dfs.chunking import DEFAULT_CHUNK_SIZE, get_chunker, sha256_hex
from dfs.db import MetadataDB, ObjectMeta
from dfs.nodeclient import NodeError
from dfs.ring import DEFAULT_VNODES, HashRing

log = logging.getLogger("dfs.objectstore")


class StoreError(Exception):
    pass


class ObjectNotFound(StoreError):
    pass


class InsufficientReplicas(StoreError):
    """Too few nodes accepted a chunk, so the upload was rolled back."""


class ChunkUnavailable(StoreError):
    """No node could supply a valid copy of a chunk."""


class Nodes(Protocol):
    """What ObjectStore needs from the node layer (NodeClient implements it)."""
    def set_urls(self, urls: dict[str, str]) -> None: ...
    def put(self, node_id: str, h: str, data: bytes) -> None: ...
    def get(self, node_id: str, h: str) -> bytes | None: ...
    def has(self, node_id: str, h: str) -> bool: ...
    def delete(self, node_id: str, h: str) -> bool: ...


class ObjectStore:
    def __init__(self, db: MetadataDB, nodes: Nodes, replicas: int = 3, min_replicas: int | None = None,
                 chunker: str = "fixed", chunk_size: int = DEFAULT_CHUNK_SIZE, vnodes: int = DEFAULT_VNODES):
        self.db, self.nodes = db, nodes
        self.replicas = replicas
        self.min_replicas = min_replicas if min_replicas is not None else min(2, replicas)
        self.chunker, self.chunk_size, self.vnodes = chunker, chunk_size, vnodes
        # cumulative counters, handy for measuring dedup
        self.counters = Counter()
        self.ring = HashRing(vnodes=vnodes)
        self.refresh_ring()

    # --- ring membership ---------------------------------------------------------------
    def refresh_ring(self) -> None:
        urls = self.db.ring_nodes()
        self.ring = HashRing(list(urls), vnodes=self.vnodes)
        self.nodes.set_urls(urls)

    def add_node(self, node_id: str, url: str) -> None:
        """Add a node to the ring. Existing chunks are NOT moved (rebalancing is a later
        task); reads still find them because get falls back to the other nodes."""
        self.db.add_ring_node(node_id, url)
        self.refresh_ring()

    def remove_node(self, node_id: str) -> None:
        self.db.remove_ring_node(node_id)
        self.refresh_ring()

    def placement(self, chunk_hash: str) -> list[str]:
        return self.ring.get_nodes(chunk_hash, self.replicas)

    # --- put ---------------------------------------------------------------------------------
    def put_object(self, bucket: str, key: str, data: bytes, chunker: str | None = None) -> ObjectMeta:
        if not len(self.ring):
            raise StoreError("no storage nodes in the ring")
        chunker_name = chunker or self.chunker
        chunks = get_chunker(chunker_name, self.chunk_size)(data)
        hashes = [sha256_hex(c) for c in chunks]
        by_hash = dict(zip(hashes, chunks))
        counts = Counter(hashes)

        # 1. reserve references first, so a concurrent delete can never free these chunks
        new = self.db.reserve_chunks({h: len(c) for h, c in by_hash.items()}, counts)
        try:
            # 2. make sure every chunk is on its target nodes
            for h, chunk in by_hash.items():
                self._store_chunk(h, chunk, is_new=h in new)
            # 3. only now does the object become visible
            meta = self.db.commit_object(bucket, key, len(data), chunker_name, hashes)
        except Exception:
            # roll back: drop our references; chunks nobody else uses are removed from the nodes
            try:
                self.db.release_chunks(counts, on_unreferenced=self._purge)
            except Exception:
                log.exception("rollback failed; references may be leaked until a cleanup pass")
            raise
        return meta

    def _store_chunk(self, h: str, chunk: bytes, is_new: bool) -> None:
        targets = self.placement(h)
        need = min(self.min_replicas, len(targets))
        ok = 0
        sent = False
        for node in targets:
            try:
                # a chunk that is new to the cluster is sent without asking; a known chunk is
                # only sent to nodes that turn out not to have it (dedup + self-repair)
                if is_new or not self.nodes.has(node, h):
                    self.nodes.put(node, h, chunk)
                    sent = True
                    self.counters["bytes_sent"] += len(chunk)
                ok += 1
            except NodeError as e:
                log.warning("could not store chunk %s on %s: %s", h[:8], node, e)
        if ok < need:
            raise InsufficientReplicas(f"chunk {h[:8]} stored on {ok}/{need} required nodes")
        self.counters["chunks_sent" if sent else "chunks_deduplicated"] += 1

    # --- get --------------------------------------------------------------------------------
    def head_object(self, bucket: str, key: str, version: int | None = None) -> ObjectMeta | None:
        return self.db.get_object(bucket, key, version)

    def get_object(self, bucket: str, key: str, version: int | None = None) -> tuple[ObjectMeta, bytes]:
        meta = self.db.get_object(bucket, key, version)
        if meta is None:
            raise ObjectNotFound(f"{bucket}/{key}" + (f" version {version}" if version else ""))
        data = b"".join(self._fetch_chunk(h) for h in meta.chunk_hashes)
        if len(data) != meta.size:
            raise StoreError(f"{bucket}/{key}: reassembled {len(data)} bytes, expected {meta.size}")
        return meta, data

    def _fetch_chunk(self, h: str) -> bytes:
        preferred = self.placement(h)
        others = [n for n in self.ring.nodes if n not in preferred]   # e.g. chunks placed before the ring changed
        for node in preferred + others:
            try:
                data = self.nodes.get(node, h)
            except NodeError as e:
                log.warning("read of %s from %s failed: %s", h[:8], node, e)
                continue
            if data is not None and sha256_hex(data) == h:            # never trust a node blindly
                return data
        raise ChunkUnavailable(f"no valid copy of chunk {h[:8]} on any node")

    # --- delete / list ---------------------------------------------------------------------
    def delete_object(self, bucket: str, key: str, version: int | None = None) -> int:
        """Delete one version, or all versions of the key when version is None.
        Returns how many versions were deleted."""
        deleted, _freed = self.db.delete_object(bucket, key, version, on_unreferenced=self._purge)
        return deleted

    def _purge(self, hashes: list[str]) -> None:
        """Remove chunks that no object references from every node (best effort)."""
        for h in hashes:
            for node in self.ring.nodes:
                try:
                    self.nodes.delete(node, h)
                except NodeError as e:
                    log.warning("could not delete orphan chunk %s on %s: %s", h[:8], node, e)
            self.counters["chunks_purged"] += 1

    def list_objects(self, bucket: str, prefix: str = "", limit: int = 1000) -> list[ObjectMeta]:
        return self.db.list_objects(bucket, prefix, limit)

    def stats(self) -> dict:
        return {**self.db.stats(), **dict(self.counters)}
