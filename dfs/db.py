"""Metadata store on PostgreSQL.

Three tables:

  objects     one row per (bucket, key, version): size, chunker used, creation time and the
              ORDERED list of chunk hashes that make up the content
  chunks      one row per unique chunk: size and `refcount`, the number of references to it
              from object versions (a chunk used twice in one object counts twice)
  ring_nodes  which storage nodes are members of the placement ring, and their URLs

Deduplication rule: a chunk is stored once no matter how many objects use it. Its row is
removed (and the caller is told to delete the bytes from the nodes) only when refcount
drops to 0.

Safety of concurrent put/delete: a put first RESERVES its chunks (refcount += n) before
writing any bytes, and only afterwards commits the object row. A delete that frees a chunk
runs its physical deletion while still holding the chunk's row lock, so a concurrent put of
the same chunk waits until the delete is finished, then stores it again. A crash between
reserve and commit leaves refcount too HIGH (a harmless leak a later GC can repair), never
too low (which would lose data).
"""

from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Iterator

import psycopg
from psycopg import sql

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ring_nodes (
    node_id   TEXT PRIMARY KEY,
    url       TEXT NOT NULL,
    added_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    hash       TEXT PRIMARY KEY CHECK (length(hash) = 64),
    size       INTEGER NOT NULL,
    refcount   INTEGER NOT NULL CHECK (refcount >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS objects (
    bucket       TEXT NOT NULL,
    key          TEXT NOT NULL,
    version      INTEGER NOT NULL,
    size         BIGINT NOT NULL,
    chunker      TEXT NOT NULL,
    chunk_hashes TEXT[] NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bucket, key, version)
);
"""

OnUnreferenced = Callable[[list[str]], None]


@dataclass
class ObjectMeta:
    bucket: str
    key: str
    version: int
    size: int
    chunker: str
    chunk_hashes: list[str]
    created_at: datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


_OBJ_COLS = "bucket, key, version, size, chunker, chunk_hashes, created_at"


def _meta(row) -> ObjectMeta:
    return ObjectMeta(*row)


class MetadataDB:
    def __init__(self, dsn: str, schema: str | None = None):
        """schema: optional Postgres schema to keep all tables in (used by tests so they
        never touch real data)."""
        self.dsn = dsn
        self.schema = schema

    # --- plumbing -------------------------------------------------------------------
    @contextlib.contextmanager
    def _tx(self) -> Iterator[psycopg.Connection]:
        """One transaction on a fresh connection: commits on success, rolls back on error."""
        kwargs = {"options": f"-c search_path={self.schema}"} if self.schema else {}
        with psycopg.connect(self.dsn, **kwargs) as conn:
            yield conn

    def init_schema(self) -> None:
        if self.schema:
            with psycopg.connect(self.dsn) as conn:
                conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
        with self._tx() as conn:
            conn.execute(SCHEMA_SQL)

    def drop_schema(self) -> None:
        if self.schema:
            with psycopg.connect(self.dsn) as conn:
                conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.schema)))

    # --- ring membership ---------------------------------------------------------------
    def add_ring_node(self, node_id: str, url: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO ring_nodes (node_id, url) VALUES (%s, %s) "
                "ON CONFLICT (node_id) DO UPDATE SET url = EXCLUDED.url", (node_id, url))

    def remove_ring_node(self, node_id: str) -> bool:
        with self._tx() as conn:
            return conn.execute("DELETE FROM ring_nodes WHERE node_id = %s", (node_id,)).rowcount > 0

    def ring_nodes(self) -> dict[str, str]:
        with self._tx() as conn:
            return dict(conn.execute("SELECT node_id, url FROM ring_nodes ORDER BY node_id").fetchall())

    # --- chunk reference counting ----------------------------------------------------
    def reserve_chunks(self, sizes: dict[str, int], counts: Counter) -> set[str]:
        """Add counts[h] references to every chunk. Returns the hashes that did not exist
        before (the caller must write those to the nodes; the others are deduplicated)."""
        new: set[str] = set()
        with self._tx() as conn:
            for h in sorted(counts):                    # fixed order avoids deadlocks
                inserted = conn.execute(
                    "INSERT INTO chunks (hash, size, refcount) VALUES (%s, %s, %s) "
                    "ON CONFLICT (hash) DO UPDATE SET refcount = chunks.refcount + EXCLUDED.refcount "
                    "RETURNING (xmax = 0)", (h, sizes[h], counts[h])).fetchone()[0]
                if inserted:
                    new.add(h)
        return new

    def release_chunks(self, counts: Counter, on_unreferenced: OnUnreferenced | None = None) -> list[str]:
        """Undo a reservation (used when an upload fails part-way)."""
        with self._tx() as conn:
            return self._release(conn, counts, on_unreferenced)

    @staticmethod
    def _release(conn, counts: Counter, on_unreferenced: OnUnreferenced | None) -> list[str]:
        freed: list[str] = []
        for h in sorted(counts):
            row = conn.execute("UPDATE chunks SET refcount = refcount - %s WHERE hash = %s RETURNING refcount",
                               (counts[h], h)).fetchone()
            if row is not None and row[0] == 0:
                freed.append(h)
        if freed:
            if on_unreferenced:
                on_unreferenced(freed)                  # row locks are still held here
            conn.execute("DELETE FROM chunks WHERE hash = ANY(%s) AND refcount = 0", (freed,))
        return freed

    def refcount(self, chunk_hash: str) -> int | None:
        with self._tx() as conn:
            row = conn.execute("SELECT refcount FROM chunks WHERE hash = %s", (chunk_hash,)).fetchone()
        return row[0] if row else None

    # --- objects --------------------------------------------------------------------
    def commit_object(self, bucket: str, key: str, size: int, chunker: str, chunk_hashes: list[str]) -> ObjectMeta:
        """Record a new version of bucket/key. The chunks must already be reserved."""
        with self._tx() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{bucket}/{key}",))
            version = conn.execute("SELECT coalesce(max(version), 0) + 1 FROM objects WHERE bucket=%s AND key=%s",
                                   (bucket, key)).fetchone()[0]
            row = conn.execute(
                f"INSERT INTO objects (bucket, key, version, size, chunker, chunk_hashes) "
                f"VALUES (%s, %s, %s, %s, %s, %s) RETURNING {_OBJ_COLS}",
                (bucket, key, version, size, chunker, chunk_hashes)).fetchone()
        return _meta(row)

    def get_object(self, bucket: str, key: str, version: int | None = None) -> ObjectMeta | None:
        with self._tx() as conn:
            if version is None:
                row = conn.execute(f"SELECT {_OBJ_COLS} FROM objects WHERE bucket=%s AND key=%s "
                                   f"ORDER BY version DESC LIMIT 1", (bucket, key)).fetchone()
            else:
                row = conn.execute(f"SELECT {_OBJ_COLS} FROM objects WHERE bucket=%s AND key=%s AND version=%s",
                                   (bucket, key, version)).fetchone()
        return _meta(row) if row else None

    def list_versions(self, bucket: str, key: str) -> list[ObjectMeta]:
        with self._tx() as conn:
            rows = conn.execute(f"SELECT {_OBJ_COLS} FROM objects WHERE bucket=%s AND key=%s ORDER BY version",
                                (bucket, key)).fetchall()
        return [_meta(r) for r in rows]

    def list_objects(self, bucket: str, prefix: str = "", limit: int = 1000) -> list[ObjectMeta]:
        """Latest version of every key in the bucket whose name starts with prefix."""
        with self._tx() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT ON (key) {_OBJ_COLS} FROM objects "
                f"WHERE bucket=%s AND starts_with(key, %s) ORDER BY key, version DESC LIMIT %s",
                (bucket, prefix, limit)).fetchall()
        return [_meta(r) for r in rows]

    def delete_object(self, bucket: str, key: str, version: int | None = None,
                      on_unreferenced: OnUnreferenced | None = None) -> tuple[int, list[str]]:
        """Delete one version (or ALL versions when version is None). Chunks whose refcount
        reaches 0 are passed to on_unreferenced and removed. Returns (versions deleted,
        chunk hashes freed)."""
        with self._tx() as conn:
            if version is None:
                rows = conn.execute("DELETE FROM objects WHERE bucket=%s AND key=%s RETURNING chunk_hashes",
                                    (bucket, key)).fetchall()
            else:
                rows = conn.execute("DELETE FROM objects WHERE bucket=%s AND key=%s AND version=%s "
                                    "RETURNING chunk_hashes", (bucket, key, version)).fetchall()
            counts = Counter(h for (hashes,) in rows for h in hashes)
            freed = self._release(conn, counts, on_unreferenced)
        return len(rows), freed

    # --- reporting ----------------------------------------------------------------
    def stats(self) -> dict:
        with self._tx() as conn:
            objects, logical, unique, physical, refs = conn.execute(
                "SELECT (SELECT count(*) FROM objects), (SELECT coalesce(sum(size), 0) FROM objects), "
                "(SELECT count(*) FROM chunks), (SELECT coalesce(sum(size), 0) FROM chunks), "
                "(SELECT coalesce(sum(refcount), 0) FROM chunks)").fetchone()
        return {
            "object_versions": objects,
            "logical_bytes": int(logical),              # what users stored, counting every copy
            "unique_chunks": unique,
            "chunk_references": int(refs),              # how many times those chunks are referenced
            "unique_bytes": int(physical),              # bytes of one copy of each unique chunk
            "saved_bytes": int(logical) - int(physical),
            "dedup_ratio": round(int(logical) / int(physical), 3) if physical else None,
        }
