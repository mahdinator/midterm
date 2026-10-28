"""Storage node: a content-addressed chunk store exposed over HTTP.

    PUT    /chunks/{sha256}   store a chunk (body must hash to the id in the URL)
    GET    /chunks/{sha256}   fetch a chunk (hash is re-verified on every read)
    HEAD   /chunks/{sha256}   existence + size, no body, no hash check
    DELETE /chunks/{sha256}   remove a chunk
    GET    /status            same data the node sends in its heartbeat

Layout on disk:  <DATA_DIR>/chunks/<first 2 hex chars>/<sha256>
Writes go to <DATA_DIR>/tmp first and are renamed into place, so a crash never
leaves a half-written chunk under its final name.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi import Path as PathParam

log = logging.getLogger("dfs.node")

ChunkId = Annotated[str, PathParam(pattern=r"^[0-9a-f]{64}$")]  # also blocks path traversal


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Settings:
    node_id: str
    data_dir: Path
    node_url: str            # how other services reach this node
    metadata_url: str        # where heartbeats are sent
    heartbeat_interval: float = 5.0
    max_chunk_bytes: int = 64 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        node_id = os.environ["NODE_ID"]
        port = os.getenv("PORT", "8000")
        return cls(
            node_id=node_id,
            data_dir=Path(os.getenv("DATA_DIR", "/data")),
            node_url=os.getenv("NODE_URL", f"http://{node_id}:{port}"),
            metadata_url=os.getenv("METADATA_URL", "http://metadata:8000"),
            heartbeat_interval=float(os.getenv("HEARTBEAT_INTERVAL", "5")),
            max_chunk_bytes=int(os.getenv("MAX_CHUNK_BYTES", 64 * 1024 * 1024)),
        )


class ChunkStore:
    def __init__(self, root: Path):
        self.root = root
        self.chunks = root / "chunks"
        self.tmp = root / "tmp"
        self.chunks.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        for leftover in self.tmp.iterdir():  # partial writes from a previous crash
            leftover.unlink(missing_ok=True)
        self._lock = threading.Lock()
        self.count = sum(1 for p in self.chunks.glob("*/*") if p.is_file())

    def path(self, h: str) -> Path:
        return self.chunks / h[:2] / h

    def put(self, h: str, data: bytes) -> bool:
        """Store data. Returns True if a new chunk was created, False if it already
        existed intact. An existing but corrupted file is overwritten (self-heal)."""
        p = self.path(h)
        tmp = self.tmp / uuid.uuid4().hex
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        with self._lock:
            existed = p.exists()
            if existed and sha256_hex(p.read_bytes()) == h:
                tmp.unlink(missing_ok=True)
                return False
            p.parent.mkdir(exist_ok=True)
            os.replace(tmp, p)
            if not existed:
                self.count += 1
            return not existed

    def read(self, h: str) -> bytes | None:
        try:
            return self.path(h).read_bytes()
        except FileNotFoundError:
            return None

    def size(self, h: str) -> int | None:
        try:
            return self.path(h).stat().st_size
        except FileNotFoundError:
            return None

    def delete(self, h: str) -> bool:
        with self._lock:
            try:
                self.path(h).unlink()
            except FileNotFoundError:
                return False
            self.count -= 1
            return True

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.root).free


def heartbeat_payload(settings: Settings, store: ChunkStore) -> dict:
    return {
        "node_id": settings.node_id,
        "url": settings.node_url,
        "free_bytes": store.free_bytes(),
        "chunk_count": store.count,
        "timestamp": time.time(),
    }


async def send_heartbeat(client: httpx.AsyncClient, settings: Settings, store: ChunkStore) -> None:
    r = await client.post(f"{settings.metadata_url}/heartbeat", json=heartbeat_payload(settings, store))
    r.raise_for_status()


async def heartbeat_loop(settings: Settings, store: ChunkStore) -> None:
    healthy: bool | None = None  # log only on state changes, not every few seconds
    async with httpx.AsyncClient(timeout=3.0) as client:
        while True:
            try:
                await send_heartbeat(client, settings, store)
                if healthy is not True:
                    log.info("heartbeat to %s ok", settings.metadata_url)
                healthy = True
            except httpx.HTTPError as e:
                if healthy is not False:
                    log.warning("heartbeat to %s failed: %r (will keep retrying)", settings.metadata_url, e)
                healthy = False
            await asyncio.sleep(settings.heartbeat_interval)


def create_app(settings: Settings | None = None, heartbeat: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    store = ChunkStore(settings.data_dir)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(heartbeat_loop(settings, store)) if heartbeat else None
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title=f"storage-node {settings.node_id}", lifespan=lifespan)
    app.state.settings, app.state.store = settings, store

    @app.put("/chunks/{h}")
    async def put_chunk(h: ChunkId, request: Request, response: Response):
        data = await request.body()
        if len(data) > settings.max_chunk_bytes:
            raise HTTPException(413, "chunk too large")
        if await asyncio.to_thread(sha256_hex, data) != h:
            raise HTTPException(400, "body does not match the sha256 in the URL")
        created = await asyncio.to_thread(store.put, h, data)
        response.status_code = 201 if created else 200
        return {"hash": h, "size": len(data), "created": created}

    @app.get("/chunks/{h}")
    async def get_chunk(h: ChunkId):
        data = await asyncio.to_thread(store.read, h)
        if data is None:
            raise HTTPException(404, "chunk not found")
        if await asyncio.to_thread(sha256_hex, data) != h:
            log.error("chunk %s is corrupted on disk, refusing to serve it", h)
            raise HTTPException(409, "chunk corrupted on disk")
        return Response(data, media_type="application/octet-stream", headers={"X-Chunk-Hash": h})

    @app.head("/chunks/{h}")
    async def head_chunk(h: ChunkId):
        size = await asyncio.to_thread(store.size, h)
        if size is None:
            raise HTTPException(404, "chunk not found")
        return Response(status_code=200, headers={"Content-Length": str(size), "X-Chunk-Hash": h})

    @app.delete("/chunks/{h}", status_code=204)
    async def delete_chunk(h: ChunkId):
        if not await asyncio.to_thread(store.delete, h):
            raise HTTPException(404, "chunk not found")

    @app.get("/status")
    async def status():
        return heartbeat_payload(settings, store)

    return app
