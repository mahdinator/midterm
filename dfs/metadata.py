"""Metadata service (part 1): receives node heartbeats and tracks which nodes are alive.

    POST /heartbeat        called by every storage node every few seconds
    GET  /nodes            all known nodes with status (?status=alive|dead to filter)
    GET  /nodes/{node_id}  one node
    GET  /health           service health + node counts

A node is "alive" while its last heartbeat is newer than DEAD_AFTER seconds,
otherwise "dead". State is kept in memory only: after a restart the registry
refills within one heartbeat interval, because nodes keep sending.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

log = logging.getLogger("dfs.metadata")


class Heartbeat(BaseModel):
    node_id: str
    url: str
    free_bytes: int
    chunk_count: int
    timestamp: float | None = None


@dataclass
class Settings:
    dead_after: float = 15.0      # seconds without a heartbeat before a node is dead (3 missed beats)
    sweep_interval: float = 2.0   # how often the background check looks for newly dead nodes

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            dead_after=float(os.getenv("DEAD_AFTER", "15")),
            sweep_interval=float(os.getenv("SWEEP_INTERVAL", "2")),
        )


class Registry:
    def __init__(self, dead_after: float, clock: Callable[[], float] = time.monotonic):
        self.dead_after = dead_after
        self.clock = clock
        self._nodes: dict[str, dict] = {}
        self._announced_dead: set[str] = set()

    def heartbeat(self, hb: Heartbeat) -> str:
        """Record a heartbeat. Returns 'joined', 'revived' or 'updated'."""
        known = hb.node_id in self._nodes
        was_dead = hb.node_id in self._announced_dead
        self._nodes[hb.node_id] = {**hb.model_dump(exclude={"timestamp"}), "_seen": self.clock()}
        self._announced_dead.discard(hb.node_id)
        return "revived" if was_dead else ("updated" if known else "joined")

    def status_of(self, node_id: str) -> str:
        age = self.clock() - self._nodes[node_id]["_seen"]
        return "alive" if age <= self.dead_after else "dead"

    def view(self, node_id: str) -> dict:
        n = self._nodes[node_id]
        return {
            **{k: v for k, v in n.items() if not k.startswith("_")},
            "status": self.status_of(node_id),
            "seconds_since_heartbeat": round(self.clock() - n["_seen"], 1),
        }

    def all(self, status: str | None = None) -> list[dict]:
        views = [self.view(i) for i in sorted(self._nodes)]
        return [v for v in views if status is None or v["status"] == status]

    def sweep(self) -> list[str]:
        """Return nodes that became dead since the last sweep (each reported once)."""
        newly = [i for i in self._nodes if self.status_of(i) == "dead" and i not in self._announced_dead]
        self._announced_dead.update(newly)
        return sorted(newly)


async def sweep_loop(registry: Registry, interval: float) -> None:
    while True:
        for node_id in registry.sweep():
            log.warning("node %s declared DEAD (no heartbeat for >%ss)", node_id, registry.dead_after)
        await asyncio.sleep(interval)


def create_app(settings: Settings | None = None, clock: Callable[[], float] = time.monotonic,
               monitor: bool = True) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    settings = settings or Settings.from_env()
    registry = Registry(settings.dead_after, clock)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(sweep_loop(registry, settings.sweep_interval)) if monitor else None
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="metadata-service", lifespan=lifespan)
    app.state.registry = registry

    @app.post("/heartbeat", status_code=204)
    async def heartbeat(hb: Heartbeat):
        event = registry.heartbeat(hb)
        if event == "joined":
            log.info("node %s joined (%s)", hb.node_id, hb.url)
        elif event == "revived":
            log.info("node %s is back alive", hb.node_id)
        return Response(status_code=204)

    @app.get("/nodes")
    async def list_nodes(status: Literal["alive", "dead"] | None = None):
        return registry.all(status)

    @app.get("/nodes/{node_id}")
    async def get_node(node_id: str):
        if node_id not in registry._nodes:
            raise HTTPException(404, "unknown node")
        return registry.view(node_id)

    @app.get("/health")
    async def health():
        nodes = registry.all()
        return {
            "status": "ok",
            "nodes_total": len(nodes),
            "nodes_alive": sum(n["status"] == "alive" for n in nodes),
        }

    return app
