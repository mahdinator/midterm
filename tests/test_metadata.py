import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from dfs import metadata
from dfs.metadata import Settings, create_app


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def hb(node_id="node1", **kw):
    return {"node_id": node_id, "url": f"http://{node_id}:8000", "free_bytes": 123,
            "chunk_count": 0, "timestamp": 1.0, **kw}


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def client(clock):
    return TestClient(create_app(Settings(dead_after=15), clock=clock, monitor=False))


def test_heartbeat_registers_node(client):
    assert client.post("/heartbeat", json=hb()).status_code == 204
    (n,) = client.get("/nodes").json()
    assert n["node_id"] == "node1" and n["status"] == "alive"
    assert n["free_bytes"] == 123 and n["url"] == "http://node1:8000"


def test_heartbeat_updates_existing_node(client):
    client.post("/heartbeat", json=hb(chunk_count=1))
    client.post("/heartbeat", json=hb(chunk_count=7))
    nodes = client.get("/nodes").json()
    assert len(nodes) == 1 and nodes[0]["chunk_count"] == 7


def test_node_goes_dead_after_timeout_and_revives(client, clock):
    client.post("/heartbeat", json=hb())
    clock.t += 15
    assert client.get("/nodes/node1").json()["status"] == "alive"   # exactly at limit
    clock.t += 0.1
    assert client.get("/nodes/node1").json()["status"] == "dead"
    client.post("/heartbeat", json=hb())
    assert client.get("/nodes/node1").json()["status"] == "alive"


def test_status_filter_and_health(client, clock):
    client.post("/heartbeat", json=hb("node1"))
    clock.t += 10
    client.post("/heartbeat", json=hb("node2"))
    clock.t += 10                                   # node1 is 20s old, node2 is 10s old
    assert [n["node_id"] for n in client.get("/nodes?status=alive").json()] == ["node2"]
    assert [n["node_id"] for n in client.get("/nodes?status=dead").json()] == ["node1"]
    assert client.get("/health").json() == {"status": "ok", "nodes_total": 2, "nodes_alive": 1}
    assert client.get("/nodes?status=bogus").status_code == 422


def test_unknown_node_is_404(client):
    assert client.get("/nodes/nope").status_code == 404


def test_invalid_heartbeat_rejected(client):
    assert client.post("/heartbeat", json={"node_id": "node1"}).status_code == 422


def test_sweep_reports_each_death_once(clock):
    reg = metadata.Registry(dead_after=15, clock=clock)
    reg.heartbeat(metadata.Heartbeat(**hb()))
    assert reg.sweep() == []
    clock.t += 20
    assert reg.sweep() == ["node1"]
    assert reg.sweep() == []                        # not announced again
    assert reg.heartbeat(metadata.Heartbeat(**hb())) == "revived"
    clock.t += 20
    assert reg.sweep() == ["node1"]                 # dies again -> announced again


def test_real_node_heartbeat_is_accepted_by_metadata(tmp_path):
    """The exact payload a storage node sends must be accepted by this service."""
    from dfs.node import Settings as NodeSettings, create_app as create_node, send_heartbeat

    meta = create_app(Settings(), monitor=False)
    ns = NodeSettings(node_id="node3", data_dir=tmp_path, node_url="http://node3:8000",
                      metadata_url="http://metadata")
    node = create_node(ns, heartbeat=False)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=meta)) as c:
            await send_heartbeat(c, ns, node.state.store)

    asyncio.run(go())
    (n,) = TestClient(meta).get("/nodes").json()
    assert n["node_id"] == "node3" and n["status"] == "alive" and n["free_bytes"] > 0
