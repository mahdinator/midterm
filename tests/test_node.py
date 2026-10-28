import asyncio
import hashlib

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from dfs.node import Settings, create_app, send_heartbeat


def h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def settings(tmp_path):
    return Settings(node_id="node1", data_dir=tmp_path, node_url="http://node1:8000",
                    metadata_url="http://metadata", heartbeat_interval=0.05)


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings, heartbeat=False))


def test_put_get_head_delete_roundtrip(client):
    data = b"hello chunk"
    r = client.put(f"/chunks/{h(data)}", content=data)
    assert r.status_code == 201

    r = client.get(f"/chunks/{h(data)}")
    assert r.status_code == 200 and r.content == data

    r = client.head(f"/chunks/{h(data)}")
    assert r.status_code == 200 and r.headers["content-length"] == str(len(data))

    assert client.delete(f"/chunks/{h(data)}").status_code == 204
    assert client.get(f"/chunks/{h(data)}").status_code == 404
    assert client.head(f"/chunks/{h(data)}").status_code == 404
    assert client.delete(f"/chunks/{h(data)}").status_code == 404


def test_put_rejects_body_that_does_not_match_hash(client):
    r = client.put(f"/chunks/{h(b'a')}", content=b"b")
    assert r.status_code == 400
    assert client.head(f"/chunks/{h(b'a')}").status_code == 404


def test_put_is_idempotent_and_counts_once(client):
    data = b"dedup me"
    assert client.put(f"/chunks/{h(data)}", content=data).status_code == 201
    assert client.put(f"/chunks/{h(data)}", content=data).status_code == 200
    assert client.get("/status").json()["chunk_count"] == 1


def test_get_rejects_corrupted_chunk(client, settings):
    data = b"precious"
    client.put(f"/chunks/{h(data)}", content=data)
    path = client.app.state.store.path(h(data))
    path.write_bytes(b"bit rot")           # corrupt on disk
    r = client.get(f"/chunks/{h(data)}")
    assert r.status_code == 409


def test_put_heals_corrupted_chunk(client):
    data = b"precious"
    client.put(f"/chunks/{h(data)}", content=data)
    client.app.state.store.path(h(data)).write_bytes(b"bit rot")
    assert client.put(f"/chunks/{h(data)}", content=data).status_code == 200
    assert client.get(f"/chunks/{h(data)}").content == data
    assert client.get("/status").json()["chunk_count"] == 1


@pytest.mark.parametrize("bad", ["abc", "../../etc/passwd", "G" * 64, "A" * 64, "a" * 63])
def test_invalid_hash_ids_rejected(client, bad):
    assert client.get(f"/chunks/{bad}").status_code in (404, 422)
    assert client.put(f"/chunks/{bad}", content=b"x").status_code in (404, 422)


def test_too_large_chunk_rejected(settings):
    settings.max_chunk_bytes = 10
    c = TestClient(create_app(settings, heartbeat=False))
    data = b"x" * 11
    assert c.put(f"/chunks/{h(data)}", content=data).status_code == 413


def test_chunk_count_survives_restart(settings):
    c1 = TestClient(create_app(settings, heartbeat=False))
    for d in (b"1", b"2", b"3"):
        c1.put(f"/chunks/{h(d)}", content=d)
    c2 = TestClient(create_app(settings, heartbeat=False))
    assert c2.get("/status").json()["chunk_count"] == 3


def test_heartbeat_payload_reaches_metadata_service(settings):
    received = []
    meta = FastAPI()

    @meta.post("/heartbeat")
    async def hb(request: Request):
        received.append(await request.json())

    node = create_app(settings, heartbeat=False)
    TestClient(node).put(f"/chunks/{h(b'z')}", content=b"z")

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=meta)) as c:
            await send_heartbeat(c, settings, node.state.store)

    asyncio.run(go())
    (hb_msg,) = received
    assert hb_msg["node_id"] == "node1"
    assert hb_msg["chunk_count"] == 1
    assert hb_msg["free_bytes"] > 0
    assert hb_msg["url"] == "http://node1:8000"


def test_heartbeat_loop_survives_unreachable_metadata(settings):
    settings.metadata_url = "http://127.0.0.1:1"   # nothing listens here
    with TestClient(create_app(settings, heartbeat=True)) as c:
        import time; time.sleep(0.3)               # several failed heartbeats
        assert c.get("/status").status_code == 200  # node keeps serving
