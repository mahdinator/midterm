"""HTTP client for the storage nodes' chunk API (see dfs/node.py)."""

from __future__ import annotations

import httpx


class NodeError(Exception):
    """A node could not be reached or answered with an unexpected error."""


class NodeClient:
    def __init__(self, timeout: float = 10.0):
        self._http = httpx.Client(timeout=timeout)
        self._urls: dict[str, str] = {}

    def set_urls(self, urls: dict[str, str]) -> None:
        self._urls = dict(urls)

    def _url(self, node_id: str, h: str) -> str:
        try:
            return f"{self._urls[node_id].rstrip('/')}/chunks/{h}"
        except KeyError:
            raise NodeError(f"unknown node {node_id}") from None

    def put(self, node_id: str, h: str, data: bytes) -> None:
        try:
            self._http.put(self._url(node_id, h), content=data).raise_for_status()
        except httpx.HTTPError as e:
            raise NodeError(f"PUT {h[:8]} on {node_id}: {e!r}") from e

    def get(self, node_id: str, h: str) -> bytes | None:
        """The chunk's bytes, or None if the node does not have it or reports it corrupted."""
        try:
            r = self._http.get(self._url(node_id, h))
        except httpx.HTTPError as e:
            raise NodeError(f"GET {h[:8]} on {node_id}: {e!r}") from e
        if r.status_code in (404, 409):               # missing / corrupted on disk
            return None
        try:
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise NodeError(f"GET {h[:8]} on {node_id}: {e!r}") from e
        return r.content

    def has(self, node_id: str, h: str) -> bool:
        try:
            r = self._http.head(self._url(node_id, h))
        except httpx.HTTPError as e:
            raise NodeError(f"HEAD {h[:8]} on {node_id}: {e!r}") from e
        if r.status_code == 404:
            return False
        try:
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise NodeError(f"HEAD {h[:8]} on {node_id}: {e!r}") from e
        return True

    def delete(self, node_id: str, h: str) -> bool:
        try:
            r = self._http.delete(self._url(node_id, h))
        except httpx.HTTPError as e:
            raise NodeError(f"DELETE {h[:8]} on {node_id}: {e!r}") from e
        if r.status_code == 404:
            return False
        try:
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise NodeError(f"DELETE {h[:8]} on {node_id}: {e!r}") from e
        return True
