import os
import uuid

import psycopg
import pytest

from dfs.db import MetadataDB


@pytest.fixture
def db():
    """A MetadataDB in a throw-away schema, so tests never touch real data.

    Needs a reachable Postgres:  DATABASE_URL=postgresql://dfs:dfs@localhost:5432/dfs
    Without it the Postgres tests are skipped (run `pytest -rs` to see why)."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set, skipping Postgres tests")
    d = MetadataDB(dsn, schema="test_" + uuid.uuid4().hex[:12])
    try:
        d.init_schema()
    except psycopg.OperationalError as e:
        pytest.skip(f"cannot reach Postgres at DATABASE_URL: {e}")
    yield d
    d.drop_schema()
