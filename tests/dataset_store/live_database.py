"""The Postgres database each live dataset_store test runs against.

`DFO_DATASET_STORE_LIVE_DSN` names a database with every migration applied. Each test gets a
copy of it, cloned as a template when the test starts and dropped when it ends. A test sees no
row, sequence or migration bookkeeping row another test left, whatever order pytest-xdist runs
them in and whichever worker runs them, so `-n 4` gives the result a serial run does. The
DSN's role needs CREATEDB and a login on the `postgres` database, and nothing may be connected
to the template while a test clones it.

Roles belong to the cluster, not a database, so a test that creates one takes `cluster_lock`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast
from urllib.parse import urlsplit
from uuid import uuid4

import pg8000.legacy
import pytest

from dataset_store import connect

SERVER_DSN = os.environ.get("DFO_DATASET_STORE_LIVE_DSN")

# Advisory locks belong to one database, so this one is taken in the maintenance database,
# which every worker reaches.
_CLUSTER_LOCK_KEY = 7_104_220_618
_MAINTENANCE_DATABASE = "postgres"


_active: str | None = None


def _template_database() -> str:
    assert SERVER_DSN
    return urlsplit(SERVER_DSN).path.lstrip("/")


def database_name() -> str:
    """The name of the database the running test uses, which `live_database` has made."""
    assert _active is not None, "a live test needs the `live_database` fixture"
    return _active


def connect_to(database: str) -> Callable[[], pg8000.legacy.Connection]:
    """A connect callable for `database` on the server `SERVER_DSN` names."""
    assert SERVER_DSN
    parts = urlsplit(SERVER_DSN)
    return lambda: cast(
        pg8000.legacy.Connection,
        connect(
            connector=None,
            instance_connection_name=None,
            host=cast(str, parts.hostname),
            port=parts.port or 5432,
            database=database,
            user=cast(str, parts.username),
            password=parts.password,
        ),
    )


def connect_live() -> pg8000.legacy.Connection:
    """A connection to the database this process's live tests use."""
    return connect_to(database_name())()


def live_params() -> dict[str, str]:
    """The `DATASET_STORE_*` settings that reach `connect_live`'s database as the DSN's role."""
    assert SERVER_DSN
    parts = urlsplit(SERVER_DSN)
    return {
        "DATASET_STORE_HOST": cast(str, parts.hostname),
        "DATASET_STORE_PORT": str(parts.port or 5432),
        "DATASET_STORE_DATABASE": database_name(),
        "DATASET_STORE_MIGRATOR_DB_USER": cast(str, parts.username),
        "DATASET_STORE_MIGRATOR_DB_PASSWORD": cast(str, parts.password),
    }


@contextmanager
def cluster_lock() -> Iterator[pg8000.legacy.Connection]:
    """A connection to the maintenance database, in autocommit, holding the cluster-wide
    lock until the block exits. Take it around anything that changes a role or clones a
    database, which Postgres runs one at a time per object and fails on a race."""
    admin = connect_to(_MAINTENANCE_DATABASE)()
    admin.autocommit = True
    try:
        admin.run("SELECT pg_advisory_lock(:key)", key=_CLUSTER_LOCK_KEY)
        try:
            yield admin
        finally:
            admin.run("SELECT pg_advisory_unlock(:key)", key=_CLUSTER_LOCK_KEY)
    finally:
        admin.close()


@pytest.fixture
def live_database() -> Iterator[str]:
    """A copy of the DSN's database for one test, dropped when the test ends."""
    global _active
    name = f"{_template_database()}_t{uuid4().hex[:12]}"
    with cluster_lock() as admin:
        admin.run(f'CREATE DATABASE "{name}" TEMPLATE "{_template_database()}"')
    _active = name
    try:
        yield name
    finally:
        _active = None
        with cluster_lock() as admin:
            admin.run(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
