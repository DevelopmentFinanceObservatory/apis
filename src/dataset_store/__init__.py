"""Postgres persistence for the Datasets API: connections, pooling, migrations,
and DB-API protocols.

Connection setup owns environment and credential access. `LiveDatasetStore`
receives its dependencies through the constructor.
"""

from __future__ import annotations

from dataset_store.connection import ConnectionSettings, build_connector, connect, default_is_broken
from dataset_store.dbapi import DBAPIConnection, DBAPICursor
from dataset_store.errors import (
    DatasetStoreConfigError,
    DatasetStoreConnectionError,
    DatasetStoreError,
    PoolTimeoutError,
)
from dataset_store.pool import ConnectionPool
from dataset_store.provider import close_shared_pool, shared_pool
from dataset_store.store import ConnectionSource, DatasetStore, LiveDatasetStore, Transaction

__all__ = [
    "ConnectionPool",
    "ConnectionSettings",
    "ConnectionSource",
    "DBAPIConnection",
    "DBAPICursor",
    "DatasetStore",
    "DatasetStoreConfigError",
    "DatasetStoreConnectionError",
    "DatasetStoreError",
    "LiveDatasetStore",
    "PoolTimeoutError",
    "Transaction",
    "build_connector",
    "close_shared_pool",
    "connect",
    "default_is_broken",
    "shared_pool",
]
