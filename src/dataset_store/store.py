"""Transactional SQL adapter for Postgres, with `DatasetStore` and `Transaction` protocols.

Queries use pg8000's `%s` placeholders. Empty `params` sends SQL unchanged.
Parameterized queries scan `%` outside single-quoted literals. `%%` escapes a
percent. A bare `%` raises `InterfaceError`.

Schema-qualify every dataset table as `datasets.<table>`.

`LiveDatasetStore.transaction` adds one to the call's `db_transactions` count
(`dfo_platform.timings.count`), so each request's timings line says how many
transactions, and so how many pool borrows, that request made.
"""

from __future__ import annotations

import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

from dataset_store.bounds import BORROW_TIMEOUT_SECONDS
from dataset_store.dbapi import DBAPIConnection
from dataset_store.errors import DatasetStoreConnectionError, DatasetStoreError
from dfo_platform.timings import count


class Transaction(Protocol):
    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        """Run DDL/DML. Use `fetch_one()` or `fetch_all()` for statements that return rows."""
        ...

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        """Return `None` for zero rows. Raise `ValueError` for multiple rows."""
        ...

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        """Return column-name mappings in selected order, or an empty list."""
        ...


class DatasetStore(Protocol):
    def transaction(self) -> AbstractContextManager[Transaction]:
        """Commit on clean exit. Roll back and re-raise on failure.

        One transaction per store per thread. Nested calls raise
        `DatasetStoreError`. Other instances or threads may run concurrently.
        Transactions are synchronous. Async callers use a worker thread to
        avoid event-loop interleaving.
        """
        ...


class ConnectionSource(Protocol):
    """Connection-pool interface required by `LiveDatasetStore`."""

    def borrow(self, *, timeout: float | None = ...) -> AbstractContextManager[DBAPIConnection]: ...


class _LiveTransaction:
    def __init__(self, conn: DBAPIConnection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, params)
        finally:
            cursor.close()

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        rows = self.fetch_all(sql, params)
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"fetch_one got {len(rows)} rows for {sql!r}")
        return rows[0]

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description or []]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        finally:
            cursor.close()


class LiveDatasetStore:
    """`connections` implements `ConnectionSource`."""

    def __init__(
        self, *, connections: ConnectionSource, borrow_timeout: float = BORROW_TIMEOUT_SECONDS
    ) -> None:
        self._connections = connections
        self._borrow_timeout = borrow_timeout
        # The guard is per store and thread. Worker threads start with independent state.
        self._local = threading.local()

    @contextmanager
    def transaction(self) -> Generator[Transaction, None, None]:
        """Synchronous operation. Async callers run it in a worker thread."""
        if getattr(self._local, "in_transaction", False):
            # Fail before a nested borrow self-deadlocks into an opaque pool timeout.
            raise DatasetStoreError(
                "transaction() called on this store while a transaction is "
                "already open on this thread; run one transaction at a time "
                "per store and start the next unit of work after the first "
                "exits. Transactions are synchronous and must not span an "
                "await -- async callers run them in worker threads"
            )
        self._local.in_transaction = True
        count("db_transactions")
        try:
            with self._connections.borrow(timeout=self._borrow_timeout) as conn:
                try:
                    yield _LiveTransaction(conn)
                except BaseException as exc:
                    # Mark failed rollbacks as broken so the pool discards the connection.
                    try:
                        conn.rollback()
                    except Exception as rollback_exc:
                        raise DatasetStoreConnectionError(
                            f"rollback failed ({rollback_exc!r}) after {exc!r}; connection unusable"
                        ) from rollback_exc
                    raise
                else:
                    conn.commit()
        finally:
            self._local.in_transaction = False
