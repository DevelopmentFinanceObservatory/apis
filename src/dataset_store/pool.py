"""Lazy, bounded DB-API pool with connection checks for Cloud Run freeze/thaw.

The dataset pool is shared across surfaces. The separate presentation pool
is lazy. Their defaults are 10 and 2, so an apis instance holds at most 12
connections. A Datasets request holds one connection at a time, so the pool
bounds the store transactions an instance runs at once.

Every service sharing the Postgres instance sizes against one budget:

    sum(pool x max instances) + reserve <= max_connections

With the `infra` repository's caps (`infra/stacks/apis`, `infra/stacks/website`,
`infra/stacks/ai`) and the pool sizes its `infra/stacks/platform` comment
records, that is 12 x 5 (apis) + 10 x 4 (datasets worker) + 5 x 5 (website)
+ 10 x 2 (ai) + 10 (migrations and built-in users) = 155, against
`max_connections` 200 in dev and 300 in prod. Raising an instance cap or a
pool size raises the left side, and `max_connections` moves with it.

Borrowing waits 30 seconds by default, then raises `PoolTimeoutError`, which doors answer as a
retryable 503. Connections idle over 300 seconds
are reopened. Those idle over 60 seconds receive a `SELECT 1` check.

A borrow records its slot wait as the `db_pool_wait` phase, a new connection's
dial (including the Cloud SQL Connector dial) as `db_connect`, and the idle
check as `db_ping`. These nest inside the caller's own phase, such as
`head_read` or `operation`, so they break that phase down rather than add to it.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress

from dataset_store.bounds import CONNECT_TIMEOUT_SECONDS
from dataset_store.connection import default_is_broken
from dataset_store.dbapi import DBAPIConnection
from dataset_store.errors import PoolTimeoutError
from dfo_platform.timings import phase


class ConnectionPool:
    """Connect lazily, with at most `size` idle or borrowed connections.

    `is_broken` decides whether an exception discards a borrowed connection.
    It defaults to `connection.default_is_broken`. `connect_timeout` is the bound `connect`
    dials and reads under (`ConnectionSettings.connect_timeout`).
    """

    def __init__(
        self,
        *,
        connect: Callable[[], DBAPIConnection],
        size: int,
        is_broken: Callable[[BaseException], bool] = default_is_broken,
        max_idle_seconds: float = 300.0,
        validate_after_idle_seconds: float = 60.0,
        connect_timeout: float = CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1; got {size!r}")
        self._connect = connect
        self.connect_timeout = connect_timeout
        self._is_broken = is_broken
        self._size = size
        self._max_idle_seconds = max_idle_seconds
        self._validate_after_idle_seconds = validate_after_idle_seconds
        self._idle: queue.LifoQueue[tuple[DBAPIConnection, float]] = queue.LifoQueue(maxsize=size)
        self._semaphore = threading.Semaphore(size)

    @contextmanager
    def borrow(self, *, timeout: float | None = 30.0) -> Generator[DBAPIConnection, None, None]:
        """`timeout` bounds slot wait. Dialling and query I/O use the connection timeout."""
        with phase("db_pool_wait"):
            acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise PoolTimeoutError(
                f"no connection freed up within {timeout}s (pool size {self._size})"
            )
        try:
            # Release the slot if connection creation fails.
            conn = self._take_or_create()
        except BaseException:
            self._semaphore.release()
            raise

        broken = False
        try:
            yield conn
        except BaseException as exc:
            try:
                broken = self._is_broken(exc)
            except Exception:
                # Discard the connection if its classifier fails.
                broken = True
            raise
        finally:
            if broken:
                self._close_quietly(conn)
            else:
                self._idle.put_nowait((conn, time.monotonic()))
            self._semaphore.release()

    def close_all(self) -> None:
        """Close idle connections. Borrowed connections close on return. Safe to repeat."""
        while True:
            try:
                conn, _ = self._idle.get_nowait()
            except queue.Empty:
                break
            self._close_quietly(conn)

    def _take_or_create(self) -> DBAPIConnection:
        while True:
            try:
                conn, returned_at = self._idle.get_nowait()
            except queue.Empty:
                with phase("db_connect"):
                    return self._connect()
            idle_for = time.monotonic() - returned_at
            if idle_for > self._max_idle_seconds:
                self._close_quietly(conn)
                continue
            if idle_for > self._validate_after_idle_seconds and not self._ping(conn):
                self._close_quietly(conn)
                continue
            return conn

    @staticmethod
    def _ping(conn: DBAPIConnection) -> bool:
        with phase("db_ping"):
            try:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT 1")
                    cursor.fetchall()
                finally:
                    cursor.close()
                return True
            except Exception:
                return False

    @staticmethod
    def _close_quietly(conn: DBAPIConnection) -> None:
        with suppress(Exception):
            conn.close()
