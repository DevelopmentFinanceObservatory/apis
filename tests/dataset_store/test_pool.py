"""Test `ConnectionPool` reuse, timeouts, connection failures, and idle validation."""

from __future__ import annotations

import threading
import time

import pytest

from dataset_store.errors import DatasetStoreConnectionError, PoolTimeoutError
from dataset_store.pool import ConnectionPool
from dfo_platform.timings import PhaseTimings, collecting


class FakeConn:
    """Track how often idle validation issues `SELECT 1`."""

    _next_id = 0

    def __init__(self, *, fail_ping: bool = False) -> None:
        FakeConn._next_id += 1
        self.id = FakeConn._next_id
        self.closed = False
        self.ping_calls = 0
        self._fail_ping = fail_ping

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    def execute(self, sql: str, params=()) -> None:
        if sql == "SELECT 1":
            self._conn.ping_calls += 1
            if self._conn._fail_ping:
                raise DatasetStoreConnectionError("ping failed")

    def fetchall(self):
        return [(1,)]

    def fetchone(self):
        return (1,)

    def close(self) -> None:
        pass


def make_factory(*conns: FakeConn | BaseException):
    """Return a connector that yields `conns` in order and raises exception entries.

    Raise `AssertionError` if the test requests more connections than supplied.
    """
    remaining = list(conns)

    def _connect():
        assert remaining, "factory called more times than the test expected"
        item = remaining.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return _connect


def test_lazy_creation_calls_factory_zero_times_before_first_borrow():
    calls = []

    def factory():
        calls.append(1)
        return FakeConn()

    ConnectionPool(connect=factory, size=2)
    assert calls == []


def test_sequential_borrows_reuse_the_same_connection():
    conn = FakeConn()
    factory = make_factory(conn)
    pool = ConnectionPool(connect=factory, size=1)

    with pool.borrow() as c1:
        assert c1 is conn
    with pool.borrow() as c2:
        assert c2 is conn


def test_cap_enforcement_raises_pool_timeout_error_when_exhausted():
    pool = ConnectionPool(connect=make_factory(FakeConn(), FakeConn()), size=1)
    released = threading.Event()
    entered = threading.Event()

    def hold():
        with pool.borrow():
            entered.set()
            released.wait(timeout=5)

    t = threading.Thread(target=hold)
    t.start()
    entered.wait(timeout=5)

    with pytest.raises(PoolTimeoutError), pool.borrow(timeout=0.2):
        pass

    released.set()
    t.join(timeout=5)

    with pool.borrow():
        pass


def test_connect_failure_releases_the_semaphore_slot_instead_of_leaking_it():
    """Two transient connect() failures on a size=1 pool leave it usable for
    a third, successful borrow.
    """
    good = FakeConn()
    factory = make_factory(
        DatasetStoreConnectionError("boom 1"),
        DatasetStoreConnectionError("boom 2"),
        good,
    )
    pool = ConnectionPool(connect=factory, size=1)

    with pytest.raises(DatasetStoreConnectionError), pool.borrow(timeout=0.5):
        pass
    with pytest.raises(DatasetStoreConnectionError), pool.borrow(timeout=0.5):
        pass
    with pool.borrow(timeout=0.5) as conn:
        assert conn is good


def test_discard_on_broken_gets_a_fresh_connection_next_borrow():
    first = FakeConn()
    second = FakeConn()
    pool = ConnectionPool(
        connect=make_factory(first, second),
        size=1,
        is_broken=lambda exc: isinstance(exc, RuntimeError),
    )

    with pytest.raises(RuntimeError), pool.borrow() as conn:
        assert conn is first
        raise RuntimeError("dead socket")
    assert first.closed

    with pool.borrow() as conn:
        assert conn is second


def test_non_broken_exception_keeps_the_connection_for_reuse():
    only = FakeConn()
    pool = ConnectionPool(
        connect=make_factory(only),
        size=1,
        is_broken=lambda exc: isinstance(exc, RuntimeError),
    )

    with pytest.raises(ValueError), pool.borrow() as conn:
        assert conn is only
        raise ValueError("ordinary application error")
    assert not only.closed

    with pool.borrow() as conn:
        assert conn is only


def test_broken_classifier_that_raises_fails_safe_and_discards():
    first = FakeConn()
    second = FakeConn()

    def flaky_is_broken(exc: BaseException) -> bool:
        raise RuntimeError("classifier itself is broken")

    pool = ConnectionPool(
        connect=make_factory(first, second),
        size=1,
        is_broken=flaky_is_broken,
    )

    with pytest.raises(ValueError, match="original"), pool.borrow() as conn:
        assert conn is first
        raise ValueError("original")
    assert first.closed

    with pool.borrow() as conn:
        assert conn is second


def test_idle_recycle_closes_and_reconnects_past_max_idle_seconds():
    first = FakeConn()
    second = FakeConn()
    pool = ConnectionPool(
        connect=make_factory(first, second),
        size=1,
        max_idle_seconds=0.05,
    )

    with pool.borrow() as conn:
        assert conn is first
    time.sleep(0.1)

    with pool.borrow() as conn:
        assert conn is second
    assert first.closed


def test_validate_on_borrow_pings_and_discards_a_failed_stale_connection():
    stale = FakeConn(fail_ping=True)
    fresh = FakeConn()
    pool = ConnectionPool(
        connect=make_factory(stale, fresh),
        size=1,
        validate_after_idle_seconds=0.02,
        max_idle_seconds=10.0,
    )

    with pool.borrow() as conn:
        assert conn is stale
    time.sleep(0.05)

    with pool.borrow() as conn:
        assert conn is fresh
    assert stale.closed


def test_validate_on_borrow_skips_the_ping_when_idle_is_short():
    only = FakeConn()
    pool = ConnectionPool(
        connect=make_factory(only),
        size=1,
        validate_after_idle_seconds=5.0,
        max_idle_seconds=10.0,
    )

    with pool.borrow() as conn:
        assert conn is only
    with pool.borrow() as conn:
        assert conn is only
    assert only.ping_calls == 0


def test_close_all_closes_every_idle_connection_and_is_idempotent():
    conn = FakeConn()
    pool = ConnectionPool(connect=make_factory(conn), size=1)

    with pool.borrow():
        pass
    assert not conn.closed

    pool.close_all()
    assert conn.closed

    pool.close_all()  # no-op the second time, must not raise


def test_a_borrow_that_dials_records_db_connect_and_a_reuse_does_not():
    pool = ConnectionPool(connect=make_factory(FakeConn()), size=1)

    dialled = PhaseTimings()
    with collecting(dialled), pool.borrow():
        pass
    reused = PhaseTimings()
    with collecting(reused), pool.borrow():
        pass

    assert set(dialled.snapshot()) == {"db_pool_wait", "db_connect"}
    assert set(reused.snapshot()) == {"db_pool_wait"}


def test_a_borrow_that_pings_an_idle_connection_records_db_ping():
    pool = ConnectionPool(
        connect=make_factory(FakeConn()),
        size=1,
        validate_after_idle_seconds=0.02,
        max_idle_seconds=10.0,
    )
    with pool.borrow():
        pass
    time.sleep(0.05)

    timings = PhaseTimings()
    with collecting(timings), pool.borrow():
        pass

    assert set(timings.snapshot()) == {"db_pool_wait", "db_ping"}
