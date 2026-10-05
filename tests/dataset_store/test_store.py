"""Test `LiveDatasetStore` with a recording connection and `FakeDatasetStore` with SQLite."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

import pytest

from dataset_store.errors import DatasetStoreConnectionError, DatasetStoreError
from dataset_store.pool import ConnectionPool
from dataset_store.store import LiveDatasetStore

from .conftest import (
    FakeDatasetStore,
    RecordingConnection,
    _to_sqlite_paramstyle,
)


class _RecordingConnectionSource:
    """Record each borrow timeout and yield `conn`."""

    def __init__(self, conn: RecordingConnection) -> None:
        self._conn = conn
        self.borrow_timeouts: list[float | None] = []

    @contextmanager
    def borrow(self, *, timeout: float | None = 30.0) -> Generator[RecordingConnection, None, None]:
        self.borrow_timeouts.append(timeout)
        yield self._conn


# LiveDatasetStore with a recording connection.


def test_commit_once_on_clean_exit():
    conn = RecordingConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (1,))

    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_rollback_once_and_propagate_on_exception():
    conn = RecordingConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with pytest.raises(ValueError, match="boom"), store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (1,))
        raise ValueError("boom")

    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_cursors_are_closed_on_clean_exit():
    conn = RecordingConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (1,))

    assert conn.cursors
    assert all(c.closed for c in conn.cursors)


def test_cursors_are_closed_on_rollback():
    conn = RecordingConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with pytest.raises(ValueError), store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (1,))
        raise ValueError("boom")

    assert conn.cursors
    assert all(c.closed for c in conn.cursors)


def test_fetch_all_maps_description_names_onto_row_tuples():
    conn = RecordingConnection(
        fetchall_result=[(1, "a"), (2, "b")],
        description=["id", "name"],
    )
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with store.transaction() as tx:
        rows = tx.fetch_all("SELECT id, name FROM t")

    assert rows == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]


def test_fetch_one_raises_on_more_than_one_row():
    conn = RecordingConnection(fetchall_result=[(1,), (2,)], description=["id"])
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with store.transaction() as tx, pytest.raises(ValueError, match="fetch_one got 2 rows"):
        tx.fetch_one("SELECT id FROM t")


def test_fetch_one_returns_none_on_zero_rows():
    conn = RecordingConnection(fetchall_result=[], description=["id"])
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with store.transaction() as tx:
        assert tx.fetch_one("SELECT id FROM t") is None


def test_borrow_is_called_with_the_configured_timeout():
    conn = RecordingConnection()
    source = _RecordingConnectionSource(conn)
    store = LiveDatasetStore(connections=source, borrow_timeout=12.5)

    with store.transaction() as tx:
        tx.execute("SELECT 1")

    assert source.borrow_timeouts == [12.5]


def test_failed_rollback_raises_connection_error_chaining_the_original():
    class _RollbackFailsConnection(RecordingConnection):
        def rollback(self) -> None:
            super().rollback()
            raise RuntimeError("dead socket during rollback")

    conn = _RollbackFailsConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with pytest.raises(DatasetStoreConnectionError) as excinfo, store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (1,))
        raise ValueError("boom")

    assert conn.rollbacks == 1
    rollback_exc = excinfo.value.__cause__
    assert isinstance(rollback_exc, RuntimeError)
    assert str(rollback_exc) == "dead socket during rollback"
    original_exc = rollback_exc.__context__
    assert isinstance(original_exc, ValueError)
    assert str(original_exc) == "boom"


def test_failed_rollback_gets_the_connection_discarded_by_a_real_pool():
    """Confirm a real pool discards a connection whose rollback fails."""

    class _RollbackFailsConnection(RecordingConnection):
        def rollback(self) -> None:
            super().rollback()
            raise RuntimeError("dead socket during rollback")

    poisoned = _RollbackFailsConnection()
    fresh = RecordingConnection()
    remaining = [poisoned, fresh]

    def factory():
        return remaining.pop(0)

    pool = ConnectionPool(connect=factory, size=1)
    store = LiveDatasetStore(connections=pool)

    with pytest.raises(DatasetStoreConnectionError), store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (1,))
        raise ValueError("boom")

    assert poisoned.closed

    with pool.borrow() as conn:
        assert conn is fresh


def test_nested_transaction_raises_immediately_without_a_second_borrow():
    conn = RecordingConnection()
    source = _RecordingConnectionSource(conn)
    store = LiveDatasetStore(connections=source, borrow_timeout=5.0)

    start = time.monotonic()
    with store.transaction() as tx:
        tx.execute("SELECT 1")
        with (
            pytest.raises(DatasetStoreError, match="one transaction at a time"),
            store.transaction(),
        ):
            pass
    elapsed = time.monotonic() - start

    # Nested transactions fail before borrowing instead of waiting for the 5 s timeout.
    assert source.borrow_timeouts == [5.0]
    assert elapsed < 1.0


def test_nested_transaction_error_points_the_caller_at_the_no_await_contract():
    conn = RecordingConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with (
        store.transaction(),
        pytest.raises(DatasetStoreError, match="synchronous and must not span an await"),
        store.transaction(),
    ):
        pass


def test_sequential_transactions_on_the_same_store_each_succeed():
    conn = RecordingConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (1,))
    with store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (2,))

    assert conn.commits == 2
    assert conn.rollbacks == 0


def test_transaction_works_again_after_a_previous_one_raised():
    conn = RecordingConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))

    with pytest.raises(ValueError, match="boom"), store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (1,))
        raise ValueError("boom")

    with store.transaction() as tx:
        tx.execute("INSERT INTO t VALUES (%s)", (2,))

    assert conn.commits == 1
    assert conn.rollbacks == 1


def test_two_threads_can_each_hold_a_transaction_concurrently():
    ready = threading.Barrier(2)
    results: dict[str, RecordingConnection] = {}
    errors: list[BaseException] = []

    def worker(name: str) -> None:
        try:
            conn = RecordingConnection()
            store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))
            with store.transaction() as tx:
                # A transaction guard leaking across threads would deadlock here.
                ready.wait(timeout=5)
                tx.execute("SELECT 1")
            results[name] = conn
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    assert results["a"].commits == 1
    assert results["b"].commits == 1


def test_a_transaction_on_a_second_independent_store_works_inside_the_first():
    conn_a = RecordingConnection()
    conn_b = RecordingConnection()
    store_a = LiveDatasetStore(connections=_RecordingConnectionSource(conn_a))
    store_b = LiveDatasetStore(connections=_RecordingConnectionSource(conn_b))

    with store_a.transaction() as tx_a:
        tx_a.execute("SELECT 1")
        with store_b.transaction() as tx_b:
            tx_b.execute("SELECT 1")

    assert conn_a.commits == 1
    assert conn_b.commits == 1


def test_a_thread_spawned_inside_an_open_transaction_runs_its_own_transaction_on_the_same_store():
    conn = RecordingConnection()
    store = LiveDatasetStore(connections=_RecordingConnectionSource(conn))
    child_error: list[BaseException] = []

    def child() -> None:
        try:
            with store.transaction() as tx:
                tx.execute("SELECT 1")
        except BaseException as exc:
            child_error.append(exc)

    with store.transaction() as tx:
        tx.execute("SELECT 1")
        # A new thread has a fresh `threading.local`, so it can transact on this store.
        t = threading.Thread(target=child)
        t.start()
        t.join(timeout=5)

    assert not child_error
    assert conn.commits == 2


# SQLite-backed FakeDatasetStore.


def test_fake_read_your_own_write_across_transactions():
    store = FakeDatasetStore()

    with store.transaction() as tx:
        tx.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        tx.execute("INSERT INTO t (id, v) VALUES (%s, %s)", (1, "first"))

    with store.transaction() as tx:
        row = tx.fetch_one("SELECT v FROM t WHERE id = %s", (1,))

    assert row == {"v": "first"}
    store.close()


def test_fake_rollback_on_exception_leaves_no_row():
    store = FakeDatasetStore()

    with store.transaction() as tx:
        tx.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")

    with pytest.raises(ValueError), store.transaction() as tx:
        tx.execute("INSERT INTO t (id, v) VALUES (%s, %s)", (1, "first"))
        raise ValueError("boom")

    with store.transaction() as tx:
        assert tx.fetch_all("SELECT * FROM t") == []
    store.close()


def test_fake_upsert_idiom_leaves_exactly_one_row_with_the_latest_value():
    store = FakeDatasetStore()

    with store.transaction() as tx:
        tx.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")

    upsert = "INSERT INTO t (id, v) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET v = excluded.v"
    with store.transaction() as tx:
        tx.execute(upsert, (1, "first"))
    with store.transaction() as tx:
        tx.execute(upsert, (1, "second"))

    with store.transaction() as tx:
        rows = tx.fetch_all("SELECT id, v FROM t")

    assert rows == [{"id": 1, "v": "second"}]
    store.close()


def test_fake_fetch_one_and_fetch_all_zero_and_multi_row_behaviour():
    store = FakeDatasetStore()

    with store.transaction() as tx:
        tx.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        assert tx.fetch_one("SELECT v FROM t WHERE id = %s", (1,)) is None
        assert tx.fetch_all("SELECT v FROM t") == []
        tx.execute("INSERT INTO t (id, v) VALUES (1, 'a')")
        tx.execute("INSERT INTO t (id, v) VALUES (2, 'b')")
        with pytest.raises(ValueError, match="fetch_one got 2 rows"):
            tx.fetch_one("SELECT v FROM t")

    store.close()


# Paramstyle translation.


def test_paramstyle_translator_s_becomes_positional_placeholder():
    assert _to_sqlite_paramstyle("SELECT %s FROM t") == "SELECT ? FROM t"


def test_paramstyle_translator_double_percent_unescapes_to_literal_percent():
    assert _to_sqlite_paramstyle("SELECT 1 WHERE 2 %% 2 = %s") == "SELECT 1 WHERE 2 % 2 = ?"


def test_paramstyle_translator_quoted_percent_passes_through_verbatim():
    assert (
        _to_sqlite_paramstyle("SELECT * FROM t WHERE a LIKE 'a%' AND id = %s")
        == "SELECT * FROM t WHERE a LIKE 'a%' AND id = ?"
    )


def test_paramstyle_translator_quoted_literal_with_escaped_quote_is_untouched():
    assert _to_sqlite_paramstyle("SELECT 'it''s 100%s' FROM t") == "SELECT 'it''s 100%s' FROM t"


def test_paramstyle_translator_unquoted_bare_percent_raises_with_pg8000s_message():
    with pytest.raises(ValueError, match=r"Only %s and %% are supported in the query\."):
        _to_sqlite_paramstyle("SELECT 100 % 3 FROM t")
