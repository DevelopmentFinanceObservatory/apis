"""Run `LiveOperationStore` SQL against a shared in-memory SQLite database.

One store lock serialises transactions. Foreign keys are not enforced. The lease-end expression
`claimed_at + make_interval(secs => lease_seconds)` runs as a registered `lease_end`
function, since SQLite has no interval arithmetic. `before_backlog_lock` runs
at the advisory-lock boundary, after the unlocked read, for deterministic race setup.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta

_OPERATION_SCHEMA = """
CREATE TABLE datasets.operation (
    id TEXT PRIMARY KEY,
    request_identity TEXT NOT NULL,
    options TEXT NOT NULL,
    coverage_stamp TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    estimate_rows INTEGER NOT NULL,
    estimate_requests INTEGER NOT NULL,
    rows_written INTEGER NOT NULL DEFAULT 0,
    subject_id TEXT,
    content_version TEXT,
    refusal_code TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    claimed_at TEXT,
    lease_seconds INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    submission_count INTEGER NOT NULL DEFAULT 0,
    retryable INTEGER,
    dispatched_at TEXT,
    submitted_at TEXT NOT NULL,
    origin_catalogue_build_id TEXT,
    refusal TEXT,
    final_delivery INTEGER,
    confirmed_at TEXT
);
"""

_LEASE_END_SQL = "claimed_at + make_interval(secs => lease_seconds)"

_QUOTED_LITERAL_OR_PERCENT = re.compile(r"('(?:[^']|'')*')|%(.)")


def _to_sqlite_paramstyle(sql: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        quoted = match.group(1)
        if quoted is not None:
            return quoted
        following = match.group(2)
        if following == "s":
            return "?"
        if following == "%":
            return "%"
        raise ValueError("Only %s and %% are supported in the query.")

    sql = sql.replace(_LEASE_END_SQL, "lease_end(claimed_at, lease_seconds)")
    # The store lock serialises transactions, so SQLite needs no row lock.
    sql = sql.replace(" FOR SHARE", "")
    return _QUOTED_LITERAL_OR_PERCENT.sub(_sub, sql)


def _lease_end(claimed_at: str | None, lease_seconds: int | None) -> str | None:
    if claimed_at is None or lease_seconds is None:
        return None
    end = datetime.fromisoformat(claimed_at) + timedelta(seconds=lease_seconds)
    return end.isoformat()


def _sqlite_param(value: object) -> object:
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


_JSON_COLUMNS = frozenset({"request_identity", "options", "refusal"})
_TIMESTAMP_COLUMNS = frozenset(
    {"claimed_at", "created_at", "updated_at", "dispatched_at", "submitted_at", "confirmed_at"}
)
_BOOL_COLUMNS = frozenset({"retryable", "final_delivery"})


def _decode_row(row: sqlite3.Row) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key in row.keys():  # noqa: SIM118 -- sqlite3.Row.__iter__ yields values, not keys
        value = row[key]
        if isinstance(value, str) and key in _JSON_COLUMNS:
            value = json.loads(value)
        elif isinstance(value, str) and key in _TIMESTAMP_COLUMNS:
            value = datetime.fromisoformat(value)
        elif isinstance(value, int) and key in _BOOL_COLUMNS:
            # SQLite stores booleans as 1/0. Keep NULL where the column is unset.
            value = bool(value)
        decoded[key] = value
    return decoded


class _OperationFakeTransaction:
    def __init__(self, conn: sqlite3.Connection, store: FakeOperationDatasetStore) -> None:
        self._conn = conn
        self._store = store

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        if sql.strip().upper().startswith("SELECT PG_ADVISORY_XACT_LOCK"):
            # The store lock serialises transactions, so SQLite needs no advisory-lock function.
            # Invoke the hook at this boundary so tests can inject a competing write.
            if self._store.before_backlog_lock is not None:
                self._store.before_backlog_lock(self._conn)
            return
        self._conn.execute(_to_sqlite_paramstyle(sql), tuple(_sqlite_param(p) for p in params))

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        rows = self.fetch_all(sql, params)
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"fetch_one got {len(rows)} rows for {sql!r}")
        return rows[0]

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        cursor = self._conn.execute(
            _to_sqlite_paramstyle(sql), tuple(_sqlite_param(p) for p in params)
        )
        return [_decode_row(row) for row in cursor.fetchall()]


class FakeOperationDatasetStore:
    """Share one SQLite connection across transactions so committed writes remain visible."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.create_function("lease_end", 2, _lease_end, deterministic=True)
        self._conn.execute("ATTACH DATABASE ':memory:' AS datasets")
        self._conn.executescript(_OPERATION_SCHEMA)
        self._lock = threading.Lock()
        self.before_backlog_lock: Callable[[sqlite3.Connection], None] | None = None

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Generator[_OperationFakeTransaction, None, None]:
        with self._lock:
            tx = _OperationFakeTransaction(self._conn, self)
            try:
                yield tx
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()
