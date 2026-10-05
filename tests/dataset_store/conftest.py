"""Provide recording DB-API doubles and a SQLite-backed store for dataset_store tests.
Domain tests should mock their own store protocol."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager


class RecordingCursor:
    def __init__(self, conn: RecordingConnection) -> None:
        self._conn = conn
        self.closed = False

    @property
    def description(self):
        return self._conn.description

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._conn.executed.append((sql, tuple(params)))
        if self._conn.on_execute is not None:
            self._conn.on_execute(sql, params)

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._conn.fetchall_result)

    def fetchone(self) -> tuple[object, ...] | None:
        rows = self.fetchall()
        return rows[0] if rows else None

    def close(self) -> None:
        self.closed = True


class RecordingConnection:
    """Return `fetchall_result` for every read; `on_execute` can change it after writes."""

    def __init__(
        self,
        *,
        fetchall_result: Sequence[tuple[object, ...]] = (),
        description: Sequence[str] = (),
        on_execute: Callable[[str, Sequence[object]], None] | None = None,
    ) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.cursors: list[RecordingCursor] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.fetchall_result: list[tuple[object, ...]] = list(fetchall_result)
        self.description = [(name,) for name in description]
        self.on_execute = on_execute

    def cursor(self) -> RecordingCursor:
        c = RecordingCursor(self)
        self.cursors.append(c)
        return c

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


# One SQLite connection backs every transaction, so commits appear in later reads.
# SQLite keeps DDL after rollback, unlike Postgres.
# Outside quotes, `%s` becomes `?`, `%%` becomes `%`, and other percent sequences raise.
# pg8000 skips this conversion when params are empty; the SQLite translator still applies it.

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

    return _QUOTED_LITERAL_OR_PERCENT.sub(_sub, sql)


class FakeTransaction:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._conn.execute(_to_sqlite_paramstyle(sql), params)

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        rows = self.fetch_all(sql, params)
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"fetch_one got {len(rows)} rows for {sql!r}")
        return rows[0]

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        cursor = self._conn.execute(_to_sqlite_paramstyle(sql), params)
        return [dict(row) for row in cursor.fetchall()]


class FakeDatasetStore:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row

    @contextmanager
    def transaction(self):
        try:
            yield FakeTransaction(self._conn)
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
