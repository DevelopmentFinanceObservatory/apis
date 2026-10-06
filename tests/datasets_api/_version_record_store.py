"""SQLite-backed store double for version, snapshot, and reference tests."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime

# The schema covers version records, artifacts, revocations, references, inventory, and subject heads.
# SQLite runs the real reader/writer SQL here, but does not enforce cross-table foreign keys.
_VERSION_RECORD_SCHEMA = """
CREATE TABLE datasets.version_record (
    subject_id TEXT NOT NULL,
    subject_kind TEXT NOT NULL CHECK (subject_kind IN ('spec', 'collection')),
    content_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    immutable_body TEXT NOT NULL,
    trace_id TEXT,
    recorded_at TEXT NOT NULL,
    base_content_version TEXT,
    PRIMARY KEY (subject_id, content_version)
);
CREATE TABLE datasets.version_artifact (
    subject_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    artifact_digest TEXT NOT NULL,
    rendering_key TEXT NOT NULL,
    rendering TEXT NOT NULL,
    media_type TEXT NOT NULL,
    PRIMARY KEY (subject_id, content_version, rendering_key)
);
CREATE TABLE datasets.version_revocation (
    subject_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    reason TEXT,
    trace_id TEXT,
    revoked_at TEXT NOT NULL,
    PRIMARY KEY (subject_id, content_version)
);
CREATE TABLE datasets.artifact_reference (
    principal TEXT NOT NULL,
    artifact_digest TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    reference_kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    removed_at TEXT,
    PRIMARY KEY (principal, artifact_digest, subject_id, content_version, reference_kind)
);
CREATE TABLE datasets.artifact_inventory (
    artifact_digest TEXT PRIMARY KEY,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE datasets.artifact_location (
    artifact_digest TEXT NOT NULL,
    tier TEXT NOT NULL,
    bucket TEXT NOT NULL,
    object_name TEXT NOT NULL,
    stored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (artifact_digest, tier)
);
CREATE TABLE datasets.subject_head (
    subject_id TEXT PRIMARY KEY,
    content_version TEXT NOT NULL,
    confirmed_at TEXT NOT NULL,
    materialised_at TEXT NOT NULL,
    signal TEXT NOT NULL
);
"""

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


def _sqlite_param(value: object) -> object:
    """Encode dicts and datetimes as SQLite text for JSONB and timestamp columns."""
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# Decode SQLite text to the dict and datetime values that pg8000 returns for JSONB and timestamps.
_JSON_COLUMNS = frozenset({"immutable_body", "rendering", "signal"})
_TIMESTAMP_COLUMNS = frozenset(
    {
        "recorded_at",
        "created_at",
        "revoked_at",
        "stored_at",
        "removed_at",
        "confirmed_at",
        "materialised_at",
    }
)


def _decode_row(row: sqlite3.Row) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key in row.keys():  # noqa: SIM118 -- sqlite3.Row.__iter__ yields values, not keys
        value = row[key]
        if isinstance(value, str) and key in _JSON_COLUMNS:
            value = json.loads(value)
        elif isinstance(value, str) and key in _TIMESTAMP_COLUMNS:
            value = datetime.fromisoformat(value)
        decoded[key] = value
    return decoded


class _VersionRecordFakeTransaction:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        if sql.strip().upper().startswith("SELECT PG_ADVISORY_XACT_LOCK"):
            # The store lock serialises transactions, so SQLite needs no advisory-lock function.
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


class FakeVersionRecordDatasetStore:
    """Share one SQLite connection across transactions so committed writes remain visible."""

    def __init__(self) -> None:
        # The mint worker uses this connection from a different thread.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("ATTACH DATABASE ':memory:' AS datasets")
        self._conn.executescript(_VERSION_RECORD_SCHEMA)
        # Mint work and the request thread share this connection, so serialize transaction spans.
        # Postgres gives each call its own pooled connection.
        self._lock = threading.Lock()

    @contextmanager
    def transaction(self) -> Generator[_VersionRecordFakeTransaction, None, None]:
        with self._lock:
            try:
                yield _VersionRecordFakeTransaction(self._conn)
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def close(self) -> None:
        self._conn.close()
