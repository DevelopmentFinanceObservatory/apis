"""SQLite-backed store double for version, snapshot, and reference tests.

It also holds `datasets.operation`, as the one Postgres database does, so a `LiveOperationStore`
over it shares the database a fenced `LiveHeadStore.advance_head` reads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime

from datasets_api.version_store import _ROLE_COLUMNS_SQL
from datasets_api.versions import MaterialisationRecord, Trigger

from ._operation_store import _OPERATION_SCHEMA, _lease_end, _to_sqlite_paramstyle

# The schema covers version records, artifacts, revocations, references, caller allowances,
# inventory, frame recipes, analysis answers, description records, materialisation runs, and subject heads. `LiveRecipeStore` and `LiveVersionLister` over this store are the recipe and
# version-list doubles.
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
    frame_grammar_version TEXT,
    prepared_at TEXT,
    definition TEXT,
    coverage TEXT,
    citation TEXT,
    trace TEXT,
    warnings TEXT,
    series_count INTEGER,
    series_coverage_stamp TEXT,
    output_profile TEXT,
    series_conversions TEXT,
    default_scope_plural INTEGER,
    input_content_versions TEXT,
    composed_stat_vars TEXT,
    continuations TEXT,
    frame_sha256 TEXT,
    frame_summed INTEGER,
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
CREATE TABLE datasets.caller_allowance (
    caller_key TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('share', 'card', 'pin', 'pin_budget', 'open_pin_budget')),
    day TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (caller_key, kind, day)
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
    renewal INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (artifact_digest, tier)
);
CREATE TABLE datasets.frame_recipe (
    subject_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    normal_form_hash TEXT NOT NULL,
    replay_key TEXT NOT NULL UNIQUE,
    recipe TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (subject_id, content_version)
);
CREATE TABLE datasets.analysis_answer (
    request_key TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    body TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE datasets.description_record (
    subject_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    basis TEXT NOT NULL CHECK (basis IN ('mint', 'publish', 'backfill')),
    record TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (subject_id, content_version)
);
CREATE TABLE datasets.materialisation (
    id INTEGER PRIMARY KEY,
    subject_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    materialised_at TEXT NOT NULL,
    "trigger" TEXT NOT NULL
        CHECK ("trigger" IN ('mint', 'get', 'options', 'refresh', 'worker')),
    materialiser_build TEXT NOT NULL,
    frame_grammar_version TEXT NOT NULL,
    code_index_stamp TEXT NOT NULL,
    origin_catalogue_build_id TEXT NOT NULL,
    coverage_stamp TEXT,
    graph_generation TEXT,
    dependencies TEXT NOT NULL,
    selection TEXT NOT NULL,
    conversion TEXT,
    trace_id TEXT
);
CREATE TABLE datasets.subject_head (
    subject_id TEXT PRIMARY KEY,
    content_version TEXT NOT NULL,
    confirmed_at TEXT NOT NULL,
    materialised_at TEXT NOT NULL,
    signal TEXT NOT NULL DEFAULT '{"members": []}',
    materialisation_id INTEGER,
    dependencies TEXT
);
"""


def _role_columns(output_profile: str | None) -> str | None:
    """SQLite's stand-in for the `jsonb_path_query_array` of `_ROLE_COLUMNS_SQL`: the names of
    the profile's columns of kind `role`, as a JSON array."""
    if output_profile is None:
        return None
    columns = json.loads(output_profile).get("columns", [])
    return json.dumps([column["name"] for column in columns if column.get("kind") == "role"])


def _to_sqlite(sql: str) -> str:
    sql = sql.replace(_ROLE_COLUMNS_SQL, "role_columns(output_profile) AS role_columns")
    return _to_sqlite_paramstyle(sql)


def _sqlite_param(value: object) -> object:
    """Encode dicts, datetimes and dates as SQLite text for JSONB, timestamp and date columns."""
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


# Decode SQLite text to the dict and datetime values that pg8000 returns for JSONB and timestamps.
_JSON_COLUMNS = frozenset(
    {
        "immutable_body",
        "definition",
        "coverage",
        "citation",
        "trace",
        "warnings",
        "output_profile",
        "role_columns",
        "series_conversions",
        "input_content_versions",
        "composed_stat_vars",
        "continuations",
        "rendering",
        "signal",
        "recipe",
        "body",
        "record",
        "dependencies",
        "selection",
        "conversion",
        "request_identity",
        "options",
        "refusal",
    }
)
_TIMESTAMP_COLUMNS = frozenset(
    {
        "recorded_at",
        "created_at",
        "revoked_at",
        "stored_at",
        "removed_at",
        "confirmed_at",
        "materialised_at",
        "prepared_at",
        "claimed_at",
        "updated_at",
        "dispatched_at",
        "submitted_at",
    }
)
_BOOL_COLUMNS = frozenset({"retryable", "final_delivery", "frame_summed"})


def _decode_row(row: sqlite3.Row) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key in row.keys():  # noqa: SIM118 -- sqlite3.Row.__iter__ yields values, not keys
        value = row[key]
        if isinstance(value, str) and key in _JSON_COLUMNS:
            value = json.loads(value)
        elif isinstance(value, str) and key in _TIMESTAMP_COLUMNS:
            value = datetime.fromisoformat(value)
        elif isinstance(value, int) and key in _BOOL_COLUMNS:
            value = bool(value)
        decoded[key] = value
    return decoded


class _VersionRecordFakeTransaction:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        if sql.strip().upper().startswith("SELECT PG_ADVISORY_XACT_LOCK"):
            # The store lock serialises transactions, so SQLite needs no advisory-lock function.
            return
        self._conn.execute(_to_sqlite(sql), tuple(_sqlite_param(p) for p in params))

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        rows = self.fetch_all(sql, params)
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"fetch_one got {len(rows)} rows for {sql!r}")
        return rows[0]

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        cursor = self._conn.execute(_to_sqlite(sql), tuple(_sqlite_param(p) for p in params))
        return [_decode_row(row) for row in cursor.fetchall()]


class FakeVersionRecordDatasetStore:
    """Share one SQLite connection across transactions so committed writes remain visible."""

    def __init__(self) -> None:
        # The mint worker uses this connection from a different thread.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.create_function("lease_end", 2, _lease_end, deterministic=True)
        self._conn.create_function("role_columns", 1, _role_columns, deterministic=True)
        self._conn.execute("ATTACH DATABASE ':memory:' AS datasets")
        self._conn.executescript(_VERSION_RECORD_SCHEMA)
        self._conn.executescript(_OPERATION_SCHEMA)
        # Mint work and the request thread share this connection, so serialize transaction spans.
        # Postgres gives each call its own pooled connection.
        self._lock = threading.Lock()

    @property
    def total_changes(self) -> int:
        """Rows every statement so far inserted, updated or deleted."""
        return self._conn.total_changes

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


def materialisation_record(
    *, subject_id: str, content_version: str, trigger: Trigger = "get"
) -> MaterialisationRecord:
    """A run record for head-store tests, carrying a fixed schema-1 fingerprint."""
    return MaterialisationRecord(
        subject_id=subject_id,
        content_version=content_version,
        trigger=trigger,
        materialiser_build="test-build",
        frame_grammar_version="1.0.0",
        code_index_stamp="test-stamp",
        origin_catalogue_build_id="test-catalogue",
        coverage_stamp=None,
        graph_generation=None,
        dependencies={"schema": 1, "members": f"sha256:{content_version}"},
        selection={"facets": []},
        conversion=None,
        trace_id=None,
    )
