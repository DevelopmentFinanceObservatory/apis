"""`python -m dataset_store.warning_formats --dry-run` against a real Postgres. Skipped unless
DFO_DATASET_STORE_LIVE_DSN is set, as `test_live.py` is."""

from __future__ import annotations

import json
import re
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from dataset_store import LiveDatasetStore, connect, warning_formats
from dataset_store.dbapi import DBAPIConnection
from dataset_store.warning_formats import OLD_FORMATS, REWRITE_SQL, dry_run
from tests.dataset_store._warning_refs import rewrite_cases
from tests.dataset_store.test_live import (
    _DSN,
    _delete_versions,
    _insert_as_the_previous_image,
    _pool,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _DSN, reason="set DFO_DATASET_STORE_LIVE_DSN to run live dataset_store tests"
    ),
]

_NAMES = {code: name for name, code, _pattern in OLD_FORMATS}
_PATTERNS = {code: pattern for _name, code, pattern in OLD_FORMATS}
_COVERAGE, _CURRENCY, _DEFLATOR, _WITHHELD, _UN, _PROJECTED = _NAMES.values()


def _connect() -> DBAPIConnection:
    parts = urlsplit(_DSN)
    return connect(
        connector=None,
        instance_connection_name=None,
        host=parts.hostname,
        port=parts.port or 5432,
        database=parts.path.lstrip("/"),
        user=parts.username,
        password=parts.password,
    )


def _execute(conn: DBAPIConnection, sql: str, params: tuple = ()) -> Any:
    cursor = conn.cursor()
    try:
        cursor.execute(sql, params)
        return cast(Any, cursor).rowcount
    finally:
        cursor.close()


def _stored(conn: DBAPIConnection, table: str) -> dict[str, list]:
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT subject_id, warnings FROM {table}")
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.rollback()
    return {cast(str, subject): cast(list, warnings) for subject, warnings in rows}


@pytest.fixture
def live_store():
    pool = _pool()
    try:
        yield LiveDatasetStore(connections=pool)
    finally:
        pool.close_all()


@pytest.fixture
def scratch_table():
    """A version table in a schema of its own, holding only what a test inserts."""
    schema = f"warning_formats_{uuid4().hex}"
    conn = _connect()
    try:
        _execute(conn, f"CREATE SCHEMA {schema}")
        _execute(
            conn,
            f"CREATE TABLE {schema}.version_record (subject_id text, content_version text, "
            "warnings jsonb, PRIMARY KEY (subject_id, content_version))",
        )
        conn.commit()
        yield f"{schema}.version_record"
    finally:
        conn.rollback()
        _execute(conn, f"DROP SCHEMA {schema} CASCADE")
        conn.commit()
        conn.close()


def test_the_dry_run_changes_nothing_and_counts_what_the_rewrite_changes(scratch_table):
    """The dry run reads in a read-only transaction, leaves the rows and no function behind,
    and counts by format the rows a real run of the same SQL then changes, the rows left for
    the task and the residual row whose list holds a name with a comma. The real run writes
    each writer's shape, and a second dry run finds nothing left for the SQL."""
    cases = rewrite_cases()
    conn = _connect()
    try:
        for case, (stored, _rewritten) in cases.items():
            _execute(
                conn,
                f"INSERT INTO {scratch_table} VALUES (%s, 'cv1', %s::jsonb)",
                (case, json.dumps(stored)),
            )
        conn.commit()
        before = _stored(conn, scratch_table)

        report = dry_run(conn, table=scratch_table)

        assert _stored(conn, scratch_table) == before
        assert (
            _execute(
                conn,
                "SELECT 1 WHERE to_regprocedure('pg_temp.warning_counted_refs_0046(jsonb)') IS NULL",
            )
            == 1
        )
        conn.rollback()
    finally:
        conn.close()

    zero = dict.fromkeys(_NAMES.values(), 0)
    assert report.read_only == "on"
    assert (report.rewritten_rows, report.rewritten) == (
        7,
        {**zero, _COVERAGE: 3, _CURRENCY: 2, _DEFLATOR: 2},
    )
    assert (report.task_rows, report.task) == (
        5,
        {**zero, _COVERAGE: 2, _WITHHELD: 1, _UN: 1, _PROJECTED: 1},
    )
    assert (report.residual_rows, report.residual) == (1, {**zero, _COVERAGE: 1})
    expected_samples: dict[str, list[tuple[str, str]]] = {name: [] for name in _NAMES.values()}
    for stored, rewritten in cases.values():
        for old, new in zip(stored, rewritten, strict=True):
            code, message = cast(str, old["code"]), cast(str, old["message"])
            if code in _PATTERNS and re.search(_PATTERNS[code], message):
                expected_samples[_NAMES[code]].append((message, cast(str, new["message"])))
    assert {name: sorted(samples) for name, samples in report.samples.items()} == {
        name: sorted(samples) for name, samples in expected_samples.items()
    }
    assert {code: (rows, refs) for code, rows, refs in report.census} == {
        "coverage_below_scope": (6, 5),
        "currency_unmapped_omitted": (3, 2),
        "deflator_missing_omitted": (3, 2),
        "deflator_projected": (1, 1),
        "empty_result": (2, 0),
        "facet_default_selected": (1, 1),
        "stat_vars_summed": (1, 1),
        "un_system_unavailable": (1, 1),
        "withheld_by_registry": (1, 1),
    }

    conn = _connect()
    try:
        _execute(conn, REWRITE_SQL.read_text())
        updated = _execute(
            conn,
            f"UPDATE {scratch_table} SET warnings = pg_temp.warnings_counted_refs_0046(warnings) "
            "WHERE warnings IS DISTINCT FROM pg_temp.warnings_counted_refs_0046(warnings)",
        )
        conn.commit()
    finally:
        conn.close()
    assert updated == report.rewritten_rows

    conn = _connect()
    try:
        assert _stored(conn, scratch_table) == {
            case: rewritten for case, (_stored_warnings, rewritten) in cases.items()
        }
        after = dry_run(conn, table=scratch_table)
    finally:
        conn.close()
    assert (after.rewritten_rows, after.task_rows, after.residual_rows) == (0, 5, 1)


def test_the_command_reports_read_only_changes_nothing_and_exits_1_on_a_residual_row(
    live_store, monkeypatch, capsys
):
    """`--dry-run` against `datasets.version_record` prints that its transaction is read
    only, leaves the stored warnings as they were, and exits 1 while a residual row exists."""
    cases = rewrite_cases()
    subjects = {case: f"spec_warning_formats_{case}_{uuid4().hex}" for case in cases}

    def _warnings() -> dict[str, object]:
        with live_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT subject_id, warnings FROM datasets.version_record "
                "WHERE subject_id = ANY(%s)",
                (list(subjects.values()),),
            )
        return {cast(str, row["subject_id"]): row["warnings"] for row in rows}

    try:
        for case, (stored, _rewritten) in cases.items():
            _insert_as_the_previous_image(
                live_store,
                subject_id=subjects[case],
                content_version="cv1",
                body={"prepared_at": "2026-10-20T00:00:00Z", "warnings": stored},
            )
        before = _warnings()
        assert len(before) == len(subjects)

        parts = urlsplit(_DSN)
        for name, value in {
            "DATASET_STORE_HOST": parts.hostname,
            "DATASET_STORE_PORT": str(parts.port or 5432),
            "DATASET_STORE_DATABASE": parts.path.lstrip("/"),
            "DATASET_STORE_MIGRATOR_DB_USER": parts.username,
            "DATASET_STORE_MIGRATOR_DB_PASSWORD": parts.password,
        }.items():
            monkeypatch.setenv(name, cast(str, value))
        monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
        capsys.readouterr()

        assert warning_formats.main(["--dry-run"]) == 1

        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "transaction read only: on"
        residual = next(line for line in lines if line.startswith("residual rows: "))
        assert int(residual.removeprefix("residual rows: ")) >= 1
        assert _warnings() == before
    finally:
        _delete_versions(live_store, list(subjects.values()))
