"""Migration 0047 and the count of old warning formats `datasets-maintenance settle` reads,
against a real Postgres. Skipped unless DFO_DATASET_STORE_LIVE_DSN is set, as `test_live.py`
is."""

from __future__ import annotations

import json
from typing import cast
from uuid import uuid4

import pytest

from dataset_store import LiveDatasetStore
from dataset_store.migrate import MIGRATIONS_DIR
from datasets_api.maintenance import run_settle
from tests.dataset_store._warning_refs import (
    COVERAGE_DETAIL,
    ONE_WITHHELD,
    UNSERVED,
    StoredWarning,
    new_coverage,
    new_withheld,
    old_unserved_sentence,
    old_withheld,
    rewrite_cases,
    trace_for,
)
from tests.dataset_store.live_database import (  # noqa: F401  (`live_database` is a fixture)
    connect_live,
    database_name,
    live_database,
)
from tests.dataset_store.test_live import (
    _DSN,
    _insert_as_the_previous_image,
    _pool,
    _rerun_migration,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.usefixtures("live_database"),
    pytest.mark.skipif(
        not _DSN, reason="set DFO_DATASET_STORE_LIVE_DSN to run live dataset_store tests"
    ),
]

_MIGRATION = "0047_warning_counted_refs.sql"
_PREPARED_AT = "2026-10-20T00:00:00Z"
_EMPTY: StoredWarning = {"code": "empty_result", "severity": "warn", "message": "No rows."}
_OLD_UNSERVED: StoredWarning = {
    "code": "coverage_below_scope",
    "severity": "warn",
    "message": COVERAGE_DETAIL + old_unserved_sentence(UNSERVED[:1]),
}
_OLD_WITHHELD: StoredWarning = {
    "code": "withheld_by_registry",
    "severity": "warn",
    "message": old_withheld("Rate", ONE_WITHHELD),
}


@pytest.fixture
def live_store():
    pool = _pool()
    try:
        yield LiveDatasetStore(connections=pool)
    finally:
        pool.close_all()


def _insert(
    live_store,
    subject_id: str,
    *,
    warnings: list[StoredWarning] | None,
    column: list[StoredWarning] | None = None,
    trace: dict[str, object] | None = None,
) -> None:
    """A version row whose body holds `warnings` (none when `None`) and `trace`, and whose
    column holds `column` when given, else what the insert trigger copies from the body."""
    body: dict[str, object] = {"prepared_at": _PREPARED_AT}
    if trace is not None:
        body["trace"] = trace
    if warnings is not None:
        body["warnings"] = warnings
    _insert_as_the_previous_image(
        live_store, subject_id=subject_id, content_version="cv1", body=body
    )
    if column is not None:
        with live_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.version_record SET warnings = %s::jsonb WHERE subject_id = %s",
                (json.dumps(column), subject_id),
            )


def _rows(live_store, subjects: list[str]) -> dict[str, tuple[str, object, object]]:
    """Each row's `xmin`, which only an UPDATE of the row moves, its column and its body."""
    with live_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, xmin::text AS xmin, warnings, immutable_body "
            "FROM datasets.version_record WHERE subject_id = ANY(%s)",
            (subjects,),
        )
    return {
        cast(str, row["subject_id"]): (
            cast(str, row["xmin"]),
            row["warnings"],
            row["immutable_body"],
        )
        for row in rows
    }


@pytest.mark.parametrize("standard_conforming_strings", ["on", "off"])
def test_0047_rewrites_each_old_shape_in_the_column_and_the_body(
    live_store, standard_conforming_strings
):
    """Every old shape but the member sentence becomes the writer's shape, in the column and in
    the body's `warnings`, each rewritten from its own stored value, whatever
    `standard_conforming_strings` the session runs under. A row whose warnings are already in
    the new shape, hold a member sentence alone or a name with a comma, or that holds no
    warnings at all, is not updated, and a rerun updates no row."""
    with live_store.transaction() as tx:
        tx.execute(
            f'ALTER DATABASE "{database_name()}" '
            f"SET standard_conforming_strings = {standard_conforming_strings}"
        )
    cases = rewrite_cases()
    suffix = uuid4().hex
    subjects = {case: f"spec_0047_{case}_{suffix}" for case in cases}
    body_differs = f"spec_0047_body_differs_{suffix}"
    no_warnings = f"spec_0047_no_warnings_{suffix}"
    for case, (stored, _rewritten) in cases.items():
        _insert(live_store, subjects[case], warnings=stored, trace=trace_for(case))
    _insert(live_store, body_differs, warnings=[_OLD_UNSERVED, _EMPTY], column=[_EMPTY])
    _insert(live_store, no_warnings, warnings=None)
    every = [*subjects.values(), body_differs, no_warnings]
    before = _rows(live_store, every)

    assert _rerun_migration(live_store, _MIGRATION) == [_MIGRATION]

    after = _rows(live_store, every)
    for case, (stored, rewritten) in cases.items():
        xmin, column, body = after[subjects[case]]
        assert column == rewritten, case
        assert body == {**cast(dict, before[subjects[case]][2]), "warnings": rewritten}, case
        assert (xmin == before[subjects[case]][0]) == (stored == rewritten), case
    xmin, column, body = after[body_differs]
    assert column == [_EMPTY]
    assert cast(dict, body)["warnings"] == [new_coverage(unserved=UNSERVED[:1]), _EMPTY]
    assert xmin != before[body_differs][0]
    assert after[no_warnings] == before[no_warnings]
    assert before[no_warnings][1] is None

    assert _rerun_migration(live_store, _MIGRATION) == [_MIGRATION]
    assert _rows(live_store, every) == after


def test_0047_leaves_no_function_behind_in_its_session():
    """The rewrite's functions live in the migration's session and are dropped before it
    commits, so a later migration in the same session defines its own."""
    conn = connect_live()
    try:
        cursor = conn.cursor()
        cursor.execute((MIGRATIONS_DIR / _MIGRATION).read_text())
        cursor.execute(
            "SELECT count(*) FROM pg_proc "
            "WHERE pronamespace = pg_my_temp_schema() AND proname LIKE 'warning%'"
        )
        assert cast(list, cursor.fetchone())[0] == 0
    finally:
        conn.rollback()
        conn.close()


def test_settle_counts_a_row_holding_an_old_warning_until_0047_rewrites_it(live_store):
    """A row holding an old shape in its column, and one holding it in its body alone, each add
    one to settle's count until migration 0047 rewrites them; a row in the new shape adds none."""
    suffix = uuid4().hex
    _insert(live_store, f"spec_settle_withheld_{suffix}", warnings=[_OLD_WITHHELD])
    _insert(
        live_store,
        f"spec_settle_body_{suffix}",
        warnings=[_OLD_UNSERVED, _EMPTY],
        column=[_EMPTY],
    )
    _insert(live_store, f"spec_settle_new_{suffix}", warnings=[new_withheld("Rate", ONE_WITHHELD)])

    assert run_settle(store=live_store) == 2

    assert _rerun_migration(live_store, _MIGRATION) == [_MIGRATION]
    assert run_settle(store=live_store) == 0
