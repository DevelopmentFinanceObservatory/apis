"""The warning rewrite `datasets-maintenance rewrite-warnings` runs, against a real Postgres. Skipped unless DFO_DATASET_STORE_LIVE_DSN is set, as `test_live.py`
is."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from dataset_store import LiveDatasetStore, warning_formats
from dataset_store.snapshot import ManifestFacts
from dataset_store.warning_formats import OLD_FORMATS, count_old_format_rows
from datasets_api.maintenance import run_rewrite_warnings
from datasets_api.version_store import _fill_year_coverage
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
    database_name,
    live_database,
)
from tests.dataset_store.test_live import _DSN, _insert_as_the_previous_image, _pool

pytestmark = [
    pytest.mark.live,
    pytest.mark.usefixtures("live_database"),
    pytest.mark.skipif(
        not _DSN, reason="set DFO_DATASET_STORE_LIVE_DSN to run live dataset_store tests"
    ),
]

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
    coverage: dict[str, object] | None = None,
) -> None:
    """A version row whose body holds `warnings` (none when `None`), `trace` and `coverage`, and
    whose column holds `column` when given, else what the insert trigger copies from the body."""
    body: dict[str, object] = {"prepared_at": _PREPARED_AT}
    if trace is not None:
        body["trace"] = trace
    if coverage is not None:
        body["coverage"] = coverage
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


def _rewritten(capsys, store: LiveDatasetStore) -> dict[str, object]:
    """Run the rewrite on `store` and return its `datasets_warnings_rewritten` line."""
    run_rewrite_warnings(store=store)
    return _line(capsys)


def _left(store: LiveDatasetStore) -> int:
    """The version rows still holding an old format, counted in a fresh transaction."""
    with store.transaction() as tx:
        return count_old_format_rows(tx)


def _line(capsys) -> dict[str, object]:
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (rewritten,) = [line for line in lines if line["event"] == "datasets_warnings_rewritten"]
    return rewritten


def _holds_old_format(warnings: list[StoredWarning]) -> bool:
    return any(
        re.search(OLD_FORMATS[code], cast(str, warning["message"])) is not None
        for warning in warnings
        if (code := cast(str, warning["code"])) in OLD_FORMATS
    )


@pytest.mark.parametrize("standard_conforming_strings", ["on", "off"])
def test_the_rewrite_rewrites_each_old_shape_in_the_column_and_the_body(
    live_store, capsys, monkeypatch, standard_conforming_strings
):
    """Every old shape but the member sentence becomes the writer's shape, in the column and in
    the body's `warnings`, each rewritten from its own stored value, whatever
    `standard_conforming_strings` the session runs under. A row whose warnings are already in
    the new shape, hold a member sentence alone or a name with a comma, or that holds no
    warnings at all, is not updated. The rewrite counts the rows it rewrote, which does not fail
    it, across pages of two rows, leaves the rows holding a shape it cannot rewrite, drops the
    rewrite's functions before each page commits, and on a rerun updates no row."""
    monkeypatch.setattr(warning_formats, "REWRITE_BATCH_ROWS", 2)
    with live_store.transaction() as tx:
        tx.execute(
            f'ALTER DATABASE "{database_name()}" '
            f"SET standard_conforming_strings = {standard_conforming_strings}"
        )
    cases = rewrite_cases()
    suffix = uuid4().hex
    subjects = {case: f"spec_rewrite_{case}_{suffix}" for case in cases}
    body_differs = f"spec_rewrite_body_differs_{suffix}"
    no_warnings = f"spec_rewrite_no_warnings_{suffix}"
    for case, (stored, _new) in cases.items():
        _insert(live_store, subjects[case], warnings=stored, trace=trace_for(case))
    _insert(live_store, body_differs, warnings=[_OLD_UNSERVED, _EMPTY], column=[_EMPTY])
    _insert(live_store, no_warnings, warnings=None)
    every = [*subjects.values(), body_differs, no_warnings]
    before = _rows(live_store, every)

    # A pool opened after the ALTER, so its one session runs under the setting.
    pool = _pool(size=1)
    try:
        rewriting = LiveDatasetStore(connections=pool)
        with rewriting.transaction() as tx:
            setting = tx.fetch_one("SHOW standard_conforming_strings")
        first = _rewritten(capsys, rewriting)
        after = _rows(live_store, every)
        left_after_first = _left(live_store)
        rerun = _rewritten(capsys, rewriting)
        with rewriting.transaction() as tx:
            functions = tx.fetch_one(
                "SELECT count(*) AS count FROM pg_proc "
                "WHERE pronamespace = pg_my_temp_schema() AND proname LIKE 'warning%'"
            )
    finally:
        pool.close_all()

    assert setting == {"standard_conforming_strings": standard_conforming_strings}
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

    changed = sum(stored != rewritten for stored, rewritten in cases.values()) + 1
    left = sum(_holds_old_format(rewritten) for _stored, rewritten in cases.values())
    assert left > 0
    assert (first["rewritten_warning_rows"], first["severity"]) == (changed, "INFO")
    assert left_after_first == left
    assert rerun["rewritten_warning_rows"] == 0
    assert _rows(live_store, every) == after
    assert _left(live_store) == left
    assert functions == {"count": 0}


def test_the_rewrite_rewrites_a_row_holding_an_old_warning_once(live_store, capsys):
    """A row holding an old shape in its column, and one holding it in its body alone, are each
    rewritten by the first rewrite and holds no old shape after it; a row in the new shape is
    left as it is."""
    suffix = uuid4().hex
    _insert(live_store, f"spec_rewrite_withheld_{suffix}", warnings=[_OLD_WITHHELD])
    _insert(
        live_store,
        f"spec_rewrite_body_{suffix}",
        warnings=[_OLD_UNSERVED, _EMPTY],
        column=[_EMPTY],
    )
    _insert(live_store, f"spec_rewrite_new_{suffix}", warnings=[new_withheld("Rate", ONE_WITHHELD)])

    first = _rewritten(capsys, live_store)
    left = _left(live_store)
    second = _rewritten(capsys, live_store)

    assert (first["rewritten_warning_rows"], left) == (2, 0)
    assert second["rewritten_warning_rows"] == 0


class _RewriteAfterRead:
    """A transaction that runs the rewrite on another connection once the coverage fill has read
    the stored coverage, before the fill writes."""

    def __init__(self, inner, *, rewriting: LiveDatasetStore, capsys) -> None:
        self._inner = inner
        self._rewriting = rewriting
        self._capsys = capsys

    def execute(self, sql: str, params=()) -> None:
        self._inner.execute(sql, params)

    def fetch_one(self, sql: str, params=()):
        row = self._inner.fetch_one(sql, params)
        if sql.startswith("SELECT coverage"):
            assert _rewritten(self._capsys, self._rewriting)["rewritten_warning_rows"] == 1
        return row


def test_a_coverage_fill_keeps_a_warning_rewrite_committed_after_its_read(live_store, capsys):
    """`_fill_year_coverage` reads the stored coverage, then writes the counted one. A rewrite
    that rewrites the row's warnings in between keeps its rewrite in the body as in the column:
    the fill sets the body's `coverage` alone."""
    subject = f"spec_fill_{uuid4().hex}"
    uncounted = {"rows": 2, "dimensions": [{"label": "provider"}]}
    counted = {"rows": 2, "dimensions": [{"label": "provider", "noun": "donors"}]}
    _insert(live_store, subject, warnings=[_OLD_WITHHELD], coverage=uncounted)
    record = SimpleNamespace(
        subject_id=subject,
        content_version="cv1",
        facts=ManifestFacts(
            prepared_at=datetime(2026, 10, 20, tzinfo=UTC),
            definition={},
            coverage=counted,
            citation={},
            trace={},
            warnings=[],
        ),
    )
    pool = _pool(size=1)
    try:
        rewriting = LiveDatasetStore(connections=pool)
        with live_store.transaction() as tx:
            _fill_year_coverage(_RewriteAfterRead(tx, rewriting=rewriting, capsys=capsys), record)
    finally:
        pool.close_all()

    ((_xmin, column, body),) = _rows(live_store, [subject]).values()
    assert column == [new_withheld("Rate", ONE_WITHHELD)]
    assert body == {
        "prepared_at": _PREPARED_AT,
        "coverage": counted,
        "warnings": [new_withheld("Rate", ONE_WITHHELD)],
    }
    with live_store.transaction() as tx:
        stored = tx.fetch_one(
            "SELECT coverage FROM datasets.version_record WHERE subject_id = %s", (subject,)
        )
    assert stored == {"coverage": counted}


def test_a_lock_timeout_on_a_late_page_keeps_the_pages_before_it(live_store, capsys, monkeypatch):
    """With pages of one row, a row lock held on the last row fails its page after
    `REWRITE_LOCK_TIMEOUT`. The pages before it stay committed, and the line is at `ERROR`
    counting their rows. The next run scans from the start, finds only the row
    left in an old format, rewrites it and leaves the committed rows as they are."""
    monkeypatch.setattr(warning_formats, "REWRITE_BATCH_ROWS", 1)
    monkeypatch.setattr(warning_formats, "REWRITE_LOCK_TIMEOUT", "200ms")
    suffix = uuid4().hex
    subjects = [f"spec_page_{position}_{suffix}" for position in range(3)]
    for subject in subjects:
        _insert(live_store, subject, warnings=[_OLD_WITHHELD])
    before = _rows(live_store, subjects)

    holder = _pool(size=1)
    try:
        with LiveDatasetStore(connections=holder).transaction() as held:
            held.execute(
                "SELECT 1 FROM datasets.version_record WHERE subject_id = %s FOR UPDATE",
                (subjects[-1],),
            )
            with pytest.raises(Exception, match="lock timeout"):
                run_rewrite_warnings(store=live_store)
    finally:
        holder.close_all()
    failed = _line(capsys)
    after = _rows(live_store, subjects)
    resumed = _rewritten(capsys, live_store)

    assert [after[subject][0] != before[subject][0] for subject in subjects] == [
        True,
        True,
        False,
    ]
    assert (failed["severity"], failed["rewritten_warning_rows"]) == ("ERROR", 2)
    assert (resumed["rewritten_warning_rows"], _left(live_store)) == (1, 0)
    assert _rows(live_store, subjects[:2]) == {subject: after[subject] for subject in subjects[:2]}
    assert _rows(live_store, subjects[2:])[subjects[2]][0] != before[subjects[2]][0]


def test_a_page_whose_commit_fails_logs_no_rewrite(live_store, capsys):
    """A constraint trigger refuses the page's commit. The line is at `ERROR`, counts no row
    rewritten, and the row keeps its old warning."""
    subject = f"spec_commit_{uuid4().hex}"
    _insert(live_store, subject, warnings=[_OLD_WITHHELD])
    before = _rows(live_store, [subject])
    with live_store.transaction() as tx:
        tx.execute(
            "CREATE FUNCTION datasets.refuse_commit() RETURNS trigger LANGUAGE plpgsql AS "
            "$$BEGIN RAISE EXCEPTION 'commit refused'; END$$"
        )
        tx.execute(
            "CREATE CONSTRAINT TRIGGER refuse_commit AFTER UPDATE ON datasets.version_record "
            "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
            "EXECUTE FUNCTION datasets.refuse_commit()"
        )

    with pytest.raises(Exception, match="commit refused"):
        run_rewrite_warnings(store=live_store)

    failed = _line(capsys)
    assert (failed["severity"], failed["rewritten_warning_rows"]) == ("ERROR", 0)
    assert _rows(live_store, [subject]) == before
