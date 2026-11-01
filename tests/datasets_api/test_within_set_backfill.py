"""`datasets-maintenance backfill-within-sets` fills the set a record written before records
held `within_set` leaves out, and its title with it.

A record whose definition binds a set and whose dimension holds neither `within` nor
`within_set` is selected. A dry run lists it with its old and new titles and writes nothing; an
apply fills the set as a late field with basis `backfill` and re-titles the record. A record
holding its set, and a record whose definition binds no set, are left alone, and a second run
selects nothing.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from datasets_api import maintenance
from datasets_api.description_record import UnrecordedSets, unrecorded_sets
from datasets_api.maintenance import run_backfill_within_sets
from description.facts import DescriptionRecord

from . import test_serve_groupby_collapse as family_fixture
from ._description_doubles import FAMILY_AXIS, read_record
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_description_backfill import _backfill, _family_body, _stored_without_record

_SPEC_ID = family_fixture._spec_id_and_sidecar()[0]
_BEFORE_SETS = "cv_" + "a1" * 16
_WITH_SET = "cv_" + "b2" * 16
_SETLESS = "cv_" + "c3" * 16
_STALE_TITLE = "Stale title composed before sets were recorded"
_OLD_TITLE = "Aid to Kenya"
_NEW_TITLE = "Aid to Kenya for Axis value A and Axis value B"
_FILLED_AT = datetime(2026, 10, 31, 6, 0, tzinfo=UTC)


def _setless_body() -> dict[str, Any]:
    """The family body with its axis bound to one code with no descendants."""
    body = _family_body()
    identity = copy.deepcopy(body["definition"]["identity"])
    (slot,) = identity["slots"]
    slot["binding"] = {"state": "value", "ref": "TestAxisA"}
    body["definition"] = {**body["definition"], "identity": identity}
    return body


def _as_written_before_sets(store: FakeVersionRecordDatasetStore, content_version: str) -> None:
    """Rewrite the stored record as a mint before sets were recorded wrote it: schema version
    1, no `within_set`, and a title composed without the set. Its one variable and the labels
    of that variable and the recipient, as a mint reads them, let its title compose."""
    with store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT record FROM datasets.description_record "
            "WHERE subject_id = %s AND content_version = %s",
            (_SPEC_ID, content_version),
        )
        assert row is not None
        record: dict[str, Any] = copy.deepcopy(cast("dict[str, Any]", row["record"]))
        for dimension in record["variables"]["dimensions"]:
            dimension.pop("within_set", None)
        record["schema_version"] = 1
        record["evidence"]["title"] = _STALE_TITLE
        record["variables"]["variable"] = family_fixture._MEMBER_A
        record["evidence"]["labels"] = sorted(
            [
                *record["evidence"]["labels"],
                {"ref": family_fixture._MEMBER_A, "label": "Aid", "read": "2026-10-20"},
                {"ref": family_fixture._RECIPIENT, "label": "Kenya", "read": "2026-10-20"},
            ],
            key=lambda label: label["ref"],
        )
        tx.execute(
            "UPDATE datasets.description_record SET schema_version = 1, record = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (json.dumps(record), _SPEC_ID, content_version),
        )


def _store() -> FakeVersionRecordDatasetStore:
    store = FakeVersionRecordDatasetStore()
    for content_version, body in (
        (_BEFORE_SETS, _family_body()),
        (_WITH_SET, _family_body()),
        (_SETLESS, _setless_body()),
    ):
        _stored_without_record(
            store, subject_id=_SPEC_ID, content_version=content_version, body=body
        )
    assert _backfill(store) == 0
    _as_written_before_sets(store, _BEFORE_SETS)
    return store


def _records(store: FakeVersionRecordDatasetStore) -> dict[str, DescriptionRecord | None]:
    return {
        cv: read_record(store, subject_id=_SPEC_ID, content_version=cv)
        for cv in (_BEFORE_SETS, _WITH_SET, _SETLESS)
    }


def _run(store: FakeVersionRecordDatasetStore, capsys, *, apply: bool) -> list[dict[str, Any]]:
    capsys.readouterr()
    assert (
        run_backfill_within_sets(
            store=store,
            code_index=family_fixture._code_index(),
            apply=apply,
            now=lambda: _FILLED_AT,
        )
        == 0
    )
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def _axis(record: DescriptionRecord | None):
    assert record is not None
    assert record.variables is not None
    (axis,) = [d for d in record.variables.dimensions or () if d.property == FAMILY_AXIS]
    return axis


def test_a_dry_run_lists_the_record_without_its_set_and_writes_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _store()
    before = _records(store)

    lines = _run(store, capsys, apply=False)

    (planned,) = [line for line in lines if line["event"] == "datasets_within_set_backfill_planned"]
    assert (planned["subject_id"], planned["content_version"]) == (_SPEC_ID, _BEFORE_SETS)
    assert (planned["old_title"], planned["new_title"]) == (_OLD_TITLE, _NEW_TITLE)
    (summary,) = [line for line in lines if line["event"] == "datasets_within_sets_backfilled"]
    assert (summary["selected"], summary["filled"], summary["applied"]) == (1, 0, False)
    assert _records(store) == before


def test_an_apply_fills_the_set_and_titles_the_record_as_a_fresh_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _store()
    before = _records(store)
    fresh = before[_WITH_SET]
    assert fresh is not None

    lines = _run(store, capsys, apply=True)

    after = _records(store)
    filled = after[_BEFORE_SETS]
    assert filled is not None
    assert _axis(filled).within_set == _axis(fresh).within_set == ["TestAxisA", "TestAxisB"]
    assert filled.schema_version == 2
    assert {(late.field, late.basis) for late in filled.late_fields} == {
        ("variables.dimensions.0.within_set", "backfill"),
        ("evidence.title", "backfill"),
    }
    (line,) = [line for line in lines if line["event"] == "datasets_within_set_backfill_filled"]
    assert (line["old_title"], line["new_title"]) == (_OLD_TITLE, _NEW_TITLE)
    assert filled.evidence.title == _NEW_TITLE
    assert after[_WITH_SET] == before[_WITH_SET]
    assert after[_SETLESS] == before[_SETLESS]

    again = _run(store, capsys, apply=True)

    (summary,) = [line for line in again if line["event"] == "datasets_within_sets_backfilled"]
    assert (summary["selected"], summary["filled"]) == (0, 0)
    assert _records(store) == after


def test_an_apply_a_concurrent_filler_beat_reports_no_fill(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record another run filled between this run's read and its write gets no
    `datasets_within_set_backfill_filled` line, so the lines match the `filled` count."""
    store = _store()
    monkeypatch.setattr(maintenance, "fill_late_fields", lambda tx, **_: [])

    lines = _run(store, capsys, apply=True)

    assert not [line for line in lines if line["event"] == "datasets_within_set_backfill_filled"]
    (summary,) = [line for line in lines if line["event"] == "datasets_within_sets_backfilled"]
    assert (summary["selected"], summary["filled"]) == (1, 0)


def test_a_collapse_that_summed_its_set_away_is_reported_not_filled() -> None:
    """A summed collapse made before sets were recorded holds no dimension for its set."""
    record = _records(_store())[_BEFORE_SETS]
    assert record is not None
    assert record.variables is not None
    summed_away = record.model_copy(
        update={
            "variables": record.variables.model_copy(
                update={
                    "dimensions": [
                        d for d in record.variables.dimensions or () if d.property != FAMILY_AXIS
                    ]
                }
            )
        }
    )
    definition = _family_body()["definition"]

    assert unrecorded_sets(
        summed_away, definition=definition, code_index=family_fixture._code_index()
    ) == UnrecordedSets(fills={}, dropped=[FAMILY_AXIS])
