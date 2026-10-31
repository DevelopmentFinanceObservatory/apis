"""`warning_backfill`'s selection and compare-and-set update against a real Postgres, with a fake
graph. Skipped unless DFO_DATASET_STORE_LIVE_DSN is set, as `test_live.py` is."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import cast
from uuid import uuid4

import pytest

from dataset_store import LiveDatasetStore
from datasets_api.maintenance import _clears_on_retry
from datasets_api.warning_backfill import Readers, run_backfill_warnings, select_rows
from tests.dataset_store.test_live import _DSN, _delete_versions, _pool
from tests.materialise._scope_availability import ScopeAvailability
from tests.materialise.conftest import FakeGraphReader

from .test_warning_backfill import _FACETS, _IN, _NOW, _OUT, _code_index, _row, _warning

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _DSN, reason="set DFO_DATASET_STORE_LIVE_DSN to run live dataset_store tests"
    ),
]


@pytest.fixture
def live_store():
    pool = _pool()
    try:
        yield LiveDatasetStore(connections=pool)
    finally:
        pool.close_all()


def _insert(live_store: LiveDatasetStore, row: dict) -> None:
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "trace_id, recorded_at, prepared_at, definition, trace, warnings) "
            "VALUES (%s, 'spec', %s, 1, %s::jsonb, NULL, now(), now(), %s::jsonb, "
            "%s::jsonb, %s::jsonb)",
            (
                row["subject_id"],
                row["content_version"],
                json.dumps(row["immutable_body"]),
                json.dumps(row["definition"]),
                json.dumps(row["trace"]),
                json.dumps(row["warnings"]),
            ),
        )


def _warnings(live_store: LiveDatasetStore, subject_id: str) -> list:
    """The row's `warnings`, which its body holds too."""
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT warnings, immutable_body -> 'warnings' AS body_warnings "
            "FROM datasets.version_record WHERE subject_id = %s",
            (subject_id,),
        )
    assert row is not None
    assert row["body_warnings"] == row["warnings"]
    return cast(list, row["warnings"])


def _run(live_store: LiveDatasetStore, rows) -> int:
    return run_backfill_warnings(
        rows,
        store=live_store,
        readers=Readers(
            graph=FakeGraphReader(facets=_FACETS),
            availability=ScopeAvailability(data=()),
            code_index=_code_index(),
            coverage=None,
        ),
        dry_run=False,
        deadline=_NOW + timedelta(minutes=45),
        now=lambda: _NOW,
        clears_on_retry=_clears_on_retry,
    )


def test_the_selection_finds_old_sentences_and_the_update_holds_to_what_it_read(live_store):
    """The selection's pattern finds the old sentence in Postgres and the new-format clause the
    split rows. The update rewrites a row as read, leaves a row changed after the read, and a
    second selection finds only the row it left."""
    suffix = uuid4().hex
    rewritten, changed, split = (
        f"spec_{name}_{suffix}" for name in ("rewritten", "changed", "split")
    )
    split_warning = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": "The data covers less. 1 of 2 series hold no observations in the requested years.",
        "member_count": 2,
        "out_of_window_members": {"count": 1, "refs": ["dcid:Var_Out"]},
    }
    try:
        _insert(live_store, _row(_warning([_IN, _OUT]), zero=[_IN, _OUT], subject_id=rewritten))
        _insert(live_store, _row(_warning([_IN]), zero=[_IN], subject_id=changed))
        _insert(live_store, _row(split_warning, zero=[_OUT], subject_id=split))
        ours = {rewritten, changed, split}

        assert {row.subject_id for row in select_rows(live_store, new_format=False)} & ours == {
            rewritten,
            changed,
        }
        assert {row.subject_id for row in select_rows(live_store, new_format=True)} & ours == ours

        rows = [row for row in select_rows(live_store, new_format=False) if row.subject_id in ours]
        concurrent = [{**_warning([_IN]), "message": f"{_warning([_IN])['message']} Again."}]
        with live_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.version_record SET warnings = %s::jsonb, "
                "immutable_body = jsonb_set(immutable_body, '{warnings}', %s::jsonb) "
                "WHERE subject_id = %s",
                (json.dumps(concurrent), json.dumps(concurrent), changed),
            )

        assert _run(live_store, rows) == 1

        (written,) = _warnings(live_store, rewritten)
        assert written["off_cadence_members"] == {"count": 1, "refs": ["dcid:Var_In"]}
        assert written["out_of_window_members"] == {"count": 1, "refs": ["dcid:Var_Out"]}
        assert _warnings(live_store, changed) == concurrent
        assert {row.subject_id for row in select_rows(live_store, new_format=False)} & ours == {
            changed
        }
    finally:
        _delete_versions(live_store, [rewritten, changed, split])
