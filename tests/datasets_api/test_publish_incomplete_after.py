"""A computed version's record of how complete its rows are
(`FrameLineage.incomplete_after`, `datasets_api.lineage_incomplete`): derived when it is
published from the records of the versions it was computed from, keyed by its own rows'
`provenance` and `stat_var` cells, and read by a card with no recipe and no release."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from dfo_frame_grammar import Frame

from artifact_store import LiveArtifactInventory
from assembler.mint.provenance import _partial_period
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame
from dataset_store.snapshot import FrameLineage, LiveSnapshotReader, SnapshotReader
from datasets_api.lineage_incomplete import derived_incomplete_after
from datasets_api.manifest import version_document
from tests.materialise._ledger import LEDGER

from ._description_doubles import datasets_client, publish
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door_currency import _RenderableArtifactStore
from .test_publish_continuations import _FIRST, _ROOT, _SECOND, _SOURCE, _rows

_PARTIAL = "dcid:Amount_Partial"
_COMPLETE = "dcid:Amount_Complete"
_RATIO = "dcid:ops_ratio"


def _entry(
    stat_vars: list[str], mark: str | None = "2025", *, source: str = _SOURCE
) -> dict[str, object]:
    """A `partial` entry from `mark` of `source`, or `unmarked` for `None`."""
    return {
        "provenance": _SOURCE,
        "stat_vars": stat_vars,
        "source": None if mark is None else source,
        "status": "unmarked" if mark is None else "partial",
        "partial_period": mark,
    }


def _projected(stat_vars: list[str], start: str = "2026") -> dict[str, object]:
    """A `projected` entry of `_SOURCE`'s rows from `start`, as an opted-in IDS mint records."""
    return {
        "provenance": _SOURCE,
        "stat_vars": stat_vars,
        "source": _SOURCE,
        "status": "projected",
        "partial_period": start,
    }


def _unknown(stat_vars: list[str], *, source: str | None = _SOURCE) -> dict[str, object]:
    return {
        "provenance": _SOURCE,
        "stat_vars": stat_vars,
        "source": source,
        "status": "unknown",
        "partial_period": None,
    }


def _record_root(
    store: FakeVersionRecordDatasetStore,
    *,
    recorded: list[object] | None,
    built_at: datetime | None = None,
    source: str = _SOURCE,
) -> None:
    """A dataset version of `_PARTIAL` and `_COMPLETE` from `source` storing `recorded`,
    or none for `None`, judged by the catalogue build `built_at`."""
    trace = {
        "facets": [{"axis": "provenance", "ref": source}],
        "resolved_stat_vars": [{"ref": _PARTIAL}, {"ref": _COMPLETE}],
    }
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record (subject_id, subject_kind, content_version, "
            "schema_version, immutable_body, recorded_at, prepared_at, definition, coverage, "
            "citation, trace, warnings, incomplete_after, incomplete_after_built_at) "
            "VALUES (%s, 'spec', %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                *_ROOT,
                json.dumps({}),
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC),
                json.dumps({}),
                json.dumps({"rows": 0, "dimensions": []}),
                json.dumps({}),
                json.dumps(trace),
                json.dumps([]),
                None if recorded is None else json.dumps(recorded),
                built_at,
            ),
        )


def _publish(
    store: FakeVersionRecordDatasetStore,
    version: tuple[str, str],
    *,
    inputs: Sequence[str],
    rows: list[dict[str, object]],
) -> tuple[FrameLineage, bool]:
    """Publish `rows` as `version` computed from `inputs`, with no recipe. Returns the
    lineage stored and whether a card over the whole frame is partial."""
    client = datasets_client(
        store, artifact_store=_RenderableArtifactStore(inventory=LiveArtifactInventory(store=store))
    )
    response = publish(
        client,
        {
            "schema_version": 1,
            "spec_id": version[0],
            "content_version": version[1],
            "role_keys": ["recipient"],
            "rows": rows,
            "input_content_versions": list(inputs),
        },
    )
    assert response.status_code == 200, response.text
    stored = LiveSnapshotReader(store=store).read_version(*version)
    assert stored is not None and isinstance(stored.facts, FrameLineage)
    frame = build_resolved_frame(
        frame_ref=FrameRef(spec_id=version[0], content_version=version[1]),
        role_keys=("recipient",),
        rows=rows,
        manifest=version_document(stored),
    )
    return stored.facts, _partial_period(resolved_frame=frame, rows=None)


def _published(*, recorded: list[object] | None, stat_var: str) -> tuple[FrameLineage, bool]:
    store = FakeVersionRecordDatasetStore()
    _record_root(store, recorded=recorded)
    return _publish(store, _FIRST, inputs=[_ROOT[1]], rows=_rows(stat_var))


@pytest.mark.parametrize(
    ("recorded", "stat_var", "stored", "partial"),
    [
        pytest.param(
            [_entry([_PARTIAL])],
            _RATIO,
            [_entry([_RATIO])],
            True,
            id="control-a-ratio-over-a-partial-root",
        ),
        pytest.param(
            [_entry([_PARTIAL])],
            _PARTIAL,
            [_entry([_PARTIAL])],
            True,
            id="a-kept-partial-cell-carries-its-entry",
        ),
        pytest.param(
            [_entry([_PARTIAL])],
            _COMPLETE,
            [_entry([_COMPLETE], None)],
            False,
            id="a-kept-complete-cell-of-a-partial-root",
        ),
        pytest.param([], _RATIO, [_entry([_RATIO], None)], False, id="over-a-complete-root"),
        pytest.param(
            [_unknown([_PARTIAL])],
            _RATIO,
            [_unknown([_RATIO], source=None)],
            False,
            id="a-ratio-over-an-unjudged-root-cell",
        ),
        pytest.param(
            [_unknown([_PARTIAL])],
            _COMPLETE,
            [_entry([_COMPLETE], None)],
            False,
            id="a-kept-cell-beside-an-unjudged-one",
        ),
        pytest.param(
            [_entry([_PARTIAL], "2025-06", source="provenance/B"), _entry([_COMPLETE], "2025")],
            _RATIO,
            [_entry([_RATIO], "2025")],
            True,
            id="the-earliest-mark-at-its-own-grain",
        ),
        pytest.param(
            [_entry([_PARTIAL], source="provenance/B"), _entry([_COMPLETE], source="provenance/A")],
            _RATIO,
            [_entry([_RATIO], source="provenance/A")],
            True,
            id="equal-marks-break-on-source",
        ),
        pytest.param(
            [_projected([_PARTIAL])],
            _RATIO,
            [_projected([_RATIO])],
            False,
            id="a-ratio-over-a-projected-root",
        ),
        pytest.param(
            [_projected([_PARTIAL])],
            _PARTIAL,
            [_projected([_PARTIAL])],
            False,
            id="a-kept-projected-cell-carries-its-entry",
        ),
        pytest.param(
            [_entry([_PARTIAL]), _projected([_COMPLETE])],
            _RATIO,
            [_entry([_RATIO]), _projected([_RATIO])],
            True,
            id="a-ratio-over-partial-and-projected-cells-is-both",
        ),
        pytest.param(None, _RATIO, None, False, id="a-root-recorded-before-the-member"),
    ],
)
def test_a_computed_version_records_its_own_rows_past_their_source_s_complete_year(
    recorded: list[object] | None,
    stat_var: str,
    stored: list[object] | None,
    partial: bool,
) -> None:
    """A computed row whose cells its root's record names takes the root's entry; a cell the
    root's rows carry with no entry is complete, and listed so; a cell the computation made
    takes the earliest partial period among the root's entries (by its start, ties broken by
    source), else unknown beside an unjudged root cell. The rows reach 2026, so a card over the
    version flags exactly when a partial entry is stored, with no recipe read."""
    lineage, flagged = _published(recorded=recorded, stat_var=stat_var)

    assert lineage.incomplete_after == stored
    assert flagged is partial


@pytest.mark.parametrize(
    ("stat_var", "stored", "partial"),
    [
        pytest.param(_PARTIAL, [_entry([_PARTIAL])], True, id="control-a-kept-partial-cell"),
        pytest.param(_COMPLETE, [_entry([_COMPLETE], None)], False, id="a-kept-complete-cell"),
    ],
)
def test_a_computation_over_a_computed_version_knows_its_input_s_cells(
    stat_var: str, stored: list[object], partial: bool
) -> None:
    """The first computation keeps both root cells; the second keeps one of them. The first's
    record lists its complete cell, so the second reads it as kept and complete, and never
    gives it the earliest partial period."""
    store = FakeVersionRecordDatasetStore()
    _record_root(store, recorded=[_entry([_PARTIAL])])
    both = _rows(_PARTIAL) + _rows(_COMPLETE)
    first, _ = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=both)
    assert first.incomplete_after == [_entry([_COMPLETE], None), _entry([_PARTIAL])]

    second, flagged = _publish(store, _SECOND, inputs=[_FIRST[1]], rows=_rows(stat_var))

    assert second.incomplete_after == stored
    assert flagged is partial


def test_a_kept_complete_cell_of_a_declared_dataset_s_partial_root_is_unmarked() -> None:
    """The root's trace selects `dataset/Ledger` and its rows carry that id, as every row of
    the dataset's member provenances does (`lineage_incomplete._dataset_cells`), so the kept
    complete cell is the root's and takes none of its partial periods."""
    store = FakeVersionRecordDatasetStore()
    partial = {**_entry([_PARTIAL], source=LEDGER), "provenance": LEDGER}
    _record_root(store, recorded=[partial], source=LEDGER)
    rows = [{**row, "provenance": LEDGER} for row in _rows(_COMPLETE)]

    lineage, flagged = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=rows)

    assert lineage.incomplete_after == [{**_entry([_COMPLETE], None), "provenance": LEDGER}]
    assert flagged is False


def _without_provenance(stat_var: str) -> list[dict[str, object]]:
    return [{k: v for k, v in row.items() if k != "provenance"} for row in _rows(stat_var)]


@pytest.mark.parametrize(
    ("stat_var", "stored", "partial"),
    [
        pytest.param(
            _PARTIAL,
            [{**_entry([_PARTIAL]), "provenance": None}],
            True,
            id="control-its-variable-partial-wherever-carried",
        ),
        pytest.param(
            _COMPLETE,
            [{**_entry([_COMPLETE], None), "provenance": None}],
            False,
            id="its-variable-complete-wherever-carried",
        ),
    ],
)
def test_a_frame_with_no_provenance_column_matches_kept_cells_by_variable(
    stat_var: str, stored: list[object], partial: bool
) -> None:
    store = FakeVersionRecordDatasetStore()
    _record_root(store, recorded=[_entry([_PARTIAL])])

    lineage, flagged = _publish(
        store, _FIRST, inputs=[_ROOT[1]], rows=_without_provenance(stat_var)
    )

    assert lineage.incomplete_after == stored
    assert flagged is partial


def test_a_computation_over_no_inputs_records_nothing() -> None:
    """With no input to take a judgement from, the record is unknown, and nothing is read."""
    frame = Frame.build(role_keys=("recipient",), rows=_rows(_RATIO))

    assert derived_incomplete_after([], reader=cast("SnapshotReader", None), frame=frame) is None


def test_a_republished_computed_version_takes_the_record_derived_then() -> None:
    """The root was recorded before versions stored a record, so the first publish stores
    none. The root's record then lands, and publishing the same version again stores the one
    derived from it."""
    store = FakeVersionRecordDatasetStore()
    _record_root(store, recorded=None)
    first, _ = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=_rows(_PARTIAL))
    assert first.incomplete_after is None
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET incomplete_after = %s WHERE content_version = %s",
            (json.dumps([_entry([_PARTIAL])]), _ROOT[1]),
        )

    again, flagged = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=_rows(_PARTIAL))

    assert again.incomplete_after == [_entry([_PARTIAL])]
    assert flagged


def _rejudge_root(
    store: FakeVersionRecordDatasetStore, recorded: list[object], built_at: datetime | None
) -> None:
    """The root's record as the catalogue build `built_at` judged it."""
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record "
            "SET incomplete_after = %s, incomplete_after_built_at = %s WHERE content_version = %s",
            (json.dumps(recorded), built_at, _ROOT[1]),
        )


def test_a_republish_reading_an_older_input_judgement_keeps_the_newer_record() -> None:
    """Two republishes of one computed version read their root under different judgements.
    Each record takes the newest build among the judgements it read, so the republish that
    read the root's older judgement, or one no known build made, leaves the newer record
    standing, and one that read a later judgement replaces it."""
    built = datetime(2026, 11, 1, tzinfo=UTC)
    store = FakeVersionRecordDatasetStore()
    _record_root(store, recorded=[], built_at=built)
    first, _ = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=_rows(_PARTIAL))
    assert (first.incomplete_after, first.incomplete_after_built_at) == (
        [_entry([_PARTIAL], None)],
        built,
    )

    for stale in (built - timedelta(days=7), None):
        _rejudge_root(store, [_entry([_PARTIAL])], stale)
        kept, flagged = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=_rows(_PARTIAL))
        assert (kept.incomplete_after, kept.incomplete_after_built_at) == (
            [_entry([_PARTIAL], None)],
            built,
        )
        assert not flagged

    later = built + timedelta(days=7)
    _rejudge_root(store, [_entry([_PARTIAL])], later)
    replaced, flagged = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=_rows(_PARTIAL))
    assert (replaced.incomplete_after, replaced.incomplete_after_built_at) == (
        [_entry([_PARTIAL])],
        later,
    )
    assert flagged


def test_an_input_judgement_naming_no_build_leaves_the_record_s_build_unknown() -> None:
    """A computation over a stamped computed version and a root whose judgement names no build
    knows no build for its record, though the other input's is known."""
    built = datetime(2026, 11, 1, tzinfo=UTC)
    store = FakeVersionRecordDatasetStore()
    _record_root(store, recorded=[], built_at=built)
    first, _ = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=_rows(_PARTIAL))
    assert first.incomplete_after_built_at == built
    _rejudge_root(store, [], None)

    second, _ = _publish(store, _SECOND, inputs=[_ROOT[1], _FIRST[1]], rows=_rows(_PARTIAL))

    assert second.incomplete_after is not None
    assert second.incomplete_after_built_at is None
