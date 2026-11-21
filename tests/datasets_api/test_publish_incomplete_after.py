"""A computed version's record of the rows past the last complete year of their source
(`FrameLineage.incomplete_after`, `datasets_api.lineage_incomplete`): derived when it is
published from the records of the versions it was computed from, keyed by its own rows'
`provenance` and `stat_var` cells, and read by a card with no recipe and no release."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
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

from ._description_doubles import datasets_client, publish
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door_currency import _RenderableArtifactStore
from .test_publish_continuations import _FIRST, _ROOT, _SECOND, _SOURCE, _rows

_PARTIAL = "dcid:Amount_Partial"
_COMPLETE = "dcid:Amount_Complete"
_RATIO = "dcid:ops_ratio"


def _entry(stat_vars: list[str], year: int | None = 2024) -> dict[str, object]:
    return {
        "provenance": _SOURCE,
        "stat_vars": stat_vars,
        "source": None if year is None else _SOURCE,
        "last_complete_year": year,
    }


def _record_root(store: FakeVersionRecordDatasetStore, *, recorded: list[object] | None) -> None:
    """A dataset version of `_PARTIAL` and `_COMPLETE` from `_SOURCE` storing `recorded`,
    or none for `None`."""
    trace = {
        "facets": [{"axis": "provenance", "ref": _SOURCE}],
        "resolved_stat_vars": [{"ref": _PARTIAL}, {"ref": _COMPLETE}],
    }
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record (subject_id, subject_kind, content_version, "
            "schema_version, immutable_body, recorded_at, prepared_at, definition, coverage, "
            "citation, trace, warnings, incomplete_after) "
            "VALUES (%s, 'spec', %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
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
    takes the earliest last complete year among the root's entries. The rows reach 2026, so a
    card over the version flags exactly when a dated entry is stored, with no recipe read."""
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
    gives it the earliest partial year."""
    store = FakeVersionRecordDatasetStore()
    _record_root(store, recorded=[_entry([_PARTIAL])])
    both = _rows(_PARTIAL) + _rows(_COMPLETE)
    first, _ = _publish(store, _FIRST, inputs=[_ROOT[1]], rows=both)
    assert first.incomplete_after == [_entry([_COMPLETE], None), _entry([_PARTIAL])]

    second, flagged = _publish(store, _SECOND, inputs=[_FIRST[1]], rows=_rows(stat_var))

    assert second.incomplete_after == stored
    assert flagged is partial


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
