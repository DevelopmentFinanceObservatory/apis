"""A computed version's record of the rows past the last complete year of their source
(`FrameLineage.incomplete_after`, `datasets_api.lineage_incomplete`): derived when it is
published from the records of the versions it was computed from, keyed by its own rows'
`provenance` and `stat_var` cells, and read by a card with no recipe and no release."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from artifact_store import LiveArtifactInventory
from assembler.mint.provenance import _partial_period
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame
from dataset_store.snapshot import FrameLineage, LiveSnapshotReader
from datasets_api.manifest import version_document

from ._description_doubles import datasets_client, publish
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door_currency import _RenderableArtifactStore
from .test_publish_continuations import _FIRST, _ROOT, _SOURCE, _rows

_PARTIAL = "dcid:Amount_Partial"
_COMPLETE = "dcid:Amount_Complete"


def _entry(stat_vars: list[str], year: int = 2024) -> dict[str, object]:
    return {
        "provenance": _SOURCE,
        "stat_vars": stat_vars,
        "source": _SOURCE,
        "last_complete_year": year,
    }


def _record_root(store: FakeVersionRecordDatasetStore, *, recorded: list[object] | None) -> None:
    """A dataset version of `_PARTIAL` and `_COMPLETE` from `_SOURCE` whose `coverage` records
    `recorded`, or no member for `None`."""
    coverage: dict[str, object] = {"rows": 0, "dimensions": []}
    if recorded is not None:
        coverage["incomplete_after"] = recorded
    trace = {
        "facets": [{"axis": "provenance", "ref": _SOURCE}],
        "resolved_stat_vars": [{"ref": _PARTIAL}, {"ref": _COMPLETE}],
    }
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record (subject_id, subject_kind, content_version, "
            "schema_version, immutable_body, recorded_at, prepared_at, definition, coverage, "
            "citation, trace, warnings) VALUES (%s, 'spec', %s, 1, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                *_ROOT,
                json.dumps({}),
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC),
                json.dumps({}),
                json.dumps(coverage),
                json.dumps({}),
                json.dumps(trace),
                json.dumps([]),
            ),
        )


def _published(*, recorded: list[object] | None, stat_var: str) -> tuple[FrameLineage, bool]:
    """The lineage stored for a frame of `stat_var` computed from the root, published with
    no recipe, and whether a card over the whole frame is partial."""
    store = FakeVersionRecordDatasetStore()
    _record_root(store, recorded=recorded)
    client = datasets_client(
        store, artifact_store=_RenderableArtifactStore(inventory=LiveArtifactInventory(store=store))
    )
    response = publish(
        client,
        {
            "schema_version": 1,
            "spec_id": _FIRST[0],
            "content_version": _FIRST[1],
            "role_keys": ["recipient"],
            "rows": _rows(stat_var),
            "input_content_versions": [_ROOT[1]],
        },
    )
    assert response.status_code == 200, response.text
    stored = LiveSnapshotReader(store=store).read_version(*_FIRST)
    assert stored is not None and isinstance(stored.facts, FrameLineage)
    frame = build_resolved_frame(
        frame_ref=FrameRef(spec_id=_FIRST[0], content_version=_FIRST[1]),
        role_keys=("recipient",),
        rows=_rows(stat_var),
        manifest=version_document(stored),
    )
    return stored.facts, _partial_period(resolved_frame=frame, rows=None)


@pytest.mark.parametrize(
    ("recorded", "stat_var", "stored", "partial"),
    [
        pytest.param(
            [_entry([_PARTIAL])],
            "dcid:ops_ratio",
            [_entry(["dcid:ops_ratio"])],
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
            [],
            False,
            id="a-kept-complete-cell-of-a-partial-root",
        ),
        pytest.param([], "dcid:ops_ratio", [], False, id="a-ratio-over-a-complete-root"),
        pytest.param(None, "dcid:ops_ratio", None, False, id="a-root-recorded-before-the-member"),
    ],
)
def test_a_computed_version_records_its_own_rows_past_their_source_s_complete_year(
    recorded: list[object] | None,
    stat_var: str,
    stored: list[object] | None,
    partial: bool,
) -> None:
    """A computed row whose cells its root's record names takes the root's entry; a cell the
    root's rows carry with no entry is complete; a cell the computation made takes the
    earliest last complete year among the root's entries. The root's rows reach 2026, so a
    card over the version flags exactly when an entry is stored, with no recipe read."""
    lineage, flagged = _published(recorded=recorded, stat_var=stat_var)

    assert lineage.incomplete_after == stored
    assert flagged is partial
