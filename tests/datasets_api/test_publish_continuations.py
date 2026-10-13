"""A computed version's continuations: derived when it is published from the versions it was
computed from, scoped to the series they continued in their root, carried unchanged through
computed inputs, stored with its lineage, and filled by `backfill-descriptions` on versions
recorded before versions stored them. Its description counts series as Presentation groups
them."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from artifact_store import LiveArtifactInventory
from assembler.frame_view import (
    compute_structural_view,
    continuation_known,
    frame_continuation,
)
from assembler.testing import build_frame_doc
from dataset_store.snapshot import FrameLineage, LiveSnapshotReader
from datasets_api.maintenance import run_backfill_descriptions
from datasets_api.manifest import version_document
from materialise.continued_series import ContinuedSeries

from ._description_doubles import datasets_client, forget_record, publish, read_record
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door_currency import _RenderableArtifactStore
from .test_serve_groupby_collapse import _code_index

_SOURCE = "provenance/IMFWorldEconomicOutlook"
_ROOT = ("spec_" + "a1" * 16, "cv_" + "b2" * 16)
_FIRST = ("ops_" + "c3" * 16, "ocv_" + "d4" * 16)
_SECOND = ("ops_" + "e5" * 16, "ocv_" + "f6" * 16)

# The root's series: reported under a null method to 2024, continued by the source's
# estimates for 2025-2026.
_CONTINUED = ContinuedSeries(method="Estimated", continues=None, provenance=_SOURCE)


def _rows(stat_var: str) -> list[dict[str, object]]:
    return [
        {
            "stat_var": stat_var,
            "recipient": "country/ZMB",
            "time_period": str(year),
            "value": float(year - 2015),
            "measurementMethod": "Estimated" if year >= 2025 else None,
            "provenance": _SOURCE,
        }
        for year in range(2016, 2027)
    ]


def _retrieve(version: tuple[str, str]) -> dict[str, object]:
    return {
        "id": "s1",
        "op": "retrieve",
        "input": {"id": version[0], "content_version": version[1]},
        "options": {},
    }


def _body(
    version: tuple[str, str], *, inputs: tuple[str, str], stat_var: str, steps: bool = True
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "spec_id": version[0],
        "content_version": version[1],
        "role_keys": ["recipient"],
        "rows": _rows(stat_var),
        "input_content_versions": [inputs[1]],
    }
    if steps:
        body["steps"] = [
            _retrieve(inputs),
            {"id": "s2", "op": "growth", "input": "s1", "stat_var": stat_var.removeprefix("dcid:")},
        ]
    return body


def _record_root(store: FakeVersionRecordDatasetStore, *, continuations: list[object]) -> None:
    """A dataset version whose trace records `continuations` of its null-method series from
    `_SOURCE`."""
    trace = {
        "facets": [
            {"axis": "provenance", "ref": _SOURCE},
            {"axis": "measurementMethod", "ref": None},
        ],
        "continuations": continuations,
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
                json.dumps({}),
                json.dumps({}),
                json.dumps(trace),
                json.dumps([]),
            ),
        )


def _published_chain(
    *, continuations: list[object], second_steps: bool = True
) -> tuple[FakeVersionRecordDatasetStore, _RenderableArtifactStore]:
    """The root, a growth computed from it, and a second growth computed from the first."""
    store = FakeVersionRecordDatasetStore()
    artifacts = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=store))
    _record_root(store, continuations=continuations)
    client = datasets_client(store, artifact_store=artifacts)
    for body in (
        _body(_FIRST, inputs=_ROOT, stat_var="dcid:ops_growth"),
        _body(_SECOND, inputs=_FIRST, stat_var="dcid:ops_growth_twice", steps=second_steps),
    ):
        response = publish(client, body)
        assert response.status_code == 200, response.text
    return store, artifacts


def _stored(store: FakeVersionRecordDatasetStore, version: tuple[str, str]) -> FrameLineage:
    stored = LiveSnapshotReader(store=store).read_version(*version)
    assert stored is not None
    assert isinstance(stored.facts, FrameLineage)
    return stored.facts


def _document(store: FakeVersionRecordDatasetStore, version: tuple[str, str]) -> dict[str, object]:
    stored = LiveSnapshotReader(store=store).read_version(*version)
    assert stored is not None
    return version_document(stored)


def _presentation_series(store: FakeVersionRecordDatasetStore, version: tuple[str, str]) -> int:
    """The series Presentation groups the version's rows into, from its stored manifest."""
    stored = LiveSnapshotReader(store=store).read_version(*version)
    assert stored is not None
    doc = build_frame_doc(role_keys=("recipient",), rows=_rows("dcid:ops_growth_twice"))
    continuation = frame_continuation(manifest=version_document(stored))
    return compute_structural_view(frame=doc, continuation=continuation).series_count


_ROOT_CONTINUATION = {
    "axis": "measurementMethod",
    "method": "Estimated",
    "periods": ["2025", "2026"],
}


@pytest.mark.parametrize(
    ("continuations", "stored", "series"),
    [
        pytest.param([_ROOT_CONTINUATION], [_CONTINUED.as_entry()], 1, id="a-continued-root"),
        pytest.param([], [], 2, id="control-a-root-nothing-continues"),
    ],
)
@pytest.mark.parametrize("second_steps", [True, False], ids=["with-steps", "lineage-only"])
def test_a_chain_of_computations_keeps_its_roots_continuation(
    continuations: list[object], stored: list[object], series: int, second_steps: bool
) -> None:
    """A growth of a growth of a continued root stores the root's scoped continuation, whether
    it names its input in a `retrieve` step or only in its lineage, and both its description
    and Presentation count one series."""
    store, _ = _published_chain(continuations=continuations, second_steps=second_steps)

    for version in (_FIRST, _SECOND):
        assert _stored(store, version).continuations == stored
        record = read_record(store, subject_id=version[0], content_version=version[1])
        assert record is not None
        assert record.series.series == series
        assert _presentation_series(store, version) == series


def test_backfill_fills_the_continuations_of_versions_recorded_before_them() -> None:
    """Versions recorded without continuations, the first with a record holding its steps and
    the second with no record, are filled oldest first, and the second's backfilled record
    counts one series."""
    store, artifacts = _published_chain(continuations=[_ROOT_CONTINUATION])
    with store.transaction() as tx:
        tx.execute("UPDATE datasets.version_record SET continuations = NULL")
        tx.execute(
            "UPDATE datasets.version_record SET recorded_at = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (datetime(2026, 2, 1, tzinfo=UTC), *_FIRST),
        )
    forget_record(store, subject_id=_SECOND[0], content_version=_SECOND[1])
    unfilled = _document(store, _SECOND)

    run_backfill_descriptions(
        store=store,
        artifact_store=artifacts,
        code_index=_code_index(),
        now=lambda: datetime(2026, 10, 12, tzinfo=UTC),
    )

    assert _stored(store, _FIRST).continuations == [_CONTINUED.as_entry()]
    assert _stored(store, _SECOND).continuations == [_CONTINUED.as_entry()]
    # Presentation reads the unfilled row's continuations as unknown, the filled one's as known.
    assert (unfilled["continuations"], continuation_known(manifest=unfilled)) == (None, False)
    assert continuation_known(manifest=_document(store, _SECOND))
    assert _presentation_series(store, _SECOND) == 1
    record = read_record(store, subject_id=_SECOND[0], content_version=_SECOND[1])
    assert record is not None
    assert (record.basis, record.series.series) == ("backfill", 1)


def test_late_steps_replace_the_continuations_a_stepless_publish_stored() -> None:
    """A computed version published without steps, whose lineage names only an unpublished
    intermediate frame, stores no continuation and its record counts two series. The publish
    that fills its steps, retrieving the continued root, stores the root's continuation and
    recounts one series, as Presentation then groups the frame."""
    store = FakeVersionRecordDatasetStore()
    _record_root(store, continuations=[_ROOT_CONTINUATION])
    client = datasets_client(store)
    stepless = _body(_FIRST, inputs=_ROOT, stat_var="dcid:ops_growth", steps=False)
    stepless["input_content_versions"] = ["cv_" + "0a" * 16]

    assert publish(client, stepless).status_code == 200
    before = read_record(store, subject_id=_FIRST[0], content_version=_FIRST[1])
    filling = _body(_FIRST, inputs=_ROOT, stat_var="dcid:ops_growth")
    filling["input_content_versions"] = stepless["input_content_versions"]
    response = publish(client, filling)
    assert response.status_code == 200, response.text
    after = read_record(store, subject_id=_FIRST[0], content_version=_FIRST[1])

    assert before is not None and after is not None
    assert (before.steps, before.series.series) == (None, 2)
    assert _stored(store, _FIRST).continuations == [_CONTINUED.as_entry()]
    assert after.steps is not None
    assert after.series.series == 1
    assert _presentation_series(store, _FIRST) == 1
