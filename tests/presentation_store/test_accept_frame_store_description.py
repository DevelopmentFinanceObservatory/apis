"""Acceptance cases for the frame store reading a version's description record (case 5).

`LiveFrameStore` takes a `description_reader`, injected as `recipe_reader` is. Its type is
`presentation_store.frame_store.DescriptionReader`, a protocol over
`read_description(*, id, content_version)` and `read_descriptions(*, versions)` that `datasets_api.description_store`'s
`LiveDescriptionStore` satisfies. When it loads a frame it reads that version's record once and caches it
with the frame, as `ResolvedFrame.description_record`. A version whose record is absent loads
with none and is read again on a load past the recheck wait, without fetching the frame's
bytes again. So is a computed version's record holding `steps: null`, which a later publish
fills.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence

import pytest

from artifact_store.store import Tier
from assembler.ports import FrameRef
from description.facts import (
    Builds,
    ComputedFacts,
    DescriptionRecord,
    Evidence,
    InputVersion,
    SeriesFacts,
    StatusCounts,
    VariableFacts,
)

from .test_frame_store import (
    _COMPOSED_ROWS,
    _QUOTIENT_ENTRY,
    _QUOTIENT_LABEL,
    _STEPS_WITHOUT_STAT_VAR,
    _SUM_ENTRY,
    _Clock,
    _computed_frame_store,
    _happy_path_fixture,
    _ops_happy_path_fixture,
    _store,
)


def _record(frame_ref: FrameRef) -> DescriptionRecord:
    """The record Datasets wrote for the version: a dataset version's variables, or what a
    computed version was computed from."""
    computed = frame_ref.spec_id.startswith("ops_")
    return DescriptionRecord(
        id=frame_ref.spec_id,
        content_version=frame_ref.content_version,
        schema_version=1,
        basis="mint",
        index_stamp=None,
        variables=None
        if computed
        else VariableFacts(
            shape=[],
            variable=None,
            fixed=[],
            dimensions=None,
            admitted_at_mint=1,
            admitted_from="resolved_variables",
            with_data=1,
        ),
        computed=ComputedFacts(
            inputs=[
                InputVersion(
                    id="spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
                    content_version="cv_f4a81c22d9075e3ba61c2ee98f83b405",
                )
            ],
            composed=None,
        )
        if computed
        else None,
        series=SeriesFacts(
            series=1,
            rows=9,
            periods=9,
            span=None,
            grain="year",
            latest_period="2024",
            series_reaching_latest=1,
            entities_by_role=None,
            by_source_unit=None,
            status=StatusCounts(estimated=None, provisional=None),
        ),
        steps=[],
        caveats=[],
        evidence=Evidence(
            prepared=datetime.date(2026, 9, 28),
            title=None,
            sources=[],
            source_release=None,
            labels=[],
            membership=None,
            builds=Builds(
                origin_catalogue=None, code_index=None, coverage=None, method_status=None
            ),
        ),
        late_fields=[],
    )


class _FakeDescriptionReader:
    """Holds records by version and counts single-version reads."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], DescriptionRecord] = {}
        self.reads = 0

    def put(self, record: DescriptionRecord) -> None:
        self.records[(record.id, record.content_version)] = record

    def read_description(self, *, id: str, content_version: str) -> DescriptionRecord | None:
        self.reads += 1
        return self.records.get((id, content_version))

    def read_descriptions(
        self, *, versions: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], DescriptionRecord]:
        return {version: self.records[version] for version in versions if version in self.records}


def _put_with_inputs(reader: _FakeDescriptionReader, frame_ref: FrameRef) -> None:
    """Holds the version's record and a record for each version it was computed from, so the
    load's answer is final."""
    record = _record(frame_ref)
    reader.put(record)
    for one in record.computed.inputs if record.computed is not None else []:
        assert one.id is not None
        reader.put(_record(FrameRef(spec_id=one.id, content_version=one.content_version)))


def _reader() -> _FakeDescriptionReader:
    """A reader that satisfies the description store's reader protocol."""
    from presentation_store.frame_store import DescriptionReader

    reader: DescriptionReader = _FakeDescriptionReader()
    return reader


_FIXTURES = [
    pytest.param(_happy_path_fixture, id="dataset-version"),
    pytest.param(_ops_happy_path_fixture, id="computed-version"),
]


@pytest.mark.parametrize("fixture", _FIXTURES)
def test_load_attaches_the_versions_record_and_caches_it_with_the_frame(fixture):
    snapshots, artifacts, frame_ref, digest = fixture()
    reader = _reader()
    _put_with_inputs(reader, frame_ref)
    store = _store(snapshots=snapshots, artifacts=artifacts, description_reader=reader)

    first = store.load(frame_ref=frame_ref)
    second = store.load(frame_ref=frame_ref)

    assert first.description_record == _record(frame_ref)
    assert second is first
    assert reader.reads == 1
    assert artifacts.calls.count(("get", Tier.working, digest)) == 1


@pytest.mark.parametrize("fixture", _FIXTURES)
def test_an_absent_record_is_read_again_past_the_recheck_wait_without_refetching_the_frame(
    fixture,
):
    """The record arrives after the first load, as one written late or by the backfill does.
    A load within the recheck wait serves the cached frame without reading the record."""
    snapshots, artifacts, frame_ref, digest = fixture()
    reader = _reader()
    clock = _Clock()
    store = _store(snapshots=snapshots, artifacts=artifacts, description_reader=reader, now=clock)

    before = store.load(frame_ref=frame_ref)
    _put_with_inputs(reader, frame_ref)
    within = store.load(frame_ref=frame_ref)
    reads_within = reader.reads
    clock.pass_recheck()
    after = store.load(frame_ref=frame_ref)
    cached = store.load(frame_ref=frame_ref)

    assert within is before
    assert reads_within == 1
    assert (before.description_record, after.description_record) == (None, _record(frame_ref))
    assert cached is after
    assert reader.reads == 2
    assert artifacts.calls.count(("get", Tier.working, digest)) == 1


def test_a_store_with_no_description_reader_loads_frames_without_a_record():
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()

    resolved = _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    assert resolved.description_record is None


def test_a_computed_record_without_steps_is_read_again_until_its_steps_arrive():
    """A publish without steps stores `steps: null`, and a later publish fills them
    (`PublishDescriber.settle`). The variable a step names takes its name on the load past the
    recheck wait that reads the filled record, which is then cached."""
    rows: list[list[object]] = [["dcid:ops_per_income", "country/ETH", "2015", 1.0, None, None]]
    steps: list[dict[str, object]] = [
        *_STEPS_WITHOUT_STAT_VAR[:2],
        {**_STEPS_WITHOUT_STAT_VAR[2], "stat_var": "ops_per_income"},
    ]
    clock = _Clock()
    store, frame_ref, _, reader = _computed_frame_store(rows=rows, steps=steps, clock=clock)
    settled = reader.record
    assert settled is not None
    reader.record = settled.model_copy(update={"steps": None})

    first = store.load(frame_ref=frame_ref)
    reader.record = settled
    clock.pass_recheck()
    second = store.load(frame_ref=frame_ref)
    reader.record = None
    third = store.load(frame_ref=frame_ref)

    assert "dcid:ops_per_income" not in first.labels
    assert second.description_record == settled
    assert second.labels["dcid:ops_per_income"] == (
        "Gross ODA disbursements per gross national income"
    )
    assert third is second


def test_composed_entries_of_a_record_arriving_after_the_first_load_are_named_from_their_sides():
    """A legacy computed record, with composed entries and `steps: null`, that arrives after
    the frame was cached reads the labels of the sides the frame does not label before it names
    the variables. A failed side read caches the new record over the labels the frame held,
    and neither the record nor the labels are read again until the recheck wait passes."""
    clock = _Clock()
    store, frame_ref, graph, reader = _computed_frame_store(
        rows=_COMPOSED_ROWS, composed=[_SUM_ENTRY, _QUOTIENT_ENTRY], clock=clock
    )
    record, reader.record = reader.record, None
    store.load(frame_ref=frame_ref)
    reader.record = record
    record_reads: list[str] = []
    read_description = reader.read_description

    def counted(*, id: str, content_version: str) -> DescriptionRecord | None:
        record_reads.append(id)
        return read_description(id=id, content_version=content_version)

    reader.read_description = counted
    clock.pass_recheck()
    graph.raise_for = frozenset({"Count_Person"})

    failed = store.load(frame_ref=frame_ref)
    graph.raise_for = frozenset()
    calls = graph.calls
    within = store.load(frame_ref=frame_ref)
    reads_within = (len(record_reads), graph.calls - calls)
    clock.pass_recheck()
    named = store.load(frame_ref=frame_ref)

    assert failed.description_record == record
    assert failed.labels["country/ETH"] == "Ethiopia"
    assert failed.labels["dcid:ops_quotient"] == "Education plus Count_Person per Count_Person"
    assert within is failed
    assert reads_within == (1, 0)
    assert named.labels["dcid:ops_quotient"] == _QUOTIENT_LABEL
    assert named.labels["dcid:ops_sum"] == "Education plus Total population"
