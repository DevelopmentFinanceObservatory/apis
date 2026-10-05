"""Direct tests for `_record_member_versions` using prepared
`MaterialisationResult`s, including per-member frame1 and version writes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from artifact_store import Tier
from code_index.shared import get_index
from datasets_api.manifest import decode_default_scope_plural
from datasets_api.manifest_models import WarningModel
from datasets_api.serve import (
    FRAME1_MAX_ROWS,
    MEMBER_VERSION_RECORD_FAILED_MARKER,
    ServeDeps,
    _record_member_versions,
)
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import PhaseTimings, collecting
from datasets_api.versions import ArtifactRecord, NullHeadWriter, VersionRecord
from materialise import EnrichmentLookup, MaterialisationResult
from materialise.content_version import MintPreimage
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.frame_build import build_frame
from materialise.profile import compute_output_profile
from materialise.rate_tables import RateTableCache
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace
from materialise.window import Coverage
from spec_registry.reader import StoredDefinition

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader

_PREPARED_AT = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _statistics_file(frame):
    sink = SeriesStatisticsSink()
    sink.write(compute_series_statistics_table(frame))
    return sink.finish()


def _member(
    *, spec_id: str, stat_var_ref: str, content_version: str
) -> tuple[StoredDefinition, MaterialisationResult]:
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": stat_var_ref,
                "recipient": "country/KEN",
                "unit": "USD",
                "measurementMethod": "GrantEquivalent",
                "provenance": "OECD_DAC2A",
                "time_period": "2015",
                "value": 1000.0,
            }
        ],
    )
    stored = StoredDefinition(
        spec_id=spec_id,
        schema_version=1,
        identity={"kind": "spec_id", "spec_id": spec_id},
        identity_bytes=b"{}",
        identity_digest="sha256:fixture",
    )
    graph = FakeMintGraphReader(labels={"OECD_DAC2A": "OECD DAC2a"})
    result = MaterialisationResult(
        frame_chunks=lambda: iter((frame,)),
        content_version=content_version,
        trace=ManifestTrace(
            applied_window=None,
            graph_revision="rev1",
            source_revisions=(),
            resolved_stat_vars=(),
            facets=(
                {
                    "axis": "provenance",
                    "ref": "OECD_DAC2A",
                    "basis": "bound",
                    "default_policy": None,
                    "default_policy_version": None,
                    "selection_reason": None,
                    "dispatch_facet_id": "facet1",
                },
            ),
            normalisations=(),
            conversion=None,
            frame_grammar_version="1.0.0",
            materialiser_build="test-build",
            capability_version="dset_test",
        ),
        coverage=Coverage(rows=len(frame.rows), dimensions=(), window=None),
        warnings=(),
        series_statistics=_statistics_file(frame),
        output_profile=compute_output_profile(
            frame, role_columns=("recipient",), enrichment=EnrichmentLookup(graph)
        ),
        conversion_record=None,
        member_conversion_records={stat_var_ref: None},
        mint_preimage=MintPreimage(
            graph_revision="rev1",
            source_revisions=(),
            facets=({"axis": "provenance", "ref": "OECD_DAC2A", "basis": "bound"},),
        ),
    )
    return stored, result


class _RecordingVersionRecordWriter:
    """Lock-guarded: `_record_member_versions` calls `record_version` from
    up to `member_concurrency` worker threads at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[VersionRecord] = []

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> None:
        with self._lock:
            self.records.append(record)

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> None:
        return None


class _FailingNthMemberVersionRecordWriter:
    """Raises `DatasetsStoreUnavailable` on the `fail_at` call
    (0-indexed, in call order) and records every other call."""

    def __init__(self, *, fail_at: int) -> None:
        self._lock = threading.Lock()
        self._fail_at = fail_at
        self._calls = 0
        self.records: list[VersionRecord] = []

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> None:
        with self._lock:
            call_index = self._calls
            self._calls += 1
        if call_index == self._fail_at:
            raise DatasetsStoreUnavailable("simulated store outage")
        with self._lock:
            self.records.append(record)

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> None:
        return None


def _deps(
    *, artifact_store, version_writer, member_concurrency: int = 4, deadline=None
) -> ServeDeps:
    from datasets_api.revision import SentinelRevisionReader
    from dc_sdmx import SdmxClient

    return ServeDeps(
        sdmx=SdmxClient(base_url="https://example.invalid"),
        graph=FakeMintGraphReader(),
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store,
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=member_concurrency,
        enrichment=EnrichmentLookup(FakeMintGraphReader()),
        code_index=get_index(),
        rate_cache=RateTableCache(),
        version_writer=version_writer,
        head_store=NullHeadWriter(),
        deadline=deadline,
    )


def test_records_one_version_and_one_frame1_per_member():
    stored_b, result_b = _member(
        spec_id="spec_b", stat_var_ref="dcid:GrantB", content_version="cv_b"
    )
    per_member = [
        _member(spec_id="spec_a", stat_var_ref="dcid:GrantA", content_version="cv_a"),
        (stored_b, replace(result_b, default_scope=("country/KEN", "country/GHA"))),
    ]
    writer = _RecordingVersionRecordWriter()
    artifact_store = FakeArtifactStore()
    deps = _deps(artifact_store=artifact_store, version_writer=writer)

    _record_member_versions(
        collection_id="collection_x",
        per_member=per_member,
        member_origin_warnings=[()] * len(per_member),
        member_prepared_at=[_PREPARED_AT] * len(per_member),
        member_on_record=[False] * len(per_member),
        source_citations={},
        deps=deps,
    )

    assert len(writer.records) == 2
    by_subject = {record.subject_id: record for record in writer.records}
    assert set(by_subject) == {"spec_a", "spec_b"}

    for spec_id, content_version in (("spec_a", "cv_a"), ("spec_b", "cv_b")):
        record = by_subject[spec_id]
        assert record.subject_kind == "spec"
        assert record.content_version == content_version
        assert record.member_versions == ()
        assert len(record.artifacts) == 1
        assert record.artifacts[0].media_type == FRAME1_MEDIA_TYPE
        assert "artifacts" not in record.immutable_body
        assert record.immutable_body["definition"]["spec_id"] == spec_id
        assert record.immutable_body["content_version"] == content_version
    # Each member's body stores whether its own default place scope was plural.
    assert [
        decode_default_scope_plural(by_subject[spec_id].immutable_body)
        for spec_id in ("spec_a", "spec_b")
    ] == [False, True]


def test_each_member_record_carries_its_own_origin_warnings():
    per_member = [
        _member(spec_id="spec_a", stat_var_ref="dcid:GrantA", content_version="cv_a"),
        _member(spec_id="spec_b", stat_var_ref="dcid:GrantB", content_version="cv_b"),
    ]
    withheld = WarningModel(
        code="withheld_by_registry", severity="warn", message="GrantB withheld at KEN"
    )
    writer = _RecordingVersionRecordWriter()
    deps = _deps(artifact_store=FakeArtifactStore(), version_writer=writer)

    _record_member_versions(
        collection_id="collection_x",
        per_member=per_member,
        member_origin_warnings=[(), (withheld,)],
        member_prepared_at=[_PREPARED_AT] * len(per_member),
        member_on_record=[False] * len(per_member),
        source_citations={},
        deps=deps,
    )

    codes = {
        record.subject_id: [w["code"] for w in record.immutable_body["warnings"]]
        for record in writer.records
    }
    assert "withheld_by_registry" in codes["spec_b"]
    assert "withheld_by_registry" not in codes["spec_a"]


def test_no_frame1_above_the_d8_row_gate_for_a_collection_member():
    """Only members at or below `FRAME1_MAX_ROWS` get frame1. The fixture puts one member
    on each side of the limit."""
    spec_a = _member(spec_id="spec_a", stat_var_ref="dcid:GrantA", content_version="cv_a")
    spec_a = (
        spec_a[0],
        replace(spec_a[1], coverage=replace(spec_a[1].coverage, rows=FRAME1_MAX_ROWS + 1)),
    )
    per_member = [
        spec_a,
        _member(spec_id="spec_b", stat_var_ref="dcid:GrantB", content_version="cv_b"),
    ]
    writer = _RecordingVersionRecordWriter()
    artifact_store = FakeArtifactStore()
    deps = _deps(artifact_store=artifact_store, version_writer=writer)

    _record_member_versions(
        collection_id="collection_x",
        per_member=per_member,
        member_origin_warnings=[()] * len(per_member),
        member_prepared_at=[_PREPARED_AT] * len(per_member),
        member_on_record=[False] * len(per_member),
        source_citations={},
        deps=deps,
    )

    by_subject = {record.subject_id: record for record in writer.records}
    assert by_subject["spec_a"].artifacts == ()
    assert len(by_subject["spec_b"].artifacts) == 1
    assert by_subject["spec_b"].artifacts[0].media_type == FRAME1_MEDIA_TYPE


def test_two_members_frame1_writes_are_content_addressed_and_distinct():
    per_member = [
        _member(spec_id="spec_a", stat_var_ref="dcid:GrantA", content_version="cv_a"),
        _member(spec_id="spec_b", stat_var_ref="dcid:GrantB", content_version="cv_b"),
    ]
    writer = _RecordingVersionRecordWriter()
    artifact_store = FakeArtifactStore()
    deps = _deps(artifact_store=artifact_store, version_writer=writer)

    _record_member_versions(
        collection_id="collection_x",
        per_member=per_member,
        member_origin_warnings=[()] * len(per_member),
        member_prepared_at=[_PREPARED_AT] * len(per_member),
        member_on_record=[False] * len(per_member),
        source_citations={},
        deps=deps,
    )

    digests = {record.artifacts[0].artifact_digest for record in writer.records}
    # Distinct member StatVars produce distinct frame1 digests.
    assert len(digests) == 2


def _members(count: int) -> list[tuple[StoredDefinition, MaterialisationResult]]:
    return [
        _member(spec_id=f"spec_{i}", stat_var_ref=f"dcid:Grant{i}", content_version=f"cv_{i}")
        for i in range(count)
    ]


def test_member_concurrency_dispatches_every_member_at_once_barrier_proof():
    """A four-party barrier inside `put()` proves four member writes run concurrently."""
    per_member = _members(8)
    barrier = threading.Barrier(4)

    class _BarrierArtifactStore(FakeArtifactStore):
        def put(self, data: bytes, *, tier: Tier):
            barrier.wait(timeout=5.0)
            return super().put(data, tier=tier)

    writer = _RecordingVersionRecordWriter()
    deps = _deps(
        artifact_store=_BarrierArtifactStore(), version_writer=writer, member_concurrency=4
    )

    _record_member_versions(
        collection_id="collection_x",
        per_member=per_member,
        member_origin_warnings=[()] * len(per_member),
        member_prepared_at=[_PREPARED_AT] * len(per_member),
        member_on_record=[False] * len(per_member),
        source_citations={},
        deps=deps,
    )

    assert len(writer.records) == 8


def test_member_concurrency_one_times_out_a_two_party_barrier():
    """With one worker, a two-party barrier in `put()` times out."""
    per_member = _members(2)
    barrier = threading.Barrier(2)

    class _BarrierArtifactStore(FakeArtifactStore):
        def put(self, data: bytes, *, tier: Tier):
            barrier.wait(timeout=0.2)
            return super().put(data, tier=tier)

    writer = _RecordingVersionRecordWriter()
    deps = _deps(
        artifact_store=_BarrierArtifactStore(), version_writer=writer, member_concurrency=1
    )

    with pytest.raises(threading.BrokenBarrierError):
        _record_member_versions(
            collection_id="collection_x",
            per_member=per_member,
            member_origin_warnings=[()] * len(per_member),
            member_prepared_at=[_PREPARED_AT] * len(per_member),
            member_on_record=[False] * len(per_member),
            source_citations={},
            deps=deps,
        )


def test_every_member_version_record_lands_in_any_order_each_with_own_frame1():
    """Eight concurrent writes may finish in any order. Each record must keep its member's
    distinct frame1 digest."""
    per_member = _members(8)
    writer = _RecordingVersionRecordWriter()
    artifact_store = FakeArtifactStore()
    deps = _deps(artifact_store=artifact_store, version_writer=writer, member_concurrency=4)

    _record_member_versions(
        collection_id="collection_x",
        per_member=per_member,
        member_origin_warnings=[()] * len(per_member),
        member_prepared_at=[_PREPARED_AT] * len(per_member),
        member_on_record=[False] * len(per_member),
        source_citations={},
        deps=deps,
    )

    assert len(writer.records) == 8
    by_subject = {record.subject_id: record for record in writer.records}
    assert set(by_subject) == {f"spec_{i}" for i in range(8)}
    digests_by_subject = {
        subject_id: record.artifacts[0].artifact_digest for subject_id, record in by_subject.items()
    }
    assert len(set(digests_by_subject.values())) == 8


def test_one_of_five_members_version_record_failure_is_best_effort(
    caplog: pytest.LogCaptureFixture,
):
    """One failed version write leaves four records and all five frame1 artifacts. The failure
    marker is logged once."""
    per_member = _members(5)
    writer = _FailingNthMemberVersionRecordWriter(fail_at=1)
    artifact_store = FakeArtifactStore()
    deps = _deps(artifact_store=artifact_store, version_writer=writer, member_concurrency=4)

    with caplog.at_level(logging.ERROR, logger="datasets_api.serve"):
        _record_member_versions(
            collection_id="collection_x",
            per_member=per_member,
            member_origin_warnings=[()] * len(per_member),
            member_prepared_at=[_PREPARED_AT] * len(per_member),
            member_on_record=[False] * len(per_member),
            source_citations={},
            deps=deps,
        )

    assert len(writer.records) == 4
    assert len(artifact_store.list(tier=Tier.working)) == 5
    marker_records = [
        record for record in caplog.records if MEMBER_VERSION_RECORD_FAILED_MARKER in record.message
    ]
    assert len(marker_records) == 1


def test_member_writes_are_recorded_under_the_record_phase():
    """Worker writes inherit the caller's timing context, so member and collection writes share
    the `record` phase name."""
    per_member = _members(3)
    writer = _RecordingVersionRecordWriter()
    artifact_store = FakeArtifactStore()
    deps = _deps(artifact_store=artifact_store, version_writer=writer, member_concurrency=2)

    timings = PhaseTimings()
    with collecting(timings):
        _record_member_versions(
            collection_id="collection_x",
            per_member=per_member,
            member_origin_warnings=[()] * len(per_member),
            member_prepared_at=[_PREPARED_AT] * len(per_member),
            member_on_record=[False] * len(per_member),
            source_citations={},
            deps=deps,
        )

    snapshot = timings.snapshot()
    assert "record" in snapshot
    assert snapshot["record"] > 0.0
    assert "member_record" not in snapshot


def test_past_deadline_still_persists_every_member_and_logs_once(
    caplog: pytest.LogCaptureFixture,
):
    """A past deadline permits every member write and emits one warning with the
    collection id and member count."""
    per_member = _members(4)
    writer = _RecordingVersionRecordWriter()
    artifact_store = FakeArtifactStore()
    deadline = time.monotonic() - 5.0
    deps = _deps(
        artifact_store=artifact_store,
        version_writer=writer,
        member_concurrency=2,
        deadline=deadline,
    )

    with caplog.at_level(logging.WARNING, logger="datasets_api.serve"):
        _record_member_versions(
            collection_id="collection_x",
            per_member=per_member,
            member_origin_warnings=[()] * len(per_member),
            member_prepared_at=[_PREPARED_AT] * len(per_member),
            member_on_record=[False] * len(per_member),
            source_citations={},
            deps=deps,
        )

    assert len(writer.records) == 4
    past_deadline_records = [
        record
        for record in caplog.records
        if "collection_id=collection_x" in record.message and "member_count=4" in record.message
    ]
    assert len(past_deadline_records) == 1
