"""Pins mint-to-writer calls and `LiveVersionRecordWriter` persistence against SQLite-backed stores."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from dataset_store.snapshot import (
    FrameLineage,
    LiveSnapshotReader,
    ManifestFacts,
    VersionFactsMissing,
)
from datasets_api.manifest_models import CoverageModel
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveVersionRecordWriter, rendering_key
from datasets_api.versions import (
    ArtifactRecord,
    NullVersionRecordWriter,
    VersionRecord,
    VersionWrite,
    WrittenVersion,
)
from materialise.graph import Facet
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from ._store_fault_doubles import DriverExplodingFakeDatasetStore, SqlstateExplodingFakeDatasetStore
from ._version_facts import facts_from_body, golden_cases
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR_A = "GrantA"
_STAT_VAR_B = "GrantB"
_REF_A = f"dcid:{_STAT_VAR_A}"
_REF_B = f"dcid:{_STAT_VAR_B}"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _csv_row(*, stat_var_ref: str, year: str, value: str) -> str:
    """`stat_var_ref` is written to the `variableMeasured` column bare. An
    SDMX response never carries the `dcid:` scheme prefix.
    """
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var_ref.removeprefix('dcid:')},country/KEN,USD,"
        f"GrantEquivalent,NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


# Keyed by the BARE stat_var. `_sdmx_transport`'s handler matches a key
# against the outgoing request's own `c[variableMeasured]` query value,
# which `sdmx.data` sends bare.
_CSV_BY_STAT_VAR = {
    _STAT_VAR_A: _CSV_HEADER + _csv_row(stat_var_ref=_REF_A, year="2015", value="1000"),
    _STAT_VAR_B: _CSV_HEADER + _csv_row(stat_var_ref=_REF_B, year="2017", value="2000"),
}

# Facts for records whose facts no assertion reads.
_FACTS = FrameLineage(input_content_versions=["cv_x"])

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2017",
)


class RecordingVersionRecordWriter:
    """Record version calls for mint-door assertions without persisting them."""

    def __init__(self) -> None:
        self.records: list[VersionRecord] = []
        self.trace_ids: list[str | None] = []

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> None:
        self.records.append(record)
        self.trace_ids.append(trace_id)

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> str:
        return artifact.artifact_digest

    def write_version(self, write: VersionWrite, *, trace_id: str | None = None) -> WrittenVersion:
        self.record_version(write.record, trace_id=trace_id)
        return NullVersionRecordWriter().write_version(write, trace_id=trace_id)


class RaisingVersionRecordWriter:
    """Raise `DatasetsStoreUnavailable` to test mint-door error mapping."""

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> None:
        raise DatasetsStoreUnavailable("exemplar diagnostics")

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> None:
        raise DatasetsStoreUnavailable("exemplar diagnostics")

    def write_version(self, write: VersionWrite, *, trace_id: str | None = None) -> WrittenVersion:
        raise DatasetsStoreUnavailable("exemplar diagnostics")


def _sdmx_transport():
    """Combine rows for every bare StatVar in the request under one CSV header."""

    def handler(request: httpx.Request) -> httpx.Response:
        requested_vars = str(request.url.params.get("c[variableMeasured]", "")).split(",")
        rows = "".join(
            _CSV_BY_STAT_VAR[stat_var].removeprefix(_CSV_HEADER)
            for stat_var in requested_vars
            if stat_var in _CSV_BY_STAT_VAR
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _coverage_service() -> FakeCoverageService:
    """Build coverage for both fixture StatVars with `_FACET` metadata and their CSV years."""
    index = build_coverage_index(
        [
            (_STAT_VAR_A, "facet1", ABSENT_ENTITY, "country/KEN", [2015]),
            (_STAT_VAR_B, "facet1", ABSENT_ENTITY, "country/KEN", [2017]),
        ],
        role_names={
            _STAT_VAR_A: RoleNames(donor=None, recipient="recipient"),
            _STAT_VAR_B: RoleNames(donor=None, recipient="recipient"),
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(*, version_writer):
    from dc_sdmx import SdmxClient

    store = FakeDatasetStore()
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR_A: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
                "name": ("Statistical variable A",),
            },
            _STAT_VAR_B: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
                "name": ("Statistical variable B",),
            },
            # Named so a download's default name columns fill and the render adds no
            # `enrichment_unavailable` warning beyond the version record's own.
            "country/KEN": {"name": ("Kenya",)},
            "GrantEquivalent": {"name": ("Grant equivalent",)},
        },
        facets={
            (_STAT_VAR_A, "country/KEN"): [_FACET],
            (_STAT_VAR_B, "country/KEN"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport())
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=FakeArtifactStore(),
        version_writer=version_writer,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _scope(*, statvars: list[str]) -> dict:
    return {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": statvars,
            "scope": {
                "entities": {"recipient": ["country/KEN"]},
                "facets": {
                    "provenance": {"ref": "OECD_DAC2A"},
                    "measurementMethod": {"ref": "GrantEquivalent"},
                    "unit": {"ref": "USD"},
                },
            },
        },
        "options": {"bundle": False},
    }


def _wire_coverage(coverage: Mapping[str, object]) -> dict:
    """A stored coverage fact as a wire response serialises it."""
    return CoverageModel.model_validate(coverage).model_dump(mode="json", exclude_none=True)


def test_definition_mint_records_version_exactly_once_with_no_live_artifacts_in_its_facts():
    writer = RecordingVersionRecordWriter()
    client = _client(version_writer=writer)

    resp = client.post("/api/datasets", json=_scope(statvars=[_REF_A]))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert len(writer.records) == 1
    record = writer.records[0]

    assert record.subject_kind == "spec"
    assert record.subject_id == manifest["definition"]["spec_id"]
    assert record.content_version == manifest["content_version"]
    facts = record.facts
    assert isinstance(facts, ManifestFacts)

    # Compare fields carried on the wire.
    assert facts.definition == manifest["definition"]
    # A stored dimension retains `axis: None`, while the wire response omits it.
    assert _wire_coverage(facts.coverage) == manifest["coverage"]
    assert facts.citation["text"] == manifest["citation"]["text"]
    # The response stamps how this call was served onto the stored trace.
    served = {"served_from": "materialise", "confirmed_at": manifest["trace"]["confirmed_at"]}
    assert {**facts.trace, **served} == manifest["trace"]
    assert facts.warnings == manifest["warnings"]


def test_mint_door_wire_maps_a_version_writer_store_failure_to_retryable_503():
    client = _client(version_writer=RaisingVersionRecordWriter())

    resp = client.post("/api/datasets", json=_scope(statvars=[_REF_A]))

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True


def test_a_repeat_definition_mint_still_records_version_exactly_once_per_call():
    """A mint writes its version record exactly once (`write_version`). A repeat mint of the
    same definition is a separate call, with its own single record.
    """
    writer = RecordingVersionRecordWriter()
    client = _client(version_writer=writer)
    body = _scope(statvars=[_REF_A])

    first = client.post("/api/datasets", json=body)
    second = client.post("/api/datasets", json=body)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert len(writer.records) == 2
    assert writer.records[0].content_version == writer.records[1].content_version


def _artifact_record(digest: str) -> ArtifactRecord:
    # Distinct rendering options avoid the `version_artifact` PK collision on `rendering_key`.
    return ArtifactRecord(
        artifact_digest=digest, rendering={"format": "csv", "digest": digest}, media_type="text/csv"
    )


@pytest.fixture
def version_store():
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()


def test_live_writer_persists_the_record_and_its_artifact_links(version_store):
    writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )

    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=_FACTS,
            artifacts=(_artifact_record("sha256:aaa"), _artifact_record("sha256:bbb")),
        ),
        trace_id="trace-1",
    )

    with version_store.transaction() as tx:
        record_row = tx.fetch_one(
            "SELECT * FROM datasets.version_record WHERE subject_id = %s AND content_version = %s",
            ("spec_a", "cv1"),
        )
        artifact_rows = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s ORDER BY artifact_digest",
            ("spec_a", "cv1"),
        )
    assert record_row["subject_kind"] == "spec"
    assert record_row["schema_version"] == 1
    assert record_row["trace_id"] == "trace-1"
    assert record_row["recorded_at"] == datetime(2026, 1, 1, tzinfo=UTC)
    assert [row["artifact_digest"] for row in artifact_rows] == ["sha256:aaa", "sha256:bbb"]
    assert record_row["base_content_version"] is None


@pytest.mark.parametrize("frame_summed", [True, False], ids=["collapse", "cut"])
def test_live_writer_round_trips_base_content_version_and_frame_sha256_off_their_columns(
    version_store, frame_summed
):
    """`base_content_version`, `frame_sha256` and `frame_summed` persist in their own columns
    and read back from them. The body holds no `frame_sha256`."""
    writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )

    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv_collapsed",
            schema_version=1,
            facts=_FACTS,
            base_content_version="cv_base",
            frame_sha256="a" * 64,
            frame_summed=frame_summed,
        )
    )

    with version_store.transaction() as tx:
        record_row = tx.fetch_one(
            "SELECT * FROM datasets.version_record WHERE subject_id = %s AND content_version = %s",
            ("spec_a", "cv_collapsed"),
        )
    assert record_row["base_content_version"] == "cv_base"
    assert record_row["frame_sha256"] == "a" * 64
    assert "frame_sha256" not in record_row["immutable_body"]

    stored = LiveSnapshotReader(store=version_store).read_version("spec_a", "cv_collapsed")
    assert stored is not None
    assert stored.base_content_version == "cv_base"
    assert stored.frame_sha256 == "a" * 64
    assert stored.frame_summed is frame_summed


@pytest.mark.parametrize(
    ("base_content_version", "frame_summed"),
    [("cv_base", None), (None, True)],
    ids=["derived-without-it", "raw-with-it"],
)
def test_a_version_record_holds_frame_summed_exactly_when_it_names_a_base(
    base_content_version, frame_summed
):
    with pytest.raises(ValueError, match="frame_summed"):
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=_FACTS,
            base_content_version=base_content_version,
            frame_summed=frame_summed,
        )


@pytest.mark.parametrize("name", ["definition", "collapsed", "conversions", "published"])
def test_live_writer_stores_facts_as_columns_and_the_body_they_encode(version_store, name):
    """A new version stores its facts in their own columns and, in `immutable_body`, the body
    the previous writer stored for them, and reads back the facts it stored."""
    case = golden_cases()[name]
    facts = facts_from_body(case["body"])
    LiveVersionRecordWriter(store=version_store).record_version(
        VersionRecord(
            subject_id=case["subject_id"],
            subject_kind="spec",
            content_version=case["content_version"],
            schema_version=1,
            facts=facts,
            base_content_version=case["base_content_version"],
            frame_summed=None if case["base_content_version"] is None else True,
        )
    )

    with version_store.transaction() as tx:
        record_row = tx.fetch_one(
            "SELECT immutable_body FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (case["subject_id"], case["content_version"]),
        )
    assert record_row == {"immutable_body": case["body"]}
    stored = LiveSnapshotReader(store=version_store).read_version(
        case["subject_id"], case["content_version"]
    )
    assert stored is not None
    assert stored.facts == facts


def test_a_version_row_holding_no_facts_refuses_to_read(version_store):
    """A row whose body names neither `prepared_at` nor `input_content_versions` holds no
    facts. Reading it refuses."""
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, 'spec', 'cv1', 1, %s, %s)",
            ("spec_a", {"revision": "first"}, datetime(2026, 1, 1, tzinfo=UTC)),
        )

    with pytest.raises(VersionFactsMissing) as refused:
        LiveSnapshotReader(store=version_store).read_version("spec_a", "cv1")
    assert (refused.value.subject_id, refused.value.content_version) == ("spec_a", "cv1")


def test_record_version_returns_whether_its_own_insert_created_the_row(version_store):
    """The return value distinguishes a new insert from an existing version row."""
    writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    record = VersionRecord(
        subject_id="spec_a",
        subject_kind="spec",
        content_version="cv1",
        schema_version=1,
        facts=_FACTS,
        artifacts=(_artifact_record("sha256:aaa"),),
    )

    assert writer.record_version(record) is True
    assert writer.record_version(record) is False


class _CapturingDescriber:
    """Records the facts of each version it is asked to describe, and builds no record."""

    def __init__(self) -> None:
        self.described: list[object] = []

    def describe(self, version: VersionRecord, *, base: object) -> None:
        self.described.append(version.facts)

    def settle(self, stored: object, *, version: VersionRecord) -> None:
        return None


def test_live_writer_does_not_rewrite_an_already_recorded_versions_facts(version_store):
    """A repeated version insert with other facts returns `False`, preserves the stored
    timestamp and facts, and adds new artifact links. Its describer describes the stored facts,
    so a writer that lost the race records the winner's."""
    writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=FrameLineage(input_content_versions=["cv_first"]),
            artifacts=(_artifact_record("sha256:aaa"),),
        )
    )

    later_writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 6, 1, tzinfo=UTC)
    )
    describer = _CapturingDescriber()
    recorded = later_writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=FrameLineage(input_content_versions=["cv_second"]),
            artifacts=(_artifact_record("sha256:aaa"), _artifact_record("sha256:ccc")),
            describer=describer,
        )
    )

    with version_store.transaction() as tx:
        record_row = tx.fetch_one(
            "SELECT * FROM datasets.version_record WHERE subject_id = %s AND content_version = %s",
            ("spec_a", "cv1"),
        )
        artifact_rows = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s ORDER BY artifact_digest",
            ("spec_a", "cv1"),
        )
    assert recorded is False
    assert record_row["recorded_at"] == datetime(2026, 1, 1, tzinfo=UTC)
    assert record_row["input_content_versions"] == ["cv_first"]
    assert [row["artifact_digest"] for row in artifact_rows] == ["sha256:aaa", "sha256:ccc"]
    assert describer.described == [FrameLineage(input_content_versions=["cv_first"])]


def test_live_writer_defaults_its_clock_to_the_real_utc_now(version_store):
    before = datetime.now(UTC)
    writer = LiveVersionRecordWriter(store=version_store)

    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=_FACTS,
        )
    )
    after = datetime.now(UTC)

    with version_store.transaction() as tx:
        record_row = tx.fetch_one(
            "SELECT recorded_at FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            ("spec_a", "cv1"),
        )
    recorded_at = record_row["recorded_at"]
    assert before <= recorded_at <= after


def test_live_writer_persists_a_definition_mint_end_to_end_through_the_real_door(version_store):
    """The mint door records both response artifacts and its internal canonical frame."""
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(version_writer=writer)

    resp = client.post("/api/datasets", json=_scope(statvars=[_REF_A]))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    with version_store.transaction() as tx:
        record_row = tx.fetch_one(
            "SELECT * FROM datasets.version_record WHERE subject_id = %s AND content_version = %s",
            (manifest["definition"]["spec_id"], manifest["content_version"]),
        )
        artifact_rows = tx.fetch_all(
            "SELECT artifact_digest, rendering FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (manifest["definition"]["spec_id"], manifest["content_version"]),
        )
    assert record_row is not None
    response_digests = {artifact["artifact_digest"] for artifact in manifest["artifacts"]}
    recorded_digests = {row["artifact_digest"] for row in artifact_rows}
    assert response_digests <= recorded_digests
    assert recorded_digests - response_digests
    assert any(row["rendering"] == {"format": "frame"} for row in artifact_rows)


def test_identical_rendering_options_resubmitted_stay_a_single_idempotent_row(version_store):
    """Identical rendering options share one idempotent artifact link."""
    writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    record = VersionRecord(
        subject_id="spec_a",
        subject_kind="spec",
        content_version="cv1",
        schema_version=1,
        facts=_FACTS,
        artifacts=(_artifact_record("sha256:aaa"),),
    )

    writer.record_version(record)
    writer.record_version(record)

    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            ("spec_a", "cv1"),
        )
    assert [row["artifact_digest"] for row in rows] == ["sha256:aaa"]


def test_two_rendering_option_sets_sharing_one_digest_still_yield_two_rows(version_store):
    """One byte digest can have two links when the rendering options differ."""
    writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=_FACTS,
            artifacts=(
                ArtifactRecord(
                    artifact_digest="sha256:shared",
                    rendering={"format": "csv"},
                    media_type="text/csv",
                ),
                ArtifactRecord(
                    artifact_digest="sha256:shared",
                    rendering={"format": "csv", "bundle": True},
                    media_type="text/csv",
                ),
            ),
        )
    )

    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest, rendering_key FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            ("spec_a", "cv1"),
        )
    assert [row["artifact_digest"] for row in rows] == ["sha256:shared", "sha256:shared"]
    assert len({row["rendering_key"] for row in rows}) == 2


def test_record_artifact_adds_a_rendering_to_an_already_recorded_version(version_store):
    """`record_artifact` adds a rendering idempotently to an existing version."""
    writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=_FACTS,
            artifacts=(_artifact_record("sha256:aaa"),),
        )
    )
    second_artifact = _artifact_record("sha256:bbb")

    writer.record_artifact(subject_id="spec_a", content_version="cv1", artifact=second_artifact)
    writer.record_artifact(subject_id="spec_a", content_version="cv1", artifact=second_artifact)

    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s ORDER BY artifact_digest",
            ("spec_a", "cv1"),
        )
    assert [row["artifact_digest"] for row in rows] == ["sha256:aaa", "sha256:bbb"]


_LINK_NOW = datetime(2026, 3, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("tier", "stored_at", "linked"),
    [
        pytest.param(
            "working", _LINK_NOW - timedelta(days=30), "sha256:old", id="servable-working"
        ),
        pytest.param("working", _LINK_NOW - timedelta(days=31), "sha256:new", id="expired-working"),
        pytest.param("pinned", _LINK_NOW - timedelta(days=400), "sha256:old", id="pinned"),
        pytest.param(None, None, "sha256:new", id="no-location"),
    ],
)
def test_a_rendering_key_keeps_its_link_only_while_the_linked_digest_is_servable(
    version_store, tier, stored_at, linked
):
    """A new digest under a linked rendering key replaces the link once the linked digest has
    no servable location, as `select_servable_locations` judges one at the writer's `now`."""
    writer = LiveVersionRecordWriter(store=version_store, now=lambda: _LINK_NOW)
    rendering = {"format": "csv", "bundle": True}
    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=_FACTS,
            artifacts=(
                ArtifactRecord(
                    artifact_digest="sha256:old", rendering=rendering, media_type="application/zip"
                ),
            ),
        )
    )
    if tier is not None:
        with version_store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.artifact_location "
                "(artifact_digest, tier, bucket, object_name, stored_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                ("sha256:old", tier, "bucket", "sha256:old", stored_at),
            )

    writer.record_artifact(
        subject_id="spec_a",
        content_version="cv1",
        artifact=ArtifactRecord(
            artifact_digest="sha256:new", rendering=rendering, media_type="application/zip"
        ),
    )

    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            ("spec_a", "cv1"),
        )
    assert [row["artifact_digest"] for row in rows] == [linked]


def test_record_artifact_against_a_failing_store_raises_datasets_store_unavailable():
    store = SqlstateExplodingFakeDatasetStore(sqlstate="23503")
    writer = LiveVersionRecordWriter(store=store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        writer.record_artifact(
            subject_id="spec_a", content_version="cv1", artifact=_artifact_record("sha256:aaa")
        )

    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert exc.sqlstate == "23503"


def test_rendering_key_matches_the_key_record_version_stored(version_store):
    writer = LiveVersionRecordWriter(
        store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            facts=_FACTS,
            artifacts=(
                ArtifactRecord(
                    artifact_digest="sha256:aaa",
                    rendering={"format": "csv", "bundle": False},
                    media_type="text/csv",
                ),
            ),
        )
    )

    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT rendering_key FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            ("spec_a", "cv1"),
        )
    assert rendering_key({"format": "csv", "bundle": False}) == row["rendering_key"]


def test_a_driver_failure_raises_datasets_store_unavailable_with_no_leaked_context():
    """The wrapped store error has no chained cause and exposes no driver message or bound data."""
    distinctive_driver_text = "distinctive_driver_boom_version_write_f7a1"
    store = DriverExplodingFakeDatasetStore(message=distinctive_driver_text)
    writer = LiveVersionRecordWriter(store=store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        writer.record_version(
            VersionRecord(
                subject_id="spec_a",
                subject_kind="spec",
                content_version="cv1",
                schema_version=1,
                facts=FrameLineage(input_content_versions=["should-never-leak"]),
            )
        )

    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert distinctive_driver_text not in str(exc)
    assert "should-never-leak" not in str(exc)


def test_a_driver_failure_carrying_a_sqlstate_attaches_it_to_the_raised_exception():
    """The pg8000 error field `args[0]["C"]` populates `DatasetsStoreUnavailable.sqlstate`."""
    store = SqlstateExplodingFakeDatasetStore(sqlstate="23503")
    writer = LiveVersionRecordWriter(store=store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        writer.record_version(
            VersionRecord(
                subject_id="spec_a",
                subject_kind="spec",
                content_version="cv1",
                schema_version=1,
                facts=_FACTS,
            )
        )

    assert excinfo.value.sqlstate == "23503"
