"""Tests manifest `series_count` omission, zero, and round-trip behaviour.

Also checks that the series-statistics Parquet artifact appears on the wire; its bytes are covered
by `tests/materialise/test_series_statistics.py`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from artifact_store import Tier
from code_index.shared import get_index
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.snapshot import build_snapshot_door, read_snapshot_manifest
from datasets_api.manifest import assemble_manifest, definition_version_record
from datasets_api.manifest_models import ArtifactEntryModel
from datasets_api.render import render_and_store_series_statistics
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord
from materialise import EnrichmentLookup, MaterialisationResult
from materialise.dependencies import Dependencies
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.frame_build import build_frame
from materialise.profile import compute_output_profile
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace
from materialise.window import Coverage
from spec_registry.reader import StoredDefinition

from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import (
    _NOW,
    FakeSnapshotReader,
    FakeSpecReader,
    _store_and_locate,
    _version,
    make_client_for_door,
    version_body,
)
from ._version_facts import facts_from_body

_PREPARED_AT = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _statistics_fixture_frame():
    return build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:GrantA",
                "recipient": "country/KEN",
                "unit": "USD",
                "measurementMethod": "GrantEquivalent",
                "provenance": "OECD_DAC2A",
                "time_period": "2015",
                "value": 1000.0,
            },
            {
                "stat_var": "dcid:GrantA",
                "recipient": "country/KEN",
                "unit": "USD",
                "measurementMethod": "GrantEquivalent",
                "provenance": "OECD_DAC2A",
                "time_period": "2016",
                "value": 1500.0,
            },
        ],
    )


def _statistics_file(frame):
    sink = SeriesStatisticsSink(continuing=())
    sink.write(compute_series_statistics_table(frame))
    return sink.finish()


def _statistics_fixture_result(frame) -> MaterialisationResult:
    return MaterialisationResult(
        frame_chunks=lambda: iter((frame,)),
        frame_sha256="0" * 64,
        content_version="cv1",
        trace=ManifestTrace(
            applied_window=None,
            graph_revision="rev1",
            source_revisions=(),
            resolved_stat_vars=(),
            facets=(),
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
            frame,
            role_columns=("recipient",),
            enrichment=EnrichmentLookup(FakeMintGraphReader(), parties=lambda: {}),
        ),
        conversion_record=None,
        member_conversion_records={"dcid:GrantA": None},
        dependencies=Dependencies(
            members="sha256:fixture", sources=(), conversion_sources=(), recipes=(), window=None
        ),
    )


def test_a_stored_body_without_series_count_validates_and_the_wire_response_omits_the_key():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pre-series-count-body")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},  # no "series_count"/"statistics" key at all
        locations={("spec_a", "cv1"): [location]},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert manifest.series_count is None

    door = build_snapshot_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    assert "series_count" not in resp.json()


def test_an_old_records_inline_statistics_key_is_dropped_not_translated():
    """A legacy inline `statistics` key is dropped; `series_count` stays unset."""
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"inline-statistics-body")
    reader = FakeSnapshotReader(
        versions={
            ("spec_a", "cv1"): _version(
                facts=facts_from_body(
                    {**version_body(), "statistics": [{"stat_var": "dcid:GrantA"}]}
                )
            )
        },
        locations={("spec_a", "cv1"): [location]},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert manifest.series_count is None


def test_a_zero_series_count_is_present_not_omitted_on_the_wire():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"zero-series-body")
    reader = FakeSnapshotReader(
        versions={
            ("spec_a", "cv1"): _version(
                facts=facts_from_body({**version_body(), "series_count": 0})
            )
        },
        locations={("spec_a", "cv1"): [location]},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert manifest.series_count == 0

    door = build_snapshot_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    assert resp.json()["series_count"] == 0


def test_minted_series_count_and_statistics_artifact_survive_a_store_and_read_round_trip(
    version_store,
):
    """Live storage preserves the count and statistics artifact."""
    artifact_store = FakeArtifactStore()
    stored = artifact_store.put(b"hello", tier=Tier.working)

    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored_definition = StoredDefinition(
        spec_id="spec_a",
        schema_version=1,
        identity={"identity_version": "spec1"},
        identity_bytes=b"{}",
        identity_digest="sha256:fixture",
    )
    statistics_artifact = render_and_store_series_statistics(
        data=result.series_statistics.data,
        subject_id="spec_a",
        content_version="cv1",
        artifact_store=artifact_store,
    )
    artifacts = [
        ArtifactEntryModel(
            artifact_digest=stored.digest,
            rendering={"format": "csv"},
            bytes=stored.size_bytes,
            media_type="text/csv",
            created_at=_NOW.isoformat(),
            tier="working",
            url="fake://placeholder",
        ),
        statistics_artifact,
    ]
    with version_store.transaction() as tx:
        for artifact in artifacts:
            tx.execute(
                "INSERT INTO datasets.artifact_inventory "
                "(artifact_digest, size_bytes, created_at) VALUES (%s, %s, %s)",
                (artifact.artifact_digest, artifact.bytes, _NOW),
            )
            tx.execute(
                "INSERT INTO datasets.artifact_location "
                "(artifact_digest, tier, bucket, object_name, stored_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    artifact.artifact_digest,
                    "working",
                    "fake-bucket",
                    artifact.artifact_digest,
                    _NOW,
                ),
            )
    minted = assemble_manifest(
        stored=stored_definition,
        materialisation=result,
        artifacts=artifacts,
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )
    LiveVersionRecordWriter(store=version_store, now=lambda: _NOW).record_version(
        definition_version_record(
            manifest=minted,
            stored=stored_definition,
            materialisation=result,
            artifacts=artifacts,
        )
    )
    assert minted.series_count == 1

    door = build_snapshot_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=LiveSnapshotReader(store=version_store),
            artifact_store=artifact_store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["series_count"] == 1
    statistics_entries = [
        a for a in body["artifacts"] if a["rendering"] == {"statistics": "series"}
    ]
    assert len(statistics_entries) == 1
    assert statistics_entries[0]["media_type"] == "application/vnd.apache.parquet"


def _bytes_with_a_digest_sorting_after(other_digest: str) -> bytes:
    """Return distinct bytes whose `sha256:` digest sorts after `other_digest`."""
    counter = 0
    while True:
        candidate = f"data-bytes-{counter}".encode()
        digest = "sha256:" + hashlib.sha256(candidate).hexdigest()
        if digest > other_digest:
            return candidate
        counter += 1


def test_snapshot_lists_the_data_artifact_before_the_statistics_artifact_regardless_of_digest_order(
    version_store,
):
    """Pre-#60 clients select the first Parquet artifact; the data entry must come first.

    The data digest sorts after the statistics digest in this fixture.
    """
    artifact_store = FakeArtifactStore()
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    statistics_artifact = render_and_store_series_statistics(
        data=result.series_statistics.data,
        subject_id="spec_a",
        content_version="cv1",
        artifact_store=artifact_store,
    )
    stored = artifact_store.put(
        _bytes_with_a_digest_sorting_after(statistics_artifact.artifact_digest), tier=Tier.working
    )
    assert stored.digest > statistics_artifact.artifact_digest

    stored_definition = StoredDefinition(
        spec_id="spec_a",
        schema_version=1,
        identity={"identity_version": "spec1"},
        identity_bytes=b"{}",
        identity_digest="sha256:fixture",
    )
    data_artifact = ArtifactEntryModel(
        artifact_digest=stored.digest,
        rendering={"format": "csv"},
        bytes=stored.size_bytes,
        media_type="text/csv",
        created_at=_NOW.isoformat(),
        tier="working",
        url="fake://placeholder",
    )
    artifacts = [data_artifact, statistics_artifact]
    with version_store.transaction() as tx:
        for artifact in artifacts:
            tx.execute(
                "INSERT INTO datasets.artifact_inventory "
                "(artifact_digest, size_bytes, created_at) VALUES (%s, %s, %s)",
                (artifact.artifact_digest, artifact.bytes, _NOW),
            )
            tx.execute(
                "INSERT INTO datasets.artifact_location "
                "(artifact_digest, tier, bucket, object_name, stored_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    artifact.artifact_digest,
                    "working",
                    "fake-bucket",
                    artifact.artifact_digest,
                    _NOW,
                ),
            )
    minted = assemble_manifest(
        stored=stored_definition,
        materialisation=result,
        artifacts=artifacts,
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )
    LiveVersionRecordWriter(store=version_store, now=lambda: _NOW).record_version(
        definition_version_record(
            manifest=minted,
            stored=stored_definition,
            materialisation=result,
            artifacts=artifacts,
        )
    )

    door = build_snapshot_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=LiveSnapshotReader(store=version_store),
            artifact_store=artifact_store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [a["rendering"] for a in body["artifacts"]] == [
        {"format": "csv"},
        {"statistics": "series"},
    ]


def test_a_freshly_minted_definitions_snapshot_lists_only_listable_renderings(version_store):
    """A frame1 record-only artifact is omitted; the requested rendering remains."""
    artifact_store = FakeArtifactStore()
    csv_stored = artifact_store.put(b"csv-bytes", tier=Tier.working)
    frame1_stored = artifact_store.put(b"frame1-bytes", tier=Tier.working)
    with version_store.transaction() as tx:
        for stored in (csv_stored, frame1_stored):
            tx.execute(
                "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes, created_at) "
                "VALUES (%s, %s, %s)",
                (stored.digest, stored.size_bytes, _NOW),
            )
            tx.execute(
                "INSERT INTO datasets.artifact_location "
                "(artifact_digest, tier, bucket, object_name, stored_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (stored.digest, "working", stored.bucket, stored.object_name, _NOW),
            )

    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored_definition = StoredDefinition(
        spec_id="spec_a",
        schema_version=1,
        identity={"identity_version": "spec1"},
        identity_bytes=b"{}",
        identity_digest="sha256:fixture",
    )
    artifacts = [
        ArtifactEntryModel(
            artifact_digest=csv_stored.digest,
            rendering={"format": "csv"},
            bytes=csv_stored.size_bytes,
            media_type="text/csv",
            created_at=_NOW.isoformat(),
            tier="working",
            url="fake://placeholder",
        )
    ]
    minted = assemble_manifest(
        stored=stored_definition,
        materialisation=result,
        artifacts=artifacts,
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )
    LiveVersionRecordWriter(store=version_store, now=lambda: _NOW).record_version(
        definition_version_record(
            manifest=minted,
            stored=stored_definition,
            materialisation=result,
            artifacts=artifacts,
            record_only_artifacts=[
                ArtifactRecord(
                    artifact_digest=frame1_stored.digest,
                    rendering={"format": "frame1"},
                    media_type=FRAME1_MEDIA_TYPE,
                )
            ],
        )
    )

    door = build_snapshot_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=LiveSnapshotReader(store=version_store),
            artifact_store=artifact_store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)

    resp = client.get(f"/api/datasets/spec_a/{minted.content_version}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["artifacts"]) == 1
    assert body["artifacts"][0]["artifact_digest"] == csv_stored.digest
    assert body["artifacts"][0]["media_type"] == "text/csv"
