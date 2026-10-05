"""Mints a frame through the datasets API and loads it through `LiveFrameStore`."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient

from artifact_store import ArtifactNotFoundError, Tier
from assembler.ports import FrameRef, FrameUnavailable
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.reference_store import ReferenceWrite
from datasets_api.revision import SentinelRevisionReader
from datasets_api.version_store import LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from presentation_store.frame_store import FRAME1_MEDIA_TYPE, LiveFrameStore
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.datasets_api._app_factories import make_client
from tests.datasets_api._artifact_doubles import FakeArtifactStore as _MintFakeArtifactStore
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._snapshot_doubles import FakeSnapshotReader
from tests.datasets_api._version_record_store import FakeVersionRecordDatasetStore
from tests.datasets_api.conftest import FakeDatasetStore
from tests.datasets_api.test_pin_door import FakeArtifactReferenceWriter
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

_BODY = {
    "schema_version": 1,
    "input": {
        "kind": "statvar_set",
        "statvars": [_STAT_VAR_REF],
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


class _ArtifactStore(_MintFakeArtifactStore):
    """Add `get` so this fake supports both mint writes and frame-store reads."""

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        key = (artifact_digest, tier.value)
        if key not in self._objects:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        return self._objects[key]


class _NoOpReferenceWriter:
    """Stub reference writes, which `load()` does not call."""

    def record_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        raise NotImplementedError("not exercised by load()")

    def record_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        raise NotImplementedError("not exercised by load()")

    def claim_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> ReferenceWrite:
        raise NotImplementedError("not exercised by load()")

    def remove_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> None:
        raise NotImplementedError("not exercised by load()")


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
    )


def _coverage() -> FakeCoverageService:
    """Build coverage for `_STAT_VAR` and `country/KEN` under `facet1`."""
    role_names = {_STAT_VAR: RoleNames(donor="donor", recipient="recipient")}
    facet_metadata = {
        "facet1": FacetMetadata(
            provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
        )
    }
    rows = [(_STAT_VAR, "facet1", "country/DONOR", "country/KEN", [2020])]
    index = build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)
    return FakeCoverageService(index)


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _mint_client(
    *, version_store: FakeVersionRecordDatasetStore, artifact_store: _ArtifactStore
) -> TestClient:
    spec_store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=spec_store),
        spec_reader=LiveSpecReader(store=spec_store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store, now=lambda: _NOW),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage(),
    )


def _seed_working_location(
    version_store: FakeVersionRecordDatasetStore, *, artifact_store: _ArtifactStore, digest: str
) -> None:
    """Seeds the SQL inventory rows; the fake artifact store only holds object bytes."""
    size = len(artifact_store.get(artifact_digest=digest, tier=Tier.working))
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory "
            "(artifact_digest, size_bytes, created_at) VALUES (%s, %s, %s)",
            (digest, size, _NOW),
        )
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name, stored_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (digest, "working", "fake-bucket", digest, _NOW),
        )


def _frame_store(
    version_store: FakeVersionRecordDatasetStore, *, artifact_store: _ArtifactStore
) -> LiveFrameStore:
    return LiveFrameStore(
        snapshots=LiveSnapshotReader(store=version_store),
        artifacts=artifact_store,
        graph=None,
        reference_writer=_NoOpReferenceWriter(),
        now=lambda: _NOW,
    )


def test_a_real_mint_frame1_rendering_loads_through_the_real_frame_store():
    version_store = FakeVersionRecordDatasetStore()
    try:
        artifact_store = _ArtifactStore()
        client = _mint_client(version_store=version_store, artifact_store=artifact_store)

        resp = client.post("/api/datasets", json=_BODY)
        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        spec_id = manifest["definition"]["spec_id"]
        content_version = manifest["content_version"]

        # C79: frame1 is internal, so read its row from `version_artifact`.
        assert len(manifest["artifacts"]) == 2
        csv_artifact = next(a for a in manifest["artifacts"] if a["media_type"] == "text/csv")
        with version_store.transaction() as tx:
            artifact_rows = tx.fetch_all(
                "SELECT artifact_digest, rendering, media_type FROM datasets.version_artifact "
                "WHERE subject_id = %s AND content_version = %s",
                (spec_id, content_version),
            )
        frame1_row = next(row for row in artifact_rows if row["rendering"] == {"format": "frame1"})
        assert frame1_row["media_type"] == FRAME1_MEDIA_TYPE
        # Content versions and artifact digests hash different inputs.
        assert content_version.startswith("cv_")
        assert frame1_row["artifact_digest"].startswith("sha256:")
        assert frame1_row["artifact_digest"] != csv_artifact["artifact_digest"]

        frame1_digest = frame1_row["artifact_digest"]
        _seed_working_location(version_store, artifact_store=artifact_store, digest=frame1_digest)
        frame_store = _frame_store(version_store, artifact_store=artifact_store)

        resolved = frame_store.load(
            frame_ref=FrameRef(spec_id=spec_id, content_version=content_version)
        )

        # Read the frame bytes written by the mint path.
        raw = artifact_store.get(artifact_digest=frame1_digest, tier=Tier.working)
        assert hashlib.sha256(raw).hexdigest() == frame1_digest.removeprefix("sha256:")
        parsed = cast("dict[str, object]", json.loads(raw))
        assert resolved.frame == parsed
        assert parsed["frame_version"] == "frame1"
        assert parsed["columns"] == [
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        ]
        assert len(cast("list[object]", parsed["rows"])) == 1
        # The stored immutable body omits expiring artifact URLs and retains null fields.
        resolved_manifest = cast("dict[str, object]", resolved.manifest)
        assert "artifacts" not in resolved_manifest
        assert resolved_manifest["content_version"] == content_version
        assert cast("dict[str, object]", resolved_manifest["definition"])["spec_id"] == spec_id
    finally:
        version_store.close()


def test_a_mint_over_the_frame1_row_ceiling_loads_as_missing_rendering(
    monkeypatch: pytest.MonkeyPatch,
):
    """With the ceiling at 0, the one-row mint is over it: the version and its CSV are recorded
    with no frame1, and the frame store refuses the version as `missing_rendering`."""
    monkeypatch.setattr("datasets_api.serve.FRAME1_MAX_ROWS", 0)
    version_store = FakeVersionRecordDatasetStore()
    try:
        artifact_store = _ArtifactStore()
        client = _mint_client(version_store=version_store, artifact_store=artifact_store)

        resp = client.post("/api/datasets", json=_BODY)
        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        spec_id = manifest["definition"]["spec_id"]
        content_version = manifest["content_version"]
        with version_store.transaction() as tx:
            artifact_rows = tx.fetch_all(
                "SELECT artifact_digest, media_type FROM datasets.version_artifact "
                "WHERE subject_id = %s AND content_version = %s",
                (spec_id, content_version),
            )
        assert artifact_rows
        assert all(row["media_type"] != FRAME1_MEDIA_TYPE for row in artifact_rows)
        for row in artifact_rows:
            _seed_working_location(
                version_store, artifact_store=artifact_store, digest=row["artifact_digest"]
            )

        with pytest.raises(FrameUnavailable) as excinfo:
            _frame_store(version_store, artifact_store=artifact_store).load(
                frame_ref=FrameRef(spec_id=spec_id, content_version=content_version)
            )

        assert excinfo.value.reason == "missing_rendering"
    finally:
        version_store.close()
