"""Stores canonical frames through the datasets API and loads them through `LiveFrameStore`."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import cast

import httpx
import pyarrow as pa
from dfo_frame_grammar import Frame, canonical_frame_bytes, canonical_frame_schema, unit_cell_array
from fastapi.testclient import TestClient

from artifact_store import ArtifactNotFoundError, Tier
from assembler.ports import FrameRef
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.reference_store import ReferenceWrite
from datasets_api.render import render_and_store_canonical_frame
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import VersionRecord
from dc_sdmx import SdmxClient
from materialise.formats import CANONICAL_FRAME_MEDIA_TYPE, read_canonical_frame
from materialise.graph import Facet
from presentation_store.frame_store import LiveFrameStore
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.datasets_api._app_factories import make_client
from tests.datasets_api._artifact_doubles import FakeArtifactStore as _MintFakeArtifactStore
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._snapshot_doubles import FakeSnapshotReader, version_body
from tests.datasets_api._version_facts import facts_from_body
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
    role_names = {_STAT_VAR: RoleNames(donor=None, recipient="recipient")}
    facet_metadata = {
        "facet1": FacetMetadata(
            provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
        )
    }
    rows = [(_STAT_VAR, "facet1", ABSENT_ENTITY, "country/KEN", [2020])]
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
        allowances=None,
        now=lambda: _NOW,
    )


def test_a_real_mint_canonical_frame_loads_through_the_real_frame_store():
    version_store = FakeVersionRecordDatasetStore()
    try:
        artifact_store = _ArtifactStore()
        client = _mint_client(version_store=version_store, artifact_store=artifact_store)

        resp = client.post("/api/datasets", json=_BODY)
        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        spec_id = manifest["definition"]["spec_id"]
        content_version = manifest["content_version"]

        # C79: the canonical frame is internal, so read its row from `version_artifact`.
        assert len(manifest["artifacts"]) == 2
        csv_artifact = next(a for a in manifest["artifacts"] if a["media_type"] == "text/csv")
        with version_store.transaction() as tx:
            artifact_rows = tx.fetch_all(
                "SELECT artifact_digest, rendering, media_type FROM datasets.version_artifact "
                "WHERE subject_id = %s AND content_version = %s",
                (spec_id, content_version),
            )
        frame_row = next(row for row in artifact_rows if row["rendering"] == {"format": "frame"})
        assert frame_row["media_type"] == CANONICAL_FRAME_MEDIA_TYPE
        assert frame_row["artifact_digest"] != csv_artifact["artifact_digest"]

        frame_digest = frame_row["artifact_digest"]
        _seed_working_location(version_store, artifact_store=artifact_store, digest=frame_digest)
        frame_store = _frame_store(version_store, artifact_store=artifact_store)

        resolved = frame_store.load(
            frame_ref=FrameRef(spec_id=spec_id, content_version=content_version)
        )

        # The frame Presentation reads is the frame1 document of the frame the mint stored.
        raw = artifact_store.get(artifact_digest=frame_digest, tier=Tier.working)
        assert hashlib.sha256(raw).hexdigest() == frame_digest.removeprefix("sha256:")
        parsed = cast(
            "dict[str, object]", json.loads(canonical_frame_bytes(read_canonical_frame(raw)))
        )
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


_LARGE_ROWS = 250_000


def _large_frame() -> Frame:
    """A one-role frame of `_LARGE_ROWS` rows built as Arrow columns, in canonical row order."""
    columns = (
        "stat_var",
        "recipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    table = pa.Table.from_arrays(
        [
            pa.array([_STAT_VAR_REF] * _LARGE_ROWS),
            pa.array([f"country/R{index:06d}" for index in range(_LARGE_ROWS)]),
            pa.array(["2020"] * _LARGE_ROWS),
            pa.array([float(index) for index in range(_LARGE_ROWS)]),
            unit_cell_array("USD", _LARGE_ROWS),
            pa.array(["GrantEquivalent"] * _LARGE_ROWS),
            pa.array(["OECD_DAC2A"] * _LARGE_ROWS),
            pa.nulls(_LARGE_ROWS),
        ],
        schema=canonical_frame_schema(columns),
    )
    return Frame.from_table(columns=columns, table=table)


def test_a_250000_row_canonical_frame_is_written_and_opens_in_presentation():
    """A version of 250,000 rows stores its canonical frame, which Presentation opens whole.
    Builds and reads 250,000 rows, about two seconds."""
    version_store = FakeVersionRecordDatasetStore()
    try:
        artifact_store = _ArtifactStore()
        frame = _large_frame()
        chunks = [
            Frame(columns=frame.columns, table=frame.table.slice(start, 60_000))
            for start in range(0, _LARGE_ROWS, 60_000)
        ]
        record = render_and_store_canonical_frame(
            frame_chunks=chunks,
            subject_id="spec_" + "a" * 32,
            content_version="cv_" + "b" * 32,
            artifact_store=artifact_store,
        )
        LiveVersionRecordWriter(store=version_store, now=lambda: _NOW).record_version(
            VersionRecord(
                subject_id="spec_" + "a" * 32,
                subject_kind="spec",
                content_version="cv_" + "b" * 32,
                schema_version=1,
                facts=facts_from_body(version_body("cv_" + "b" * 32)),
                artifacts=(record,),
            )
        )
        _seed_working_location(
            version_store, artifact_store=artifact_store, digest=record.artifact_digest
        )

        resolved = _frame_store(version_store, artifact_store=artifact_store).load(
            frame_ref=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)
        )

        rows = cast("list[list[object]]", resolved.frame["rows"])
        assert len(rows) == _LARGE_ROWS
        assert rows[-1] == [
            _STAT_VAR_REF,
            f"country/R{_LARGE_ROWS - 1:06d}",
            "2020",
            _LARGE_ROWS - 1,
            "USD",
            "GrantEquivalent",
            "OECD_DAC2A",
            None,
        ]
    finally:
        version_store.close()
