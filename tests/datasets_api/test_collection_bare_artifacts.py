"""A bare collection records one artifact per member under a shared subject and version.
Member rendering keys keep those artifacts distinct, and head hits serve them without SDMX reads.
"""

from __future__ import annotations

from datetime import datetime

import httpx

from artifact_store import LiveArtifactInventory, StoredArtifact, Tier
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.artifact_delivery import artifact_content_disposition, artifact_file_name
from datasets_api.revision import SentinelRevisionReader
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter, rendering_key
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR_A = "BareCollectionGrantA"
_STAT_VAR_A_REF = f"dcid:{_STAT_VAR_A}"
_STAT_VAR_B = "BareCollectionGrantB"
_STAT_VAR_B_REF = f"dcid:{_STAT_VAR_B}"
_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

_MEMBER_FACETS = (
    (_STAT_VAR_A, "facet1", "OECD_DAC2A"),
    (_STAT_VAR_B, "facet2", "OECD_CRS"),
)
"""Each member cites its own source, so member files carry distinct source names."""

_FACET_BY_STAT_VAR = {
    stat_var: Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
    )
    for stat_var, facet_id, provenance in _MEMBER_FACETS
}

# The scope leaves the provenance axis unbound, so each member takes its own facet's.
_SCOPE_DICT = {
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


_OBS_VALUE_BY_STAT_VAR = {_STAT_VAR_A: "1000", _STAT_VAR_B: "2000"}
"""Distinct values give each member different CSV bytes and artifact digests."""


def _row(*, stat_var: str) -> str:
    facet = _FACET_BY_STAT_VAR[stat_var]
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{_RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,{facet.provenance_id},2020,{_OBS_VALUE_BY_STAT_VAR[stat_var]},,"
        f"{facet.facet_id}\r\n"
    )


def _graph() -> FakeMintGraphReader:
    arcs = {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("recipient",),
        "measuredProperty": ("amount",),
    }
    return FakeMintGraphReader(
        arcs={_STAT_VAR_A: arcs, _STAT_VAR_B: arcs},
        facets={(stat_var, _RECIPIENT): [facet] for stat_var, facet in _FACET_BY_STAT_VAR.items()},
        labels={"OECD_DAC2A": "OECD DAC2a", "OECD_CRS": "OECD CRS", _RECIPIENT: "Kenya"},
    )


class _CountingTransport:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.n += 1
        variable_param = request.url.params.get("c[variableMeasured]", "")
        rows = "".join(
            _row(stat_var=stat_var)
            for stat_var in (_STAT_VAR_A, _STAT_VAR_B)
            if stat_var in variable_param
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )


class _InventoryRecordingArtifactStore(FakeArtifactStore):
    """Records inventory rows on `put()` so `LiveSnapshotReader` can find stored artifacts."""

    def __init__(self, *, inventory: LiveArtifactInventory) -> None:
        super().__init__()
        self._inventory = inventory

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        stored = super().put(data, tier=tier)
        self._inventory.record(
            digest=stored.digest,
            tier=tier,
            bucket=stored.bucket,
            object_name=stored.object_name,
            size_bytes=stored.size_bytes,
            uploaded=not stored.deduplicated,
        )
        return stored


def _coverage_service() -> FakeCoverageService:
    """Builds coverage matching `_MEMBER_FACETS` for both StatVars' 2020 rows."""
    index = build_coverage_index(
        [
            (stat_var, facet_id, "country/USA", _RECIPIENT, [2020])
            for stat_var, facet_id, _provenance in _MEMBER_FACETS
        ],
        role_names={
            _STAT_VAR_A: RoleNames(donor="provider", recipient="recipient"),
            _STAT_VAR_B: RoleNames(donor="provider", recipient="recipient"),
        },
        facet_metadata={
            facet_id: FacetMetadata(
                provenance_id=provenance, measurement_method="GrantEquivalent", unit="USD"
            )
            for _stat_var, facet_id, provenance in _MEMBER_FACETS
        },
    )
    return FakeCoverageService(index)


def _client(*, graph: FakeMintGraphReader, transport: _CountingTransport, version_store):
    """Builds the client with live version stores and returns its artifact store for inspection."""
    store = FakeDatasetStore()
    head_store = LiveHeadStore(store=version_store)
    sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(transport))
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        head_writer=head_store,
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )
    return client, artifact_store


def _signed_dispositions(
    artifact_store: _InventoryRecordingArtifactStore, *, artifacts: list[dict]
):
    """Maps artifact digests to their recorded `Content-Disposition` headers."""
    return {
        artifact["artifact_digest"]: artifact_store.signed_headers[
            (Tier(artifact["tier"]), artifact["artifact_digest"])
        ][1]
        for artifact in artifacts
    }


def _version_artifact_rows(version_store, *, subject_id: str, content_version: str):
    with version_store.transaction() as tx:
        return tx.fetch_all(
            "SELECT artifact_digest, rendering_key, rendering, media_type "
            "FROM datasets.version_artifact WHERE subject_id = %s AND content_version = %s "
            "ORDER BY rendering_key",
            (subject_id, content_version),
        )


def _mint_two_member_collection(client, *, bundle: bool) -> dict:
    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [_STAT_VAR_A_REF, _STAT_VAR_B_REF],
                "scope": _SCOPE_DICT,
            },
            "options": {"bundle": bundle},
        },
    )
    assert minted.status_code == 200, minted.text
    return minted.json()


def test_bare_collection_mint_records_one_artifact_per_member_with_distinct_rendering_keys():
    graph = _graph()
    transport = _CountingTransport()
    version_store = FakeVersionRecordDatasetStore()
    client, _artifact_store = _client(graph=graph, transport=transport, version_store=version_store)

    manifest = _mint_two_member_collection(client, bundle=False)
    collection_id = manifest["collection"]["collection_id"]
    member_spec_ids = {
        member["definition"]["spec_id"] for member in manifest["collection"]["members"]
    }
    assert len(member_spec_ids) == 2

    rows = _version_artifact_rows(
        version_store, subject_id=collection_id, content_version=manifest["content_version"]
    )

    # Each member has a data row and a series-statistics row.
    assert len(rows) == 4
    assert len({row["rendering_key"] for row in rows}) == 4
    assert {row["rendering"]["member"] for row in rows} == member_spec_ids
    data_rows = [row for row in rows if row["rendering"].get("statistics") != "series"]
    statistics_rows = [row for row in rows if row["rendering"].get("statistics") == "series"]
    assert len(data_rows) == 2
    assert len(statistics_rows) == 2
    assert {row["rendering"]["member"] for row in data_rows} == member_spec_ids
    assert {row["rendering"]["member"] for row in statistics_rows} == member_spec_ids
    for row in data_rows:
        rendering = dict(row["rendering"])
        member = rendering.pop("member")
        assert member in member_spec_ids
        assert rendering == {"format": "csv", "bundle": False}
        assert rendering_key(dict(row["rendering"])) == row["rendering_key"]
    for row in statistics_rows:
        rendering = dict(row["rendering"])
        member = rendering.pop("member")
        assert member in member_spec_ids
        assert rendering == {"statistics": "series"}
        assert rendering_key(dict(row["rendering"])) == row["rendering_key"]


def test_bare_collection_head_hit_serves_both_member_artifacts():
    graph = _graph()
    transport = _CountingTransport()
    version_store = FakeVersionRecordDatasetStore()
    client, artifact_store = _client(graph=graph, transport=transport, version_store=version_store)

    manifest = _mint_two_member_collection(client, bundle=False)
    collection_id = manifest["collection"]["collection_id"]
    sdmx_calls_after_mint = transport.n
    mint_dispositions = _signed_dispositions(artifact_store, artifacts=manifest["artifacts"])

    got = client.get(f"/api/datasets/{collection_id}?bundle=false")

    assert got.status_code == 200, got.text
    body = got.json()
    assert body["trace"]["served_from"] == "head"
    # Each member has one data artifact and one statistics artifact.
    assert len(body["artifacts"]) == 4
    data_artifacts = [a for a in body["artifacts"] if a["rendering"].get("statistics") != "series"]
    assert {artifact["media_type"] for artifact in data_artifacts} == {"text/csv"}
    assert transport.n == sdmx_calls_after_mint

    head_dispositions = _signed_dispositions(artifact_store, artifacts=body["artifacts"])
    assert len(set(head_dispositions.values())) == 4
    for digest, disposition in head_dispositions.items():
        assert disposition == mint_dispositions[digest]
    # Each member's files are named by that member's own sources, version and first mint.
    labels = {source["ref"]: source["label"] for source in manifest["citation"]["sources"]}
    members = {
        member["definition"]["spec_id"]: member for member in manifest["collection"]["members"]
    }
    for artifact in body["artifacts"]:
        member = members[artifact["rendering"]["member"]]
        statistics = artifact["rendering"].get("statistics") == "series"
        expected = artifact_file_name(
            source_names=[labels[ref] for ref in member["provenance_refs"]],
            prepared_at=datetime.fromisoformat(member["prepared_at"]),
            content_version=member["content_version"],
            media_type=artifact["media_type"],
            rendering=artifact["rendering"],
            statistics=statistics,
        )
        assert head_dispositions[artifact["artifact_digest"]] == artifact_content_disposition(
            expected
        )


def test_bare_collection_head_hit_re_materialises_an_unstored_rendering():
    graph = _graph()
    transport = _CountingTransport()
    version_store = FakeVersionRecordDatasetStore()
    client, _artifact_store = _client(graph=graph, transport=transport, version_store=version_store)

    manifest = _mint_two_member_collection(client, bundle=False)
    collection_id = manifest["collection"]["collection_id"]
    sdmx_calls_after_mint = transport.n

    got = client.get(f"/api/datasets/{collection_id}?format=parquet&bundle=false")

    assert got.status_code == 200, got.text
    body = got.json()
    assert body["trace"]["served_from"] == "materialise"
    assert transport.n > sdmx_calls_after_mint


def test_definition_mint_bare_records_one_artifact_with_no_member_key():
    graph = _graph()
    transport = _CountingTransport()
    version_store = FakeVersionRecordDatasetStore()
    client, artifact_store = _client(graph=graph, transport=transport, version_store=version_store)

    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_A_REF], "scope": _SCOPE_DICT},
            "options": {"bundle": False},
        },
    )
    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    spec_id = manifest["definition"]["spec_id"]

    rows = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=manifest["content_version"]
    )
    listable_rows = [row for row in rows if row["media_type"] == "text/csv"]
    assert len(listable_rows) == 1
    assert "member" not in listable_rows[0]["rendering"]

    got = client.get(f"/api/datasets/{spec_id}?bundle=false")
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["trace"]["served_from"] == "head"
    expected_disposition = artifact_content_disposition(
        artifact_file_name(
            source_names=[source["label"] for source in manifest["citation"]["sources"]],
            prepared_at=datetime.fromisoformat(manifest["prepared_at"]),
            content_version=manifest["content_version"],
            media_type="text/csv",
            rendering={"format": "csv", "bundle": False},
            statistics=False,
        )
    )
    data_artifacts = [a for a in body["artifacts"] if a["rendering"].get("statistics") != "series"]
    dispositions = _signed_dispositions(artifact_store, artifacts=data_artifacts)
    assert set(dispositions.values()) == {expected_disposition}


def test_bundled_collection_mint_records_one_artifact_with_no_member_key():
    graph = _graph()
    transport = _CountingTransport()
    version_store = FakeVersionRecordDatasetStore()
    client, artifact_store = _client(graph=graph, transport=transport, version_store=version_store)

    manifest = _mint_two_member_collection(client, bundle=True)
    collection_id = manifest["collection"]["collection_id"]

    rows = _version_artifact_rows(
        version_store, subject_id=collection_id, content_version=manifest["content_version"]
    )

    # The bundle has no member key. Statistics remain per-member.
    assert len(rows) == 3
    bundle_rows = [row for row in rows if row["media_type"] == "application/zip"]
    statistics_rows = [row for row in rows if row["rendering"].get("statistics") == "series"]
    assert len(bundle_rows) == 1
    assert "member" not in bundle_rows[0]["rendering"]
    assert len(statistics_rows) == 2
    assert all("member" in row["rendering"] for row in statistics_rows)

    got = client.get(f"/api/datasets/{collection_id}?bundle=true")
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["trace"]["served_from"] == "head"
    expected_disposition = artifact_content_disposition(
        artifact_file_name(
            source_names=[source["label"] for source in manifest["citation"]["sources"]],
            prepared_at=datetime.fromisoformat(manifest["prepared_at"]),
            content_version=manifest["content_version"],
            media_type="application/zip",
            rendering={"format": "csv", "bundle": True},
            statistics=False,
        )
    )
    bundle_artifacts = [a for a in body["artifacts"] if a["media_type"] == "application/zip"]
    dispositions = _signed_dispositions(artifact_store, artifacts=bundle_artifacts)
    assert set(dispositions.values()) == {expected_disposition}
