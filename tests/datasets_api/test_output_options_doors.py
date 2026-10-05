"""Mint, definition, and snapshot responses carry `output_options`; collection responses
carry it per member. Reads derive options from the current capabilities, so a capability
change affects an existing snapshot. Registry and materialisation are real; graph, SDMX,
and artifact storage are faked.
"""

from __future__ import annotations

import httpx

import datasets_api.capabilities as capabilities_module
from artifact_store import LiveArtifactInventory, StoredArtifact, Tier
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.adapter import build_identity_document
from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from datasets_api.shape import resolve_shape
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.version_store import LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

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

_SCOPE = {
    "entities": {"recipient": ["country/KEN"]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}

_FRAME_COLUMN_NAMES = (
    "stat_var",
    "recipient",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            "country/KEN": {
                "countryAlpha3Code": ("KEN",),
                "countryAlpha2Code": ("KE",),
                "name": ("Kenya",),
            },
            # The collection fixture binds UGA, which has only a name arc.
            "country/UGA": {"name": ("Uganda",)},
        },
        facets={
            (_STAT_VAR, "country/KEN"): [_FACET],
            (_STAT_VAR, "country/UGA"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya", "country/UGA": "Uganda"},
    )


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _body() -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False},
    }


def _coverage_service() -> FakeCoverageService:
    """Builds 2020 coverage rows for Kenya and Uganda."""
    index = build_coverage_index(
        [
            (_STAT_VAR, "facet1", "country/USA", "country/KEN", [2020]),
            (_STAT_VAR, "facet1", "country/USA", "country/UGA", [2020]),
        ],
        role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(
    *,
    store: FakeDatasetStore | None = None,
    artifact_store=None,
    graph: FakeMintGraphReader | None = None,
    version_writer=None,
    snapshot_reader=None,
):
    """Defaults `snapshot_reader` to a fake when it is not injected.

    The snapshot and pin doors need `spec_reader`, `snapshot_reader`, and `artifact_store`
    injected together.
    """
    store = store if store is not None else FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        version_writer=version_writer,
        snapshot_reader=snapshot_reader if snapshot_reader is not None else FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _assert_output_options_shape(output_options: dict) -> None:
    assert output_options["capability_version"] == DATASETS_CAPABILITY_VERSION
    assert {"columns", "formats", "bundle", "enrichments", "pivots", "groupby"} <= set(
        output_options
    )
    assert [c["name"] for c in output_options["columns"]] == list(_FRAME_COLUMN_NAMES)


def test_a_mint_response_carries_output_options():
    client = _client()

    resp = client.post("/api/datasets", json=_body())

    assert resp.status_code == 200, resp.text
    _assert_output_options_shape(resp.json()["output_options"])


def test_definition_retrieval_carries_output_options():
    store = FakeDatasetStore()
    client = _client(store=store)
    mint_resp = client.post("/api/datasets", json=_body())
    assert mint_resp.status_code == 200, mint_resp.text
    spec_id = mint_resp.json()["definition"]["spec_id"]

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    _assert_output_options_shape(resp.json()["output_options"])


class _InventoryRecordingArtifactStore(FakeArtifactStore):
    """Records each stored artifact in the inventory used by `LiveSnapshotReader`."""

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


def _mint_and_snapshot_client(version_store):
    """Wires a minting client and snapshot reader to the shared `version_store`."""
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    return _client(
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
    )


def test_the_snapshot_serves_output_options(version_store):
    client = _mint_and_snapshot_client(version_store)
    mint_resp = client.post("/api/datasets", json=_body())
    assert mint_resp.status_code == 200, mint_resp.text
    manifest = mint_resp.json()
    spec_id = manifest["definition"]["spec_id"]
    content_version = manifest["content_version"]

    resp = client.get(f"/api/datasets/{spec_id}/{content_version}")

    assert resp.status_code == 200, resp.text
    _assert_output_options_shape(resp.json()["output_options"])


def test_a_capability_change_since_minting_reaches_an_existing_snapshot_without_recompute(
    monkeypatch, version_store
):
    """A snapshot derives enrichment options from current capabilities.

    Kenya has ISO2, ISO3, and name arcs, so restoring capabilities exposes `recipient_iso3`
    and `recipient_label`.
    """
    client = _mint_and_snapshot_client(version_store)
    with monkeypatch.context() as write_time:
        write_time.setattr(capabilities_module, "ENABLED_ENRICHMENTS", frozenset())
        mint_resp = client.post("/api/datasets", json=_body())
        assert mint_resp.status_code == 200, mint_resp.text
        manifest = mint_resp.json()
        assert manifest["output_options"]["enrichments"] == []

    spec_id = manifest["definition"]["spec_id"]
    content_version = manifest["content_version"]

    resp = client.get(f"/api/datasets/{spec_id}/{content_version}")

    assert resp.status_code == 200, resp.text
    tokens = {e["token"] for e in resp.json()["output_options"]["enrichments"]}
    assert "recipient_iso3" in tokens
    assert "recipient_label" in tokens


def _member_sidecar(
    stat_var_ref: str, *, entities: dict[str, list[str]], graph: FakeMintGraphReader
) -> tuple[str, dict]:
    """Builds a matching definition sidecar for a collection member."""
    shape = resolve_shape(stat_var_ref, graph=graph)
    document = build_identity_document(
        shape, scope=Scope(entities=entities, facets=_SCOPE["facets"])
    )
    recomputed = recompute_identity(document)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": document,
    }


def _collection_body(collection_id: str) -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "collection_id", "collection_id": collection_id},
        "options": {"bundle": False},
    }


def _register_collection(client, *definitions: dict) -> str:
    resp = client.post(
        "/api/datasets/register", json={"schema_version": 1, "members": list(definitions)}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["collection"]["collection_id"]


def test_a_collection_manifest_carries_output_options_per_member_never_at_the_top_level():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)

    resp = client.post("/api/datasets", json=_collection_body(collection_id))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert "output_options" not in manifest
    members = manifest["collection"]["members"]
    assert len(members) == 2
    for member in members:
        _assert_output_options_shape(member["output_options"])
