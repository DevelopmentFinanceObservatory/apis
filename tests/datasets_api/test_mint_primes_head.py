"""Each mint kind primes `datasets.subject_head`, so its first GET serves the stored
`bundle=false` rendering without graph or SDMX reads. Stores and materialisation are real;
graph, SDMX, and artifact storage are faked.
"""

from __future__ import annotations

import httpx

from artifact_store import LiveArtifactInventory
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.adapter import build_identity_document
from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from datasets_api.shape import resolve_shape
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_STAT_VAR2 = "AnotherFlowIndicator"
_STAT_VAR2_REF = f"dcid:{_STAT_VAR2}"
_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
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

_SCOPE_DICT = {
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


def _row(*, stat_var: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{_RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
    )


def _graph() -> FakeMintGraphReader:
    arcs = {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("recipient",),
        "measuredProperty": ("amount",),
    }
    return FakeMintGraphReader(
        arcs={_STAT_VAR: arcs, _STAT_VAR2: arcs},
        facets={
            (_STAT_VAR, _RECIPIENT): [_FACET],
            (_STAT_VAR2, _RECIPIENT): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


class _CountingTransport:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.n += 1
        variable_param = request.url.params.get("c[variableMeasured]", "")
        rows = "".join(
            _row(stat_var=stat_var)
            for stat_var in (_STAT_VAR, _STAT_VAR2)
            if stat_var in variable_param
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )


def _coverage_service() -> FakeCoverageService:
    """Builds one coverage facet per StatVar, matching `_FACET` and its 2020 row."""
    index = build_coverage_index(
        [
            (_STAT_VAR, "facet1", "country/USA", _RECIPIENT, [2020]),
            (_STAT_VAR2, "facet1", "country/USA", _RECIPIENT, [2020]),
        ],
        role_names={
            _STAT_VAR: RoleNames(donor="provider", recipient="recipient"),
            _STAT_VAR2: RoleNames(donor="provider", recipient="recipient"),
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(*, graph: FakeMintGraphReader, transport: _CountingTransport):
    """Use real head, version, and snapshot stores over one shared SQL-backed fake store."""
    store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    head_store = LiveHeadStore(store=version_store)
    sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(transport))
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=_InventoryRecordingArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        head_writer=head_store,
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _spec_sidecar(*, graph: FakeMintGraphReader) -> tuple[str, dict]:
    shape = resolve_shape(_STAT_VAR_REF, graph=graph)
    document = build_identity_document(shape, scope=Scope.model_validate(_SCOPE_DICT))
    recomputed = recompute_identity(document)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": document,
    }


def _member_sidecar(stat_var_ref: str, *, graph: FakeMintGraphReader) -> tuple[str, dict]:
    shape = resolve_shape(stat_var_ref, graph=graph)
    document = build_identity_document(shape, scope=Scope.model_validate(_SCOPE_DICT))
    recomputed = recompute_identity(document)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": document,
    }


def test_spec_id_mint_primes_the_head_so_the_first_get_serves_from_it():
    graph = _graph()
    transport = _CountingTransport()
    client = _client(graph=graph, transport=transport)
    spec_id, sidecar = _spec_sidecar(graph=graph)

    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
            "options": {"bundle": False},
        },
    )
    assert minted.status_code == 200, minted.text
    sdmx_calls_after_mint = transport.n
    graph_calls_after_mint = dict(graph.calls)

    got = client.get(f"/api/datasets/{spec_id}?bundle=false")

    body = got.json()
    assert got.status_code == 200, got.text
    assert body["trace"]["served_from"] == "head"
    # Definition mints also store the `series-statistics/1` artifact.
    assert len(body["artifacts"]) == 2
    requested = next(a for a in body["artifacts"] if a["rendering"] != {"statistics": "series"})
    assert requested["media_type"] == "text/csv"
    assert transport.n == sdmx_calls_after_mint
    assert (
        graph.calls["observation_facets_batch_many"]
        == (graph_calls_after_mint["observation_facets_batch_many"])
    )


def test_collection_id_mint_primes_the_head_so_the_first_get_serves_from_it():
    graph = _graph()
    transport = _CountingTransport()
    client = _client(graph=graph, transport=transport)
    _spec_a, def_a = _member_sidecar(_STAT_VAR_REF, graph=graph)
    _spec_b, def_b = _member_sidecar(_STAT_VAR2_REF, graph=graph)

    registered = client.post(
        "/api/datasets/register", json={"schema_version": 1, "members": [def_a, def_b]}
    )
    assert registered.status_code == 200, registered.text
    collection_id = registered.json()["collection"]["collection_id"]

    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "collection_id", "collection_id": collection_id},
            "options": {"bundle": False},
        },
    )
    assert minted.status_code == 200, minted.text
    sdmx_calls_after_mint = transport.n
    graph_calls_after_mint = dict(graph.calls)

    got = client.get(f"/api/datasets/{collection_id}?bundle=false")

    body = got.json()
    assert got.status_code == 200, got.text
    assert body["trace"]["served_from"] == "head"
    # Each member has one data artifact and one `series-statistics/1` artifact.
    assert len(body["artifacts"]) == 4
    data_artifacts = [a for a in body["artifacts"] if a["rendering"].get("statistics") != "series"]
    assert {artifact["media_type"] for artifact in data_artifacts} == {"text/csv"}
    assert transport.n == sdmx_calls_after_mint
    assert (
        graph.calls["observation_facets_batch_many"]
        == (graph_calls_after_mint["observation_facets_batch_many"])
    )


def test_statvar_set_single_member_mint_primes_the_head_so_the_first_get_serves_from_it():
    """A one-member `statvar_set` mint primes the definition head on initial creation."""
    graph = _graph()
    transport = _CountingTransport()
    client = _client(graph=graph, transport=transport)

    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE_DICT},
            "options": {"bundle": False},
        },
    )
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    sdmx_calls_after_mint = transport.n
    graph_calls_after_mint = dict(graph.calls)

    got = client.get(f"/api/datasets/{spec_id}?bundle=false")

    body = got.json()
    assert got.status_code == 200, got.text
    assert body["trace"]["served_from"] == "head"
    # Definition mints also store the `series-statistics/1` artifact.
    assert len(body["artifacts"]) == 2
    requested = next(a for a in body["artifacts"] if a["rendering"] != {"statistics": "series"})
    assert requested["media_type"] == "text/csv"
    assert transport.n == sdmx_calls_after_mint
    assert (
        graph.calls["observation_facets_batch_many"]
        == graph_calls_after_mint["observation_facets_batch_many"]
    )


def test_statvar_set_collection_mint_primes_the_head_so_the_first_get_serves_from_it():
    """A multi-member `statvar_set` mint primes the collection head on initial creation."""
    graph = _graph()
    transport = _CountingTransport()
    client = _client(graph=graph, transport=transport)

    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [_STAT_VAR_REF, _STAT_VAR2_REF],
                "scope": _SCOPE_DICT,
            },
            "options": {"bundle": False},
        },
    )
    assert minted.status_code == 200, minted.text
    collection_id = minted.json()["collection"]["collection_id"]
    sdmx_calls_after_mint = transport.n
    graph_calls_after_mint = dict(graph.calls)

    got = client.get(f"/api/datasets/{collection_id}?bundle=false")

    body = got.json()
    assert got.status_code == 200, got.text
    assert body["trace"]["served_from"] == "head"
    # Each member has one data artifact and one `series-statistics/1` artifact.
    assert len(body["artifacts"]) == 4
    data_artifacts = [a for a in body["artifacts"] if a["rendering"].get("statistics") != "series"]
    assert {artifact["media_type"] for artifact in data_artifacts} == {"text/csv"}
    assert transport.n == sdmx_calls_after_mint
    assert (
        graph.calls["observation_facets_batch_many"]
        == graph_calls_after_mint["observation_facets_batch_many"]
    )
