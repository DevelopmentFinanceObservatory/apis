"""Tests `spec_id` registry hits and misses, sidecar checks, member
reproducibility, scope surgery, and wide-member enumeration. Registry and
materialisation are real; graph, SDMX, and artifact storage are faked.
"""

from __future__ import annotations

import httpx
import pytest

import datasets_api.doors.mint as mint_module
from code_index.query import CodeIndex
from code_index.types import IndexStamp
from datasets_api.adapter import build_identity_document
from datasets_api.config import DatasetsBuildSettings
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import Scope
from datasets_api.shape import resolve_shape
from datasets_api.surgery import apply_scope_surgery
from dc_sdmx import SdmxClient
from dfo_platform.named import MAX_VALUE_CHARS, cut
from graph_generation import FakeGeneration
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.rate_tables import RateTableCache
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from statvar_members.enumerate import CANDIDATE_CAP
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import described_identity, pinned_identity, register_body
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore, wire_definition_for
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
    "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)

# Defaults apply when scope surgery clears the `facets` member.
_FACET_POLICIES: dict[tuple[str, str], FacetDefaultPolicy] = {
    (_STAT_VAR_REF, "provenance"): FacetDefaultPolicy(
        resolved_ref="OECD_DAC2A",
        policy_node="policy/provenance",
        policy_version="v1",
        selection_reason="declared_default",
    ),
    (_STAT_VAR_REF, "measurementMethod"): FacetDefaultPolicy(
        resolved_ref="GrantEquivalent",
        policy_node="policy/measurementMethod",
        policy_version="v1",
        selection_reason="declared_default",
    ),
    (_STAT_VAR_REF, "unit"): FacetDefaultPolicy(
        resolved_ref="USD",
        policy_node="policy/unit",
        policy_version="v1",
        selection_reason="declared_default",
    ),
}

_SCOPE_DICT = {
    "entities": {"recipient": ["country/KEN"]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}

_STATVAR_SET_BODY = {
    "schema_version": 1,
    "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE_DICT},
    "options": {"bundle": False},
}


def _arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        _STAT_VAR: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
        }
    }


def _graph(arcs: dict[str, dict[str, tuple[str, ...]]] | None = None) -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs=arcs if arcs is not None else _arcs(),
        # Facet probes use the bare dcid; policy lookup keeps the prefixed id.
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        policies=_FACET_POLICIES,
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
    )


def _sdmx_transport(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _coverage_service(
    *,
    stat_vars: tuple[str, ...] = (_STAT_VAR,),
    recipient: str = "country/KEN",
) -> FakeCoverageService:
    """Builds one 2015 coverage row per StatVar for the requested recipient."""
    index = build_coverage_index(
        [(stat_var, "facet1", ABSENT_ENTITY, recipient, [2015]) for stat_var in stat_vars],
        role_names={
            stat_var: RoleNames(donor=None, recipient="recipient") for stat_var in stat_vars
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(
    *,
    artifact_store,
    store: FakeDatasetStore | None = None,
    graph: FakeMintGraphReader | None = None,
    sdmx_requests: list[httpx.Request] | None = None,
    rate_cache: RateTableCache | None = None,
    code_index: CodeIndex | None = None,
    coverage: FakeCoverageService | None = None,
    settings: DatasetsBuildSettings | None = None,
):
    store = store if store is not None else FakeDatasetStore()
    requests = sdmx_requests if sdmx_requests is not None else []
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(requests))
    kwargs: dict[str, object] = {}
    if rate_cache is not None:
        kwargs["rate_cache"] = rate_cache
    if code_index is not None:
        kwargs["code_index"] = code_index
    if settings is not None:
        kwargs["settings"] = settings
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _graph(),
        sdmx=sdmx,
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage if coverage is not None else _coverage_service(),
        **kwargs,
    )


def _spec_id_body(
    spec_id: str, *, definition: dict | None = None, scope: dict | None = None
) -> dict:
    input_: dict = {"kind": "spec_id", "spec_id": spec_id}
    if definition is not None:
        input_["definition"] = definition
    if scope is not None:
        input_["scope"] = scope
    return {"schema_version": 1, "input": input_, "options": {"bundle": False}}


def _sidecar_for(*, graph: FakeMintGraphReader) -> tuple[str, dict]:
    """Builds a matching spec id and sidecar for `_SCOPE_DICT`."""
    shape = resolve_shape(_STAT_VAR_REF, graph=graph)
    document = build_identity_document(shape, scope=Scope.model_validate(_SCOPE_DICT))
    recomputed = recompute_identity(document)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": document,
    }


def test_registry_hit_materialises_the_registered_definition():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    base_manifest = base.json()
    spec_id = base_manifest["definition"]["spec_id"]

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] == spec_id
    assert manifest["content_version"] == base_manifest["content_version"]
    assert (
        manifest["artifacts"][0]["artifact_digest"]
        == base_manifest["artifacts"][0]["artifact_digest"]
    )


def test_unregistered_spec_id_with_no_sidecar_refuses_spec_not_found():
    client = _client(artifact_store=FakeArtifactStore())

    resp = client.post("/api/datasets", json=_spec_id_body("spec_" + "0" * 32))

    assert resp.status_code == 404, resp.text
    problem = resp.json()
    assert problem["code"] == "spec_not_found"
    assert problem["pointer"] == "/input/spec_id"


def test_unregistered_spec_id_with_a_matching_sidecar_mints_it_then_serves():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    spec_id, definition = _sidecar_for(graph=graph)

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] == spec_id
    assert spec_id in store.state.spec_registry
    assert store.state.spec_definition[(spec_id, 1)]["definition"]["identity"] == (
        recompute_identity(definition["identity"]).identity
    )
    assert len(store.state.spec_registry) == 1


def test_a_sidecar_identity_that_carries_entity_groups_is_refused():
    """A named group is request context beside an identity, so the mint refuses a sidecar
    whose identity holds one and writes no registry row."""
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    spec_id, definition = _sidecar_for(graph=graph)
    groups = {"recipient": {"label": "recipient countries"}}
    definition = {**definition, "identity": {**definition["identity"], "entity_groups": groups}}

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert (problem["code"], problem["pointer"]) == ("request_invalid", "/input/spec_id")
    assert store.state.spec_registry == {}


def test_a_sidecar_that_does_not_reproduce_against_the_graph_leaves_no_registry_row():
    """A sidecar whose role shape differs from the graph is refused before registry write."""
    # The sidecar binds recipient; the graph below declares provider.
    stale_arcs = {
        _STAT_VAR: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
        }
    }
    stale_shape = resolve_shape(_STAT_VAR_REF, graph=_graph(stale_arcs))
    document = build_identity_document(
        stale_shape, scope=Scope(entities={"recipient": ["country/KEN"]})
    )
    spec_id = recompute_identity(document).spec_id
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": document}

    store = FakeDatasetStore()
    drifted_arcs = {
        _STAT_VAR: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("provider",),
            "measuredProperty": ("amount",),
        }
    }
    graph = _graph(drifted_arcs)
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "definition_not_reproducible"
    assert store.state.spec_registry == {}


def test_a_sidecar_whose_hash_disagrees_with_an_unregistered_id_refuses_spec_id_mismatch():
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), graph=graph)
    _real_spec_id, definition = _sidecar_for(graph=graph)
    wrong_spec_id = "spec_" + "0" * 32
    # Keep the sidecar id aligned with the outer id; its identity still hashes to `_real_spec_id`.
    definition = {**definition, "spec_id": wrong_spec_id}

    resp = client.post("/api/datasets", json=_spec_id_body(wrong_spec_id, definition=definition))

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "spec_id_mismatch"


def test_a_registered_spec_id_with_a_matching_sidecar_still_serves():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    definition = base.json()["definition"]
    spec_id = definition["spec_id"]

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 200, resp.text
    assert resp.json()["definition"]["spec_id"] == spec_id


def test_a_registered_spec_id_with_a_mismatched_sidecar_still_refuses_spec_id_mismatch():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    # This valid identity hashes to a different id from the registered one.
    foreign_document = build_identity_document(
        resolve_shape(_STAT_VAR_REF, graph=graph), scope=None
    )
    foreign_definition = {"schema_version": 1, "spec_id": spec_id, "identity": foreign_document}

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=foreign_definition))

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "spec_id_mismatch"


def test_member_uniqueness_drift_since_minting_refuses_definition_not_reproducible():
    store = FakeDatasetStore()
    arcs = _arcs()
    graph = _graph(arcs)
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    # The StatVar drops `recipient` as a role while the stored identity still binds it.
    arcs[_STAT_VAR] = {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("provider",),
        "measuredProperty": ("amount",),
    }

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "definition_not_reproducible"
    assert {issue["member"] for issue in problem["issues"]} == {spec_id}
    assert "pointer" not in problem
    assert all("pointer" not in issue for issue in problem["issues"])


def test_scope_entities_on_a_hit_still_checks_reproducibility_against_drift_since_minting():
    """A registry hit checks shape reproducibility when `scope.entities` is supplied."""
    store = FakeDatasetStore()
    arcs = _arcs()
    graph = _graph(arcs)
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    # The StatVar drops `recipient` as a role while the stored identity still binds it.
    arcs[_STAT_VAR] = {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("provider",),
        "measuredProperty": ("amount",),
    }

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body(spec_id, scope={"entities": {"recipient": ["country/KEN"]}}),
    )

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "definition_not_reproducible"
    assert {issue["member"] for issue in problem["issues"]} == {spec_id}


def test_a_breadth_only_stored_identity_refuses_spec_not_exportable_through_the_mint_door():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    identity = described_identity()
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id))

    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "spec_not_exportable"


_ODA_GRANTS = "ODAGrantsAmount"
_ODA_GRANTS_REF = f"dcid:{_ODA_GRANTS}"
_ODA_LOANS = "ODALoansAmount"
_ODA_LOANS_REF = f"dcid:{_ODA_LOANS}"


def _described_two_member_graph() -> FakeMintGraphReader:
    """Members share `populationType=DevelopmentFinance` and `measuredProperty=amount`,
    but differ by `DevelopmentFinanceScheme`. A set-bound slot enumerates both.
    """
    member_arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("recipient",),
        "constraintProperties": ("DevelopmentFinanceScheme",),
    }
    return FakeMintGraphReader(
        arcs={
            _ODA_GRANTS: {**member_arcs, "DevelopmentFinanceScheme": ("ODAGrants",)},
            _ODA_LOANS: {**member_arcs, "DevelopmentFinanceScheme": ("ODALoans",)},
        },
        facets={
            (_ODA_GRANTS, "country/KEN"): [_FACET],
            (_ODA_LOANS, "country/KEN"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
    )


_ODA_ROWS_BY_STAT_VAR = {
    _ODA_GRANTS: (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_GRANTS},country/KEN,USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    ),
    _ODA_LOANS: (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_LOANS},country/KEN,USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2015,500,,facet1\r\n"
    ),
}


def _described_two_member_sdmx_transport(requests: list[httpx.Request]):
    """Returns only rows named by each request's `c[variableMeasured]` parameter."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        requested = request.url.params.get("c[variableMeasured]", "").split(",")
        body = _CSV_HEADER + "".join(
            _ODA_ROWS_BY_STAT_VAR[stat_var]
            for stat_var in requested
            if stat_var in _ODA_ROWS_BY_STAT_VAR
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return httpx.MockTransport(handler)


def test_an_unregistered_described_arm_spec_id_with_a_sidecar_mints_every_enumerated_member():
    """A described-arm sidecar mints both graph members in one group and lists them in the trace."""
    store = FakeDatasetStore()
    graph = _described_two_member_graph()
    sdmx_requests: list[httpx.Request] = []
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(
            base_url=_BASE_URL, transport=_described_two_member_sdmx_transport(sdmx_requests)
        ),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(stat_vars=(_ODA_GRANTS, _ODA_LOANS)),
    )
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "amount",
        },
        "slots": [
            {
                "property": "DevelopmentFinanceScheme",
                "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
            }
        ],
        "entities": {"recipient": ["country/KEN"]},
        "facets": {
            "provenance": {"state": "value", "ref": "OECD_DAC2A"},
            "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
            "unit": {"state": "value", "ref": "USD"},
        },
        "requested_window": {"kind": "all"},
    }
    spec_id = recompute_identity(identity).spec_id
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": identity}

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] == spec_id
    resolved_refs = {entry["ref"] for entry in manifest["trace"]["resolved_stat_vars"]}
    assert resolved_refs == {_ODA_GRANTS_REF, _ODA_LOANS_REF}
    assert spec_id in store.state.spec_registry


def test_a_graph_write_between_two_mints_reaches_the_next_graph_generation():
    """Member groups are cached per graph generation, so a repeat mint under the same
    generation serves the group it resolved, and the first mint after the generation moves
    reads the write."""
    store = FakeDatasetStore()
    generation = FakeGeneration(token="rev1")
    member_arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("recipient",),
        "constraintProperties": ("DevelopmentFinanceScheme",),
    }
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        _ODA_GRANTS: {**member_arcs, "DevelopmentFinanceScheme": ("ODAGrants",)},
    }
    graph = FakeMintGraphReader(
        arcs=arcs,
        facets={
            (_ODA_GRANTS, "country/KEN"): [_FACET],
            (_ODA_LOANS, "country/KEN"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
        generation=generation,
    )
    sdmx_requests: list[httpx.Request] = []
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(
            base_url=_BASE_URL, transport=_described_two_member_sdmx_transport(sdmx_requests)
        ),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(stat_vars=(_ODA_GRANTS, _ODA_LOANS)),
    )
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "amount",
        },
        "slots": [
            {
                "property": "DevelopmentFinanceScheme",
                "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
            }
        ],
        "entities": {"recipient": ["country/KEN"]},
        "facets": {
            "provenance": {"state": "value", "ref": "OECD_DAC2A"},
            "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
            "unit": {"state": "value", "ref": "USD"},
        },
        "requested_window": {"kind": "all"},
    }
    spec_id = recompute_identity(identity).spec_id
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": identity}

    first = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))
    assert first.status_code == 200, first.text
    first_refs = {entry["ref"] for entry in first.json()["trace"]["resolved_stat_vars"]}
    assert first_refs == {_ODA_GRANTS_REF}

    arcs[_ODA_LOANS] = {**member_arcs, "DevelopmentFinanceScheme": ("ODALoans",)}

    def mint_refs() -> set[str]:
        response = client.post("/api/datasets", json=_spec_id_body(spec_id))
        assert response.status_code == 200, response.text
        return {entry["ref"] for entry in response.json()["trace"]["resolved_stat_vars"]}

    assert mint_refs() == {_ODA_GRANTS_REF}
    generation.advance("rev2")
    assert mint_refs() == {_ODA_GRANTS_REF, _ODA_LOANS_REF}


def test_every_mint_shares_the_surfaces_process_level_dependencies(monkeypatch):
    """Each request's `MintDeps` wraps the one graph, SDMX client, artifact store, enrichment
    cache, and materialise semaphore the surface built."""
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]
    views: list[MintDeps] = []
    real_mint_deps = mint_module.mint_deps

    def _spy(deps, **kwargs):
        view = real_mint_deps(deps, **kwargs)
        views.append(view)
        return view

    monkeypatch.setattr(mint_module, "mint_deps", _spy)

    client.post("/api/datasets", json=_spec_id_body(spec_id))
    client.post("/api/datasets", json=_spec_id_body(spec_id))

    first, second = views
    for name in ("graph", "sdmx", "artifact_store", "enrichment", "materialise_semaphore"):
        assert getattr(first, name) is getattr(second, name), name


def test_a_malformed_stat_var_ref_refuses_member_not_resolvable_not_a_500():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    identity = pinned_identity(stat_var="not-a-known-statvar")
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "member_not_resolvable"
    assert problem["issues"][0]["pointer"] == "/input/spec_id"


@pytest.mark.parametrize(
    "identity",
    [
        {"slots": 1},
        described_identity(entities={"creditor": ["country/CHN"]}, open_roles=["origin"]),
    ],
    ids=["slots-not-a-list", "role-bound-and-open-on-one-axis"],
)
def test_a_malformed_sidecar_identity_on_a_miss_refuses_request_invalid_not_a_500(identity):
    """A malformed sidecar identity is rejected by schema validation on a registry miss."""
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    spec_id = "spec_" + "0" * 32
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": identity}

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/spec_id"
    assert store.state.spec_registry == {}


def test_a_malformed_sidecar_identity_on_a_hit_refuses_request_invalid_not_a_500():
    """A malformed sidecar identity is rejected by schema validation on a registry hit."""
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": {"slots": 1}}

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/spec_id"


def test_a_miss_with_a_malformed_ref_sidecar_and_no_scope_leaves_no_registry_row():
    """An unresolvable sidecar is refused before registry write on a miss."""
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    identity = pinned_identity(stat_var="not-a-known-statvar")
    spec_id = recompute_identity(identity).spec_id
    definition = wire_definition_for(spec_id=spec_id, identity=identity)

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "member_not_resolvable"
    assert problem["issues"][0]["pointer"] == "/input/spec_id"
    assert store.state.spec_registry == {}


def test_a_miss_with_a_breadth_only_sidecar_and_no_scope_leaves_no_registry_row():
    """A breadth-only sidecar is refused before its registry write."""
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    identity = described_identity()
    spec_id = recompute_identity(identity).spec_id
    definition = wire_definition_for(spec_id=spec_id, identity=identity)

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "spec_not_exportable"
    assert store.state.spec_registry == {}


def test_a_miss_with_sidecar_and_a_bad_scope_role_leaves_no_registry_row():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    spec_id, definition = _sidecar_for(graph=graph)

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body(
            spec_id, definition=definition, scope={"entities": {"donor": ["country/FRA"]}}
        ),
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "invalid_role_ref"
    assert store.state.spec_registry == {}


def test_a_miss_with_a_multi_member_sidecar_checks_scope_against_every_enumerated_member():
    """Pre-write scope validation checks every enumerated member. Here, `ODAGrantsAmount`
    leaves `recipient` unbound, while `ODALoansAmount` pins it to `country/KEN`.
    The scope must be refused against the pinned member too.
    """
    member_arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("recipient",),
    }
    graph = FakeMintGraphReader(
        arcs={
            "ODAGrantsAmount": {
                **member_arcs,
                "constraintProperties": ("DevelopmentFinanceModality", "DevelopmentFinanceScheme"),
                "DevelopmentFinanceScheme": ("ODAGrants",),
                "DevelopmentFinanceModality": ("Loan",),
            },
            "ODALoansAmount": {
                **member_arcs,
                "constraintProperties": ("DevelopmentFinanceScheme", "recipient"),
                "DevelopmentFinanceScheme": ("ODALoans",),
                "recipient": ("country/KEN",),
            },
        }
    )
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "amount",
        },
        "slots": [
            {
                "property": "DevelopmentFinanceScheme",
                "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
            },
            {"property": "recipient", "binding": {"state": "unbound"}},
            {"property": "DevelopmentFinanceModality", "binding": {"state": "unbound"}},
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    spec_id = recompute_identity(identity).spec_id
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": identity}
    store = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(
            base_url=_BASE_URL, transport=httpx.MockTransport(lambda request: httpx.Response(200))
        ),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
    )

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body(
            spec_id, definition=definition, scope={"entities": {"recipient": ["country/KEN"]}}
        ),
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "role_incompatible"
    assert store.state.spec_registry == {}


def test_an_unregistered_spec_id_with_a_scope_block_and_no_sidecar_still_refuses_spec_not_found():
    client = _client(artifact_store=FakeArtifactStore())

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body("spec_" + "0" * 32, scope={"window": {"kind": "latest"}}),
    )

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "spec_not_found"


def test_a_repeat_mint_through_this_door_returns_the_same_content_version():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]
    body = _spec_id_body(spec_id)

    first = client.post("/api/datasets", json=body)
    assert first.status_code == 200, first.text
    second = client.post("/api/datasets", json=body)
    assert second.status_code == 200, second.text

    assert second.json()["content_version"] == first.json()["content_version"]
    assert (
        second.json()["artifacts"][0]["artifact_digest"]
        == first.json()["artifacts"][0]["artifact_digest"]
    )


def test_a_sidecar_spec_id_disagreeing_with_the_outer_spec_id_refuses_request_invalid():
    """`SpecIdInput`'s own `_sidecar_spec_id_matches_outer` validator refuses
    this shape before any registry read -- the disagreement is caught at
    the wire boundary, not left to `register_definition`'s own hash check
    (which only ever sees the outer `spec_id`).
    """
    graph = _graph()
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    outer_spec_id, definition = _sidecar_for(graph=graph)
    definition = {**definition, "spec_id": "spec_" + "9" * 32}

    resp = client.post("/api/datasets", json=_spec_id_body(outer_spec_id, definition=definition))

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"
    assert store.state.spec_registry == {}


def test_no_scope_block_leaves_spec_id_mint_behaviour_unchanged():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] == spec_id
    assert manifest["definition"]["identity"] == base.json()["definition"]["identity"]


def test_scope_surgery_replacing_the_window_member_renames_it_to_requested_window():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    base_identity = base.json()["definition"]["identity"]
    spec_id = base.json()["definition"]["spec_id"]

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body(
            spec_id,
            scope={"window": {"kind": "range", "start_year": 2015, "end_year": 2020}},
        ),
    )

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    new_spec_id = manifest["definition"]["spec_id"]
    assert new_spec_id != spec_id
    identity = manifest["definition"]["identity"]
    assert identity["requested_window"] == {"kind": "range", "start_year": 2015, "end_year": 2020}
    assert "window" not in identity
    # Members the scope block did not name carry forward unchanged.
    assert identity["entities"] == base_identity["entities"]
    assert identity["facets"] == base_identity["facets"]
    assert new_spec_id in store.state.spec_registry
    assert spec_id in store.state.spec_registry


def test_scope_surgery_replacing_multiple_members_together():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body(
            spec_id,
            scope={
                "window": {"kind": "latest"},
                "entities": {"recipient": ["country/KEN", "country/UGA"]},
            },
        ),
    )

    assert resp.status_code == 200, resp.text
    identity = resp.json()["definition"]["identity"]
    assert identity["requested_window"] == {"kind": "latest"}
    assert identity["entities"] == {"recipient": ["country/KEN", "country/UGA"]}
    # `facets` was not named by this scope, so it carries forward unchanged.
    assert identity["facets"] == base.json()["definition"]["identity"]["facets"]
    assert resp.json()["definition"]["spec_id"] != spec_id


def test_scope_surgery_clearing_facets_omits_the_key_from_the_manifest_identity():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    base_identity = base.json()["definition"]["identity"]
    assert "facets" in base_identity
    spec_id = base.json()["definition"]["spec_id"]

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, scope={"facets": {}}))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] != spec_id
    assert "facets" not in manifest["definition"]["identity"]


def test_scope_surgery_clearing_entities_and_facets_together_registers_the_cleared_identity():
    """The cleared identity is registered before materialisation refuses the unbound request."""
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    base_identity = base.json()["definition"]["identity"]
    assert base_identity["entities"]
    assert "facets" in base_identity
    spec_id = base.json()["definition"]["spec_id"]

    client.post(
        "/api/datasets",
        json=_spec_id_body(
            spec_id, scope={"entities": {}, "facets": {}, "window": {"kind": "all"}}
        ),
    )

    surgered_rows = [
        row["definition"]
        for (stored_spec_id, _schema_version), row in store.state.spec_definition.items()
        if stored_spec_id != spec_id
    ]
    assert len(surgered_rows) == 1
    surgered_identity = surgered_rows[0]["identity"]
    assert surgered_identity["entities"] == {}
    assert "facets" not in surgered_identity
    assert surgered_identity["requested_window"] == {"kind": "all"}


@pytest.mark.parametrize(
    ("scope_entities", "open_roles"),
    [
        ({"creditor": ["country/CHN"]}, None),
        ({"origin": ["country/CHN"], "destination": ["country/GHA"]}, None),
        ({"destination": ["country/GHA"]}, ["origin"]),
        ({"debtor": ["country/GHA"]}, ["origin"]),
        ({}, ["origin"]),
    ],
)
def test_apply_scope_surgery_closes_an_open_role_the_scope_binds(scope_entities, open_roles):
    """A role the scope binds leaves `open_roles`, matched by axis whatever the key's
    spelling, and every other open role stays."""
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "measured_property": "amount"},
        "slots": [],
        "entities": {"destination": ["country/NGA"]},
        "open_roles": ["origin"],
        "requested_window": {"kind": "all"},
    }

    surgered = apply_scope_surgery(identity, scope=Scope(entities=scope_entities))

    assert surgered["entities"] == scope_entities
    assert surgered.get("open_roles") == open_roles
    recompute_identity(surgered)


def test_apply_scope_surgery_clears_facets_by_deleting_the_key():
    """Clearing `facets` deletes the key, matching fresh identity documents.

    An empty object hashes to the same spec id but differs from the fresh document's sparse shape.
    """
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "TestStatVar"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
        "facets": {"unit": {"state": "value", "ref": "USD"}},
    }

    surgered = apply_scope_surgery(identity, scope=Scope(facets={}))

    assert "facets" not in surgered
    cleared_by_delete = recompute_identity(surgered)
    cleared_by_empty_dict = recompute_identity({**surgered, "facets": {}})
    assert cleared_by_delete.spec_id == cleared_by_empty_dict.spec_id


def test_scope_restating_the_stored_state_exactly_serves_the_base_spec_id_with_no_second_mint():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]
    rows_before = len(store.state.spec_registry)

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body(spec_id, scope={**_SCOPE_DICT, "window": {"kind": "all"}}),
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["definition"]["spec_id"] == spec_id
    assert len(store.state.spec_registry) == rows_before


def test_an_unavailable_unit_ref_refusal_names_the_posted_ref_cut():
    """Every detail of the refusal names a 1,000-character posted ref by its first 256
    characters (`cut`)."""
    ref = "U" * 1000
    client = _client(artifact_store=FakeArtifactStore())
    scope = {**_SCOPE_DICT, "facets": {**_SCOPE_DICT["facets"], "unit": {"ref": ref}}}
    body = {**_STATVAR_SET_BODY, "input": {**_STATVAR_SET_BODY["input"], "scope": scope}}

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "facet_not_available"
    assert problem["issues"]
    details = [problem["detail"], *(issue["detail"] for issue in problem["issues"])]
    assert any(cut(ref) in detail for detail in details)
    assert all(ref[: MAX_VALUE_CHARS + 1] not in detail for detail in details)


class TestScopeSurgeryRoleRefusals:
    """Scope roles are checked against the stored shape before surgery or registry writes."""

    def test_an_undeclared_role_refuses_invalid_role_ref_with_no_registry_write(self):
        store = FakeDatasetStore()
        client = _client(artifact_store=FakeArtifactStore(), store=store)
        base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
        assert base.status_code == 200, base.text
        spec_id = base.json()["definition"]["spec_id"]
        rows_before = len(store.state.spec_registry)

        resp = client.post(
            "/api/datasets",
            json=_spec_id_body(spec_id, scope={"entities": {"donor": ["country/FRA"]}}),
        )

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "invalid_role_ref"
        assert len(problem["issues"]) == 1
        assert problem["issues"][0]["pointer"] == "/input/scope/entities/donor"
        assert problem["issues"][0]["member"] == "donor"
        assert len(store.state.spec_registry) == rows_before

    @pytest.mark.parametrize(
        ("dcid", "types", "kind"),
        [
            pytest.param(
                "Count_Person", ("StatisticalVariable",), "a statistical variable", id="variable"
            ),
            pytest.param(
                "geoId/06",
                ("AdministrativeArea1", "Place", "State"),
                "a place other than a country or region",
                id="state",
            ),
        ],
    )
    @pytest.mark.parametrize("entry", ["scope", "statvar_set", "sidecar"])
    def test_a_recipient_of_a_kind_no_recipient_takes_refuses_with_no_registry_write(
        self, entry: str, dcid: str, types: tuple[str, ...], kind: str
    ):
        """A recipient whose kind none of the code index's recipients take
        (`search.role_kinds`) refuses before any registry write, whether a scope, a statvar
        set or a sidecar's identity binds it."""
        store = FakeDatasetStore()
        graph = _graph({**_arcs(), dcid: {"typeOf": types}})
        client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
        scope = {**_SCOPE_DICT, "entities": {"recipient": [dcid]}}
        pointer = "/input/scope/entities/recipient"
        if entry == "scope":
            base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
            assert base.status_code == 200, base.text
            body = _spec_id_body(base.json()["definition"]["spec_id"], scope=scope)
        elif entry == "statvar_set":
            body = {
                **_STATVAR_SET_BODY,
                "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": scope},
            }
        else:
            shape = resolve_shape(_STAT_VAR_REF, graph=graph)
            document = build_identity_document(shape, scope=Scope.model_validate(scope))
            spec_id = recompute_identity(document).spec_id
            definition = {"schema_version": 1, "spec_id": spec_id, "identity": document}
            body = _spec_id_body(spec_id, definition=definition)
            pointer = "/input/definition/identity/entities/recipient"
        rows_before = len(store.state.spec_registry)

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "invalid_role_ref"
        assert problem["pointer"] == pointer
        assert problem["detail"].startswith(f"{dcid} is {kind}; a recipient is a country")
        assert len(store.state.spec_registry) == rows_before

    def test_a_slot_pinned_role_refuses_role_incompatible_with_no_registry_write(self):
        store = FakeDatasetStore()
        arcs: dict[str, dict[str, tuple[str, ...]]] = {
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
                "constraintProperties": ("recipient",),
                "recipient": ("country/KEN",),
            }
        }
        graph = _graph(arcs)
        client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
        identity = pinned_identity(stat_var=_STAT_VAR)
        spec_id = recompute_identity(identity).spec_id
        reg = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )
        assert reg.status_code == 200, reg.text
        rows_before = len(store.state.spec_registry)

        resp = client.post(
            "/api/datasets",
            json=_spec_id_body(spec_id, scope={"entities": {"recipient": ["country/KEN"]}}),
        )

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "role_incompatible"
        assert len(problem["issues"]) == 1
        assert problem["issues"][0]["pointer"] == "/input/scope/entities/recipient"
        assert problem["issues"][0]["member"] == "recipient"
        assert len(store.state.spec_registry) == rows_before


def test_round_trip_a_surgered_specs_id_recomputes_identically_from_its_stored_canonical_form():
    store = FakeDatasetStore()
    client = _client(artifact_store=FakeArtifactStore(), store=store)
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body(spec_id, scope={"window": {"kind": "range", "start_year": 2015}}),
    )

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    recomputed = recompute_identity(manifest["definition"]["identity"])
    assert recomputed.spec_id == manifest["definition"]["spec_id"]


# More than the 4,000-candidate cap, with a set-bound slot that prevents the
# truncated reverse-arc page from proving completeness, so the walk reads the whole set.
_CRS_GRANTS = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_CRS_LOANS = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance210_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)


def _wide_crs_identity() -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "grantEquivalent",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {
                "property": "crsTypeOfFinance",
                "binding": {
                    "state": "set",
                    "refs": ["CRSTypeOfFinance100", "CRSTypeOfFinance200"],
                },
            }
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def _wide_crs_graph() -> FakeMintGraphReader:
    """Adds two matching members and `CANDIDATE_CAP + 1` vintage variants to trigger page truncation."""
    member_arcs: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("FinancialFlow",),
        "measuredProperty": ("grantEquivalent",),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "observationProperties": ("financingProvider", "financingRecipient"),
    }
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        _CRS_GRANTS: {
            **member_arcs,
            "constraintProperties": ("crsTypeOfFinance",),
            "crsTypeOfFinance": ("CRSTypeOfFinance110",),
        },
        _CRS_LOANS: {
            **member_arcs,
            "constraintProperties": ("crsTypeOfFinance",),
            "crsTypeOfFinance": ("CRSTypeOfFinance210",),
        },
    }
    for i in range(CANDIDATE_CAP + 1):
        arcs[f"{_CRS_GRANTS}_Vintage{i:04d}"] = {
            **member_arcs,
            "constraintProperties": ("vintage",),
            "vintage": (f"Vintage{i:04d}",),
        }
    return FakeMintGraphReader(arcs=arcs)


_NO_INDEX_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _no_index_coverage() -> CodeIndex:
    """Forces member resolution through the injected graph."""
    return CodeIndex(stamp=_NO_INDEX_STAMP, declarations=(), codes=(), statvars=())


def test_three_identical_wide_crs_asks_converge_on_one_non_retryable_refusal():
    """Three asks whose whole set exceeds the subject ceiling return the same non-retryable
    candidate-window 422, which no cache holds."""
    ceiling = CANDIDATE_CAP + 1
    store = FakeDatasetStore()
    client = _client(
        artifact_store=FakeArtifactStore(),
        store=store,
        graph=_wide_crs_graph(),
        code_index=_no_index_coverage(),
        settings=DatasetsBuildSettings(member_subject_ceiling=ceiling),
    )
    identity = _wide_crs_identity()
    spec_id = recompute_identity(identity).spec_id
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": identity}
    body = _spec_id_body(spec_id, definition=definition)

    outcomes = []
    for _ in range(3):
        resp = client.post("/api/datasets", json=body)
        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["ceiling_kind"] == "candidate_window"
        assert problem["count"] == ceiling
        assert problem["cap"] == ceiling
        outcomes.append((problem["code"], problem["retryable"]))

    assert outcomes == [("over_ceiling", False)] * 3
