"""Tests collection-id minting, member sidecars, scope changes, and member-level refusals.
The composed app and materialiser are real. Graph, SDMX, and artifact storage are fakes.
"""

from __future__ import annotations

import httpx

from datasets_api.adapter import build_identity_document
from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from datasets_api.shape import resolve_shape
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry, mint_collection_id
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import described_identity, pinned_identity
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore, wire_definition_for
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_STAT_VAR2 = "AnotherFlowIndicator"
_STAT_VAR2_REF = f"dcid:{_STAT_VAR2}"
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

_FACETS_DICT = {
    "provenance": {"ref": "OECD_DAC2A"},
    "measurementMethod": {"ref": "GrantEquivalent"},
    "unit": {"ref": "USD"},
}


def _arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        _STAT_VAR: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
        },
        _STAT_VAR2: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("provider",),
            "measuredProperty": ("amount",),
        },
    }


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs=_arcs(),
        # The graph facet probe keys on the bare dcid.
        facets={
            (_STAT_VAR, "country/KEN"): [_FACET],
            (_STAT_VAR, "country/UGA"): [_FACET],
            (_STAT_VAR2, "country/KEN"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya", "country/UGA": "Uganda"},
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


def _coverage_service() -> FakeCoverageService:
    """Coverage matching `_FACET` for each StatVar and entity.
    `_STAT_VAR2` uses `provider` as donor.
    """
    index = build_coverage_index(
        [
            (_STAT_VAR, "facet1", "country/USA", "country/KEN", [2015]),
            (_STAT_VAR, "facet1", "country/USA", "country/UGA", [2015]),
            (_STAT_VAR2, "facet1", "country/KEN", "country/UGA", [2015]),
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


def _client(
    *,
    artifact_store,
    store: FakeDatasetStore | None = None,
    graph: FakeMintGraphReader | None = None,
    sdmx_requests: list[httpx.Request] | None = None,
):
    store = store if store is not None else FakeDatasetStore()
    requests = sdmx_requests if sdmx_requests is not None else []
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(requests))
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _graph(),
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _member_sidecar(
    stat_var_ref: str, *, entities: dict[str, list[str]], graph: FakeMintGraphReader
) -> tuple[str, dict]:
    """A spec_id and its own matching canonical-definition sidecar for one
    collection member, built the same way `kinds.statvar_set` builds one.
    """
    shape = resolve_shape(stat_var_ref, graph=graph)
    document = build_identity_document(shape, scope=Scope(entities=entities, facets=_FACETS_DICT))
    recomputed = recompute_identity(document)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": document,
    }


def _collection_body(
    collection_id: str, *, members: list[dict] | None = None, scope: dict | None = None
) -> dict:
    input_: dict = {"kind": "collection_id", "collection_id": collection_id}
    if members is not None:
        input_["members"] = members
    if scope is not None:
        input_["scope"] = scope
    return {"schema_version": 1, "input": input_, "options": {"bundle": False}}


def _register_collection(client, *definitions: dict) -> str:
    resp = client.post(
        "/api/datasets/register", json={"schema_version": 1, "members": list(definitions)}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["collection"]["collection_id"]


def test_registered_collection_materialises_every_member():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
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
    assert manifest["collection"]["collection_id"] == collection_id
    assert len(manifest["collection"]["members"]) == 2
    assert manifest["content_version"].startswith("cv_")
    # Bare mode returns one data file and one `series-statistics/1` artifact per member.
    assert len(manifest["artifacts"]) == 4
    assert len([a for a in manifest["artifacts"] if a["rendering"].get("statistics")]) == 2
    # The members are separate definitions, so the collection's trace states no one family.
    assert manifest["trace"]["family_member_digest"] is None
    assert manifest["trace"]["family_member_count"] is None


def test_a_repeat_mint_of_the_same_collection_is_idempotent():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)
    body = _collection_body(collection_id)
    rows_before = len(store.state.spec_registry)
    collections_before = len(store.state.collection_registry)

    first = client.post("/api/datasets", json=body)
    assert first.status_code == 200, first.text
    second = client.post("/api/datasets", json=body)
    assert second.status_code == 200, second.text

    assert second.json()["content_version"] == first.json()["content_version"]
    assert {a["artifact_digest"] for a in second.json()["artifacts"]} == {
        a["artifact_digest"] for a in first.json()["artifacts"]
    }
    assert len(store.state.spec_registry) == rows_before
    assert len(store.state.collection_registry) == collections_before


def test_unregistered_collection_with_no_sidecar_refuses_spec_not_found():
    client = _client(artifact_store=FakeArtifactStore())

    resp = client.post("/api/datasets", json=_collection_body("col_" + "0" * 32))

    assert resp.status_code == 404, resp.text
    problem = resp.json()
    assert problem["code"] == "spec_not_found"
    assert problem["pointer"] == "/input/collection_id"


def test_unregistered_collection_with_a_matching_sidecar_mints_all_rows_then_serves():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = mint_collection_id([spec_ken, spec_uga])

    resp = client.post(
        "/api/datasets", json=_collection_body(collection_id, members=[def_ken, def_uga])
    )

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["collection"]["collection_id"] == collection_id
    assert spec_ken in store.state.spec_registry
    assert spec_uga in store.state.spec_registry
    assert collection_id in store.state.collection_registry


def test_a_member_whose_hash_disagrees_with_its_own_spec_id_refuses_spec_id_mismatch():
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), graph=graph)
    spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    # The identity's scope changes while its claimed `spec_id` stays fixed.
    _other_spec, other_def = _member_sidecar(
        _STAT_VAR_REF,
        entities={"recipient": ["country/KEN", "country/UGA"]},
        graph=graph,
    )
    corrupted_ken = {**def_ken, "identity": other_def["identity"]}
    collection_id = mint_collection_id([spec_ken, spec_uga])

    resp = client.post(
        "/api/datasets",
        json=_collection_body(collection_id, members=[corrupted_ken, def_uga]),
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "spec_id_mismatch"


def test_a_collection_id_disagreeing_with_its_members_refuses_spec_id_mismatch():
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    wrong_collection_id = "col_" + "0" * 32

    resp = client.post(
        "/api/datasets",
        json=_collection_body(wrong_collection_id, members=[def_ken, def_uga]),
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "spec_id_mismatch"


def test_scope_surgery_moves_every_members_own_id():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)
    original_member_ids = {spec_ken, spec_uga}

    resp = client.post(
        "/api/datasets",
        json=_collection_body(collection_id, scope={"window": {"kind": "latest"}}),
    )

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["collection"]["collection_id"] != collection_id
    new_member_ids = {m["definition"]["spec_id"] for m in manifest["collection"]["members"]}
    assert new_member_ids.isdisjoint(original_member_ids)
    assert len(new_member_ids) == 2
    for member in manifest["collection"]["members"]:
        assert member["definition"]["identity"]["requested_window"] == {"kind": "latest"}


def test_convergence_across_members_deduplicates_the_collection():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)

    resp = client.post(
        "/api/datasets",
        json=_collection_body(collection_id, scope={"entities": {"recipient": ["country/KEN"]}}),
    )

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["collection"]["collection_id"] != collection_id
    assert len(manifest["collection"]["members"]) == 1
    assert manifest["collection"]["members"][0]["definition"]["spec_id"] == spec_ken
    assert any(w["code"] == "members_deduplicated" for w in manifest["warnings"])


def test_round_trip_a_surgered_collections_id_recomputes_identically_from_its_stored_canonical_form():
    """Scope surgery converges two members to one.
    The resulting collection id must recompute.
    """
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)

    resp = client.post(
        "/api/datasets",
        json=_collection_body(collection_id, scope={"entities": {"recipient": ["country/KEN"]}}),
    )

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    member_ids = []
    for member in manifest["collection"]["members"]:
        recomputed = recompute_identity(member["definition"]["identity"])
        assert recomputed.spec_id == member["definition"]["spec_id"]
        member_ids.append(recomputed.spec_id)
    assert mint_collection_id(member_ids) == manifest["collection"]["collection_id"]


def test_a_role_refusal_from_one_member_names_that_member():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    spec_other, def_other = _member_sidecar(
        _STAT_VAR2_REF, entities={"provider": ["country/KEN"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_other)

    resp = client.post(
        "/api/datasets",
        json=_collection_body(collection_id, scope={"entities": {"recipient": ["country/KEN"]}}),
    )

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "invalid_role_ref"
    assert len(problem["issues"]) == 1
    assert problem["issues"][0]["axis"] == "recipient"
    assert problem["issues"][0]["member"] == spec_other


def test_scope_restating_the_stored_state_exactly_serves_the_base_collection_id():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)
    rows_before = len(store.state.spec_registry)
    collections_before = len(store.state.collection_registry)

    resp = client.post(
        "/api/datasets",
        json=_collection_body(collection_id, scope={"window": {"kind": "all"}}),
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["collection"]["collection_id"] == collection_id
    assert len(store.state.spec_registry) == rows_before
    assert len(store.state.collection_registry) == collections_before


def test_a_registered_collection_with_a_matching_members_sidecar_still_serves():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)

    resp = client.post(
        "/api/datasets", json=_collection_body(collection_id, members=[def_ken, def_uga])
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["collection"]["collection_id"] == collection_id


def test_a_registered_collection_with_a_wrong_members_sidecar_refuses_spec_id_mismatch():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)
    # The valid member hash disagrees with the registered collection id.
    _spec_other, def_other = _member_sidecar(
        _STAT_VAR2_REF, entities={"provider": ["country/KEN"]}, graph=graph
    )

    resp = client.post(
        "/api/datasets", json=_collection_body(collection_id, members=[def_ken, def_other])
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "spec_id_mismatch"


def test_a_miss_with_sidecar_and_a_bad_scope_role_leaves_no_registry_row():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = mint_collection_id([spec_ken, spec_uga])

    resp = client.post(
        "/api/datasets",
        json=_collection_body(
            collection_id,
            members=[def_ken, def_uga],
            scope={"entities": {"donor": ["country/FRA"]}},
        ),
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "invalid_role_ref"
    assert store.state.spec_registry == {}
    assert store.state.collection_registry == {}


def test_a_malformed_member_ref_refuses_member_not_resolvable_not_a_500():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    identity = pinned_identity(stat_var="not-a-dcid")
    member_spec_id = recompute_identity(identity).spec_id
    collection_id = mint_collection_id([member_spec_id])
    reg = client.post(
        "/api/datasets/register",
        json={
            "schema_version": 1,
            "members": [wire_definition_for(spec_id=member_spec_id, identity=identity)],
        },
    )
    assert reg.status_code == 200, reg.text

    resp = client.post("/api/datasets", json=_collection_body(collection_id))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "member_not_resolvable"
    assert problem["issues"][0]["pointer"] == "/input/collection_id"


def test_a_malformed_member_identity_refuses_request_invalid_not_a_500():
    """An invalid identity fails before shape resolution on both registry paths."""
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    collection_id = "col_" + "0" * 32
    member = {"schema_version": 1, "spec_id": "spec_" + "1" * 32, "identity": {"slots": 1}}

    resp = client.post("/api/datasets", json=_collection_body(collection_id, members=[member]))

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["issues"][0]["pointer"] == "/input/collection_id"
    assert problem["issues"][0]["member"] == member["spec_id"]
    assert store.state.spec_registry == {}
    assert store.state.collection_registry == {}


def test_a_miss_with_a_malformed_member_ref_and_no_scope_leaves_no_registry_row():
    """A registry miss with an unresolvable member leaves both registries empty."""
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    identity = pinned_identity(stat_var="not-a-dcid")
    member_spec_id = recompute_identity(identity).spec_id
    collection_id = mint_collection_id([member_spec_id])
    member = wire_definition_for(spec_id=member_spec_id, identity=identity)

    resp = client.post("/api/datasets", json=_collection_body(collection_id, members=[member]))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "member_not_resolvable"
    assert problem["issues"][0]["pointer"] == "/input/collection_id"
    assert store.state.spec_registry == {}
    assert store.state.collection_registry == {}


def test_a_miss_with_a_breadth_only_member_and_no_scope_leaves_no_registry_row():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    identity = described_identity()
    member_spec_id = recompute_identity(identity).spec_id
    collection_id = mint_collection_id([member_spec_id])
    member = wire_definition_for(spec_id=member_spec_id, identity=identity)

    resp = client.post("/api/datasets", json=_collection_body(collection_id, members=[member]))

    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "spec_not_exportable"
    assert store.state.spec_registry == {}
    assert store.state.collection_registry == {}


def test_a_miss_with_a_multi_member_described_arm_member_refuses_over_ceiling():
    """A collection member must resolve to one StatVar.
    A described arm yielding two refuses `over_ceiling`.
    """
    store = FakeDatasetStore()
    member_arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("recipient",),
        "constraintProperties": ("DevelopmentFinanceScheme",),
    }
    graph = FakeMintGraphReader(
        arcs={
            "ODAGrantsAmount": {**member_arcs, "DevelopmentFinanceScheme": ("ODAGrants",)},
            "ODALoansAmount": {**member_arcs, "DevelopmentFinanceScheme": ("ODALoans",)},
        }
    )
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
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
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    member_spec_id = recompute_identity(identity).spec_id
    collection_id = mint_collection_id([member_spec_id])
    member = wire_definition_for(spec_id=member_spec_id, identity=identity)

    resp = client.post("/api/datasets", json=_collection_body(collection_id, members=[member]))

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "over_ceiling"
    assert store.state.spec_registry == {}
    assert store.state.collection_registry == {}


def test_a_member_that_does_not_reproduce_against_the_graph_leaves_no_registry_row():
    """A self-consistent member identity with a stale role fails reproducibility before any write."""
    store = FakeDatasetStore()
    graph = _graph()
    graph._arcs = {
        **graph._arcs,
        # `_STAT_VAR` declares `provider`. The stale identity binds `recipient`.
        _STAT_VAR: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("provider",),
            "measuredProperty": ("amount",),
        },
    }
    client = _client(artifact_store=FakeArtifactStore(), store=store, graph=graph)

    stale_identity = pinned_identity(stat_var=_STAT_VAR, entities={"recipient": ["country/KEN"]})
    stale_spec_id = recompute_identity(stale_identity).spec_id
    stale_member = wire_definition_for(spec_id=stale_spec_id, identity=stale_identity)
    good_spec_id, good_member = _member_sidecar(
        _STAT_VAR2_REF, entities={"provider": ["country/KEN"]}, graph=graph
    )
    collection_id = mint_collection_id([stale_spec_id, good_spec_id])

    resp = client.post(
        "/api/datasets",
        json=_collection_body(collection_id, members=[stale_member, good_member]),
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "definition_not_reproducible"
    assert store.state.spec_registry == {}
    assert store.state.collection_registry == {}


def test_empty_members_list_refuses_at_the_wire():
    client = _client(artifact_store=FakeArtifactStore())

    resp = client.post("/api/datasets", json=_collection_body("col_" + "0" * 32, members=[]))

    assert resp.status_code == 400, resp.text
