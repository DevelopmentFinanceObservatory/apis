"""The composed mint and definition routes report the injected `CodeIndex` consistently.
Described-arm tests also pin member enumeration to that same index.
"""

from __future__ import annotations

import httpx

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import CodeSystemDeclaration, IndexStamp
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
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
_RECIPIENT = "country/KEN"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
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
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}

_MINT_BODY = {
    "schema_version": 1,
    "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
    "options": {"bundle": False},
}


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, _RECIPIENT): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


def _fixture_code_index(*, code_system: str = "syntheticAxis") -> CodeIndex:
    """Builds an index with a code system absent from the packaged singleton."""
    assert code_system not in get_index().declarations
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-09-15T00:00:00Z",
            code_counts={code_system: 0},
            statvar_shape_counts={},
            content_hash="sha256:fixture",
        ),
        declarations=[
            CodeSystemDeclaration(
                code_system=code_system,
                hierarchical=False,
                disjoint_from_children=True,
                summable=True,
                absent_reading="not_declared",
                plain_name="synthetic axis",
            )
        ],
        codes=[],
        statvars=[],
    )


def _coverage_service(*, stat_vars: tuple[str, ...] = (_STAT_VAR,)) -> FakeCoverageService:
    """Builds one `_RECIPIENT` coverage facet per StatVar, matching the 2020 CSV fixtures."""
    index = build_coverage_index(
        [(stat_var, "facet1", ABSENT_ENTITY, _RECIPIENT, [2020]) for stat_var in stat_vars],
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


def _client(*, code_index: CodeIndex, store: FakeDatasetStore | None = None):
    store = store if store is not None else FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(),
        sdmx=SdmxClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text=_CSV_BODY,
                    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                )
            ),
        ),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        code_index=code_index,
        coverage=_coverage_service(),
    ), store


def test_mint_response_reports_the_injected_code_indexs_own_declarations():
    fixture = _fixture_code_index()
    client, _store = _client(code_index=fixture)

    resp = client.post("/api/datasets", json=_MINT_BODY)

    assert resp.status_code == 200, resp.text
    axes = resp.json()["output_options"]["groupby"]["axes"]
    assert axes == [{"name": "syntheticAxis"}]


def test_definition_retrieval_reports_the_same_injected_code_index_the_mint_used():
    fixture = _fixture_code_index()
    store = FakeDatasetStore()
    client, _store = _client(code_index=fixture, store=store)
    minted = client.post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]

    # A fresh client exercises the definition door's index resolution.
    retrieving_client, _store = _client(code_index=fixture, store=store)

    resp = retrieving_client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    axes = resp.json()["output_options"]["groupby"]["axes"]
    assert axes == [{"name": "syntheticAxis"}]


def test_definition_options_route_reports_the_same_injected_code_index():
    # `/options` runs the definition door's write-free prewrite path.
    fixture = _fixture_code_index()
    store = FakeDatasetStore()
    client, _store = _client(code_index=fixture, store=store)
    minted = client.post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]

    resp = client.get(f"/api/datasets/{spec_id}/options")

    assert resp.status_code == 200, resp.text
    axes = resp.json()["output_options"]["groupby"]["axes"]
    assert axes == [{"name": "syntheticAxis"}]


def test_two_clients_injecting_different_code_indexes_report_their_own():
    # Distinct fixtures prove each client reports its own index.
    first = _fixture_code_index(code_system="syntheticAxisOne")
    second = _fixture_code_index(code_system="syntheticAxisTwo")
    first_client, _store = _client(code_index=first)
    second_client, _store = _client(code_index=second)

    first_resp = first_client.post("/api/datasets", json=_MINT_BODY)
    second_resp = second_client.post("/api/datasets", json=_MINT_BODY)

    assert first_resp.status_code == 200, first_resp.text
    assert second_resp.status_code == 200, second_resp.text
    assert first_resp.json()["output_options"]["groupby"]["axes"] == [{"name": "syntheticAxisOne"}]
    assert second_resp.json()["output_options"]["groupby"]["axes"] == [{"name": "syntheticAxisTwo"}]


# Enumeration uses the injected index's `absent_reading`. The default "aggregate" lets the
# unconstrained member dominate. `_modality_code_index` uses "not_declared", so both survive.

_MODALITY_PROPERTY = "syntheticModality"
_GRANTS_TOTAL = "GrantsTotalDisbursements"
_GRANTS_TOTAL_REF = f"dcid:{_GRANTS_TOTAL}"
_GRANTS_BY_MODALITY = "GrantsByModalityDisbursements"
_GRANTS_BY_MODALITY_REF = f"dcid:{_GRANTS_BY_MODALITY}"

_DESCRIBED_CSV_ROWS = {
    _GRANTS_TOTAL: (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_GRANTS_TOTAL},{_RECIPIENT},USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2020,500,,facet1\r\n"
    ),
    _GRANTS_BY_MODALITY: (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_GRANTS_BY_MODALITY},{_RECIPIENT},USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2020,500,,facet1\r\n"
    ),
}


def _described_two_member_graph() -> FakeMintGraphReader:
    member_arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("recipient",),
    }
    return FakeMintGraphReader(
        arcs={
            _GRANTS_TOTAL: member_arcs,
            _GRANTS_BY_MODALITY: {
                **member_arcs,
                "constraintProperties": (_MODALITY_PROPERTY,),
                _MODALITY_PROPERTY: ("Grant",),
            },
        },
        facets={
            (_GRANTS_TOTAL, _RECIPIENT): [_FACET],
            (_GRANTS_BY_MODALITY, _RECIPIENT): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


def _modality_code_index() -> CodeIndex:
    assert _MODALITY_PROPERTY not in get_index().declarations
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-09-15T00:00:00Z",
            code_counts={_MODALITY_PROPERTY: 0},
            statvar_shape_counts={},
            content_hash="sha256:fixture-modality",
        ),
        declarations=[
            CodeSystemDeclaration(
                code_system=_MODALITY_PROPERTY,
                hierarchical=False,
                disjoint_from_children=True,
                summable=True,
                absent_reading="not_declared",
                plain_name="synthetic modality",
            )
        ],
        codes=[],
        statvars=[],
    )


def _described_transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        requested = request.url.params.get("c[variableMeasured]", "").split(",")
        body = _CSV_HEADER + "".join(
            _DESCRIBED_CSV_ROWS[stat_var]
            for stat_var in requested
            if stat_var in _DESCRIBED_CSV_ROWS
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return httpx.MockTransport(handler)


def _described_ask_identity() -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "amount",
        },
        "slots": [{"property": _MODALITY_PROPERTY, "binding": {"state": "unbound"}}],
        "entities": {"recipient": [_RECIPIENT]},
        "facets": {
            "provenance": {"state": "value", "ref": "OECD_DAC2A"},
            "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
            "unit": {"state": "value", "ref": "USD"},
        },
        "requested_window": {"kind": "all"},
    }


def _described_mint_body(identity: dict, *, spec_id: str) -> dict:
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": identity}
    return {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": spec_id, "definition": definition},
        "options": {"bundle": False},
    }


def _described_client(
    *, code_index: CodeIndex | None = None, store: FakeDatasetStore | None = None
):
    store = store if store is not None else FakeDatasetStore()
    kwargs: dict[str, object] = {}
    if code_index is not None:
        kwargs["code_index"] = code_index
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_described_two_member_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_described_transport([])),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(stat_vars=(_GRANTS_TOTAL, _GRANTS_BY_MODALITY)),
        **kwargs,
    )
    return client, store


def test_described_ask_mint_enumerates_using_the_injected_code_indexs_own_readings():
    identity = _described_ask_identity()
    spec_id = recompute_identity(identity).spec_id
    client, _store = _described_client(code_index=_modality_code_index())

    resp = client.post("/api/datasets", json=_described_mint_body(identity, spec_id=spec_id))

    assert resp.status_code == 200, resp.text
    resolved_refs = {entry["ref"] for entry in resp.json()["trace"]["resolved_stat_vars"]}
    assert resolved_refs == {_GRANTS_TOTAL_REF, _GRANTS_BY_MODALITY_REF}


def test_described_ask_mint_against_the_real_singleton_enumerates_only_the_dominating_member():
    # The packaged index's default "aggregate" reading keeps only the unconstrained member.
    identity = _described_ask_identity()
    spec_id = recompute_identity(identity).spec_id
    client, _store = _described_client()

    resp = client.post("/api/datasets", json=_described_mint_body(identity, spec_id=spec_id))

    assert resp.status_code == 200, resp.text
    resolved_refs = {entry["ref"] for entry in resp.json()["trace"]["resolved_stat_vars"]}
    assert resolved_refs == {_GRANTS_TOTAL_REF}


def test_described_ask_definition_retrieval_re_enumerates_with_the_same_injected_index():
    # Retrieval must re-enumerate both members with the injected index.
    identity = _described_ask_identity()
    spec_id = recompute_identity(identity).spec_id
    fixture = _modality_code_index()
    store = FakeDatasetStore()
    client, _store = _described_client(code_index=fixture, store=store)
    minted = client.post("/api/datasets", json=_described_mint_body(identity, spec_id=spec_id))
    assert minted.status_code == 200, minted.text

    retrieving_client, _store = _described_client(code_index=fixture, store=store)
    resp = retrieving_client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    resolved_refs = {entry["ref"] for entry in resp.json()["trace"]["resolved_stat_vars"]}
    assert resolved_refs == {_GRANTS_TOTAL_REF, _GRANTS_BY_MODALITY_REF}
