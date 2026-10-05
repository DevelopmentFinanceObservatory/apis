"""Tests `source_codes` graph lookup, optional source scoping, and its
`code_ambiguous`, `code_not_found`, and `request_invalid` refusals.
"""

from __future__ import annotations

import httpx

from datasets_api.revision import SentinelRevisionReader
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
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
_CSV_ROW = (
    f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
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


def _sdmx_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_CSV_HEADER + _CSV_ROW,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _coverage_service() -> FakeCoverageService:
    """Builds one 2015 coverage facet for `_STAT_VAR`."""
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", "country/USA", "country/KEN", [2015])],
        role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(*, graph_arcs: dict) -> object:
    from dc_sdmx import SdmxClient

    store = FakeDatasetStore()
    graph = FakeMintGraphReader(
        arcs=graph_arcs,
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "dcid:a": "Candidate A", "dcid:b": "Candidate B"},
    )
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport())
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


_BODY_TEMPLATE = {
    "schema_version": 1,
    "input": {
        "kind": "source_codes",
        "codes": ["CODE1"],
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


def test_bare_code_resolves_and_mints_like_statvar_set():
    client = _client(
        graph_arcs={
            "CODE1": {"member": (_STAT_VAR_REF,)},
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
        }
    )

    resp = client.post("/api/datasets", json=_BODY_TEMPLATE)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    identity = manifest["definition"]["identity"]
    assert identity["shape"] == {"kind": "stat_var", "id": _STAT_VAR}
    assert identity["slots"] == []


def test_code_scoped_by_source_resolves_a_different_lookup_key():
    body = {**_BODY_TEMPLATE, "input": {**_BODY_TEMPLATE["input"], "source": "OECD"}}
    client = _client(
        graph_arcs={
            "OECD:CODE1": {"member": (_STAT_VAR_REF,)},
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
        }
    )

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text


def test_code_matching_nothing_refuses_code_not_found():
    client = _client(graph_arcs={})

    resp = client.post("/api/datasets", json=_BODY_TEMPLATE)

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "code_not_found"


def test_code_matching_more_than_one_statvar_refuses_code_ambiguous_with_candidates():
    client = _client(graph_arcs={"CODE1": {"member": ("dcid:a", "dcid:b")}})

    resp = client.post("/api/datasets", json=_BODY_TEMPLATE)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "code_ambiguous"
    assert len(problem["issues"]) == 1
    issue = problem["issues"][0]
    assert issue["member"] == "CODE1"
    assert {alt["ref"] for alt in issue["alternatives"]} == {"dcid:a", "dcid:b"}


def test_a_dcid_prefixed_code_is_request_invalid():
    client = _client(graph_arcs={})
    body = {
        "schema_version": 1,
        "input": {"kind": "source_codes", "codes": ["dcid:not-a-real-code"]},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"


def test_source_codes_shares_scope_entities_validation_with_statvar_set():
    """`source_codes.scope` is the same `datasets_api.mint_models.Scope` as
    `statvar_set.scope` -- an entity role bound to no values refuses
    `request_invalid` here too, before the door ever resolves a code or
    touches the graph/SDMX transports.
    """
    client = _client(graph_arcs={})
    body = {
        "schema_version": 1,
        "input": {
            "kind": "source_codes",
            "codes": ["CODE1"],
            "scope": {"entities": {"recipient": []}},
        },
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/scope"
