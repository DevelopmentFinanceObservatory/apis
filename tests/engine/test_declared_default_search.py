"""A StatVar whose graph declares a default facet is answered under that default: the answer's
read plan elects the declared value where a candidate carries it, as a mint of its Spec
does (`retrieval.facet_policy`, `read_plan.in_elected_datasets`), and nothing on the Search
path refuses it."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from dc_graph import FacetDefaultPolicy
from materialise.facets import FACET_AXES
from promptlab.graphcache import GraphCache
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.errors import GraphInfraError
from search.engine.region.pipeline import resolve_variable
from series_coverage.schema import FacetMetadata
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine import test_measure_election_inapplicable_slots as measure_election
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import PINNED_DATE, door_roles_for_test_query
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.fixtures.record import _RecordingGraph
from tests.materialise.conftest import FakeGraphReader

_FERTILITY = "FertilityRate_Person_Female"
_POLICY = FacetDefaultPolicy(
    resolved_ref="Percent",
    policy_node="dc/policy/fertility-unit",
    policy_version="1",
    selection_reason="declared",
)
_RECORDED = "Recorded"
_ARC_POLICY = FacetDefaultPolicy(
    resolved_ref=_RECORDED,
    policy_node="defaultMeasurementMethod",
    policy_version="1",
    selection_reason="graph_declared",
)


def _declaring_graph(**kwargs) -> FakeGraph:
    return FakeGraph(default_methods={_FERTILITY: _RECORDED}, **kwargs)


def _post(graph: FakeGraph, body: dict):
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = None
        return TestClient(app, raise_server_exceptions=False).post("/api/search/resolve", json=body)


def test_a_resolve_over_a_declared_default_answers_definite() -> None:
    """The graph declares a default method for the fertility rate, and the resolve answers
    as it does with no declaration."""
    body = {"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}}

    resp = _post(_declaring_graph(), body)

    assert resp.status_code == 200
    assert resp.json()["status"] == "definite"


_CRS = "provenance/OECDCreditorReportingSystem"
_ONE_WEIGHTING = "ONEClimateFinanceWeighting"
_FULL_VALUE = "OECDClimateFinanceFullValue"


def _resolve_declaring(monkeypatch, declared: str | None, *, unread: bool = False):
    """The measure-elected grant-equivalent answer to Kenya, whose CRS member holds its flows
    from the United States under two methods, the full value from 2010 and ONE Data's
    weighting from 2018, and declares `declared` its default method in the graph. With
    `unread` the graph fails the read of that declaration."""
    sv = measure_election._EXTRA_SV
    index = patch_coverage(
        monkeypatch,
        [
            (sv, facet_id, "country/USA", measure_election._KEN_DCID, list(range(first, 2021)))
            for facet_id, first in (("fFull", 2010), ("fWeighting", 2018))
        ],
        role_names={sv: measure_election._ROLE_NAMES},
        facet_metadata={
            facet_id: FacetMetadata(
                provenance_id=_CRS,
                measurement_method=method,
                unit="USDollar",
                observation_period="P1Y",
            )
            for facet_id, method in (("fFull", _FULL_VALUE), ("fWeighting", _ONE_WEIGHTING))
        },
    )
    graph = FakeGraph(
        nodes=measure_election._NODES,
        detect=measure_election._DETECT,
        resolve={"Kenya": measure_election._KEN_DCID},
        obs={},
        default_methods={} if declared is None else {sv: declared},
    )
    if unread:

        def failing_declaration_read(stat_vars, *, deadline=None):
            raise GraphInfraError("simulated transport error")

        monkeypatch.setattr(graph, "default_facet_policies", failing_declaration_read)
    query = measure_election._QUERY
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            entity_roles=door_roles_for_test_query(query, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations=measure_election._OBS, availability={}),
            llm=measure_election._BindPurposeToExtraLLM(),
            base_timing={},
            coverage=FakeCoverageService(index).snapshot(),
        )
    )


@pytest.mark.parametrize(
    ("declared", "basis", "ref"),
    [
        pytest.param(_ONE_WEIGHTING, "declared_default", _ONE_WEIGHTING, id="declared"),
        pytest.param("NoSuchMethod", "fallback", _FULL_VALUE, id="declared-missing"),
        pytest.param(None, "fallback", _FULL_VALUE, id="undeclared"),
    ],
)
def test_a_role_answer_serves_the_method_its_member_declares(
    monkeypatch, declared: str | None, basis: str, ref: str
) -> None:
    """Ranking alone serves the full value, which holds more years. Declaring ONE Data's
    weighting, the answer serves it and names the declaration as its basis; declaring a
    method no facet carries, it serves the ranking's one method."""
    result = _resolve_declaring(monkeypatch, declared)

    assert result.status == "definite"
    method = result.specs[0].resolved.facets["measurementMethod"]
    assert (method.basis, method.ref) == (basis, ref)


def test_a_declaration_the_graph_fails_to_read_fails_the_resolve(monkeypatch) -> None:
    """The graph fails the read of the member's declaration. The resolve raises the graph's
    error: an answer under the ranking's method would name a Spec whose mint, reading the
    declaration once the graph answers, serves ONE Data's weighting."""
    with pytest.raises(GraphInfraError):
        _resolve_declaring(monkeypatch, _ONE_WEIGHTING, unread=True)


# The declared method's facet ends in 2000; another provenance's Estimated projection
# covers 2020 to 2032, so 2027 holds a projected point only.
_RECORDED_AND_PROJECTED = {
    f"{_FERTILITY}|country/KEN": [
        {
            "facetId": "recorded",
            "provenanceId": "dc/base/RecordedSource",
            "measurementMethod": _RECORDED,
            "earliestDate": "1960",
            "latestDate": "2000",
            "obsCount": 41,
            "dates": [str(y) for y in range(1960, 2001)],
            "values": [5.0] * 41,
        },
        {
            "facetId": "projected",
            "provenanceId": "dc/base/ProjectionSource",
            "measurementMethod": "Estimated",
            "earliestDate": "2020",
            "latestDate": "2032",
            "obsCount": 13,
            "dates": [str(y) for y in range(2020, 2033)],
            "values": [4.0] * 13,
        },
    ]
}
_FERTILITY_IN_2027 = {
    "input": {
        "kind": "parsed",
        "parts": [
            {
                "variable_text": "fertility rate",
                "entities": [{"text": "Kenya"}],
                "time_text": "2027",
            }
        ],
    },
    "response_options": {},
}


def test_a_2027_request_over_a_declared_recorded_default_never_serves_the_projection() -> None:
    """Undeclared, 2027 elects the other provenance's Estimated projection, the one facet
    with a point that year. Declared, the variable's default method is the recorded facet's,
    ending in 2000, which a mint serves whatever the window, so the answer holds no 2027
    point and answers no data: no coverage or preview names the projection."""
    control = _post(FakeGraph(obs=_RECORDED_AND_PROJECTED), _FERTILITY_IN_2027)
    assert control.status_code == 200
    assert "ProjectionSource" in control.text

    resp = _post(_declaring_graph(obs=_RECORDED_AND_PROJECTED), _FERTILITY_IN_2027)

    assert resp.status_code == 200
    assert resp.json()["parts"][0]["status"] == "no_data"
    assert "ProjectionSource" not in resp.text


def test_the_live_client_passes_the_graphs_policy_through(monkeypatch) -> None:
    """The live client reports what its graph client declares."""
    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(client._client, "default_facet_policy", lambda **_: _POLICY)
    assert client.default_facet_policy(stat_var="dcid:Count_Person", axis="unit") == _POLICY


def _stat_vars_and_axes() -> list[tuple[str, str]]:
    return [
        (f"dcid:{sv}", axis)
        for sv in (_FERTILITY, "Count_Person", "Unrelated_StatVar")
        for axis in FACET_AXES
    ]


def test_a_graph_declaring_no_default_returns_no_policy_and_one_declaring_returns_it(
    tmp_path: Path,
) -> None:
    """Promptlab's graph cache, the recording proxy and every fake built without policies
    return no policy; the same wrappers over a graph given a
    policy, and the fakes given one, return it. The live clients read the graph's
    `defaultMeasurementMethod` arc (`tests/origin_catalogue/test_graph.py`,
    `tests/dc_graph/test_client_default_facet_policy.py`)."""
    silent = [
        GraphCache(FakeGraph(), path=tmp_path / "silent.sqlite"),
        _RecordingGraph(FakeGraph(), nodes={}, obs={}, detect={}, resolve={}),
        FakeGraph(),
        FakeGraph(raise_on_call=True),
        FakeGraphReader(),
        FakeMintGraphReader(arcs={}),
    ]
    for graph in silent:
        for stat_var, axis in _stat_vars_and_axes():
            assert graph.default_facet_policy(stat_var=stat_var, axis=axis) is None, graph

    declaring = [
        (_declaring_graph(), "measurementMethod", _ARC_POLICY),
        (
            GraphCache(_declaring_graph(), path=tmp_path / "declaring.sqlite"),
            "measurementMethod",
            _ARC_POLICY,
        ),
        (
            _RecordingGraph(_declaring_graph(), nodes={}, obs={}, detect={}, resolve={}),
            "measurementMethod",
            _ARC_POLICY,
        ),
        (FakeGraphReader(policies={(f"dcid:{_FERTILITY}", "unit"): _POLICY}), "unit", _POLICY),
        (
            FakeMintGraphReader(arcs={}, policies={(f"dcid:{_FERTILITY}", "unit"): _POLICY}),
            "unit",
            _POLICY,
        ),
    ]
    for graph, axis, policy in declaring:
        assert graph.default_facet_policy(stat_var=f"dcid:{_FERTILITY}", axis=axis) == policy
