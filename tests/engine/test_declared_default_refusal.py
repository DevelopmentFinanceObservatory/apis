"""A StatVar whose graph declares a default facet is refused at resolve time, before any
coverage, headline or sentence is built, as the registered `facet_binding_unsupported`
refusal (501, not retryable). No fan-out leg or UN pool election degrades it into a
no-data answer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from dc_graph import FacetDefaultPolicy
from materialise.facets import FACET_AXES
from promptlab.graphcache import GraphCache
from promptlab.report import _NoLiveGraph
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.errors import DeclaredDefaultUnsupportedError, EngineInfraError
from search.engine.events import SentenceEvent
from search.engine.region.pipeline import resolve_variable
from search.engine.retrieval.facet_policy import refuse_declared_defaults
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine import test_candidates_fold_after_unit_filter as fold
from tests.engine import test_measure_election_inapplicable_slots as measure_election
from tests.engine import test_pool_election as pools
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


# A policy on a StatVar no test answer reads, so no answer here is refused for it.
_UNRELATED_POLICY = {("dcid:Unrelated_StatVar", "unit"): _POLICY}


def _declaring_graph() -> FakeGraph:
    return FakeGraph(facet_policies={(f"dcid:{_FERTILITY}", "unit"): _POLICY})


def _client(graph: FakeGraph) -> TestClient:
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    return TestClient(app, raise_server_exceptions=False)


def _post(client: TestClient, body: dict, **kwargs):
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = None
        return client.post("/api/search/resolve", json=body, **kwargs)


def test_the_refusal_is_a_non_retryable_facet_refusal() -> None:
    assert not issubclass(DeclaredDefaultUnsupportedError, EngineInfraError)
    refusal = DeclaredDefaultUnsupportedError("declared")
    assert (refusal.code, refusal.status) == ("facet_binding_unsupported", 501)
    with pytest.raises(DeclaredDefaultUnsupportedError, match="Percent"):
        refuse_declared_defaults(_declaring_graph(), [_FERTILITY])
    refuse_declared_defaults(FakeGraph(), [_FERTILITY])


def _assert_refused(resp) -> None:
    assert resp.status_code == 501
    body = resp.json()
    assert (body["code"], body["retryable"]) == ("facet_binding_unsupported", False)
    assert "Percent" in body["detail"]


def test_a_resolve_over_a_declared_default_is_refused() -> None:
    client = _client(_declaring_graph())
    control = _post(
        _client(FakeGraph()),
        {"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}},
    )
    assert control.status_code == 200 and control.json()["status"] == "definite"

    resp = _post(client, {"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}})

    _assert_refused(resp)


def test_a_dataset_entry_over_a_declared_default_is_not_degraded_to_no_data() -> None:
    """Dataset entries fan out with `degrade_on_error`, which turns a leg's ordinary error
    into a placeholder `no_data` part. The refusal fails the request instead."""
    body = {
        "input": {
            "kind": "dataset_resubmit",
            "entries": [
                {
                    "entry_kind": "parsed",
                    "part": {
                        "variable_text": "fertility rate",
                        "entities": [{"text": "Kenya"}],
                    },
                }
            ],
        },
        "response_options": {"include_observations": True},
    }
    control = _post(_client(FakeGraph()), body)
    assert control.status_code == 200 and control.json()["parts"][0]["status"] == "definite"

    resp = _post(_client(_declaring_graph()), body)

    _assert_refused(resp)


def test_a_streamed_resolve_refuses_before_any_sentence() -> None:
    resp = _post(
        _client(_declaring_graph()),
        {"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}},
        headers={"Accept": "text/event-stream"},
    )

    events = [
        line.removeprefix("event: ").strip()
        for line in resp.text.splitlines()
        if line.startswith("event: ")
    ]
    assert events[-1] == "error"
    assert "sentence" not in events
    assert "response" not in events
    error = json.loads(
        [line for line in resp.text.splitlines() if line.startswith("data: ")][-1].removeprefix(
            "data: "
        )
    )
    assert (error["code"], error["retryable"]) == ("facet_binding_unsupported", False)


def test_the_live_client_passes_the_graphs_policy_through(monkeypatch) -> None:
    """The live client reports what the graph declares and leaves the refusal to resolve."""
    client = LiveGraphClient(base="http://fake")
    assert client.default_facet_policy(stat_var="dcid:Count_Person", axis="unit") is None
    monkeypatch.setattr(client._client, "default_facet_policy", lambda **_: _POLICY)
    assert client.default_facet_policy(stat_var="dcid:Count_Person", axis="unit") == _POLICY


def test_the_standard_no_data_answer_also_refuses() -> None:
    """With no observation at the place, the standard path still refuses a declared default
    rather than answering no data, as the role path refuses before confirmation."""
    graph = FakeGraph(obs={}, facet_policies={(f"dcid:{_FERTILITY}", "unit"): _POLICY})

    _assert_refused(
        _post(_client(graph), {"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}})
    )


def _measure_elected_stream(monkeypatch, events: list, *, facet_policies: dict) -> None:
    """Stream the measure-elected grant-equivalent resolve into `events`."""
    index = patch_coverage(
        monkeypatch,
        measure_election._COVERAGE_ROWS,
        role_names={measure_election._EXTRA_SV: measure_election._ROLE_NAMES},
    )
    graph = FakeGraph(
        nodes=measure_election._NODES,
        detect=measure_election._DETECT,
        resolve={"Kenya": measure_election._KEN_DCID},
        obs={},
        facet_policies=facet_policies,
    )

    async def _emit(event) -> None:
        events.append(event)

    asyncio.run(
        resolve_variable(
            measure_election._QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=measure_election._QUERY,
            role_query=measure_election._QUERY,
            entity_roles=door_roles_for_test_query(measure_election._QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations=measure_election._OBS, availability={}),
            llm=measure_election._BindPurposeToExtraLLM(),
            base_timing={},
            coverage=FakeCoverageService(index).snapshot(),
            emit=_emit,
        )
    )


def test_a_measure_elected_stream_emits_no_sentence_before_the_refusal(monkeypatch) -> None:
    """The pipeline emits no sentence of its own: the part's one sentence follows its final
    result (`fanout._stream_one_variable`), so a refusal reaches the client before any."""
    control: list = []
    _measure_elected_stream(monkeypatch, control, facet_policies={})
    assert control, "the stream emits its shape and step events"
    assert not [e for e in control if isinstance(e, SentenceEvent)]

    events: list = []
    with pytest.raises(DeclaredDefaultUnsupportedError):
        _measure_elected_stream(
            monkeypatch,
            events,
            facet_policies={(f"dcid:{measure_election._EXTRA_SV}", "unit"): _POLICY},
        )
    assert events, "the stream emits its shape and step events before materialise"
    assert not [e for e in events if isinstance(e, SentenceEvent)]


# ---------------------------------------------------------------------------
# A refusal raised by a later step, over a graph that declares default facets: the
# pipeline streams no sentence, so none reaches the stream ahead of the refusal.
# ---------------------------------------------------------------------------


def test_a_refusal_in_the_candidates_fan_out_streams_no_sentence(monkeypatch) -> None:
    """The candidates fan-out of the measure-elected grant-equivalent answer materialises the
    plausible amount sibling, whose member declares a default: the request fails and the
    stream carries no sentence."""
    index = patch_coverage(
        monkeypatch,
        [
            (fold._EXTRA_SV, "f1", "country/USA", fold._KEN_DCID, [2018, 2019, 2020]),
            (fold._ALT_SV, "f2", "country/USA", fold._KEN_DCID, [2020, 2021]),
        ],
        role_names={fold._EXTRA_SV: fold._ROLE_NAMES, fold._ALT_SV: fold._ROLE_NAMES},
    )

    def stream(facet_policies: dict, events: list) -> None:
        graph = FakeGraph(
            nodes=fold._NODES,
            detect=fold._DETECT,
            resolve={"Kenya": fold._KEN_DCID},
            obs={},
            facet_policies=facet_policies,
        )

        async def _emit(event) -> None:
            events.append(event)

        asyncio.run(
            resolve_variable(
                fold._QUERY,
                entities=["Kenya"],
                date_request=None,
                detect_query=fold._QUERY,
                role_query=fold._QUERY,
                entity_roles=door_roles_for_test_query(fold._QUERY, ["Kenya"]),
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations=fold._SDMX_OBS, availability={}),
                llm=fold._ElectGrantEquivalentAlsoPlausibleAmountLLM(),
                base_timing={},
                emit=_emit,
                coverage=FakeCoverageService(index).snapshot(),
            )
        )

    control: list = []
    stream({}, control)
    assert control and not [e for e in control if isinstance(e, SentenceEvent)]
    events: list = []
    with pytest.raises(DeclaredDefaultUnsupportedError, match=fold._ALT_SV):
        stream({(f"dcid:{fold._ALT_SV}", "unit"): _POLICY}, events)
    assert events, "the stream carries shape and step events before the refusal"
    assert not [e for e in events if isinstance(e, SentenceEvent)]


def test_a_refusal_on_an_elected_un_variable_streams_no_sentence() -> None:
    """The part elects a UN variable declaring a default and refuses, so the stream carries
    no sentence."""
    graph = pools._graph(facet_policies={(f"dcid:{pools._UN}", "unit"): _POLICY})
    events: list = []

    async def collect(event) -> None:
        events.append(event)

    with pytest.raises(DeclaredDefaultUnsupportedError, match=pools._UN):
        pools._resolve(graph, emit=collect, llm=pools._ElectLLM(pools._UN))
    assert not [e for e in events if isinstance(e, SentenceEvent)]


# The declared provenance's facet is recorded and ends in 2000; another provenance's
# Estimated projection covers 2020 to 2032, so 2027 holds a projected point only.
_DECLARED_SOURCE = "dc/base/DeclaredSource"
_PROVENANCE_POLICY = FacetDefaultPolicy(
    resolved_ref=_DECLARED_SOURCE,
    policy_node="dc/policy/fertility-provenance",
    policy_version="1",
    selection_reason="declared",
)
_RECORDED_AND_PROJECTED = {
    f"{_FERTILITY}|country/KEN": [
        {
            "facetId": "recorded",
            "provenanceId": _DECLARED_SOURCE,
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
    with a point that year. Declared, the variable's default is the recorded facet ending in
    2000, so search refuses it before any sentence: no coverage, sentence or preview names
    the projection. Materialise serves the declared facet when asked
    (`tests/materialise/test_declared_default_window.py`)."""
    control = _post(_client(FakeGraph(obs=_RECORDED_AND_PROJECTED)), _FERTILITY_IN_2027)
    assert control.status_code == 200
    part = control.json()["parts"][0]
    assert part["status"] == "definite"
    assert part["spec"]["resolved"]["coverage"]["window"] == {"start_year": 2027, "end_year": 2027}

    declared = FakeGraph(
        obs=_RECORDED_AND_PROJECTED,
        facet_policies={(f"dcid:{_FERTILITY}", "provenance"): _PROVENANCE_POLICY},
    )
    buffered = _post(_client(declared), _FERTILITY_IN_2027)
    streamed = _post(_client(declared), _FERTILITY_IN_2027, headers={"Accept": "text/event-stream"})

    assert buffered.status_code == 501
    assert (buffered.json()["code"], buffered.json()["retryable"]) == (
        "facet_binding_unsupported",
        False,
    )
    assert _DECLARED_SOURCE in buffered.json()["detail"]
    events = [
        line.removeprefix("event: ").strip()
        for line in streamed.text.splitlines()
        if line.startswith("event: ")
    ]
    assert events[-1] == "error"
    assert "sentence" not in events
    assert "ProjectionSource" not in streamed.text


# ---------------------------------------------------------------------------
# A graph given no default facet returns no policy, and one given a policy returns it.
# ---------------------------------------------------------------------------


def _stat_vars_and_axes() -> list[tuple[str, str]]:
    return [
        (f"dcid:{sv}", axis)
        for sv in (_FERTILITY, "Count_Person", "Unrelated_StatVar")
        for axis in FACET_AXES
    ]


def test_a_graph_declaring_no_default_returns_no_policy_and_one_declaring_returns_it(
    tmp_path: Path,
) -> None:
    """The production client, its origin delegate, the live wrapper, promptlab's graph cache
    and its read-only placeholder, the recording proxy and every fake built without policies
    return no policy; the same wrappers over a graph given a policy, and the fakes given one,
    return it."""
    live = LiveGraphClient(base="http://fake")
    silent = [
        live,
        live._client,
        live._client._ours,
        _NoLiveGraph(),
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
        _declaring_graph(),
        GraphCache(_declaring_graph(), path=tmp_path / "declaring.sqlite"),
        _RecordingGraph(_declaring_graph(), nodes={}, obs={}, detect={}, resolve={}),
        FakeGraphReader(policies={(f"dcid:{_FERTILITY}", "unit"): _POLICY}),
        FakeMintGraphReader(arcs={}, policies={(f"dcid:{_FERTILITY}", "unit"): _POLICY}),
    ]
    for graph in declaring:
        assert graph.default_facet_policy(stat_var=f"dcid:{_FERTILITY}", axis="unit") == _POLICY
