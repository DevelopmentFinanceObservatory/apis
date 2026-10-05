"""A StatVar whose graph declares a default facet is refused at resolve time, before any
coverage, headline or sentence is built, as the registered `facet_binding_unsupported`
refusal (501, not retryable). No fan-out leg or UN gap-fill pass degrades it into a
no-data answer."""

from __future__ import annotations

import ast
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
from search.engine.events import SentenceEvent, StepEvent
from search.engine.region import phase_place_materialise, pipeline
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.retrieval.facet_policy import refuse_declared_defaults
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine import test_candidates_fold_after_unit_filter as fold
from tests.engine import test_gap_fill as gap_fill
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


# A policy on a StatVar no test answer reads: the graph declares defaults, so the first pass
# holds its sentences, and no answer here is refused for it.
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
            base_steps=[],
            base_timing={},
            coverage=FakeCoverageService(index).snapshot(),
            emit=_emit,
        )
    )


def test_a_measure_elected_stream_emits_no_sentence_before_the_refusal(monkeypatch) -> None:
    """A measure-elected resolve builds its sentence while materialise runs and emits it only
    once materialise returns, so the refusal reaches the client before any sentence."""
    control: list = []
    _measure_elected_stream(monkeypatch, control, facet_policies={})
    assert [e for e in control if isinstance(e, SentenceEvent)]

    events: list = []
    with pytest.raises(DeclaredDefaultUnsupportedError):
        _measure_elected_stream(
            monkeypatch,
            events,
            facet_policies={(f"dcid:{measure_election._EXTRA_SV}", "unit"): _POLICY},
        )
    assert events, "the stream emits its shape and step events before materialise"
    assert not [e for e in events if isinstance(e, SentenceEvent)]


def _sentences_and_materialise(events: list) -> list[str]:
    """The stream's sentence events and its materialise step event, in order."""
    return [
        "sentence" if isinstance(e, SentenceEvent) else "materialise"
        for e in events
        if isinstance(e, SentenceEvent) or (isinstance(e, StepEvent) and e.step == "materialise")
    ]


def test_a_graph_declaring_no_default_streams_the_bind_stage_sentence_before_materialise(
    monkeypatch,
) -> None:
    """A graph that declares no default facet cannot refuse, so the measure-elected
    bind-stage sentence streams while materialise runs: it precedes the materialise step
    event, and the definite answer's final sentence follows that event."""
    events: list = []
    _measure_elected_stream(monkeypatch, events, facet_policies={})

    order = _sentences_and_materialise(events)
    assert order[:2] == ["sentence", "materialise"]
    assert order[-1] == "sentence"


def test_a_graph_declaring_a_default_holds_the_bind_stage_sentence_until_the_answer_stands(
    monkeypatch,
) -> None:
    """A graph declaring a default on a StatVar this answer does not read can still refuse a
    later step, so the first pass holds its sentences: none precedes the materialise step
    event, and the answer's sentences arrive once it stands."""
    events: list = []
    _measure_elected_stream(monkeypatch, events, facet_policies=_UNRELATED_POLICY)

    order = _sentences_and_materialise(events)
    assert order[0] == "materialise"
    assert "sentence" in order


# ---------------------------------------------------------------------------
# A refusal raised after the bind-stage sentence over a graph that declares default facets:
# each pass holds its sentences until the part's answer stands, so none reaches the stream
# ahead of the refusal.
# ---------------------------------------------------------------------------


def test_a_refusal_in_the_candidates_fan_out_streams_no_sentence(monkeypatch) -> None:
    """The measure-elected grant-equivalent answer builds its bind-stage sentence, then the
    candidates fan-out materialises the plausible amount sibling, whose member declares a
    default: the request fails and the stream carries no sentence."""
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
                base_steps=[],
                base_timing={},
                emit=_emit,
                coverage=FakeCoverageService(index).snapshot(),
            )
        )

    control: list = []
    stream({}, control)
    assert [e for e in control if isinstance(e, SentenceEvent)]
    events: list = []
    with pytest.raises(DeclaredDefaultUnsupportedError, match=fold._ALT_SV):
        stream({(f"dcid:{fold._ALT_SV}", "unit"): _POLICY}, events)
    assert events, "the stream carries shape and step events before the refusal"
    assert not [e for e in events if isinstance(e, SentenceEvent)]


def _with_a_bind_stage_sentence(monkeypatch) -> None:
    """Have the first pass emit a sentence before its phases run, as a measure-elected bind
    stage does once its materialise returns."""
    real = pipeline._run_phases

    async def run(state):
        if state.pool == "ours" and state.emit is not None:
            await state.emit(
                SentenceEvent(
                    variable_index=0, spec_id="spec_bind_stage", sentence_plan=[], slots=[]
                )
            )
        return await real(state)

    monkeypatch.setattr(pipeline, "_run_phases", run)


def _gap_fill_stream(graph: FakeGraph) -> tuple[RegionResult, list]:
    events: list = []

    async def collect(event) -> None:
        events.append(event)

    return gap_fill._resolve(graph, emit=collect), events


def test_a_refusal_in_the_un_second_pass_streams_no_sentence(monkeypatch) -> None:
    """The first pass streams a sentence and ends without data; the UN second pass finds a
    variable declaring a default and refuses. The first pass's sentence is held, so the
    stream carries none."""
    _with_a_bind_stage_sentence(monkeypatch)
    graph = FakeGraph(
        **gap_fill._graph_kwargs(), facet_policies={(f"dcid:{gap_fill._UN}", "unit"): _POLICY}
    )
    events: list = []

    async def collect(event) -> None:
        events.append(event)

    with pytest.raises(DeclaredDefaultUnsupportedError, match=gap_fill._UN):
        gap_fill._resolve(graph, emit=collect)
    assert not [e for e in events if isinstance(e, SentenceEvent)]


def test_a_filled_answer_streams_only_the_second_passs_sentences(monkeypatch) -> None:
    """Over a graph declaring default facets, without the refusal the second pass answers,
    and the stream carries its sentences alone: the first pass's held sentence describes an
    answer that does not stand."""
    _with_a_bind_stage_sentence(monkeypatch)
    result, events = _gap_fill_stream(
        FakeGraph(**gap_fill._graph_kwargs(), facet_policies=_UNRELATED_POLICY)
    )

    assert result.status == "definite"
    spec_ids = [e.spec_id for e in events if isinstance(e, SentenceEvent)]
    assert spec_ids and "spec_bind_stage" not in spec_ids
    assert spec_ids[-1] == result.specs[0].spec_id


def test_a_filled_answer_over_a_graph_declaring_no_default_ends_on_its_own_sentence(
    monkeypatch,
) -> None:
    """A graph declaring no default streams the first pass's sentence as it is built; the
    second pass's sentences follow once its answer stands, so the last sentence on the
    stream is the answer's."""
    _with_a_bind_stage_sentence(monkeypatch)
    result, events = _gap_fill_stream(gap_fill._graph())

    assert result.status == "definite"
    spec_ids = [e.spec_id for e in events if isinstance(e, SentenceEvent)]
    assert spec_ids[0] == "spec_bind_stage"
    assert spec_ids[-1] == result.specs[0].spec_id


def test_a_refusal_in_sibling_re_election_streams_no_sentence(monkeypatch) -> None:
    """The first pass streams a sentence, its elected shape has no data at Kenya, and the
    re-election of a zero-observation sibling meets a declared default and refuses."""
    _with_a_bind_stage_sentence(monkeypatch)
    calls: list[str] = []

    async def refuse(state, mat_result):
        calls.append(state.shape_draft.shape_id)
        raise DeclaredDefaultUnsupportedError("a sibling declares a default")

    # Re-election runs when the elected shape has a sibling to try; the fixture recalls one
    # shape, so the elected shape stands in as its own sibling.
    monkeypatch.setattr(phase_place_materialise, "_reelection_siblings", lambda state: [0])
    monkeypatch.setattr(phase_place_materialise, "_reelect_zero_observation_sibling", refuse)
    # The re-election is replaced, so no sibling is walked ahead of it.
    monkeypatch.setattr(phase_place_materialise, "_sibling_shapes_of", lambda state: {})
    events: list = []

    async def collect(event) -> None:
        events.append(event)

    with pytest.raises(DeclaredDefaultUnsupportedError):
        gap_fill._resolve(
            FakeGraph(**gap_fill._graph_kwargs(un=()), facet_policies=_UNRELATED_POLICY),
            emit=collect,
        )
    assert calls, "the elected shape's no-data result runs the sibling re-election"
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
    "response_options": {"include_sentence": True},
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
# A graph that declares no default facet returns no policy: the hold rests on it.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_SOURCE_ROOTS = ("src", "packages/dfo-data-tools/src", "promptlab", "tests", "scripts")


def _graph_classes() -> dict[str, tuple[set[str], list[str]]]:
    """Every class in the repository's Python sources that name `default_facet_policy`, the
    graphs and the bases they inherit it from: its methods and its base names."""
    classes: dict[str, tuple[set[str], list[str]]] = {}
    for root in _SOURCE_ROOTS:
        for path in (_REPO / root).rglob("*.py"):
            text = path.read_text()
            if "default_facet_policy" not in text:
                continue
            for node in ast.walk(ast.parse(text, filename=str(path))):
                if isinstance(node, ast.ClassDef):
                    methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
                    bases = [ast.unparse(base).rsplit(".", 1)[-1] for base in node.bases]
                    known, known_bases = classes.get(node.name, (set(), []))
                    classes[node.name] = (known | methods, known_bases + bases)
    return classes


def test_every_graph_answering_default_facet_policy_says_whether_it_declares_any() -> None:
    """A class defining `default_facet_policy` also defines `declares_facet_defaults`, or
    inherits it from a base that does, so no graph the pipeline holds its sentences by can
    leave the question unanswered."""
    classes = _graph_classes()

    def answers(name: str, seen: frozenset[str] = frozenset()) -> bool:
        if name not in classes or name in seen:
            return False
        methods, bases = classes[name]
        return "declares_facet_defaults" in methods or any(
            answers(base, seen | {name}) for base in bases
        )

    graphs = sorted(
        name for name, (methods, _) in classes.items() if "default_facet_policy" in methods
    )
    assert len(graphs) >= 15, graphs
    assert [name for name in graphs if not answers(name)] == []


def _stat_vars_and_axes() -> list[tuple[str, str]]:
    return [
        (f"dcid:{sv}", axis)
        for sv in (_FERTILITY, "Count_Person", "Unrelated_StatVar")
        for axis in FACET_AXES
    ]


def test_a_graph_declaring_no_default_returns_no_policy_and_one_declaring_says_so(
    tmp_path: Path,
) -> None:
    """The production client, its origin delegate, the live wrapper, promptlab's graph cache
    and its read-only placeholder, the recording proxy and every fake built without policies
    declare no default and return no policy; the same wrappers over a graph given a policy,
    and the fakes given one, declare it."""
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
        assert graph.declares_facet_defaults() is False, graph
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
        assert graph.declares_facet_defaults() is True, graph
        assert graph.default_facet_policy(stat_var=f"dcid:{_FERTILITY}", axis="unit") == _POLICY
