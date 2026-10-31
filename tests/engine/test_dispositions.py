"""Pins the raw-text door's catalogue and declined answers: they return straight
after extract with no recall, bind or second cache lookup, read the graph only
through the entity resolver's place lookup and only when the question names a
place, carry the one confirmed country the query named as `scope.place`, stream
`step(extract)`, `response` and `done` only, and round-trip through the layer-1
response cache.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from entity_resolution import Entity, Group, NotFound, Resolved, shared_membership_policy
from entity_resolution import Resolution as EntityResolution
from entity_resolution.membership import MembershipPolicy
from entity_resolution.types import GraphLookup
from search.engine.core import resolve_async, resolve_events
from search.engine.events import DoneEvent, Event, ResponseEvent, StepEvent
from search.engine.extract import Extraction
from search.engine.response_cache import ResponseCache
from search.models import (
    CatalogueAnswer,
    DatasetResubmitInput,
    DeclinedAnswer,
    GraphRef,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolveRequest,
    ResubmitParsedEntry,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import EMPTY_SDMX, make_request, parse_sse
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph

_KENYA = Entity(dcid="country/KEN", name="Kenya", kind="country", graph_type="Country")
_UGANDA = Entity(dcid="country/UGA", name="Uganda", kind="country", graph_type="Country")
_NAIROBI = Entity(dcid="wikidataId/Q3870", name="Nairobi", kind="place_other", graph_type="City")
_SAHEL = Entity(dcid="undata-geo/G00134000", name="Sahel", kind="subregion", graph_type="Place")
_WORLD_BANK = Entity(
    dcid="dfo/org/WorldBank", name="World Bank", kind="organisation", graph_type="Organization"
)

_RESOLUTIONS: dict[str, EntityResolution] = {
    "Kenya": Resolved(_KENYA),
    "Kenyan": Resolved(_KENYA),
    "Uganda": Resolved(_UGANDA),
    "Nairobi": Resolved(_NAIROBI),
    "Sahel": Group(members=(_KENYA,), entity=_SAHEL),
    "World Bank": Resolved(_WORLD_BANK),
}


class _ExtractOnlyLLM:
    """Answers the extract call with one fixed extraction and records every schema asked for.
    Any other call, such as bind, fails the test."""

    def __init__(self, extraction: dict[str, Any]) -> None:
        self._extraction = extraction
        self.schemas: list[str] = []

    async def generate_structured[T: BaseModel](
        self, *, prompt: str, system: str, schema: type[T]
    ) -> tuple[T, dict | None]:
        self.schemas.append(schema.__name__)
        assert schema is Extraction, f"a disposition run made a {schema.__name__} call"
        usage = {"input_tokens": 11, "output_tokens": 7, "cached_tokens": 0}
        return schema.model_validate(self._extraction), usage


class _StubResolver:
    """Resolves names from `_RESOLUTIONS` and records every call."""

    def __init__(self, *, error: Exception | None = None, delay_s: float = 0.0) -> None:
        self._error = error
        self._delay_s = delay_s
        self.calls: list[list[str]] = []

    @property
    def membership(self) -> MembershipPolicy:
        """The shared membership policy, which says which resolved places are regions."""
        return shared_membership_policy()

    async def resolve(
        self, texts: list[str], *, deadline: float | None = None
    ) -> list[EntityResolution]:
        self.calls.append(list(texts))
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._error is not None:
            raise self._error
        return [_RESOLUTIONS.get(text, NotFound()) for text in texts]


_RESOLVER_READS = frozenset(name for name in vars(GraphLookup) if not name.startswith("_"))
"""The graph reads the entity resolver makes: the only ones a disposition run may make."""


class _CountingGraph:
    """The offline graph, recording every attribute a run reads. A read the entity resolver
    does not make, such as recall, shape or materialise, fails the test."""

    def __init__(self) -> None:
        self._graph = FakeGraph()
        self.reads: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.reads.append(name)
        if name not in _RESOLVER_READS:
            raise AssertionError(f"a disposition run read graph.{name} outside the resolver")
        return getattr(self._graph, name)


def _extraction(kind: str, *, entities: Sequence[str] = (), variables: Sequence[str] = ()) -> dict:
    return {
        "kind": kind,
        "variables": list(variables),
        "entities": [{"text": text} for text in entities],
        "dates": [],
    }


def _events(
    query: str,
    llm: _ExtractOnlyLLM,
    resolver: _StubResolver | None,
    *,
    graph: Any = None,
    response_cache: ResponseCache | None = None,
) -> list[Event]:
    """Run the raw-text door. With no `resolver`, the engine builds its own entity resolver
    over `graph`, as it does in service."""

    async def _collect() -> list[Event]:
        return [
            ev
            async for ev in resolve_events(
                make_request(query),
                graph=graph if graph is not None else _CountingGraph(),
                resolver=resolver,
                llm=llm,
                sdmx=EMPTY_SDMX,
                response_cache=response_cache,
            )
        ]

    return asyncio.run(_collect())


def _response(events: list[Event]) -> Any:
    (response,) = [ev.response for ev in events if isinstance(ev, ResponseEvent)]
    return response


def _done(events: list[Event]) -> DoneEvent:
    (done,) = [ev for ev in events if isinstance(ev, DoneEvent)]
    return done


def _shape(events: list[Event]) -> list[str]:
    return [f"step({ev.step})" if isinstance(ev, StepEvent) else type(ev).__name__ for ev in events]


_SHORT_CIRCUIT_SEQUENCE = ["step(extract)", "ResponseEvent", "DoneEvent"]


# ---------------------------------------------------------------------------
# Each disposition kind answers with its own body and three events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["out_of_scope", "requires_analysis"])
def test_a_declining_kind_answers_declined_with_its_reason(reason: str):
    query = "What should the next IDA replenishment prioritise?"
    llm = _ExtractOnlyLLM(_extraction(reason))
    graph = _CountingGraph()

    events = _events(query, llm, None, graph=graph)

    assert _shape(events) == _SHORT_CIRCUIT_SEQUENCE
    response = _response(events)
    assert isinstance(response, DeclinedAnswer)
    assert response.reason == reason
    assert response.scope.place is None
    assert response.query_echo.raw_query == query
    assert response.query_echo.extract_skipped is False
    assert response.diagnostics.llm_usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "cached_tokens": 0,
    }
    assert response.diagnostics.timing_ms.by_step is not None
    assert set(response.diagnostics.timing_ms.by_step) == {"coverage", "extract", "entity_check"}
    assert llm.schemas == ["Extraction"], "only extract ran: no bind or other LLM call"
    assert graph.reads == [], "a question naming no place reads nothing from the graph"
    assert _done(events).timed_out is False


def test_the_catalogue_kind_answers_catalogue():
    llm = _ExtractOnlyLLM(_extraction("catalogue"))
    graph = _CountingGraph()

    events = _events("What data do you have?", llm, None, graph=graph)

    assert _shape(events) == _SHORT_CIRCUIT_SEQUENCE
    response = _response(events)
    assert isinstance(response, CatalogueAnswer)
    assert response.scope.place is None
    assert llm.schemas == ["Extraction"]
    assert graph.reads == []


@pytest.mark.parametrize(
    ("entities", "expected_reads", "expected_place"),
    [
        (["Kenya"], ["node_arcs_many"], GraphRef(ref="country/KEN", label="Kenya")),
        (["Nairobi", "Zzyzx land"], ["resolve_by_arc"] * 4, None),
    ],
    ids=["country", "fallback-ladder"],
)
def test_a_question_naming_a_place_reads_the_graph_only_through_the_resolver(
    entities: list[str], expected_reads: list[str], expected_place: GraphRef | None
):
    llm = _ExtractOnlyLLM(_extraction("catalogue", entities=entities))
    graph = _CountingGraph()

    events = _events("What can I find there?", llm, None, graph=graph)

    assert _shape(events) == _SHORT_CIRCUIT_SEQUENCE
    response = _response(events)
    assert isinstance(response, CatalogueAnswer)
    assert response.scope.place == expected_place
    assert graph.reads == expected_reads
    assert llm.schemas == ["Extraction"]


def test_the_sse_wire_carries_step_response_done_and_nothing_else():
    llm = _ExtractOnlyLLM(_extraction("catalogue", entities=["Kenya"]))
    client = TestClient(
        search_app(graph=FakeGraph(), resolver=_StubResolver(), llm=llm, sdmx=FakeSdmxClient()),
        raise_server_exceptions=True,
    )

    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "What can I find on Kenya?"}},
        headers={"Accept": "text/event-stream"},
    )

    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    assert [name for name, _ in frames] == ["step", "response", "done"]
    assert frames[0][1]["step"] == "extract"
    body = frames[1][1]["response"]
    assert body["kind"] == "catalogue"
    assert body["scope"] == {"place": {"ref": "country/KEN", "label": "Kenya"}}
    assert "parts" not in body


# ---------------------------------------------------------------------------
# scope.place is the one country the resolver confirms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entities", "expected"),
    [
        (["Kenya"], GraphRef(ref="country/KEN", label="Kenya")),
        (["Nairobi", "Kenya"], GraphRef(ref="country/KEN", label="Kenya")),
        (["Kenya", "Kenyan"], GraphRef(ref="country/KEN", label="Kenya")),
        (["Nairobi"], None),
        (["Sahel"], None),
        (["World Bank"], None),
        (["Atlantis"], None),
        (["Kenya", "Uganda"], None),
    ],
    ids=[
        "country",
        "city-and-its-country",
        "two-names-one-country",
        "city",
        "region",
        "organisation",
        "not-found",
        "two-countries",
    ],
)
def test_scope_place_is_set_only_for_a_single_confirmed_country(
    entities: list[str], expected: GraphRef | None
):
    llm = _ExtractOnlyLLM(_extraction("out_of_scope", entities=entities))
    resolver = _StubResolver()

    response = _response(_events("a question naming places", llm, resolver))

    assert isinstance(response, DeclinedAnswer)
    assert response.scope.place == expected
    assert resolver.calls == [entities], "every distinct name goes to one resolver call"


def test_a_resolver_failure_answers_without_a_place_and_is_not_cached():
    llm = _ExtractOnlyLLM(_extraction("catalogue", entities=["Kenya"]))
    resolver = _StubResolver(error=RuntimeError("graph unreachable"))
    cache = ResponseCache()

    events = _events("What can I find on Kenya?", llm, resolver, response_cache=cache)

    assert _shape(events) == _SHORT_CIRCUIT_SEQUENCE
    response = _response(events)
    assert isinstance(response, CatalogueAnswer)
    assert response.scope.place is None
    assert _done(events).timed_out is False
    assert _done(events).degraded_leg is True

    repeat = _events("What can I find on Kenya?", llm, _StubResolver(), response_cache=cache)
    assert _shape(repeat) == _SHORT_CIRCUIT_SEQUENCE, "a degraded answer is not stored"
    assert _response(repeat).scope.place == GraphRef(ref="country/KEN", label="Kenya")


def test_a_resolver_timeout_answers_without_a_place(search_settings):
    search_settings(request_timeout_s=0.2)
    llm = _ExtractOnlyLLM(_extraction("out_of_scope", entities=["Kenya"]))
    resolver = _StubResolver(delay_s=5.0)

    events = _events("Temperature in Kenya right now", llm, resolver)

    assert _shape(events) == _SHORT_CIRCUIT_SEQUENCE
    response = _response(events)
    assert isinstance(response, DeclinedAnswer)
    assert response.scope.place is None
    assert _done(events).timed_out is True


# ---------------------------------------------------------------------------
# Layer 1 stores and serves a disposition answer
# ---------------------------------------------------------------------------


def test_a_repeated_catalogue_question_is_a_layer1_hit():
    cache = ResponseCache()
    llm = _ExtractOnlyLLM(_extraction("catalogue", entities=["Kenya"]))
    resolver = _StubResolver()

    first = _events("What can I find on Kenya?", llm, resolver, response_cache=cache)
    second = _events("What can I find  on Kenya?", llm, resolver, response_cache=cache)

    assert _shape(second) == ["ResponseEvent", "DoneEvent"], "a hit emits what any hit emits"
    assert llm.schemas == ["Extraction"], "the hit ran no extract"
    assert resolver.calls == [["Kenya"]], "the hit made no resolver call"
    stored, served = _response(first), _response(second)
    assert isinstance(served, CatalogueAnswer)
    assert served.scope == stored.scope
    assert set(stored.diagnostics.timing_ms.by_step) == {
        "coverage",
        "extract",
        "entity_check",
        "cache_lookup",
    }, "a miss books the layer-1 lookup that preceded it"
    assert served.query_echo.raw_query == "What can I find  on Kenya?"
    assert served.query_echo.extract_skipped is True
    assert served.diagnostics.timing_ms.by_step is not None
    assert set(served.diagnostics.timing_ms.by_step) == {"coverage", "response_cache_layer1"}


def test_a_repeated_declined_question_is_a_layer1_hit():
    cache = ResponseCache()
    llm = _ExtractOnlyLLM(_extraction("requires_analysis"))

    _events("Should donors fund IDA?", llm, _StubResolver(), response_cache=cache)
    second = _events("Should donors fund IDA?", llm, _StubResolver(), response_cache=cache)

    served = _response(second)
    assert isinstance(served, DeclinedAnswer)
    assert served.reason == "requires_analysis"
    assert llm.schemas == ["Extraction"]


# ---------------------------------------------------------------------------
# What still becomes a resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["catalogue", "requires_analysis", "out_of_scope"])
def test_a_yielding_kind_that_names_a_measure_resolves(kind: str):
    """An unrecorded measure recalls nothing on the fake graph, so it ends no_data without
    bind. An out_of_scope reading that names a measure is no exception, so text in the
    query cannot decline a data question."""
    llm = _ExtractOnlyLLM(_extraction(kind, variables=["zzq unrecorded measure"]))

    response = _response(_events("zzq unrecorded measure", llm, _StubResolver(), graph=FakeGraph()))

    assert isinstance(response, Resolution)
    assert [part.status for part in response.parts] == ["no_data"]


def test_a_resolution_with_no_variables_answers_as_a_catalogue():
    llm = _ExtractOnlyLLM(_extraction("resolution"))

    events = _events("hello", llm, _StubResolver())

    response = _response(events)
    assert isinstance(response, CatalogueAnswer)
    assert response.scope.place is None


_UNRECORDED_PART = ParsedPart(variable_text="zzq unrecorded measure")


@pytest.mark.parametrize(
    "request_input",
    [
        ParsedInput(parts=[_UNRECORDED_PART]),
        DatasetResubmitInput(entries=[ResubmitParsedEntry(part=_UNRECORDED_PART)]),
    ],
    ids=["parsed", "dataset_resubmit"],
)
def test_the_parsed_and_resubmit_doors_always_resolve(request_input):
    """Neither door runs extract, so an LLM that would decline everything is never asked."""
    llm = _ExtractOnlyLLM(_extraction("out_of_scope"))

    response = asyncio.run(
        resolve_async(
            ResolveRequest(input=request_input),
            graph=FakeGraph(),
            resolver=_StubResolver(),
            llm=llm,
            sdmx=EMPTY_SDMX,
        )
    )

    assert isinstance(response, Resolution)
    assert llm.schemas == []
