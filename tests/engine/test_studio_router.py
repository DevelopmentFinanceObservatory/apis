"""Pins Studio routing on the raw_text door (`search.engine.studio_router`): only a request
that sets `studio_routing` while the router is on asks Jev, the route is decided from the
first parse's kind and Jev's studio probability when the parse returns, a Jev call that has
not answered by then routes nothing and leaves its verdict in the memo, and the response
cache keeps routed and unrouted answers apart and replays an offer's `route` event.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from search.engine import studio_router
from search.engine.binders.jev_client import JevError, JevLane, JevResponse, JevUsage
from search.engine.core import ResolveRecord, resolve_events
from search.engine.events import (
    Event,
    ResponseEvent,
    RouteEvent,
    serialize_sse,
)
from search.engine.extract import Extraction
from search.engine.response_cache import ResponseCache
from search.models import (
    CatalogueAnswer,
    DeclinedAnswer,
    RawTextInput,
    Resolution,
    ResolveRequest,
    StudioRoute,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import EMPTY_SDMX, member_walk_reader, parse_sse, resolve_log_lines
from tests.engine._search_app import search_app
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM, make_resolver

_QUERY = "fertility rate in Kenya"


class _RouterAsk:
    """The router's `jev_client.ask`, answering every studio signal with `probability` (a
    binary fraction, so their mean is exactly it), or
    raising `fail`, after `gate` is set when one is given. It records each call's state,
    model and lane."""

    def __init__(
        self,
        probability: float,
        *,
        fail: Exception | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.probability = probability
        self.fail = fail
        self.gate = gate
        self.calls: list[tuple[object, str | None, JevLane | None]] = []

    async def __call__(
        self,
        state,
        questions: Mapping[str, Any],
        *,
        settings,
        model: str | None = None,
        lane: JevLane | None = None,
    ) -> JevResponse:
        self.calls.append((state, model, lane))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        p = self.probability
        return JevResponse(
            nouls={"wants_analysis": p, "needs_synthesis": p, "in_scope": 0.99},
            choices={"route": {"analysis": p, "retrieval": 1 - p}},
            model="jev-1.14.0",
            usage=JevUsage(input_tokens=300, output_tokens=3),
            seconds=0.1,
            chunks=1,
        )


class _KindLLM(FakeLLM):
    """The recorded fake, whose extraction of every query reads as `kind` with nothing named."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self._kind = kind

    async def generate_structured(self, *, prompt: str, system: str, schema):
        if schema is Extraction:
            await asyncio.sleep(0)
            extraction = {"kind": self._kind, "variables": [], "entities": [], "dates": []}
            usage = {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}
            return Extraction.model_validate(extraction), usage
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


@pytest.fixture(autouse=True)
def _fresh_router(monkeypatch: pytest.MonkeyPatch):
    """A fresh memo and lane for every test."""
    studio_router.clear_memo()
    monkeypatch.setattr(studio_router, "_LANE", JevLane.fresh())
    yield
    studio_router.clear_memo()


def _install(monkeypatch: pytest.MonkeyPatch, ask: _RouterAsk) -> _RouterAsk:
    monkeypatch.setattr(studio_router, "_ask", ask)
    return ask


def _request(query: str = _QUERY, *, routed: bool = True) -> ResolveRequest:
    return ResolveRequest(input=RawTextInput(query=query, studio_routing=routed))


async def _collect(
    request: ResolveRequest,
    *,
    llm=None,
    cache: ResponseCache | None = None,
    record: ResolveRecord | None = None,
) -> list[Event]:
    graph = FakeGraph()
    return [
        event
        async for event in resolve_events(
            request,
            graph=graph,
            llm=llm if llm is not None else FakeLLM(),
            sdmx=EMPTY_SDMX,
            materialise_graph=member_walk_reader(graph),
            response_cache=cache,
            record=record,
        )
    ]


def _run(coroutine):
    """Run `coroutine` under the pinned extraction date the recorded fixtures were keyed on."""
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        return asyncio.run(coroutine)


def _response(events: list[Event]):
    (response,) = [event.response for event in events if isinstance(event, ResponseEvent)]
    return response


def _kinds(events: list[Event]) -> list[str]:
    return [event.kind for event in events]


@pytest.mark.parametrize(
    ("routed", "router"), [(False, "on"), (True, "off")], ids=["not-asked", "router-off"]
)
def test_a_request_that_does_not_route_asks_jev_nothing_and_carries_no_route(
    monkeypatch, search_settings, routed: bool, router: str
):
    search_settings(studio_router=router)
    ask = _install(monkeypatch, _RouterAsk(0.75))
    record = ResolveRecord()

    events = _run(_collect(_request(routed=routed), record=record))

    assert ask.calls == []
    assert "route" not in _kinds(events)
    response = _response(events)
    assert isinstance(response, Resolution)
    assert response.route is None
    assert record.studio_route is None


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.75, StudioRoute(mode="auto", probability=0.75)),
        (0.5, StudioRoute(mode="offer", probability=0.5)),
        (0.25, None),
    ],
    ids=["auto", "offer", "none"],
)
def test_a_resolution_routes_on_the_studio_probability(
    monkeypatch, probability: float, expected: StudioRoute | None
):
    ask = _install(monkeypatch, _RouterAsk(probability))
    record = ResolveRecord()

    events = _run(_collect(_request(), record=record))

    response = _response(events)
    assert response.route == expected
    ((state, model, lane),) = ask.calls
    assert state == studio_router.ROUTER_QUESTIONS.state(_QUERY)
    assert model == studio_router.ROUTER_QUESTIONS.model
    assert lane is studio_router._LANE
    assert record.studio_route is not None
    assert record.studio_route.log_fields() == {
        "decision": expected.mode if expected is not None else "none",
        "probability": probability,
        "extract_kind": "resolution",
        "jev": "answered",
        "jev_error": None,
        "jev_ms": record.studio_route.jev_ms,
    }
    if expected is None:
        assert isinstance(response, Resolution)
        assert "route" not in _kinds(events)
    elif expected.mode == "auto":
        assert isinstance(response, DeclinedAnswer)
        assert response.reason == "requires_analysis"
        assert _kinds(events) == ["step", "response", "done"]
    else:
        assert isinstance(response, Resolution)
        kinds = _kinds(events)
        assert kinds[:3] == ["step", "route", "interpretation"]
        assert kinds.count("route") == 1
        (route,) = [event for event in events if isinstance(event, RouteEvent)]
        assert route.probability == probability


@pytest.mark.parametrize(
    ("kind", "answer", "route_on_the_wire"),
    [
        (
            "requires_analysis",
            DeclinedAnswer,
            {"target": "studio", "mode": "auto", "probability": None},
        ),
        ("out_of_scope", DeclinedAnswer, None),
        ("catalogue", CatalogueAnswer, None),
    ],
)
def test_a_parse_kind_other_than_resolution_routes_on_the_parse_alone(
    monkeypatch, kind: str, answer: type, route_on_the_wire: dict | None
):
    """Jev's high probability changes nothing: the parse decides scope, and a parse reading
    the question as requiring analysis moves it with no router probability, stated as null
    on the wire."""
    _install(monkeypatch, _RouterAsk(0.75))

    events = _run(
        _collect(_request("What should the next IDA replenishment prioritise?"), llm=_KindLLM(kind))
    )

    response = _response(events)
    assert isinstance(response, answer)
    assert _kinds(events) == ["step", "response", "done"]
    (response_event,) = [event for event in events if isinstance(event, ResponseEvent)]
    wire = json.loads(serialize_sse(response_event).split("data: ", 1)[1])
    assert wire["response"].get("route") == route_on_the_wire


@pytest.mark.parametrize(
    ("fail", "gated", "jev", "jev_error"),
    [
        (None, True, "late", None),
        (JevError(429, "rate_limited"), False, "failed", "rate_limited"),
    ],
    ids=["late", "failed"],
)
def test_a_jev_call_without_an_answer_when_the_parse_returns_routes_nothing(
    monkeypatch, fail: Exception | None, gated: bool, jev: str, jev_error: str | None
):
    _install(monkeypatch, _RouterAsk(0.75, fail=fail, gate=asyncio.Event() if gated else None))
    record = ResolveRecord()

    events = _run(_collect(_request(), record=record))

    response = _response(events)
    assert isinstance(response, Resolution)
    assert response.route is None
    assert "route" not in _kinds(events)
    assert record.studio_route is not None
    assert (record.studio_route.jev, record.studio_route.jev_error) == (jev, jev_error)
    assert record.studio_route.decision == "none"


def test_a_late_verdict_reaches_the_memo_and_routes_the_repeat_layer1_did_not_store(
    monkeypatch,
):
    """The first run decides without Jev, so layer 1 stores nothing under the query; Jev's
    answer, landing after, routes the repeat without a second call."""
    gate = asyncio.Event()
    ask = _install(monkeypatch, _RouterAsk(0.75, gate=gate))
    cache = ResponseCache()
    first, second = ResolveRecord(), ResolveRecord()

    async def scenario() -> tuple[list[Event], list[Event]]:
        late = await _collect(_request(), cache=cache, record=first)
        gate.set()
        await asyncio.gather(*studio_router._BACKGROUND)
        return late, await _collect(_request(), cache=cache, record=second)

    late, repeat = _run(scenario())

    assert _response(late).route is None
    assert first.studio_route is not None and first.studio_route.jev == "late"
    assert second.cache == "miss", "a run routed without Jev's answer is not stored in layer 1"
    assert second.studio_route is not None and second.studio_route.jev == "memo"
    assert _response(repeat).route == StudioRoute(mode="auto", probability=0.75)
    assert len(ask.calls) == 1


def test_layer1_keeps_routed_and_unrouted_answers_apart_and_replays_the_offer(
    monkeypatch, search_settings
):
    ask = _install(monkeypatch, _RouterAsk(0.5))
    cache = ResponseCache()

    def run(*, routed: bool) -> tuple[list[Event], ResolveRecord]:
        record = ResolveRecord()
        return _run(_collect(_request(routed=routed), cache=cache, record=record)), record

    _, unrouted = run(routed=False)
    _, routed = run(routed=True)
    hit, routed_again = run(routed=True)
    unrouted_hit, unrouted_again = run(routed=False)
    search_settings(studio_router_offer=0.5)
    _, rethresholded = run(routed=True)

    assert [unrouted.cache, routed.cache] == ["miss", "miss"]
    assert routed_again.cache == "hit layer1"
    assert _kinds(hit) == ["route", "response", "done"]
    assert _response(hit).route == StudioRoute(mode="offer", probability=0.5)
    assert routed_again.studio_route is not None
    assert routed_again.studio_route.log_fields()["jev"] == "cached"
    assert unrouted_again.cache == "hit layer1"
    assert "route" not in _kinds(unrouted_hit)
    assert _response(unrouted_hit).route is None
    assert rethresholded.cache == "miss", "a threshold change starts fresh entries"
    assert len(ask.calls) == 1, "the hit and the memo asked Jev nothing"


def test_the_sse_wire_sends_the_offer_with_its_trace_id_live_and_on_a_hit(monkeypatch):
    """The `route` frame precedes `interpretation` on a live run and `response` on a layer-1
    hit, each carrying its own request's trace id, the one its response states; the live
    run's `resolve` line logs the decision."""
    _install(monkeypatch, _RouterAsk(0.5))
    graph = FakeGraph()
    client = TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=member_walk_reader(graph),
            response_cache=ResponseCache(),
        ),
        raise_server_exceptions=True,
    )

    def post() -> list[tuple[str, dict]]:
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": _QUERY, "studio_routing": True}},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        return parse_sse(resp.text)

    with patch("search.engine.extract.date") as mock_date, resolve_log_lines() as lines:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        live = post()
        hit = post()

    route = {"target": "studio", "mode": "offer", "probability": 0.5}
    for frames, leading in (
        (live, ["step", "route", "interpretation"]),
        (hit, ["route", "response"]),
    ):
        names = [name for name, _ in frames]
        assert names[: len(leading)] == leading
        frame = dict(frames)["route"]
        response = dict(frames)["response"]["response"]
        assert frame == {"kind": "route", **route, "trace_id": response["diagnostics"]["trace_id"]}
        assert response["route"] == route
    assert dict(live)["route"]["trace_id"] != dict(hit)["route"]["trace_id"]
    assert [line["cache"] for line in lines] == ["miss", "hit layer1"]
    assert lines[0]["studio_route"]["decision"] == "offer"
    assert lines[0]["studio_route"]["extract_kind"] == "resolution"
    assert lines[0]["studio_route"]["jev"] == "answered"


def test_a_layer2_hit_carries_its_own_requests_route(monkeypatch):
    """A late first run stores its unrouted answer in layer 2 alone. The repeat routes an
    offer from the memo and is served that entry, carrying its own route and sending its
    `route` event once."""
    gate = asyncio.Event()
    _install(monkeypatch, _RouterAsk(0.5, gate=gate))
    cache = ResponseCache()
    record = ResolveRecord()

    async def scenario() -> list[Event]:
        await _collect(_request(), cache=cache)
        gate.set()
        await asyncio.gather(*studio_router._BACKGROUND)
        return await _collect(_request(), cache=cache, record=record)

    events = _run(scenario())

    assert record.cache == "hit layer2"
    assert _kinds(events).count("route") == 1
    assert _response(events).route == StudioRoute(mode="offer", probability=0.5)


_RECORDED_D = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "jev" / "router_d_answers.json").read_text()
)["records"]


@pytest.mark.parametrize("record", _RECORDED_D, ids=[r["query"] for r in _RECORDED_D])
def test_a_recorded_jev_answer_scores_as_the_router_eval_scored_it(record: dict):
    """The route choice's `analysis`, `wants_analysis` and `needs_synthesis` average to the
    studio probability the eval reported, and `in_scope` takes no part."""
    answers = record["answers"]
    response = JevResponse(
        nouls={qid: a["noul"] for qid, a in answers.items() if a["type"] == "noul"},
        choices={qid: a["probabilities"] for qid, a in answers.items() if a["type"] == "choice"},
        model="jev-1.14.0",
        usage=JevUsage(input_tokens=0, output_tokens=0),
        seconds=0.1,
        chunks=1,
    )

    probability = studio_router.ROUTER_QUESTIONS.studio_probability(response)

    assert round(probability, 2) == record["eval_score"]
    assert set(answers) == set(studio_router.ROUTER_QUESTIONS.questions)
