"""An SSE search stores its event bundle on `done` and replays it for an identical request.

Every corpus case that reaches the cache runs twice through `offline_events` with the SSE
options the website posts (option coverage and observations). The second run must come from
the cache and send the first run's events from `response` on, differing only in the fields
that describe the request. Place options are not in the bundle: a hit streams them after the
replayed events, and they must equal the first run's by spec id and scope. With
`SSE_BUNDLE_METRICS` set to a file path, each case appends one JSON line of entry bytes and
run times to it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from typing import Any, NoReturn
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from materialise.rate_tables import RateTableCache
from origin_catalogue import note_un_blocked_places
from search.engine import enrichment
from search.engine.app import create_app
from search.engine.degraded_reads import DegradedReads, note_degraded_read
from search.engine.deps import SEARCH_DEPS_SLOT
from search.engine.errors import EngineInfraError, EngineInputError, GraphInfraError
from search.engine.events import DoneEvent, Event, ResponseEvent
from search.engine.options import surface
from search.engine.options.options_response_cache import OptionsResponseCache
from search.engine.options.population_memo import PopulationInFlight
from search.engine.options.surface import ComputedOptions
from search.engine.region import option_coverage, pipeline, place_options
from search.engine.region.observations import dispatch as obs_dispatch
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.region.result import RegionResult
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_keys import is_bundle_storable
from search.eval.cases import Case, load_cases
from search.eval.runner import build_request
from search.models import (
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    ResolveOptions,
    ResolveRequest,
    Warning,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    PINNED_DATE,
    build_materialise_double,
    member_walk_reader,
    offline_events,
)
from tests.engine.test_place_options_trailing import _fertility_kenya_materialise_with_population
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

_CASES = [c for c in load_cases() if c.door in ("raw_text", "parsed")]

# What a replay may change, as paths inside an event's payload: the request's own echo
# wording, trace id, timing and token usage.
_FRESH = (
    "response.query_echo.extract_skipped",
    "response.diagnostics.trace_id",
    "response.diagnostics.timing_ms",
    "response.diagnostics.llm_usage",
    "elapsed_ms",
    "trace_id",
    "telemetry",
)


def _from_response(events: list[Event]) -> list[dict]:
    """The events from `response` on, as the SSE wire dumps them."""
    start = next(i for i, e in enumerate(events) if isinstance(e, ResponseEvent))
    return [
        json.loads(e.model_dump_json(exclude_none=True, context={"surface": "sse"}))
        for e in events[start:]
    ]


def _without_place_options(events: list[dict]) -> list[dict]:
    return [e for e in events if e["kind"] != "place_options"]


def _place_options(events: list[dict]) -> dict[tuple[str, str], dict]:
    """The place-options events by spec id and scope, the keys a client reads them by."""
    return {(e["spec_id"], e["scope"]): e for e in events if e["kind"] == "place_options"}


def _replays(first: list[dict], second: list[dict]) -> bool:
    """Whether `second` sends `first`'s events, apart from the fields a replay refreshes and
    the place options' position, which a hit streams after the replayed events."""
    changed = _differences(_without_place_options(first), _without_place_options(second))
    return not {p for p in changed if not p.startswith(_FRESH)} and _place_options(
        first
    ) == _place_options(second)


def _differences(a: object, b: object, path: str = "") -> set[str]:
    """Paths where two JSON values differ, with list positions elided."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {
            d for k in a.keys() | b.keys() for d in _differences(a.get(k), b.get(k), f"{path}.{k}")
        }
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return {d for x, y in zip(a, b, strict=True) for d in _differences(x, y, path)}
    return set() if a == b else {path.lstrip(".")}


def _sse_request(case: Case) -> ResolveRequest:
    body = build_request(case)
    body["response_options"] = {
        "include_option_coverage": True,
        "include_observations": True,
    }
    return ResolveRequest.model_validate(body)


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_second_identical_sse_request_replays_the_first_bundle(case: Case) -> None:
    request = _sse_request(case)
    cache = ResponseCache()
    # Enough permits for two Specs' target and scope stages at once, so neither run drops a
    # Spec's place options on a full pool and the two runs' place options compare.
    pool = PlaceOptionsPool.build(max_workers=6)
    _, materialise_sdmx, _ = build_materialise_double({})
    materialise_graph = member_walk_reader(FakeGraph())

    def run() -> tuple[list[Event], float]:
        began = time.perf_counter()
        events = offline_events(
            request,
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
            stream_slot_options=True,
            response_cache=cache,
            place_options_pool=pool,
        )
        return events, (time.perf_counter() - began) * 1000

    try:
        first, first_ms = run()
    except EngineInputError:
        pytest.skip("a refusal is an answer, never stored")
    stored = cache.stats().entries > 0
    second, second_ms = run()
    pool.shutdown()

    stats = cache.stats()
    done = first[-1]
    assert isinstance(done, DoneEvent)
    if stored:
        assert stats.hits == 1
        assert _replays(_from_response(first), _from_response(second))
    else:
        # A response the cache refuses is computed again.
        assert stats.hits == 0

    if path := os.environ.get("SSE_BUNDLE_METRICS"):
        with open(path, "a") as out:
            out.write(
                json.dumps(
                    {
                        "case": case.id,
                        "door": case.door,
                        "stored": stored,
                        "bytes": stats.total_bytes,
                        "events": [e.kind for e in first],
                        "first_ms": first_ms,
                        "second_ms": second_ms,
                        "warnings": sorted(
                            {
                                w.code
                                for e in first
                                if isinstance(e, ResponseEvent)
                                for w in [
                                    *e.response.diagnostics.warnings,
                                    *(
                                        w
                                        for p in getattr(e.response, "parts", [])
                                        for w in p.warnings
                                    ),
                                ]
                            }
                            | {w.code for w in done.warnings}
                        ),
                        "timed_out": done.timed_out,
                        "degraded_leg": done.degraded_leg,
                        "first_done_ms": done.elapsed_ms,
                        "second_done_ms": second[-1].elapsed_ms,
                    }
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# A request whose answer carries every trailing event: slot options, observations and
# place options (the fertility-in-Kenya fixture the place-options tests use).
# ---------------------------------------------------------------------------

_QUERY = "fertility rate in Kenya"
# The website's submit (`buildRequest(input, true)`) and its suggestion (`DataSearch.tsx`).
_SUBMIT_OPTIONS = {"include_option_coverage": True, "include_observations": True}
_SUGGEST_OPTIONS = {"include_option_coverage": False, "include_observations": True}


def _kenya_request(options: dict[str, bool], query: str = _QUERY) -> ResolveRequest:
    return ResolveRequest.model_validate(
        {
            "schema_version": 1,
            "input": {"kind": "raw_text", "query": query},
            "response_options": options,
        }
    )


def _kenya_events(
    cache: ResponseCache,
    options: dict[str, bool],
    *,
    stream: bool,
    query: str = _QUERY,
    pool: PlaceOptionsPool | None = None,
    options_cache: OptionsResponseCache | None = None,
) -> list[Event]:
    materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
    if pool is None:
        pool = PlaceOptionsPool.build(max_workers=2, options_cache=options_cache)
    try:
        return offline_events(
            _kenya_request(options, query),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
            stream_slot_options=stream,
            response_cache=cache,
            place_options_pool=pool,
        )
    finally:
        pool.shutdown()


def test_every_trailing_event_kind_is_replayed_in_order() -> None:
    """A hit replays the slot-options and observations events in order, then streams the
    place options a fresh run streams."""
    cache = ResponseCache()
    first = _kenya_events(cache, _SUBMIT_OPTIONS, stream=True)
    kinds = [e.kind for e in first if e.kind not in ("step", "interpretation", "frame", "shape")]
    assert {"slot_options", "observations", "place_options"} <= set(kinds)
    entry_bytes = cache.stats().total_bytes
    second = _kenya_events(cache, _SUBMIT_OPTIONS, stream=True)
    assert cache.stats().hits == 1
    a, b = _from_response(first), _from_response(second)
    assert [e["kind"] for e in _without_place_options(a)] == [
        e["kind"] for e in _without_place_options(b)
    ]
    assert [e["kind"] for e in b][-len(_place_options(a)) - 1 :] == [
        *["place_options"] * len(_place_options(a)),
        "done",
    ], "a hit streams its place options after every replayed event"
    assert len(_place_options(b)) == [e["kind"] for e in b].count("place_options"), (
        "the bundle holds no place options, so a hit sends each once"
    )
    assert _replays(a, b)
    if path := os.environ.get("SSE_BUNDLE_METRICS"):
        with open(path, "a") as out:
            out.write(json.dumps({"case": "kenya-trailing", "bytes": entry_bytes}) + "\n")


@pytest.mark.parametrize("close_after", ["response", "slot_options"])
def test_a_stream_closed_before_done_stores_nothing_and_releases_its_lease(
    close_after: str,
) -> None:
    """A client that disconnects before `done` leaves no entry, and the next identical
    request leads its own run rather than waiting on the abandoned one."""
    cache = ResponseCache()
    materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
    closed = offline_events(
        _kenya_request(_SUBMIT_OPTIONS),
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        stream_slot_options=True,
        response_cache=cache,
        close_after=close_after,
    )
    assert closed[-1].kind == close_after
    assert cache.stats().entries == 0

    complete = _kenya_events(cache, _SUBMIT_OPTIONS, stream=True)
    assert isinstance(complete[-1], DoneEvent)
    stats = cache.stats()
    assert (stats.hits, stats.follower_waits) == (0, 0)
    assert stats.entries > 0


def test_a_suggestion_serves_the_submit_only_as_an_sse_request_with_the_submit_options() -> None:
    # A buffered suggestion with coverage off stores an entry that cannot serve the submit.
    today = ResponseCache()
    _kenya_events(today, _SUGGEST_OPTIONS, stream=False)
    assert today.stats().entries > 0
    _kenya_events(today, _SUBMIT_OPTIONS, stream=True)
    assert today.stats().hits == 0

    # A buffered suggestion carrying the submit's options still misses. It caches the
    # response, and the submit needs the bundle.
    buffered = ResponseCache()
    _kenya_events(buffered, _SUBMIT_OPTIONS, stream=False)
    _kenya_events(buffered, _SUBMIT_OPTIONS, stream=True)
    assert buffered.stats().hits == 0

    # An SSE suggestion with the submit's options stores the bundle. The submit hits it,
    # and so does a submit whose query differs only in whitespace.
    proposed = ResponseCache()
    _kenya_events(proposed, _SUBMIT_OPTIONS, stream=True)
    _kenya_events(proposed, _SUBMIT_OPTIONS, stream=True, query=f"  {_QUERY}\t")
    assert proposed.stats().hits == 1

    # Dropping coverage from the suggestion breaks the match even over SSE.
    partial = ResponseCache()
    _kenya_events(partial, _SUGGEST_OPTIONS, stream=True)
    _kenya_events(partial, _SUBMIT_OPTIONS, stream=True)
    assert partial.stats().hits == 0


# ---------------------------------------------------------------------------
# What the cache refuses to store.
# ---------------------------------------------------------------------------


@pytest.fixture
def kenya_bundle() -> list[Event]:
    events = _kenya_events(ResponseCache(), _SUBMIT_OPTIONS, stream=True)
    return events[next(i for i, e in enumerate(events) if isinstance(e, ResponseEvent)) :]


def _warning(code: str) -> Warning:
    return Warning(code=code, severity="warn", message="m")


@pytest.mark.parametrize(
    ("done_update", "timed_out", "degraded_leg", "trailing_incomplete", "storable"),
    [
        ({}, False, False, False, True),
        ({"timed_out": True}, False, False, False, False),
        ({}, True, False, False, False),
        ({}, False, True, False, False),
        ({}, False, False, True, False),
        ({"warnings": [_warning("option_coverage_probe_failed")]}, False, False, False, False),
        ({"warnings": [_warning("option_coverage_timed_out")]}, False, False, False, False),
        ({"warnings": [_warning("observations_probe_failed")]}, False, False, False, False),
        ({"warnings": [_warning("observations_timed_out")]}, False, False, False, False),
    ],
    ids=[
        "complete",
        "done-timed-out",
        "run-timed-out",
        "degraded-leg",
        "trailing-incomplete",
        "coverage-probe-failed",
        "coverage-timed-out",
        "observations-probe-failed",
        "observations-timed-out",
    ],
)
def test_a_bundle_is_stored_only_when_complete(
    kenya_bundle: list[Event],
    done_update: dict,
    timed_out: bool,
    degraded_leg: bool,
    trailing_incomplete: bool,
    storable: bool,
) -> None:
    events = [*kenya_bundle[:-1], kenya_bundle[-1].model_copy(update=done_update)]
    assert (
        is_bundle_storable(
            events,
            timed_out=timed_out,
            degraded_leg=degraded_leg,
            trailing_incomplete=trailing_incomplete,
            refused_at_build=frozenset(),
        )
        is storable
    )


_ESH_REFUSED = {"places": ["country/ESH"], "variables": [], "variable_labels": []}


@pytest.mark.parametrize(
    ("params", "refused_at_build", "storable"),
    [
        (None, frozenset({"country/ESH"}), False),
        (_ESH_REFUSED, frozenset({"country/ESH"}), True),
        (_ESH_REFUSED, frozenset(), False),
        (
            {"places": [], "variables": ["undata/sv"], "variable_labels": [""]},
            frozenset({"country/ESH"}),
            False,
        ),
    ],
    ids=["failed-un-leg", "refused-by-the-build", "refused-while-serving", "failed-un-read"],
)
@pytest.mark.parametrize("on", ["response", "done"])
def test_a_un_warning_blocks_storage_only_for_a_failed_read(
    kenya_bundle: list[Event],
    params: dict | None,
    refused_at_build: frozenset[str],
    storable: bool,
    on: str,
) -> None:
    """`un_system_unavailable` naming only places the origin catalogue build found refused
    is a fact of that build, so the answer is stored. One naming a place refused only while
    serving, failed reads in its `variables`, or a failed UN leg's warning, which has no
    `params`, is not."""
    warning = Warning(code="un_system_unavailable", severity="warn", message="m", params=params)
    response_event, done = kenya_bundle[0], kenya_bundle[-1]
    assert isinstance(response_event, ResponseEvent) and isinstance(done, DoneEvent)
    if on == "response":
        response = response_event.response
        diagnostics = response.diagnostics.model_copy(
            update={"warnings": [*response.diagnostics.warnings, warning]}
        )
        response_event = ResponseEvent(
            response=response.model_copy(update={"diagnostics": diagnostics})
        )
    else:
        done = done.model_copy(update={"warnings": [*done.warnings, warning]})
    events = [response_event, *kenya_bundle[1:-1], done]
    assert (
        is_bundle_storable(
            events,
            timed_out=False,
            degraded_leg=False,
            trailing_incomplete=False,
            refused_at_build=refused_at_build,
        )
        is storable
    )


# Each fault drops or degrades one trailing event of the Kenya stream without a warning.
# It patches the stream's own code in `mp`, appends to `fired` when it acts, and may
# return the place-options pool the faulted request runs with.
type _Fault = Callable[[pytest.MonkeyPatch, list[str]], PlaceOptionsPool | None]


def _raising(exc: BaseException, fired: list[str]) -> Callable[..., NoReturn]:
    def raise_(*_args: object, **_kwargs: object) -> NoReturn:
        fired.append(type(exc).__name__)
        raise exc

    return raise_


def _slot_options_graph_failure(mp: pytest.MonkeyPatch, fired: list[str]) -> None:
    mp.setattr(
        option_coverage, "_stream_slot_options_events", _raising(GraphInfraError("x"), fired)
    )


def _observations_graph_failure(mp: pytest.MonkeyPatch, fired: list[str]) -> None:
    mp.setattr(obs_dispatch, "_observations_events", _raising(GraphInfraError("x"), fired))


def _place_options_budget_expired(mp: pytest.MonkeyPatch, fired: list[str]) -> None:
    kwdefaults = enrichment._stream_slot_options.__kwdefaults__
    assert kwdefaults is not None
    mp.setitem(kwdefaults, "place_options_budget_s", 0.0)
    fired.append("budget")


def _place_options_pool_full(_mp: pytest.MonkeyPatch, fired: list[str]) -> PlaceOptionsPool:
    pool = PlaceOptionsPool.build(max_workers=1)
    assert pool.admission.acquire(blocking=False)
    fired.append("pool")
    return pool


def _target(exc: BaseException) -> _Fault:
    def fault(mp: pytest.MonkeyPatch, fired: list[str]) -> None:
        mp.setattr(place_options, "resolve_options_target", _raising(exc, fired))

    return fault


def _target_degraded(mp: pytest.MonkeyPatch, fired: list[str]) -> None:
    real = place_options.resolve_options_target

    def resolve(*args: Any, **kwargs: Any) -> Any:
        target = real(*args, **kwargs)
        fired.append("target")
        note_degraded_read("population")
        return target

    mp.setattr(place_options, "resolve_options_target", resolve)
    _scope_record_closes_inside_the_region(mp)


def _scope_graph_failure(mp: pytest.MonkeyPatch, fired: list[str]) -> None:
    mp.setattr(place_options, "compute_options", _raising(EngineInfraError("x"), fired))


def _scope(*, degraded: bool) -> _Fault:
    def fault(mp: pytest.MonkeyPatch, fired: list[str]) -> None:
        real = place_options.compute_options

        def build(*args: Any, **kwargs: Any) -> ComputedOptions:
            computed = real(*args, **kwargs)
            fired.append("scope")
            if degraded:
                note_degraded_read("breadth")
                return computed
            return replace(
                computed, warnings=(_warning("option_coverage_probe_failed"),), transient=True
            )

        mp.setattr(place_options, "compute_options", build)
        if degraded:
            _scope_record_closes_inside_the_region(mp)

    return fault


def _scope_record_closes_inside_the_region(mp: pytest.MonkeyPatch) -> None:
    """Hold the region's resolution open until a place-options job has closed its
    degraded-reads record, so the job's notes would reach the region's record if they
    passed to it, the order the worker threads run in now and then on CI."""
    closed = threading.Event()
    record = place_options.recording_degraded_reads
    fill_gap = pipeline._fill_gap

    @contextmanager
    def recorded(**kwargs: Any) -> Iterator[DegradedReads]:
        with record(**kwargs) as lost:
            yield lost
        closed.set()

    async def filled_after_a_scope_closed(*args: Any, **kwargs: Any) -> RegionResult:
        result = await fill_gap(*args, **kwargs)
        assert await asyncio.to_thread(closed.wait, 5), "no place-options job closed"
        return result

    mp.setattr(place_options, "recording_degraded_reads", recorded)
    mp.setattr(pipeline, "_fill_gap", filled_after_a_scope_closed)


@pytest.mark.parametrize(
    ("fault", "stored"),
    [
        (_target(EngineInputError("no place options for this anchor")), True),
        (_slot_options_graph_failure, False),
        (_observations_graph_failure, False),
        (_place_options_budget_expired, True),
        (_place_options_pool_full, True),
        (_target(PopulationInFlight()), True),
        (_target(EngineInfraError("x")), True),
        (_target(TimeoutError()), True),
        (_target_degraded, True),
        (_scope_graph_failure, True),
        (_scope(degraded=True), True),
        (_scope(degraded=False), True),
    ],
    ids=[
        "complete-empty-place-options",
        "slot-options-graph-failure",
        "observations-graph-failure",
        "place-options-budget-expired",
        "place-options-pool-full",
        "population-in-flight",
        "target-graph-failure",
        "target-timeout",
        "target-degraded",
        "scope-graph-failure",
        "scope-degraded",
        "scope-warned",
    ],
)
def test_a_stream_that_lost_a_trailing_event_stores_no_bundle(
    monkeypatch: pytest.MonkeyPatch, fault: _Fault, stored: bool
) -> None:
    """A slot-options or observations event dropped without a warning leaves no entry, and
    the next request computes afresh. Place options are not in the bundle, so a place-options
    drop or a degraded or transient place-options read leaves it stored."""
    cache = ResponseCache()
    fired: list[str] = []
    with monkeypatch.context() as mp:
        pool = fault(mp, fired)
        faulted = _kenya_events(cache, _SUBMIT_OPTIONS, stream=True, pool=pool)
    assert fired, "the fault never acted on the stream"
    assert isinstance(faulted[-1], DoneEvent)
    assert (cache.stats().entries > 0) is stored

    _kenya_events(cache, _SUBMIT_OPTIONS, stream=True)
    stats = cache.stats()
    assert (stats.hits, stats.entries > 0) == (int(stored), True)


class _CountingCompute:
    """`place_options.compute_options`, counting its calls, optionally holding each call
    until `release` is set and noting places the UN front door refuses inside its read."""

    def __init__(self, *, refused: tuple[str, ...] = (), hold: bool = False) -> None:
        self.calls = 0
        self.refused = refused
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self._real = place_options.compute_options
        self._observation_about = surface._observation_about_options_set

    def __call__(self, *args: Any, **kwargs: Any) -> ComputedOptions:
        self.calls += 1
        assert self.release.wait(timeout=5), "test setup: the compute was never released"
        return self._real(*args, **kwargs)

    def options_set(self, *args: Any, **kwargs: Any) -> Any:
        note_un_blocked_places(self.refused)
        return self._observation_about(*args, **kwargs)

    def install(self, mp: pytest.MonkeyPatch) -> None:
        mp.setattr(place_options, "compute_options", self)
        mp.setattr(surface, "_observation_about_options_set", self.options_set)


def _wait_for_entries(cache: OptionsResponseCache, entries: int) -> None:
    waited_until = time.monotonic() + 5
    while cache.stats()["entries"] < entries and time.monotonic() < waited_until:
        time.sleep(0.01)
    assert cache.stats()["entries"] == entries


def _fresh_place_options() -> dict[tuple[str, str], dict]:
    """The place options a fresh run with no cache streams."""
    return _place_options(
        _from_response(_kenya_events(ResponseCache(), _SUBMIT_OPTIONS, stream=True))
    )


@pytest.mark.parametrize(
    ("build_refuses", "options_stored"),
    [
        (frozenset({"country/ESH", "country/GRC"}), True),
        (frozenset({"country/ESH"}), False),
    ],
    ids=["refused-by-the-build", "refused-while-serving"],
)
def test_place_options_naming_refused_places_are_stored_and_served_on_a_hit(
    monkeypatch: pytest.MonkeyPatch,
    bind_catalogue: Callable[[FakeCatalogue], None],
    build_refuses: frozenset[str],
    options_stored: bool,
) -> None:
    """Peers whose read names places the UN front door refuses, as Ghana's inflation peers
    do, carry `un_system_unavailable`. When the origin catalogue build found every such place
    refused, that is a fact of the build, so the options cache keeps both scopes and the
    hit's place options come from it with no compute. A place refused only while serving
    keeps both scopes out, and the hit computes them again. The bundle is stored either way."""
    fresh = _fresh_place_options()
    bind_catalogue(FakeCatalogue(un_blocked_places=build_refuses))
    compute = _CountingCompute(refused=("country/ESH", "country/GRC"))
    compute.install(monkeypatch)
    cache, options_cache = ResponseCache(), OptionsResponseCache(enabled=True, max_entries=10)

    first = _kenya_events(cache, _SUBMIT_OPTIONS, stream=True, options_cache=options_cache)
    assert (cache.stats().entries > 0, options_cache.stats()["entries"]) == (
        True,
        2 if options_stored else 0,
    )
    calls = compute.calls
    second = _kenya_events(cache, _SUBMIT_OPTIONS, stream=True, options_cache=options_cache)

    assert cache.stats().hits == 1
    assert (compute.calls == calls) is options_stored
    assert _replays(_from_response(first), _from_response(second))
    assert _place_options(_from_response(second)) == fresh


@pytest.mark.parametrize("degraded", [False, True], ids=["clean-target", "degraded-target"])
def test_a_degraded_place_options_target_keeps_both_scopes_out_of_the_options_cache(
    monkeypatch: pytest.MonkeyPatch, degraded: bool
) -> None:
    """Every scope's options stand on the resolved target, so a read the target's resolution
    lost marks both scopes degraded and the options cache stores neither. The bundle is
    stored either way, since place options are not in it."""
    real = place_options.resolve_options_target

    def resolve(*args: Any, **kwargs: Any) -> Any:
        target = real(*args, **kwargs)
        if degraded:
            note_degraded_read("population")
        return target

    monkeypatch.setattr(place_options, "resolve_options_target", resolve)
    cache, options_cache = ResponseCache(), OptionsResponseCache(enabled=True, max_entries=10)

    _kenya_events(cache, _SUBMIT_OPTIONS, stream=True, options_cache=options_cache)

    assert cache.stats().entries > 0
    assert options_cache.stats()["entries"] == (0 if degraded else 2)


def test_a_cold_run_past_the_place_options_budget_stores_and_fills_the_options_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first run's place options overrun the budget and stream nothing. The bundle is
    stored anyway, and the worker jobs fill the options cache after the stream moved on, so
    the next request is a hit whose place options come from the options cache.

    The run keeps the 1 s production budget, so both scope jobs are running when it expires:
    a job still queued when its stream gives up is cancelled and computes nothing."""
    fresh = _fresh_place_options()
    compute = _CountingCompute(hold=True)
    compute.install(monkeypatch)
    cache, options_cache = ResponseCache(), OptionsResponseCache(enabled=True, max_entries=10)

    first = _kenya_events(cache, _SUBMIT_OPTIONS, stream=True, options_cache=options_cache)
    assert compute.calls == 2, "both scope jobs were running when the budget expired"
    compute.release.set()
    assert "place_options" not in [e.kind for e in first]
    assert cache.stats().entries > 0
    _wait_for_entries(options_cache, 2)

    calls = compute.calls
    second = _kenya_events(cache, _SUBMIT_OPTIONS, stream=True, options_cache=options_cache)
    assert cache.stats().hits == 1
    assert compute.calls == calls, "the hit's place options come from the options cache"
    assert _place_options(_from_response(second)) == fresh


def _merged_places_events(cache: ResponseCache) -> list[dict]:
    """A parsed request for one measure at Kenya and Ethiopia, whose parts merge into one Spec
    whose `observationAbout` holds both places, from `response` on."""
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text=place)])
                for place in ("Kenya", "Ethiopia")
            ]
        ),
        response_options=ResolveOptions(**_SUBMIT_OPTIONS),
    )
    materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
    # Enough permits that no Spec's stages meet a full pool.
    pool = PlaceOptionsPool.build(max_workers=6)
    try:
        return _from_response(
            offline_events(
                request,
                sdmx=FakeSdmxClient(),
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
                stream_slot_options=True,
                response_cache=cache,
                place_options_pool=pool,
            )
        )
    finally:
        pool.shutdown()


def test_a_merged_spec_streams_the_same_place_options_live_and_on_a_hit() -> None:
    """A where merge widens the merged Spec's `observationAbout` to both places. The live run
    and a hit both build place options from that Spec's identity, so they stream the same
    events, and `children`, which a two-anchor identity refuses, on neither."""
    cache = ResponseCache()
    live = _merged_places_events(cache)
    hit = _merged_places_events(cache)

    response = live[0]["response"]
    (spec_id,) = {part["spec"]["spec_id"] for part in response["parts"]}
    anchors = response["parts"][0]["spec"]["definition"]["identity"]["entities"]
    assert sorted(anchors["observationAbout"]) == ["country/ETH", "country/KEN"]
    assert cache.stats().hits == 1
    assert set(_place_options(live)) == {(spec_id, "peers")}
    assert _replays(live, hit)


class _FailingRelatedSubjects(FakeGraph):
    """The related-measures read's reverse read of the group fails (a degraded leg)."""

    def node_subjects_many(self, *args, **kwargs):
        raise GraphInfraError("the reverse read failed")


@pytest.mark.parametrize(
    ("graph_type", "stored"), [(FakeGraph, True), (_FailingRelatedSubjects, False)]
)
def test_an_sse_answer_that_swallowed_a_read_is_not_stored(graph_type, stored: bool) -> None:
    """The degraded-leg flag rides on no event, so the scope's own health note decides."""
    cache = ResponseCache()
    materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
    for _ in range(2):
        offline_events(
            _kenya_request(_SUBMIT_OPTIONS),
            graph=graph_type(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            stream_slot_options=True,
            response_cache=cache,
        )
    assert (cache.stats().entries > 0, cache.stats().hits) == (stored, int(stored))


# ---------------------------------------------------------------------------
# The SSE route, in process.
# ---------------------------------------------------------------------------


def _sse_frames(text: str) -> list[dict]:
    return [
        {"kind": kind, **json.loads(data)}
        for kind, data in re.findall(r"^event: (\w+)\ndata: (.*)$", text, re.MULTILINE)
    ]


def test_the_sse_route_serves_an_identical_request_from_the_cache() -> None:
    materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
    graph = FakeGraph()
    app = create_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    body = {
        "schema_version": 1,
        "input": {"kind": "raw_text", "query": _QUERY},
        "response_options": _SUBMIT_OPTIONS,
    }
    with (
        TestClient(app, raise_server_exceptions=False) as client,
        patch("search.engine.extract.date") as mock_date,
    ):
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        cache = getattr(app.state, SEARCH_DEPS_SLOT.name).response_cache
        runs = []
        for _ in range(2):
            resp = client.post(
                "/api/search/resolve", json=body, headers={"Accept": "text/event-stream"}
            )
            assert resp.status_code == 200
            runs.append(_sse_frames(resp.text))
    first, second = (r[[f["kind"] for f in r].index("response") :] for r in runs)
    assert "place_options" in [f["kind"] for f in first]
    assert cache.stats().hits == 1
    assert _replays(first, second)
