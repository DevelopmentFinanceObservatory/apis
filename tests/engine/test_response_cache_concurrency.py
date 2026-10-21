"""Pins concurrent follower behaviour in both response-cache layers."""

from __future__ import annotations

import asyncio
import time
from datetime import date
from unittest.mock import patch

import search.engine.fanout as fanout_module
from search.engine.binders import Fallbacks
from search.engine.core import resolve_async, resolve_events
from search.engine.envelope import now_ms
from search.engine.events import DoneEvent, Event, ResponseEvent, StepEvent
from search.engine.extract import Extraction, ExtractionResult
from search.engine.region import RegionResult
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_flow import CacheScope
from search.models import RawTextInput, ResolveRequest, ResolveResponse
from tests.engine._binders import KeyedBinder
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    DEADLINE_TIMEOUT_S,
    EMPTY_SDMX,
    frozen_monotonic_clock,
    make_request,
    offline_resolve,
)
from tests.engine.test_response_cache_e2e import _QUERY_A, _QUERY_B, _AliasedLLM, _CountingLLM
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM

_BUILD_COMMIT = "test-build"
_MODEL = "test-model"
_SEED = 1
_TODAY = "2026-06-23"
_MAX_CANDIDATES = 6
_SEAM_DEFAULT = True
_DEADLINE_HORIZON_S = 60.0


def _base_response() -> ResolveResponse:
    return offline_resolve(make_request("total population India"))


def _new_cache() -> ResponseCache:
    return ResponseCache(max_bytes=10_000_000, max_entry_bytes=10_000_000, ttl_s=60)


def _make_scope(cache: ResponseCache | None, *, deadline: float | None = None) -> CacheScope:
    return CacheScope(
        cache,
        request=ResolveRequest(input=RawTextInput(query="placeholder")),
        start_ms=now_ms(),
        deadline=deadline if deadline is not None else time.monotonic() + _DEADLINE_HORIZON_S,
        build_commit=_BUILD_COMMIT,
        binder=KeyedBinder(model=_MODEL),
        fallbacks=Fallbacks(),
        seed=_SEED,
        today=_TODAY,
        max_candidates=_MAX_CANDIDATES,
        seam_default=_SEAM_DEFAULT,
        coverage=empty_coverage_snapshot(),
        trailing=False,
    )


async def _test_leader_raising_lets_exactly_one_follower_become_the_new_leader(
    base: ResolveResponse,
):
    cache = _new_cache()
    query = "q-stampede"
    pipeline_runs = 0
    raised_once = False

    async def _run_one() -> ResolveResponse | None:
        nonlocal pipeline_runs, raised_once
        scope = _make_scope(cache)
        hit = await scope.lookup_raw(query)
        if hit is not None:
            return hit
        if scope._layer1_lease is None:
            return None  # this deadline is generous; not exercised here
        pipeline_runs += 1
        # Let all callers register as followers before the leader settles.
        await asyncio.sleep(0.02)
        if not raised_once:
            raised_once = True
            await scope.release()  # this leader's pipeline run raised
            return None
        await scope.store(base)
        return base

    n = 5
    results = await asyncio.gather(*[_run_one() for _ in range(n)])

    assert pipeline_runs == 2  # initial attempt plus one retry
    assert results.count(None) == 1  # the original leader gets no result
    for result in results:
        if result is not None:
            assert result.parts == base.parts


async def _test_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time(
    base: ResolveResponse,
):
    cache = _new_cache()
    query = "q-short-deadline"
    leader = _make_scope(cache, deadline=time.monotonic() + 60)
    assert await leader.lookup_raw(query) is None

    follower = _make_scope(cache, deadline=time.monotonic() + 0.05)
    started = time.monotonic()
    result = await asyncio.wait_for(follower.lookup_raw(query), timeout=5)
    elapsed = time.monotonic() - started

    assert result is None
    assert follower._layer1_lease is None  # gave up; never bid for the key either
    assert elapsed < 1.0  # bounded by the follower's own ~0.05s deadline

    await leader.store(base)  # the leader is unaffected by the follower giving up


def _no_data_region(variable: str) -> RegionResult:
    return RegionResult(
        variable_text=variable,
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
    )


def _mock_extract(variables: list[str]):
    async def _inner(query: str, *, llm=None):
        return (
            ExtractionResult(
                extraction=Extraction(
                    kind="resolution", variables=variables, entities=[], dates=[]
                ),
                qualifiers={},
            ),
            None,
        )

    return _inner


async def _collect(gen) -> list[Event]:
    return [ev async for ev in gen]


def _stall_leader_past_deadline_once_followers_wait(monkeypatch, cache, clock, *, n_followers):
    """A `resolve_variable` stand-in for the leader's fan-out, and the variables it was given.

    It waits until `n_followers` requests are waiting on a leader in `cache`, then moves the
    fake `clock` past the request deadline and wakes the fan-out loop. The deadline passes
    only once the followers wait, so no step of the leader or a follower races it.
    """
    waiting = 0
    all_waiting = asyncio.Event()
    real_await_leader = cache.await_leader

    async def _counting_await_leader(lease):
        nonlocal waiting
        waiting += 1
        if waiting >= n_followers:
            all_waiting.set()
        return await real_await_leader(lease)

    monkeypatch.setattr(cache, "await_leader", _counting_await_leader)

    calls: list[str] = []

    async def _stub(variable, **kwargs):
        calls.append(variable)
        await all_waiting.wait()
        clock.now += DEADLINE_TIMEOUT_S * 10
        await kwargs["emit"](StepEvent(step="materialise", ran=True))
        await asyncio.Event().wait()
        return _no_data_region(variable)  # pragma: no cover (the deadline fires first)

    return _stub, calls


def test_leader_timeout_wakes_every_follower_with_timed_out_done(monkeypatch, search_settings):
    """N concurrent identical raw_text requests sharing one ResponseCache.
    The leader's own fan-out hits SEARCH_REQUEST_TIMEOUT_S and produces a
    partial, is_storable-rejected response. Every follower gets that same
    partial and its own done(timed_out=true) rather than waiting past its
    own deadline into a 504. The fake clock passes the deadline once every follower waits.
    """
    clock = frozen_monotonic_clock(monkeypatch)
    search_settings(request_timeout_s=DEADLINE_TIMEOUT_S)
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["only"]))

    cache = ResponseCache()
    query = "test leader timeout wakes every follower"
    n_followers = 2
    stub, _ = _stall_leader_past_deadline_once_followers_wait(
        monkeypatch, cache, clock, n_followers=n_followers
    )
    monkeypatch.setattr("search.engine.fanout.resolve_variable", stub)

    async def _run() -> tuple[list[Event], list[list[Event]]]:
        leader_task = asyncio.ensure_future(
            _collect(
                resolve_events(
                    make_request(query),
                    graph=FakeGraph(),
                    llm=FakeLLM(),
                    sdmx=EMPTY_SDMX,
                    response_cache=cache,
                )
            )
        )
        await asyncio.sleep(0)  # let the leader claim the layer-1 lease first

        follower_tasks = [
            asyncio.ensure_future(
                _collect(
                    resolve_events(
                        make_request(query),
                        graph=FakeGraph(),
                        llm=FakeLLM(),
                        sdmx=EMPTY_SDMX,
                        response_cache=cache,
                    )
                )
            )
            for _ in range(n_followers)
        ]
        await asyncio.sleep(0)  # let every follower register as a non-leader lease

        gathered = await asyncio.gather(leader_task, *follower_tasks)
        return gathered[0], list(gathered[1:])

    leader_events, follower_events = asyncio.run(asyncio.wait_for(_run(), timeout=10))

    leader_dones = [e for e in leader_events if isinstance(e, DoneEvent)]
    assert len(leader_dones) == 1
    assert leader_dones[0].timed_out is True

    for events in follower_events:
        responses = [e for e in events if isinstance(e, ResponseEvent)]
        dones = [e for e in events if isinstance(e, DoneEvent)]
        assert len(responses) == 1  # followers receive the partial answer
        assert len(dones) == 1
        assert dones[0].timed_out is True


def test_two_differently_worded_queries_run_concurrently_bind_once(monkeypatch):
    """Sequential layer-2 hits are covered by
    test_response_cache_e2e.py's test_differently_worded_query_hits_layer2;
    this drives the same query pair through asyncio.gather instead, so the
    second wording's own lookup_parsed call races the first's while it is
    still in flight rather than finding it already stored. Slows the
    leader's own fan-out (fanout.resolve_variable) so the second query's
    extract() and lookup_parsed call are guaranteed to land before the
    leader's bind finishes and stores.
    """
    original_resolve_variable = fanout_module.resolve_variable

    async def _slow_resolve_variable(*args, **kwargs):
        await asyncio.sleep(0.1)
        return await original_resolve_variable(*args, **kwargs)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _slow_resolve_variable)

    llm = _CountingLLM(_AliasedLLM(alias_query=_QUERY_B, canonical_query=_QUERY_A))
    cache = ResponseCache()

    async def _run_one(query: str) -> ResolveResponse:
        return await resolve_async(
            make_request(query),
            graph=FakeGraph(),
            llm=llm,
            sdmx=EMPTY_SDMX,
            response_cache=cache,
        )

    async def _run_both() -> tuple[ResolveResponse, ResolveResponse]:
        return await asyncio.gather(_run_one(_QUERY_A), _run_one(_QUERY_B))

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        r1, r2 = asyncio.run(asyncio.wait_for(_run_both(), timeout=10))

    assert llm.calls.count("_UniversalBindOutput") == 1  # single-flight, not once per wording
    assert r1.parts == r2.parts
    assert r1.query_echo.raw_query == _QUERY_A
    assert r2.query_echo.raw_query == _QUERY_B


def test_layer2_leader_timeout_wakes_the_layer2_follower_with_timed_out_done(
    monkeypatch, search_settings
):
    clock = frozen_monotonic_clock(monkeypatch)
    search_settings(request_timeout_s=DEADLINE_TIMEOUT_S)
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["only"]))

    cache = ResponseCache()
    stub, fan_out_calls = _stall_leader_past_deadline_once_followers_wait(
        monkeypatch, cache, clock, n_followers=1
    )
    monkeypatch.setattr("search.engine.fanout.resolve_variable", stub)

    query_a = "layer two timeout alpha wording"
    query_b = "layer two timeout beta wording"

    async def _run() -> tuple[list[Event], list[Event]]:
        leader_task = asyncio.ensure_future(
            _collect(
                resolve_events(
                    make_request(query_a),
                    graph=FakeGraph(),
                    llm=FakeLLM(),
                    sdmx=EMPTY_SDMX,
                    response_cache=cache,
                )
            )
        )
        await asyncio.sleep(0)  # let the leader claim the layer-2 lease first

        follower_task = asyncio.ensure_future(
            _collect(
                resolve_events(
                    make_request(query_b),
                    graph=FakeGraph(),
                    llm=FakeLLM(),
                    sdmx=EMPTY_SDMX,
                    response_cache=cache,
                )
            )
        )
        await asyncio.sleep(0)  # let the follower register as a non-leader layer-2 lease

        return await asyncio.gather(leader_task, follower_task)

    leader_events, follower_events = asyncio.run(asyncio.wait_for(_run(), timeout=10))

    assert len(fan_out_calls) == 1  # the follower never ran its own fan-out

    leader_dones = [e for e in leader_events if isinstance(e, DoneEvent)]
    assert len(leader_dones) == 1
    assert leader_dones[0].timed_out is True

    follower_responses = [e for e in follower_events if isinstance(e, ResponseEvent)]
    follower_dones = [e for e in follower_events if isinstance(e, DoneEvent)]
    assert len(follower_responses) == 1  # a real, shared answer, never a fresh run
    assert len(follower_dones) == 1
    assert follower_dones[0].timed_out is True


async def _test_layer2_leader_raising_lets_exactly_one_follower_become_the_new_leader(
    base: ResolveResponse,
):
    cache = _new_cache()
    extraction = {
        "variables": ["population"],
        "entities": [],
        "dates": [],
        "dropped_qualifiers": [],
        "variable_qualifiers": [()],
        "operations": [()],
        "entity_roles": {},
        "sources": [()],
        "query_text": None,
    }
    pipeline_runs = 0
    raised_once = False

    async def _run_one() -> ResolveResponse | None:
        nonlocal pipeline_runs, raised_once
        scope = _make_scope(cache)
        hit = await scope.lookup_parsed(query="q-layer2-stampede", **extraction)
        if hit is not None:
            return hit
        if scope._layer2_lease is None:
            return None  # this deadline is generous; not exercised here
        pipeline_runs += 1
        # Let all callers register as followers before the leader settles.
        await asyncio.sleep(0.02)
        if not raised_once:
            raised_once = True
            await scope.release()  # this leader's pipeline run raised
            return None
        await scope.store(base)
        return base

    n = 5
    results = await asyncio.gather(*[_run_one() for _ in range(n)])

    assert pipeline_runs == 2  # initial attempt plus one retry
    assert results.count(None) == 1  # the original leader gets no result
    for result in results:
        if result is not None:
            assert result.parts == base.parts


def test_leader_raising_lets_exactly_one_follower_become_the_new_leader():
    asyncio.run(
        _test_leader_raising_lets_exactly_one_follower_become_the_new_leader(_base_response())
    )


def test_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time():
    asyncio.run(
        _test_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time(_base_response())
    )


def test_layer2_leader_raising_lets_exactly_one_follower_become_the_new_leader():
    asyncio.run(
        _test_layer2_leader_raising_lets_exactly_one_follower_become_the_new_leader(
            _base_response()
        )
    )
