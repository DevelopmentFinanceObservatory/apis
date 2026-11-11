"""Pins `CacheScope` key lookup, publication, lease handling, and health tracking."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import search.engine.clients.llm as llm_module
from graph_generation import UNKNOWN_GENERATION, FakeGeneration, GenerationProvider
from origin_catalogue import bound_catalogue
from search.engine import config, response_cache_flow
from search.engine.binders import Binder, Fallbacks, configured_binder
from search.engine.clients.llm import (
    LLM,
    UNCOUNTED_CALLS,
    Agreement,
    SecondSample,
    SecondSamples,
    SupportsLLM,
    two_samples,
)
from search.engine.degraded_reads import DegradedReads
from search.engine.envelope import _terminal_events, now_ms
from search.engine.errors import LLMInfraError
from search.engine.events import DoneEvent
from search.engine.extract import ExtractedEntity
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_flow import CacheScope
from search.engine.response_cache_keys import to_entry
from search.engine.stated_scope import part_scope_intakes
from search.engine.unit.conversion_catalogue import (
    ConversionCatalogue,
    bound_conversion_catalogue,
)
from search.models import (
    DatasetResubmitInput,
    ParsedPart,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResolveResponse,
    ResubmitParsedEntry,
    Warning,
)
from search.scope.state import ScopeIntake
from series_coverage.shared import CoverageSnapshot
from tests.engine._binders import KeyedBinder
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    make_request,
    member_walk_reader,
    observe_answer,
    offline_resolve,
)
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

_BUILD_COMMIT = "test-build"
_MODEL = "test-model"
_SEED = 1
_TODAY = "2026-06-23"
_MAX_CANDIDATES = 6
_SEAM_DEFAULT = True


def _base_response() -> ResolveResponse:
    return offline_resolve(
        make_request("total population India"), materialise_graph=member_walk_reader()
    )


def _new_cache() -> ResponseCache:
    return ResponseCache(max_bytes=10_000_000, max_entry_bytes=10_000_000, ttl_s=60)


_DEADLINE_HORIZON_S = 60.0  # generous: only the deadline-specific tests below need a tight one


_COVERAGE_SNAPSHOT = empty_coverage_snapshot()


def _make_scope(
    cache: ResponseCache | None,
    *,
    request: ResolveRequest | None = None,
    deadline: float | None = None,
    generation: GenerationProvider = UNKNOWN_GENERATION,
    coverage: CoverageSnapshot = _COVERAGE_SNAPSHOT,
    trailing: bool = False,
    binder: Binder | None = None,
    fallbacks: Fallbacks | None = None,
    request_reads: DegradedReads | None = None,
) -> CacheScope:
    return CacheScope(
        cache,
        request=request or ResolveRequest(input=RawTextInput(query="placeholder")),
        start_ms=now_ms(),
        deadline=deadline if deadline is not None else time.monotonic() + _DEADLINE_HORIZON_S,
        build_commit=_BUILD_COMMIT,
        binder=binder if binder is not None else KeyedBinder(model=_MODEL),
        fallbacks=fallbacks if fallbacks is not None else Fallbacks(),
        request_reads=request_reads if request_reads is not None else DegradedReads(),
        seed=_SEED,
        today=_TODAY,
        max_candidates=_MAX_CANDIDATES,
        seam_default=_SEAM_DEFAULT,
        generation=generation,
        coverage=coverage,
        trailing=trailing,
    )


async def _assert_inactive_noop(scope: CacheScope, base: ResolveResponse) -> None:
    assert scope.active is False
    assert await scope.lookup_raw("q") is None
    assert (
        await scope.lookup_parsed(
            variables=[],
            entities=[],
            dates=[],
            dropped_qualifiers=[],
            variable_qualifiers=[],
            operations=[],
            query="q",
            entity_roles=(),
            sources=[],
            query_text=None,
        )
        is None
    )
    scope.note_health(timed_out=True, degraded_leg=True)
    await observe_answer(scope, base)
    await scope.release()


async def _test_inactive_when_cache_is_none(base: ResolveResponse):
    await _assert_inactive_noop(_make_scope(None), base)


async def _test_inactive_for_a_dataset_resubmit(base: ResolveResponse):
    resubmit_request = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[ResubmitParsedEntry(part=ParsedPart(variable_text="gdp"))]
        )
    )
    await _assert_inactive_noop(_make_scope(_new_cache(), request=resubmit_request), base)


async def _test_a_transport_with_trailing_events_never_reads_a_response_entry(
    base: ResolveResponse,
):
    cache = _new_cache()
    options = ResolveOptions(include_observations=True)
    request = ResolveRequest(input=RawTextInput(query="q"), response_options=options)
    stored = _make_scope(cache, request=request)
    assert await stored.lookup_raw("q") is None
    await observe_answer(stored, base)
    assert cache.stats().entries == 1

    streaming = _make_scope(cache, request=request, trailing=True)
    assert streaming.active
    assert await streaming.lookup_raw("q") is None
    await streaming.release()


async def _test_full_miss_then_store_then_layer1_hit_carries_the_new_wording(base: ResolveResponse):
    cache = _new_cache()
    query_1 = "total population india"
    scope_1 = _make_scope(cache)
    assert await scope_1.lookup_raw(query_1) is None
    await observe_answer(scope_1, base)

    query_2 = "total  population   india"  # same normalised key (whitespace only), different literal wording
    scope_2 = _make_scope(cache)
    restored = await scope_2.lookup_raw(query_2)

    assert restored is not None
    assert restored.query_echo.raw_query == query_2
    assert restored.query_echo.extract_skipped is True


async def _test_layer2_hit_for_a_reworded_query_with_the_same_extraction(base: ResolveResponse):
    cache = _new_cache()
    extraction = {
        "variables": ["population"],
        "entities": [ExtractedEntity(text="India")],
        "dates": [],
        "dropped_qualifiers": [],
        "variable_qualifiers": [()],
        "operations": [()],
        "entity_roles": (),
        "sources": [()],
        "query_text": None,
    }

    query_1 = "total population india"
    scope_1 = _make_scope(cache)
    assert await scope_1.lookup_raw(query_1) is None
    assert await scope_1.lookup_parsed(query=query_1, **extraction) is None
    await observe_answer(scope_1, base)

    query_2 = "how many people live in india"
    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query_2) is None  # a different query string misses layer 1
    hit = await scope_2.lookup_parsed(query=query_2, **extraction)
    await scope_2.release()  # the layer-1 lease scope_2 claimed above is never filled

    assert hit is not None
    assert hit.query_echo.raw_query == query_2
    assert hit.query_echo.extract_skipped is False


async def _test_layer2_hit_does_not_reset_the_shared_entrys_ttl(base: ResolveResponse):
    """A layer-2 hit does not refresh its entry's TTL, even when `observe_answer` runs."""
    cache = _new_cache()
    extraction = {
        "variables": ["population"],
        "entities": [ExtractedEntity(text="India")],
        "dates": [],
        "dropped_qualifiers": [],
        "variable_qualifiers": [()],
        "operations": [()],
        "entity_roles": (),
        "sources": [()],
        "query_text": None,
    }

    query_1 = "total population india"
    scope_1 = _make_scope(cache)
    assert await scope_1.lookup_raw(query_1) is None
    assert await scope_1.lookup_parsed(query=query_1, **extraction) is None
    await observe_answer(scope_1, base)

    layer2_key = scope_1._layer2_key
    assert layer2_key is not None
    expires_at_after_first_store = cache._store[layer2_key].expires_at

    query_2 = "how many people live in india"
    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query_2) is None
    hit = await scope_2.lookup_parsed(query=query_2, **extraction)
    assert hit is not None
    await observe_answer(scope_2, hit)  # the hit path publishing, as core.py drives it

    assert cache._store[layer2_key].expires_at == expires_at_after_first_store
    # The layer-1 write for query_2's own literal wording still happens: a
    # request answered from layer 2 is a legitimate first write for its own
    # wording, and a repeat of that same wording now hits layer 1 instead.
    assert await _new_scope_hits_layer1(cache, query_2)


async def _new_scope_hits_layer1(cache: ResponseCache, query: str) -> bool:
    scope = _make_scope(cache)
    return await scope.lookup_raw(query) is not None


def _population_india_extraction() -> dict:
    return {
        "variables": ["population"],
        "entities": [ExtractedEntity(text="India")],
        "dates": [],
        "dropped_qualifiers": [],
        "variable_qualifiers": [()],
        "operations": [()],
        "entity_roles": (),
        "sources": [()],
        "query_text": None,
    }


def test_single_variable_queries_cannot_share_an_unindexed_value_grounding_result():
    base = _base_response()

    async def run():
        cache = _new_cache()
        extraction = _population_india_extraction()
        female = "female population India"
        male = "male population India"
        writer = _make_scope(cache)
        assert await writer.lookup_raw(female) is None
        extraction["query_text"] = female
        assert await writer.lookup_parsed(query=female, **extraction) is None
        await observe_answer(writer, base)

        reader = _make_scope(cache)
        assert await reader.lookup_raw(male) is None
        extraction["query_text"] = male
        assert await reader.lookup_parsed(query=male, **extraction) is None
        await reader.release()

    asyncio.run(run())


def test_queries_whose_words_state_another_scope_never_share_a_layer2_entry():
    """Two queries of several variables with one extraction, one of them stating
    "bilateral" in a variable's words, key apart, and a repeat of the scope shares."""
    base = _base_response()

    async def run():
        cache = _new_cache()
        extraction = _population_india_extraction()
        extraction.update(
            variables=["ODA to India", "GDP of India"],
            variable_qualifiers=[(), ()],
            operations=[(), ()],
            sources=[(), ()],
        )
        bilateral = part_scope_intakes(
            message=None, part_texts=["bilateral ODA to India", "GDP of India"]
        )
        plain = [ScopeIntake(), ScopeIntake()]
        writer = _make_scope(cache)
        assert await writer.lookup_raw("bilateral ODA and GDP of India") is None
        assert (
            await writer.lookup_parsed(
                query="bilateral ODA and GDP of India", stated_scopes=bilateral, **extraction
            )
            is None
        )
        await observe_answer(writer, base)

        reader = _make_scope(cache)
        assert await reader.lookup_raw("ODA and GDP of India") is None
        assert (
            await reader.lookup_parsed(
                query="ODA and GDP of India", stated_scopes=plain, **extraction
            )
            is None
        )
        await reader.release()

        repeat = _make_scope(cache)
        assert await repeat.lookup_raw("bilateral ODA, and GDP of India") is None
        hit = await repeat.lookup_parsed(
            query="bilateral ODA, and GDP of India", stated_scopes=bilateral, **extraction
        )
        await repeat.release()
        assert hit is not None

    asyncio.run(run())


async def _woken_layer2_follower(
    cache: ResponseCache, extraction: dict, *, query_1: str, query_2: str
) -> tuple[CacheScope, CacheScope, ResolveResponse]:
    """Starts a layer-2 leader and registers a differently worded follower before publication."""
    leader = _make_scope(cache)
    assert await leader.lookup_raw(query_1) is None
    assert await leader.lookup_parsed(query=query_1, **extraction) is None

    follower = _make_scope(cache)
    assert await follower.lookup_raw(query_2) is None  # its own, distinct layer-1 key
    follower_task = asyncio.ensure_future(follower.lookup_parsed(query=query_2, **extraction))
    await asyncio.sleep(0)  # let the follower register as a non-leader layer-2 lease
    return leader, follower, follower_task


async def _done_of_hit(scope: CacheScope, response: ResolveResponse) -> DoneEvent:
    [*_, done] = [event async for event in scope.hit_events(response, start_ms=now_ms())]
    assert isinstance(done, DoneEvent)
    return done


async def _test_layer2_follower_of_a_timed_out_leader_gets_no_layer1_entry(base: ResolveResponse):
    cache = _new_cache()
    extraction = _population_india_extraction()
    query_1, query_2 = "total population india", "how many people live in india"

    leader, follower, follower_task = await _woken_layer2_follower(
        cache, extraction, query_1=query_1, query_2=query_2
    )
    leader.note_health(timed_out=True, degraded_leg=False)
    await observe_answer(leader, base)

    result = await follower_task
    assert result is not None
    assert (await _done_of_hit(follower, result)).timed_out is True
    assert follower.outcome == "hit layer2"
    await observe_answer(
        follower, result
    )  # the ResponseEvent branch core.py drives on every response

    # Neither wording's own layer-1 entry persists: the leader's result was
    # unstorable, and the publish's served-verdict check inherits that rather
    # than caching a truncated answer as if it were complete.
    assert await _new_scope_hits_layer1(cache, query_1) is False
    assert await _new_scope_hits_layer1(cache, query_2) is False


async def _test_layer2_follower_of_a_degraded_leader_gets_no_layer1_entry(base: ResolveResponse):
    """A follower preserves the leader's degraded verdict when caching its own wording."""
    cache = _new_cache()
    extraction = _population_india_extraction()
    query_1, query_2 = "total population india", "how many people live in india"

    leader, follower, follower_task = await _woken_layer2_follower(
        cache, extraction, query_1=query_1, query_2=query_2
    )
    leader.note_health(timed_out=False, degraded_leg=True)
    await observe_answer(leader, base)

    result = await follower_task
    assert result is not None
    assert follower.outcome == "hit layer2"
    assert (await _done_of_hit(follower, result)).degraded_leg is True
    # A follower that publishes passes the verdict on to its own followers.
    second = _make_scope(cache)
    second_task = asyncio.ensure_future(second.lookup_raw(query_2))
    await asyncio.sleep(0)
    await observe_answer(follower, result)
    second_result = await second_task
    assert second_result is not None
    assert (await _done_of_hit(second, second_result)).degraded_leg is True

    assert await _new_scope_hits_layer1(cache, query_1) is False
    assert await _new_scope_hits_layer1(cache, query_2) is False


async def _test_layer2_follower_of_a_healthy_leader_gets_its_own_layer1_entry(
    base: ResolveResponse,
):
    """Control for the two tests above: a leader with nothing to report to
    `note_health` produces a genuinely storable answer, and its woken
    follower's own literal wording DOES get a layer-1 entry.
    """
    cache = _new_cache()
    extraction = _population_india_extraction()
    query_1, query_2 = "total population india", "how many people live in india"

    leader, follower, follower_task = await _woken_layer2_follower(
        cache, extraction, query_1=query_1, query_2=query_2
    )
    await observe_answer(leader, base)  # healthy: no note_health call at all

    result = await follower_task
    assert result is not None
    assert follower.outcome == "hit layer2"
    await observe_answer(follower, result)

    assert await _new_scope_hits_layer1(cache, query_1)
    assert await _new_scope_hits_layer1(cache, query_2)


async def _test_layer2_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time(
    base: ResolveResponse,
):
    cache = _new_cache()
    extraction = _population_india_extraction()

    query_1 = "total population india"
    leader = _make_scope(cache, deadline=time.monotonic() + 60)
    assert await leader.lookup_raw(query_1) is None
    assert await leader.lookup_parsed(query=query_1, **extraction) is None

    query_2 = "how many people live in india"
    follower = _make_scope(cache, deadline=time.monotonic() + 0.05)
    assert await follower.lookup_raw(query_2) is None
    started = time.monotonic()
    result = await follower.lookup_parsed(query=query_2, **extraction)
    elapsed = time.monotonic() - started

    assert result is None
    assert follower._layer2_lease is None  # gave up, and never bid for the key either
    assert elapsed < 1.0  # bounded by the follower's own ~0.05s deadline, not left hanging

    # The leader is still free to finish afterward, unaffected by the
    # follower having given up on it.
    await observe_answer(leader, base)
    assert await _new_scope_hits_layer1(cache, query_1)


async def _test_store_publishes_both_leases_when_this_request_holds_both(
    base: ResolveResponse, *, agreed: bool, pending: bool
):
    """Layer 2 stores the answer either way; layer 1 stores it under the query only when two
    extraction samples agreed on its reading, and a layer-1 waiter receives it either way.
    A second sample still running when the answer is published is awaited first."""
    cache = _new_cache()
    extraction = _population_india_extraction()
    query = "total population india"
    scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    assert await scope.lookup_parsed(query=query, **extraction) is None
    assert scope._layer1_lease is not None
    assert scope._layer2_lease is not None
    if pending:

        async def second() -> tuple[None, None]:
            await asyncio.sleep(0.01)
            return None, None

        scope.note_extraction(Agreement(_held(second), lambda _second: agreed))
    else:
        scope.note_extraction(Agreement.fixed(agreed))
    waiter = _make_scope(cache)
    waiting = asyncio.ensure_future(waiter.lookup_raw(query))
    await asyncio.sleep(0)

    await observe_answer(scope, base)

    assert await waiting is not None
    assert scope._layer1_lease is None
    assert scope._layer2_lease is None
    assert await _new_scope_hits_layer1(cache, query) is agreed

    other = _make_scope(cache)
    assert await other.lookup_raw("a wording that is not the one stored above") is None
    hit = await other.lookup_parsed(
        query="a wording that is not the one stored above", **extraction
    )
    await other.release()
    assert hit is not None  # the layer-2 write from the same observed answer also landed


async def _test_release_abandons_both_leases_when_this_request_holds_both():
    cache = _new_cache()
    extraction = _population_india_extraction()
    query = "q-release-both-layers"
    scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    assert await scope.lookup_parsed(query=query, **extraction) is None
    assert scope._layer1_lease is not None
    assert scope._layer2_lease is not None

    await scope.release()

    assert scope._layer1_lease is None
    assert scope._layer2_lease is None

    layer1_successor = _make_scope(cache)
    assert await layer1_successor.lookup_raw(query) is None
    assert layer1_successor._layer1_lease is not None and layer1_successor._layer1_lease.is_leader

    layer2_successor = _make_scope(cache)
    assert await layer2_successor.lookup_parsed(query=query, **extraction) is None
    assert layer2_successor._layer2_lease is not None and layer2_successor._layer2_lease.is_leader

    await layer1_successor.release()
    await layer2_successor.release()


async def _test_store_does_not_persist_when_timed_out_was_noted(base: ResolveResponse):
    cache = _new_cache()
    query = "q-timed-out"
    scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    scope.note_health(timed_out=True, degraded_leg=False)
    await observe_answer(scope, base)

    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query) is None


async def _test_store_does_not_persist_when_a_leg_degraded(base: ResolveResponse):
    cache = _new_cache()
    query = "q-degraded"
    scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    scope.note_health(timed_out=False, degraded_leg=True)
    await observe_answer(scope, base)

    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query) is None


async def _test_note_health_true_sticks_even_after_a_later_false_report(base: ResolveResponse):
    cache = _new_cache()
    query = "q-sticky"
    scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    scope.note_health(timed_out=True, degraded_leg=False)
    scope.note_health(timed_out=False, degraded_leg=False)
    await observe_answer(scope, base)

    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query) is None


async def _test_store_wakes_a_waiter_with_the_response_when_unstorable(base: ResolveResponse):
    """An unstorable response reaches current waiters but is absent from later lookups."""
    cache = _new_cache()
    query = "q-unstorable-wakes-waiter"
    leader = _make_scope(cache)
    assert await leader.lookup_raw(query) is None

    waiter = _make_scope(cache)
    waiter_task = asyncio.ensure_future(waiter.lookup_raw(query))
    await asyncio.sleep(0)  # let the waiter register as a non-leader lease

    leader.note_health(timed_out=True, degraded_leg=False)
    await observe_answer(leader, base)

    result = await waiter_task
    assert result is not None
    assert result.query_echo.raw_query == query
    assert (await _done_of_hit(waiter, result)).timed_out is True

    later = _make_scope(cache)
    assert await later.lookup_raw(query) is None  # nothing was ever stored under the key


async def _test_waiter_becomes_the_new_leader_when_its_leader_abandons():
    cache = _new_cache()
    query = "q-waiter"
    leader = _make_scope(cache)
    assert await leader.lookup_raw(query) is None

    waiter = _make_scope(cache)
    waiter_task = asyncio.ensure_future(waiter.lookup_raw(query))
    await asyncio.sleep(
        0
    )  # let the waiter register as a non-leader lease before the leader releases

    await leader.release()
    result = await waiter_task

    # No stored value and no fresh error either: the waiter re-acquired the
    # key when the leader's release() woke it with nothing, and won. It is
    # the new leader now, not a plain unleased miss.
    assert result is None
    assert waiter._layer1_lease is not None and waiter._layer1_lease.is_leader


async def _test_a_cancelled_leaders_finally_still_releases_the_lease():
    """Cancellation runs the leader's `finally`, releases its lease, and lets the waiter lead."""
    cache = _new_cache()
    query = "q-cancelled"
    leader = _make_scope(cache)

    async def _leader_body() -> None:
        try:
            assert await leader.lookup_raw(query) is None
            await asyncio.sleep(3600)  # stands in for the pipeline run; never reached
        finally:
            await leader.release()

    leader_task = asyncio.ensure_future(_leader_body())
    await asyncio.sleep(0)  # let the leader claim the lease and reach its sleep

    waiter = _make_scope(cache)
    waiter_task = asyncio.ensure_future(waiter.lookup_raw(query))
    await asyncio.sleep(0)  # let the waiter register as a non-leader lease

    leader_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await leader_task

    result = await asyncio.wait_for(waiter_task, timeout=1)

    assert result is None
    assert waiter._layer1_lease is not None and waiter._layer1_lease.is_leader
    await waiter.release()
    assert cache._inflight == {}


async def _test_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time(
    base: ResolveResponse,
):
    """A follower's bounded wait races its OWN deadline, computed once per
    request in `core.py` before the layer-1 lookup, not the leader's. A
    follower whose deadline is much sooner than the leader's actual run time
    stops waiting right on schedule and proceeds as a plain, unleased miss,
    rather than hanging until the leader eventually publishes or the leader
    itself abandons.
    """
    cache = _new_cache()
    query = "q-short-deadline-follower"
    leader = _make_scope(cache, deadline=time.monotonic() + 60)
    assert await leader.lookup_raw(query) is None

    follower = _make_scope(cache, deadline=time.monotonic() + 0.05)
    started = time.monotonic()
    result = await follower.lookup_raw(query)
    elapsed = time.monotonic() - started

    assert result is None
    assert follower._layer1_lease is None  # gave up, and never bid for the key either
    assert elapsed < 1.0  # bounded by the follower's own ~0.05s deadline, not left hanging

    # The leader is still free to finish afterward, unaffected by the
    # follower having given up on it.
    await observe_answer(leader, base)
    assert await _new_scope_hits_layer1(cache, query)


class _RaisingOnPublishCache(ResponseCache):
    async def publish(self, *args, **kwargs):
        raise RuntimeError("boom")


async def _test_store_swallows_a_publish_failure(base: ResolveResponse):
    cache = _RaisingOnPublishCache(max_bytes=10_000_000, max_entry_bytes=10_000_000, ttl_s=60)
    scope = _make_scope(cache)
    assert await scope.lookup_raw("q-raises") is None

    await observe_answer(scope, base)  # must not propagate the RuntimeError


async def _test_release_after_a_miss_frees_the_key_for_a_new_leader():
    cache = _new_cache()
    query = "q-release"
    scope_1 = _make_scope(cache)
    assert await scope_1.lookup_raw(query) is None
    assert scope_1._layer1_lease is not None and scope_1._layer1_lease.is_leader
    await scope_1.release()

    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query) is None
    assert scope_2._layer1_lease is not None and scope_2._layer1_lease.is_leader


async def _test_outcome_is_inactive_when_the_scope_never_activates():
    scope = _make_scope(None)
    assert scope.outcome == "inactive"
    assert await scope.lookup_raw("q") is None
    assert scope.outcome == "inactive"


async def _test_outcome_is_miss_for_a_plain_leader():
    cache = _new_cache()
    scope = _make_scope(cache)
    assert await scope.lookup_raw("q-outcome-miss") is None
    assert scope.outcome == "miss"
    await scope.release()


async def _test_outcome_is_hit_layer1_after_a_layer1_hit(base: ResolveResponse):
    cache = _new_cache()
    query = "q-outcome-layer1"
    writer = _make_scope(cache)
    assert await writer.lookup_raw(query) is None
    await observe_answer(writer, base)

    reader = _make_scope(cache)
    assert await reader.lookup_raw(query) is not None
    assert reader.outcome == "hit layer1"


async def _test_outcome_is_hit_layer2_after_a_layer2_hit(base: ResolveResponse):
    cache = _new_cache()
    extraction = {
        "variables": ["population"],
        "entities": [ExtractedEntity(text="India")],
        "dates": [],
        "dropped_qualifiers": [],
        "variable_qualifiers": [()],
        "operations": [()],
        "entity_roles": (),
        "sources": [()],
        "query_text": None,
    }
    query_1 = "total population india"
    writer = _make_scope(cache)
    assert await writer.lookup_raw(query_1) is None
    assert await writer.lookup_parsed(query=query_1, **extraction) is None
    await observe_answer(writer, base)

    query_2 = "how many people live in india"
    reader = _make_scope(cache)
    assert await reader.lookup_raw(query_2) is None
    hit = await reader.lookup_parsed(query=query_2, **extraction)
    await reader.release()  # the layer-1 lease reader claimed above is never filled

    assert hit is not None
    assert reader.outcome == "hit layer2"


def test_inactive_when_cache_is_none():
    asyncio.run(_test_inactive_when_cache_is_none(_base_response()))


def test_inactive_for_a_dataset_resubmit():
    asyncio.run(_test_inactive_for_a_dataset_resubmit(_base_response()))


def test_a_transport_with_trailing_events_never_reads_a_response_entry():
    asyncio.run(
        _test_a_transport_with_trailing_events_never_reads_a_response_entry(_base_response())
    )


def test_full_miss_then_store_then_layer1_hit_carries_the_new_wording():
    asyncio.run(
        _test_full_miss_then_store_then_layer1_hit_carries_the_new_wording(_base_response())
    )


def test_layer2_hit_for_a_reworded_query_with_the_same_extraction():
    asyncio.run(_test_layer2_hit_for_a_reworded_query_with_the_same_extraction(_base_response()))


def test_layer2_hit_does_not_reset_the_shared_entrys_ttl():
    asyncio.run(_test_layer2_hit_does_not_reset_the_shared_entrys_ttl(_base_response()))


def test_layer2_follower_of_a_timed_out_leader_gets_no_layer1_entry():
    asyncio.run(_test_layer2_follower_of_a_timed_out_leader_gets_no_layer1_entry(_base_response()))


def test_layer2_follower_of_a_degraded_leader_gets_no_layer1_entry():
    asyncio.run(_test_layer2_follower_of_a_degraded_leader_gets_no_layer1_entry(_base_response()))


def test_layer2_follower_of_a_healthy_leader_gets_its_own_layer1_entry():
    asyncio.run(
        _test_layer2_follower_of_a_healthy_leader_gets_its_own_layer1_entry(_base_response())
    )


def test_layer2_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time():
    asyncio.run(
        _test_layer2_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time(
            _base_response()
        )
    )


@pytest.mark.parametrize("pending", [False, True], ids=["settled", "second-pending"])
@pytest.mark.parametrize("agreed", [True, False], ids=["agreed", "samples-disagreed"])
def test_store_publishes_both_leases_when_this_request_holds_both(agreed, pending):
    asyncio.run(
        _test_store_publishes_both_leases_when_this_request_holds_both(
            _base_response(), agreed=agreed, pending=pending
        )
    )


def test_an_unstreamed_response_passes_before_its_second_sample_and_publishes_at_done():
    """A request without trailing events takes its `response` without waiting on a second
    sample still running, and publishes at `done`, once that sample has settled."""

    base = _base_response()

    async def run() -> tuple[bool, bool, bool]:
        cache = _new_cache()
        query = "total population india"
        scope = _make_scope(cache)
        assert await scope.lookup_raw(query) is None
        gate = asyncio.Event()

        async def second() -> tuple[None, None]:
            await gate.wait()
            return None, None

        scope.note_extraction(Agreement(_held(second), lambda _second: True))
        response, done = _terminal_events(base, start_ms=now_ms(), degraded_leg=False)
        await asyncio.wait_for(scope.observe(response), 5)
        published_at_response = scope._layer1_lease is None
        gate.set()
        await scope.observe(done)
        return (
            published_at_response,
            scope._layer1_lease is None,
            await _new_scope_hits_layer1(cache, query),
        )

    assert asyncio.run(run()) == (False, True, True)


def test_release_cancels_and_awaits_a_second_sample_still_running():
    """A request cancelled after its read served the first sample releases its scope in
    `finally`, which cancels the second sample its holder owns and awaits it."""

    async def sample(sampler: SupportsLLM) -> tuple[str, None]:
        assert isinstance(sampler, LLM)
        if sampler._seed == 2:
            await asyncio.Event().wait()
        return "first", None

    async def run() -> list[bool]:
        scope = _make_scope(_new_cache())
        llm = scope.hold_samples(LLM(seed=1))
        served = asyncio.Event()

        async def request() -> None:
            try:
                await two_samples(
                    sample, llm, stage="extraction", read=lambda a: a, judge=lambda *_: True
                )
                served.set()
                await asyncio.Event().wait()
            finally:
                await scope.release()

        task = asyncio.create_task(request())
        await served.wait()
        seconds = [second.task for second in scope._samples._running]
        assert [second.done() for second in seconds] == [False]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # Read before `asyncio.run` returns, since it cancels whatever task is left.
        return [second.cancelled() for second in seconds]

    assert asyncio.run(run()) == [True]


def _held(call: Callable[[], Coroutine[Any, Any, tuple[Any, None]]]) -> SecondSample:
    """`call()` run as an extraction's second sample, held apart from any request."""
    return SecondSamples().start(call, stage="extraction")


async def _never() -> tuple[None, None]:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


async def _agreeing() -> tuple[str, None]:
    return "second", None


def _stalling_second(fail: bool = False):
    """A sample whose seed-2 call never answers, or fails with `fail`, and whose seed-1 call
    answers at once."""

    async def sample(sampler: SupportsLLM) -> tuple[str, None]:
        assert isinstance(sampler, LLM)
        if sampler._seed == 2:
            if fail:
                raise LLMInfraError("503")
            await asyncio.Event().wait()
        return "first", None

    return sample


def test_a_second_sample_that_never_finishes_is_cut_at_the_settle_bound_and_stores_nothing(
    search_settings, caplog
):
    """A request's second sample that never answers ends `llm_second_sample_settle_s` after
    its first was read, on its own deadline, so `done` waits at most that bound. The reading
    is cut: judged by nobody, neither agreed nor disagreed, logged, and layer 1 stores
    nothing under the query."""
    search_settings(llm_second_sample_settle_s=0.05)
    caplog.set_level(logging.INFO, logger=llm_module.__name__)
    base = _base_response()

    async def run() -> tuple[float, str, list[object], bool | None, bool]:
        cache = _new_cache()
        query = "total population india"
        scope = _make_scope(cache)
        assert await scope.lookup_raw(query) is None
        judged: list[object] = []
        _served, _usage, agreement = await two_samples(
            _stalling_second(),
            scope.hold_samples(LLM(seed=1)),
            stage="extraction",
            read=lambda answer: answer,
            judge=lambda *answers: judged.append(answers) or True,
        )
        scope.note_extraction(agreement)
        response, done = _terminal_events(base, start_ms=now_ms(), degraded_leg=False)
        await scope.observe(response)
        started = time.monotonic()
        await asyncio.wait_for(scope.observe(done), 5)
        elapsed = time.monotonic() - started
        await scope.release()
        return (
            elapsed,
            await agreement.outcome(),
            judged,
            scope.extraction_agreed,
            await _new_scope_hits_layer1(cache, query),
        )

    elapsed, outcome, judged, agreed, layer1 = asyncio.run(run())

    assert elapsed < 1.0
    assert (outcome, judged, agreed, layer1) == ("cut", [], None, False)
    assert [r.levelname for r in caplog.records if r.name == llm_module.__name__] == ["INFO"]


def test_an_unheld_callers_stalled_second_sample_is_cut_at_the_settle_bound(search_settings):
    """A caller outside a request settles its read before `two_samples` returns. A second
    sample that never answers is cut `llm_second_sample_settle_s` after the first was read,
    not at `llm_timeout_s`, and counts one uncounted call."""
    search_settings(llm_second_sample_settle_s=0.05, llm_timeout_s=30)

    async def run() -> tuple[float, str, dict | None]:
        started = time.monotonic()
        _served, usage, agreement = await asyncio.wait_for(
            two_samples(
                _stalling_second(),
                LLM(seed=1),
                stage="extraction",
                read=lambda answer: answer,
                judge=lambda *_answers: True,
            ),
            5,
        )
        return time.monotonic() - started, await agreement.outcome(), usage

    elapsed, outcome, usage = asyncio.run(run())

    assert elapsed < 1.0
    assert (outcome, usage) == ("cut", {UNCOUNTED_CALLS: 1})


def test_a_held_second_sample_is_cut_at_the_settle_bound_before_done(search_settings, caplog):
    """A request's second sample that never answers is cut `llm_second_sample_settle_s`
    after its first was read, by its own deadline, while the request has not reached `done`
    and settles nothing. The cut logs one line naming its stage."""
    search_settings(llm_second_sample_settle_s=0.05)
    caplog.set_level(logging.INFO, logger=llm_module.__name__)

    async def run() -> tuple[bool, str]:
        scope = _make_scope(_new_cache())
        _served, _usage, agreement = await two_samples(
            _stalling_second(),
            scope.hold_samples(LLM(seed=1)),
            stage="extraction",
            read=lambda answer: answer,
            judge=lambda *_answers: True,
        )
        [second] = scope._samples._running
        await asyncio.sleep(0.3)
        ended = second.task.done()
        await scope.release()
        return ended, await agreement.outcome()

    assert asyncio.run(run()) == (True, "cut")
    assert [
        (record.levelname, record.getMessage().split(":")[0])
        for record in caplog.records
        if record.name == llm_module.__name__
    ] == [("INFO", "extraction")]


def test_an_unheld_callers_cut_logs_its_stage(search_settings, caplog):
    """A caller outside a request whose second sample is cut sees the same log line as a
    request does, naming the stage."""
    search_settings(llm_second_sample_settle_s=0.05)
    caplog.set_level(logging.INFO, logger=llm_module.__name__)

    asyncio.run(
        two_samples(
            _stalling_second(),
            LLM(seed=1),
            stage="exclusion_read",
            read=lambda answer: answer,
            judge=lambda *_answers: True,
        )
    )

    assert [
        (record.levelname, record.getMessage().split(":")[0])
        for record in caplog.records
        if record.name == llm_module.__name__
    ] == [("INFO", "exclusion_read")]


def test_a_cut_set_while_the_model_deadline_expires_does_not_raise(search_settings):
    """A first sample read just as its second's own `llm_timeout_s` deadline fires sets the
    cut on a timeout already expiring. The cut is ignored, and the second ends as the model
    deadline ended it: failed."""
    search_settings(llm_timeout_s=0.01, llm_second_sample_settle_s=1.0)

    async def run() -> tuple[bool, str]:
        second = _held(_never)
        await asyncio.sleep(0)
        time.sleep(0.03)
        # The first yield runs the due timeout, which marks its scope expiring and
        # cancels the task; this task resumes before the cancelled one does.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        expiring = second._scope is not None and second._scope.expired()
        second.cut_after(1.0)
        return expiring, await Agreement(second, lambda _answer: True).outcome()

    assert asyncio.run(run()) == (True, "failed")


def test_a_second_sample_raising_a_defect_raises_from_its_outcome():
    """Only a model failure, a refusal, a cut or a cancellation leaves a read unjudged. Any
    other exception from the second sample is a defect, and `outcome()` raises it."""

    async def broken() -> tuple[None, None]:
        raise KeyError("missing")

    async def run() -> str:
        return await Agreement(_held(broken), lambda _answer: True).outcome()

    with pytest.raises(KeyError):
        asyncio.run(run())


def test_an_unstreamed_response_is_encoded_once_for_both_layers(monkeypatch):
    """A request holding both leases publishes layer 2 at `response` and layer 1 at `done`
    from one encoding of its response."""
    encoded: list[ResolveResponse] = []

    def counting(response: ResolveResponse) -> tuple[bytes, int]:
        encoded.append(response)
        return to_entry(response)

    monkeypatch.setattr(response_cache_flow, "to_entry", counting)
    base = _base_response()

    async def run() -> tuple[bool, bool]:
        cache = _new_cache()
        query = "total population india"
        scope = _make_scope(cache)
        assert await scope.lookup_raw(query) is None
        assert await scope.lookup_parsed(query=query, **_population_india_extraction()) is None
        await observe_answer(scope, base)
        return scope._layer1_lease is None, scope._layer2_lease is None

    assert asyncio.run(run()) == (True, True)
    assert len(encoded) == 1


def test_a_bundle_publishes_layer2_at_done_before_its_second_sample_settles():
    """A request with trailing events publishes its bundle to layer 2 at `done` before
    settling its second samples, so a reworded request is served while the second sample
    still runs. Layer 1 publishes once it has settled."""
    base = _base_response()
    options = ResolveOptions(include_observations=True)
    request = ResolveRequest(input=RawTextInput(query="q"), response_options=options)

    async def run() -> tuple[bool, bool, bool]:
        cache = _new_cache()
        extraction = _population_india_extraction()
        query = "total population india"
        scope = _make_scope(cache, request=request, trailing=True)
        assert await scope.lookup_raw(query) is None
        assert await scope.lookup_parsed(query=query, **extraction) is None
        gate = asyncio.Event()

        async def second() -> tuple[None, None]:
            await gate.wait()
            return None, None

        scope.note_extraction(Agreement(_held(second), lambda _answer: True))
        response, done = _terminal_events(base, start_ms=now_ms(), degraded_leg=False)
        await scope.observe(response)
        finishing = asyncio.create_task(scope.observe(done))
        await asyncio.sleep(0.05)
        other = _make_scope(cache, request=request, trailing=True)
        layer2 = await other.lookup_parsed(query="another wording", **extraction) is not None
        await other.release()
        layer1_held = scope._layer1_lease is not None
        gate.set()
        await asyncio.wait_for(finishing, 5)
        return layer2, layer1_held, scope._layer1_lease is None

    assert asyncio.run(run()) == (True, True, True)


class _TimedModels:
    """A Gemini double answering `{"value": 1}` after the next delay listed for the
    request's seed, or never for `None`."""

    def __init__(self, delays: dict[int, list[float | None]]) -> None:
        self.delays = delays

    async def generate_content(self, *, model: str, contents: object, config: Any) -> object:
        delay = self.delays[config.seed].pop(0)
        if delay is None:
            await asyncio.Event().wait()
        await asyncio.sleep(delay)
        return SimpleNamespace(text='{"value": 1}', model_version=model, usage_metadata=None)


class _Value(BaseModel):
    value: int = 0


def _timed_sample(sampler: SupportsLLM) -> Coroutine[Any, Any, tuple[_Value, dict | None]]:
    return llm_module.generate_structured_async(sampler, prompt="q", system="s", schema=_Value)


def test_a_repeat_whose_first_the_memo_serves_gives_its_second_the_live_calls_time(
    monkeypatch, search_settings
):
    """A request whose first sample took 0.2 s live and whose second sample was cut leaves
    only the first in the LLM memo. Its repeat is served that first at once, and its second
    runs no less than the 0.2 s the memo's call took plus the settle bound: a second needing
    0.15 s answers, the samples agree, and layer 1 stores the answer, so the next repeat is a
    layer-1 hit. A second that never answers is cut no earlier than that."""
    search_settings(llm_second_sample_settle_s=0.05)
    models = _TimedModels({1: [0.2, 0.2], 2: [0.4, 0.15, None, None]})
    monkeypatch.setattr(
        llm_module, "_get_client", lambda: SimpleNamespace(aio=SimpleNamespace(models=models))
    )
    base = _base_response()
    query = "total population india"

    async def ask(cache: ResponseCache, memo: llm_module.LLMMemo) -> tuple[str, float]:
        scope = _make_scope(cache)
        assert await scope.lookup_raw(query) is None
        started = time.monotonic()
        _served, _usage, agreement = await two_samples(
            _timed_sample,
            scope.hold_samples(LLM(seed=1, memo=memo)),
            stage="extraction",
            read=lambda answer: answer,
            judge=lambda *_answers: True,
        )
        scope.note_extraction(agreement)
        await observe_answer(scope, base)
        outcome = await agreement.outcome()
        elapsed = time.monotonic() - started
        await scope.release()
        return outcome, elapsed

    async def run() -> tuple[str, str, bool, str, float]:
        cache, memo = _new_cache(), llm_module.build_memo()
        assert memo is not None
        cut, _ = await ask(cache, memo)
        agreed, _ = await ask(cache, memo)
        repeat_hits = await _new_scope_hits_layer1(cache, query)
        cache, memo = _new_cache(), llm_module.build_memo()
        assert memo is not None
        await ask(cache, memo)
        stalled, waited = await ask(cache, memo)
        return cut, agreed, repeat_hits, stalled, waited

    cut, agreed, repeat_hits, stalled, waited = asyncio.run(run())

    assert (cut, agreed, repeat_hits, stalled) == ("cut", "agreed", True, "cut")
    assert waited >= 0.2 + 0.05


def test_a_failed_second_sample_leaves_the_reading_unjudged():
    """A second sample that fails is unjudged, not a disagreement: the request's reading is
    neither agreed nor disagreed, and layer 1 stores nothing under the query."""
    base = _base_response()

    async def run() -> tuple[str, bool | None, bool]:
        cache = _new_cache()
        query = "total population india"
        scope = _make_scope(cache)
        assert await scope.lookup_raw(query) is None
        _served, _usage, agreement = await two_samples(
            _stalling_second(fail=True),
            scope.hold_samples(LLM(seed=1)),
            stage="extraction",
            read=lambda answer: answer,
            judge=lambda *_answers: True,
        )
        scope.note_extraction(agreement)
        await observe_answer(scope, base)
        await scope.release()
        return (
            await agreement.outcome(),
            scope.extraction_agreed,
            await _new_scope_hits_layer1(cache, query),
        )

    assert asyncio.run(run()) == ("failed", None, False)


def test_an_unstreamed_response_publishes_layer2_at_response_and_layer1_at_done():
    """A request without trailing events publishes layer 2, keyed on the reading, when its
    `response` passes, so a reworded request is served before this one's second sample has
    answered. Layer 1 publishes at `done`, once the samples agreed."""
    base = _base_response()

    async def run() -> tuple[bool, bool, bool, bool]:
        cache = _new_cache()
        extraction = _population_india_extraction()
        query = "total population india"
        scope = _make_scope(cache)
        assert await scope.lookup_raw(query) is None
        assert await scope.lookup_parsed(query=query, **extraction) is None
        gate = asyncio.Event()

        async def second() -> tuple[None, None]:
            await gate.wait()
            return None, None

        scope.note_extraction(Agreement(_held(second), lambda _answer: True))
        response, done = _terminal_events(base, start_ms=now_ms(), degraded_leg=False)
        await scope.observe(response)
        other = _make_scope(cache)
        layer2 = await other.lookup_parsed(query="another wording", **extraction) is not None
        await other.release()
        layer1_held = scope._layer1_lease is not None
        gate.set()
        await scope.observe(done)
        return (
            layer2,
            layer1_held,
            scope._layer1_lease is None,
            await _new_scope_hits_layer1(cache, query),
        )

    assert asyncio.run(run()) == (True, True, True, True)


def test_a_request_without_cache_or_extraction_store_settles_nothing_at_done():
    """With no response cache and no extraction store, `done` does not wait for a second
    sample still running: nothing would keep its verdict."""
    base = _base_response()

    async def run() -> tuple[bool, bool | None]:
        scope = _make_scope(None)
        scope.hold_samples(LLM(seed=1))
        second = _held(_never)
        scope.note_extraction(Agreement(second, lambda _answer: True))
        response, done = _terminal_events(base, start_ms=now_ms(), degraded_leg=False)
        await scope.observe(response)
        # Well under the default 1 s settle bound.
        await asyncio.wait_for(scope.observe(done), 0.5)
        running = not second.task.done()
        await scope.release()
        return running, scope.extraction_agreed

    assert asyncio.run(run()) == (True, None)


def test_a_finished_agreeing_sample_is_stored_past_the_soft_deadline():
    """A request that reaches `done` after its soft deadline settles a second sample that
    already answered: the agreed reading is judged, which stores it, and layer 1 stores the
    answer under the query."""
    base = _base_response()

    async def run() -> tuple[list[object], bool | None, bool]:
        cache = _new_cache()
        query = "total population india"
        scope = _make_scope(cache, deadline=time.monotonic() - 1)
        assert await scope.lookup_raw(query) is None
        judged: list[object] = []
        second = _held(_agreeing)
        await asyncio.wait([second.task])
        scope.note_extraction(Agreement(second, lambda answer: judged.append(answer) or True))
        await observe_answer(scope, base)
        return judged, scope.extraction_agreed, await _new_scope_hits_layer1(cache, query)

    assert asyncio.run(run()) == (["second"], True, True)


def test_an_extraction_store_write_that_fails_does_not_fail_the_request(caplog):
    """A judge whose extraction-store write raises `OSError` is logged, and leaves the read
    unjudged. The request publishes its answer: layer 2 keeps it, a waiter receives it, and
    layer 1, whose reading is not agreed, stores nothing under the query."""
    caplog.set_level(logging.INFO, logger=llm_module.__name__)
    base = _base_response()

    def judge(_answer: object) -> bool:
        raise OSError("disk full")

    async def run() -> tuple[bool | None, bool, bool, bool]:
        cache = _new_cache()
        extraction = _population_india_extraction()
        query = "total population india"
        scope = _make_scope(cache)
        assert await scope.lookup_raw(query) is None
        assert await scope.lookup_parsed(query=query, **extraction) is None
        scope.note_extraction(Agreement(_held(_agreeing), judge))
        waiter = _make_scope(cache)
        waiting = asyncio.ensure_future(waiter.lookup_raw(query))
        await asyncio.sleep(0)
        await observe_answer(scope, base)
        other = _make_scope(cache)
        layer2 = await other.lookup_parsed(query="another wording", **extraction) is not None
        await other.release()
        return (
            scope.extraction_agreed,
            await waiting is not None,
            layer2,
            await _new_scope_hits_layer1(cache, query),
        )

    assert asyncio.run(run()) == (None, True, True, False)
    assert [
        record.levelname for record in caplog.records if record.name == llm_module.__name__
    ] == ["ERROR"]


def test_release_abandons_both_leases_when_this_request_holds_both():
    asyncio.run(_test_release_abandons_both_leases_when_this_request_holds_both())


def test_store_does_not_persist_when_timed_out_was_noted():
    asyncio.run(_test_store_does_not_persist_when_timed_out_was_noted(_base_response()))


def test_store_does_not_persist_when_a_leg_degraded():
    asyncio.run(_test_store_does_not_persist_when_a_leg_degraded(_base_response()))


def test_note_health_true_sticks_even_after_a_later_false_report():
    asyncio.run(_test_note_health_true_sticks_even_after_a_later_false_report(_base_response()))


def test_store_wakes_a_waiter_with_the_response_when_unstorable():
    asyncio.run(_test_store_wakes_a_waiter_with_the_response_when_unstorable(_base_response()))


def test_waiter_becomes_the_new_leader_when_its_leader_abandons():
    asyncio.run(_test_waiter_becomes_the_new_leader_when_its_leader_abandons())


def test_a_cancelled_leaders_finally_still_releases_the_lease():
    asyncio.run(_test_a_cancelled_leaders_finally_still_releases_the_lease())


def test_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time():
    asyncio.run(
        _test_follower_deadline_shorter_than_leaders_run_proceeds_unleased_on_time(_base_response())
    )


def test_store_swallows_a_publish_failure():
    asyncio.run(_test_store_swallows_a_publish_failure(_base_response()))


def test_release_after_a_miss_frees_the_key_for_a_new_leader():
    asyncio.run(_test_release_after_a_miss_frees_the_key_for_a_new_leader())


def test_outcome_is_inactive_when_the_scope_never_activates():
    asyncio.run(_test_outcome_is_inactive_when_the_scope_never_activates())


def test_outcome_is_miss_for_a_plain_leader():
    asyncio.run(_test_outcome_is_miss_for_a_plain_leader())


def test_outcome_is_hit_layer1_after_a_layer1_hit():
    asyncio.run(_test_outcome_is_hit_layer1_after_a_layer1_hit(_base_response()))


def test_outcome_is_hit_layer2_after_a_layer2_hit():
    asyncio.run(_test_outcome_is_hit_layer2_after_a_layer2_hit(_base_response()))


async def _test_a_new_stamp_does_not_join_a_lease_opened_under_the_old_stamp():
    """The single-flight registry is keyed by the full cache key
    (response_cache_keys.scope_document folds `stamp.key` into the shared
    scope), so a caller building its scope under a newer stamp never joins a
    lease a caller under an OLDER stamp already opened for the identical
    query. It becomes a leader on its own key instead, the same way a
    differently-worded query would. `scope_1`'s own leader lease is left
    unresolved (no observe_answer/release()) for the whole test, so a scope_2 that
    joined it would report `is_leader=False` here. It does not.
    """
    cache = _new_cache()
    query = "gdp kenya"

    scope_1 = _make_scope(cache, generation=FakeGeneration(token="gen-a"))
    assert await scope_1.lookup_raw(query) is None
    assert scope_1._layer1_lease is not None
    assert scope_1._layer1_lease.is_leader is True

    scope_2 = _make_scope(cache, generation=FakeGeneration(token="gen-b"))
    assert await scope_2.lookup_raw(query) is None
    assert scope_2._layer1_lease is not None
    assert scope_2._layer1_lease.is_leader is True  # its own lease, never a follower

    assert scope_1._layer1_lease.key != scope_2._layer1_lease.key


async def _test_unknown_and_known_stamps_never_share_a_lease_either():
    """The companion case. `UNKNOWN_GENERATION` (a never-configured
    provider) and a `FakeGeneration` reporting a real token must not share a
    lease for the identical query, the same way two distinct known tokens
    do not above."""
    cache = _new_cache()
    query = "gdp kenya"

    unknown_scope = _make_scope(cache)  # generation defaults to UNKNOWN_GENERATION
    assert await unknown_scope.lookup_raw(query) is None
    assert unknown_scope._layer1_lease is not None
    assert unknown_scope._layer1_lease.is_leader is True

    known_scope = _make_scope(cache, generation=FakeGeneration(token="gen-a"))
    assert await known_scope.lookup_raw(query) is None
    assert known_scope._layer1_lease is not None
    assert known_scope._layer1_lease.is_leader is True

    assert unknown_scope._layer1_lease.key != known_scope._layer1_lease.key


async def _test_entry_lifetime_equals_the_stamp_max_age_captured_at_construction(
    base: ResolveResponse,
):
    """`observe_answer` writes with `max_age_s=self._stamp.max_age_s`, the stamp
    `__init__` captured once, before the generation could move on, not
    whatever `generation.stamp()` would report by the time `observe_answer` runs.
    """
    cache = _new_cache()
    generation = FakeGeneration(token="gen-a")
    scope = _make_scope(cache, generation=generation)
    query = "gdp kenya"
    assert await scope.lookup_raw(query) is None
    lease_key = scope._layer1_lease.key

    # The real generation moves on mid-request (an ingestion lands while this
    # request's own fan-out is still running). This scope's own stamp does
    # not move with it, since __init__ already captured it.
    generation.advance("gen-b")

    before = time.monotonic()
    await observe_answer(scope, base)
    expected_max_age = FakeGeneration(token="gen-a").stamp().max_age_s

    entry = cache._store[lease_key]
    assert abs(entry.expires_at - (before + expected_max_age)) < 5.0


def test_a_new_stamp_does_not_join_a_lease_opened_under_the_old_stamp():
    asyncio.run(_test_a_new_stamp_does_not_join_a_lease_opened_under_the_old_stamp())


def test_unknown_and_known_stamps_never_share_a_lease_either():
    asyncio.run(_test_unknown_and_known_stamps_never_share_a_lease_either())


def test_entry_lifetime_equals_the_stamp_max_age_captured_at_construction():
    asyncio.run(
        _test_entry_lifetime_equals_the_stamp_max_age_captured_at_construction(_base_response())
    )


def _conversion_catalogue(default_base_year: int) -> ConversionCatalogue:
    return ConversionCatalogue(
        recipes={}, tables={}, national_currencies={}, default_base_year=default_base_year
    )


async def _test_an_answer_is_served_only_under_the_catalogue_it_was_computed_from(
    base: ResolveResponse, live_base_year: int
) -> ResolveResponse | None:
    cache = _new_cache()
    generation = FakeGeneration(token="gen-a")
    query = "gdp kenya"
    with bound_conversion_catalogue(_conversion_catalogue(2024)):
        scope = _make_scope(cache, generation=generation)
    assert await scope.lookup_raw(query) is None
    await observe_answer(scope, base)

    with bound_conversion_catalogue(_conversion_catalogue(live_base_year)):
        return await _make_scope(cache, generation=generation).lookup_raw(query)


@pytest.mark.parametrize(
    ("live_base_year", "served"),
    [
        pytest.param(2024, True, id="same-catalogue-facts"),
        pytest.param(2025, False, id="reloaded-catalogue"),
    ],
)
def test_an_answer_is_served_only_under_the_catalogue_it_was_computed_from(
    live_base_year: int, served: bool
):
    """A catalogue reload keeps the graph generation, so the revision of the catalogue a
    request bound keys its answer: one computed with default base year 2024 is not served
    once a catalogue declaring 2025 is live, and a reload holding the same facts still
    serves it."""
    restored = asyncio.run(
        _test_an_answer_is_served_only_under_the_catalogue_it_was_computed_from(
            _base_response(), live_base_year
        )
    )

    assert (restored is not None) is served


async def _test_an_answer_is_served_only_under_the_origin_catalogue_it_was_admitted_by(
    base: ResolveResponse, live: FakeCatalogue
) -> ResolveResponse | None:
    cache = _new_cache()
    generation = FakeGeneration(token="gen-a")
    query = "gdp kenya"
    with bound_catalogue(FakeCatalogue()):
        scope = _make_scope(cache, generation=generation)
    assert await scope.lookup_raw(query) is None
    await observe_answer(scope, base)

    with bound_catalogue(live):
        return await _make_scope(cache, generation=generation).lookup_raw(query)


@pytest.mark.parametrize(
    ("live", "served"),
    [
        pytest.param(FakeCatalogue(), True, id="same-catalogue"),
        pytest.param(FakeCatalogue(build_id="build-b"), False, id="new-build"),
        pytest.param(FakeCatalogue(un_system_enabled=False), False, id="un-system-off"),
    ],
)
def test_an_answer_is_served_only_under_the_origin_catalogue_it_was_admitted_by(
    live: FakeCatalogue, served: bool
):
    """The origin catalogue reloads by its build id without a new graph generation, and its
    build and UN System setting decide what an answer admits, so both key the answer."""
    restored = asyncio.run(
        _test_an_answer_is_served_only_under_the_origin_catalogue_it_was_admitted_by(
            _base_response(), live
        )
    )

    assert (restored is not None) is served


async def _test_a_un_refusal_is_stored_only_under_a_build_that_refuses_the_place(
    base: ResolveResponse, build_refuses: frozenset[str]
) -> ResolveResponse | None:
    cache = _new_cache()
    query = "inflation ghana"
    refusal = Warning(
        code="un_system_unavailable",
        severity="warn",
        message="m",
        params={"places": ["country/ESH"], "variables": [], "variable_labels": []},
    )
    diagnostics = base.diagnostics.model_copy(
        update={"warnings": [*base.diagnostics.warnings, refusal]}
    )
    refused = base.model_copy(update={"diagnostics": diagnostics})
    catalogue = FakeCatalogue(un_blocked_places=build_refuses)
    with bound_catalogue(catalogue):
        scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    await observe_answer(scope, refused)

    with bound_catalogue(catalogue):
        return await _make_scope(cache).lookup_raw(query)


@pytest.mark.parametrize(
    ("build_refuses", "stored"),
    [
        pytest.param(frozenset({"country/ESH"}), True, id="refused-by-the-build"),
        pytest.param(frozenset(), False, id="refused-while-serving"),
    ],
)
def test_a_un_refusal_is_stored_only_under_a_build_that_refuses_the_place(
    build_refuses: frozenset[str], stored: bool
):
    """A place the origin catalogue build found refused is a fact of that build, which the
    key names. A place refused only while serving may be a stray refusal, so the answer
    naming it is not stored."""
    restored = asyncio.run(
        _test_a_un_refusal_is_stored_only_under_a_build_that_refuses_the_place(
            _base_response(), build_refuses
        )
    )
    assert (restored is not None) is stored


def test_a_binder_reading_the_query_text_looks_up_nothing_on_layer_two():
    """A reworded query with the same extraction misses under a binder whose calls read the
    query's own text, since its answer may differ. The stored query's repeat still hits
    layer 1."""
    jev = KeyedBinder(name="jev", model="jev-1.13.0", reads_query_text=True)

    base = _base_response()

    async def run() -> tuple[ResolveResponse | None, bool, bool]:
        cache = _new_cache()
        writer = _make_scope(cache, binder=jev)
        assert await writer.lookup_raw("total population india") is None
        assert await writer.lookup_parsed(query="total population india", **_EXTRACTION) is None
        await observe_answer(writer, base)

        reader = _make_scope(cache, binder=jev)
        assert await reader.lookup_raw("how many people live in india") is None
        hit = await reader.lookup_parsed(query="how many people live in india", **_EXTRACTION)
        await reader.release()
        repeat = _make_scope(cache, binder=jev)
        return (
            hit,
            reader._layer2_lease is None,
            await repeat.lookup_raw("total population india") is not None,
        )

    hit, no_layer2_lease, repeat_hits = asyncio.run(run())

    assert hit is None
    assert no_layer2_lease
    assert repeat_hits


def test_a_request_whose_binder_fell_back_stores_nothing():
    """An answer a fallback served is not the one its binder's key names, so neither
    layer keeps it."""

    base = _base_response()

    async def run() -> tuple[bool, ResolveResponse | None]:
        cache = _new_cache()
        scope = _make_scope(cache, fallbacks=Fallbacks(reasons=["timeout"]))
        assert await scope.lookup_raw("total population india") is None
        assert await scope.lookup_parsed(query="total population india", **_EXTRACTION) is None
        await observe_answer(scope, base)

        later = _make_scope(cache)
        layer1 = await later.lookup_raw("total population india") is not None
        layer2 = await later.lookup_parsed(query="how many people live in india", **_EXTRACTION)
        await later.release()
        return layer1, layer2

    layer1, layer2 = asyncio.run(run())

    assert not layer1
    assert layer2 is None


_EXTRACTION = {
    "variables": ["population"],
    "entities": [ExtractedEntity(text="India")],
    "dates": [],
    "dropped_qualifiers": [],
    "variable_qualifiers": [()],
    "operations": [()],
    "entity_roles": (),
    "sources": [()],
    "query_text": None,
}


def test_the_default_route_keys_on_the_gemini_binder_and_the_engine_model():
    """Under the default settings the key names the Gemini binder, and both its model and
    the engine model are `engine_model`."""
    scope = _make_scope(_new_cache(), binder=configured_binder("total population india"))
    engine_model = config.settings().engine_model

    assert scope._scope is not None
    assert {k: scope._scope[k] for k in ("binder", "model", "engine_model")} == {
        "binder": "gemini",
        "model": engine_model,
        "engine_model": engine_model,
    }
