"""Pins `CacheScope` key lookup, publication, lease handling, and health tracking."""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from graph_generation import UNKNOWN_GENERATION, FakeGeneration, GenerationProvider
from origin_catalogue import bound_catalogue
from search.engine import config
from search.engine.binders import Binder, Fallbacks, configured_binder
from search.engine.envelope import now_ms
from search.engine.extract import ExtractedEntity
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_flow import CacheScope
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
from series_coverage.shared import CoverageSnapshot
from tests.engine._binders import KeyedBinder
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import make_request, member_walk_reader, offline_resolve
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
) -> CacheScope:
    return CacheScope(
        cache,
        request=request or ResolveRequest(input=RawTextInput(query="placeholder")),
        start_ms=now_ms(),
        deadline=deadline if deadline is not None else time.monotonic() + _DEADLINE_HORIZON_S,
        build_commit=_BUILD_COMMIT,
        binder=binder if binder is not None else KeyedBinder(model=_MODEL),
        fallbacks=fallbacks if fallbacks is not None else Fallbacks(),
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
            entity_roles={},
            sources=[],
            query_text=None,
        )
        is None
    )
    scope.note_health(timed_out=True, degraded_leg=True)
    await scope.store(base)
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
    await stored.store(base)
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
    await scope_1.store(base)

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
        "entity_roles": {},
        "sources": [()],
        "query_text": None,
    }

    query_1 = "total population india"
    scope_1 = _make_scope(cache)
    assert await scope_1.lookup_raw(query_1) is None
    assert await scope_1.lookup_parsed(query=query_1, **extraction) is None
    await scope_1.store(base)

    query_2 = "how many people live in india"
    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query_2) is None  # a different query string misses layer 1
    hit = await scope_2.lookup_parsed(query=query_2, **extraction)
    await scope_2.release()  # the layer-1 lease scope_2 claimed above is never filled

    assert hit is not None
    assert hit.query_echo.raw_query == query_2
    assert hit.query_echo.extract_skipped is False


async def _test_layer2_hit_does_not_reset_the_shared_entrys_ttl(base: ResolveResponse):
    """A layer-2 hit does not refresh its entry's TTL, even when `store()` runs."""
    cache = _new_cache()
    extraction = {
        "variables": ["population"],
        "entities": [ExtractedEntity(text="India")],
        "dates": [],
        "dropped_qualifiers": [],
        "variable_qualifiers": [()],
        "operations": [()],
        "entity_roles": {},
        "sources": [()],
        "query_text": None,
    }

    query_1 = "total population india"
    scope_1 = _make_scope(cache)
    assert await scope_1.lookup_raw(query_1) is None
    assert await scope_1.lookup_parsed(query=query_1, **extraction) is None
    await scope_1.store(base)

    layer2_key = scope_1._layer2_key
    assert layer2_key is not None
    expires_at_after_first_store = cache._store[layer2_key].expires_at

    query_2 = "how many people live in india"
    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query_2) is None
    hit = await scope_2.lookup_parsed(query=query_2, **extraction)
    assert hit is not None
    await scope_2.store(hit)  # the hit path calling store(), as core.py does

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
        "entity_roles": {},
        "sources": [()],
        "query_text": None,
    }


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


async def _test_layer2_follower_of_a_timed_out_leader_gets_no_layer1_entry(base: ResolveResponse):
    cache = _new_cache()
    extraction = _population_india_extraction()
    query_1, query_2 = "total population india", "how many people live in india"

    leader, follower, follower_task = await _woken_layer2_follower(
        cache, extraction, query_1=query_1, query_2=query_2
    )
    leader.note_health(timed_out=True, degraded_leg=False)
    await leader.store(base)

    result = await follower_task
    assert result is not None
    assert follower.served_timed_out is True
    assert follower.outcome == "hit layer2"
    await follower.store(result)  # the ResponseEvent branch core.py drives on every response

    # Neither wording's own layer-1 entry persists: the leader's result was
    # unstorable, and store()'s served-verdict check inherits that rather
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
    await leader.store(base)

    result = await follower_task
    assert result is not None
    assert follower.outcome == "hit layer2"
    await follower.store(result)

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
    await leader.store(base)  # healthy: no note_health call at all

    result = await follower_task
    assert result is not None
    assert follower.outcome == "hit layer2"
    await follower.store(result)

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
    await leader.store(base)
    assert await _new_scope_hits_layer1(cache, query_1)


async def _test_store_publishes_both_leases_when_this_request_holds_both(base: ResolveResponse):
    cache = _new_cache()
    extraction = _population_india_extraction()
    query = "total population india"
    scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    assert await scope.lookup_parsed(query=query, **extraction) is None
    assert scope._layer1_lease is not None
    assert scope._layer2_lease is not None

    await scope.store(base)

    assert scope._layer1_lease is None
    assert scope._layer2_lease is None
    assert await _new_scope_hits_layer1(cache, query)

    other = _make_scope(cache)
    assert await other.lookup_raw("a wording that is not the one stored above") is None
    hit = await other.lookup_parsed(
        query="a wording that is not the one stored above", **extraction
    )
    await other.release()
    assert hit is not None  # the layer-2 write from the same store() call also landed


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
    await scope.store(base)

    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query) is None


async def _test_store_does_not_persist_when_a_leg_degraded(base: ResolveResponse):
    cache = _new_cache()
    query = "q-degraded"
    scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    scope.note_health(timed_out=False, degraded_leg=True)
    await scope.store(base)

    scope_2 = _make_scope(cache)
    assert await scope_2.lookup_raw(query) is None


async def _test_note_health_true_sticks_even_after_a_later_false_report(base: ResolveResponse):
    cache = _new_cache()
    query = "q-sticky"
    scope = _make_scope(cache)
    assert await scope.lookup_raw(query) is None
    scope.note_health(timed_out=True, degraded_leg=False)
    scope.note_health(timed_out=False, degraded_leg=False)
    await scope.store(base)

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
    await leader.store(base)

    result = await waiter_task
    assert result is not None
    assert result.query_echo.raw_query == query
    assert waiter.served_timed_out is True

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
    await leader.store(base)
    assert await _new_scope_hits_layer1(cache, query)


class _RaisingOnPublishCache(ResponseCache):
    async def publish(self, *args, **kwargs):
        raise RuntimeError("boom")


async def _test_store_swallows_a_publish_failure(base: ResolveResponse):
    cache = _RaisingOnPublishCache(max_bytes=10_000_000, max_entry_bytes=10_000_000, ttl_s=60)
    scope = _make_scope(cache)
    assert await scope.lookup_raw("q-raises") is None

    await scope.store(base)  # must not propagate the RuntimeError


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
    await writer.store(base)

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
        "entity_roles": {},
        "sources": [()],
        "query_text": None,
    }
    query_1 = "total population india"
    writer = _make_scope(cache)
    assert await writer.lookup_raw(query_1) is None
    assert await writer.lookup_parsed(query=query_1, **extraction) is None
    await writer.store(base)

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


def test_store_publishes_both_leases_when_this_request_holds_both():
    asyncio.run(_test_store_publishes_both_leases_when_this_request_holds_both(_base_response()))


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
    unresolved (no store()/release()) for the whole test, so a scope_2 that
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
    """`store()` writes with `max_age_s=self._stamp.max_age_s`, the stamp
    `__init__` captured once, before the generation could move on, not
    whatever `generation.stamp()` would report by the time `store()` runs.
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
    await scope.store(base)
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
    await scope.store(base)

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
    await scope.store(base)

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
    await scope.store(refused)

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
        await writer.store(base)

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
        await scope.store(base)

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
    "entity_roles": {},
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
