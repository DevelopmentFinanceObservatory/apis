"""Tests bounded, generation-keyed sync and async behavior in `SingleFlightCache`.

Concurrency cases use `threading.Event` and asyncio tasks.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from dfo_platform.single_flight_cache import AlreadyInFlight, SingleFlightCache
from graph_generation import FakeGeneration, GenerationStamp

# ---------------------------------------------------------------------------
# Sync entry point: get_or_fetch (single key)
# ---------------------------------------------------------------------------


def test_sync_concurrent_misses_for_the_same_key_fetch_once():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def fetch() -> str:
        calls.append("call")
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return "value"

    results: list[str] = []
    results_lock = threading.Lock()

    def worker() -> None:
        result = cache.get_or_fetch("k", fetch)
        with results_lock:
            results.append(result)

    leader = threading.Thread(target=worker)
    leader.start()
    assert started.wait(timeout=5), "leader never reached fetch"

    follower = threading.Thread(target=worker)
    follower.start()
    time.sleep(0.05)  # give the follower a chance to park on the leader's slot

    release.set()
    leader.join(timeout=5)
    follower.join(timeout=5)

    assert len(calls) == 1, "two concurrent callers for the same key must share one fetch"
    assert results == ["value", "value"]


def test_sync_failing_leader_wakes_followers_with_the_error_and_caches_nothing():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def failing_fetch() -> str:
        calls.append("call")
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        raise RuntimeError("leader's own failure")

    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker() -> None:
        try:
            cache.get_or_fetch("k", failing_fetch)
        except BaseException as exc:
            with errors_lock:
                errors.append(exc)

    leader = threading.Thread(target=worker)
    leader.start()
    assert started.wait(timeout=5), "leader never reached fetch"

    follower = threading.Thread(target=worker)
    follower.start()
    time.sleep(0.05)

    release.set()
    leader.join(timeout=5)
    follower.join(timeout=5)

    assert len(errors) == 2
    assert all(isinstance(exc, RuntimeError) for exc in errors)

    # A failed fetch leaves the cache empty.
    calls_before = len(calls)
    result = cache.get_or_fetch("k", lambda: "recovered")
    assert result == "recovered"
    assert len(calls) == calls_before, "the failed leader's own fetch must not have been reused"


def test_sync_follower_deadline_expires_while_the_leader_keeps_running():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)
    started = threading.Event()
    release = threading.Event()

    def fetch() -> str:
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return "value"

    leader_result: list[str] = []

    def leader() -> None:
        leader_result.append(cache.get_or_fetch("k", fetch))

    t = threading.Thread(target=leader)
    t.start()
    assert started.wait(timeout=5), "leader never reached fetch"

    before = time.monotonic()
    with pytest.raises(TimeoutError):
        cache.get_or_fetch("k", fetch, deadline=time.monotonic() - 1.0)
    assert time.monotonic() - before < 2, "the follower must not have waited for the leader"

    release.set()
    t.join(timeout=5)
    assert leader_result == ["value"]


def test_sync_follower_deadline_is_measured_on_the_real_clock_not_the_injected_ttl_clock():
    """Follower deadlines use `time.monotonic()`, independent of the injected TTL clock."""
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, clock=lambda: time.monotonic() + 10_000
    )
    started = threading.Event()
    release = threading.Event()

    def fetch() -> str:
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return "value"

    leader_result: list[str] = []

    def leader() -> None:
        leader_result.append(cache.get_or_fetch("k", fetch))

    t = threading.Thread(target=leader)
    t.start()
    assert started.wait(timeout=5), "leader never reached fetch"

    follower_result: list[str] = []
    follower_error: list[BaseException] = []

    def follower() -> None:
        try:
            follower_result.append(cache.get_or_fetch("k", fetch, deadline=time.monotonic() + 5))
        except BaseException as exc:
            follower_error.append(exc)

    f = threading.Thread(target=follower)
    f.start()
    time.sleep(0.05)  # give the follower a chance to park on the leader's slot

    release.set()
    t.join(timeout=5)
    f.join(timeout=5)

    assert follower_error == [], f"follower must not time out: {follower_error}"
    assert follower_result == ["value"]


def test_a_cached_none_is_served_as_a_hit_not_re_fetched():
    """A cached `None` is a hit and must not trigger another fetch."""
    cache: SingleFlightCache[str, None] = SingleFlightCache(maxsize=10)
    calls: list[str] = []

    def fetch() -> None:
        calls.append("call")
        return None

    first = cache.get_or_fetch("k", fetch)
    second = cache.get_or_fetch("k", fetch)

    assert first is None
    assert second is None
    assert len(calls) == 1, "a cached None must be served as a hit, not re-fetched"


def test_sync_concurrent_followers_get_a_leaders_none_result():
    """A parked follower receives the leader's cached `None` result."""
    cache: SingleFlightCache[str, None] = SingleFlightCache(maxsize=10)
    started = threading.Event()
    release = threading.Event()

    def fetch() -> None:
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return None

    results: list[None] = []
    results_lock = threading.Lock()

    def worker() -> None:
        result = cache.get_or_fetch("k", fetch)
        with results_lock:
            results.append(result)

    leader = threading.Thread(target=worker)
    leader.start()
    assert started.wait(timeout=5), "leader never reached fetch"

    follower = threading.Thread(target=worker)
    follower.start()
    time.sleep(0.05)  # give the follower a chance to park on the leader's slot

    release.set()
    leader.join(timeout=5)
    follower.join(timeout=5)

    assert results == [None, None]


def test_sync_owner_pops_its_slot_only_if_still_owned_after_a_clear():
    """An old owner must leave a new in-flight slot intact after `clear()` races its fetch."""
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)
    old_owner_started = threading.Event()
    old_owner_release = threading.Event()

    def old_fetch() -> str:
        old_owner_started.set()
        assert old_owner_release.wait(timeout=5), "test deadlocked waiting for release"
        raise RuntimeError("old owner's own failure")

    old_owner_error: list[BaseException] = []

    def old_owner() -> None:
        try:
            cache.get_or_fetch("k", old_fetch)
        except BaseException as exc:
            old_owner_error.append(exc)

    old_thread = threading.Thread(target=old_owner)
    old_thread.start()
    assert old_owner_started.wait(timeout=5), "old owner never reached fetch"

    cache.clear()  # empties _sync_inflight while the old owner's fetch is still running

    new_owner_started = threading.Event()
    new_owner_release = threading.Event()
    new_owner_calls: list[str] = []

    def new_fetch() -> str:
        new_owner_calls.append("call")
        new_owner_started.set()
        assert new_owner_release.wait(timeout=5), "test deadlocked waiting for release"
        return "new-value"

    new_owner_result: list[str] = []

    def new_owner() -> None:
        new_owner_result.append(cache.get_or_fetch("k", new_fetch))

    new_thread = threading.Thread(target=new_owner)
    new_thread.start()
    assert new_owner_started.wait(timeout=5), "new owner never reached fetch"

    old_owner_release.set()
    old_thread.join(timeout=5)
    assert not old_thread.is_alive(), "old owner hung after being released"
    assert len(old_owner_error) == 1

    # A third caller joins the new owner's slot; one fetch proves old-owner cleanup preserved it.
    third_caller_result: list[str] = []

    def third_caller() -> None:
        third_caller_result.append(cache.get_or_fetch("k", new_fetch))

    third_thread = threading.Thread(target=third_caller)
    third_thread.start()
    time.sleep(0.05)  # give the third caller a chance to park, if it's going to

    new_owner_release.set()
    new_thread.join(timeout=5)
    third_thread.join(timeout=5)

    assert new_owner_result == ["new-value"]
    assert third_caller_result == ["new-value"]
    assert len(new_owner_calls) == 1, (
        f"the third caller must have joined the new owner's fetch, not issued its own; "
        f"got {len(new_owner_calls)} call(s)"
    )


def test_sync_join_inflight_false_raises_at_once_without_touching_the_leader():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)
    started = threading.Event()
    release = threading.Event()

    def fetch() -> str:
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return "value"

    t = threading.Thread(target=lambda: cache.get_or_fetch("k", fetch))
    t.start()
    assert started.wait(timeout=5), "leader never reached fetch"

    with pytest.raises(AlreadyInFlight):
        cache.get_or_fetch("k", fetch, join_inflight=False)

    release.set()
    t.join(timeout=5)


# ---------------------------------------------------------------------------
# Sync entry point: get_or_fetch_many (batch)
# ---------------------------------------------------------------------------


def _echo(keys: list[str]) -> dict[str, str]:
    return {k: f"value-{k}" for k in keys}


def test_get_or_fetch_many_fetches_only_the_misses():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)
    calls: list[list[str]] = []

    def fetch(keys: list[str]) -> dict[str, str]:
        calls.append(list(keys))
        return _echo(keys)

    first = cache.get_or_fetch_many(["a", "b"], fetch)
    assert calls == [["a", "b"]]
    assert first == {"a": "value-a", "b": "value-b"}

    second = cache.get_or_fetch_many(["a", "c"], fetch)
    assert calls == [["a", "b"], ["c"]], "only the genuine miss reaches fetch"
    assert second == {"a": "value-a", "c": "value-c"}


def test_get_or_fetch_many_copy_result_isolates_stored_values():
    cache: SingleFlightCache[str, list[int]] = SingleFlightCache(maxsize=10)

    def fetch(keys: list[str]) -> dict[str, list[int]]:
        return {k: [1, 2, 3] for k in keys}

    import copy

    r1 = cache.get_or_fetch_many(["a"], fetch, copy_result=copy.deepcopy)
    r1["a"].append(4)
    r2 = cache.get_or_fetch_many(["a"], fetch, copy_result=copy.deepcopy)
    assert r2["a"] == [1, 2, 3]


def test_get_or_fetch_many_concurrent_leader_survives_a_clear_mid_fetch():
    """A leader publishes to its local slot and wakes followers after `clear()` empties the map."""
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)
    started = threading.Event()
    release = threading.Event()

    def fetch(keys: list[str]) -> dict[str, str]:
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return _echo(keys)

    leader_result: dict[str, dict[str, str]] = {}

    def leader() -> None:
        leader_result["value"] = cache.get_or_fetch_many(["a", "b"], fetch)

    leader_thread = threading.Thread(target=leader)
    leader_thread.start()
    assert started.wait(timeout=5), "leader never reached fetch"

    follower_result: dict[str, dict[str, str]] = {}

    def follower() -> None:
        follower_result["value"] = cache.get_or_fetch_many(
            ["a"], _echo, deadline=time.monotonic() + 5
        )

    follower_thread = threading.Thread(target=follower)
    follower_thread.start()
    time.sleep(0.05)  # best-effort: give the follower a chance to park before clear() races it

    cache.clear()  # simulates a graph-generation token-change clear firing mid-fetch

    release.set()
    leader_thread.join(timeout=5)
    follower_thread.join(timeout=5)

    assert not leader_thread.is_alive()
    assert not follower_thread.is_alive()
    assert leader_result["value"] == {"a": "value-a", "b": "value-b"}
    assert follower_result["value"] == {"a": "value-a"}


# ---------------------------------------------------------------------------
# Async entry point: aget_or_fetch
# ---------------------------------------------------------------------------


def test_async_concurrent_misses_for_the_same_key_fetch_once():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls: list[str] = []

        def fetch() -> str:
            calls.append("call")
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5), "test deadlocked waiting for release"
            return "value"

        deadline = time.monotonic() + 5
        leader = asyncio.create_task(cache.aget_or_fetch("k", fetch, deadline=deadline))
        await started.wait()
        follower = asyncio.create_task(cache.aget_or_fetch("k", fetch, deadline=deadline))
        release.set()
        results = await asyncio.gather(leader, follower)
        return results, calls

    results, calls = asyncio.run(scenario())
    assert len(calls) == 1, "two concurrent callers for the same key must share one fetch"
    assert results == ["value", "value"]


def test_async_failing_leader_wakes_followers_with_the_error_and_caches_nothing():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls: list[str] = []

        def failing_fetch() -> str:
            calls.append("call")
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5), "test deadlocked waiting for release"
            raise RuntimeError("leader's own failure")

        deadline = time.monotonic() + 5
        leader = asyncio.create_task(cache.aget_or_fetch("k", failing_fetch, deadline=deadline))
        await started.wait()
        follower = asyncio.create_task(cache.aget_or_fetch("k", failing_fetch, deadline=deadline))
        release.set()

        errors = []
        for task in (leader, follower):
            try:
                await task
            except BaseException as exc:
                errors.append(exc)
        return errors, calls

    errors, calls = asyncio.run(scenario())
    assert len(errors) == 2
    assert all(isinstance(exc, RuntimeError) for exc in errors)
    calls_before = len(calls)

    async def recovered_call():
        return await cache.aget_or_fetch("k", lambda: "recovered", deadline=time.monotonic() + 5)

    result = asyncio.run(recovered_call())
    assert result == "recovered"
    assert len(calls) == calls_before, "the failed leader's own fetch must not have been reused"


def test_async_follower_deadline_expires_while_the_leader_keeps_running():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def fetch() -> str:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5), "test deadlocked waiting for release"
            return "value"

        leader = asyncio.create_task(cache.aget_or_fetch("k", fetch, deadline=time.monotonic() + 5))
        await started.wait()

        before = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await cache.aget_or_fetch("k", fetch, deadline=time.monotonic() + 0.05)
        elapsed = time.monotonic() - before
        assert elapsed < 2, "the follower must not have waited for the leader"

        release.set()
        return await leader

    result = asyncio.run(scenario())
    assert result == "value"


def test_async_follower_deadline_is_measured_on_the_real_clock_not_the_injected_ttl_clock():
    """Follower deadlines use `time.monotonic()`, independent of the injected TTL clock."""
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, clock=lambda: time.monotonic() + 10_000
    )

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def fetch() -> str:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5), "test deadlocked waiting for release"
            return "value"

        leader = asyncio.create_task(cache.aget_or_fetch("k", fetch, deadline=time.monotonic() + 5))
        await started.wait()

        follower = asyncio.create_task(
            cache.aget_or_fetch("k", fetch, deadline=time.monotonic() + 5)
        )
        await asyncio.sleep(0.05)  # give the follower a chance to join the leader's task

        release.set()
        return await leader, await follower

    leader_result, follower_result = asyncio.run(scenario())
    assert leader_result == "value"
    assert follower_result == "value"


def test_async_cancelling_a_follower_does_not_cancel_the_leader():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls: list[str] = []

        def fetch() -> str:
            calls.append("call")
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5), "test deadlocked waiting for release"
            return "value"

        deadline = time.monotonic() + 5
        leader = asyncio.create_task(cache.aget_or_fetch("k", fetch, deadline=deadline))
        await started.wait()
        follower = asyncio.create_task(cache.aget_or_fetch("k", fetch, deadline=deadline))
        await asyncio.sleep(0.05)  # give the follower a chance to park on the shielded task

        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower

        release.set()
        leader_result = await leader
        later = await cache.aget_or_fetch("k", fetch, deadline=time.monotonic() + 5)
        return leader_result, later, calls

    leader_result, later, calls = asyncio.run(scenario())
    assert leader_result == "value"
    assert later == "value"
    assert len(calls) == 1, "neither the cancelled follower nor the later hit may re-fetch"


def test_async_is_cacheable_false_never_stores_but_still_answers_every_racer():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10)

    async def scenario():
        calls = []

        def fetch() -> str:
            calls.append("call")
            return "degraded"

        deadline = time.monotonic() + 5
        first = await cache.aget_or_fetch(
            "k", fetch, deadline=deadline, is_cacheable=lambda value: value != "degraded"
        )
        second = await cache.aget_or_fetch(
            "k", fetch, deadline=deadline, is_cacheable=lambda value: value != "degraded"
        )
        return first, second, calls

    first, second, calls = asyncio.run(scenario())
    assert first == second == "degraded"
    assert len(calls) == 2, "a rejected result must never be served from the store"


# ---------------------------------------------------------------------------
# Generation keying, bound, and TTL: entry-point-agnostic store behaviour,
# exercised once through the sync single-key entry point.
# ---------------------------------------------------------------------------


def test_a_stale_generation_is_a_miss_for_an_identical_key():
    generation = FakeGeneration(token="gen-a")
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10, generation=generation)
    calls: list[str] = []

    def fetch() -> str:
        calls.append("call")
        return "value"

    cache.get_or_fetch("k", fetch)
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 1, "the same token must serve the second call from the store"

    generation.advance("gen-b")
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 2, "a token change must re-fetch rather than reuse the old entry"


def test_unknown_and_known_generations_never_share_an_entry():
    generation = FakeGeneration()  # starts unknown
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=10, generation=generation)
    calls: list[str] = []

    def fetch() -> str:
        calls.append("call")
        return "value"

    cache.get_or_fetch("k", fetch)
    generation.advance("gen-a")  # unknown -> known, not itself a token "change"
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 2, "unknown and a known token must never share the identical key"


def test_the_bound_evicts_the_least_recently_used_entry():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=2)
    calls: list[str] = []

    def fetch(name: str):
        def _fetch() -> str:
            calls.append(name)
            return f"value-{name}"

        return _fetch

    cache.get_or_fetch("a", fetch("a"))
    cache.get_or_fetch("b", fetch("b"))
    cache.get_or_fetch("c", fetch("c"))  # evicts "a": least recently used
    assert len(calls) == 3

    cache.get_or_fetch("a", fetch("a"))
    assert len(calls) == 4, "a evicted entry must re-fetch"

    cache.get_or_fetch("c", fetch("c"))
    assert len(calls) == 4, "c (most recently used before a's re-fetch) must still be cached"


def test_ttl_from_the_generation_stamp_expires_an_entry():
    generation = FakeGeneration(token="gen-a")
    clock = [0.0]
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, generation=generation, clock=lambda: clock[0]
    )
    max_age_s = GenerationStamp(token="gen-a").max_age_s
    calls: list[str] = []

    def fetch() -> str:
        calls.append("call")
        return "value"

    cache.get_or_fetch("k", fetch)
    clock[0] = max_age_s - 1
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 1, "still within the stamp's own lifetime: must stay a hit"

    clock[0] = max_age_s + 1
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 2, "past the stamp's own lifetime: must re-fetch"


def test_custom_ttl_for_stamp_policy_overrides_the_stamps_own_max_age():
    clock = [0.0]
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, clock=lambda: clock[0], ttl_for_stamp=lambda stamp: 5.0
    )
    calls: list[str] = []

    def fetch() -> str:
        calls.append("call")
        return "value"

    cache.get_or_fetch("k", fetch)
    clock[0] = 4.0
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 1

    clock[0] = 6.0
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 2


def test_stats_and_clear():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=1)

    cache.get_or_fetch("a", lambda: "value-a")  # miss
    cache.get_or_fetch("a", lambda: "value-a")  # hit
    cache.get_or_fetch("b", lambda: "value-b")  # miss, evicts "a"

    assert cache.stats() == {"hits": 1, "misses": 2, "evictions": 1, "entries": 1}

    cache.clear()
    assert cache.stats()["entries"] == 0


def test_disabled_cache_bypasses_storage_on_every_entry_point():
    cache: SingleFlightCache[str, str] = SingleFlightCache(maxsize=0)
    calls: list[str] = []

    def fetch() -> str:
        calls.append("call")
        return "value"

    cache.get_or_fetch("a", fetch)
    cache.get_or_fetch("a", fetch)
    assert len(calls) == 2

    async def async_call():
        return await cache.aget_or_fetch("a", fetch, deadline=time.monotonic() + 5)

    asyncio.run(async_call())
    assert len(calls) == 3
