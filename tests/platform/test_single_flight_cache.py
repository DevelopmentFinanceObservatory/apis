"""Tests bounded, generation-keyed sync and async behavior in `SingleFlightCache`.

Concurrency cases use `threading.Event` and asyncio tasks.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from typing import Any

import pytest

from dfo_platform.request_deadline import bounded_deadline, request_deadline
from dfo_platform.single_flight_cache import AlreadyInFlight, SingleFlightCache, WaitTimeout
from graph_generation import FakeGeneration, GenerationStamp

_FLIGHTS = ThreadPoolExecutor(max_workers=8, thread_name_prefix="test-flights")
_FLIGHT_TIMEOUT_S = 5.0


def _cache(**kwargs: Any) -> SingleFlightCache[Any, Any]:
    """A cache whose sync flights run on the test pool under a 5 s flight deadline."""
    return SingleFlightCache(executor=_FLIGHTS, flight_timeout_s=_FLIGHT_TIMEOUT_S, **kwargs)


# ---------------------------------------------------------------------------
# Sync entry point: get_or_fetch (single key)
# ---------------------------------------------------------------------------


def test_sync_concurrent_misses_for_the_same_key_fetch_once():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def failing_fetch(_deadline: float) -> str:
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
    result = cache.get_or_fetch("k", lambda _deadline: "recovered")
    assert result == "recovered"
    assert len(calls) == calls_before, "the failed leader's own fetch must not have been reused"


def test_sync_follower_deadline_expires_while_the_leader_keeps_running():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    started = threading.Event()
    release = threading.Event()

    def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10, clock=lambda: time.monotonic() + 10_000)
    started = threading.Event()
    release = threading.Event()

    def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, None] = _cache(maxsize=10)
    calls: list[str] = []

    def fetch(_deadline: float) -> None:
        calls.append("call")
        return None

    first = cache.get_or_fetch("k", fetch)
    second = cache.get_or_fetch("k", fetch)

    assert first is None
    assert second is None
    assert len(calls) == 1, "a cached None must be served as a hit, not re-fetched"


def test_sync_concurrent_followers_get_a_leaders_none_result():
    """A parked follower receives the leader's cached `None` result."""
    cache: SingleFlightCache[str, None] = _cache(maxsize=10)
    started = threading.Event()
    release = threading.Event()

    def fetch(_deadline: float) -> None:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    old_owner_started = threading.Event()
    old_owner_release = threading.Event()

    def old_fetch(_deadline: float) -> str:
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

    def new_fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    started = threading.Event()
    release = threading.Event()

    def fetch(_deadline: float) -> str:
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


def _echo(keys: list[str], _deadline: float = 0.0) -> dict[str, str]:
    return {k: f"value-{k}" for k in keys}


def test_get_or_fetch_many_fetches_only_the_misses():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    calls: list[list[str]] = []

    def fetch(keys: list[str], _deadline: float) -> dict[str, str]:
        calls.append(list(keys))
        return _echo(keys)

    first = cache.get_or_fetch_many(["a", "b"], fetch)
    assert calls == [["a", "b"]]
    assert first == {"a": "value-a", "b": "value-b"}

    second = cache.get_or_fetch_many(["a", "c"], fetch)
    assert calls == [["a", "b"], ["c"]], "only the genuine miss reaches fetch"
    assert second == {"a": "value-a", "c": "value-c"}


def test_get_or_fetch_many_is_cacheable_false_answers_but_stores_only_the_accepted_keys():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    calls: list[list[str]] = []

    def fetch(keys: list[str], _deadline: float) -> dict[str, str]:
        calls.append(list(keys))
        return {k: "degraded" if k == "b" else f"value-{k}" for k in keys}

    def accepted(value: str) -> bool:
        return value != "degraded"

    first = cache.get_or_fetch_many(["a", "b", "c"], fetch, is_cacheable=accepted)
    second = cache.get_or_fetch_many(["a", "b", "c"], fetch, is_cacheable=accepted)

    assert first == second == {"a": "value-a", "b": "degraded", "c": "value-c"}
    assert calls == [["a", "b", "c"], ["b"]]


@pytest.mark.parametrize("awaited", [False, True], ids=["sync", "async"])
def test_get_or_fetch_many_flight_per_key_fails_a_key_alone(awaited: bool):
    """Each miss reads in its own flight, and a failed flight resolves its key through
    `on_error` while the other keys resolve and store."""
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    calls: list[list[str]] = []

    def fetch(keys: list[str], _deadline: float) -> dict[str, str]:
        calls.append(list(keys))
        if keys == ["b"]:
            raise ValueError("b failed")
        return _echo(keys)

    def on_error(key: str, exc: Exception) -> str:
        return f"failed-{key}-{type(exc).__name__}"

    def call() -> dict[str, str]:
        kwargs: dict[str, Any] = {"flight_per_key": True, "on_error": on_error}
        if awaited:
            return asyncio.run(cache.aget_or_fetch_many(["a", "b", "c"], fetch, **kwargs))
        return cache.get_or_fetch_many(["a", "b", "c"], fetch, **kwargs)

    first = call()
    second = call()

    assert first == second == {"a": "value-a", "b": "failed-b-ValueError", "c": "value-c"}
    assert sorted(calls[:3]) == [["a"], ["b"], ["c"]]
    assert calls[3:] == [["b"]]


def test_get_or_fetch_many_copy_result_isolates_stored_values():
    cache: SingleFlightCache[str, list[int]] = _cache(maxsize=10)

    def fetch(keys: list[str], _deadline: float) -> dict[str, list[int]]:
        return {k: [1, 2, 3] for k in keys}

    import copy

    r1 = cache.get_or_fetch_many(["a"], fetch, copy_result=copy.deepcopy)
    r1["a"].append(4)
    r2 = cache.get_or_fetch_many(["a"], fetch, copy_result=copy.deepcopy)
    assert r2["a"] == [1, 2, 3]


def test_get_or_fetch_many_concurrent_leader_survives_a_clear_mid_fetch():
    """A leader publishes to its local slot and wakes followers after `clear()` empties the map."""
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    started = threading.Event()
    release = threading.Event()

    def fetch(keys: list[str], _deadline: float) -> dict[str, str]:
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


def test_an_async_many_caller_joins_a_sync_flight_and_its_wait_holds_no_thread():
    """The async caller joins a sync caller's flight and starts one for its other key; while it
    waits, the loop's one default-executor thread stays free."""
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    started = threading.Event()
    release = threading.Event()
    calls: list[list[str]] = []

    def fetch(keys: list[str], _deadline: float) -> dict[str, str]:
        calls.append(list(keys))
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return _echo(keys)

    sync_result: dict[str, dict[str, str]] = {}
    sync = threading.Thread(
        target=lambda: sync_result.update(value=cache.get_or_fetch_many(["a"], fetch))
    )
    sync.start()
    assert started.wait(timeout=5), "the sync caller never reached fetch"

    async def run() -> dict[str, str]:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        waiting = asyncio.create_task(
            cache.aget_or_fetch_many(["a", "b"], fetch, deadline=time.monotonic() + 5)
        )
        assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), timeout=1) == "free"
        assert not waiting.done()
        release.set()
        return await waiting

    async_result = asyncio.run(run())
    sync.join(timeout=5)

    assert sync_result["value"] == {"a": "value-a"}
    assert async_result == {"a": "value-a", "b": "value-b"}
    assert calls == [["a"], ["b"]]


def test_an_async_many_wait_past_its_deadline_raises_while_the_flight_publishes():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    release = threading.Event()

    def fetch(keys: list[str], _deadline: float) -> dict[str, str]:
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return _echo(keys)

    async def run() -> None:
        with pytest.raises(WaitTimeout):
            await cache.aget_or_fetch_many(["a"], fetch, deadline=time.monotonic() + 0.05)

    asyncio.run(run())
    release.set()
    wait_deadline = time.monotonic() + 5
    while not cache.cached_many(["a"]):
        assert time.monotonic() < wait_deadline, "the flight never published"
        time.sleep(0.01)
    assert cache.cached_many(["a"]) == {"a": "value-a"}


# ---------------------------------------------------------------------------
# Async entry point: aget_or_fetch
# ---------------------------------------------------------------------------


def test_async_concurrent_misses_for_the_same_key_fetch_once():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls: list[str] = []

        def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls: list[str] = []

        def failing_fetch(_deadline: float) -> str:
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
        return await cache.aget_or_fetch(
            "k", lambda _deadline: "recovered", deadline=time.monotonic() + 5
        )

    result = asyncio.run(recovered_call())
    assert result == "recovered"
    assert len(calls) == calls_before, "the failed leader's own fetch must not have been reused"


def test_async_follower_deadline_expires_while_the_leader_keeps_running():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10, clock=lambda: time.monotonic() + 10_000)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls: list[str] = []

        def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)

    async def scenario():
        calls = []

        def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10, generation=generation)
    calls: list[str] = []

    def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=10, generation=generation)
    calls: list[str] = []

    def fetch(_deadline: float) -> str:
        calls.append("call")
        return "value"

    cache.get_or_fetch("k", fetch)
    generation.advance("gen-a")  # unknown -> known, not itself a token "change"
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 2, "unknown and a known token must never share the identical key"


def test_the_bound_evicts_the_least_recently_used_entry():
    cache: SingleFlightCache[str, str] = _cache(maxsize=2)
    calls: list[str] = []

    def fetch(name: str):
        def _fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(
        maxsize=10, generation=generation, clock=lambda: clock[0]
    )
    max_age_s = GenerationStamp(token="gen-a").max_age_s
    calls: list[str] = []

    def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(
        maxsize=10, clock=lambda: clock[0], ttl_for_stamp=lambda stamp: 5.0
    )
    calls: list[str] = []

    def fetch(_deadline: float) -> str:
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
    cache: SingleFlightCache[str, str] = _cache(maxsize=1)

    cache.get_or_fetch("a", lambda _deadline: "value-a")  # miss
    cache.get_or_fetch("a", lambda _deadline: "value-a")  # hit
    cache.get_or_fetch("b", lambda _deadline: "value-b")  # miss, evicts "a"

    assert cache.stats() == {"hits": 1, "misses": 2, "evictions": 1, "entries": 1}

    cache.clear()
    assert cache.stats()["entries"] == 0


def test_disabled_cache_bypasses_storage_on_every_entry_point():
    cache: SingleFlightCache[str, str] = _cache(maxsize=0)
    calls: list[str] = []

    def fetch(_deadline: float) -> str:
        calls.append("call")
        return "value"

    cache.get_or_fetch("a", fetch)
    cache.get_or_fetch("a", fetch)
    assert len(calls) == 2

    async def async_call():
        return await cache.aget_or_fetch("a", fetch, deadline=time.monotonic() + 5)

    asyncio.run(async_call())
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Flights: detached from the caller that starts them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["single", "many"])
@pytest.mark.parametrize("bound", ["ambient", "explicit"])
def test_a_starter_near_its_deadline_waits_like_any_caller_and_never_cuts_the_flight(
    entry: str, bound: str
):
    """The caller that starts a flight stops at its own deadline, ambient or explicit, while
    the flight runs under its own deadline, publishes, and serves a later caller in full."""
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    release = threading.Event()
    seen: list[tuple[float, float | None]] = []

    def fetch(flight_deadline: float) -> str:
        seen.append((flight_deadline, bounded_deadline(None)))
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return "full"

    def call(deadline: float | None) -> str:
        if entry == "single":
            return cache.get_or_fetch("k", fetch, deadline=deadline)
        return cache.get_or_fetch_many(
            ["k"], lambda keys, dl: {key: fetch(dl) for key in keys}, deadline=deadline
        )["k"]

    started_at = time.monotonic()
    starter_deadline = started_at + 0.1
    with pytest.raises(TimeoutError):
        if bound == "ambient":
            with request_deadline(starter_deadline):
                call(None)
        else:
            call(starter_deadline)
    assert time.monotonic() - started_at < 2, "the starter must stop at its own deadline"

    release.set()
    assert call(time.monotonic() + 5) == "full"
    assert len(seen) == 1, "the later caller must reuse the starter's flight"
    flight_deadline, ambient_in_fetch = seen[0]
    assert flight_deadline >= started_at + _FLIGHT_TIMEOUT_S
    assert ambient_in_fetch == flight_deadline, "the flight binds its own request deadline"


def test_an_async_flight_runs_under_its_own_deadline_not_the_starters():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    seen: list[tuple[float, float | None]] = []

    def fetch(flight_deadline: float) -> str:
        seen.append((flight_deadline, bounded_deadline(None)))
        time.sleep(0.2)
        return "full"

    async def scenario() -> str:
        started_at = time.monotonic()
        with request_deadline(started_at + 0.05), pytest.raises(asyncio.TimeoutError):
            await cache.aget_or_fetch("k", fetch, deadline=started_at + 5)
        return await cache.aget_or_fetch("k", fetch, deadline=time.monotonic() + 5)

    started_at = time.monotonic()
    assert asyncio.run(scenario()) == "full"
    assert len(seen) == 1
    flight_deadline, ambient_in_fetch = seen[0]
    assert flight_deadline >= started_at + _FLIGHT_TIMEOUT_S
    assert ambient_in_fetch == flight_deadline


def _blocked_pool() -> tuple[ThreadPoolExecutor, threading.Event]:
    """A one-thread pool whose thread waits on the returned event."""
    pool = ThreadPoolExecutor(max_workers=1)
    unblock = threading.Event()
    pool.submit(unblock.wait, 5)
    return pool, unblock


def _call_in_thread(call: Any) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            call()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)  # a stranded call must not hold exit
    thread.start()
    return thread, errors


def test_a_queued_flight_a_shutdown_cancels_releases_its_waiters():
    pool, unblock = _blocked_pool()
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, executor=pool, flight_timeout_s=5.0
    )
    calls: list[str] = []
    thread, errors = _call_in_thread(
        lambda: cache.get_or_fetch("k", lambda _deadline: calls.append("call") or "value")
    )
    time.sleep(0.05)  # let the caller queue its flight behind the blocked thread

    pool.shutdown(wait=False, cancel_futures=True)
    thread.join(timeout=5)
    unblock.set()

    assert not thread.is_alive(), "a cancelled flight must not strand its waiter"
    assert [type(exc) for exc in errors] == [CancelledError]
    assert calls == []


def test_a_flight_the_executor_refuses_fails_and_leaves_the_key_free():
    pool = ThreadPoolExecutor(max_workers=1)
    pool.shutdown()
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, executor=pool, flight_timeout_s=5.0
    )
    with pytest.raises(RuntimeError):
        cache.get_or_fetch("k", lambda _deadline: "value")
    # A fresh flight refused again, where a dead slot would time out the wait.
    with pytest.raises(RuntimeError):
        cache.get_or_fetch("k", lambda _deadline: "value", deadline=time.monotonic() + 1)


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_a_flight_that_never_gets_a_worker_ends_at_its_deadline_and_never_fetches(entry: str):
    pool, unblock = _blocked_pool()
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, executor=pool if entry == "sync" else None, flight_timeout_s=0.05
    )
    calls: list[str] = []
    elapsed: list[float] = []

    def fetch(_deadline: float) -> str:
        calls.append("call")
        return "value"

    def sync_calls() -> str:
        started = time.monotonic()
        try:
            cache.get_or_fetch("k", fetch)  # no deadline: only the flight's bounds the wait
        finally:
            elapsed.append(time.monotonic() - started)
        return "unreachable"

    async def async_calls() -> str:
        # The blocked pool is the loop's default executor, which async flights queue on.
        loop = asyncio.get_running_loop()
        loop.set_default_executor(pool)
        started = time.monotonic()
        try:
            await cache.aget_or_fetch("k", fetch, deadline=time.monotonic() + 5)
        finally:
            elapsed.append(time.monotonic() - started)
            # A free pool for the next flight; joining the expired one would raise instead.
            loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        return "unreachable"

    thread, errors = _call_in_thread(
        sync_calls if entry == "sync" else (lambda: asyncio.run(async_calls()))
    )
    thread.join(timeout=1)
    try:
        assert not thread.is_alive(), "a flight with no worker must answer by its deadline"
        assert [type(exc) for exc in errors] == [TimeoutError]
        assert 0.04 <= elapsed[0] < 0.5
        # The key is free: a new caller starts a new flight, where joining the old one would
        # raise `AlreadyInFlight` (sync) or the old flight's `TimeoutError` (async).
        if entry == "sync":
            with pytest.raises(TimeoutError):
                cache.get_or_fetch("k", fetch, join_inflight=False)
        else:

            async def fresh_call() -> str:
                return await cache.aget_or_fetch(
                    "k", lambda _deadline: "fresh", deadline=time.monotonic() + 5
                )

            assert asyncio.run(fresh_call()) == "fresh"
    finally:
        unblock.set()
        pool.shutdown()
    assert calls == []


def test_a_caller_arriving_after_a_flights_deadline_starts_a_new_flight():
    pool, unblock = _blocked_pool()
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, executor=pool, flight_timeout_s=0.05
    )
    try:
        # The only waiter leaves early, so nothing expires the flight before the next caller.
        with pytest.raises(WaitTimeout):
            cache.get_or_fetch("k", lambda _deadline: "value", deadline=time.monotonic() + 0.01)
        time.sleep(0.1)
        with pytest.raises(TimeoutError) as raised:
            cache.get_or_fetch("k", lambda _deadline: "value", join_inflight=False)
        assert not isinstance(raised.value, AlreadyInFlight | WaitTimeout)
    finally:
        unblock.set()
        pool.shutdown()


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_a_fetch_that_ends_after_its_flight_deadline_publishes_nothing(entry: str):
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=10, executor=_FLIGHTS, flight_timeout_s=0.05
    )
    release = threading.Event()
    calls: list[str] = []

    def slow_fetch(_deadline: float) -> str:
        calls.append("slow")
        release.wait(5)
        return "late"

    def fast_fetch(_deadline: float) -> str:
        calls.append("fast")
        return "fresh"

    def sync_scenario() -> str:
        try:
            with pytest.raises(TimeoutError):
                cache.get_or_fetch("k", slow_fetch)
        finally:
            release.set()
        time.sleep(0.05)  # let the late fetch return and settle
        return cache.get_or_fetch("k", fast_fetch)

    async def async_scenario() -> str:
        # One loop throughout, so the expired flight's task could still publish if it ran on.
        try:
            with pytest.raises(TimeoutError):
                await cache.aget_or_fetch("k", slow_fetch, deadline=time.monotonic() + 1)
        finally:
            release.set()
        await asyncio.sleep(0.05)  # let the late fetch return
        return await cache.aget_or_fetch("k", fast_fetch, deadline=time.monotonic() + 5)

    fresh = sync_scenario() if entry == "sync" else asyncio.run(async_scenario())
    assert fresh == "fresh"
    assert calls == ["slow", "fast"]


def test_a_settle_that_raises_resolves_the_waiters_and_leaves_the_key_free():
    cache: SingleFlightCache[str, str] = _cache(maxsize=10)
    returns: list[Any] = [None, {"k": "value"}]

    def fetch(_keys: list[str], _deadline: float) -> Any:
        return returns.pop(0)

    thread, errors = _call_in_thread(lambda: cache.get_or_fetch_many(["k"], fetch))
    thread.join(timeout=5)  # no request deadline, so a stranded slot would wait forever

    assert not thread.is_alive()
    assert [type(exc) for exc in errors] == [TypeError]
    assert cache.get_or_fetch_many(["k"], fetch, deadline=time.monotonic() + 5) == {"k": "value"}
