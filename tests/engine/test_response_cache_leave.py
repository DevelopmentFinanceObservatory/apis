"""Pins lease interest counts, orphan callbacks, and late settlement."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.response_cache import Lease, ResponseCache


def _cache() -> ResponseCache:
    return ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)


async def _claim(cache: ResponseCache, key: str = "k") -> Lease:
    value, lease = await cache.acquire(key)
    assert value is None and lease is not None
    return lease


async def _not_woken(cache: ResponseCache, lease: Lease) -> bool:
    """True when `lease`'s waiter stays unset for a short wait."""
    try:
        await asyncio.wait_for(cache.await_leader(lease), timeout=0.05)
    except TimeoutError:
        return True
    return False


def test_leave_with_interest_remaining_does_nothing():
    async def scenario():
        cache = _cache()
        leader = await _claim(cache)
        follower = await _claim(cache)
        orphaned: list[str] = []
        cache.on_orphaned(leader, lambda: orphaned.append("cancel"))

        cache.leave(leader)

        assert orphaned == []
        # The flight is still in place: a newcomer joins it as a follower.
        newcomer = await _claim(cache)
        assert not newcomer.is_leader
        await cache.publish(leader, b"answer", 6)
        assert await cache.await_leader(follower) == (b"answer", False, True)
        assert await cache.await_leader(newcomer) == (b"answer", False, True)

    asyncio.run(scenario())


def test_last_leave_detaches_the_flight_and_calls_the_callback_once():
    async def scenario():
        cache = _cache()
        leader = await _claim(cache)
        follower = await _claim(cache)
        orphaned: list[str] = []
        cache.on_orphaned(leader, lambda: orphaned.append("cancel"))

        cache.leave(leader)
        cache.leave(follower)
        # A second leave from the same caller is not a second departure.
        cache.leave(follower)
        cache.leave(leader)

        assert orphaned == ["cancel"]
        fresh = await _claim(cache)
        assert fresh.is_leader
        assert fresh._waiter is not leader._waiter

    asyncio.run(scenario())


def test_leave_after_publish_does_not_call_the_callback():
    async def scenario():
        cache = _cache()
        leader = await _claim(cache)
        orphaned: list[str] = []
        cache.on_orphaned(leader, lambda: orphaned.append("cancel"))

        await cache.publish(leader, b"answer", 6)
        cache.leave(leader)

        assert orphaned == []
        assert await cache.get("k") == b"answer"

    asyncio.run(scenario())


@pytest.mark.parametrize("settle", ["abandon", "publish"])
def test_a_detached_leases_late_settle_leaves_a_newer_flight_untouched(settle: str):
    async def scenario():
        cache = _cache()
        old = await _claim(cache)
        cache.on_orphaned(old, lambda: None)
        cache.leave(old)

        newer = await _claim(cache)
        assert newer.is_leader
        parked = await _claim(cache)
        assert not parked.is_leader

        if settle == "abandon":
            await cache.abandon(old)
        else:
            await cache.publish(old, b"late", 4)

        # The newer flight's waiter is neither woken nor unregistered.
        assert await _not_woken(cache, parked)
        joiner = await _claim(cache) if settle == "abandon" else None
        if joiner is not None:
            assert not joiner.is_leader
            assert joiner._waiter is newer._waiter

        await cache.publish(newer, b"newer", 5)
        assert await cache.await_leader(parked) == (b"newer", False, True)

    asyncio.run(scenario())


def test_a_detached_leases_late_publish_still_stores_its_value():
    """A fetch that finished before its cancellation landed publishes a
    correct answer, so the store keeps it even though nobody is parked."""

    async def scenario():
        cache = _cache()
        old = await _claim(cache)
        cache.on_orphaned(old, lambda: None)
        cache.leave(old)

        await cache.publish(old, b"late", 4)

        assert await cache.get("k") == b"late"

    asyncio.run(scenario())


def test_without_leave_calls_a_flight_outlives_every_cancelled_follower():
    """The search response cache never calls `leave`: every follower it
    parks can be cancelled and the flight stays claimed until the leader
    settles it."""

    async def scenario():
        cache = _cache()
        leader = await _claim(cache)
        followers = [await _claim(cache) for _ in range(3)]
        waits = [asyncio.create_task(cache.await_leader(f)) for f in followers]
        await asyncio.sleep(0)
        for wait in waits:
            wait.cancel()
        await asyncio.gather(*waits, return_exceptions=True)

        late = await _claim(cache)
        assert not late.is_leader
        await cache.publish(leader, b"answer", 6)
        assert await cache.await_leader(late) == (b"answer", False, True)

    asyncio.run(scenario())
