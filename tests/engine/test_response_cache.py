"""Pins `ResponseCache` storage, expiry, single-flight, and thread-safe clearing."""

from __future__ import annotations

import asyncio
import threading
import time

import graph_generation.config as generation_config
from search.engine.response_cache import ResponseCache


async def _test_get_returns_stored_value_and_miss_returns_none():
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)
    _, lease = await cache.acquire("k1")
    assert lease is not None and lease.is_leader
    await cache.publish(lease, b"answer-1", 10)

    assert await cache.get("k1") == b"answer-1"
    assert await cache.get("missing") is None


async def _test_entry_past_ttl_is_a_miss():
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=0)
    _, lease = await cache.acquire("k1")
    await cache.publish(lease, b"answer-1", 10)

    # A zero TTL expires the entry immediately.
    assert await cache.get("k1") is None
    value, lease = await cache.acquire("k1")
    assert value is None
    assert lease is not None and lease.is_leader


async def _test_eviction_drops_lru_and_keeps_mru():
    # Each entry costs 10 bytes; a 25-byte budget holds two but not three.
    cache = ResponseCache(max_bytes=25, max_entry_bytes=1_000_000, ttl_s=60)

    _, lease_a = await cache.acquire("a")
    await cache.publish(lease_a, b"a", 10)
    _, lease_b = await cache.acquire("b")
    await cache.publish(lease_b, b"b", 10)
    # Touch "a" so "b" becomes the least-recently-used entry.
    assert await cache.get("a") == b"a"
    _, lease_c = await cache.acquire("c")
    await cache.publish(lease_c, b"c", 10)

    assert await cache.get("b") is None  # evicted: was LRU at the time "c" landed
    assert await cache.get("a") == b"a"
    assert await cache.get("c") == b"c"
    stats = cache.stats()
    assert stats.evictions == 1
    assert stats.entries == 2
    assert stats.total_bytes == 20


async def _test_entry_over_max_entry_bytes_is_not_stored():
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=100, ttl_s=60)
    _, lease = await cache.acquire("big")
    await cache.publish(lease, b"too big", 101)

    assert await cache.get("big") is None
    assert cache.stats().entries == 0


async def _test_oversized_publish_counts_but_does_not_store_and_still_wakes_a_waiter():
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=100, ttl_s=60)

    _, lease_a = await cache.acquire("big")
    _, lease_b = await cache.acquire("big")
    assert lease_a is not None and lease_b is not None

    waiter_task = asyncio.ensure_future(cache.await_leader(lease_b))
    await asyncio.sleep(0)

    await cache.publish(lease_a, b"too big", 101)

    # Oversize affects storage, not the response's storable verdict.
    assert await waiter_task == (b"too big", False, True)
    assert await cache.get("big") is None
    stats = cache.stats()
    assert stats.entries == 0
    assert stats.oversized == 1


async def _test_unstorable_publish_wakes_a_waiter_with_timed_out_but_stores_nothing():
    """An unstorable timed-out result wakes waiters without incrementing the oversize counter."""
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)

    _, lease_a = await cache.acquire("k1")
    _, lease_b = await cache.acquire("k1")
    assert lease_a is not None and lease_b is not None

    waiter_task = asyncio.ensure_future(cache.await_leader(lease_b))
    await asyncio.sleep(0)

    await cache.publish(lease_a, b"partial", 7, storable=False, timed_out=True)

    assert await waiter_task == (b"partial", True, False)
    assert await cache.get("k1") is None
    stats = cache.stats()
    assert stats.entries == 0
    assert stats.oversized == 0


async def _test_stored_size_equals_the_actual_retained_byte_length():
    """The stored size equals the retained payload length."""
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)
    payload = b'{"parts": [1, 2, 3], "note": "a real encoded response payload"}'

    _, lease = await cache.acquire("k1")
    await cache.publish(lease, payload, len(payload))

    stored = await cache.get("k1")
    assert stored == payload
    assert cache.stats().total_bytes == len(payload)


async def _test_one_bytes_object_under_two_keys_is_charged_once():
    """Two keys holding one bytes object charge its size once, and the last key to go frees it.

    The 15-byte budget fits one 10-byte value but not two.
    """
    cache = ResponseCache(max_bytes=15, max_entry_bytes=1_000_000, ttl_s=60)
    shared = bytes(10)
    for key in ("layer1", "layer2"):
        _, lease = await cache.acquire(key)
        await cache.publish(lease, shared, 10)
    stats = cache.stats()
    assert (stats.entries, stats.total_bytes, stats.evictions) == (2, 10, 0)

    # An equal value in another object is charged again. Evicting "layer1" frees nothing
    # while "layer2" holds the shared value, so both go.
    _, lease = await cache.acquire("other")
    await cache.publish(lease, bytes(bytearray(10)), 10)
    stats = cache.stats()
    assert (stats.entries, stats.total_bytes, stats.evictions) == (1, 10, 2)
    assert await cache.get("other") == bytes(10)


async def _test_concurrent_acquire_single_flight_and_await_leader():
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)

    value_a, lease_a = await cache.acquire("k1")
    value_b, lease_b = await cache.acquire("k1")

    assert value_a is None and value_b is None
    assert lease_a is not None and lease_b is not None
    assert lease_a.is_leader is True
    assert lease_b.is_leader is False
    assert cache.stats().follower_waits == 1

    waiter_task = asyncio.ensure_future(cache.await_leader(lease_b))
    await asyncio.sleep(0)  # let the waiter actually park before publishing
    assert not waiter_task.done()

    await cache.publish(lease_a, b"answer-42", 10)

    assert await waiter_task == (b"answer-42", False, True)


async def _test_abandon_wakes_waiter_with_none():
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)

    _, lease_a = await cache.acquire("k1")
    _, lease_b = await cache.acquire("k1")
    assert lease_a is not None and lease_b is not None

    waiter_task = asyncio.ensure_future(cache.await_leader(lease_b))
    await asyncio.sleep(0)

    await cache.abandon(lease_a)

    assert await waiter_task is None
    assert await cache.get("k1") is None  # abandon stores nothing


def test_bare_constructor_defaults_ttl_s_to_the_unknown_generation_lifetime():
    """A bare cache uses the unknown-generation lifetime until a write sets `max_age_s`."""
    cache = ResponseCache()
    assert cache._ttl_s == generation_config.GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS


async def _test_publish_max_age_s_overrides_the_constructor_ttl_per_entry():
    """A per-entry `max_age_s` overrides the constructor TTL for that entry only."""
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=3600)
    before = time.monotonic()

    _, lease_short = await cache.acquire("short")
    await cache.publish(lease_short, b"short-lived", 11, max_age_s=0.0)

    _, lease_default = await cache.acquire("default")
    await cache.publish(lease_default, b"default-lived", 13)

    # The short-lived entry is already expired; the default one, written
    # with no max_age_s, still lives under the constructor's own ttl_s.
    assert await cache.get("short") is None
    assert await cache.get("default") == b"default-lived"
    assert abs(cache._store["default"].expires_at - (before + 3600)) < 5.0


async def _test_clear_sync_from_a_foreign_thread_drops_the_store():
    """A real background thread can schedule `clear_sync` on the event loop."""
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)
    _, lease = await cache.acquire("k1")
    await cache.publish(lease, b"answer-1", 10)
    assert await cache.get("k1") == b"answer-1"

    loop = asyncio.get_running_loop()
    scheduled = threading.Event()

    def _from_foreign_thread() -> None:
        loop.call_soon_threadsafe(cache.clear_sync)
        scheduled.set()

    thread = threading.Thread(target=_from_foreign_thread)
    thread.start()
    thread.join(timeout=5.0)
    assert scheduled.is_set()

    # Poll until the scheduled callback runs.
    for _ in range(100):
        if await cache.get("k1") is None:
            break
        await asyncio.sleep(0.01)

    assert await cache.get("k1") is None
    assert cache.stats().entries == 0
    assert cache.stats().total_bytes == 0


async def _test_clear_sync_does_not_disturb_an_in_flight_lease():
    """Clearing published entries leaves an active leader and waiter intact."""
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)

    _, lease_a = await cache.acquire("k1")
    _, lease_b = await cache.acquire("k1")
    assert lease_a is not None and lease_b is not None

    waiter_task = asyncio.ensure_future(cache.await_leader(lease_b))
    await asyncio.sleep(0)

    cache.clear_sync()  # inline here, only the store-vs-inflight split is under test

    await cache.publish(lease_a, b"answer-1", 10)

    assert await waiter_task == (b"answer-1", False, True)
    assert await cache.get("k1") == b"answer-1"


async def _test_acquire_on_a_different_key_does_not_block_on_an_in_flight_leader():
    """A leader waiting on one key does not block acquisition of another."""
    cache = ResponseCache(max_bytes=1_000_000, max_entry_bytes=1_000_000, ttl_s=60)
    leader_may_finish = asyncio.Event()

    async def leader_work():
        _, lease = await cache.acquire("x")
        assert lease is not None and lease.is_leader
        await leader_may_finish.wait()  # simulates a whole search pipeline, in flight
        await cache.publish(lease, b"x", 10)

    leader_task = asyncio.ensure_future(leader_work())
    await asyncio.sleep(0)  # let the leader claim "x" and start waiting

    # "y" is a fully independent key. Its acquire() must complete promptly
    # rather than waiting on "x"'s in-flight leader.
    value_y, lease_y = await asyncio.wait_for(cache.acquire("y"), timeout=1.0)
    assert value_y is None
    assert lease_y is not None and lease_y.is_leader

    leader_may_finish.set()
    await leader_task
    assert await cache.get("x") == b"x"


def test_get_returns_stored_value_and_miss_returns_none():
    asyncio.run(_test_get_returns_stored_value_and_miss_returns_none())


def test_entry_past_ttl_is_a_miss():
    asyncio.run(_test_entry_past_ttl_is_a_miss())


def test_eviction_drops_lru_and_keeps_mru():
    asyncio.run(_test_eviction_drops_lru_and_keeps_mru())


def test_entry_over_max_entry_bytes_is_not_stored():
    asyncio.run(_test_entry_over_max_entry_bytes_is_not_stored())


def test_oversized_publish_counts_but_does_not_store_and_still_wakes_a_waiter():
    asyncio.run(_test_oversized_publish_counts_but_does_not_store_and_still_wakes_a_waiter())


def test_unstorable_publish_wakes_a_waiter_with_timed_out_but_stores_nothing():
    asyncio.run(_test_unstorable_publish_wakes_a_waiter_with_timed_out_but_stores_nothing())


def test_stored_size_equals_the_actual_retained_byte_length():
    asyncio.run(_test_stored_size_equals_the_actual_retained_byte_length())


def test_one_bytes_object_under_two_keys_is_charged_once():
    asyncio.run(_test_one_bytes_object_under_two_keys_is_charged_once())


def test_concurrent_acquire_single_flight_and_await_leader():
    asyncio.run(_test_concurrent_acquire_single_flight_and_await_leader())


def test_abandon_wakes_waiter_with_none():
    asyncio.run(_test_abandon_wakes_waiter_with_none())


def test_acquire_on_a_different_key_does_not_block_on_an_in_flight_leader():
    asyncio.run(_test_acquire_on_a_different_key_does_not_block_on_an_in_flight_leader())


def test_clear_sync_from_a_foreign_thread_drops_the_store():
    asyncio.run(_test_clear_sync_from_a_foreign_thread_drops_the_store())


def test_clear_sync_does_not_disturb_an_in_flight_lease():
    asyncio.run(_test_clear_sync_does_not_disturb_an_in_flight_lease())


def test_publish_max_age_s_overrides_the_constructor_ttl_per_entry():
    asyncio.run(_test_publish_max_age_s_overrides_the_constructor_ttl_per_entry())
