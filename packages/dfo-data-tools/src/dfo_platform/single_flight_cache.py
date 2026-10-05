"""Bounded, generation-keyed cache that coalesces concurrent fetches per key.

Each call captures one generation stamp before lookup and keys entries by `(stamp.key, key)`.
The default TTL is `stamp.max_age_s`. `max_entries` bounds the LRU store. One lock protects O(1)
bookkeeping and is released before fetches or awaits.

Sync callers share a `threading.Event` per in-flight key. Async callers share a cache-owned task
through `asyncio.shield`. Caller cancellation detaches its await while the task runs to completion.
Sync and async in-flight maps are independent. Deadlines use real `time.monotonic()`. The injectable
`clock` controls TTL only.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field

from graph_generation import UNKNOWN_GENERATION, GenerationProvider, GenerationStamp

_DEFAULT_TIMEOUT_MESSAGE = (
    "single-flight cache: deadline elapsed while the leader was still running"
)


def _default_ttl_for_stamp(stamp: GenerationStamp) -> float:
    return stamp.max_age_s


class AlreadyInFlight(Exception):
    """Raised when `join_inflight=False` and another caller owns the key."""


@dataclass
class _SyncSlot[V]:
    """One in-flight key's event and result.
    `ready` releases followers after the owner stores a value or error.
    `_MISS` distinguishes a stored `None` from an unset slot.
    """

    ready: threading.Event = field(default_factory=threading.Event)
    value: V | _Miss = field(default_factory=lambda: _MISS)
    error: BaseException | None = None


def _retrieve_exception[V](task: asyncio.Task[V]) -> None:
    """Consume a failed task's exception to suppress asyncio's warning if all callers detach.

    Later awaits raise the exception normally. Cancelled tasks carry no unhandled exception warning.
    """
    if not task.cancelled():
        task.exception()


class SingleFlightCache[K, V]:
    """Bounded per-key cache. `maxsize=0` calls `fetch` directly without lookup or coalescing."""

    def __init__(
        self,
        *,
        maxsize: int,
        generation: GenerationProvider = UNKNOWN_GENERATION,
        clock: Callable[[], float] = time.monotonic,
        ttl_for_stamp: Callable[[GenerationStamp], float] = _default_ttl_for_stamp,
        timeout_message: str = _DEFAULT_TIMEOUT_MESSAGE,
    ) -> None:
        self._maxsize = maxsize
        self._generation = generation
        self._clock = clock
        self._ttl_for_stamp = ttl_for_stamp
        self._timeout_message = timeout_message
        self._lock = threading.Lock()
        self._store: OrderedDict[tuple[str, K], tuple[float, V]] = OrderedDict()
        self._sync_inflight: dict[tuple[str, K], _SyncSlot[V]] = {}
        self._async_inflight: dict[tuple[str, K], asyncio.Task[V]] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        return self._maxsize > 0

    # -------------------------------------------------------------------
    # Sync entry points
    # -------------------------------------------------------------------

    def get_or_fetch(
        self,
        key: K,
        fetch: Callable[[], V],
        *,
        deadline: float | None = None,
        join_inflight: bool = True,
    ) -> V:
        """Return a cached value or fetch it once. A follower deadline uses real
        `time.monotonic()` and raises `TimeoutError` without stopping the leader. With
        `join_inflight=False`, a live hit returns and an unowned miss fetches. An owned miss raises
        `AlreadyInFlight` immediately.
        """
        if not self.enabled:
            return fetch()

        stamp = self._generation.stamp()
        full_key = (stamp.key, key)
        is_owner, slot, cached = self._claim_sync(full_key)
        if not isinstance(cached, _Miss):
            return cached
        assert slot is not None  # every non-hit path claims or joins a slot

        if not is_owner:
            if not join_inflight:
                raise AlreadyInFlight(key)
            self._await_sync_slot(slot, deadline=deadline)
            if slot.error is not None:
                raise slot.error
            assert not isinstance(slot.value, _Miss)  # a slot with no error always has a value
            return slot.value

        self._run_sync_owner(full_key, slot, fetch, ttl=self._ttl_for_stamp(stamp))
        if slot.error is not None:
            raise slot.error
        assert not isinstance(slot.value, _Miss)
        return slot.value

    def get_or_fetch_many(
        self,
        keys: Sequence[K],
        fetch: Callable[[list[K]], dict[K, V]],
        *,
        deadline: float | None = None,
        copy_result: Callable[[V], V] | None = None,
    ) -> dict[K, V]:
        """Resolve unique keys in input order, fetching only misses this call owns. Fetch failures
        propagate to all followers. The cache stores successful fetches only.
        `deadline` bounds follower waits.
        `copy_result`, when given, copies hits and fetched values.
        """
        if not self.enabled:
            return fetch(list(dict.fromkeys(keys)))

        stamp = self._generation.stamp()
        ttl = self._ttl_for_stamp(stamp)
        unique_keys = list(dict.fromkeys(keys))
        full_keys = {key: (stamp.key, key) for key in unique_keys}
        hits: dict[K, V] = {}
        owned: list[K] = []
        owned_slots: dict[K, _SyncSlot[V]] = {}
        waiting: list[tuple[K, _SyncSlot[V]]] = []
        with self._lock:
            now = self._clock()
            for key in unique_keys:
                full_key = full_keys[key]
                value = self._lookup_locked(full_key, now=now)
                if not isinstance(value, _Miss):
                    hits[key] = value
                    continue
                slot = self._sync_inflight.get(full_key)
                if slot is not None:
                    waiting.append((key, slot))
                else:
                    slot = _SyncSlot()
                    self._sync_inflight[full_key] = slot
                    owned.append(key)
                    owned_slots[key] = slot

        if owned:
            fetch_error: BaseException | None = None
            fetched: dict[K, V] = {}
            try:
                fetched = fetch(owned)
            except BaseException as exc:
                fetch_error = exc
            finally:
                # Keep owned slots locally because `clear()` can empty the registry during fetch.
                # Remove only the matching entry, then signal its followers.
                with self._lock:
                    now = self._clock()
                    for key in owned:
                        full_key = full_keys[key]
                        slot = owned_slots[key]
                        if self._sync_inflight.get(full_key) is slot:
                            del self._sync_inflight[full_key]
                        if fetch_error is None:
                            slot.value = fetched[key]
                            self._publish_locked(full_key, fetched[key], ttl=ttl, now=now)
                        else:
                            slot.error = fetch_error
                        slot.ready.set()
            if fetch_error is not None:
                raise fetch_error
            hits.update(fetched)

        for key, slot in waiting:
            self._await_sync_slot(slot, deadline=deadline)
            if slot.error is not None:
                raise slot.error
            hits[key] = slot.value  # ty: ignore[invalid-assignment] -- set before ready, under the lock

        if copy_result is None:
            return {key: hits[key] for key in unique_keys}
        return {key: copy_result(hits[key]) for key in unique_keys}

    def _await_sync_slot(self, slot: _SyncSlot[V], *, deadline: float | None) -> None:
        if deadline is None:
            slot.ready.wait()
            return
        remaining = max(0.0, deadline - time.monotonic())
        if not slot.ready.wait(timeout=remaining):
            raise TimeoutError(self._timeout_message)

    def _claim_sync(self, full_key: tuple[str, K]) -> tuple[bool, _SyncSlot[V] | None, V | _Miss]:
        """Under the lock, return `(False, None, value)` for a hit, `(False, slot, _MISS)` to join,
        or `(True, slot, _MISS)` for the owner. `_MISS` distinguishes cached `None`.
        """
        with self._lock:
            now = self._clock()
            value = self._lookup_locked(full_key, now=now)
            if not isinstance(value, _Miss):
                return False, None, value
            existing = self._sync_inflight.get(full_key)
            if existing is not None:
                return False, existing, _MISS
            slot = _SyncSlot()
            self._sync_inflight[full_key] = slot
            return True, slot, _MISS

    def _run_sync_owner(
        self, full_key: tuple[str, K], slot: _SyncSlot[V], fetch: Callable[[], V], *, ttl: float
    ) -> None:
        try:
            value = fetch()
        except BaseException as exc:
            with self._lock:
                if self._sync_inflight.get(full_key) is slot:
                    del self._sync_inflight[full_key]
            slot.error = exc
            slot.ready.set()
            return
        with self._lock:
            if self._sync_inflight.get(full_key) is slot:
                del self._sync_inflight[full_key]
            self._publish_locked(full_key, value, ttl=ttl, now=self._clock())
        slot.value = value
        slot.ready.set()

    # -------------------------------------------------------------------
    # Async entry point
    # -------------------------------------------------------------------

    async def aget_or_fetch(
        self,
        key: K,
        fetch: Callable[[], V],
        *,
        deadline: float,
        is_cacheable: Callable[[V], bool] | None = None,
    ) -> V:
        """Resolve on the event loop, running blocking `fetch` in a worker thread. Callers share one
        shielded task, so their cancellation or deadline detaches only their own await. A value
        rejected by `is_cacheable` reaches current callers and leaves the store unchanged.
        """
        if not self.enabled:
            return await self._await_bounded(asyncio.to_thread(fetch), deadline=deadline)

        stamp = self._generation.stamp()
        full_key = (stamp.key, key)
        ttl = self._ttl_for_stamp(stamp)
        task, cached = self._claim_task(full_key, fetch, ttl=ttl, is_cacheable=is_cacheable)
        if not isinstance(cached, _Miss):
            return cached
        assert task is not None  # every non-hit path claims or joins a task

        return await self._await_bounded(self._await_shielded(task), deadline=deadline)

    @staticmethod
    async def _await_shielded(task: asyncio.Task[V]) -> V:
        return await asyncio.shield(task)

    async def _await_bounded(self, coro: Coroutine[object, object, V], *, deadline: float) -> V:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            coro.close()  # Close the unawaited coroutine to avoid an asyncio warning.
            raise TimeoutError(self._timeout_message)
        return await asyncio.wait_for(coro, timeout=remaining)

    def _claim_task(
        self,
        full_key: tuple[str, K],
        fetch: Callable[[], V],
        *,
        ttl: float,
        is_cacheable: Callable[[V], bool] | None,
    ) -> tuple[asyncio.Task[V] | None, V | _Miss]:
        """Under the lock, return `(None, value)` for a hit or `(task, _MISS)` for the leader or a
        follower. `_MISS` distinguishes cached `None`.
        """
        with self._lock:
            now = self._clock()
            value = self._lookup_locked(full_key, now=now)
            if not isinstance(value, _Miss):
                return None, value
            existing = self._async_inflight.get(full_key)
            if existing is not None:
                return existing, _MISS
            task = asyncio.create_task(
                self._run_and_store(full_key, fetch, ttl=ttl, is_cacheable=is_cacheable)
            )
            task.add_done_callback(_retrieve_exception)
            self._async_inflight[full_key] = task
            return task, _MISS

    async def _run_and_store(
        self,
        full_key: tuple[str, K],
        fetch: Callable[[], V],
        *,
        ttl: float,
        is_cacheable: Callable[[V], bool] | None,
    ) -> V:
        this_task = asyncio.current_task()
        try:
            value = await asyncio.to_thread(fetch)
        except BaseException:
            with self._lock:
                if self._async_inflight.get(full_key) is this_task:
                    del self._async_inflight[full_key]
            raise
        with self._lock:
            if self._async_inflight.get(full_key) is this_task:
                del self._async_inflight[full_key]
            if is_cacheable is None or is_cacheable(value):
                self._publish_locked(full_key, value, ttl=ttl, now=self._clock())
        return value

    # -------------------------------------------------------------------
    # Shared store (always accessed under `self._lock`)
    # -------------------------------------------------------------------

    def _lookup_locked(self, full_key: tuple[str, K], *, now: float) -> V | _Miss:
        """A live hit moves `full_key` to most-recently-used, counts a hit,
        and returns its value. Anything else (absent, or expired, dropped
        here) counts a miss and returns the `_MISS` sentinel."""
        entry = self._store.get(full_key)
        if entry is not None:
            expires_at, value = entry
            if expires_at > now:
                self._store.move_to_end(full_key)
                self._hits += 1
                return value
            del self._store[full_key]  # expired: fall through, treat as a miss
        self._misses += 1
        return _MISS

    def _publish_locked(self, full_key: tuple[str, K], value: V, *, ttl: float, now: float) -> None:
        self._store[full_key] = (now + ttl, value)
        self._store.move_to_end(full_key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)
            self._evictions += 1

    # -------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------

    def stats(self) -> Mapping[str, int]:
        """Hit, miss and eviction counters plus the current entry count."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "entries": len(self._store),
            }

    def clear(self) -> None:
        """Clear entries and in-flight bookkeeping. Active fetches continue to completion."""
        with self._lock:
            self._store.clear()
            self._sync_inflight.clear()
            self._async_inflight.clear()


class _Miss:
    """The sentinel `_lookup_locked` returns on a miss, distinct from any
    real `V` (including `None`, a legitimate cached value)."""


_MISS = _Miss()
