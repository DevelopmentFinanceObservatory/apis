"""Bounded async cache of finished search responses with per-key single-flight.

The lock guards cache bookkeeping only; callers run pipeline work outside it,
and followers wait on a per-key event. Different keys never block one another.
`leave()` can cancel a flight with no interested callers; the search cache does
not use it, while the LLM memo does.
"""

from __future__ import annotations

import time
from asyncio import Event, Lock
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import graph_generation.config as generation_config  # the constructor's own unknown-generation TTL default
import search.engine.config as config  # module-attribute access for runtime-resolved values


@dataclass
class Lease:
    """A cache miss claim. Leaders fetch and publish; followers await the leader.

    `_waiter` survives removal of the in-flight entry, and `_left` makes
    `leave()` idempotent.
    """

    key: str
    is_leader: bool
    _waiter: _Waiter | None = field(default=None, repr=False, compare=False)
    _left: bool = field(default=False, repr=False, compare=False)


@dataclass
class ResponseCacheStats:
    hits: int
    misses: int
    evictions: int
    entries: int
    total_bytes: int
    oversized: int


@dataclass
class _Entry:
    value: bytes
    size: int
    expires_at: float


@dataclass
class _Waiter:
    """Shared signal and result for one in-flight key.

    `event` wakes all followers. `value=None` means abandonment; otherwise
    `timed_out` and `storable` carry the leader's verdict. `interest` counts
    leases not yet released; `on_orphaned` runs when that count reaches zero.
    """

    event: Event = field(default_factory=Event)
    value: bytes | None = None
    timed_out: bool = False
    storable: bool = True
    interest: int = 1
    on_orphaned: Callable[[], object] | None = None


class ResponseCache:
    """Byte-bounded, single-flight cache with per-entry lifetimes.

    Size limits default to `SEARCH_RESPONSE_CACHE_*` settings. `ttl_s` defaults
    to the unknown-generation lifetime and applies when `publish()` omits
    `max_age_s`. The search cache passes its generation lifetime; the LLM memo
    supplies its own `ttl_s` and does not use `max_age_s`.
    """

    def __init__(
        self,
        *,
        max_bytes: int | None = None,
        max_entry_bytes: int | None = None,
        ttl_s: float | None = None,
    ) -> None:
        self._max_bytes = (
            config.settings().response_cache_max_bytes if max_bytes is None else max_bytes
        )
        self._max_entry_bytes = (
            config.settings().response_cache_max_entry_bytes
            if max_entry_bytes is None
            else max_entry_bytes
        )
        self._ttl_s = (
            generation_config.GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS
            if ttl_s is None
            else ttl_s
        )
        self._lock = Lock()
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._inflight: dict[str, _Waiter] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversized = 0

    async def get(self, key: str) -> bytes | None:
        """Return a live value or `None`; a hit becomes most-recently-used."""
        async with self._lock:
            value = self._get_locked(key)
            if value is None:
                self._misses += 1
            else:
                self._hits += 1
            return value

    async def acquire(self, key: str) -> tuple[bytes | None, Lease | None]:
        """Return a live hit, or a leader/follower lease for the miss."""
        async with self._lock:
            value = self._get_locked(key)
            if value is not None:
                self._hits += 1
                return value, None

            self._misses += 1
            waiter = self._inflight.get(key)
            if waiter is not None:
                waiter.interest += 1
                return None, Lease(key=key, is_leader=False, _waiter=waiter)
            waiter = _Waiter()
            self._inflight[key] = waiter
            return None, Lease(key=key, is_leader=True, _waiter=waiter)

    def _get_locked(self, key: str) -> bytes | None:
        """Return a live value and update LRU order; drop expired entries.

        Callers update hit and miss counters themselves.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._drop_locked(key, entry)
            return None
        self._store.move_to_end(key)
        return entry.value

    async def await_leader(self, lease: Lease) -> tuple[bytes, bool, bool] | None:
        """Wait for a follower's leader; return `(value, timed_out, storable)` or `None` on abandon.

        `storable` is the leader's verdict, even when this store rejects an
        oversized value. Calling this with a leader lease raises `ValueError`.
        The wait is unbounded; callers enforce their own deadlines.
        """
        if lease.is_leader:
            raise ValueError(
                "await_leader() called on a leader lease; the leader runs the "
                "fetch itself and calls publish() or abandon(), never await_leader()"
            )
        assert lease._waiter is not None  # every lease acquire() returns carries one
        await lease._waiter.event.wait()
        value = lease._waiter.value
        if value is None:
            return None
        return value, lease._waiter.timed_out, lease._waiter.storable

    def on_orphaned(self, lease: Lease, callback: Callable[[], object]) -> None:
        """Register the leader's orphan callback; non-leader leases raise `ValueError`.

        Synchronous registration lets the leader install it before any caller can leave.
        """
        if not lease.is_leader:
            raise ValueError("on_orphaned() called on a non-leader lease")
        assert lease._waiter is not None  # every lease acquire() returns carries one
        lease._waiter.on_orphaned = callback

    def leave(self, lease: Lease) -> None:
        """Release this lease once. The last caller detaches an unsettled flight,
        wakes followers with `None`, and runs the orphan callback. A settled
        flight only loses interest.

        Synchronous and lock-free on the event-loop thread, so it is safe in a
        cancellation `finally` block.
        """
        if lease._left:
            return
        lease._left = True
        waiter = lease._waiter
        assert waiter is not None  # every lease acquire() returns carries one
        waiter.interest -= 1
        if waiter.interest > 0 or self._inflight.get(lease.key) is not waiter:
            return
        del self._inflight[lease.key]
        waiter.value = None
        waiter.event.set()
        if waiter.on_orphaned is not None:
            waiter.on_orphaned()

    async def publish(
        self,
        lease: Lease,
        value: bytes,
        size: int,
        *,
        also: Sequence[str] = (),
        storable: bool = True,
        timed_out: bool = False,
        max_age_s: float | None = None,
    ) -> None:
        """Wake followers and optionally store `value` under this and `also` keys.

        A detached lease may still store its correct result, but wakes no
        waiters; any in-flight entry then belongs to a newer leader. `max_age_s`
        sets the lifetime for all keys, while `None` uses the cache's `ttl_s`.

        `storable=False` still publishes the value to followers. They also
        receive the leader's `timed_out` and `storable` verdicts, which the
        response bytes cannot show. Oversized storable values are not stored
        and increment `oversized`; non-storable values do not.

        Each `also` key is charged `size` independently, though both keys share
        the same bytes object. Two keys can therefore overstate retained bytes
        by up to 2x, avoiding shared-value eviction bookkeeping. Only a leader
        lease may publish; others raise `ValueError`.
        """
        if not lease.is_leader:
            raise ValueError(
                "publish() called on a non-leader lease; only the leader that "
                "owns the fetch may publish its result"
            )
        now = time.monotonic()
        async with self._lock:
            if storable:
                if size <= self._max_entry_bytes:
                    for k in (lease.key, *also):
                        self._set_locked(k, value, size, now, max_age_s)
                else:
                    self._oversized += 1
            waiter = lease._waiter
            if waiter is not None and self._inflight.get(lease.key) is waiter:
                del self._inflight[lease.key]
                waiter.value = value
                waiter.timed_out = timed_out
                waiter.storable = storable
                waiter.event.set()

    async def abandon(self, lease: Lease) -> None:
        """Wake followers with `None` and end this flight without storing.

        A detached lease leaves any newer leader's entry untouched.
        """
        if not lease.is_leader:
            raise ValueError(
                "abandon() called on a non-leader lease; only the leader that "
                "owns the fetch may abandon it"
            )
        async with self._lock:
            waiter = lease._waiter
            if waiter is not None and self._inflight.get(lease.key) is waiter:
                del self._inflight[lease.key]
                waiter.value = None
                waiter.event.set()

    def stats(self) -> ResponseCacheStats:
        return ResponseCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            entries=len(self._store),
            total_bytes=self._bytes,
            oversized=self._oversized,
        )

    def clear_sync(self) -> None:
        """Clear stored entries and byte accounting; in-flight work is untouched.

        Run on the store's event-loop thread. A background refresh thread must
        schedule this callback with `loop.call_soon_threadsafe()`; no lock is
        needed because the callback runs to completion without coroutine interleaving.
        """
        self._store.clear()
        self._bytes = 0

    def _drop_locked(self, key: str, entry: _Entry) -> None:
        del self._store[key]
        self._bytes -= entry.size

    def _set_locked(
        self, key: str, value: bytes, size: int, now: float, max_age_s: float | None
    ) -> None:
        existing = self._store.pop(key, None)
        if existing is not None:
            self._bytes -= existing.size
        lifetime = self._ttl_s if max_age_s is None else max_age_s
        self._store[key] = _Entry(value=value, size=size, expires_at=now + lifetime)
        self._bytes += size
        while self._bytes > self._max_bytes and self._store:
            _evicted_key, evicted = self._store.popitem(last=False)
            self._bytes -= evicted.size
            self._evictions += 1
