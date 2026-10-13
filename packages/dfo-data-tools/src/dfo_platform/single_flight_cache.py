"""Bounded, generation-keyed cache that coalesces concurrent fetches per key.

Each call captures one generation stamp before lookup and keys entries by `(stamp.key, key)`.
The default TTL is `stamp.max_age_s`. `maxsize` bounds the LRU store. One lock protects O(1)
bookkeeping and is released before fetches or awaits.

A miss starts a flight, which belongs to no caller. Its deadline is `flight_timeout_s` from the
moment it starts, queue time included. Its fetch receives that deadline as an argument and runs
in `detached_context(deadline=...)`, so the deadline also bounds every read that narrows through
`bounded_deadline`, and no caller's request deadline reaches it.

The flight deadline bounds every wait on the flight, whether or not its fetch ever got a worker.
At that deadline the flight expires: its unresolved waiters get `TimeoutError` with the
flight-expired message, its keys leave the in-flight registry, and a job still queued is
cancelled. A caller that finds an expired flight in the registry starts a new one. A fetch that
starts after the deadline fails without fetching, and one that ends after it publishes nothing,
because its key may already belong to a newer flight whose value must not be overwritten.

Sync flights run on the `executor` the cache is given, which must be a pool no caller of the
cache runs on. The caller that starts a flight waits for it as every joining caller does, until
the earlier of the flight deadline and its own deadline narrowed by its bound request deadline
(`bounded_deadline`). A wait that reaches the caller's own deadline first raises `WaitTimeout`,
a `TimeoutError`, while the flight runs on and publishes. Expiry needs no timer: the first waiter,
joining caller, or job completion to see the deadline pass expires the flight, once, under the
lock. Every end of a flight resolves its waiters: a value, a fetch error, a submission the
executor refuses, a queued job a shutdown cancels, or the flight deadline.

`aget_or_fetch_many` claims and starts sync flights as `get_or_fetch_many` does, sharing them
with sync callers, and awaits each on the event loop: a slot's resolution wakes the loop
(`_SyncSlot.wakers`), so the wait holds no thread.

Async callers of `aget_or_fetch` share a cache-owned task created in the flight's context,
which runs the fetch on the cache's `executor`, or the event loop's default executor when it has none, under
`asyncio.timeout` set to the flight deadline. At that deadline the task ends with the
flight-expired `TimeoutError` and leaves the registry, which cancels a job still queued; a thread
that already started finishes on its own. Each caller awaits the task through `asyncio.shield` until its own deadline,
so a timeout or cancellation detaches only that caller.

Failures reach every caller waiting and are never stored. The sync flights and the tasks of
`aget_or_fetch` are registered apart and never join one another. Deadlines use real
`time.monotonic()`. The injectable `clock` controls TTL only.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import CancelledError, Executor, Future
from dataclasses import dataclass, field

from dfo_platform.executor import run_in_executor_with_context
from dfo_platform.request_deadline import bounded_deadline, detached_context
from graph_generation import UNKNOWN_GENERATION, GenerationProvider, GenerationStamp

_DEFAULT_TIMEOUT_MESSAGE = (
    "single-flight cache: deadline elapsed while the leader was still running"
)
_FLIGHT_EXPIRED_MESSAGE = "single-flight cache: the flight's deadline passed before it ended"


def _flight_expired() -> TimeoutError:
    return TimeoutError(_FLIGHT_EXPIRED_MESSAGE)


def _raise_if_flight_expired(flight_deadline: float) -> None:
    if time.monotonic() >= flight_deadline:
        raise _flight_expired()


def _default_ttl_for_stamp(stamp: GenerationStamp) -> float:
    return stamp.max_age_s


class AlreadyInFlight(Exception):
    """Raised when `join_inflight=False` and another caller owns the key."""


class WaitTimeout(TimeoutError):
    """A sync caller's wait reached its own deadline before the flight ended. The flight runs on.
    A `TimeoutError` the flight itself raises, including its expiry, reaches its waiters as that
    error, so an owner can tell the two apart."""


@dataclass(eq=False)
class _SyncFlight[K, V]:
    """One sync flight: its deadline, the TTL its values store under, which values it stores
    (`is_cacheable`, every value when `None`), the slots it owns, and its executor job once
    submitted. A flight is filled under the lock by the caller that starts it and never gains
    slots afterwards."""

    deadline: float
    ttl: float
    is_cacheable: Callable[[V], bool] | None = None
    owned: dict[K, tuple[tuple[str, K], _SyncSlot[K, V]]] = field(default_factory=dict)
    job: Future[dict[K, V]] | None = None


@dataclass(eq=False)
class _SyncSlot[K, V]:
    """One in-flight key's event and result.
    `ready` is set, under the lock, once the slot is resolved with a value or error; a resolved
    slot is never written again. `_MISS` distinguishes a stored `None` from an unset slot.
    `wakers` are the async waiters' callbacks, called under the lock as the slot resolves.
    """

    flight: _SyncFlight[K, V]
    ready: threading.Event = field(default_factory=threading.Event)
    value: V | _Miss = field(default_factory=lambda: _MISS)
    error: BaseException | None = None
    wakers: list[Callable[[], None]] = field(default_factory=list)


def _cancel_job[K, V](flight: _SyncFlight[K, V]) -> None:
    """Cancel an expired flight's job if it is still queued. Called without the lock, since a
    successful cancel runs the job's done callback, which takes it. A job not yet recorded is
    left to `_raise_if_flight_expired`, which stops it before it fetches."""
    if flight.job is not None:
        flight.job.cancel()


def _retrieve_exception[V](task: asyncio.Task[V]) -> None:
    """Consume a failed task's exception to suppress asyncio's warning if all callers detach.

    Later awaits raise the exception normally. Cancelled tasks carry no unhandled exception warning.
    """
    if not task.cancelled():
        task.exception()


class SingleFlightCache[K, V]:
    """Bounded per-key cache. `maxsize=0` calls `fetch` directly on the caller's thread, without
    lookup or coalescing, under the caller's deadline or else `flight_timeout_s` from now.

    `flight_timeout_s` sets each flight's deadline, which bounds every wait on the flight.
    `executor` runs sync flights and is required by `get_or_fetch` and `get_or_fetch_many` on an
    enabled cache. Async flights run on `executor` when it is given, else on the event loop's
    default executor.
    """

    def __init__(
        self,
        *,
        maxsize: int,
        flight_timeout_s: float,
        executor: Executor | None = None,
        generation: GenerationProvider = UNKNOWN_GENERATION,
        clock: Callable[[], float] = time.monotonic,
        ttl_for_stamp: Callable[[GenerationStamp], float] = _default_ttl_for_stamp,
        timeout_message: str = _DEFAULT_TIMEOUT_MESSAGE,
    ) -> None:
        if flight_timeout_s <= 0:
            raise ValueError(f"flight_timeout_s must be positive, got {flight_timeout_s!r}")
        self._maxsize = maxsize
        self._flight_timeout_s = flight_timeout_s
        self._executor = executor
        self._generation = generation
        self._clock = clock
        self._ttl_for_stamp = ttl_for_stamp
        self._timeout_message = timeout_message
        self._lock = threading.Lock()
        self._store: OrderedDict[tuple[str, K], tuple[float, V]] = OrderedDict()
        self._sync_inflight: dict[tuple[str, K], _SyncSlot[K, V]] = {}
        # Each async flight's task and deadline, so a caller never joins an expired flight.
        self._async_inflight: dict[tuple[str, K], tuple[asyncio.Task[V], float]] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        return self._maxsize > 0

    def _direct_deadline(self, deadline: float | None) -> float:
        """The deadline a disabled cache passes to `fetch`: the caller's, else a flight's."""
        bound = bounded_deadline(deadline)
        return bound if bound is not None else time.monotonic() + self._flight_timeout_s

    def _new_flight_deadline(self) -> float:
        return time.monotonic() + self._flight_timeout_s

    # -------------------------------------------------------------------
    # Sync entry points
    # -------------------------------------------------------------------

    def get_or_fetch(
        self,
        key: K,
        fetch: Callable[[float], V],
        *,
        deadline: float | None = None,
        join_inflight: bool = True,
    ) -> V:
        """Return a cached value or fetch it once in a flight. `fetch` receives the flight's
        deadline. Every caller, starter or joiner alike, waits until the earlier of the flight
        deadline, which raises the flight's `TimeoutError`, and `deadline`, which raises
        `WaitTimeout` without stopping the flight. With `join_inflight=False`, a live hit returns
        and an unowned miss starts a flight. An owned miss raises `AlreadyInFlight` immediately.
        """
        if not self.enabled:
            return fetch(self._direct_deadline(deadline))

        stamp = self._generation.stamp()
        full_key = (stamp.key, key)
        is_owner, slot, cached = self._claim_sync(key, full_key, ttl=self._ttl_for_stamp(stamp))
        if not isinstance(cached, _Miss):
            return cached
        assert slot is not None  # every non-hit path claims or joins a slot

        if is_owner:
            self._start_sync_flight(
                slot.flight, lambda _keys, flight_deadline: {key: fetch(flight_deadline)}
            )
        elif not join_inflight:
            raise AlreadyInFlight(key)
        return self._slot_result(slot, deadline=deadline)

    def get_or_fetch_many(
        self,
        keys: Sequence[K],
        fetch: Callable[[list[K], float], dict[K, V]],
        *,
        deadline: float | None = None,
        copy_result: Callable[[V], V] | None = None,
        is_cacheable: Callable[[V], bool] | None = None,
        flight_per_key: bool = False,
        on_error: Callable[[K, Exception], V] | None = None,
    ) -> dict[K, V]:
        """Resolve unique keys in input order. The misses no live flight holds start one flight,
        or with `flight_per_key` one flight each, whose `fetch` receives its keys and its
        deadline. Fetch failures reach every caller waiting on the flight. The cache stores
        successful fetches only. A fetched value `is_cacheable` rejects reaches the flight's
        callers and leaves the store unchanged, while the flight's other values store. Each wait
        ends at the earlier of its flight's deadline and `deadline`, as in `get_or_fetch`. The
        first key whose flight fails or whose wait ends raises that error, unless `on_error` is
        given: then such a key resolves to `on_error(key, error)`, which is not stored, and the
        other keys resolve on their own. `copy_result`, when given, copies hits and fetched
        values.
        """
        unique_keys = list(dict.fromkeys(keys))
        if not self.enabled:
            return fetch(unique_keys, self._direct_deadline(deadline))

        hits, waiting = self._claim_many(
            unique_keys, fetch, is_cacheable=is_cacheable, flight_per_key=flight_per_key
        )
        for key, slot in waiting:
            try:
                hits[key] = self._slot_result(slot, deadline=deadline)
            except Exception as exc:
                if on_error is None:
                    raise
                hits[key] = on_error(key, exc)
        return _ordered(hits, unique_keys, copy_result)

    async def aget_or_fetch_many(
        self,
        keys: Sequence[K],
        fetch: Callable[[list[K], float], dict[K, V]],
        *,
        deadline: float | None = None,
        copy_result: Callable[[V], V] | None = None,
        is_cacheable: Callable[[V], bool] | None = None,
        flight_per_key: bool = False,
        on_error: Callable[[K, Exception], V] | None = None,
    ) -> dict[K, V]:
        """`get_or_fetch_many` awaited on the event loop. It joins and starts the same sync
        flights, on the cache's executor, and each wait holds no thread. A disabled cache runs
        `fetch` in `asyncio.to_thread`."""
        unique_keys = list(dict.fromkeys(keys))
        if not self.enabled:
            return await asyncio.to_thread(fetch, unique_keys, self._direct_deadline(deadline))

        hits, waiting = self._claim_many(
            unique_keys, fetch, is_cacheable=is_cacheable, flight_per_key=flight_per_key
        )
        for key, slot in waiting:
            try:
                hits[key] = await self._aslot_result(slot, deadline=deadline)
            except Exception as exc:
                if on_error is None:
                    raise
                hits[key] = on_error(key, exc)
        return _ordered(hits, unique_keys, copy_result)

    def _claim_many(
        self,
        unique_keys: list[K],
        fetch: Callable[[list[K], float], dict[K, V]],
        *,
        is_cacheable: Callable[[V], bool] | None,
        flight_per_key: bool,
    ) -> tuple[dict[K, V], list[tuple[K, _SyncSlot[K, V]]]]:
        """The hits among `unique_keys`, and the slot each miss waits on: a live flight's, or
        one of the new flights over the misses no live flight holds, started here, one over them
        all or, with `flight_per_key`, one each."""
        stamp = self._generation.stamp()
        ttl = self._ttl_for_stamp(stamp)

        def new_flight() -> _SyncFlight[K, V]:
            return _SyncFlight(
                deadline=self._new_flight_deadline(), ttl=ttl, is_cacheable=is_cacheable
            )

        flights: list[_SyncFlight[K, V]] = []
        hits: dict[K, V] = {}
        waiting: list[tuple[K, _SyncSlot[K, V]]] = []
        expired: list[_SyncFlight[K, V]] = []
        with self._lock:
            now = self._clock()
            for key in unique_keys:
                full_key = (stamp.key, key)
                value = self._lookup_locked(full_key, now=now)
                if not isinstance(value, _Miss):
                    hits[key] = value
                    continue
                slot = self._live_slot_locked(full_key, expired=expired)
                if slot is None:
                    if flight_per_key or not flights:
                        flights.append(new_flight())
                    slot = self._own_slot_locked(flights[-1], key, full_key)
                waiting.append((key, slot))
        for stale in expired:
            _cancel_job(stale)

        for flight in flights:
            self._start_sync_flight(flight, fetch)
        return hits, waiting

    def cached_many(self, keys: Sequence[K]) -> dict[K, V]:
        """The live stored values of `keys`, in input order, without fetching or joining a
        flight. Each value found counts a hit. A key not found counts nothing, since a caller
        that wants it fetches it next and counts the miss there."""
        if not self.enabled:
            return {}
        stamp = self._generation.stamp()
        found: dict[K, V] = {}
        with self._lock:
            now = self._clock()
            for key in dict.fromkeys(keys):
                full_key = (stamp.key, key)
                entry = self._store.get(full_key)
                if entry is None or entry[0] <= now:
                    continue
                self._store.move_to_end(full_key)
                self._hits += 1
                found[key] = entry[1]
        return found

    def _claim_sync(
        self, key: K, full_key: tuple[str, K], *, ttl: float
    ) -> tuple[bool, _SyncSlot[K, V] | None, V | _Miss]:
        """Under the lock, return `(False, None, value)` for a hit, `(False, slot, _MISS)` to join
        a live flight, or `(True, slot, _MISS)` for the owner of a new flight. An expired flight
        found here is expired and replaced. `_MISS` distinguishes cached `None`.
        """
        expired: list[_SyncFlight[K, V]] = []
        with self._lock:
            now = self._clock()
            value = self._lookup_locked(full_key, now=now)
            if not isinstance(value, _Miss):
                return False, None, value
            slot = self._live_slot_locked(full_key, expired=expired)
            is_owner = slot is None
            if slot is None:
                flight: _SyncFlight[K, V] = _SyncFlight(
                    deadline=self._new_flight_deadline(), ttl=ttl
                )
                slot = self._own_slot_locked(flight, key, full_key)
        for stale in expired:
            _cancel_job(stale)
        return is_owner, slot, _MISS

    def _live_slot_locked(
        self, full_key: tuple[str, K], *, expired: list[_SyncFlight[K, V]]
    ) -> _SyncSlot[K, V] | None:
        """The registry's slot for `full_key` if its flight is still live. A flight past its
        deadline is expired here and appended to `expired`, whose jobs the caller cancels once it
        releases the lock."""
        slot = self._sync_inflight.get(full_key)
        if slot is None:
            return None
        if time.monotonic() < slot.flight.deadline:
            return slot
        self._resolve_sync_flight_locked(slot.flight, {}, _flight_expired())
        expired.append(slot.flight)
        return None

    def _own_slot_locked(
        self, flight: _SyncFlight[K, V], key: K, full_key: tuple[str, K]
    ) -> _SyncSlot[K, V]:
        slot = _SyncSlot(flight=flight)
        flight.owned[key] = (full_key, slot)
        self._sync_inflight[full_key] = slot
        return slot

    def _start_sync_flight(
        self,
        flight: _SyncFlight[K, V],
        fetch: Callable[[list[K], float], dict[K, V]],
    ) -> None:
        """Submit `flight` to the executor, in a context that binds its deadline. The job's done
        callback resolves every slot still unresolved, whatever ended the job. The caller holds
        no lock, so an executor that runs the job inline cannot deadlock on it."""
        keys = list(flight.owned)
        context = detached_context(deadline=flight.deadline)

        def fetch_unless_expired() -> dict[K, V]:
            _raise_if_flight_expired(flight.deadline)
            return fetch(keys, flight.deadline)

        def run() -> dict[K, V]:
            return context.run(fetch_unless_expired)

        try:
            if self._executor is None:
                raise RuntimeError("SingleFlightCache: sync flights need an executor")
            job = self._executor.submit(run)
        except BaseException as exc:
            self._resolve_sync_flight(flight, {}, exc)
            return
        flight.job = job
        job.add_done_callback(lambda done: self._settle_sync_flight(flight, done))

    def _settle_sync_flight(self, flight: _SyncFlight[K, V], done: Future[dict[K, V]]) -> None:
        """Resolve the flight's unresolved slots from `done`. A failure while settling resolves
        the slots still unresolved with that error, because a done callback that raises is only
        logged and would strand the waiters."""
        try:
            fetched, error = self._job_outcome(flight, done)
            self._resolve_sync_flight(flight, fetched, error)
        except BaseException as exc:
            self._resolve_sync_flight(flight, {}, exc)

    @staticmethod
    def _job_outcome(
        flight: _SyncFlight[K, V], done: Future[dict[K, V]]
    ) -> tuple[Mapping[K, V], BaseException | None]:
        """What `done` resolves the flight's slots with. A job that ends at or after the flight
        deadline expires the flight, whatever its own outcome."""
        if time.monotonic() >= flight.deadline:
            return {}, _flight_expired()
        if done.cancelled():
            return {}, CancelledError("single-flight cache: flight cancelled")
        error = done.exception()
        fetched = done.result() if error is None else {}
        missing = [key for key in flight.owned if key not in fetched]
        if error is None and missing:
            error = KeyError(f"single-flight cache: fetch returned no value for {missing!r}")
        return fetched, error

    def _expire_sync_flight(self, flight: _SyncFlight[K, V]) -> None:
        with self._lock:
            self._resolve_sync_flight_locked(flight, {}, _flight_expired())
        _cancel_job(flight)

    def _resolve_sync_flight(
        self,
        flight: _SyncFlight[K, V],
        fetched: Mapping[K, V],
        error: BaseException | None,
    ) -> None:
        with self._lock:
            self._resolve_sync_flight_locked(flight, fetched, error)

    def _resolve_sync_flight_locked(
        self,
        flight: _SyncFlight[K, V],
        fetched: Mapping[K, V],
        error: BaseException | None,
    ) -> None:
        """Publish `fetched` or record `error` in every unresolved slot of `flight`, then wake its
        waiters. A resolved slot is skipped, so whichever end of the flight comes first wins and
        a later one changes nothing, and a value publishes only with its slot. Only a registry
        entry that is still this slot is removed, because `clear()` or a newer flight can have
        replaced it."""
        now = self._clock()
        for key, (full_key, slot) in flight.owned.items():
            if slot.ready.is_set():
                continue
            if self._sync_inflight.get(full_key) is slot:
                del self._sync_inflight[full_key]
            if error is None:
                slot.value = fetched[key]
                if flight.is_cacheable is None or flight.is_cacheable(fetched[key]):
                    self._publish_locked(full_key, fetched[key], ttl=flight.ttl, now=now)
            else:
                slot.error = error
            slot.ready.set()
            for wake in slot.wakers:
                wake()
            slot.wakers.clear()

    def _slot_result(self, slot: _SyncSlot[K, V], *, deadline: float | None) -> V:
        """Wait for `slot` until the earlier of its flight's deadline and `deadline` narrowed by
        the bound request deadline, then return its value or raise its error. A wait that reaches
        the caller's own deadline first raises `WaitTimeout`. One that reaches the flight deadline
        expires the flight and raises its `TimeoutError`."""
        bound, wait_until = _wait_bounds(slot, deadline)
        ready = slot.ready.wait(timeout=max(0.0, wait_until - time.monotonic()))
        return self._slot_outcome(slot, ready=ready, bound=bound)

    async def _aslot_result(self, slot: _SyncSlot[K, V], *, deadline: float | None) -> V:
        """`_slot_result` awaited on the running loop, which the slot's resolution wakes."""
        bound, wait_until = _wait_bounds(slot, deadline)
        loop = asyncio.get_running_loop()
        woken: asyncio.Future[None] = loop.create_future()

        def wake() -> None:
            # A loop closed while its waiter detached has no one left to wake.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(_set_done, woken)

        with self._lock:
            ready = slot.ready.is_set()
            if not ready:
                slot.wakers.append(wake)
        if not ready:
            try:
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(max(0.0, wait_until - time.monotonic())):
                        await woken
            finally:
                with self._lock:
                    if wake in slot.wakers:
                        slot.wakers.remove(wake)
            ready = slot.ready.is_set()
        return self._slot_outcome(slot, ready=ready, bound=bound)

    def _slot_outcome(self, slot: _SyncSlot[K, V], *, ready: bool, bound: float | None) -> V:
        """`slot`'s value or error once a wait on it ends, `ready` or not by its end."""
        if not ready:
            if bound is not None and bound < slot.flight.deadline:
                raise WaitTimeout(self._timeout_message)
            self._expire_sync_flight(slot.flight)
        if slot.error is not None:
            raise slot.error
        assert not isinstance(slot.value, _Miss)  # a slot with no error always has a value
        return slot.value

    # -------------------------------------------------------------------
    # Async entry point
    # -------------------------------------------------------------------

    async def aget_or_fetch(
        self,
        key: K,
        fetch: Callable[[float], V],
        *,
        deadline: float,
        is_cacheable: Callable[[V], bool] | None = None,
    ) -> V:
        """Resolve on the event loop, running blocking `fetch` in a worker thread with the
        flight's deadline. Callers share one shielded task, which ends with `TimeoutError` at the
        flight deadline, so their cancellation or deadline detaches only their own await. A value
        rejected by `is_cacheable` reaches current callers and leaves the store unchanged.
        """
        if not self.enabled:
            return await self._await_bounded(
                self._off_loop(fetch, self._direct_deadline(deadline)), deadline=deadline
            )

        stamp = self._generation.stamp()
        full_key = (stamp.key, key)
        ttl = self._ttl_for_stamp(stamp)
        task, cached = self._claim_task(full_key, fetch, ttl=ttl, is_cacheable=is_cacheable)
        if not isinstance(cached, _Miss):
            return cached
        assert task is not None  # every non-hit path claims or joins a task

        return await self._await_bounded(self._await_shielded(task), deadline=deadline)

    def _off_loop[*Ts](self, fn: Callable[[*Ts], V], /, *args: *Ts) -> Coroutine[object, object, V]:
        """Run `fn` on the cache's executor, or in `asyncio.to_thread` when it has none."""
        if self._executor is None:
            return asyncio.to_thread(fn, *args)
        return run_in_executor_with_context(self._executor, fn, *args)

    @staticmethod
    async def _await_shielded(task: asyncio.Task[V]) -> V:
        return await asyncio.shield(task)

    async def _await_bounded(self, coro: Coroutine[object, object, V], *, deadline: float) -> V:
        """Await `coro` until `deadline` narrowed by the bound request deadline."""
        bound = bounded_deadline(deadline)
        assert bound is not None  # `deadline` is set, so the narrowed bound is too
        remaining = bound - time.monotonic()
        if remaining <= 0:
            coro.close()  # Close the unawaited coroutine to avoid an asyncio warning.
            raise TimeoutError(self._timeout_message)
        return await asyncio.wait_for(coro, timeout=remaining)

    def _claim_task(
        self,
        full_key: tuple[str, K],
        fetch: Callable[[float], V],
        *,
        ttl: float,
        is_cacheable: Callable[[V], bool] | None,
    ) -> tuple[asyncio.Task[V] | None, V | _Miss]:
        """Under the lock, return `(None, value)` for a hit or `(task, _MISS)` for the leader or a
        follower. `_MISS` distinguishes cached `None`. A task past its flight deadline is never
        joined; it is replaced by a new task, which runs in its own flight's context.
        """
        with self._lock:
            now = self._clock()
            value = self._lookup_locked(full_key, now=now)
            if not isinstance(value, _Miss):
                return None, value
            existing = self._async_inflight.get(full_key)
            if existing is not None and time.monotonic() < existing[1]:
                return existing[0], _MISS
            flight_deadline = self._new_flight_deadline()
            task = asyncio.create_task(
                self._run_and_store(
                    full_key, fetch, flight_deadline, ttl=ttl, is_cacheable=is_cacheable
                ),
                context=detached_context(deadline=flight_deadline),
            )
            task.add_done_callback(_retrieve_exception)
            self._async_inflight[full_key] = (task, flight_deadline)
            return task, _MISS

    async def _run_and_store(
        self,
        full_key: tuple[str, K],
        fetch: Callable[[float], V],
        flight_deadline: float,
        *,
        ttl: float,
        is_cacheable: Callable[[V], bool] | None,
    ) -> V:
        """Run `fetch` in a worker thread until the flight deadline. At the deadline the await is
        cancelled, which cancels a job still queued on the executor, and the task raises the
        flight-expired `TimeoutError`; a thread already running finishes on its own."""
        this_task = asyncio.current_task()

        def fetch_unless_expired() -> V:
            _raise_if_flight_expired(flight_deadline)
            return fetch(flight_deadline)

        scope = asyncio.timeout(flight_deadline - time.monotonic())
        try:
            async with scope:
                value = await self._off_loop(fetch_unless_expired)
        except BaseException as exc:
            with self._lock:
                self._forget_task_locked(full_key, this_task)
            if isinstance(exc, TimeoutError) and scope.expired():
                raise _flight_expired() from None
            raise
        with self._lock:
            self._forget_task_locked(full_key, this_task)
            if time.monotonic() >= flight_deadline:  # its key may belong to a newer flight
                raise _flight_expired()
            if is_cacheable is None or is_cacheable(value):
                self._publish_locked(full_key, value, ttl=ttl, now=self._clock())
        return value

    def _forget_task_locked(
        self, full_key: tuple[str, K], task: asyncio.Task[object] | None
    ) -> None:
        entry = self._async_inflight.get(full_key)
        if entry is not None and entry[0] is task:
            del self._async_inflight[full_key]

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
        """Clear entries and in-flight bookkeeping. Active flights continue to completion."""
        with self._lock:
            self._store.clear()
            self._sync_inflight.clear()
            self._async_inflight.clear()


def _ordered[K, V](
    hits: Mapping[K, V], keys: list[K], copy_result: Callable[[V], V] | None
) -> dict[K, V]:
    if copy_result is None:
        return {key: hits[key] for key in keys}
    return {key: copy_result(hits[key]) for key in keys}


def _wait_bounds[K, V](slot: _SyncSlot[K, V], deadline: float | None) -> tuple[float | None, float]:
    """`deadline` narrowed by the bound request deadline, and when a wait on `slot` ends: the
    earlier of that and the flight's deadline."""
    bound = bounded_deadline(deadline)
    flight_deadline = slot.flight.deadline
    return bound, flight_deadline if bound is None else min(bound, flight_deadline)


def _set_done(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class _Miss:
    """The sentinel `_lookup_locked` returns on a miss, distinct from any
    real `V` (including `None`, a legitimate cached value)."""


_MISS = _Miss()
