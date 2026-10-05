"""Live generation provider backed by an injected source.

`stamp()` reads an immutable snapshot without I/O and starts at most one refresh thread per
60-second poll interval. With a catalogue, each stamp also carries the build id of the catalogue
the calling request is bound to, and a change of the loaded build clears every registered cache.
Polling is request-triggered because `cpu_idle=true` starves idle loops. Requests can await `refresh_if_stale`. The default wait is 1 second.
A token becomes unknown after 300 seconds without a successful read. `UNKNOWN_GENERATION` is the
no-op default for unconfigured caches."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx

from graph_generation.protocol import CatalogueBuilds, GenerationProvider
from graph_generation.registry import CacheRegistry, RegistrationHandle
from graph_generation.source import GenerationReadResult, GenerationSource
from graph_generation.stamp import GenerationStamp

logger = logging.getLogger(__name__)

# Maximum age of a successfully read token before `stamp()` returns unknown.
STALE_AFTER_SECONDS = 300.0

POLL_INTERVAL_SECONDS = 60.0

_STATS_LOG_INTERVAL_SECONDS = 900.0

# Backs off repeat read-failure WARNINGs to the first failure, then roughly
# once every this-many consecutive failures.
_FAILURE_LOG_EVERY_N = 10


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """Lock-free snapshot of the last successful token and its read time.

    Both values start unknown."""

    token: str | None
    read_at: float | None


class GraphGeneration:
    def __init__(
        self,
        *,
        source: GenerationSource | None,
        catalogue: CatalogueBuilds | None = None,
        expected_graph_base: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        stale_after: float = STALE_AFTER_SECONDS,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        """Without a source, tokens stay unknown and refreshes stop. The catalogue's build id
        still joins every stamp. Cache registration continues for stats."""
        self._source = source
        self._catalogue = catalogue
        self._expected_graph_base = expected_graph_base
        self._clock = clock
        self._stale_after = stale_after
        self._poll_interval = poll_interval
        self._registry = CacheRegistry()
        # Protect refresh bookkeeping. Release before network reads and callbacks.
        self._refresh_lock = threading.Lock()
        self._in_flight: concurrent.futures.Future[None] | None = None
        self._last_attempt_at: float | None = None
        # `stamp()` reads or replaces this snapshot in one assignment under the GIL.
        self._snapshot: _Snapshot = _Snapshot(token=None, read_at=None)
        # Preserve the last successful token to distinguish change from same-token recovery.
        self._last_known_token: str | None = None
        self._consecutive_failures = 0
        self._last_stats_log_at: float | None = None
        # Tracks staleness transitions for logs. A race can duplicate one warning.
        self._displayed_known = False
        if catalogue is not None:
            catalogue.on_swap(self._on_catalogue_swap)

    def stamp(self) -> GenerationStamp:
        snapshot = self._snapshot
        now = self._clock()
        known = snapshot.read_at is not None and (now - snapshot.read_at) < self._stale_after
        if known:
            self._displayed_known = True
        elif self._displayed_known:
            self._displayed_known = False
            logger.warning(
                "graph generation: known -> unknown (last token %r, stale after %.0fs "
                "with no successful read)",
                snapshot.token,
                self._stale_after,
            )
        self._maybe_start_refresh(now=now, force=False)
        return GenerationStamp(
            token=snapshot.token if known else None,
            catalogue=None if self._catalogue is None else self._catalogue.request_build_id(),
        )

    async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
        now = self._clock()
        snapshot = self._snapshot
        is_stale = snapshot.read_at is None or (now - snapshot.read_at) >= self._stale_after
        if not is_stale:
            return
        poll_due = (
            self._last_attempt_at is None or (now - self._last_attempt_at) >= self._poll_interval
        )
        # Join an in-flight read. Start a new read when the next poll is due.
        future = self._maybe_start_refresh(now=now, force=poll_due)
        if future is None:
            return
        # Bound this caller's wait without cancelling the refresh future, which keeps running.
        with contextlib.suppress(TimeoutError):
            await asyncio.to_thread(future.result, timeout_s)

    def register(
        self,
        name: str,
        *,
        stats: Callable[[], Mapping[str, int]],
        clear: Callable[[], None],
    ) -> RegistrationHandle:
        handle = self._registry.register(name, stats=stats, clear=clear)
        # Report current state without starting a refresh or staleness warning.
        snapshot = self._snapshot
        now = self._clock()
        known = snapshot.read_at is not None and (now - snapshot.read_at) < self._stale_after
        if known:
            logger.info("graph generation: registered cache %r (token=%r)", name, snapshot.token)
        else:
            logger.info("graph generation: registered cache %r (token unknown)", name)
        return handle

    def registered_names(self) -> tuple[str, ...]:
        return self._registry.registered_names()

    def refresh_now(self) -> None:
        """Read and publish synchronously for the initial startup token.

        Return immediately when no source is configured."""
        if self._source is None:
            return
        with self._refresh_lock:
            self._last_attempt_at = self._clock()
        self._run_read()

    def _maybe_start_refresh(
        self, *, now: float, force: bool
    ) -> concurrent.futures.Future[None] | None:
        if self._source is None:
            return None
        with self._refresh_lock:
            if self._in_flight is not None:
                return self._in_flight
            due = (
                force
                or self._last_attempt_at is None
                or (now - self._last_attempt_at) >= self._poll_interval
            )
            if not due:
                return None
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            self._in_flight = future
            self._last_attempt_at = now
        threading.Thread(
            target=self._run_refresh_thread,
            args=(future,),
            daemon=True,
            name="graph-generation-refresh",
        ).start()
        return future

    def _on_catalogue_swap(self) -> None:
        """Entries under the old build are unreachable from requests bound to the new one, so
        this clears them instead of waiting for their lifetime to end."""
        logger.info("graph generation: origin catalogue build changed; clearing every cache")
        self._registry.clear_all()

    def _run_refresh_thread(self, future: concurrent.futures.Future[None]) -> None:
        try:
            self._run_read()
        finally:
            with self._refresh_lock:
                self._in_flight = None
            future.set_result(None)

    def _run_read(self) -> None:
        assert self._source is not None
        try:
            result = self._source.read()
        except Exception as exc:
            # Contain source, authentication, and network errors on startup and refresh threads.
            self._on_read_failure(exc)
            return
        self._on_read_success(result)

    def _on_read_success(self, result: GenerationReadResult) -> None:
        now = self._clock()
        previous_known_token = self._last_known_token
        changed = previous_known_token is not None and previous_known_token != result.token
        # Track display-state recovery separately from changes to the token itself.
        recovered_from_unknown = not self._displayed_known
        # Atomic publication gives lock-free readers a complete old or new snapshot.
        self._snapshot = _Snapshot(token=result.token, read_at=now)
        self._last_known_token = result.token
        self._consecutive_failures = 0
        if changed:
            logger.info(
                "graph generation: token change (%r -> %r); clearing every registered cache",
                previous_known_token,
                result.token,
            )
            self._registry.clear_all()
        elif recovered_from_unknown:
            logger.info("graph generation: unknown -> known (token=%r)", result.token)
        self._log_urls_and_check_host(result)
        self._maybe_log_stats(now, force=(changed or recovered_from_unknown))

    def _on_read_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        if (
            self._consecutive_failures == 1
            or self._consecutive_failures % _FAILURE_LOG_EVERY_N == 0
        ):
            logger.warning(
                "graph generation: read failed (%d consecutive failure(s)): %s",
                self._consecutive_failures,
                exc,
            )

    def _log_urls_and_check_host(self, result: GenerationReadResult) -> None:
        if not result.urls:
            return
        logger.info("graph generation: service urls %s", result.urls)
        if self._expected_graph_base is None:
            return
        expected_host = httpx.URL(self._expected_graph_base).host
        hosts = {httpx.URL(url).host for url in result.urls}
        if expected_host not in hosts:
            logger.warning(
                "graph generation: expected graph base %r (host %r) is not among this "
                "service's own urls %s -- DFO_GRAPH_GENERATION_SERVICE may name the "
                "wrong service, which would give a known token that never moves",
                self._expected_graph_base,
                expected_host,
                result.urls,
            )

    def _maybe_log_stats(self, now: float, *, force: bool) -> None:
        if (
            not force
            and self._last_stats_log_at is not None
            and (now - self._last_stats_log_at) < _STATS_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_stats_log_at = now
        logger.info("graph generation: registered cache stats %s", self._registry.stats_summary())


class _NoopRegistration:
    def close(self) -> None:
        return None


class _UnknownGeneration:
    """Always-unknown provider with no-op registration and refresh.

    Its cache TTL is 3,600 seconds by default."""

    def stamp(self) -> GenerationStamp:
        return GenerationStamp(token=None)

    def register(
        self,
        name: str,
        *,
        stats: Callable[[], Mapping[str, int]],
        clear: Callable[[], None],
    ) -> RegistrationHandle:
        return _NoopRegistration()

    async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
        return None


UNKNOWN_GENERATION: GenerationProvider = _UnknownGeneration()
