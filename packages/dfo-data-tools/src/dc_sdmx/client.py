"""HTTP client for the mixer's SDMX endpoints.

OR-lists use commas. Containment filters expand to `containedInPlace+` and
`typeOf` selectors. Values containing `+` or `,` are rejected because the
mixer treats them as filter syntax.
"""

from __future__ import annotations

import io
import json
import math
import socket
import threading
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx
import pyarrow as pa
import pydantic

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

from dc_sdmx.errors import (
    SdmxCapExceededError,
    SdmxContainmentFilterError,
    SdmxHttpError,
    SdmxIncompatibleVariablesError,
    SdmxMissingFilterError,
    SdmxQueryError,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxTransportError,
    SdmxUnexpectedResponseShapeError,
    SdmxUnsupportedFilterError,
)
from dc_sdmx.results import (
    SdmxAvailabilityResult,
    SdmxDataResult,
    SdmxObservation,
    iter_data_csv_batches,
    iter_data_csv_rows,
    observation_schema,
    parse_availability_json,
)
from dfo_platform.request_deadline import bounded_deadline
from dfo_platform.timings import carrying_timings, count

_DATAFLOW_AGENCY = "DC"
_DATAFLOW_ID = "DF_OBS"
_DATAFLOW_VERSION = "1.0.0"
_KEY_WILDCARD = "*"

_DATA_PATH = (
    f"/sdmx/v3/data/dataflow/{_DATAFLOW_AGENCY}/{_DATAFLOW_ID}/{_DATAFLOW_VERSION}/{_KEY_WILDCARD}"
)
_AVAILABILITY_PATH_PREFIX = f"/sdmx/v3/availability/dataflow/{_DATAFLOW_AGENCY}/{_DATAFLOW_ID}/{_DATAFLOW_VERSION}/{_KEY_WILDCARD}"

_ACCEPT_DATA_CSV = "application/vnd.sdmx.data+csv;version=2.0.0"
_ACCEPT_AVAILABILITY_JSON = "application/vnd.sdmx.structure+json;version=2.0.0"

# Some deployments return HTTP 200 `{}` on `/data` before Envoy routes SDMX.
# This marker makes `data()` reject that response as `SdmxResponseError`.
_CSV_CONTENT_TYPE_MARKER = "csv"

_TOKEN_HEADER = "x-graph-token"

# Keep this literal to avoid importing Search. WAF-fronted deployments can
# challenge it. Callers can override it with the `user_agent` argument.
_DEFAULT_USER_AGENT = "dfo-apis (+https://github.com/DevelopmentFinanceObservatory/apis)"

# Content-Type marker for HTML challenge pages in place of mixer error JSON.
_HTML_CONTENT_TYPE_MARKER = "html"

# Match upstream gRPC message substrings to typed HTTP errors, in order.
_ERROR_SUBSTRING_MAP: tuple[tuple[tuple[str, ...], type[SdmxHttpError]], ...] = (
    (("missing required SDMX component filter",), SdmxMissingFilterError),
    (("require containedInPlace+ and typeOf",), SdmxContainmentFilterError),
    (("incompatible observationProperties",), SdmxIncompatibleVariablesError),
    (("max supported entity slots is",), SdmxRoleCeilingError),
    (
        ("unsupported SDMX component filter", "unsupported SDMX availability component"),
        SdmxUnsupportedFilterError,
    ),
)

_DEADLINE_EXCEEDED_DETAIL = "the SDMX read exceeded its absolute caller deadline"

# PyArrow requires rows to fit within two blocks. 64 KiB is the smallest
# power of two above the renderers' 32,767-character Excel cell cap.
# `open_csv()` primes about 34 blocks of read-ahead regardless of threads,
# bounding early over-read to roughly 2.2 MiB.
_STREAM_BATCH_BLOCK_SIZE = 1 << 16

# The mixer returns 501 for these `/availability` range operators. Reject before HTTP.
_TIME_PERIOD_OPERATOR_PREFIXES = ("ge:", "le:", "gt:", "lt:", "ne:")

# The proxy returns HTTP 414 above roughly 8 KiB. Sixty CRS StatVars averaging
# 126 characters fit at 7,980 query characters. 61 fail. The 7,800-byte budget
# covers encoded query parameters only, below the measured 8,192-byte ceiling.
# It excludes the endpoint path and host. Override with `url_budget_bytes`.
_DEFAULT_URL_BUDGET_BYTES = 7_800

DEFAULT_PLANNING_LANES = 1
"""`plan_request_chunks`'s own default `lanes`: no target size beyond the
URL byte budget (`ceil(n / 1)` never binds before `_pack_variable_chunks`
does), for a caller issuing one combined request and letting the byte
budget alone decide how many chunks that becomes."""

# A purpose-open CRS read has 354 chunks at a 0.31 s median, or 108 s sequential.
# Eight workers fit the 45 s deadline and httpx's default pool. Head-of-line
# waits make actual time exceed 108/8. Override with `chunk_concurrency`.
_DEFAULT_CHUNK_CONCURRENCY = 8

# Per-instance GET cap, shared by all methods and concurrent calls. A shared
# semaphore can coordinate several clients. Each instance owns its own by default.
_DEFAULT_TOTAL_CONCURRENCY = 8

# Poll every 20 ms so coordinators notice completed siblings while waiting
# for a contended slot, including when `total_concurrency < chunk_concurrency`.
_SLOT_POLL_INTERVAL = 0.02


def _count_request(_request: httpx.Request) -> None:
    """Count each request this client sends, retries included, on the calling door's timings
    as `sdmx_requests`."""
    count("sdmx_requests")


class _ReadStoppedError(SdmxTransportError):
    """Worker marker for a request stopped by a sibling failure or cap abort.

    Coordinators report the earliest genuine failure in chunk-plan order
    when a pop encounters this marker. It never reaches a caller of `data()` or `availability()`.
    """

    def __init__(self) -> None:
        super().__init__("the chunked read was aborted before this chunk's request was served")


class _DeadlineGuard:
    """Share one deadline and abort signal across a read's open responses.

    httpx times each transport operation separately. This guard bounds the
    whole chunked read and closes every armed response on expiry or sibling
    failure. With `deadline=None`, time checks are disabled. Concurrent reads
    retain the guard for sibling cancellation. Sequential reads pass no guard.
    A read's deadline is the earlier of its caller's and the bound request
    deadline (`dfo_platform.request_deadline`).

    `window_seconds` bounds the read from the moment its first GET holds a slot
    (`open_window`), so time spent waiting for a slot does not count against it.
    """

    def __init__(
        self, deadline: float | None = None, *, window_seconds: float | None = None
    ) -> None:
        self.deadline = deadline
        self.window_seconds = window_seconds
        self._window_opened = False
        self.expired = threading.Event()
        self.stopped = threading.Event()
        self._lock = threading.Lock()
        # Serialise worker admission with failure registration so a landed failure
        # prevents later workers from starting GETs.
        self._admission_lock = threading.RLock()
        self._active = True
        self._responses: set[httpx.Response] = set()
        # Genuine worker failures, keyed by plan index, decide which error
        # coordinators report. The callback writes from worker threads. Pops read.
        self.failures: dict[int, BaseException] = {}
        self._failures_lock = threading.Lock()
        self.failure_landed = threading.Event()
        if deadline is not None:
            self._timer = threading.Timer(self.remaining(), self._expire)
            self._timer.daemon = True
            self._timer.start()
        else:
            self._timer = None

    def open_window(self) -> None:
        """Narrow `deadline` to `window_seconds` from now, on the first call only. Called once
        a GET holds its slot."""
        with self._lock:
            if self.window_seconds is None or self._window_opened or not self._active:
                return
            self._window_opened = True
            window_end = time.monotonic() + self.window_seconds
            if self.deadline is not None and self.deadline <= window_end:
                return
            self.deadline = window_end
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.window_seconds, self._expire)
            self._timer.daemon = True
            self._timer.start()

    def remaining(self) -> float:
        if self.deadline is None:
            return float("inf")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self._expire()
            raise TimeoutError(_DEADLINE_EXCEEDED_DETAIL)
        return remaining

    def arm(self, response: httpx.Response) -> None:
        with self._lock:
            stopped = self.stopped.is_set()
            expired = self.expired.is_set()
            active = self._active
            if active:
                self._responses.add(response)
        if not active:
            # Close a response that arrives after guard teardown, before reading its body.
            self._abort(response)
            raise _ReadStoppedError()
        if expired:
            self._abort(response)
            raise TimeoutError(_DEADLINE_EXCEEDED_DETAIL)
        if stopped:
            # `abort()` already took its response snapshot. Close this late arrival locally.
            self._abort(response)
            raise _ReadStoppedError()

    def disarm(self, response: httpx.Response) -> None:
        """Remove a completed response from the abort set. Safe after expiry or abort."""
        with self._lock:
            self._responses.discard(response)

    def check(self) -> None:
        if self.deadline is not None and (
            self.expired.is_set() or time.monotonic() >= self.deadline
        ):
            self._expire()
            raise TimeoutError(_DEADLINE_EXCEEDED_DETAIL)

    def raise_if_expired(self, exc: BaseException) -> None:
        if self.deadline is not None and (
            self.expired.is_set() or time.monotonic() >= self.deadline
        ):
            self._expire()
            raise TimeoutError(_DEADLINE_EXCEEDED_DETAIL) from exc

    def abort(self) -> None:
        """Set `stopped`, release slot waiters, and close every armed response.

        Failure watchers and coordinators may both call this. It is
        idempotent. Closing a response interrupts a blocked body read.
        """
        with self._admission_lock, self._lock:
            if not self._active:
                return
            self.stopped.set()
            responses = list(self._responses)
        for response in responses:
            self._abort(response)

    def preflight(self) -> None:
        """Refuse a worker that was admitted before a sibling failed."""
        with self._admission_lock:
            self.check()
            if self.stopped.is_set() or not self._active:
                raise _ReadStoppedError()

    def cancel(self) -> None:
        with self._lock:
            self._active = False
            self._responses.clear()
        if self._timer is not None:
            self._timer.cancel()

    def _expire(self) -> None:
        with self._lock:
            if not self._active:
                return
            self.expired.set()
            responses = list(self._responses)
        for response in responses:
            self._abort(response)

    @staticmethod
    def _abort(response: httpx.Response) -> None:
        # Socket shutdown interrupts blocked reads. `response.close()` releases the connection.
        network_stream = response.extensions.get("network_stream")
        get_extra_info = getattr(network_stream, "get_extra_info", None)
        if get_extra_info is not None:
            try:
                active_socket = get_extra_info("socket")
                if active_socket is not None:
                    shutdown = getattr(active_socket, "shutdown", None)
                    if shutdown is not None:
                        shutdown(socket.SHUT_RDWR)
            except (OSError, RuntimeError):
                pass
        with suppress(OSError, RuntimeError, ValueError):
            response.close()


def _on_chunk_future_done(future: Future[Any], *, plan_index: int, guard: _DeadlineGuard) -> None:
    """Record a genuine future failure and abort immediately, even if an earlier chunk is blocked.

    Successes, cancellations, and `_ReadStoppedError` markers leave the
    registry unchanged. Retrieving each exception also prevents unobserved-
    future warnings after an aborted read returns.
    """

    try:
        exc = future.exception()
    except CancelledError:
        return
    if exc is None or isinstance(exc, _ReadStoppedError):
        return
    # Make failure registration atomic with slot-backed worker admission.
    with guard._admission_lock:
        with guard._failures_lock:
            guard.failures.setdefault(plan_index, exc)
        guard.failure_landed.set()
        guard.abort()


class _TotalSlotLease:
    """One pre-acquired total-concurrency slot, released exactly once."""

    def __init__(self, semaphore: threading.BoundedSemaphore) -> None:
        self._semaphore = semaphore
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._semaphore.release()


class _ConcurrentChunkRead:
    """Shared worker/slot lifecycle for the two concurrent chunked reads."""

    def __init__(
        self,
        client: SdmxClient,
        guard: _DeadlineGuard,
        *,
        max_workers: int,
    ) -> None:
        self._client = client
        self.guard = guard
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sdmx-chunk")
        self._futures: set[Future[Any]] = set()
        self._finished = False
        self._lifecycle_ended = False
        self._failure_settled = threading.Event()
        self._lock = threading.Lock()
        # Keep the client open until workers stalled before response headers settle.
        self._client._retain_request()

    def submit(
        self,
        worker: Callable[[], Any],
        *,
        plan_index: int,
        wake_when: Callable[[], bool] | None,
    ) -> Future[Any] | None:
        """Submit an already-slot-backed worker while the read is active."""
        lease = _TotalSlotLease(self._client._total_semaphore)
        with self.guard._admission_lock:
            if self.guard.stopped.is_set() or (wake_when is not None and wake_when()):
                lease.release()
                return None
            future = self.pool.submit(self._run_worker, carrying_timings(worker))
            with self._lock:
                self._futures.add(future)
            # Close admission before the released permit can wake another coordinator.
            self._client._watch_chunk_future(future, plan_index=plan_index, guard=self.guard)
            future.add_done_callback(lambda done: self._settle(done, lease))
            return future

    def _run_worker(self, worker: Callable[[], Any]) -> Any:
        self.guard.preflight()
        return worker()

    def _settle(self, future: Future[Any], lease: _TotalSlotLease) -> None:
        lease.release()
        # Publish the released lease before `result()` reports a sibling failure.
        if self.guard.failure_landed.is_set():
            self._failure_settled.set()
        with self._lock:
            self._futures.discard(future)
            should_end = self._finished and not self._futures and not self._lifecycle_ended
            if should_end:
                self._lifecycle_ended = True
        if should_end:
            self.guard.cancel()
            self._client._end_request()

    def result(self, future: Future[Any]) -> Any:
        """Wait for the plan-head result without hiding another chunk's failure.

        Successful results merge in plan order. Completion callbacks record
        failures immediately, so the coordinator can report a later chunk's
        error while an earlier worker is stuck before response headers.
        """
        while True:
            try:
                return future.result(timeout=_SLOT_POLL_INTERVAL)
            except TimeoutError:
                if future.done():
                    # Return the result or worker error from the completed future.
                    return future.result()
                with self.guard._failures_lock:
                    if self.guard.failures:
                        self._failure_settled.wait()
                        raise self.guard.failures[min(self.guard.failures)] from None

    def finish(self, *, aborted: bool) -> None:
        if aborted:
            self.guard.abort()
            self.pool.shutdown(wait=False, cancel_futures=True)
        else:
            self.pool.shutdown(wait=True)
        with self._lock:
            self._finished = True
            should_end = not self._futures and not self._lifecycle_ended
            if should_end:
                self._lifecycle_ended = True
        if should_end:
            self.guard.cancel()
            self._client._end_request()


def _rows_before_deadline(
    rows: Iterator[SdmxObservation], guard: _DeadlineGuard
) -> Iterator[SdmxObservation]:
    """Yield rows while checking the same absolute bound between chunks."""
    while True:
        guard.check()
        try:
            row = next(rows)
        except StopIteration:
            guard.check()
            return
        except BaseException as exc:
            guard.raise_if_expired(exc)
            raise
        guard.check()
        yield row


def _batches_before_deadline(
    batches: Iterator[pa.RecordBatch], guard: _DeadlineGuard
) -> Iterator[pa.RecordBatch]:
    """Check the absolute deadline between batches."""
    while True:
        guard.check()
        try:
            batch = next(batches)
        except StopIteration:
            guard.check()
            return
        except BaseException as exc:
            guard.raise_if_expired(exc)
            raise
        guard.check()
        yield batch


@pydantic.dataclasses.dataclass(frozen=True)
class ContainedIn:
    """Select `child_type` entities contained in `parent`.

    Requests require both `<role>.containedInPlace+` and `<role>.typeOf`.
    """

    parent: str
    child_type: str


type EntityFilter = Sequence[str] | ContainedIn


@dataclass(frozen=True, slots=True)
class _BufferedChunk:
    """One fully read `/data` chunk, before its roles join the result's union."""

    roles: tuple[str, ...]
    rows: tuple[SdmxObservation, ...]
    bytes_read: int


class SdmxClient:
    """Sync client for the mixer's `/data` and `/availability` endpoints on `DC/DF_OBS/1.0.0`.

    Each instance owns one `httpx.Client`. `base_url` is the API root and
    `transport` can supply an `httpx.MockTransport` for tests. Query strings
    over `url_budget_bytes` are split and merged. `chunk_concurrency` limits
    concurrent chunks in `data()` and `availability()`. Single-chunk calls
    skip the pool. Streaming methods read chunks sequentially so callers can
    stop early.

    `total_concurrency` limits all GETs from this instance. Pass a shared
    `threading.BoundedSemaphore` to coordinate several instances. An integer
    creates a per-instance limit. Defaults are defined by the module constants.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        user_agent: str = _DEFAULT_USER_AGENT,
        url_budget_bytes: int = _DEFAULT_URL_BUDGET_BYTES,
        chunk_concurrency: int = _DEFAULT_CHUNK_CONCURRENCY,
        total_concurrency: int | threading.BoundedSemaphore = _DEFAULT_TOTAL_CONCURRENCY,
    ) -> None:
        headers = {"User-Agent": user_agent}
        if token:
            headers[_TOKEN_HEADER] = token
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
            event_hooks={"request": [_count_request]},
        )
        self._timeout = timeout
        self._url_budget_bytes = url_budget_bytes
        self._chunk_concurrency = chunk_concurrency
        self._total_semaphore = (
            total_concurrency
            if isinstance(total_concurrency, threading.BoundedSemaphore)
            else threading.BoundedSemaphore(total_concurrency)
        )
        self._lifecycle_lock = threading.Lock()
        self._close_started = False
        self._client_closed = False
        self._active_requests = 0

    @property
    def url_budget_bytes(self) -> int:
        """This instance's query-string byte limit. Construct a new client to change it."""
        return self._url_budget_bytes

    def source_key(self) -> str:
        """The API root this client reads, which identifies its backend."""
        return str(self._client.base_url)

    def close(self) -> None:
        """Reject new work and close once active requests have settled."""
        with self._lifecycle_lock:
            self._close_started = True
            close_now = self._active_requests == 0 and not self._client_closed
            if close_now:
                self._client_closed = True
        if close_now:
            self._client.close()

    def _begin_request(self) -> None:
        with self._lifecycle_lock:
            if self._close_started:
                raise RuntimeError("SdmxClient is closed")
            self._active_requests += 1

    def _retain_request(self) -> None:
        """Extend an operation already admitted before close() started."""
        with self._lifecycle_lock:
            self._active_requests += 1

    def _end_request(self) -> None:
        with self._lifecycle_lock:
            self._active_requests -= 1
            close_now = (
                self._close_started and self._active_requests == 0 and not self._client_closed
            )
            if close_now:
                self._client_closed = True
        if close_now:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def data(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None = None,
        time_period: Literal["LATEST"] | Sequence[str] | None = None,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> SdmxDataResult:
        """Fetch observations for `variables` filtered by entity roles.

        `entities` maps roles to OR-lists of dcids or `ContainedIn` filters.
        Without `time_period`, the full matched series is returned. Explicit
        periods form an OR-list. `"LATEST"` returns one latest row per
        variable/entity/facet series. A USA `Count_Person` query returned 16
        latest rows from its 496-row full series. This deployment ignores
        standard SDMX time parameters, so `time_period` is its only time filter.

        Query strings over `url_budget_bytes` split into chunks. One chunk
        makes one request through `stream_data`. Multiple chunks use a bounded
        pool and merge rows and role columns in plan order, issuing exactly
        one request per chunk. `chunk_concurrency` bounds active chunks.

        `max_rows` and `max_bytes` default to uncapped. The mixer has no
        server-side `/data` cap. Chunked reads cap each response body and
        check cumulative totals as chunks land. A cap raises
        `SdmxCapExceededError` and aborts sibling requests. Concurrent worker
        byte-cap errors report a lower bound for total
        bytes consumed. `deadline` is an absolute `time.monotonic()` value
        for the full read.

        Raises `SdmxQueryError` before HTTP for empty `variables` or `entities`,
        duplicate variables, reserved filter characters, a non-`LATEST` bare
        `time_period` string, an empty period list, or `LATEST` in a list.
        A variable that exceeds the URL budget also raises this error.
        Non-2xx responses raise a typed `SdmxHttpError`, transport failures
        raise `SdmxTransportError`, and malformed 2xx responses raise
        `SdmxResponseError` with the upstream HTTP status. A non-2xx body
        over `max_bytes` raises `SdmxCapExceededError` before HTTP mapping.
        """
        self._begin_request()
        try:
            chunks = self._plan_variable_chunks(
                variables=variables, entities=entities, facet_ids=facet_ids, time_period=time_period
            )
            if len(chunks) == 1:
                with self.stream_data(
                    variables=variables,
                    entities=entities,
                    facet_ids=facet_ids,
                    time_period=time_period,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    deadline=deadline,
                    _operation_active=True,
                ) as (roles, rows):
                    return SdmxDataResult(roles=roles, rows=tuple(rows))
            return self._buffered_chunked_data(
                chunks,
                entities=entities,
                facet_ids=facet_ids,
                time_period=time_period,
                max_rows=max_rows,
                max_bytes=max_bytes,
                deadline=deadline,
            )
        finally:
            self._end_request()

    def _buffered_chunked_data(
        self,
        chunks: Sequence[tuple[str, ...]],
        *,
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
        max_rows: int | None,
        max_bytes: int | None,
        deadline: float | None,
    ) -> SdmxDataResult:
        """Fetch and merge multiple chunks in plan order on a bounded thread pool.

        The driving loop checks each landed chunk's byte and row totals in
        plan order, independent of `chunk_concurrency`. Each response is
        capped at `max_bytes`. Cumulative bytes are checked before rows, as
        in the sequential reader. Rows are checked against the remaining
        `max_rows`. These incremental checks limit over-read to at most
        `chunk_concurrency - 1` in-flight chunks. A worker byte-cap error is
        rescaled to the cumulative total, a lower bound under concurrency.

        The loop acquires each total-concurrency slot before submission and
        transfers it to a future lease released when that worker settles.
        Slot waits wake when a sibling completes. A future failure aborts all
        armed responses immediately. The coordinator reports the earliest
        genuine failure in plan order. Open body reads are interrupted, while
        workers stalled in transport setup drain in the background with their
        leases held. The abort path cancels queued futures and returns without
        waiting for those workers. One shared guard enforces the whole-read
        deadline and sibling cancellation, including when no deadline is set.
        """
        guard = _DeadlineGuard(bounded_deadline(deadline))
        max_workers = max(1, min(self._chunk_concurrency, len(chunks)))
        controller = _ConcurrentChunkRead(self, guard, max_workers=max_workers)
        seen: set[str] = set()
        union: list[str] = []
        all_rows: list[SdmxObservation] = []
        rows_so_far = 0
        bytes_so_far = 0
        try:
            pending: list[Future[_BufferedChunk]] = []
            for i in range(min(max_workers, len(chunks))):
                # The coordinator acquires each slot before submit. `wake_when`
                # lets a completed sibling pre-empt a contended acquire.
                if not self._acquire_total_slot(
                    guard, wake_when=lambda: any(f.done() for f in pending)
                ):
                    # Pop a completed sibling after a wake or abort.
                    break
                future = controller.submit(
                    lambda chunk=chunks[i]: self._fetch_chunk_buffered(
                        chunk,
                        entities=entities,
                        facet_ids=facet_ids,
                        time_period=time_period,
                        guard=guard,
                        max_bytes=max_bytes,
                    ),
                    plan_index=i,
                    wake_when=lambda: any(f.done() for f in pending),
                )
                if future is None:
                    break
                pending.append(future)
            next_chunk = len(pending)
            for _chunk in chunks:
                # Pop in plan order so completion timing cannot reorder output.
                future = pending.pop(0)
                try:
                    chunk = controller.result(future)
                except SdmxCapExceededError as exc:
                    # Add bytes from earlier chunks to this worker's byte-cap count.
                    raise _rescale_worker_byte_cap_error(exc, bytes_so_far=bytes_so_far) from exc
                except _ReadStoppedError:
                    # This worker stopped after a sibling failed. Report the earliest
                    # genuine failure recorded by chunk-plan index.
                    with guard._failures_lock:
                        idx = min(guard.failures) if guard.failures else None
                    if idx is not None:
                        exc = guard.failures[idx]
                        if isinstance(exc, SdmxCapExceededError) and exc.cap_kind == "bytes":
                            exc = _rescale_worker_byte_cap_error(exc, bytes_so_far=bytes_so_far)
                        raise exc from None
                    raise
                for role in chunk.roles:
                    if role not in seen:
                        seen.add(role)
                        union.append(role)
                # Check each landed chunk before submitting another to bound over-read.
                bytes_so_far += chunk.bytes_read
                if max_bytes is not None and bytes_so_far > max_bytes:
                    raise SdmxCapExceededError(
                        f"SDMX /data read exceeded the byte cap of {max_bytes} "
                        f"(observed {bytes_so_far} bytes before aborting)",
                        cap_kind="bytes",
                        cap=max_bytes,
                        observed=bytes_so_far,
                    )
                if max_rows is None:
                    all_rows.extend(chunk.rows)
                else:
                    remaining = max_rows - rows_so_far
                    for local_count, row in enumerate(chunk.rows, start=1):
                        if local_count > remaining:
                            observed = rows_so_far + local_count
                            raise SdmxCapExceededError(
                                f"SDMX /data read exceeded the row cap of {max_rows} "
                                f"(observed {observed} rows before aborting)",
                                cap_kind="rows",
                                cap=max_rows,
                                observed=observed,
                            )
                        all_rows.append(row)
                rows_so_far += len(chunk.rows)
                if next_chunk < len(chunks):
                    if not self._acquire_total_slot(
                        guard, wake_when=lambda: any(f.done() for f in pending)
                    ):
                        # Pop the completed sibling before submitting another chunk.
                        continue
                    future = controller.submit(
                        lambda chunk=chunks[next_chunk]: self._fetch_chunk_buffered(
                            chunk,
                            entities=entities,
                            facet_ids=facet_ids,
                            time_period=time_period,
                            guard=guard,
                            max_bytes=max_bytes,
                        ),
                        plan_index=next_chunk,
                        wake_when=lambda: any(f.done() for f in pending),
                    )
                    if future is None:
                        continue
                    pending.append(future)
                    next_chunk += 1
        except BaseException:
            # Abort open responses and cancel queued futures without joining workers
            # stalled before headers. Those workers retain their leases until they settle.
            controller.finish(aborted=True)
            raise
        controller.finish(aborted=False)
        return SdmxDataResult(roles=tuple(union), rows=tuple(all_rows))

    def _watch_chunk_future(
        self, future: Future[Any], *, plan_index: int, guard: _DeadlineGuard
    ) -> None:
        """Watch a submitted chunk so any genuine failure aborts the read immediately."""
        future.add_done_callback(
            lambda done: _on_chunk_future_done(done, plan_index=plan_index, guard=guard)
        )

    def _fetch_chunk_buffered(
        self,
        chunk: tuple[str, ...],
        *,
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
        guard: _DeadlineGuard,
        max_bytes: int | None,
    ) -> _BufferedChunk:
        """Read one chunk fully on a worker and return its rows and byte count.

        `max_bytes` caps this response. Cumulative bytes and all row counts
        are checked by the coordinator as chunks land. The shared guard
        enforces the deadline and aborts sibling responses. The coordinator
        holds this worker's total-concurrency slot until its future settles.
        Raises `TimeoutError` when the deadline expires mid-read.
        """
        with self._open_chunk_rows(
            variables=chunk,
            entities=entities,
            facet_ids=facet_ids,
            time_period=time_period,
            max_rows=None,
            max_bytes=max_bytes,
            guard=guard,
            slot_preacquired=True,
        ) as (roles, rows, reader):
            rows_iter = rows if guard is None else _rows_before_deadline(rows, guard)
            buffered_rows = tuple(rows_iter)
        return _BufferedChunk(roles=roles, rows=buffered_rows, bytes_read=reader.bytes_read)

    @contextmanager
    def stream_data(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None = None,
        time_period: Literal["LATEST"] | Sequence[str] | None = None,
        roles: Sequence[str] | None = None,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        _operation_active: bool = False,
    ) -> Iterator[tuple[tuple[str, ...], Iterator[SdmxObservation]]]:
        """Yield `(roles, rows)` with roles resolved eagerly and rows parsed lazily.

        Callers may stop `rows` early. The open response closes when the
        context exits. For one chunk, the header is resolved in one request.
        Multiple chunks are read sequentially, with one response open at a
        time, and merged in request order. The method pre-scans every header
        to return the full role union before yielding rows. This doubles
        chunked request count to avoid holding all connections open. Supplying
        `roles` skips the pre-scan and requires each chunk's header to match
        that role set, in the caller's order. Mismatches raise
        `SdmxResponseError` with the chunk's variables and differing roles.
        Pre-scanning avoids holding every response open, which could exhaust
        httpx's connection pool and deadlock a large read.

        `entities`, `facet_ids`, and `time_period` follow `data()`. Row and
        decoded-byte caps apply across the full read, including pre-scan bytes.
        Byte caps can trip on the header before rows are yielded. Row caps are
        checked as rows are pulled. Caps raise `SdmxCapExceededError`, and a
        non-2xx body over `max_bytes` raises it before HTTP error mapping.
        `deadline` is an absolute monotonic bound covering requests and the
        pre-scan. Expiry aborts the open response.

        Raises `SdmxQueryError` before HTTP for empty or duplicate variables,
        empty entities, reserved filter characters, invalid time filters, or
        a variable that exceeds the URL budget. Non-2xx responses raise typed
        `SdmxHttpError` subclasses. Transport failures raise
        `SdmxTransportError`. Malformed 2xx responses raise
        `SdmxResponseError` with the upstream status.
        """
        if not _operation_active:
            self._begin_request()
        caller_roles = None if roles is None else tuple(roles)
        guard: _DeadlineGuard | None = None
        try:
            chunks = self._plan_variable_chunks(
                variables=variables, entities=entities, facet_ids=facet_ids, time_period=time_period
            )
            deadline = bounded_deadline(deadline)
            guard = _DeadlineGuard(deadline) if deadline is not None else None
            if len(chunks) == 1:
                with self._open_chunk_rows(
                    variables=chunks[0],
                    entities=entities,
                    facet_ids=facet_ids,
                    time_period=time_period,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    guard=guard,
                    expected_roles=caller_roles,
                ) as (resolved_roles, rows, _reader):
                    if guard is not None:
                        guard.check()
                    out_roles = caller_roles if caller_roles is not None else resolved_roles
                    yield out_roles, rows if guard is None else _rows_before_deadline(rows, guard)
            else:
                if caller_roles is not None:
                    known_roles: tuple[str, ...] = caller_roles
                    prescan_bytes = 0
                else:
                    known_roles, prescan_bytes = self._resolve_chunk_roles(
                        chunks,
                        entities=entities,
                        facet_ids=facet_ids,
                        time_period=time_period,
                        max_bytes=max_bytes,
                        guard=guard,
                    )
                rows = self._merged_rows(
                    chunks,
                    entities=entities,
                    facet_ids=facet_ids,
                    time_period=time_period,
                    roles=known_roles,
                    expected_roles=caller_roles,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    initial_bytes_read=prescan_bytes,
                    guard=guard,
                )
                try:
                    if guard is not None:
                        guard.check()
                    yield known_roles, rows if guard is None else _rows_before_deadline(rows, guard)
                finally:
                    # Close the child generator on early exit to release its open response.
                    rows.close()
        finally:
            if guard is not None:
                guard.cancel()
            if not _operation_active:
                self._end_request()

    @contextmanager
    def stream_data_batches(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None = None,
        time_period: Literal["LATEST"] | Sequence[str] | None = None,
        roles: Sequence[str] | None = None,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> Iterator[tuple[tuple[str, ...], Iterator[pa.RecordBatch]]]:
        """Yield `(roles, batches)` with eager roles and lazy `RecordBatch` output.

        Shares `stream_data()`'s request planning, chunk order, parameters,
        deadline, and errors. Batches use `observation_schema(roles)`. With
        inferred roles, missing columns in a chunk become null. Supplied
        roles are checked against each header and allow reordering without
        null fills.

        `max_rows` is checked per batch, so `observed` can exceed the cap by
        up to one batch. PyArrow's pinned block size bounds read-ahead. If
        byte and row caps trip within that window, `max_bytes` fires first.
        The row-based `stream_data()` checks rows before requesting more bytes.
        """
        self._begin_request()
        caller_roles = None if roles is None else tuple(roles)
        guard: _DeadlineGuard | None = None
        try:
            chunks = self._plan_variable_chunks(
                variables=variables, entities=entities, facet_ids=facet_ids, time_period=time_period
            )
            deadline = bounded_deadline(deadline)
            guard = _DeadlineGuard(deadline) if deadline is not None else None
            if len(chunks) == 1:
                with self._open_chunk_batches(
                    variables=chunks[0],
                    entities=entities,
                    facet_ids=facet_ids,
                    time_period=time_period,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    guard=guard,
                    expected_roles=caller_roles,
                ) as (resolved_roles, batches, _reader):
                    if guard is not None:
                        guard.check()
                    out_roles = caller_roles if caller_roles is not None else resolved_roles
                    if caller_roles is not None and resolved_roles != caller_roles:
                        batches = (_reproject_batch(b, caller_roles) for b in batches)
                    yield (
                        out_roles,
                        batches if guard is None else _batches_before_deadline(batches, guard),
                    )
            else:
                if caller_roles is not None:
                    known_roles: tuple[str, ...] = caller_roles
                    prescan_bytes = 0
                else:
                    known_roles, prescan_bytes = self._resolve_chunk_roles(
                        chunks,
                        entities=entities,
                        facet_ids=facet_ids,
                        time_period=time_period,
                        max_bytes=max_bytes,
                        guard=guard,
                    )
                batches = self._merged_batches(
                    chunks,
                    entities=entities,
                    facet_ids=facet_ids,
                    time_period=time_period,
                    roles=known_roles,
                    expected_roles=caller_roles,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    initial_bytes_read=prescan_bytes,
                    guard=guard,
                )
                try:
                    if guard is not None:
                        guard.check()
                    yield (
                        known_roles,
                        batches if guard is None else _batches_before_deadline(batches, guard),
                    )
                finally:
                    # Close the child generator on early exit to release its open response.
                    batches.close()
        finally:
            if guard is not None:
                guard.cancel()
            self._end_request()

    def _plan_variable_chunks(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
    ) -> list[tuple[str, ...]]:
        """Pack variables by the actual encoded query-string byte cost, preserving input order.

        The measured query includes entity, facet, and time filters. A list
        that fits returns as one unchanged chunk. An individually oversized
        variable raises `SdmxQueryError`. Empty inputs, duplicate variables,
        and reserved filter characters also raise before HTTP.

        Uses `plan_request_chunks` with `lanes=1`, so only the URL budget
        controls chunk sizes. Callers that need lane-sized requests split
        their variables before calling this method.
        """
        if not variables:
            raise SdmxQueryError("data() requires at least one variable")
        # Check before chunking, which would split a bare string character by character.
        _reject_bare_string(variables, label="variables")
        _reject_duplicate_variables(variables)
        if not entities:
            raise SdmxQueryError(
                "data() requires at least one entity filter; the mixer rejects a "
                "data request with none"
            )
        return plan_request_chunks(
            variables,
            entities=entities,
            facet_ids=facet_ids,
            time_period=time_period,
            url_budget_bytes=self._url_budget_bytes,
            lanes=1,
        )

    def _plan_availability_chunks(
        self,
        *,
        variables: Sequence[str],
        filters: Mapping[str, EntityFilter] | None,
        time_period: str | Sequence[str] | None,
    ) -> list[tuple[str, ...]]:
        """Pack `/availability` variables against the encoded URL budget.

        Uses the shared packing algorithm with availability filters and time
        parameters. Empty `filters` are allowed. `availability()` validates
        variables and rejects `variableMeasured` as a filter key first.
        Oversized variables and duplicates raise `SdmxQueryError` before HTTP.
        """
        # Check before packing, which would split a bare string character by character.
        _reject_bare_string(variables, label="variables")
        _reject_duplicate_variables(variables)
        fixed_params: dict[str, str] = {}
        if filters:
            fixed_params.update(_entity_params(filters))
        if time_period is not None:
            fixed_params["c[TIME_PERIOD]"] = _availability_time_period_param(time_period)

        def _query_len(chunk: Sequence[str]) -> int:
            params = {
                "c[variableMeasured]": _or_list("variableMeasured", chunk),
                **fixed_params,
            }
            return len(str(httpx.QueryParams(params)))

        return _pack_variable_chunks(variables, budget=self._url_budget_bytes, query_len=_query_len)

    @contextmanager
    def _open_chunk_rows(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
        max_rows: int | None,
        max_bytes: int | None,
        guard: _DeadlineGuard | None,
        slot_preacquired: bool = False,
        expected_roles: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[tuple[str, ...], Iterator[SdmxObservation], _CappedByteReader]]:
        """Open one `/data` chunk and resolve its header before returning `(roles, rows, reader)`.

        Header errors carry the response status. `reader.bytes_read` lets
        merge callers count cumulative bytes after draining rows. Caps apply
        to this chunk. Merge callers pass the remaining budget, while buffered
        workers cap each body at the caller's full `max_bytes`.

        `slot_preacquired` means the coordinator owns the concurrency slot for
        this GET. With `expected_roles`, the header must match the supplied role
        set. A mismatch raises `SdmxResponseError` naming the variables and
        roles. Otherwise the header is unchecked against an expected set.
        """
        with self._open_data_response(
            variables=variables,
            entities=entities,
            facet_ids=facet_ids,
            time_period=time_period,
            max_bytes=max_bytes,
            guard=guard,
            slot_preacquired=slot_preacquired,
        ) as response:
            try:
                roles, rows, reader = _stream_rows(
                    response, max_rows=max_rows, max_bytes=max_bytes, deadline_guard=guard
                )
            except SdmxResponseError as exc:
                raise SdmxResponseError(str(exc), upstream_status=response.status_code) from exc
            if expected_roles is not None:
                _check_expected_roles(
                    roles, expected_roles, variables=variables, upstream_status=response.status_code
                )
            # Keep the yield outside parse-error handling so row-consumption
            # errors retain their type.
            yield roles, rows, reader

    @contextmanager
    def _open_chunk_batches(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
        max_rows: int | None,
        max_bytes: int | None,
        guard: _DeadlineGuard | None,
        slot_preacquired: bool = False,
        expected_roles: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[tuple[str, ...], Iterator[pa.RecordBatch], _CappedByteReader]]:
        """`_open_chunk_rows`'s batch-oriented counterpart. See its
        docstring, including `expected_roles`."""
        with self._open_data_response(
            variables=variables,
            entities=entities,
            facet_ids=facet_ids,
            time_period=time_period,
            max_bytes=max_bytes,
            guard=guard,
            slot_preacquired=slot_preacquired,
        ) as response:
            try:
                roles, batches, reader = _stream_batches(
                    response, max_rows=max_rows, max_bytes=max_bytes, deadline_guard=guard
                )
            except SdmxResponseError as exc:
                raise SdmxResponseError(str(exc), upstream_status=response.status_code) from exc
            if expected_roles is not None:
                _check_expected_roles(
                    roles, expected_roles, variables=variables, upstream_status=response.status_code
                )
            # Keep the yield outside parse-error handling as in `_open_chunk_rows`.
            yield roles, batches, reader

    def _resolve_chunk_roles(
        self,
        chunks: Sequence[tuple[str, ...]],
        *,
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
        max_bytes: int | None,
        guard: _DeadlineGuard | None,
    ) -> tuple[tuple[str, ...], int]:
        """Read each chunk's header to compute the role union before yielding rows.

        Returns the roles in first-seen order and header bytes consumed.
        Merge readers include those bytes in the total cap and give each
        pre-scan request only the remaining byte budget.
        """
        seen: set[str] = set()
        union: list[str] = []
        bytes_so_far = 0
        for chunk in chunks:
            chunk_max_bytes = None if max_bytes is None else max_bytes - bytes_so_far
            try:
                with self._open_chunk_rows(
                    variables=chunk,
                    entities=entities,
                    facet_ids=facet_ids,
                    time_period=time_period,
                    max_rows=None,
                    max_bytes=chunk_max_bytes,
                    guard=guard,
                ) as (chunk_roles, _rows, reader):
                    for role in chunk_roles:
                        if role not in seen:
                            seen.add(role)
                            union.append(role)
            except SdmxCapExceededError as exc:
                raise _rescale_chunk_cap_error(
                    exc,
                    max_rows=None,
                    max_bytes=max_bytes,
                    rows_so_far=0,
                    bytes_so_far=bytes_so_far,
                ) from exc
            bytes_so_far += reader.bytes_read
        return tuple(union), bytes_so_far

    def _merged_rows(
        self,
        chunks: Sequence[tuple[str, ...]],
        *,
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
        roles: tuple[str, ...],
        max_rows: int | None,
        max_bytes: int | None,
        guard: _DeadlineGuard | None,
        initial_bytes_read: int = 0,
        expected_roles: tuple[str, ...] | None = None,
    ) -> Generator[SdmxObservation, None, None]:
        """Stream chunks in request order with cumulative caps, one response at a time.

        `roles` is the resolved union or the caller's supplied tuple.
        Rows omit role keys absent from their chunk. Supplied roles are
        checked against each chunk's header. `initial_bytes_read` carries
        pre-scan bytes into the total byte cap.
        """
        rows_so_far = 0
        bytes_so_far = initial_bytes_read
        for chunk in chunks:
            chunk_max_rows = None if max_rows is None else max_rows - rows_so_far
            chunk_max_bytes = None if max_bytes is None else max_bytes - bytes_so_far
            chunk_row_count = 0
            try:
                # Header resolution can trip `max_bytes` in `__enter__`, before the target binds.
                with self._open_chunk_rows(
                    variables=chunk,
                    entities=entities,
                    facet_ids=facet_ids,
                    time_period=time_period,
                    max_rows=chunk_max_rows,
                    max_bytes=chunk_max_bytes,
                    guard=guard,
                    expected_roles=expected_roles,
                ) as (_chunk_roles, chunk_rows, reader):
                    for row in chunk_rows:
                        chunk_row_count += 1
                        yield row
            except SdmxCapExceededError as exc:
                raise _rescale_chunk_cap_error(
                    exc,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    rows_so_far=rows_so_far,
                    bytes_so_far=bytes_so_far,
                ) from exc
            rows_so_far += chunk_row_count
            bytes_so_far += reader.bytes_read

    def _merged_batches(
        self,
        chunks: Sequence[tuple[str, ...]],
        *,
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
        roles: tuple[str, ...],
        max_rows: int | None,
        max_bytes: int | None,
        guard: _DeadlineGuard | None,
        initial_bytes_read: int = 0,
        expected_roles: tuple[str, ...] | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Stream chunk batches in request order with cumulative caps and fixed columns.

        Chunks missing union roles receive null columns. Differing role order
        is reprojected by name. Supplied roles are checked as a set. Header
        pre-scan bytes seed the cumulative byte count.
        """
        rows_so_far = 0
        bytes_so_far = initial_bytes_read
        for chunk in chunks:
            chunk_max_rows = None if max_rows is None else max_rows - rows_so_far
            chunk_max_bytes = None if max_bytes is None else max_bytes - bytes_so_far
            chunk_row_count = 0
            try:
                # Header resolution can trip `max_bytes` in `__enter__`, before the target binds.
                with self._open_chunk_batches(
                    variables=chunk,
                    entities=entities,
                    facet_ids=facet_ids,
                    time_period=time_period,
                    max_rows=chunk_max_rows,
                    max_bytes=chunk_max_bytes,
                    guard=guard,
                    expected_roles=expected_roles,
                ) as (chunk_roles, chunk_batches, reader):
                    needs_reprojection = chunk_roles != roles
                    for batch in chunk_batches:
                        chunk_row_count += batch.num_rows
                        yield _reproject_batch(batch, roles) if needs_reprojection else batch
            except SdmxCapExceededError as exc:
                raise _rescale_chunk_cap_error(
                    exc,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    rows_so_far=rows_so_far,
                    bytes_so_far=bytes_so_far,
                ) from exc
            rows_so_far += chunk_row_count
            bytes_so_far += reader.bytes_read

    def _acquire_total_slot(
        self,
        guard: _DeadlineGuard | None,
        *,
        wake_when: Callable[[], bool] | None = None,
    ) -> bool:
        """Acquire the slot every GET uses. Expiry raises `TimeoutError`.

        Without a guard, wait until a slot is free. With a guard, poll for
        slots and recheck expiry and abort between waits. `wake_when` lets a
        coordinator pop a completed sibling while a slot is contended. Return
        `True` with a slot held. Return `False` on abort or wake.
        Recheck conditions after acquisition and release the slot if they
        changed during the wait.
        """
        if guard is None:
            self._total_semaphore.acquire()
            return True
        while True:
            guard.check()  # raises TimeoutError once the deadline itself lands
            if guard.stopped.is_set():
                # Return `False` so the coordinator can report the sibling failure.
                return False
            if wake_when is not None and wake_when():
                return False
            if self._total_semaphore.acquire(timeout=min(_SLOT_POLL_INTERVAL, guard.remaining())):
                # Recheck expiry and abort after acquisition, releasing the slot on either.
                try:
                    guard.check()
                except BaseException:
                    self._total_semaphore.release()
                    raise
                if guard.stopped.is_set():
                    self._total_semaphore.release()
                    return False
                if wake_when is not None and wake_when():
                    self._total_semaphore.release()
                    return False
                return True

    @contextmanager
    def _open_data_response(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None,
        time_period: Literal["LATEST"] | Sequence[str] | None,
        max_bytes: int | None,
        guard: _DeadlineGuard | None,
        slot_preacquired: bool = False,
    ) -> Iterator[httpx.Response]:
        """Build and open one `/data` request, map errors, and check its content type.

        Callers validate variables and entities first. The caller owns a
        shared guard across chunks. `max_bytes` caps successful and error
        bodies. The caller reads successful bodies and applies row caps.

        By default this method acquires and releases the GET slot. With
        `slot_preacquired`, the coordinator's future lease owns it until the
        worker settles. Guarded slot waits and requests use the remaining
        deadline.
        """
        params = {"c[variableMeasured]": _or_list("variableMeasured", variables)}
        params.update(_entity_params(entities))
        if facet_ids:
            params["c[facetId]"] = _or_list("facetId", facet_ids)
        if time_period is not None:
            params["c[TIME_PERIOD]"] = _time_period_param(time_period)
        if not slot_preacquired:
            self._acquire_total_slot(guard)
        try:
            try:
                if guard is not None:
                    guard.preflight()
                request_timeout = (
                    self._timeout if guard is None else min(self._timeout, guard.remaining())
                )
                with self._client.stream(
                    "GET",
                    _DATA_PATH,
                    params=params,
                    headers={"Accept": _ACCEPT_DATA_CSV},
                    timeout=request_timeout,
                ) as response:
                    if guard is not None:
                        guard.arm(response)
                    try:
                        if not response.is_success:
                            _raise_for_non_success(response, guard=guard, max_bytes=max_bytes)
                        _check_csv_content_type(response)
                        yield response
                    finally:
                        # Drop this response before other chunks finish using the shared guard.
                        if guard is not None:
                            guard.disarm(response)
            except httpx.RequestError as exc:
                if guard is not None and guard.stopped.is_set():
                    # Closing an aborted response raises `RequestError`. Treat it as a stop marker.
                    raise _ReadStoppedError() from exc
                if guard is not None:
                    guard.raise_if_expired(exc)
                raise SdmxTransportError(
                    f"SDMX transport error calling {_DATA_PATH!r}: {exc}"
                ) from exc
        except BaseException:
            raise
        finally:
            if not slot_preacquired:
                self._total_semaphore.release()

    def availability(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        time_period: str | Sequence[str] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> SdmxAvailabilityResult:
        """Fetch member values for one component and a set of variables.

        `component` is a filterable SDMX component id. The `all` wildcard and
        non-filterable `TIME_PERIOD` are unsupported. Optional `filters` use
        role/value lists or `ContainedIn`. Omit them for an unfiltered read.
        Supply `variableMeasured` through the `variables` parameter. A period
        string is exact. A sequence is a deduplicated
        first-seen OR-list. `LATEST`, empty values, and range operators are
        rejected before HTTP.

        Query strings over `url_budget_bytes` split across concurrent chunks.
        Components are unioned in chunk-plan order with duplicate values
        removed. A multi-chunk result has `raw={}`. `max_values` caps the
        requested component's union. `max_bytes` caps each body and the
        cumulative bytes. Caps raise `SdmxCapExceededError`. `deadline` is an
        absolute monotonic bound over the full read. `read_seconds` also bounds
        it, counted from when its first GET holds a slot, so a wait for a slot
        does not count. Either expiry raises `TimeoutError`.

        Raises `SdmxQueryError` for empty or duplicate variables, invalid
        component/filter keys, reserved filter characters, or an oversized
        variable. Unsupported time filters raise `SdmxUnsupportedFilterError`.
        Non-2xx responses raise typed `SdmxHttpError` subclasses, transport
        failures raise `SdmxTransportError`, and malformed 2xx bodies raise
        `SdmxResponseError` with the upstream status.
        """
        if not variables:
            raise SdmxQueryError("availability() requires at least one variable")
        _validate_component_id(component, what="availability component")
        if filters and "variableMeasured" in filters:
            raise SdmxQueryError(
                "availability() filters must not include 'variableMeasured'; "
                "pass it via the variables parameter instead"
            )
        self._begin_request()
        try:
            chunks = self._plan_availability_chunks(
                variables=variables, filters=filters, time_period=time_period
            )
            deadline = bounded_deadline(deadline)
            guard = (
                _DeadlineGuard(deadline, window_seconds=read_seconds)
                if deadline is not None or read_seconds is not None
                else None
            )
            if len(chunks) == 1:
                try:
                    components, raw, _bytes_read = self._fetch_availability_chunk(
                        variables=chunks[0],
                        component=component,
                        filters=filters,
                        time_period=time_period,
                        max_bytes=max_bytes,
                        guard=guard,
                    )
                    if max_values is not None:
                        value_count = len(components.get(component, ()))
                        if value_count > max_values:
                            raise SdmxCapExceededError(
                                f"SDMX /availability read exceeded the value cap of "
                                f"{max_values} (observed {value_count} values)",
                                cap_kind="values",
                                cap=max_values,
                                observed=value_count,
                            )
                    return SdmxAvailabilityResult(components=components, raw=raw)
                finally:
                    if guard is not None:
                        guard.cancel()
            return self._chunked_availability(
                chunks,
                component=component,
                filters=filters,
                time_period=time_period,
                max_values=max_values,
                max_bytes=max_bytes,
                guard=guard,
            )
        finally:
            self._end_request()

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        """`availability`'s values for `component` alone, in its order."""
        result = self.availability(
            variables=variables,
            component=component,
            filters=filters,
            max_values=max_values,
            max_bytes=max_bytes,
            deadline=deadline,
            read_seconds=read_seconds,
        )
        return list(result.components.get(component, ()))

    def _fetch_availability_chunk(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None,
        time_period: str | Sequence[str] | None,
        max_bytes: int | None,
        guard: _DeadlineGuard | None,
        slot_preacquired: bool = False,
    ) -> tuple[Mapping[str, tuple[str, ...]], dict, int]:
        """Fetch, cap, and parse one `/availability` chunk.

        Returns its component map, raw body, and decoded bytes read. Each
        chunk is capped at the caller's `max_bytes`. The coordinator checks
        cumulative bytes and value unions as chunks land. This method reads
        the full body before returning.

        By default it acquires the GET slot itself. With `slot_preacquired`,
        the coordinator's future lease holds the slot until the worker
        settles. A guard bounds slot acquisition and the request deadline.
        """
        params = {"c[variableMeasured]": _or_list("variableMeasured", variables)}
        if filters:
            params.update(_entity_params(filters))
        if time_period is not None:
            params["c[TIME_PERIOD]"] = _availability_time_period_param(time_period)
        path = f"{_AVAILABILITY_PATH_PREFIX}/{component}"

        if not slot_preacquired:
            self._acquire_total_slot(guard)
        try:
            try:
                if guard is not None:
                    guard.open_window()
                    guard.preflight()
                request_timeout = (
                    self._timeout if guard is None else min(self._timeout, guard.remaining())
                )
                with self._client.stream(
                    "GET",
                    path,
                    params=params,
                    headers={"Accept": _ACCEPT_AVAILABILITY_JSON},
                    timeout=request_timeout,
                ) as response:
                    if guard is not None:
                        guard.arm(response)
                    try:
                        if not response.is_success:
                            _raise_for_non_success(response, guard=guard, max_bytes=max_bytes)
                        reader = _CappedByteReader(
                            response, max_bytes=max_bytes, deadline_guard=guard
                        )
                        raw_bytes = reader.readall()
                        if guard is not None:
                            guard.check()
                        body = _parse_availability_json_bytes(
                            raw_bytes, upstream_status=response.status_code
                        )
                        try:
                            result = parse_availability_json(body)
                        except SdmxResponseError as exc:
                            raise SdmxResponseError(
                                str(exc), upstream_status=response.status_code
                            ) from exc
                        return result.components, result.raw, reader.bytes_read
                    finally:
                        if guard is not None:
                            guard.disarm(response)
            except httpx.RequestError as exc:
                if guard is not None and guard.stopped.is_set():
                    # Closing an aborted response raises `RequestError`. Treat it as a stop marker.
                    raise _ReadStoppedError() from exc
                if guard is not None:
                    guard.raise_if_expired(exc)
                raise SdmxTransportError(f"SDMX transport error calling {path!r}: {exc}") from exc
        except BaseException:
            raise
        finally:
            if not slot_preacquired:
                self._total_semaphore.release()

    def _chunked_availability(
        self,
        chunks: Sequence[tuple[str, ...]],
        *,
        component: str,
        filters: Mapping[str, EntityFilter] | None,
        time_period: str | Sequence[str] | None,
        max_values: int | None,
        max_bytes: int | None,
        guard: _DeadlineGuard | None,
    ) -> SdmxAvailabilityResult:
        """Fetch multiple chunks concurrently and union component values in plan order.

        `max_values` applies to the deduplicated union for `component`.
        `max_bytes` caps each response and the cumulative read. Incremental
        checks limit over-read to `chunk_concurrency - 1` chunks. A worker's
        byte-cap error reports a lower bound under concurrency. The result's
        `raw` is `{}` because a chunked read has no single response body. Keep
        `url_budget_bytes` small relative to `max_bytes` for a tighter byte bound.

        The coordinator acquires each GET slot before submission and assigns
        it to a future lease. Completed siblings wake slot waits. Any genuine
        failure aborts the read immediately. The reported failure is the
        earliest in plan order. Armed responses close, while workers stalled
        in transport setup drain in the background with their leases held.
        The abort path cancels queued futures without waiting for those
        workers. A shared guard enforces the whole-read deadline and sibling
        cancellation, including when no deadline is set.
        """
        if guard is None:
            guard = _DeadlineGuard(None)
        max_workers = max(1, min(self._chunk_concurrency, len(chunks)))
        controller = _ConcurrentChunkRead(self, guard, max_workers=max_workers)
        seen: dict[str, set[str]] = {}
        union: dict[str, list[str]] = {}
        bytes_so_far = 0
        try:
            pending: list[Future[tuple[Mapping[str, tuple[str, ...]], dict, int]]] = []
            for i in range(min(max_workers, len(chunks))):
                # See `_buffered_chunked_data`'s matching comment: one slot
                # per chunk, acquired in chunk-plan order before the worker
                # is scheduled. `wake_when` notices a sibling already done
                # while this acquire waits on a contended semaphore.
                if not self._acquire_total_slot(
                    guard, wake_when=lambda: any(f.done() for f in pending)
                ):
                    # Pop a completed sibling after a wake or abort.
                    break
                future = controller.submit(
                    lambda chunk=chunks[i]: self._fetch_availability_chunk(
                        variables=chunk,
                        component=component,
                        filters=filters,
                        time_period=time_period,
                        guard=guard,
                        max_bytes=max_bytes,
                        slot_preacquired=True,
                    ),
                    plan_index=i,
                    wake_when=lambda: any(f.done() for f in pending),
                )
                if future is None:
                    break
                pending.append(future)
            next_chunk = len(pending)
            for _chunk in chunks:
                # Waits on the next chunk in submission (== chunk-plan)
                # order regardless of which worker's future resolves first:
                # the union merges in plan order, which is what keeps the
                # merged output deterministic under the completion-driven
                # failure contract (which chunk's failure surfaces is
                # race-determined. The merge remains in plan order.)
                future = pending.pop(0)
                try:
                    chunk_components, _raw, bytes_read = controller.result(future)
                except SdmxCapExceededError as exc:
                    # Add bytes from earlier chunks to this worker's byte-cap count.
                    raise _rescale_worker_byte_cap_error(exc, bytes_so_far=bytes_so_far) from exc
                except _ReadStoppedError:
                    # This worker stopped after a sibling failed. Report the earliest
                    # genuine failure recorded by chunk-plan index.
                    with guard._failures_lock:
                        idx = min(guard.failures) if guard.failures else None
                    if idx is not None:
                        exc = guard.failures[idx]
                        if isinstance(exc, SdmxCapExceededError) and exc.cap_kind == "bytes":
                            exc = _rescale_worker_byte_cap_error(exc, bytes_so_far=bytes_so_far)
                        raise exc from None
                    raise
                # Checked per landed chunk, in the order bytes then values,
                # before merging (see the docstring above): the incremental
                # check bounds the overread to `chunk_concurrency - 1`.
                bytes_so_far += bytes_read
                if max_bytes is not None and bytes_so_far > max_bytes:
                    raise SdmxCapExceededError(
                        f"SDMX /availability read exceeded the byte cap of {max_bytes} "
                        f"(observed {bytes_so_far} bytes before aborting)",
                        cap_kind="bytes",
                        cap=max_bytes,
                        observed=bytes_so_far,
                    )
                for comp_id, values in chunk_components.items():
                    comp_seen = seen.setdefault(comp_id, set())
                    comp_union = union.setdefault(comp_id, [])
                    for value in values:
                        if value not in comp_seen:
                            comp_seen.add(value)
                            comp_union.append(value)
                if max_values is not None:
                    value_count = len(union.get(component, ()))
                    if value_count > max_values:
                        raise SdmxCapExceededError(
                            f"SDMX /availability read exceeded the value cap of "
                            f"{max_values} (observed {value_count} values)",
                            cap_kind="values",
                            cap=max_values,
                            observed=value_count,
                        )
                if next_chunk < len(chunks):
                    if not self._acquire_total_slot(
                        guard, wake_when=lambda: any(f.done() for f in pending)
                    ):
                        # Pop the completed sibling before submitting another chunk.
                        continue
                    future = controller.submit(
                        lambda chunk=chunks[next_chunk]: self._fetch_availability_chunk(
                            variables=chunk,
                            component=component,
                            filters=filters,
                            time_period=time_period,
                            guard=guard,
                            max_bytes=max_bytes,
                            slot_preacquired=True,
                        ),
                        plan_index=next_chunk,
                        wake_when=lambda: any(f.done() for f in pending),
                    )
                    if future is None:
                        continue
                    pending.append(future)
                    next_chunk += 1
        except BaseException:
            # Abort open responses and cancel queued futures without joining workers
            # stalled before headers. Those workers retain their leases until they settle.
            controller.finish(aborted=True)
            raise
        controller.finish(aborted=False)
        return SdmxAvailabilityResult(
            components={comp_id: tuple(values) for comp_id, values in union.items()},
            raw={},
        )


def _raise_for_non_success(
    response: httpx.Response,
    *,
    guard: _DeadlineGuard | None,
    max_bytes: int | None = None,
) -> None:
    """Read a non-2xx response and raise its typed error.

    Error bodies obey `max_bytes` and the absolute deadline. Error mapping
    consumes the bytes collected by the capped reader.
    """
    try:
        raw_bytes = _CappedByteReader(response, max_bytes=max_bytes, deadline_guard=guard).readall()
    except BaseException as exc:
        if guard is not None:
            guard.raise_if_expired(exc)
        raise
    if guard is not None:
        guard.check()
    raise _typed_http_error(response, raw_bytes=raw_bytes)


def _typed_http_error(response: httpx.Response, *, raw_bytes: bytes) -> SdmxHttpError:
    unexpected_shape = _unexpected_shape_error(response, raw_bytes=raw_bytes)
    if unexpected_shape is not None:
        return unexpected_shape
    message = _upstream_message(response, raw_bytes=raw_bytes)
    for substrings, error_cls in _ERROR_SUBSTRING_MAP:
        if any(substring in message for substring in substrings):
            return error_cls(message, upstream_status=response.status_code)
    return SdmxHttpError(message, upstream_status=response.status_code)


def _unexpected_shape_error(
    response: httpx.Response, *, raw_bytes: bytes
) -> SdmxUnexpectedResponseShapeError | None:
    """Recognize HTML challenge pages when the content type is HTML and JSON parsing fails."""
    content_type = response.headers.get("content-type", "")
    if _HTML_CONTENT_TYPE_MARKER not in content_type.lower():
        return None
    try:
        json.loads(raw_bytes)
    except ValueError:
        pass
    else:
        return None
    snippet = _response_text(response, raw_bytes=raw_bytes)[:200]
    return SdmxUnexpectedResponseShapeError(
        f"SDMX request failed with status {response.status_code} and content-type "
        f"{content_type!r}, not the transcoded-gRPC JSON error shape -- body starts "
        f"with {snippet!r}",
        upstream_status=response.status_code,
    )


def _stream_rows(
    response: httpx.Response,
    *,
    max_rows: int | None,
    max_bytes: int | None,
    deadline_guard: _DeadlineGuard | None = None,
) -> tuple[tuple[str, ...], Iterator[SdmxObservation], _CappedByteReader]:
    """Resolve the header eagerly and return lazy, row-capped streaming output.

    Preserve line terminators in `TextIOWrapper` so `csv.reader` can rejoin
    quoted-field newlines. `max_bytes` is enforced while bytes arrive and can
    trip on the header. `max_rows` is checked one row at a time. Both raise
    `SdmxCapExceededError`. The returned reader exposes bytes consumed for
    chunk merging.
    """
    capped = _CappedByteReader(
        response, max_bytes=max_bytes, deadline_guard=deadline_guard, require_line_end=True
    )
    reader = io.TextIOWrapper(capped, encoding="utf-8", newline="")
    try:
        roles, rows = iter_data_csv_rows(reader)
    except UnicodeDecodeError as exc:
        raise SdmxResponseError(
            f"SDMX /data response body contained malformed UTF-8: {exc}"
        ) from exc

    def _capped_rows() -> Iterator[SdmxObservation]:
        count = 0
        try:
            for row in rows:
                count += 1
                if max_rows is not None and count > max_rows:
                    raise SdmxCapExceededError(
                        f"SDMX /data read exceeded the row cap of {max_rows} "
                        f"(observed {count} rows before aborting)",
                        cap_kind="rows",
                        cap=max_rows,
                        observed=count,
                    )
                yield row
        except UnicodeDecodeError as exc:
            raise SdmxResponseError(
                f"SDMX /data response body contained malformed UTF-8: {exc}",
                upstream_status=response.status_code,
            ) from exc
        except SdmxResponseError as exc:
            # Attach the response status to lazy row-parse errors.
            raise SdmxResponseError(str(exc), upstream_status=response.status_code) from exc

    return roles, _capped_rows(), capped


def _stream_batches(
    response: httpx.Response,
    *,
    max_rows: int | None,
    max_bytes: int | None,
    deadline_guard: _DeadlineGuard | None = None,
) -> tuple[tuple[str, ...], Iterator[pa.RecordBatch], _CappedByteReader]:
    """Resolve the header eagerly and return lazy, batch-capped Arrow output.

    PyArrow reads the capped byte stream directly and reports malformed
    UTF-8 as `ArrowInvalid`. The parser maps it and column-count errors to
    `SdmxResponseError`. Byte caps apply as chunks arrive. Row caps are
    checked at each batch's end and can overshoot by one batch.
    """
    source = _CappedByteReader(
        response, max_bytes=max_bytes, deadline_guard=deadline_guard, require_line_end=True
    )
    roles, batches = iter_data_csv_batches(source, block_size=_STREAM_BATCH_BLOCK_SIZE)

    def _capped_batches() -> Iterator[pa.RecordBatch]:
        cumulative = 0
        try:
            for batch in batches:
                cumulative += batch.num_rows
                if max_rows is not None and cumulative > max_rows:
                    raise SdmxCapExceededError(
                        f"SDMX /data read exceeded the row cap of {max_rows} "
                        f"(observed {cumulative} rows before aborting)",
                        cap_kind="rows",
                        cap=max_rows,
                        observed=cumulative,
                    )
                yield batch
        except SdmxResponseError as exc:
            # Attach the response status to lazy batch-parse errors.
            raise SdmxResponseError(str(exc), upstream_status=response.status_code) from exc

    return roles, _capped_batches(), source


def _check_expected_roles(
    roles: tuple[str, ...],
    expected_roles: tuple[str, ...],
    *,
    variables: Sequence[str],
    upstream_status: int | None,
) -> None:
    """Require matching role sets. Mismatches raise `SdmxResponseError` with both differences.

    Order is ignored for comparison. The caller's supplied order is used in
    output, so matching batches only need column reordering.
    """
    reported = set(roles)
    expected = set(expected_roles)
    if reported == expected:
        return
    detail = []
    unexpected = sorted(reported - expected)
    if unexpected:
        detail.append(f"unexpected role(s) {unexpected!r}")
    missing = sorted(expected - reported)
    if missing:
        detail.append(f"missing role(s) {missing!r}")
    raise SdmxResponseError(
        f"SDMX response for variables {list(variables)!r} declared role(s) "
        f"{sorted(reported)!r}, {' and '.join(detail)}, against the "
        f"caller-supplied roles={sorted(expected)!r}",
        upstream_status=upstream_status,
    )


def plan_request_chunks(
    variables: Sequence[str],
    *,
    entities: Mapping[str, EntityFilter],
    facet_ids: Sequence[str] | None = None,
    time_period: Literal["LATEST"] | Sequence[str] | None = None,
    url_budget_bytes: int = _DEFAULT_URL_BUDGET_BYTES,
    lanes: int = DEFAULT_PLANNING_LANES,
) -> list[tuple[str, ...]]:
    """Split variables in order into lane-sized groups, then pack by URL byte budget.

    Each group has `ceil(len(variables) / lanes)` entries before the encoded
    query cost is applied. Fixed entity, facet, and time filters count toward
    the URL budget. The client uses `lanes=1`. The coverage estimator passes
    its dispatch lane count so both share the same request plan.
    """
    target = max(1, math.ceil(len(variables) / lanes))
    grouped: list[tuple[str, ...]] = [
        tuple(variables[i : i + target]) for i in range(0, len(variables), target)
    ]
    fixed_params = _entity_params(entities)
    if facet_ids:
        fixed_params["c[facetId]"] = _or_list("facetId", facet_ids)
    if time_period is not None:
        fixed_params["c[TIME_PERIOD]"] = _time_period_param(time_period)

    def _query_len(chunk: Sequence[str]) -> int:
        params = {"c[variableMeasured]": _or_list("variableMeasured", chunk), **fixed_params}
        return len(str(httpx.QueryParams(params)))

    chunks: list[tuple[str, ...]] = []
    for group in grouped:
        chunks.extend(_pack_variable_chunks(group, budget=url_budget_bytes, query_len=_query_len))
    return chunks


def place_slices(
    variables: Sequence[str],
    *,
    role: str,
    places: Sequence[str],
    facet_ids: Sequence[str] | None = None,
    time_period: Literal["LATEST"] | Sequence[str] | None = None,
    url_budget_bytes: int = _DEFAULT_URL_BUDGET_BYTES,
) -> list[tuple[str, ...]]:
    """Split `places` in order so a `/data` request naming one slice on `role`, the longest of
    `variables`, `facet_ids` and `time_period` fits the URL budget. `data()` packs the
    variables beside each slice, so a fetch over many places issues one or more requests per
    slice. A list that fits returns as one slice; a place that cannot fit beside one variable
    raises `SdmxQueryError`."""
    longest = max(variables, key=len)
    fixed_params = {"c[variableMeasured]": _or_list("variableMeasured", [longest])}
    if facet_ids:
        fixed_params["c[facetId]"] = _or_list("facetId", facet_ids)
    if time_period is not None:
        fixed_params["c[TIME_PERIOD]"] = _time_period_param(time_period)

    def _query_len(chunk: Sequence[str]) -> int:
        params = {**fixed_params, **_entity_params({role: chunk})}
        return len(str(httpx.QueryParams(params)))

    return _pack_variable_chunks(
        places, budget=url_budget_bytes, query_len=_query_len, label="place"
    )


def _pack_variable_chunks(
    variables: Sequence[str],
    *,
    budget: int,
    query_len: Callable[[Sequence[str]], int],
    label: str = "variable",
) -> list[tuple[str, ...]]:
    """Greedily pack items in input order against the encoded query budget.

    Shared by `/data`, `/availability`, place, and facet-id planning. `label`
    names the item in an oversized-item error. A single item over budget
    raises `SdmxQueryError`. A list that fits returns as one chunk.
    """
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    for variable in variables:
        if query_len([*current, variable]) <= budget:
            current.append(variable)
            continue
        if current:
            chunks.append(tuple(current))
        current = [variable]
        if query_len(current) > budget:
            raise SdmxQueryError(
                f"SDMX {label} {variable!r} alone exceeds the URL budget of "
                f"{budget} bytes once the request's other fixed params are "
                "included; no single request can carry it"
            )
    if current:
        chunks.append(tuple(current))
    return chunks


def _reproject_batch(batch: pa.RecordBatch, roles: tuple[str, ...]) -> pa.RecordBatch:
    """Rebuild a chunk's batch on the union schema, looking up columns by name.

    Missing roles become all-null columns. Name lookup preserves values when
    chunks declare roles in different orders.
    """
    schema = observation_schema(roles)
    names = frozenset(batch.schema.names)
    arrays = [
        batch.column(name) if name in names else pa.nulls(batch.num_rows, type=pa.string())
        for name in schema.names
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _rescale_chunk_cap_error(
    exc: SdmxCapExceededError,
    *,
    max_rows: int | None,
    max_bytes: int | None,
    rows_so_far: int,
    bytes_so_far: int,
) -> SdmxCapExceededError:
    """Translate a chunk-local cap error to the caller's cumulative limit and count.

    A chunk cap can only trip when the matching caller limit is set.
    """
    if exc.cap_kind == "rows":
        assert max_rows is not None
        cap, observed = max_rows, rows_so_far + exc.observed
        message = f"SDMX /data read exceeded the row cap of {cap} (observed {observed} rows before aborting)"
    else:
        assert max_bytes is not None
        cap, observed = max_bytes, bytes_so_far + exc.observed
        message = f"SDMX /data read exceeded the byte cap of {cap} (observed {observed} bytes before aborting)"
    return SdmxCapExceededError(message, cap_kind=exc.cap_kind, cap=cap, observed=observed)


def _rescale_worker_byte_cap_error(
    exc: SdmxCapExceededError, *, bytes_so_far: int
) -> SdmxCapExceededError:
    """Add already-landed bytes to a worker's byte-cap count.

    Under concurrency, unpopped chunks are excluded, so `observed` is a
    lower bound. Non-byte cap errors pass through unchanged.
    """
    if exc.cap_kind != "bytes":
        return exc
    cap = exc.cap
    observed = bytes_so_far + exc.observed
    message = f"SDMX /data or /availability read exceeded the byte cap of {cap} (observed {observed} bytes before aborting)"
    return SdmxCapExceededError(message, cap_kind="bytes", cap=cap, observed=observed)


class _CappedByteReader(io.RawIOBase):
    """Stream decoded response bytes and enforce `max_bytes` before parsing them.

    The counter includes headers and partial records. It is maintained here
    because `response.num_bytes_downloaded` misses pre-buffered bodies.
    `bytes_read` lets chunk mergers carry the total into the next response.

    With `require_line_end`, a non-empty body whose last byte is not a line
    feed raises `SdmxResponseError` at end of stream. The mixer ends every
    CSV record, the last included, with CRLF and sends no `Content-Length`,
    so a body that stops mid-record is truncated. The CSV parser alone
    accepts a record cut inside its last field as a complete row.
    """

    # TextIOWrapper requires `.name`. Streamed responses have no filesystem path.
    name = "<sdmx-data-stream>"

    def __init__(
        self,
        response: httpx.Response,
        *,
        max_bytes: int | None,
        deadline_guard: _DeadlineGuard | None = None,
        require_line_end: bool = False,
    ) -> None:
        self._chunks = response.iter_bytes()
        self._max_bytes = max_bytes
        self._deadline_guard = deadline_guard
        self._require_line_end = require_line_end
        self._last_byte = b""
        self._pending = b""
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: WriteableBuffer, /) -> int:
        if not self._pending:
            if self._deadline_guard is not None:
                self._deadline_guard.check()
            try:
                self._pending = next(self._chunks)
            except StopIteration:
                if self._deadline_guard is not None:
                    self._deadline_guard.check()
                if self._require_line_end and self._last_byte not in (b"", b"\n"):
                    raise SdmxResponseError(
                        f"SDMX /data response body ended mid-record after {self.bytes_read} "
                        "bytes; the body is truncated"
                    ) from None
                return 0
            except BaseException as exc:
                if self._deadline_guard is not None:
                    self._deadline_guard.raise_if_expired(exc)
                raise
            if self._deadline_guard is not None:
                self._deadline_guard.check()
            if self._pending:
                self._last_byte = self._pending[-1:]
            self.bytes_read += len(self._pending)
            if self._max_bytes is not None and self.bytes_read > self._max_bytes:
                raise SdmxCapExceededError(
                    f"SDMX /data read exceeded the byte cap of {self._max_bytes} "
                    f"(observed {self.bytes_read} bytes before aborting)",
                    cap_kind="bytes",
                    cap=self._max_bytes,
                    observed=self.bytes_read,
                )
        view = memoryview(buffer)
        chunk, self._pending = self._pending[: len(view)], self._pending[len(view) :]
        view[: len(chunk)] = chunk
        return len(chunk)


def _check_csv_content_type(response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "")
    if _CSV_CONTENT_TYPE_MARKER in content_type.lower():
        return
    raise SdmxResponseError(
        f"SDMX /data returned a 2xx response with content-type {content_type!r}, "
        "not SDMX-CSV -- refusing to parse this as an empty result",
        upstream_status=response.status_code,
    )


def _parse_availability_json_bytes(raw: bytes, *, upstream_status: int) -> dict:
    """Decode an already byte-capped `/availability` body and require a JSON object."""
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise SdmxResponseError(
            f"SDMX availability returned a 2xx response that isn't valid JSON: {exc}",
            upstream_status=upstream_status,
        ) from exc
    if not isinstance(body, dict):
        raise SdmxResponseError(
            f"SDMX availability returned a 2xx response whose JSON body is not an object: {body!r}",
            upstream_status=upstream_status,
        )
    return body


def _response_text(response: httpx.Response, *, raw_bytes: bytes) -> str:
    """Decode already-bounded response bytes as httpx would for `.text`."""
    return raw_bytes.decode(response.encoding or "utf-8", errors="replace")


def _upstream_message(response: httpx.Response, *, raw_bytes: bytes) -> str:
    """Return the transcoded gRPC message or the response body text as fallback."""
    try:
        body = json.loads(raw_bytes)
    except ValueError:
        return _response_text(response, raw_bytes=raw_bytes)
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return body["message"]
    return _response_text(response, raw_bytes=raw_bytes)


def _entity_params(entities: Mapping[str, EntityFilter]) -> dict[str, str]:
    params: dict[str, str] = {}
    for role, filt in entities.items():
        _validate_component_id(role, what="entity role")
        if isinstance(filt, ContainedIn):
            _check_filter_value(role, filt.parent)
            _check_filter_value(role, filt.child_type)
            params[f"c[{role}.containedInPlace+]"] = filt.parent
            params[f"c[{role}.typeOf]"] = filt.child_type
        else:
            if not filt:
                raise SdmxQueryError(f"entity role {role!r} has no values")
            params[f"c[{role}]"] = _or_list(role, filt)
    return params


def _or_list(label: str, values: Sequence[str]) -> str:
    _reject_bare_string(values, label=label)
    for value in values:
        _check_filter_value(label, value)
    return ",".join(values)


def _time_period_param(time_period: Literal["LATEST"] | Sequence[str]) -> str:
    """Render `LATEST` or a period OR-list as `c[TIME_PERIOD]`.

    Raises `SdmxQueryError` for other bare strings, empty lists, `LATEST`
    inside a list, or values containing reserved `+` or `,` characters.
    """
    if isinstance(time_period, str):
        if time_period != "LATEST":
            raise SdmxQueryError(
                f"time_period string value must be exactly 'LATEST', got {time_period!r}; "
                "pass a sequence of explicit period strings for anything else"
            )
        return time_period
    _reject_bare_string(time_period, label="time_period")
    if not time_period:
        raise SdmxQueryError("time_period sequence must not be empty")
    for value in time_period:
        if value == "LATEST":
            raise SdmxQueryError(
                "time_period sequence must not contain 'LATEST'; pass the literal "
                "string 'LATEST' alone instead of mixing it into a sequence"
            )
        _check_filter_value("TIME_PERIOD", value)
    return ",".join(time_period)


def _availability_time_period_param(time_period: str | Sequence[str]) -> str:
    """Render exact periods as `c[TIME_PERIOD]`. Sequences become deduplicated OR-lists.

    Raises `SdmxUnsupportedFilterError` for `LATEST`, range operators, empty
    sequences, or blank values. Raises `SdmxQueryError` for bytes or reserved
    `+` and `,` characters.
    """
    if isinstance(time_period, str):
        _check_availability_time_period_value(time_period)
        return time_period
    _reject_bare_string(time_period, label="time_period")
    if not time_period:
        raise SdmxUnsupportedFilterError(
            "SDMX availability time_period sequence must not be empty; pass "
            "an exact period string, a non-empty sequence of period "
            "strings, or omit time_period entirely for an unfiltered read",
            upstream_status=None,
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for value in time_period:
        _check_availability_time_period_value(value)
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return ",".join(deduped)


def _check_availability_time_period_value(value: str) -> None:
    """Validate one `/availability` period before HTTP.

    `LATEST` gets `SdmxUnsupportedFilterError` (mixer status 400). `ge:`,
    `le:`, `gt:`, `lt:`, and `ne:` prefixes get the same error (status 501).
    Empty or whitespace-only values also raise it. Reserved `+` or `,`
    raises `SdmxQueryError`.
    """
    if not value.strip():
        raise SdmxUnsupportedFilterError(
            f"SDMX availability time_period value {value!r} must not be "
            "empty or whitespace-only; pass an exact period string, a "
            "non-empty sequence of period strings, or omit time_period "
            "entirely for an unfiltered read",
            upstream_status=None,
        )
    if value == "LATEST":
        raise SdmxUnsupportedFilterError(
            "SDMX availability time_period does not support the literal "
            "'LATEST' -- live-verified: the mixer returns 400 and requires "
            "explicit dates; pass an exact period string or a sequence of "
            "them instead",
            upstream_status=None,
        )
    if value.startswith(_TIME_PERIOD_OPERATOR_PREFIXES):
        raise SdmxUnsupportedFilterError(
            f"SDMX availability time_period value {value!r} uses a "
            "range-operator prefix -- live-verified: the mixer returns 501 "
            "for range operators on /availability; pass an exact period, "
            "or an OR-list of exact periods, instead",
            upstream_status=None,
        )
    _check_filter_value("TIME_PERIOD", value)


def _reject_bare_string(values: object, *, label: str) -> None:
    """Reject bare strings and bytes where callers must pass a sequence of strings.

    A string satisfies `Sequence[str]` but iterates character by character,
    which would corrupt the comma-joined query value.
    """
    if isinstance(values, (str, bytes)):
        raise SdmxQueryError(
            f"{label!r} got a bare string/bytes ({values!r}); "
            "pass a list of strings, not a bare string"
        )


def _reject_duplicate_variables(variables: Sequence[str]) -> None:
    """Reject repeated variables before HTTP.

    Duplicates within a chunk collapse in the mixer. Duplicates split across
    chunks fetch and merge the same rows twice. Raises `SdmxQueryError` with
    the variable and repeat count.
    """
    seen: set[str] = set()
    for variable in variables:
        if variable in seen:
            raise SdmxQueryError(
                f"variables repeats {variable!r} ({variables.count(variable)} times); "
                "a variable must appear at most once per request"
            )
        seen.add(variable)


def _check_filter_value(label: str, value: str) -> None:
    if "+" in value:
        raise SdmxQueryError(
            f"SDMX filter value for {label!r} contains '+': {value!r}; the mixer "
            "parses '+' as an (unimplemented) AND-filter separator"
        )
    if "," in value:
        raise SdmxQueryError(
            f"SDMX filter value for {label!r} contains ',': {value!r}; this client "
            "uses ',' to join OR-list values, so a raw value can't contain one"
        )


def _validate_component_id(component_id: str, *, what: str) -> None:
    if not component_id or not all(ch.isalnum() or ch == "_" for ch in component_id):
        raise SdmxQueryError(f"invalid SDMX {what}: {component_id!r}")
