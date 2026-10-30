"""Test chunked `availability()` unions, limits, ordering, deadlines, and failures.

With no filters or period, 10-character A/B/C/D variables take 34, 47, 60,
or 73 query bytes. Each extra variable adds 13 bytes.
Responses are keyed by variable so concurrent reads merge in chunk-plan order.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Mapping

import httpx
import pytest

from dc_sdmx.client import SdmxClient, _DeadlineGuard, _ReadStoppedError
from dc_sdmx.errors import SdmxCapExceededError, SdmxHttpError
from dc_sdmx.results import parse_availability_json
from dfo_platform.timings import PhaseTimings, collecting

_BASE_URL = "https://datacommons.one.org/core/api"
_CONTENT_TYPE = "application/json"

_VAR_A = "AAAAAAAAAA"
_VAR_B = "BBBBBBBBBB"
_VAR_C = "CCCCCCCCCC"
_VAR_D = "DDDDDDDDDD"

_BUDGET_ONE_PER_CHUNK = 34  # one variable per chunk, no filters/time_period


def _body(component_values: dict[str, list[str]]) -> bytes:
    body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {
                            "keyValues": [
                                {"id": component, "values": list(values)}
                                for component, values in component_values.items()
                            ]
                        }
                    ]
                }
            ]
        }
    }
    return json.dumps(body).encode("utf-8")


def _client(transport: httpx.BaseTransport, *, url_budget_bytes: int) -> SdmxClient:
    return SdmxClient(base_url=_BASE_URL, transport=transport, url_budget_bytes=url_budget_bytes)


def _error_body(code: int, message: str) -> bytes:
    # This message avoids `_ERROR_SUBSTRING_MAP` entries and maps to plain `SdmxHttpError`.
    return json.dumps({"code": code, "message": message}).encode("utf-8")


class _KeyedErrorTransport(httpx.BaseTransport):
    """Return keyed error responses for concurrent chunk reads.

    Per-key delays control completion order; a lock protects request counts.
    """

    def __init__(
        self,
        statuses: Mapping[str, int],
        bodies: Mapping[str, bytes],
        *,
        delay: Mapping[str, float] | None = None,
    ) -> None:
        self._statuses = dict(statuses)
        self._bodies = dict(bodies)
        self._delay = dict(delay or {})
        self._lock = threading.Lock()
        self.requests: list[httpx.Request] = []
        self._in_flight = 0
        self.peak_concurrent = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = httpx.QueryParams(str(request.url.query, "utf-8"))["c[variableMeasured]"]
        assert key is not None, "request carried no c[variableMeasured]"
        with self._lock:
            self.requests.append(request)
            self._in_flight += 1
            self.peak_concurrent = max(self.peak_concurrent, self._in_flight)
        try:
            time.sleep(self._delay.get(key, 0.0))
            return httpx.Response(
                self._statuses[key],
                headers={"content-type": _CONTENT_TYPE},
                content=self._bodies[key],
            )
        finally:
            with self._lock:
                self._in_flight -= 1


# --- The union equals a single request's own result -------------------------


def test_three_chunk_union_equals_a_single_requests_result(keyed_concurrent_transport):
    # Three chunks, one variable each; financingProvider values overlap
    # across chunks (p2 appears in both chunk 1 and chunk 2).
    bodies = {
        _VAR_A: _body({"financingProvider": ["p1", "p2"]}),
        _VAR_B: _body({"financingProvider": ["p2", "p3"]}),
        _VAR_C: _body({"financingProvider": ["p4"]}),
    }
    transport = keyed_concurrent_transport(bodies, content_type=_CONTENT_TYPE)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    result = client.availability(variables=[_VAR_A, _VAR_B, _VAR_C], component="financingProvider")

    assert len(transport.requests) == 3

    # What a single request over the same combined result would parse to:
    # the mixer's own dedup, first-seen order across the whole variable list.
    single_request_body = _body({"financingProvider": ["p1", "p2", "p3", "p4"]})
    expected = parse_availability_json(json.loads(single_request_body))
    assert result.components == expected.components
    # Under concurrency, union order is chunk-plan order regardless of
    # which worker's request arrived first.
    assert result.components["financingProvider"] == ("p1", "p2", "p3", "p4")


def test_every_chunk_request_counts_on_the_callers_timings(keyed_concurrent_transport):
    """Chunk reads run on the client's own threads, which carry the caller's timings, so the
    call's `sdmx_requests` count holds every request sent."""
    bodies = {
        variable: _body({"financingProvider": ["p1"]}) for variable in (_VAR_A, _VAR_B, _VAR_C)
    }
    transport = keyed_concurrent_transport(bodies, content_type=_CONTENT_TYPE)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)
    timings = PhaseTimings()

    with collecting(timings):
        client.availability(variables=[_VAR_A, _VAR_B, _VAR_C], component="financingProvider")

    assert len(transport.requests) == 3
    assert "sdmx_requests=3" in timings.as_log_field().split()


def test_chunked_union_merges_every_component_the_response_carries(keyed_concurrent_transport):
    # Each chunk also echoes a second component (provenance) alongside the
    # requested one (variableMeasured is the caller's own axis here), and
    # The merge includes both response components.
    bodies = {
        _VAR_A: _body({"variableMeasured": ["v1"], "provenance": ["dc/base"]}),
        _VAR_B: _body({"variableMeasured": ["v2"], "provenance": ["dc/base", "dc/alt"]}),
    }
    transport = keyed_concurrent_transport(bodies, content_type=_CONTENT_TYPE)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    result = client.availability(variables=[_VAR_A, _VAR_B], component="variableMeasured")

    assert len(transport.requests) == 2
    assert result.components["variableMeasured"] == ("v1", "v2")
    assert result.components["provenance"] == ("dc/base", "dc/alt")


# --- max_values counts the running union -----------------------------------


def test_overlapping_chunks_do_not_double_count_toward_max_values(keyed_concurrent_transport):
    # Two overlapping 30-value chunks form a 30-value union; cap 30 accepts it.
    values = [f"country/E{i}" for i in range(30)]
    bodies = {
        _VAR_A: _body({"observationAbout": values}),
        _VAR_B: _body({"observationAbout": values}),
    }
    transport = keyed_concurrent_transport(bodies, content_type=_CONTENT_TYPE)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    result = client.availability(
        variables=[_VAR_A, _VAR_B], component="observationAbout", max_values=30
    )

    assert len(transport.requests) == 2
    assert len(result.components["observationAbout"]) == 30


# --- A cap bounds overread --------------------------------------------------


def test_max_values_cap_reports_identical_numbers_regardless_of_concurrency(
    keyed_concurrent_transport,
):
    # Chunk 1's 3 values meet the cap; chunk 2 adds 2, taking the union to 5.
    chunk1_values = ["country/USA", "country/KEN", "country/NGA"]
    chunk2_values = ["country/KEN", "country/NGA", "country/BRA", "country/CHN"]
    bodies = {
        _VAR_A: _body({"observationAbout": chunk1_values}),
        _VAR_B: _body({"observationAbout": chunk2_values}),
    }

    def _run(chunk_concurrency: int) -> SdmxCapExceededError:
        transport = keyed_concurrent_transport(bodies, content_type=_CONTENT_TYPE)
        client = SdmxClient(
            base_url=_BASE_URL,
            transport=transport,
            url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
            chunk_concurrency=chunk_concurrency,
        )
        with pytest.raises(SdmxCapExceededError) as excinfo:
            client.availability(
                variables=[_VAR_A, _VAR_B],
                component="observationAbout",
                max_values=3,
            )
        return excinfo.value

    sequential_error = _run(1)
    concurrent_error = _run(8)

    assert sequential_error.cap_kind == concurrent_error.cap_kind == "values"
    assert sequential_error.cap == concurrent_error.cap == 3
    # The running union is 3 + 2 = 5 at either concurrency level.
    assert sequential_error.observed == concurrent_error.observed == 5


def test_max_bytes_cap_reports_identical_numbers_regardless_of_concurrency(
    keyed_concurrent_transport,
):
    # Each worker gets the full byte cap, so chunk 2 trips inside its own read.
    # The error's observed count is cumulative on the sequential path.
    # Under concurrency, observed is a lower bound when chunk 1 completes later.
    small_body = _body({"observationAbout": ["country/USA"]})
    large_body = _body({"observationAbout": [f"country/E{i}" for i in range(200)]})
    bodies = {_VAR_A: small_body, _VAR_B: large_body}
    cap = len(small_body) + 10  # under chunk 1 alone; chunk 2's own body exceeds it

    def _run(chunk_concurrency: int) -> SdmxCapExceededError:
        transport = keyed_concurrent_transport(bodies, content_type=_CONTENT_TYPE)
        client = SdmxClient(
            base_url=_BASE_URL,
            transport=transport,
            url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
            chunk_concurrency=chunk_concurrency,
        )
        with pytest.raises(SdmxCapExceededError) as excinfo:
            client.availability(
                variables=[_VAR_A, _VAR_B],
                component="observationAbout",
                max_bytes=cap,
            )
        return excinfo.value

    sequential_error = _run(1)
    concurrent_error = _run(8)

    assert sequential_error.cap_kind == concurrent_error.cap_kind == "bytes"
    assert sequential_error.cap == concurrent_error.cap == cap
    # The cumulative pin: chunk 1's full body plus the tripping chunk's own
    # bytes at its abort (which, from this transport, is its whole body).
    expected_observed = len(small_body) + len(large_body)
    assert sequential_error.observed == expected_observed
    assert len(large_body) <= concurrent_error.observed <= expected_observed


def test_both_caps_trip_same_chunk_bytes_wins(keyed_concurrent_transport):
    # Chunk 2 exceeds the byte cap before the coordinator checks the value cap.
    # Observed is a lower bound when chunk 2 fails before chunk 1 completes.
    chunk1_values = [f"country/E{i}" for i in range(30)]
    chunk2_values = [f"country/E{i}" for i in range(30, 230)]
    chunk1_body = _body({"observationAbout": chunk1_values})
    chunk2_body = _body({"observationAbout": chunk2_values})
    bodies = {_VAR_A: chunk1_body, _VAR_B: chunk2_body}
    max_bytes = len(chunk1_body) + 10  # under chunk 1 alone; chunk 2's own body trips it
    max_values = 30  # chunk 1's union hits it exactly; chunk 2's NEW values trip it

    def _run(chunk_concurrency: int) -> SdmxCapExceededError:
        transport = keyed_concurrent_transport(bodies, content_type=_CONTENT_TYPE)
        client = SdmxClient(
            base_url=_BASE_URL,
            transport=transport,
            url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
            chunk_concurrency=chunk_concurrency,
        )
        with pytest.raises(SdmxCapExceededError) as excinfo:
            client.availability(
                variables=[_VAR_A, _VAR_B],
                component="observationAbout",
                max_values=max_values,
                max_bytes=max_bytes,
            )
        return excinfo.value

    sequential_error = _run(1)
    concurrent_error = _run(8)

    assert sequential_error.cap_kind == concurrent_error.cap_kind == "bytes"
    assert sequential_error.cap == concurrent_error.cap == max_bytes
    expected_observed = len(chunk1_body) + len(chunk2_body)
    assert sequential_error.observed == expected_observed
    assert len(chunk2_body) <= concurrent_error.observed <= expected_observed


def test_cap_trip_bounds_requests_by_concurrency_not_chunk_count(keyed_concurrent_transport):
    # Chunk 5 trips the cap in a 20-chunk read at concurrency 4. Chunks 0-5
    # must finish; at most 3 additional requests can already be in flight.
    n_chunks = 20
    concurrency = 4
    trip_index = 5
    variables = [f"V{i:09d}" for i in range(n_chunks)]

    def _small_body(i: int) -> bytes:
        return _body({"observationAbout": [f"country/E{i:02d}"]})

    small_len = len(_small_body(0))
    big_body = _body({"observationAbout": [f"country/E{i:02d}" for i in range(50)]})
    bodies = {
        var: (big_body if i == trip_index else _small_body(i)) for i, var in enumerate(variables)
    }
    transport = keyed_concurrent_transport(bodies, content_type=_CONTENT_TYPE)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=concurrency,
    )
    # Fits chunks 0-4 (5 small chunks) exactly; chunk 5's bulk body exceeds
    # the cap.
    cap = trip_index * small_len + 5

    with pytest.raises(SdmxCapExceededError) as excinfo:
        client.availability(variables=variables, component="observationAbout", max_bytes=cap)

    assert excinfo.value.cap_kind == "bytes"
    minimum_requests = trip_index + 1  # chunks 0..5 each completed before the trip fired
    maximum_requests = concurrency + trip_index  # + the initial batch's own overread bound
    assert minimum_requests <= len(transport.requests) <= maximum_requests
    assert len(transport.requests) < n_chunks


# --- A deadline expiring mid-read aborts every in-flight chunk -------------


class _KeyedBlockingStream(httpx.SyncByteStream):
    """Yield a header, then wait for close; 5s is a failure safety bound."""

    def __init__(self, header: bytes) -> None:
        self._header = header
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        yield self._header
        self.closed.wait(timeout=5)

    def close(self) -> None:
        self.closed.set()


class _KeyedBlockingTransport(httpx.BaseTransport):
    """Return one blocking stream per variable key for concurrent abort tests."""

    def __init__(self, *, header: bytes) -> None:
        self._header = header
        self._lock = threading.Lock()
        self.streams: dict[str, _KeyedBlockingStream] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = httpx.QueryParams(str(request.url.query, "utf-8"))["c[variableMeasured]"]
        assert key is not None
        stream = _KeyedBlockingStream(self._header)
        with self._lock:
            self.streams[key] = stream
        return httpx.Response(200, headers={"content-type": _CONTENT_TYPE}, stream=stream)


def test_deadline_aborts_every_in_flight_chunk_under_concurrency():
    # 2 chunks, chunk_concurrency=2: both reach the transport and block
    # after their own header at once. The shared _DeadlineGuard must abort
    # both blocked streams when the deadline fires; a guard tracking one
    # armed response would leave the other running past the deadline.
    header = _body({"observationAbout": ["country/USA"]})
    transport = _KeyedBlockingTransport(header=header)
    client = SdmxClient(
        base_url=_BASE_URL,
        timeout=10,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.availability(
            variables=[_VAR_A, _VAR_B], component="observationAbout", deadline=started + 0.08
        )

    assert time.monotonic() - started < 0.25
    assert set(transport.streams) == {_VAR_A, _VAR_B}
    assert all(stream.closed.is_set() for stream in transport.streams.values())


# --- The first completed chunk failure aborts the read ---------------------


def test_first_completing_chunk_failure_surfaces_when_earlier_chunk_stalls_unarmed():
    # A (plan-first) returns 500 after a 0.15s delay; B returns 404 first.
    # The first genuine failure aborts the read, so B's 404 surfaces while
    # A's late response is stopped before its status is examined.
    bodies = {
        _VAR_A: _error_body(500, "internal mixer failure"),
        _VAR_B: _error_body(404, "no such dataflow"),
    }
    transport = _KeyedErrorTransport({_VAR_A: 500, _VAR_B: 404}, bodies, delay={_VAR_A: 0.15})
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
    )
    started = time.monotonic()

    with pytest.raises(SdmxHttpError) as excinfo:
        client.availability(variables=[_VAR_A, _VAR_B], component="observationAbout")

    # B's 404, the only chunk in the failure registry (A's marker is
    # ignored), surfaces as soon as A's stalled future resolves.
    assert excinfo.value.upstream_status == 404
    assert "no such dataflow" in str(excinfo.value)
    assert "internal mixer failure" not in str(excinfo.value)
    assert time.monotonic() - started < 1.0


# --- A failed chunk aborts its siblings and stops the window ---------------


class _KeyedMixedTransport(httpx.BaseTransport):
    """Return a keyed error or blocking response for each concurrent chunk.

    Hold the error until the sibling stream is registered to guarantee overlap.
    """

    def __init__(
        self,
        *,
        error_key: str,
        error_status: int,
        error_body: bytes,
        block_key: str,
        block_header: bytes,
    ) -> None:
        self._error_key = error_key
        self._error_status = error_status
        self._error_body = error_body
        self._block_key = block_key
        self._block_header = block_header
        self._lock = threading.Lock()
        self.blocked_ready = threading.Event()
        self.requests: list[httpx.Request] = []
        self.streams: dict[str, _KeyedBlockingStream] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = httpx.QueryParams(str(request.url.query, "utf-8"))["c[variableMeasured]"]
        assert key is not None
        with self._lock:
            self.requests.append(request)
        if key == self._error_key:
            assert self.blocked_ready.wait(timeout=5), "the blocking stream was never registered"
            return httpx.Response(
                self._error_status,
                headers={"content-type": _CONTENT_TYPE},
                content=self._error_body,
            )
        assert key == self._block_key
        stream = _KeyedBlockingStream(self._block_header)
        with self._lock:
            self.streams[key] = stream
        self.blocked_ready.set()
        return httpx.Response(200, headers={"content-type": _CONTENT_TYPE}, stream=stream)


def test_failed_chunk_aborts_blocked_sibling():
    # A's 500 waits until B's stream is blocked. The abort must close B
    # before its 5s safety timeout.
    transport = _KeyedMixedTransport(
        error_key=_VAR_A,
        error_status=500,
        error_body=_error_body(500, "internal mixer failure"),
        block_key=_VAR_B,
        block_header=_body({"observationAbout": ["country/USA"]}),
    )
    client = SdmxClient(
        base_url=_BASE_URL,
        timeout=10,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
    )
    started = time.monotonic()

    with pytest.raises(SdmxHttpError) as excinfo:
        client.availability(variables=[_VAR_A, _VAR_B], component="observationAbout")

    assert excinfo.value.upstream_status == 500
    assert "internal mixer failure" in str(excinfo.value)
    assert time.monotonic() - started < 1.0
    assert set(transport.streams) == {_VAR_B}
    assert transport.streams[_VAR_B].closed.is_set()
    assert len(transport.requests) == 2


def test_queued_worker_never_issues_get_after_a_sibling_chunk_fails():
    # The coordinator acquires the sole slot in chunk order. A's 500 must
    # propagate before B is submitted, so only one request reaches transport.
    bodies = {
        _VAR_A: _error_body(500, "internal mixer failure"),
        _VAR_B: _error_body(404, "no such dataflow"),
    }
    transport = _KeyedErrorTransport({_VAR_A: 500, _VAR_B: 404}, bodies)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
        total_concurrency=1,
    )
    started = time.monotonic()

    with pytest.raises(SdmxHttpError) as excinfo:
        client.availability(variables=[_VAR_A, _VAR_B], component="observationAbout")

    assert excinfo.value.upstream_status == 500
    assert "internal mixer failure" in str(excinfo.value)
    assert "no such dataflow" not in str(excinfo.value)
    assert time.monotonic() - started < 1.0
    assert len(transport.requests) == 1


def test_later_chunk_failure_aborts_earlier_blocked_chunk_promptly():
    # A blocks after its header while B's 500 waits for A to register.
    # Completion-driven abort must close A and surface B's error promptly.
    transport = _KeyedMixedTransport(
        error_key=_VAR_B,
        error_status=500,
        error_body=_error_body(500, "internal mixer failure"),
        block_key=_VAR_A,
        block_header=_body({"observationAbout": ["country/USA"]}),
    )
    client = SdmxClient(
        base_url=_BASE_URL,
        timeout=10,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
    )
    started = time.monotonic()

    with pytest.raises(SdmxHttpError) as excinfo:
        client.availability(variables=[_VAR_A, _VAR_B], component="observationAbout")

    assert excinfo.value.upstream_status == 500
    assert "internal mixer failure" in str(excinfo.value)
    # The call returns before the 5s stream-safety timeout.
    assert time.monotonic() - started < 1.5
    assert set(transport.streams) == {_VAR_A}
    assert transport.streams[_VAR_A].closed.is_set()
    assert len(transport.requests) == 2


def test_abort_path_does_not_wait_for_an_uninterruptible_worker():
    # B (first in plan) waits 0.05s so A reaches transport, then fails.
    # A stalls 0.3s before headers; abort must return before it settles.
    bodies = {
        _VAR_A: _body({"observationAbout": ["country/USA"]}),
        _VAR_B: _error_body(404, "no such dataflow"),
    }
    transport = _KeyedErrorTransport(
        {_VAR_A: 200, _VAR_B: 404}, bodies, delay={_VAR_A: 0.3, _VAR_B: 0.05}
    )
    client = SdmxClient(
        base_url=_BASE_URL,
        timeout=10,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
    )
    started = time.monotonic()

    # Variables ordered so the failing chunk comes first in plan order.
    with pytest.raises(SdmxHttpError) as excinfo:
        client.availability(variables=[_VAR_B, _VAR_A], component="observationAbout")

    assert excinfo.value.upstream_status == 404
    assert "no such dataflow" in str(excinfo.value)
    assert time.monotonic() - started < 0.25
    assert len(transport.requests) == 2


def test_coordinator_slot_wait_wakes_on_a_landed_sibling():
    # With one slot, A's success wakes the coordinator to submit B and surface its 404.
    bodies = {
        _VAR_A: _body({"observationAbout": ["country/USA"]}),
        _VAR_B: _error_body(404, "no such dataflow"),
    }
    transport = _KeyedErrorTransport({_VAR_A: 200, _VAR_B: 404}, bodies)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
        total_concurrency=1,
    )
    started = time.monotonic()

    with pytest.raises(SdmxHttpError) as excinfo:
        client.availability(variables=[_VAR_A, _VAR_B], component="observationAbout")

    assert excinfo.value.upstream_status == 404
    assert "no such dataflow" in str(excinfo.value)
    assert "internal mixer failure" not in str(excinfo.value)
    assert time.monotonic() - started < 1.0
    assert len(transport.requests) == 2


class _ObservedLateStream(httpx.SyncByteStream):
    """A response that arrives only after its read already failed elsewhere."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.iterated = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        self.iterated.set()
        yield self._body

    def close(self) -> None:
        self.closed.set()


class _FailThenLateTransport(httpx.BaseTransport):
    """Hold plan-head A before headers, then fail plan-later B."""

    def __init__(self) -> None:
        self.late_entered = threading.Event()
        self.release_late = threading.Event()
        self.late_stream: _ObservedLateStream | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = httpx.QueryParams(str(request.url.query, "utf-8"))["c[variableMeasured]"]
        assert key is not None
        if key == _VAR_A:
            self.late_entered.set()
            assert self.release_late.wait(timeout=2)
            self.late_stream = _ObservedLateStream(_body({"observationAbout": ["country/USA"]}))
            return httpx.Response(
                200, headers={"content-type": _CONTENT_TYPE}, stream=self.late_stream
            )
        assert key == _VAR_B
        assert self.late_entered.wait(timeout=2)
        return httpx.Response(
            500,
            headers={"content-type": _CONTENT_TYPE},
            content=_error_body(500, "internal mixer failure"),
        )


def test_abort_retains_the_guard_and_defers_close_for_a_late_response():
    # B fails while A waits for headers. The call returns, but A keeps its
    # lease and guard until its late response is closed without reading.
    transport = _FailThenLateTransport()
    semaphore = threading.BoundedSemaphore(2)
    client = SdmxClient(
        base_url=_BASE_URL,
        timeout=10,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
        total_concurrency=semaphore,
    )

    with pytest.raises(SdmxHttpError, match="internal mixer failure"):
        client.availability(variables=[_VAR_A, _VAR_B], component="observationAbout")

    # The failed chunk releases one lease; the late worker holds the other until its future settles.
    assert semaphore.acquire(blocking=False)
    assert not semaphore.acquire(blocking=False)
    semaphore.release()
    client.close()
    assert not client._client.is_closed
    transport.release_late.set()
    deadline = time.monotonic() + 1
    while transport.late_stream is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert transport.late_stream is not None
    assert transport.late_stream.closed.wait(timeout=1)
    assert not transport.late_stream.iterated.is_set()
    deadline = time.monotonic() + 1
    while not client._client.is_closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client._client.is_closed


def test_close_keeps_an_already_admitted_availability_call_safe_while_queued(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": _CONTENT_TYPE},
            content=_body({"observationAbout": ["country/USA"]}),
        )
    )
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        total_concurrency=semaphore,
    )
    admitted = threading.Event()
    original_begin = client._begin_request

    def mark_admitted() -> None:
        admitted.set()
        original_begin()

    monkeypatch.setattr(client, "_begin_request", mark_admitted)
    outcome: list[BaseException] = []

    def read() -> None:
        try:
            client.availability(variables=[_VAR_A], component="observationAbout")
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=read)
    thread.start()
    assert admitted.wait(timeout=1)
    client.close()
    semaphore.release()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not outcome
    assert client._client.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        client.availability(variables=[_VAR_A], component="observationAbout")


def _stopped_availability_read(client: SdmxClient, guard: _DeadlineGuard) -> None:
    client._fetch_availability_chunk(
        variables=[_VAR_A],
        component="financingProvider",
        filters=None,
        time_period=None,
        max_bytes=None,
        guard=guard,
    )


def _stopped_data_read(client: SdmxClient, guard: _DeadlineGuard) -> None:
    with client._open_data_response(
        variables=[_VAR_A],
        entities={"observationAbout": ["country/USA"]},
        facet_ids=None,
        time_period=None,
        max_bytes=None,
        guard=guard,
    ):
        pass


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(_stopped_availability_read, id="availability"),
        pytest.param(_stopped_data_read, id="data"),
    ],
)
def test_a_stopped_read_that_takes_its_own_slot_leaves_the_slot_count_unchanged(read):
    """A read stopped before it gets a total-concurrency slot sends nothing and releases no
    slot, so the semaphore keeps its bound."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a stopped read sent {request.url}")

    client = SdmxClient(
        base_url=_BASE_URL, transport=httpx.MockTransport(refuse), total_concurrency=2
    )
    assert client._total_semaphore.acquire(blocking=False)
    guard = _DeadlineGuard(None)
    guard.abort()

    with pytest.raises(_ReadStoppedError):
        read(client, guard)

    # One of the two slots is still taken, so exactly one more can be acquired.
    assert [client._total_semaphore.acquire(blocking=False) for _ in range(2)] == [True, False]
