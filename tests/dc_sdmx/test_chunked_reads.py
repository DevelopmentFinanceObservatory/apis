"""Test variable chunking, cap accounting, role merging, and concurrency.

Query-string lengths use 10-character A/B/C variables and `observationAbout`:
    USA: 1 variable=72 bytes, 2=85, 3=98
    USA/KEN/NGA: 1=104 bytes, 2=117, 3=130

Drained chunked stream calls make two requests per chunk; `data()` makes one.
Sequential tests use arrival-ordered `SequencedTransport`; concurrent tests route by key.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future

import httpx
import pyarrow as pa
import pytest

from dc_sdmx.client import (
    SdmxClient,
    _ConcurrentChunkRead,
    _DeadlineGuard,
)
from dc_sdmx.errors import SdmxCapExceededError, SdmxHttpError, SdmxQueryError
from dc_sdmx.results import observation_columns

from .conftest import KeyedConcurrentTransport

_BASE_URL = "https://datacommons.one.org/core/api"
_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_ENTITIES = {"observationAbout": ["country/USA"]}
_BIG_ENTITIES = {"observationAbout": ["country/USA", "country/KEN", "country/NGA"]}


class _PollBoundaryFuture(Future[str]):
    """Complete just as the controller's first timed wait reports timeout."""

    def __init__(self) -> None:
        super().__init__()
        self._first_timed_result = True

    def result(self, timeout: float | None = None) -> str:
        if timeout is not None and self._first_timed_result:
            self._first_timed_result = False
            self.set_result("completed")
            raise TimeoutError
        return super().result(timeout)


def test_controller_result_returns_a_future_that_completes_at_the_poll_boundary():
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
    )
    controller = _ConcurrentChunkRead(client, _DeadlineGuard(), max_workers=1)
    try:
        assert controller.result(_PollBoundaryFuture()) == "completed"
    finally:
        controller.finish(aborted=False)


_VAR_A = "AAAAAAAAAA"
_VAR_B = "BBBBBBBBBB"
_VAR_C = "CCCCCCCCCC"

# Budgets pinned against the query-string lengths in the module docstring.
_BUDGET_SPLIT_2_1 = 85  # small entities: (A,B) then (C,)
_BUDGET_ENTITY_AWARE = 110  # small entities: 1 chunk; big entities: 3 chunks
_BUDGET_TOO_SMALL = 50  # below even a single variable's own query length
_BUDGET_ONE_PER_CHUNK = 72  # small entities: exactly 1 variable per chunk


def _header(role: str) -> bytes:
    return (
        f"STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,{role},unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
        "scalingFactor,facetId\r\n"
    ).encode()


def _two_role_header(role_a: str, role_b: str) -> bytes:
    return (
        f"STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,{role_a},{role_b},unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
        "scalingFactor,facetId\r\n"
    ).encode()


def _row(variable: str, entity: str, *, year: int = 2020) -> bytes:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{entity},Person,Census,P1Y,"
        f"dc/base,{year},1.50,0,facet-{variable}\r\n"
    ).encode()


def _two_role_row(variable: str, value_a: str, value_b: str, *, year: int = 2020) -> bytes:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{value_a},{value_b},Person,Census,"
        f"P1Y,dc/base,{year},1.50,0,facet-{variable}\r\n"
    ).encode()


def _client(transport: httpx.BaseTransport, *, url_budget_bytes: int) -> SdmxClient:
    # `SequencedTransport` maps responses by arrival order, so these tests use
    # one worker; concurrency cases use the key-routed transport below.
    return SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=url_budget_bytes,
        chunk_concurrency=1,
    )


def _variable_measured(request: httpx.Request) -> str:
    value = httpx.QueryParams(str(request.url.query, "utf-8"))["c[variableMeasured]"]
    assert value is not None
    return value


# --- One request stays one request -----------------------------------------


def test_one_request_stays_one_request(sequenced_transport):
    # A read that already fits the budget issues exactly one HTTP call with
    # the exact same URL the unchunked, single-request path always built.
    body = _header("observationAbout") + _row("Count_Person", "country/USA")
    transport = sequenced_transport([body])
    client = _client(transport, url_budget_bytes=7_800)

    result = client.data(variables=["Count_Person"], entities=_ENTITIES)

    assert len(transport.requests) == 1
    assert (
        str(transport.requests[0].url) == f"{_BASE_URL}/sdmx/v3/data/dataflow/DC/DF_OBS/1.0.0/*"
        "?c%5BvariableMeasured%5D=Count_Person"
        "&c%5BobservationAbout%5D=country%2FUSA"
    )
    assert len(result.rows) == 1
    assert transport.responses[0].is_closed


# --- Split boundary and entity-filter accounting ----------------------------


def test_split_boundary_on_built_url_length(sequenced_transport):
    # budget=85: (A,B) alone is 85 bytes, (A,B,C) is 98 -- so the split lands
    # between B and C, not on a round variable count.
    bodies = [
        _header("observationAbout") + _row(_VAR_A, "country/USA") + _row(_VAR_B, "country/USA"),
        _header("observationAbout") + _row(_VAR_C, "country/USA"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_SPLIT_2_1)

    result = client.data(variables=[_VAR_A, _VAR_B, _VAR_C], entities=_ENTITIES)

    # Each chunk gets one data request; no header pre-scan runs.
    assert len(transport.requests) == 2
    assert _variable_measured(transport.requests[0]) == f"{_VAR_A},{_VAR_B}"
    assert _variable_measured(transport.requests[1]) == _VAR_C
    assert [row.variable for row in result.rows] == [_VAR_A, _VAR_B, _VAR_C]
    assert all(response.is_closed for response in transport.responses)


def test_budget_accounts_for_entity_filter_size(sequenced_transport):
    # Same budget, same variables: small entities fit all three variables in
    # one request; the larger entity filter alone pushes even two variables
    # over budget, forcing one variable per request.
    small_body = _header("observationAbout") + b"".join(
        _row(v, "country/USA") for v in (_VAR_A, _VAR_B, _VAR_C)
    )
    small_transport = sequenced_transport([small_body])
    small_client = _client(small_transport, url_budget_bytes=_BUDGET_ENTITY_AWARE)
    small_client.data(variables=[_VAR_A, _VAR_B, _VAR_C], entities=_ENTITIES)
    assert len(small_transport.requests) == 1
    assert _variable_measured(small_transport.requests[0]) == f"{_VAR_A},{_VAR_B},{_VAR_C}"

    big_bodies = [
        _header("observationAbout") + _row(v, "country/USA") for v in (_VAR_A, _VAR_B, _VAR_C)
    ]
    big_transport = sequenced_transport(big_bodies)
    big_client = _client(big_transport, url_budget_bytes=_BUDGET_ENTITY_AWARE)
    big_client.data(variables=[_VAR_A, _VAR_B, _VAR_C], entities=_BIG_ENTITIES)
    # Each chunk gets one data request; no header pre-scan runs.
    assert len(big_transport.requests) == 3
    assert [_variable_measured(r) for r in big_transport.requests] == [_VAR_A, _VAR_B, _VAR_C]


# --- data() reads each chunk once ----------------------------------------


def test_data_issues_exactly_len_chunks_requests(sequenced_transport):
    # Each of the 3 chunks gets one request. `data()` buffers rows and builds
    # the role union as it reads, so it needs no header pre-scan.
    var_c = "DDDDDDDDDD"
    bodies = [
        _header("observationAbout") + _row(_VAR_A, "country/USA"),
        _header("observationAbout") + _row(_VAR_B, "country/USA"),
        _header("observationAbout") + _row(var_c, "country/USA"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    result = client.data(variables=[_VAR_A, _VAR_B, var_c], entities=_ENTITIES)

    assert len(transport.requests) == 3
    assert [row.variable for row in result.rows] == [_VAR_A, _VAR_B, var_c]
    assert all(response.is_closed for response in transport.responses)


# --- Oversized variables are refused before HTTP ---------------------------


def test_single_oversized_variable_is_refused_before_any_http_call(sequenced_transport):
    transport = sequenced_transport([])
    client = _client(transport, url_budget_bytes=_BUDGET_TOO_SMALL)

    with pytest.raises(SdmxQueryError, match=rf"{_VAR_A}.*budget of {_BUDGET_TOO_SMALL}"):
        client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES)

    assert transport.requests == []


def test_oversized_variable_past_the_first_chunk_is_also_refused(sequenced_transport):
    # A and B (short names) together fit under budget=90 (85 bytes) and form
    # chunk 1; a much longer third variable, which alone already exceeds the
    # budget (122 bytes), is refused when it starts chunk 2 -- the refusal
    # The planner rejects oversized variables in later chunks too.
    long_variable = "C" * 60
    transport = sequenced_transport([])
    client = _client(transport, url_budget_bytes=90)

    with pytest.raises(SdmxQueryError, match=rf"{long_variable!r}.*budget of 90"):
        client.data(variables=[_VAR_A, _VAR_B, long_variable], entities=_ENTITIES)

    assert transport.requests == []


# --- Duplicate variables are refused --------------------------------------


def test_duplicate_variable_that_would_split_across_chunks_is_refused(sequenced_transport):
    # budget=85: (A,B) is 85 bytes; adding a second A would make 98, over
    # budget, so the greedy packer puts A,B in chunk 1 and A alone in
    # chunk 2. Without the refusal, A's rows would be fetched and merged
    # twice (issue #849). The duplicate is refused when the chunks are
    # planned, before either request is issued.
    transport = sequenced_transport([])
    client = _client(transport, url_budget_bytes=_BUDGET_SPLIT_2_1)

    with pytest.raises(SdmxQueryError, match=rf"repeats {_VAR_A!r} \(2 times\)"):
        client.data(variables=[_VAR_A, _VAR_B, _VAR_A], entities=_ENTITIES)

    assert transport.requests == []


def test_duplicate_variable_within_one_chunk_is_refused(sequenced_transport):
    # Both copies fit one request, but the mixer returns one series for A.
    # The planner rejects the duplicate.
    transport = sequenced_transport([])
    client = _client(transport, url_budget_bytes=_BUDGET_ENTITY_AWARE)

    with pytest.raises(SdmxQueryError, match=rf"repeats {_VAR_A!r} \(2 times\)"):
        client.data(variables=[_VAR_A, _VAR_A], entities=_ENTITIES)

    assert transport.requests == []


# --- Caps span the whole chunked read -------------------------------------


def test_max_rows_spans_chunks(sequenced_transport):
    bodies = [
        _header("observationAbout")
        + b"".join(_row(_VAR_A, "country/USA", year=y) for y in range(3)),
        _header("observationAbout")
        + b"".join(_row(_VAR_B, "country/USA", year=y) for y in range(3)),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with pytest.raises(SdmxCapExceededError) as excinfo:
        client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, max_rows=4)

    err = excinfo.value
    assert err.cap_kind == "rows"
    assert err.cap == 4
    # 3 rows from chunk 1 (under budget) + trips 1 row into chunk 2's excess,
    # the same off-by-one point a single 6-row request would trip at.
    assert err.observed == 5
    # Each chunk gets one data request; no header pre-scan runs.
    assert len(transport.requests) == 2
    assert all(response.is_closed for response in transport.responses)


def test_max_bytes_spans_chunks(sequenced_transport):
    chunk1_body = _header("observationAbout") + b"".join(
        _row(_VAR_A, "country/USA", year=y) for y in range(3)
    )
    chunk2_body = _header("observationAbout") + b"".join(
        _row(_VAR_B, "country/USA", year=y) for y in range(3)
    )
    bodies = [chunk1_body, chunk2_body]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    cap = len(chunk1_body) + len(_header("observationAbout")) // 2
    with pytest.raises(SdmxCapExceededError) as excinfo:
        client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, max_bytes=cap)

    err = excinfo.value
    assert err.cap_kind == "bytes"
    assert err.cap == cap
    # Tripped partway into chunk 2, past everything chunk 1 alone carried.
    assert err.observed > cap
    assert err.observed > len(chunk1_body)


def test_prescan_bytes_count_toward_max_bytes(sequenced_transport):
    # The header pre-scan consumes bytes before yielding rows, so its bytes
    # count toward max_bytes. Here chunk 2's header trips the cap.
    header = _header("observationAbout")
    bodies = [header, header]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    cap = len(header) + len(header) // 2
    with (
        pytest.raises(SdmxCapExceededError) as excinfo,
        client.stream_data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, max_bytes=cap) as (
            _roles,
            _rows,
        ),
    ):
        pass

    err = excinfo.value
    assert err.cap_kind == "bytes"
    assert err.cap == cap
    # Chunk 1's header (fully counted) plus part of chunk 2's, past the cap.
    assert err.observed > cap
    # Only the two header pre-scans were issued -- the cap tripped before
    # either chunk's real, row-carrying request was ever made.
    assert len(transport.requests) == 2


def test_max_rows_cap_not_reached_lets_a_chunked_read_complete(sequenced_transport):
    bodies = [
        _header("observationAbout") + _row(_VAR_A, "country/USA"),
        _header("observationAbout") + _row(_VAR_B, "country/USA"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    result = client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, max_rows=2)

    assert [row.variable for row in result.rows] == [_VAR_A, _VAR_B]


# --- Deadline bounds the whole chunked sequence -----------------------------


class _BlockingStream(httpx.SyncByteStream):
    """Yield one chunk, then wait for close; 5s is a failure safety bound."""

    def __init__(self, header: bytes) -> None:
        self.header = header
        self.closed = threading.Event()
        self._entered: list[bool] = []

    def block_until_closed(self, timeout: float = 5.0) -> bool:
        """Record entry and wait for the client to close the stream."""
        self._entered.append(True)
        return self.closed.wait(timeout=timeout)

    @property
    def entered(self) -> int:
        return len(self._entered)

    def __iter__(self) -> Iterator[bytes]:
        yield self.header
        self.block_until_closed()

    def close(self) -> None:
        self.closed.set()


class _ChunkSpanningBlockingTransport(httpx.BaseTransport):
    """Block the request at `block_at_index` after its header."""

    def __init__(
        self, *, fast_bodies: list[bytes], block_at_index: int, block_header: bytes
    ) -> None:
        self._fast_bodies = fast_bodies
        self._block_at_index = block_at_index
        self._block_header = block_header
        self.requests: list[httpx.Request] = []
        self.blocked_stream: _BlockingStream | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        index = len(self.requests)
        self.requests.append(request)
        if index == self._block_at_index:
            self.blocked_stream = _BlockingStream(self._block_header)
            return httpx.Response(
                200, headers={"content-type": _CONTENT_TYPE}, stream=self.blocked_stream
            )
        return httpx.Response(
            200, headers={"content-type": _CONTENT_TYPE}, content=self._fast_bodies[index]
        )


def test_deadline_fires_while_blocked_on_a_later_chunk():
    # Chunk 1 completes; chunk 2 blocks so the deadline fires mid-read.
    header = _header("observationAbout")
    fast_bodies = [header + _row(_VAR_A, "country/USA")]
    transport = _ChunkSpanningBlockingTransport(
        fast_bodies=fast_bodies, block_at_index=1, block_header=header
    )
    # This transport assigns bodies by arrival index, so use one worker.
    client = SdmxClient(
        base_url=_BASE_URL,
        timeout=10,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=1,
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, deadline=started + 0.08)

    assert time.monotonic() - started < 0.25
    assert len(transport.requests) == 2
    assert transport.blocked_stream is not None
    # The entry count and close event prove the abort closed chunk 2 after its header arrived.
    assert transport.blocked_stream.entered == 1
    assert transport.blocked_stream.closed.is_set()


# --- Role reconciliation: union in first-seen order -------------------------


def test_role_union_first_seen_order_on_rows(sequenced_transport):
    # Chunk 1 declares `donor`; chunk 2 declares `recipient`. Only chunking exposes this shape.
    chunk1_body = _header("donor") + _row(_VAR_A, "country/USA")
    chunk2_body = _header("recipient") + _row(_VAR_B, "country/KEN")
    bodies = [chunk1_body, chunk2_body]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    result = client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES)

    assert result.roles == ("donor", "recipient")
    assert len(result.rows) == 2
    assert result.rows[0].variable == _VAR_A
    assert result.rows[0].role_values == {"donor": "country/USA"}
    assert result.rows[1].variable == _VAR_B
    assert result.rows[1].role_values == {"recipient": "country/KEN"}


def test_role_reprojection_on_batches_is_by_name_not_position(sequenced_transport):
    # Chunk 3 reverses the donor/recipient header order; reprojection must map columns by name.
    var_c = "DDDDDDDDDD"
    chunk1_body = _header("donor") + _row(_VAR_A, "country/USA")
    chunk2_body = _header("recipient") + _row(_VAR_B, "country/KEN")
    chunk3_body = _two_role_header("recipient", "donor") + _two_role_row(
        var_c, "country/NGA", "country/FRA"
    )
    bodies = [
        _header("donor"),
        _header("recipient"),
        _two_role_header("recipient", "donor"),
        chunk1_body,
        chunk2_body,
        chunk3_body,
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with client.stream_data_batches(variables=[_VAR_A, _VAR_B, var_c], entities=_ENTITIES) as (
        roles,
        batches,
    ):
        table = pa.Table.from_batches(
            list(batches),
            schema=pa.schema([(name, pa.string()) for name in observation_columns(roles)]),
        )

    assert roles == ("donor", "recipient")
    assert table.column_names == list(observation_columns(("donor", "recipient")))
    rows = table.to_pylist()
    by_variable = {row["variableMeasured"]: row for row in rows}
    assert by_variable[_VAR_A]["donor"] == "country/USA"
    assert by_variable[_VAR_A]["recipient"] is None
    assert by_variable[_VAR_B]["recipient"] == "country/KEN"
    assert by_variable[_VAR_B]["donor"] is None
    # Map chunk 3's values to columns by role name.
    assert by_variable[var_c]["donor"] == "country/FRA"
    assert by_variable[var_c]["recipient"] == "country/NGA"


# --- Early stop closes every response opened so far -------------------------


def test_early_stop_closes_every_response_and_issues_no_further_chunk_requests(sequenced_transport):
    var_c = "DDDDDDDDDD"
    header = _header("observationAbout")
    bodies = [
        header,  # chunk 1 pre-scan
        header,  # chunk 2 pre-scan
        header,  # chunk 3 pre-scan
        header
        + _row(_VAR_A, "country/USA")
        + _row(_VAR_A, "country/USA", year=2021),  # real chunk 1
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with client.stream_data(variables=[_VAR_A, _VAR_B, var_c], entities=_ENTITIES) as (
        _roles,
        rows,
    ):
        first = next(rows)
        assert first.variable == _VAR_A

    # All headers are pre-scanned, but only chunk 1's data request is issued.
    assert len(transport.requests) == 4
    assert all(response.is_closed for response in transport.responses)


# --- True concurrency: chunk_concurrency > 1 -------------------------------


class _EnteredTrackingTransport(KeyedConcurrentTransport):
    """Add per-key arrival events and a signal for two in-flight requests."""

    def __init__(self, bodies, **kwargs: object) -> None:
        super().__init__(bodies, **kwargs)
        self.entered = {key: threading.Event() for key in bodies}
        self.reached_two = threading.Event()
        self._track_lock = threading.Lock()
        self._tracked_in_flight = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = httpx.QueryParams(str(request.url.query, "utf-8"))["c[variableMeasured]"]
        with self._track_lock:
            self._tracked_in_flight += 1
            if self._tracked_in_flight >= 2:
                self.reached_two.set()
        self.entered[key].set()
        try:
            return super().handle_request(request)
        finally:
            with self._track_lock:
                self._tracked_in_flight -= 1


def test_concurrency_is_bounded_and_issues_exactly_len_chunks_requests():
    # Three chunks prove the pool reaches, but never exceeds, concurrency 2.
    var_c = "DDDDDDDDDD"
    bodies = {
        _VAR_A: _header("observationAbout") + _row(_VAR_A, "country/USA"),
        _VAR_B: _header("observationAbout") + _row(_VAR_B, "country/USA"),
        var_c: _header("observationAbout") + _row(var_c, "country/USA"),
    }
    release = {_VAR_A: threading.Event(), _VAR_B: threading.Event(), var_c: threading.Event()}
    transport = _EnteredTrackingTransport(bodies, release=release)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
    )

    results: list[object] = []
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            results.append(client.data(variables=[_VAR_A, _VAR_B, var_c], entities=_ENTITIES))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=_run)
    worker.start()
    try:
        # Wait until chunks A and B are both in flight before releasing either.
        assert transport.reached_two.wait(timeout=5)
        assert transport.entered[_VAR_A].is_set()
        assert transport.entered[_VAR_B].is_set()
        release[_VAR_A].set()
        release[_VAR_B].set()
        release[var_c].set()
    finally:
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []

    assert len(transport.requests) == 3
    assert transport.peak_concurrent == 2
    (result,) = results
    assert [row.variable for row in result.rows] == [_VAR_A, _VAR_B, var_c]


def test_out_of_order_completion_preserves_chunk_plan_order():
    # Release C, B, then A to prove results retain chunk-plan order.
    var_c = "DDDDDDDDDD"
    bodies = {
        _VAR_A: _header("donor") + _row(_VAR_A, "country/USA"),
        _VAR_B: _header("recipient") + _row(_VAR_B, "country/KEN"),
        var_c: _header("observationAbout") + _row(var_c, "country/NGA"),
    }
    release = {_VAR_A: threading.Event(), _VAR_B: threading.Event(), var_c: threading.Event()}
    transport = _EnteredTrackingTransport(bodies, release=release)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=3,
    )

    results: list[object] = []

    def _run() -> None:
        results.append(client.data(variables=[_VAR_A, _VAR_B, var_c], entities=_ENTITIES))

    worker = threading.Thread(target=_run)
    worker.start()
    try:
        # Wait for all three requests, then release them in reverse order.
        assert transport.reached_two.wait(timeout=5)
        for key in (_VAR_A, _VAR_B, var_c):
            assert transport.entered[key].wait(timeout=5), f"chunk {key!r} never arrived"
        release[var_c].set()
        release[_VAR_B].set()
        release[_VAR_A].set()
    finally:
        worker.join(timeout=5)
    assert not worker.is_alive()

    (result,) = results
    assert result.roles == ("donor", "recipient", "observationAbout")
    assert [row.variable for row in result.rows] == [_VAR_A, _VAR_B, var_c]
    assert result.rows[0].role_values == {"donor": "country/USA"}
    assert result.rows[1].role_values == {"recipient": "country/KEN"}
    assert result.rows[2].role_values == {"observationAbout": "country/NGA"}


def test_max_rows_cap_reports_identical_numbers_regardless_of_concurrency(
    keyed_concurrent_transport,
):
    # The post-fetch row-cap pass runs in chunk order at every concurrency level.
    bodies = {
        _VAR_A: _header("observationAbout")
        + b"".join(_row(_VAR_A, "country/USA", year=y) for y in range(3)),
        _VAR_B: _header("observationAbout")
        + b"".join(_row(_VAR_B, "country/USA", year=y) for y in range(3)),
    }

    def _run(chunk_concurrency: int) -> SdmxCapExceededError:
        transport = keyed_concurrent_transport(bodies)
        client = SdmxClient(
            base_url=_BASE_URL,
            transport=transport,
            url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
            chunk_concurrency=chunk_concurrency,
        )
        with pytest.raises(SdmxCapExceededError) as excinfo:
            client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, max_rows=4)
        return excinfo.value

    sequential_error = _run(1)
    concurrent_error = _run(8)

    assert sequential_error.cap_kind == concurrent_error.cap_kind == "rows"
    assert sequential_error.cap == concurrent_error.cap == 4
    # Chunk 1 contributes 3 rows; the first row from chunk 2 trips cap 4.
    assert sequential_error.observed == concurrent_error.observed == 5


def test_max_bytes_cap_reports_identical_numbers_regardless_of_concurrency(
    keyed_concurrent_transport,
):
    # Each worker gets the full byte cap; these bodies fit it individually.
    # Only the coordinator's cumulative check trips, so observed is their sum.
    chunk1_body = _header("observationAbout") + b"".join(
        _row(_VAR_A, "country/USA", year=y) for y in range(3)
    )
    chunk2_body = _header("observationAbout") + b"".join(
        _row(_VAR_B, "country/USA", year=y) for y in range(3)
    )
    bodies = {_VAR_A: chunk1_body, _VAR_B: chunk2_body}
    cap = len(chunk1_body) + 10  # under chunk 1 alone; the two-body sum trips it
    # Sanity-pin the mechanism the comment above depends on: chunk 2's own
    # body stays under the per-worker ceiling, so the trip is cumulative.
    assert len(chunk2_body) <= cap

    def _run(chunk_concurrency: int) -> SdmxCapExceededError:
        transport = keyed_concurrent_transport(bodies)
        client = SdmxClient(
            base_url=_BASE_URL,
            transport=transport,
            url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
            chunk_concurrency=chunk_concurrency,
        )
        with pytest.raises(SdmxCapExceededError) as excinfo:
            client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, max_bytes=cap)
        return excinfo.value

    sequential_error = _run(1)
    concurrent_error = _run(8)

    assert sequential_error.cap_kind == concurrent_error.cap_kind == "bytes"
    assert sequential_error.cap == concurrent_error.cap == cap
    expected_observed = len(chunk1_body) + len(chunk2_body)
    assert sequential_error.observed == concurrent_error.observed == expected_observed


def test_cap_trip_bounds_requests_by_concurrency_not_chunk_count(keyed_concurrent_transport):
    # Chunk 5's bulk body is held until chunks 0-4 have been answered, so all six requests
    # reach the transport before the trip. Each of chunks 0-4 that lands frees a slot for one
    # more chunk (6-8), and chunk 5's landing raises before it submits another, so 6-9 of 20
    # reach the transport.
    n_chunks = 20
    concurrency = 4
    trip_index = 5
    variables = [f"V{i:09d}" for i in range(n_chunks)]

    def _small_body(i: int) -> bytes:
        return _header("observationAbout") + _row(variables[i], "country/USA")

    small_len = len(_small_body(0))
    big_body = _header("observationAbout") + b"".join(
        _row(variables[trip_index], "country/USA", year=y) for y in range(50)
    )
    bodies = {
        var: (big_body if i == trip_index else _small_body(i)) for i, var in enumerate(variables)
    }
    trip_released = threading.Event()
    transport = keyed_concurrent_transport(bodies, release={variables[trip_index]: trip_released})
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=concurrency,
    )
    # Fits chunks 0-4 (5 small chunks) exactly; chunk 5's bulk body blows it.
    cap = trip_index * small_len + 5

    def release_trip_after_earlier_chunks() -> None:
        if transport.wait_completed(trip_index):
            trip_released.set()

    releaser = threading.Thread(target=release_trip_after_earlier_chunks)
    releaser.start()
    try:
        with pytest.raises(SdmxCapExceededError) as excinfo:
            client.data(variables=variables, entities=_ENTITIES, max_bytes=cap)
    finally:
        trip_released.set()
        releaser.join()

    assert excinfo.value.cap_kind == "bytes"
    minimum_requests = trip_index + 1  # chunks 0..5 each reached the transport before the trip
    maximum_requests = concurrency + trip_index  # + the initial batch's own overread bound
    assert minimum_requests <= len(transport.requests) <= maximum_requests
    assert len(transport.requests) < n_chunks


def test_both_caps_trip_same_chunk_bytes_wins(keyed_concurrent_transport):
    # Same bodies/budget as test_max_rows_cap_... and test_max_bytes_cap_...
    # above, but with BOTH caps set so they'd each trip on their own. Neither
    # chunk body exceeds `max_bytes` individually (same bodies as the
    # bytes-only test), so no worker trips; the coordinator checks the
    # cumulative byte total -- which chunk 2's landing takes over the cap --
    # before this chunk's rows are weighed against `max_rows`, the same
    # interleaving the sequential path gets from `_CappedByteReader` sitting
    # in front of the CSV parser. A "rows checked first" precedence would
    # report "rows" here instead; bytes trips first in every run.
    chunk1_body = _header("observationAbout") + b"".join(
        _row(_VAR_A, "country/USA", year=y) for y in range(3)
    )
    chunk2_body = _header("observationAbout") + b"".join(
        _row(_VAR_B, "country/USA", year=y) for y in range(3)
    )
    bodies = {_VAR_A: chunk1_body, _VAR_B: chunk2_body}
    max_bytes = len(chunk1_body) + 10  # under chunk 1 alone; the two-body sum trips it
    max_rows = 4  # chunk 2's rows would trip it if bytes were checked second
    assert len(chunk2_body) <= max_bytes  # Each body fits; the combined total trips the cap.

    transport = keyed_concurrent_transport(bodies)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=2,
    )

    with pytest.raises(SdmxCapExceededError) as excinfo:
        client.data(
            variables=[_VAR_A, _VAR_B],
            entities=_ENTITIES,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )

    assert excinfo.value.cap_kind == "bytes"


class _KeyedBlockingStream(httpx.SyncByteStream):
    """Yield a header, then wait for close; 5s is a test safety bound."""

    def __init__(self, header: bytes) -> None:
        self.header = header
        self.closed = threading.Event()
        self._entered: list[bool] = []

    def block_until_closed(self, timeout: float = 5.0) -> bool:
        """Record entry and wait for the client to close the stream."""
        self._entered.append(True)
        return self.closed.wait(timeout=timeout)

    @property
    def entered(self) -> int:
        return len(self._entered)

    def __iter__(self) -> Iterator[bytes]:
        yield self.header
        self.block_until_closed()

    def close(self) -> None:
        self.closed.set()


class _KeyedBlockingTransport(httpx.BaseTransport):
    """Return one blocking stream per variable key for concurrent deadline tests."""

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
    # Both chunk streams block after their headers; the shared guard must close both at deadline.
    header = _header("observationAbout")
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
        client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, deadline=started + 0.08)

    assert time.monotonic() - started < 0.25
    assert set(transport.streams) == {_VAR_A, _VAR_B}
    # Both streams were entered and both were then closed by the abort, so
    # the guard reached every in-flight chunk rather than one of them.
    assert all(stream.entered == 1 for stream in transport.streams.values())
    assert all(stream.closed.is_set() for stream in transport.streams.values())


class _KeyedMixedTransport(httpx.BaseTransport):
    """Route keys to an error response or a stream blocked after its header.

    Hold the error until the stream registers to guarantee overlap.
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
        self.blocked_stream: _KeyedBlockingStream | None = None

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
            self.blocked_stream = stream
        self.blocked_ready.set()
        return httpx.Response(200, headers={"content-type": _CONTENT_TYPE}, stream=stream)


def test_failed_chunk_aborts_blocked_sibling_under_concurrency():
    # A's 500 waits for B's stream to register, so both chunks are in flight.
    # Abort must close B and surface A's error before the 5s safety timeout.
    transport = _KeyedMixedTransport(
        error_key=_VAR_A,
        error_status=500,
        error_body=json.dumps({"code": 500, "message": "internal mixer failure"}).encode("utf-8"),
        block_key=_VAR_B,
        block_header=_header("observationAbout"),
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
        client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES)

    assert excinfo.value.upstream_status == 500
    assert "internal mixer failure" in str(excinfo.value)
    assert time.monotonic() - started < 1.0
    assert transport.blocked_stream is not None
    assert transport.blocked_stream.closed.is_set()
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
        self.late_stream_ready = threading.Event()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = _variable_measured(request)
        if key == _VAR_A:
            self.late_entered.set()
            assert self.release_late.wait(timeout=2)
            self.late_stream = _ObservedLateStream(
                _header("observationAbout") + _row(_VAR_A, "country/USA")
            )
            self.late_stream_ready.set()
            return httpx.Response(
                200, headers={"content-type": _CONTENT_TYPE}, stream=self.late_stream
            )
        assert key == _VAR_B
        assert self.late_entered.wait(timeout=2)
        return httpx.Response(
            500,
            headers={"content-type": _CONTENT_TYPE},
            content=json.dumps({"code": 500, "message": "internal mixer failure"}).encode(),
        )


def test_abort_retains_the_guard_and_defers_close_for_a_late_response(
    monkeypatch: pytest.MonkeyPatch,
):
    # A is stalled before headers while later plan chunk B fails. The public
    # call must return before A is released, but A's worker still owns its slot and
    # guard. Its later response is closed before any body bytes are read,
    # and close() waits to physically close httpx until that worker settles.
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

    # Signal the request's completion to know when the late worker closes httpx.
    settled = threading.Event()
    original_end_request = client._end_request

    def _end_request_and_signal() -> None:
        original_end_request()
        settled.set()

    monkeypatch.setattr(client, "_end_request", _end_request_and_signal)

    with pytest.raises(SdmxHttpError, match="internal mixer failure"):
        client.data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES)

    # The failed chunk releases one lease; the late worker holds the other until its future settles.
    assert semaphore.acquire(blocking=False)
    assert not semaphore.acquire(blocking=False)
    semaphore.release()
    client.close()
    assert not client._client.is_closed
    transport.release_late.set()
    assert transport.late_stream_ready.wait(timeout=2)
    assert transport.late_stream is not None
    assert transport.late_stream.closed.wait(timeout=1)
    assert not transport.late_stream.iterated.is_set()
    assert settled.wait(timeout=2)
    assert client._client.is_closed


def test_close_keeps_an_already_admitted_data_call_safe_while_queued(
    sequenced_transport, monkeypatch
):
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    transport = sequenced_transport([_header("observationAbout") + _row(_VAR_A, "country/USA")])
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
            client.data(variables=[_VAR_A], entities=_ENTITIES)
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
        client.data(variables=[_VAR_A], entities=_ENTITIES)


def test_single_chunk_does_not_touch_the_pool(sequenced_transport, monkeypatch):
    # A single-chunk data() call routes through stream_data() directly (see
    # data()'s own docstring) and must never construct a thread pool, even
    # though _buffered_chunked_data now uses one for the multi-chunk case.
    body = _header("observationAbout") + _row("Count_Person", "country/USA")
    transport = sequenced_transport([body])
    client = _client(transport, url_budget_bytes=7_800)

    def _fail_if_constructed(*args: object, **kwargs: object) -> None:
        raise AssertionError("a single-chunk data() call must not construct a thread pool")

    monkeypatch.setattr("dc_sdmx.client.ThreadPoolExecutor", _fail_if_constructed)

    result = client.data(variables=["Count_Person"], entities=_ENTITIES)

    assert len(result.rows) == 1


# --- url_budget_bytes property ----------------------------------------------


def test_url_budget_bytes_property_matches_constructor_argument():
    # A caller planning its own requests against this client's wire cost
    # (search.engine.clients.sdmx._split_facet_ids_for_request) reads the
    # real, possibly caller-configured budget through this public property
    # rather than the leading-underscore _url_budget_bytes attribute.
    client = SdmxClient(base_url=_BASE_URL, url_budget_bytes=1234)

    assert client.url_budget_bytes == 1234
    assert client.url_budget_bytes == client._url_budget_bytes
