"""Hedged `/data` reads. The first success or lasting client error decides, and each request
gets at most one duplicate."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from dc_sdmx import SdmxHttpError
from dc_sdmx.client import _HEDGE_LOSER_CAP, _HEDGE_SLOTS, SdmxClient

_BASE_URL = "https://datacommons.one.org/core/api"
_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_ENTITIES = {"observationAbout": ["country/USA"]}
_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_HEDGE_AFTER = 0.05


def _body(variable: str, value: str) -> bytes:
    return (
        f"{_HEADER}dataflow,DC:DF_OBS(1.0.0),I,{variable},country/USA,Person,Census,P1Y,"
        f"dc/base,2020,{value},0,facet-{variable}\r\n"
    ).encode()


class _ClosingStream(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        yield self._body

    def close(self) -> None:
        self.closed.set()


class _StalledFirstRequest(httpx.BaseTransport):
    """The first request for each variable waits for `release` and answers 1; every later
    one answers 2 at once. Variables in `stall_every` stall on every request, and a variable
    in `gates` waits for its own event in place of `release`."""

    def __init__(self, *, stall_every: frozenset[str] = frozenset()) -> None:
        self.stall_every = stall_every
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.requests: Counter[str] = Counter()
        self.stalled_streams: list[_ClosingStream] = []
        self.closed = threading.Event()
        self.gates: dict[str, threading.Event] = {}
        self.streams: dict[str, _ClosingStream] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        variable = str(request.url.params["c[variableMeasured]"])
        with self.lock:
            self.requests[variable] += 1
            first = self.requests[variable] == 1
        stalls = first or variable in self.stall_every
        if stalls:
            assert self.gates.get(variable, self.release).wait(timeout=5)
        stream = _ClosingStream(_body(variable, "1" if first else "2"))
        if stalls:
            self.stalled_streams.append(stream)
            self.streams[variable] = stream
        return httpx.Response(200, headers={"content-type": _CONTENT_TYPE}, stream=stream)

    def close(self) -> None:
        self.closed.set()


def _value(client: SdmxClient, variable: str, *, deadline: float | None = None) -> str | None:
    rows = client.data(variables=[variable], entities=_ENTITIES, deadline=deadline).rows
    assert len(rows) == 1
    return rows[0].value


def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(_HEDGE_AFTER / 5)


def _total_slots_free(client: SdmxClient, slots: int) -> bool:
    taken = [client._total_semaphore.acquire(blocking=False) for _ in range(slots)]
    for got in taken:
        if got:
            client._total_semaphore.release()
    return all(taken)


def test_a_stalled_request_is_answered_by_its_duplicate_and_the_loser_frees_its_slot_at_once():
    transport = _StalledFirstRequest()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)
    hedged = [f"V{i}" for i in range(_HEDGE_SLOTS + 1)]

    # Each read returns its duplicate's answer while its first request is still stalled. The
    # stalled losers hold no slot, so more reads hedge than there are slots.
    assert [_value(client, variable) for variable in hedged] == ["2"] * len(hedged)
    assert transport.requests == Counter(dict.fromkeys(hedged, 2))

    # The losers' responses are closed unread once they arrive.
    transport.release.set()
    for stream in transport.stalled_streams:
        assert stream.closed.wait(timeout=5)


def test_a_read_waits_for_its_own_answer_while_every_hedge_slot_is_taken():
    holders = [f"H{i}" for i in range(_HEDGE_SLOTS)]
    transport = _StalledFirstRequest(stall_every=frozenset(holders))
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)

    with ThreadPoolExecutor(max_workers=_HEDGE_SLOTS + 1) as pool:
        # Both sends of each holder stall, so the holders keep every slot.
        held = [pool.submit(_value, client, variable) for variable in holders]
        _wait_until(lambda: all(transport.requests[variable] == 2 for variable in holders))
        unhedged = pool.submit(_value, client, "V_unhedged")
        time.sleep(4 * _HEDGE_AFTER)
        assert not unhedged.done()
        assert transport.requests["V_unhedged"] == 1
        transport.release.set()
        assert unhedged.result(timeout=5) == "1"
        assert all(future.result(timeout=5) in {"1", "2"} for future in held)


def test_hedging_pauses_while_the_losers_still_running_reach_the_cap_and_resumes_after_one_ends():
    transport = _StalledFirstRequest()
    losers = [f"L{i}" for i in range(_HEDGE_LOSER_CAP)]
    transport.gates = {variable: threading.Event() for variable in [*losers, "V_unhedged"]}
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)

    # Each read is answered by its duplicate and leaves its first request running as a loser.
    assert [_value(client, variable) for variable in losers] == ["2"] * len(losers)

    # With the cap reached, a slow read sends no duplicate and waits for its own answer.
    with ThreadPoolExecutor(max_workers=1) as pool:
        unhedged = pool.submit(_value, client, "V_unhedged")
        time.sleep(4 * _HEDGE_AFTER)
        assert transport.requests["V_unhedged"] == 1
        transport.gates["V_unhedged"].set()
        assert unhedged.result(timeout=5) == "1"

    # Once one loser settles, the next slow read hedges again.
    transport.gates[losers[0]].set()
    _wait_until(lambda: losers[0] in transport.streams)
    assert transport.streams[losers[0]].closed.wait(timeout=5)
    assert _value(client, "V_after") == "2"
    assert transport.requests["V_after"] == 2
    transport.release.set()
    for gate in transport.gates.values():
        gate.set()


def test_a_chunked_read_sends_each_slow_chunk_at_most_twice():
    """Every request takes four hedge intervals. Each chunk's request is sent at most
    twice, and the extra sends total no more than the hedge slots."""
    variables = [f"V{i:09d}" for i in range(4)]
    lock = threading.Lock()
    requests: Counter[str] = Counter()

    def slow(request: httpx.Request) -> httpx.Response:
        variable = str(request.url.params["c[variableMeasured]"])
        with lock:
            requests[variable] += 1
        time.sleep(4 * _HEDGE_AFTER)
        return httpx.Response(
            200, headers={"content-type": _CONTENT_TYPE}, content=_body(variable, "1")
        )

    client = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(slow),
        url_budget_bytes=72,
        hedge_after=_HEDGE_AFTER,
    )

    rows = client.data(variables=variables, entities=_ENTITIES).rows

    assert [row.variable for row in rows] == variables
    assert set(requests) == set(variables)
    assert max(requests.values()) == 2
    assert sum(requests.values()) <= len(variables) + _HEDGE_SLOTS


def _reset(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection reset", request=request)


@pytest.mark.parametrize(
    "fail",
    [
        pytest.param(_reset, id="transport error"),
        pytest.param(lambda _request: httpx.Response(503, text="unavailable"), id="503"),
        pytest.param(lambda _request: httpx.Response(429, text="slow down"), id="429"),
        pytest.param(lambda _request: httpx.Response(408, text="request timeout"), id="408"),
    ],
)
def test_a_duplicate_that_fails_at_once_loses_to_a_primary_that_answers_late(fail):
    lock = threading.Lock()
    requests: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            requests["sent"] += 1
            first = requests["sent"] == 1
        if not first:
            return fail(request)
        time.sleep(4 * _HEDGE_AFTER)
        return httpx.Response(
            200, headers={"content-type": _CONTENT_TYPE}, content=_body("V0", "1")
        )

    client = SdmxClient(
        base_url=_BASE_URL, transport=httpx.MockTransport(handler), hedge_after=_HEDGE_AFTER
    )

    assert _value(client, "V0") == "1"
    assert requests["sent"] == 2


def test_a_read_whose_two_sends_fail_raises_the_first_requests_error():
    lock = threading.Lock()
    requests: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            requests["sent"] += 1
            first = requests["sent"] == 1
        if first:
            time.sleep(4 * _HEDGE_AFTER)
            return httpx.Response(500, text="primary failed")
        return httpx.Response(503, text="duplicate failed")

    client = SdmxClient(
        base_url=_BASE_URL, transport=httpx.MockTransport(handler), hedge_after=_HEDGE_AFTER
    )

    with pytest.raises(SdmxHttpError) as excinfo:
        _value(client, "V0")
    assert excinfo.value.upstream_status == 500
    assert requests["sent"] == 2


def test_the_duplicate_times_out_with_what_remains_of_the_read_deadline():
    lock = threading.Lock()
    read_timeouts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            read_timeouts.append(request.extensions["timeout"]["read"])
            first = len(read_timeouts) == 1
        if first:
            time.sleep(4 * _HEDGE_AFTER)
        return httpx.Response(
            200, headers={"content-type": _CONTENT_TYPE}, content=_body("V0", "1")
        )

    client = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
        timeout=90,
        hedge_after=_HEDGE_AFTER,
    )

    _value(client, "V0", deadline=time.monotonic() + 10)

    primary, duplicate = read_timeouts
    assert primary <= 10
    assert duplicate <= primary - _HEDGE_AFTER


def test_closing_the_client_waits_for_a_losing_send_to_settle():
    transport = _StalledFirstRequest()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)

    assert _value(client, "V0") == "2"
    client.close()
    assert not transport.closed.is_set()

    transport.release.set()
    assert transport.closed.wait(timeout=5)
    assert transport.stalled_streams[0].closed.is_set()


def test_a_duplicate_whose_thread_fails_to_start_frees_its_slot_and_closes_the_first_request(
    monkeypatch,
):
    transport = _StalledFirstRequest()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)
    real_start = threading.Thread.start
    starts = 0

    def start(thread: threading.Thread) -> None:
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("can't start new thread")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        _value(client, "V0")
    monkeypatch.undo()

    transport.release.set()
    _wait_until(lambda: bool(transport.stalled_streams))
    assert transport.stalled_streams[0].closed.wait(timeout=5)
    slots = [client._hedge_slots.acquire(blocking=False) for _ in range(_HEDGE_SLOTS)]
    assert slots == [True] * _HEDGE_SLOTS


@pytest.mark.parametrize("failing_send", ["primary", "duplicate"])
def test_a_client_error_from_either_send_raises_without_waiting_for_the_other(failing_send):
    lock = threading.Lock()
    requests: Counter[str] = Counter()
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            requests["sent"] += 1
            send = "primary" if requests["sent"] == 1 else "duplicate"
        if send == "primary":
            # The primary outlasts the hedge interval so a duplicate is sent.
            time.sleep(2 * _HEDGE_AFTER)
        if send == failing_send:
            return httpx.Response(404, text="no such variable")
        assert release.wait(timeout=5)
        return httpx.Response(
            200, headers={"content-type": _CONTENT_TYPE}, content=_body("V0", "1")
        )

    client = SdmxClient(
        base_url=_BASE_URL, transport=httpx.MockTransport(handler), hedge_after=_HEDGE_AFTER
    )

    started = time.monotonic()
    with pytest.raises(SdmxHttpError) as excinfo:
        _value(client, "V0")
    assert time.monotonic() - started < 1
    assert excinfo.value.upstream_status == 404
    assert requests["sent"] == 2
    release.set()


def test_a_race_ends_when_the_read_deadline_passes():
    transport = _StalledFirstRequest(stall_every=frozenset({"V0"}))
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _value(client, "V0", deadline=time.monotonic() + 4 * _HEDGE_AFTER)
    assert time.monotonic() - started < 1
    assert transport.requests["V0"] == 2

    # Both abandoned sends have their responses closed when they arrive.
    transport.release.set()
    _wait_until(lambda: len(transport.stalled_streams) == 2)
    assert all(stream.closed.wait(timeout=5) for stream in transport.stalled_streams)


def test_a_race_ends_when_a_sibling_chunk_fails():
    variables = [f"V{i:09d}" for i in range(2)]
    failing, stalled = variables
    lock = threading.Lock()
    requests: Counter[str] = Counter()
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        variable = str(request.url.params["c[variableMeasured]"])
        with lock:
            requests[variable] += 1
        if variable == failing:
            # Fail once the stalled chunk has sent its duplicate.
            _wait_until(lambda: requests[stalled] == 2)
            return httpx.Response(400, text="bad request")
        # Outlast `_wait_until`, so only leaving the race frees the slot in time.
        assert release.wait(timeout=10)
        return httpx.Response(
            200, headers={"content-type": _CONTENT_TYPE}, content=_body(variable, "1")
        )

    client = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
        url_budget_bytes=72,
        total_concurrency=len(variables),
        hedge_after=_HEDGE_AFTER,
    )

    started = time.monotonic()
    with pytest.raises(SdmxHttpError) as excinfo:
        client.data(variables=variables, entities=_ENTITIES, deadline=time.monotonic() + 30)
    assert time.monotonic() - started < 1
    assert excinfo.value.upstream_status == 400
    assert requests[stalled] == 2

    # The stalled chunk leaves its race and frees its slot while both its sends still stall.
    _wait_until(lambda: _total_slots_free(client, len(variables)))
    release.set()


def test_concurrent_reads_never_run_more_losers_than_the_cap():
    """With one loser place left, two slow reads reach the hedge decision together and only
    one of them sends a duplicate."""
    transport = _StalledFirstRequest()
    losers = [f"L{i}" for i in range(_HEDGE_LOSER_CAP - 1)]
    racers = ["R0", "R1"]
    transport.gates = {variable: threading.Event() for variable in losers}
    transport.stall_every = frozenset(racers)
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)
    assert [_value(client, variable) for variable in losers] == ["2"] * len(losers)

    with ThreadPoolExecutor(max_workers=len(racers)) as pool:
        reads = [pool.submit(_value, client, variable) for variable in racers]
        time.sleep(4 * _HEDGE_AFTER)
        assert sum(transport.requests[variable] for variable in racers) == len(racers) + 1
        transport.release.set()
        assert all(read.result(timeout=5) in {"1", "2"} for read in reads)
    for gate in transport.gates.values():
        gate.set()


def test_a_request_answered_after_the_hedge_interval_sends_no_duplicate(monkeypatch):
    """A primary that answers between the hedge interval and the hedge decision wins alone."""
    transport = _StalledFirstRequest()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)
    real_hedge_timeout = client._hedge_timeout

    def hedge_timeout_after_the_primary_answers(guard):
        transport.release.set()
        _wait_until(lambda: "V0" in transport.streams)
        time.sleep(_HEDGE_AFTER)
        return real_hedge_timeout(guard)

    monkeypatch.setattr(client, "_hedge_timeout", hedge_timeout_after_the_primary_answers)

    assert _value(client, "V0") == "1"
    assert transport.requests["V0"] == 1


def test_a_wait_on_an_unhedged_request_ends_when_a_sibling_chunk_fails():
    """With every hedge slot taken, a slow chunk waits on its request alone, and a sibling's
    failure still ends that wait."""
    variables = [f"V{i:09d}" for i in range(2)]
    failing, stalled = variables
    lock = threading.Lock()
    requests: Counter[str] = Counter()
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        variable = str(request.url.params["c[variableMeasured]"])
        with lock:
            requests[variable] += 1
        if variable == failing:
            # Fail once the stalled chunk is past its hedge interval.
            _wait_until(lambda: requests[stalled] == 1)
            time.sleep(4 * _HEDGE_AFTER)
            return httpx.Response(400, text="bad request")
        # Outlast `_wait_until`, so only leaving the wait frees the slot in time.
        assert release.wait(timeout=10)
        return httpx.Response(
            200, headers={"content-type": _CONTENT_TYPE}, content=_body(variable, "1")
        )

    client = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
        url_budget_bytes=72,
        total_concurrency=len(variables),
        hedge_after=_HEDGE_AFTER,
    )
    assert all(client._hedge_slots.acquire(blocking=False) for _ in range(_HEDGE_SLOTS))

    started = time.monotonic()
    with pytest.raises(SdmxHttpError) as excinfo:
        client.data(variables=variables, entities=_ENTITIES, deadline=time.monotonic() + 30)
    assert time.monotonic() - started < 1
    assert excinfo.value.upstream_status == 400
    assert requests[stalled] == 1

    _wait_until(lambda: _total_slots_free(client, len(variables)))
    release.set()
