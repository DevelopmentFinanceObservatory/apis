"""Hedged `/data` reads. The first success or lasting client error decides, and each request
gets at most one duplicate."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import httpx
import pytest

import dc_sdmx.client
from dc_sdmx import SdmxHttpError
from dc_sdmx.client import (
    _HEDGE_LOSER_CAP,
    _HEDGE_SLOTS,
    SdmxClient,
    _close_sent_response,
    _DeadlineGuard,
    _race,
    _ReadStoppedError,
)
from dfo_platform.timings import PhaseTimings, collecting

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


@pytest.mark.parametrize("hedge_after", [0.0, -0.5])
def test_a_hedge_delay_that_is_not_positive_is_refused(hedge_after):
    with pytest.raises(ValueError, match="hedge_after"):
        SdmxClient(base_url=_BASE_URL, hedge_after=hedge_after)


def test_a_stalled_request_is_answered_by_its_duplicate_and_the_loser_frees_its_slot_at_once():
    transport = _StalledFirstRequest()
    hedged = [f"V{i}" for i in range(_HEDGE_SLOTS + 1)]
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        total_concurrency=len(hedged),
        hedge_after=_HEDGE_AFTER,
    )
    timings = PhaseTimings()

    # Each read returns its duplicate's answer while its first request is still stalled. The
    # stalled losers hold no hedge slot, so more reads hedge than there are hedge slots.
    with collecting(timings):
        assert [_value(client, variable) for variable in hedged] == ["2"] * len(hedged)
    assert transport.requests == Counter(dict.fromkeys(hedged, 2))
    counts = timings.count_snapshot()
    assert (counts["sdmx_hedges"], counts["sdmx_hedge_wins"]) == (len(hedged), len(hedged))

    # Each stalled first request gave its total slot back when its race was decided and runs
    # on in a loser place.
    assert _total_slots_free(client, len(hedged))
    assert client._running_losers == len(hedged)

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
        pytest.param(lambda _request: httpx.Response(401, text="unauthorized"), id="401"),
        pytest.param(lambda _request: httpx.Response(403, text="forbidden"), id="403"),
        pytest.param(
            lambda _request: httpx.Response(200, text="<html></html>"), id="2xx without csv"
        ),
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
    timings = PhaseTimings()

    with collecting(timings):
        assert _value(client, "V0") == "1"
    assert requests["sent"] == 2
    counts = timings.count_snapshot()
    assert (counts["sdmx_hedges"], counts.get("sdmx_hedge_wins", 0)) == (1, 0)


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


def _failing_thread_start(monkeypatch, *, failing_start: int) -> list[threading.Thread]:
    """Make the `failing_start`-th `Thread.start` raise as when no thread can be created, and
    return every thread that started."""
    real_start = threading.Thread.start
    started: list[threading.Thread] = []
    starts = 0

    def start(thread: threading.Thread) -> None:
        nonlocal starts
        starts += 1
        if starts == failing_start:
            raise RuntimeError("can't start new thread")
        started.append(thread)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    return started


def test_a_duplicate_whose_thread_fails_to_start_leaves_the_read_waiting_on_its_request(
    monkeypatch,
):
    transport = _StalledFirstRequest()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)
    _failing_thread_start(monkeypatch, failing_start=2)
    timings = PhaseTimings()

    def read_counted() -> str | None:
        with collecting(timings):
            return _value(client, "V0")

    with ThreadPoolExecutor(max_workers=1) as pool:
        read = pool.submit(read_counted)
        time.sleep(4 * _HEDGE_AFTER)
        assert not read.done()
        transport.release.set()
        assert read.result(timeout=5) == "1"
    monkeypatch.undo()

    assert transport.requests["V0"] == 1
    assert timings.count_snapshot().get("sdmx_hedges", 0) == 0
    assert (client._undecided_races, client._running_losers) == (0, 0)
    assert transport.stalled_streams[0].closed.wait(timeout=5)


def test_a_request_whose_thread_fails_to_start_is_sent_on_the_callers_thread(monkeypatch):
    sent_on: list[threading.Thread] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_on.append(threading.current_thread())
        return httpx.Response(
            200, headers={"content-type": _CONTENT_TYPE}, content=_body("V0", "1")
        )

    client = SdmxClient(
        base_url=_BASE_URL, transport=httpx.MockTransport(handler), hedge_after=_HEDGE_AFTER
    )
    started = _failing_thread_start(monkeypatch, failing_start=1)

    assert _value(client, "V0") == "1"
    assert sent_on == [threading.current_thread()]
    assert started == []


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

    timings = PhaseTimings()

    started = time.monotonic()
    with collecting(timings), pytest.raises(SdmxHttpError) as excinfo:
        _value(client, "V0")
    assert time.monotonic() - started < 1
    assert excinfo.value.upstream_status == 404
    assert requests["sent"] == 2
    # Only a duplicate's 2xx SDMX-CSV response counts as a hedge win.
    counts = timings.count_snapshot()
    assert (counts["sdmx_hedges"], counts.get("sdmx_hedge_wins", 0)) == (1, 0)
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


def test_the_wait_before_a_duplicate_ends_when_the_read_deadline_passes():
    transport = _StalledFirstRequest()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=10)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _value(client, "V0", deadline=time.monotonic() + 4 * _HEDGE_AFTER)
    assert time.monotonic() - started < 1
    assert transport.requests["V0"] == 1
    transport.release.set()


def test_a_read_with_less_than_the_hedge_interval_left_sends_no_duplicate():
    transport = _StalledFirstRequest()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)

    with pytest.raises(TimeoutError):
        _value(client, "V0", deadline=time.monotonic() + 1.5 * _HEDGE_AFTER)
    assert transport.requests["V0"] == 1
    transport.release.set()


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

    # The stalled chunk leaves its race, but its first send keeps its slot until it settles.
    time.sleep(4 * _HEDGE_AFTER)
    assert not _total_slots_free(client, len(variables))
    release.set()
    _wait_until(lambda: _total_slots_free(client, len(variables)))


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
    client._undecided_races = _HEDGE_SLOTS

    started = time.monotonic()
    with pytest.raises(SdmxHttpError) as excinfo:
        client.data(variables=variables, entities=_ENTITIES, deadline=time.monotonic() + 30)
    assert time.monotonic() - started < 1
    assert excinfo.value.upstream_status == 400
    assert requests[stalled] == 1

    # The abandoned request keeps its slot until it settles.
    time.sleep(4 * _HEDGE_AFTER)
    assert not _total_slots_free(client, len(variables))
    release.set()
    _wait_until(lambda: _total_slots_free(client, len(variables)))


@pytest.mark.parametrize("duplicates", ["stall", "answer"])
def test_reads_never_run_more_sends_than_the_slots_and_the_loser_cap(duplicates):
    """Every first request stalls past its read's deadline. When duplicates stall too, every
    read is abandoned and each send keeps its total slot or loser place until it settles. When
    duplicates answer at once, each decided race's request runs on in the winner's place and
    the reads that find no place left are abandoned. Either way the sends running together
    never exceed `total_concurrency` plus `_HEDGE_LOSER_CAP`."""
    total = 2
    release = threading.Event()
    lock = threading.Lock()
    sent: Counter[str] = Counter()
    running = 0
    peak = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal running, peak
        variable = str(request.url.params["c[variableMeasured]"])
        with lock:
            sent[variable] += 1
            stalls = duplicates == "stall" or sent[variable] == 1
            running += 1
            peak = max(peak, running)
        try:
            if stalls:
                assert release.wait(timeout=10)
            return httpx.Response(
                200, headers={"content-type": _CONTENT_TYPE}, content=_body(variable, "1")
            )
        finally:
            with lock:
                running -= 1

    client = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
        total_concurrency=total,
        hedge_after=_HEDGE_AFTER,
    )
    reads = 3 * (total + _HEDGE_LOSER_CAP)

    def read(i: int) -> bool:
        try:
            _value(client, f"V{i}", deadline=time.monotonic() + 4 * _HEDGE_AFTER)
        except TimeoutError:
            return False
        return True

    try:
        with ThreadPoolExecutor(max_workers=total + _HEDGE_LOSER_CAP) as pool:
            answered = list(pool.map(read, range(reads), timeout=10))
        assert total < peak <= total + _HEDGE_LOSER_CAP
        assert (sum(answered) > 0) == (duplicates == "answer")
    finally:
        release.set()


@pytest.mark.parametrize("ending", ["stopped", "deadline"])
def test_a_guarded_race_wakes_when_its_read_ends_without_polling(monkeypatch, ending):
    """With the slot poll interval far beyond the test, the race still ends at once when the
    read is stopped or its deadline passes."""
    monkeypatch.setattr(dc_sdmx.client, "_SLOT_POLL_INTERVAL", 30)
    guard = _DeadlineGuard(time.monotonic() + 0.1 if ending == "deadline" else None)
    unanswered: Future[httpx.Response] = Future()

    with ThreadPoolExecutor(max_workers=1) as pool:
        race = pool.submit(_race, (unanswered,), guard=guard)
        if ending == "stopped":
            time.sleep(_HEDGE_AFTER)
            guard.abort()
        error = race.exception(timeout=1)
    assert isinstance(error, _ReadStoppedError if ending == "stopped" else TimeoutError)
    guard.cancel()


class _GatedStream(httpx.SyncByteStream):
    """Yields the header row, then waits for `gate` before the rest of the body."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.reading = threading.Event()
        self.gate = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        head, rest = self._body[: len(_HEADER)], self._body[len(_HEADER) :]
        yield head
        self.reading.set()
        assert self.gate.wait(timeout=5)
        yield rest


def test_a_losing_response_is_closed_unread_while_the_winners_body_is_still_read():
    """The duplicate wins and its body is read slowly. The stalled request's 200 arrives
    meanwhile and is closed and its loser place freed before the winner's body is read."""
    transport = _StalledFirstRequest()
    winner_stream = _GatedStream(_body("V0", "2"))

    class _SlowWinner(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if transport.requests["V0"] == 0:
                return transport.handle_request(request)
            transport.requests["V0"] += 1
            return httpx.Response(
                200, headers={"content-type": _CONTENT_TYPE}, stream=winner_stream
            )

    client = SdmxClient(base_url=_BASE_URL, transport=_SlowWinner(), hedge_after=_HEDGE_AFTER)

    with ThreadPoolExecutor(max_workers=1) as pool:
        read = pool.submit(_value, client, "V0")
        assert winner_stream.reading.wait(timeout=5)
        transport.release.set()
        _wait_until(lambda: "V0" in transport.streams)
        assert transport.streams["V0"].closed.wait(timeout=1)
        _wait_until(lambda: client._running_losers == 0)
        assert not read.done()
        winner_stream.gate.set()
        assert read.result(timeout=5) == "2"
    assert transport.requests["V0"] == 2


def test_a_duplicate_that_wins_after_its_request_ended_holds_no_loser_place_while_its_body_is_read():
    """The request answers 503 after the duplicate was sent, and the duplicate answers later.
    No send is left running, so the winner's slow body read counts against no loser place."""
    winner_stream = _GatedStream(_body("V0", "2"))
    duplicate_sent = threading.Event()
    lock = threading.Lock()
    sent = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal sent
        with lock:
            sent += 1
            first = sent == 1
        if first:
            assert duplicate_sent.wait(timeout=5)
            return httpx.Response(503)
        duplicate_sent.set()
        time.sleep(4 * _HEDGE_AFTER)
        return httpx.Response(200, headers={"content-type": _CONTENT_TYPE}, stream=winner_stream)

    client = SdmxClient(
        base_url=_BASE_URL, transport=httpx.MockTransport(handler), hedge_after=_HEDGE_AFTER
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        read = pool.submit(_value, client, "V0")
        assert winner_stream.reading.wait(timeout=5)
        assert client._running_losers == 0
        winner_stream.gate.set()
        assert read.result(timeout=5) == "2"
    assert client._running_losers == 0


class _RaisingCloseStream(httpx.SyncByteStream):
    def __init__(self, error: Exception) -> None:
        self._error = error

    def __iter__(self) -> Iterator[bytes]:
        yield b""

    def close(self) -> None:
        raise self._error


@pytest.mark.parametrize(
    "error",
    [OSError("reset"), httpx.ReadError("reset"), RuntimeError("closed"), ValueError("closed")],
    ids=type,
)
def test_closing_a_settled_send_whose_close_raises_leaves_no_error_for_the_callback(error):
    sent: Future[httpx.Response] = Future()
    sent.set_result(httpx.Response(200, stream=_RaisingCloseStream(error)))

    _close_sent_response(sent)
