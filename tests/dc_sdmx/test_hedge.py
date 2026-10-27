"""Test hedged `/data` reads: the first answer wins and each request gets one duplicate at most."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import httpx

from dc_sdmx.client import _HEDGE_SLOTS, SdmxClient

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
    one answers 2 at once."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.requests: Counter[str] = Counter()
        self.stalled_streams: list[_ClosingStream] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        variable = str(request.url.params["c[variableMeasured]"])
        with self.lock:
            self.requests[variable] += 1
            first = self.requests[variable] == 1
        if first:
            assert self.release.wait(timeout=5)
        stream = _ClosingStream(_body(variable, "1" if first else "2"))
        if first:
            self.stalled_streams.append(stream)
        return httpx.Response(200, headers={"content-type": _CONTENT_TYPE}, stream=stream)


def _value(client: SdmxClient, variable: str) -> str | None:
    rows = client.data(variables=[variable], entities=_ENTITIES).rows
    assert len(rows) == 1
    return rows[0].value


def test_a_stalled_request_is_answered_by_its_one_duplicate_while_hedge_slots_last():
    transport = _StalledFirstRequest()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, hedge_after=_HEDGE_AFTER)
    hedged = [f"V{i}" for i in range(_HEDGE_SLOTS)]

    # Each read returns its duplicate's answer while its first request is still stalled.
    assert [_value(client, variable) for variable in hedged] == ["2"] * _HEDGE_SLOTS
    assert transport.requests == Counter(dict.fromkeys(hedged, 2))

    # Every hedge slot is held by a stalled loser, so the next read waits for its own answer.
    with ThreadPoolExecutor(max_workers=1) as pool:
        unhedged = pool.submit(_value, client, "V_unhedged")
        time.sleep(4 * _HEDGE_AFTER)
        assert not unhedged.done()
        assert transport.requests["V_unhedged"] == 1
        transport.release.set()
        assert unhedged.result(timeout=5) == "1"

    # The losers' responses are closed unread, and their slots serve the next stalled read.
    for stream in transport.stalled_streams[:_HEDGE_SLOTS]:
        assert stream.closed.wait(timeout=5)
    transport.release.clear()
    assert _value(client, "V_after") == "2"
    assert transport.requests["V_after"] == 2
    transport.release.set()


def test_a_chunked_read_sends_each_slow_chunk_at_most_twice():
    """Every request takes four hedge intervals. Each chunk's request is sent once more, and
    never a third time, while a hedge slot is free."""
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
