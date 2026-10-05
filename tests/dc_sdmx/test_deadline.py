"""Absolute deadlines for streamed SDMX reads."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import httpx
import pytest

from dc_sdmx import SdmxClient

_BASE_URL = "https://example.test"
_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_HEADER = (
    b"STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    b"observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_ROW = (
    b"dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/KEN,Count,Measured,"
    b"P1Y,Source,2020,1,,facet1\r\n"
)


class _SlowStreamingTransport(httpx.BaseTransport):
    def __init__(self, *, delay: float, rows: int) -> None:
        self.delay = delay
        self.rows = rows
        self.chunks_yielded = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": _CONTENT_TYPE},
            content=self._chunks(),
        )

    def _chunks(self) -> Iterator[bytes]:
        for chunk in (_HEADER, *(_ROW for _ in range(self.rows))):
            time.sleep(self.delay)
            self.chunks_yielded += 1
            yield chunk


class _RecordingTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(200, headers={"content-type": _CONTENT_TYPE}, content=_HEADER)


_AVAILABILITY_BODY = b'{"data": {"dataConstraints": []}}'


class _AvailabilityRecordingTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=_AVAILABILITY_BODY
        )


class _BlockingStream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        yield _HEADER
        self.closed.wait(timeout=5)

    def close(self) -> None:
        self.closed.set()


class _BlockingTransport(httpx.BaseTransport):
    def __init__(self, *, status_code: int = 200) -> None:
        self.stream = _BlockingStream()
        self.status_code = status_code

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            self.status_code,
            headers={"content-type": _CONTENT_TYPE},
            stream=self.stream,
        )


def test_streaming_deadline_enforces_total_elapsed_time_despite_arriving_chunks() -> None:
    transport = _SlowStreamingTransport(delay=0.01, rows=100)
    client = SdmxClient(base_url=_BASE_URL, timeout=10, transport=transport)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.data(
            variables=["Count_Person"],
            entities={"recipient": ["country/KEN"]},
            deadline=started + 0.08,
        )

    elapsed = time.monotonic() - started
    assert elapsed < 0.25
    assert transport.chunks_yielded < transport.rows + 1

    # Expiry closes the response that crossed its deadline. The shared client
    # and transport remain available to later materialisation members.
    transport.delay = 0
    transport.rows = 0
    assert (
        client.data(variables=["Count_Person"], entities={"recipient": ["country/KEN"]}).rows == ()
    )


def test_deadline_closes_a_response_blocked_waiting_for_its_next_chunk() -> None:
    transport = _BlockingTransport()
    client = SdmxClient(base_url=_BASE_URL, timeout=10, transport=transport)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.data(
            variables=["Count_Person"],
            entities={"recipient": ["country/KEN"]},
            deadline=started + 0.08,
        )

    assert time.monotonic() - started < 0.25
    assert transport.stream.closed.is_set()


def test_deadline_classifies_a_blocked_error_body_as_timeout() -> None:
    transport = _BlockingTransport(status_code=500)
    client = SdmxClient(base_url=_BASE_URL, timeout=10, transport=transport)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.data(
            variables=["Count_Person"],
            entities={"recipient": ["country/KEN"]},
            deadline=started + 0.08,
        )

    assert time.monotonic() - started < 0.25
    assert transport.stream.closed.is_set()


def test_availability_deadline_classifies_a_blocked_error_body_as_timeout() -> None:
    transport = _BlockingTransport(status_code=500)
    client = SdmxClient(base_url=_BASE_URL, timeout=10, transport=transport)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.availability(
            variables=["Count_Person"],
            component="observationAbout",
            deadline=started + 0.08,
        )

    assert time.monotonic() - started < 0.25
    assert transport.stream.closed.is_set()


def test_deadline_uses_the_lower_of_deadline_and_transport_timeouts() -> None:
    short_deadline_transport = _RecordingTransport()
    short_deadline_client = SdmxClient(
        base_url=_BASE_URL, timeout=90, transport=short_deadline_transport
    )
    short_deadline_client.data(
        variables=["Count_Person"],
        entities={"recipient": ["country/KEN"]},
        deadline=time.monotonic() + 5,
    )
    assert short_deadline_transport.request is not None
    assert 0 < short_deadline_transport.request.extensions["timeout"]["read"] <= 5

    transport_bound_transport = _RecordingTransport()
    transport_bound_client = SdmxClient(
        base_url=_BASE_URL, timeout=2, transport=transport_bound_transport
    )
    transport_bound_client.data(
        variables=["Count_Person"],
        entities={"recipient": ["country/KEN"]},
        deadline=time.monotonic() + 30,
    )
    assert transport_bound_transport.request is not None
    assert transport_bound_transport.request.extensions["timeout"]["read"] == 2


def test_expired_deadline_raises_before_transport_dispatch() -> None:
    transport = _RecordingTransport()
    client = SdmxClient(base_url=_BASE_URL, transport=transport)

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.data(
            variables=["Count_Person"],
            entities={"recipient": ["country/KEN"]},
            deadline=time.monotonic() - 1,
        )

    assert transport.request is None


def test_availability_expired_deadline_raises_before_transport_dispatch() -> None:
    transport = _AvailabilityRecordingTransport()
    client = SdmxClient(base_url=_BASE_URL, transport=transport)

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.availability(
            variables=["Count_Person"],
            component="observationAbout",
            deadline=time.monotonic() - 1,
        )

    assert transport.request is None


def test_availability_deadline_uses_the_lower_of_deadline_and_transport_timeouts() -> None:
    short_deadline_transport = _AvailabilityRecordingTransport()
    short_deadline_client = SdmxClient(
        base_url=_BASE_URL, timeout=90, transport=short_deadline_transport
    )
    short_deadline_client.availability(
        variables=["Count_Person"],
        component="observationAbout",
        deadline=time.monotonic() + 5,
    )
    assert short_deadline_transport.request is not None
    assert 0 < short_deadline_transport.request.extensions["timeout"]["read"] <= 5


def test_availability_read_seconds_counts_from_when_the_get_holds_its_slot() -> None:
    # The one GET slot is held for 0.3 s, longer than the read's 0.2 s window. The window opens
    # once the read holds the slot, so the read answers.
    slots = threading.BoundedSemaphore(1)
    transport = _AvailabilityRecordingTransport()
    client = SdmxClient(base_url=_BASE_URL, transport=transport, total_concurrency=slots)
    slots.acquire()
    threading.Timer(0.3, slots.release).start()
    started = time.monotonic()

    client.availability(variables=["Count_Person"], component="observationAbout", read_seconds=0.2)

    assert time.monotonic() - started >= 0.3
    assert transport.request is not None


def test_availability_read_seconds_closes_a_blocked_body() -> None:
    transport = _BlockingTransport()
    client = SdmxClient(base_url=_BASE_URL, timeout=10, transport=transport)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute caller deadline"):
        client.availability(
            variables=["Count_Person"], component="observationAbout", read_seconds=0.08
        )

    assert time.monotonic() - started < 0.25
    assert transport.stream.closed.is_set()
