"""Shared SDMX fixture readers and HTTP transports."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import httpx
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def read_fixture_bytes(name: str) -> bytes:
    """Read a fixture file's raw bytes (preserves CRLF line endings verbatim)."""
    return (FIXTURES_DIR / name).read_bytes()


def read_fixture_text(name: str) -> str:
    """Read a fixture file as text, without normalising its line endings."""
    return read_fixture_bytes(name).decode("utf-8")


class RecordingTransport(httpx.BaseTransport):
    """Record each request and replay a fixed response for offline request tests."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._response

    @property
    def last_request(self) -> httpx.Request:
        return self.requests[-1]


@pytest.fixture
def recording_transport():
    """Factory: build a RecordingTransport that always replays `response`."""

    def _make(response: httpx.Response) -> RecordingTransport:
        return RecordingTransport(response)

    return _make


class StreamingTransport(httpx.BaseTransport):
    """Serve fresh responses as 16-byte chunks for incremental-read tests.

    Fresh generators avoid `httpx.StreamConsumed` when a client makes several requests.
    """

    def __init__(
        self,
        *,
        body: bytes,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        chunk_size: int = 16,
    ) -> None:
        self._body = body
        self._status_code = status_code
        self._headers = dict(
            headers or {"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )
        self._chunk_size = chunk_size
        self.requests: list[httpx.Request] = []
        self.chunks_yielded = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self._status_code,
            headers=self._headers,
            content=self._iter_chunks(),
        )

    def _iter_chunks(self) -> Iterator[bytes]:
        for i in range(0, len(self._body), self._chunk_size):
            self.chunks_yielded += 1
            yield self._body[i : i + self._chunk_size]

    @property
    def total_chunks(self) -> int:
        return -(-len(self._body) // self._chunk_size) if self._body else 0


@pytest.fixture
def streaming_transport():
    """Factory: build a `StreamingTransport` replaying `body` as chunked bytes."""

    def _make(body: bytes, **kwargs: object) -> StreamingTransport:
        return StreamingTransport(body=body, **kwargs)

    return _make


class SequencedTransport(httpx.BaseTransport):
    """Replay chunked response bodies in request order and record requests and responses.

    An extra request raises `IndexError`, exposing a test with too few bodies.
    """

    def __init__(
        self,
        bodies: Sequence[bytes],
        *,
        content_type: str = "application/vnd.sdmx.data+csv;version=2.0.0",
        chunk_size: int = 16,
    ) -> None:
        self._bodies = list(bodies)
        self._content_type = content_type
        self._chunk_size = chunk_size
        self.requests: list[httpx.Request] = []
        self.responses: list[httpx.Response] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        index = len(self.requests)
        self.requests.append(request)
        body = self._bodies[index]
        response = httpx.Response(
            200,
            headers={"content-type": self._content_type},
            content=self._iter_chunks(body),
        )
        self.responses.append(response)
        return response

    def _iter_chunks(self, body: bytes) -> Iterator[bytes]:
        for i in range(0, len(body), self._chunk_size):
            yield body[i : i + self._chunk_size]


@pytest.fixture
def sequenced_transport():
    """Factory: build a `SequencedTransport` replaying `bodies` in order, one per call."""

    def _make(bodies: Sequence[bytes], **kwargs: object) -> SequencedTransport:
        return SequencedTransport(bodies, **kwargs)

    return _make


class KeyedConcurrentTransport(httpx.BaseTransport):
    """Route concurrent requests by `c[variableMeasured]`, not arrival order.

    `release` events can force a completion order. A lock protects request and concurrency counts.
    `completed` lists the keys whose responses have been returned, and `wait_completed` blocks
    until a number of them have.
    """

    def __init__(
        self,
        bodies: Mapping[str, bytes],
        *,
        release: Mapping[str, threading.Event] | None = None,
        content_type: str = "application/vnd.sdmx.data+csv;version=2.0.0",
    ) -> None:
        self._bodies = dict(bodies)
        self._release = dict(release or {})
        self._content_type = content_type
        self._lock = threading.Lock()
        self._completion = threading.Condition(self._lock)
        self.requests: list[httpx.Request] = []
        self.completed: list[str] = []
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
            event = self._release.get(key)
            if event is not None:
                # The 5s timeout fails the test if it omits this release event.
                assert event.wait(timeout=5), f"release event for {key!r} was never set"
            return httpx.Response(
                200, headers={"content-type": self._content_type}, content=self._bodies[key]
            )
        finally:
            with self._completion:
                self._in_flight -= 1
                self.completed.append(key)
                self._completion.notify_all()

    def wait_completed(self, count: int, *, timeout: float = 5) -> bool:
        """Whether `count` requests have been answered within `timeout` seconds."""
        with self._completion:
            return self._completion.wait_for(lambda: len(self.completed) >= count, timeout=timeout)


@pytest.fixture
def keyed_concurrent_transport():
    """Factory: build a `KeyedConcurrentTransport` keyed on each request's
    own `c[variableMeasured]` value.
    """

    def _make(bodies: Mapping[str, bytes], **kwargs: object) -> KeyedConcurrentTransport:
        return KeyedConcurrentTransport(bodies, **kwargs)

    return _make
