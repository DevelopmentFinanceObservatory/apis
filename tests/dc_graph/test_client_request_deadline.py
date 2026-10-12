"""The bound request deadline (`dfo_platform.request_deadline`) bounds graph reads whose caller
passes no deadline, on the calling thread, on a worker thread, and in the client's chunk pool."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from dc_graph import GraphClient, GraphDeadlineExceededError
from dfo_platform.request_deadline import request_deadline

_CLIENT_TIMEOUT_S = 30.0


class _RecordingTransport(httpx.BaseTransport):
    """Answer every POST with an empty body and record each request's read timeout."""

    def __init__(self) -> None:
        self.read_timeouts: list[float] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.read_timeouts.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json={"data": {}})


def _client(transport: _RecordingTransport) -> GraphClient:
    # A chunk of one dcid makes a two-dcid read run in the client's own chunk pool.
    return GraphClient(
        base_url="http://fake/core/api",
        timeout=_CLIENT_TIMEOUT_S,
        transport=transport,
        single_arc_batch_chunk_size=1,
    )


def _read_on_the_calling_thread(client: GraphClient) -> None:
    client.node_labels_batch(["country/KEN"])


def _read_in_the_chunk_pool(client: GraphClient) -> None:
    client.node_labels_batch(["country/KEN", "country/ETH"])


def _read_on_a_worker_thread(client: GraphClient) -> None:
    async def _read() -> None:
        await asyncio.to_thread(client.node_labels_batch, ["country/KEN"])

    asyncio.run(_read())


_READS = pytest.mark.parametrize(
    "read",
    [_read_on_the_calling_thread, _read_in_the_chunk_pool, _read_on_a_worker_thread],
    ids=["calling_thread", "chunk_pool", "worker_thread"],
)


@_READS
def test_a_read_without_a_deadline_stops_at_the_passed_request_deadline(read) -> None:
    """The read refuses before sending anything."""
    transport = _RecordingTransport()
    with request_deadline(time.monotonic() - 1), pytest.raises(GraphDeadlineExceededError):
        read(_client(transport))
    assert transport.read_timeouts == []


@_READS
def test_a_read_without_a_deadline_times_out_by_the_request_deadline(read) -> None:
    transport = _RecordingTransport()
    with request_deadline(time.monotonic() + 5):
        read(_client(transport))
    assert transport.read_timeouts
    assert all(timeout <= 5 for timeout in transport.read_timeouts)


def test_an_earlier_explicit_deadline_still_bounds_the_read() -> None:
    transport = _RecordingTransport()
    with request_deadline(time.monotonic() + 20):
        _client(transport).post(
            "/v2/node", {"nodes": ["country/KEN"]}, deadline=time.monotonic() + 5
        )
    assert transport.read_timeouts and transport.read_timeouts[0] <= 5


@_READS
def test_outside_a_request_a_read_keeps_the_client_timeout(read) -> None:
    transport = _RecordingTransport()
    read(_client(transport))
    assert transport.read_timeouts
    assert all(timeout == _CLIENT_TIMEOUT_S for timeout in transport.read_timeouts)


_HELD_SLOT_READS = pytest.mark.parametrize(
    ("semaphore_attr", "read"),
    [
        ("_node_arc_semaphore", lambda client: client.node_arcs_batch(["country/KEN"])),
        (
            "_node_arc_semaphore",
            lambda client: client.node_observation_properties_batch(["country/KEN"]),
        ),
        (
            "_node_arc_many_semaphore",
            lambda client: client.node_arcs_many(["country/KEN"], properties=["name"]),
        ),
    ],
    ids=["node_arcs_batch", "node_observation_properties_batch", "node_arcs_many"],
)


@_HELD_SLOT_READS
def test_a_semaphore_wait_stops_at_the_request_deadline(semaphore_attr, read) -> None:
    """A read queued behind a held slot raises the deadline error at the request deadline,
    without waiting for the slot to free."""
    transport = _RecordingTransport()
    client = _client(transport)
    semaphore = getattr(client, semaphore_attr)
    while semaphore.acquire(blocking=False):
        pass
    outcome: list[BaseException | None] = []

    def _read() -> None:
        try:
            with request_deadline(time.monotonic() + 0.1):
                read(client)
        except BaseException as exc:
            outcome.append(exc)
        else:
            outcome.append(None)

    worker = threading.Thread(target=_read)
    worker.start()
    worker.join(timeout=2)
    still_waiting = worker.is_alive()
    while True:  # free every slot so a still-waiting read finishes
        try:
            semaphore.release()
        except ValueError:
            break
    worker.join(timeout=5)
    assert not still_waiting
    assert isinstance(outcome[0], GraphDeadlineExceededError)
    assert transport.read_timeouts == []
