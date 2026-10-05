"""Test the per-client GET semaphore across calls, including single-chunk reads."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from dc_sdmx.client import SdmxClient

_BASE_URL = "https://datacommons.one.org/core/api"
_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_ENTITIES = {"observationAbout": ["country/USA"]}

# A 72-byte budget fits each 10-character variable used below.
_BUDGET_ONE_PER_CHUNK = 72


def _header(role: str) -> bytes:
    return (
        f"STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,{role},unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
        "scalingFactor,facetId\r\n"
    ).encode()


def _row(variable: str, entity: str) -> bytes:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{entity},Person,Census,P1Y,"
        f"dc/base,2020,1.50,0,facet-{variable}\r\n"
    ).encode()


def _variable_measured(request: httpx.Request) -> str:
    value = httpx.QueryParams(str(request.url.query, "utf-8"))["c[variableMeasured]"]
    assert value is not None
    return value


def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.005) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not met before the timeout")
        time.sleep(interval)


def test_call_concurrency_reaches_but_never_exceeds_chunk_concurrency(keyed_concurrent_transport):
    # Set both limits to 3. The barrier proves all 3 workers run concurrently;
    # its 5s timeout fails if the pool starts fewer.
    n = 3
    variables = [f"V{i:09d}" for i in range(n)]
    bodies = {var: _header("observationAbout") + _row(var, "country/USA") for var in variables}
    barrier = threading.Barrier(n, timeout=5)

    class _BarrierTransport(httpx.BaseTransport):
        def __init__(self) -> None:
            self.requests: list[httpx.Request] = []
            self._lock = threading.Lock()

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            with self._lock:
                self.requests.append(request)
            barrier.wait()
            key = _variable_measured(request)
            return httpx.Response(200, headers={"content-type": _CONTENT_TYPE}, content=bodies[key])

    transport = _BarrierTransport()
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        chunk_concurrency=n,
        total_concurrency=n,
    )

    result = client.data(variables=variables, entities=_ENTITIES)

    assert len(transport.requests) == n
    assert {row.variable for row in result.rows} == set(variables)


def test_several_concurrent_calls_never_exceed_the_total(keyed_concurrent_transport):
    # Four single-chunk calls isolate the shared cap of 2 across top-level calls.
    variables = [f"V{i:09d}" for i in range(4)]
    bodies = {var: _header("observationAbout") + _row(var, "country/USA") for var in variables}
    release = {var: threading.Event() for var in variables}
    transport = keyed_concurrent_transport(bodies, release=release)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        total_concurrency=2,
    )

    results: list[object] = []
    errors: list[BaseException] = []

    def _run(var: str) -> None:
        try:
            results.append(client.data(variables=[var], entities=_ENTITIES))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=_run, args=(var,)) for var in variables]
    for worker in workers:
        worker.start()
    try:
        # Confirm two GETs overlap before asserting a third stays queued.
        _wait_until(lambda: transport.peak_concurrent >= 2)
        time.sleep(0.05)
        assert transport.peak_concurrent == 2
        for event in release.values():
            event.set()
    finally:
        for worker in workers:
            worker.join(timeout=5)

    assert not any(worker.is_alive() for worker in workers)
    assert errors == []
    assert len(results) == 4
    assert transport.peak_concurrent == 2


def test_no_deadlock_when_concurrent_calls_outnumber_the_total(keyed_concurrent_transport):
    # Six calls queue behind 2 slots; each must finish within its 5s join timeout.
    variables = [f"V{i:09d}" for i in range(6)]
    bodies = {var: _header("observationAbout") + _row(var, "country/USA") for var in variables}
    transport = keyed_concurrent_transport(bodies)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        total_concurrency=2,
    )

    results: list[object] = []
    errors: list[BaseException] = []

    def _run(var: str) -> None:
        try:
            results.append(client.data(variables=[var], entities=_ENTITIES))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=_run, args=(var,)) for var in variables]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert not any(worker.is_alive() for worker in workers)
    assert errors == []
    assert len(results) == 6
    assert transport.peak_concurrent <= 2


def test_get_waiting_for_a_slot_past_the_deadline_raises_the_deadline_error(
    keyed_concurrent_transport,
):
    # The first call holds the sole slot; the second call's 0.08s deadline
    # expires while it waits, before its GET reaches the transport.
    hold_var = "HHHHHHHHHH"
    wait_var = "WWWWWWWWWW"
    bodies = {
        hold_var: _header("observationAbout") + _row(hold_var, "country/USA"),
        wait_var: _header("observationAbout") + _row(wait_var, "country/USA"),
    }
    release = {hold_var: threading.Event()}
    transport = keyed_concurrent_transport(bodies, release=release)
    client = SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=_BUDGET_ONE_PER_CHUNK,
        total_concurrency=1,
    )

    holder_errors: list[BaseException] = []

    def _hold() -> None:
        try:
            client.data(variables=[hold_var], entities=_ENTITIES)
        except BaseException as exc:
            holder_errors.append(exc)

    holder = threading.Thread(target=_hold)
    holder.start()
    try:
        _wait_until(lambda: transport.peak_concurrent >= 1)

        with pytest.raises(TimeoutError, match="absolute caller deadline"):
            client.data(
                variables=[wait_var],
                entities=_ENTITIES,
                deadline=time.monotonic() + 0.08,
            )

        # The waiting call never got a slot, so its GET never reached the
        # transport at all -- the deadline fired on the semaphore wait
        # itself, not on a request that was issued and then ran long.
        assert all(_variable_measured(r) != wait_var for r in transport.requests)
    finally:
        release[hold_var].set()
        holder.join(timeout=5)

    assert not holder.is_alive()
    assert holder_errors == []
