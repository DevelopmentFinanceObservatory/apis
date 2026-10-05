"""Prove per-call and instance-wide chunk limits with barriers and in-flight counts.
`node_arc_many_total_concurrency` is separate from the pool for batch reads.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from dc_graph import GraphClient, GraphDeadlineExceededError


def _reader(
    *,
    chunk_size: int,
    call_concurrency: int,
    total_concurrency: int,
    transport: httpx.BaseTransport | None = None,
) -> GraphClient:
    return GraphClient(
        base_url="https://graph.example.org/core/api",
        node_arc_chunk_size=chunk_size,
        node_arc_call_concurrency=call_concurrency,
        node_arc_many_total_concurrency=total_concurrency,
        transport=transport,
    )


def _fake_arcs_body(dcids: list[str]) -> dict:
    """Return one arc per dcid, avoiding the empty-map group retry."""
    return {
        "data": {
            d: {"arcs": {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}}} for d in dcids
        }
    }


def test_call_concurrency_bounded_and_saturated(monkeypatch) -> None:
    """The per-call cap stays at 3 and reaches 3 concurrent chunks across 5 chunks."""
    cap = 3
    dcids = [f"dcid/{i}" for i in range(10)]
    barrier = threading.Barrier(cap, timeout=5.0)
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0
    started = 0

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal in_flight, max_in_flight, started
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            my_index = started
            started += 1
        if my_index < cap:
            barrier.wait(timeout=5.0)
        body = _fake_arcs_body(list(payload["nodes"]))
        with lock:
            in_flight -= 1
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    reader = _reader(chunk_size=2, call_concurrency=cap, total_concurrency=100)

    result = reader.node_arcs_many(dcids, properties=("typeOf",))

    assert set(result.keys()) == set(dcids)
    assert max_in_flight <= cap


def test_total_concurrency_is_instance_wide_bounded(monkeypatch) -> None:
    """The instance cap of 4 bounds POSTs across three calls and 15 chunks.

    Each call's cap is 8, so only the shared semaphore limits in-flight POSTs.
    """
    total_cap = 4
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        body = _fake_arcs_body(list(payload["nodes"]))
        with lock:
            in_flight -= 1
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    # The constructor requires total_concurrency >= call_concurrency, so set
    # the tighter test cap on the instance semaphore after construction.
    reader = _reader(chunk_size=2, call_concurrency=8, total_concurrency=8)
    reader._node_arc_many_semaphore = threading.BoundedSemaphore(total_cap)

    results: list[dict] = [None, None, None]  # type: ignore[list-item]

    def _worker(call_index: int) -> None:
        dcids = [f"thread{call_index}/dcid/{i}" for i in range(10)]
        results[call_index] = reader.node_arcs_many(dcids, properties=("typeOf",))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung past the join timeout"

    for call_index, result in enumerate(results):
        expected = {f"thread{call_index}/dcid/{i}" for i in range(10)}
        assert set(result.keys()) == expected
    assert max_in_flight <= total_cap


def test_no_deadlock_when_callers_outnumber_the_total(monkeypatch) -> None:
    """Six calls queued behind a cap of 2 all finish without deadlock."""
    total_cap = 2

    def fake_post(self, path, payload, *, deadline=None):
        time.sleep(0.02)
        return _fake_arcs_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)
    reader = _reader(chunk_size=2, call_concurrency=4, total_concurrency=4)
    reader._node_arc_many_semaphore = threading.BoundedSemaphore(total_cap)

    results: list[dict] = [None] * 6  # type: ignore[list-item]

    def _worker(call_index: int) -> None:
        dcids = [f"thread{call_index}/dcid/{i}" for i in range(8)]
        results[call_index] = reader.node_arcs_many(dcids, properties=("typeOf",))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung past the join timeout"

    for call_index, result in enumerate(results):
        expected = {f"thread{call_index}/dcid/{i}" for i in range(8)}
        assert set(result.keys()) == expected


def test_merged_result_identical_regardless_of_chunk_completion_order(monkeypatch) -> None:
    """The same 12-dcid result holds when four chunks finish out of order."""
    dcids = [f"dcid/{i}" for i in range(12)]

    def fake_post(self, path, payload, *, deadline=None):
        chunk_dcids = list(payload["nodes"])
        # Sleep longer for a chunk whose dcids sort earlier, so later
        # chunks finish first under real concurrency.
        time.sleep(0.01 * (12 - int(chunk_dcids[0].split("/")[1])))
        return _fake_arcs_body(chunk_dcids)

    monkeypatch.setattr(GraphClient, "post", fake_post)

    serial_reader = _reader(chunk_size=2, call_concurrency=1, total_concurrency=1)
    serial_result = serial_reader.node_arcs_many(dcids, properties=("typeOf",))

    concurrent_reader = _reader(chunk_size=2, call_concurrency=4, total_concurrency=8)
    concurrent_result = concurrent_reader.node_arcs_many(dcids, properties=("typeOf",))

    assert concurrent_result == serial_result
    assert set(concurrent_result.keys()) == set(dcids)


def test_one_chunk_calls_serialize_at_total_concurrency_one(monkeypatch) -> None:
    """The instance cap of 1 serialises two concurrent one-chunk reads."""
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        body = _fake_arcs_body(list(payload["nodes"]))
        with lock:
            in_flight -= 1
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    reader = _reader(chunk_size=10, call_concurrency=4, total_concurrency=4)
    reader._node_arc_many_semaphore = threading.BoundedSemaphore(1)

    results: list[dict] = [None, None]  # type: ignore[list-item]

    def _worker(call_index: int) -> None:
        results[call_index] = reader.node_arcs_many([f"dcid/{call_index}"], properties=("typeOf",))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung past the join timeout"

    assert max_in_flight <= 1
    assert set(results[0].keys()) == {"dcid/0"}
    assert set(results[1].keys()) == {"dcid/1"}


def test_group_retry_also_holds_a_node_arc_slot(monkeypatch) -> None:
    """Initial reads and empty-map retries share the instance cap of 1."""
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0
    seen_once: set[str] = set()

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal in_flight, max_in_flight
        nodes = list(payload["nodes"])
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        if len(nodes) == 1 and nodes[0] in seen_once:
            # The group retry, scoped to just the suspect dcid.
            return _fake_arcs_body(nodes)
        with lock:
            seen_once.update(nodes)
        empty_dcid = nodes[0]
        full_dcid = nodes[1]
        return {
            "data": {
                empty_dcid: {"arcs": {}},
                full_dcid: _fake_arcs_body([full_dcid])["data"][full_dcid],
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    reader = _reader(chunk_size=10, call_concurrency=4, total_concurrency=4)
    reader._node_arc_many_semaphore = threading.BoundedSemaphore(1)

    results: list[dict] = [None, None]  # type: ignore[list-item]

    def _worker(call_index: int) -> None:
        dcids = [f"thread{call_index}/empty", f"thread{call_index}/full"]
        results[call_index] = reader.node_arcs_many(dcids, properties=("typeOf",))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung past the join timeout"

    assert max_in_flight <= 1
    for call_index, result in enumerate(results):
        assert result[f"thread{call_index}/empty"] is not None
        assert result[f"thread{call_index}/full"] is not None


def test_deadline_exceeded_while_waiting_for_a_node_arc_slot(monkeypatch) -> None:
    """A spent deadline raises while waiting for the sole slot, before POST."""

    def _unreachable_post(self, path, payload, *, deadline=None):
        raise AssertionError("must not be called: no slot was ever free")

    monkeypatch.setattr(GraphClient, "post", _unreachable_post)
    reader = _reader(chunk_size=10, call_concurrency=1, total_concurrency=1)
    assert reader._node_arc_many_semaphore.acquire(timeout=1.0)
    try:
        short_deadline = time.monotonic() + 0.1
        with pytest.raises(GraphDeadlineExceededError):
            reader.node_arcs_many(["dcid/a"], properties=("typeOf",), deadline=short_deadline)
    finally:
        reader._node_arc_many_semaphore.release()


def test_multi_chunk_read_completes_at_total_concurrency_one(monkeypatch) -> None:
    """Six chunks complete with one slot; nested acquisition would deadlock."""
    monkeypatch.setattr(
        GraphClient,
        "post",
        lambda self, path, payload, *, deadline=None: _fake_arcs_body(list(payload["nodes"])),
    )
    reader = _reader(chunk_size=2, call_concurrency=1, total_concurrency=1)
    dcids = [f"dcid/{i}" for i in range(6)]

    result: dict | None = None

    def _run() -> None:
        nonlocal result
        result = reader.node_arcs_many(dcids, properties=("typeOf",))

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=10.0)
    assert not thread.is_alive(), "multi-chunk read hung at total_concurrency=1 (self-deadlock)"
    assert result is not None
    assert set(result.keys()) == set(dcids)


def test_deadline_is_honoured_across_concurrent_chunks() -> None:
    """An expired deadline raises before any of the 5 chunk requests reaches HTTP."""

    def _unreachable_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called: the deadline was already spent")

    reader = _reader(
        chunk_size=2,
        call_concurrency=4,
        total_concurrency=8,
        transport=httpx.MockTransport(_unreachable_handler),
    )
    dcids = [f"dcid/{i}" for i in range(10)]
    past_deadline = time.monotonic() - 1.0

    with pytest.raises(GraphDeadlineExceededError):
        reader.node_arcs_many(dcids, properties=("typeOf",), deadline=past_deadline)
