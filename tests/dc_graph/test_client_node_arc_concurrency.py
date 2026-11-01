"""Pins `SEARCH_NODE_ARC_CALL_CONCURRENCY` and `SEARCH_NODE_ARC_TOTAL_CONCURRENCY` with barriers.
Label and type batch reads do not use the node-arc semaphore."""

from __future__ import annotations

import threading
import time

import pytest

import dc_graph.client as graph_module
from dc_graph import GraphClient

_CHUNK_SIZE = graph_module._DEFAULT_NODE_ARC_BATCH_CHUNK_SIZE


def _fake_post_body(dcids: list[str]) -> dict:
    """One non-empty arcs row per dcid -- an empty arcs map would read as
    suspicious and trigger node_arcs_batch's own retry POST, which would
    confound the in-flight count this module measures."""
    return {"data": {d: {"arcs": {"p": {"nodes": [{"dcid": f"v/{d}"}]}}} for d in dcids}}


def test_node_arcs_batch_per_call_concurrency_bounded_and_saturated(monkeypatch) -> None:
    """A barrier fills all 3 slots across five chunks (60 dcids).
    The in-flight count stays bounded."""
    cap = 3
    dcids = [f"dcid/{i}" for i in range(5 * _CHUNK_SIZE)]
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
        body = _fake_post_body(list(payload["nodes"]))
        with lock:
            in_flight -= 1
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", node_arc_call_concurrency=cap)
    # Keep the process-wide total bound wide open so only the per-call cap binds.
    client._node_arc_semaphore = threading.BoundedSemaphore(100)

    result = client.node_arcs_batch(dcids)

    assert set(result.keys()) == set(dcids)
    assert max_in_flight <= cap


def test_node_arcs_batch_total_concurrency_is_process_wide_bounded(monkeypatch) -> None:
    """Three calls send 15 chunks through one client. The shared semaphore caps POSTs at 4.
    Each call's cap is 8, so only the client-wide bound constrains their combined traffic."""
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
        body = _fake_post_body(list(payload["nodes"]))
        with lock:
            in_flight -= 1
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", node_arc_call_concurrency=8)
    client._node_arc_semaphore = threading.BoundedSemaphore(total_cap)

    results: list[dict] = [None, None, None]  # type: ignore[list-item]

    def _worker(call_index: int) -> None:
        dcids = [f"thread{call_index}/dcid/{i}" for i in range(5 * _CHUNK_SIZE)]
        results[call_index] = client.node_arcs_batch(dcids)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung past the join timeout"

    for call_index, result in enumerate(results):
        expected = {f"thread{call_index}/dcid/{i}" for i in range(5 * _CHUNK_SIZE)}
        assert set(result.keys()) == expected
    assert max_in_flight <= total_cap


def test_one_chunk_calls_serialize_at_total_concurrency_1(monkeypatch) -> None:
    """Two concurrent single-chunk reads serialize through a shared semaphore of capacity 1."""
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.1)
        body = _fake_post_body(list(payload["nodes"]))
        with lock:
            in_flight -= 1
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")
    client._node_arc_semaphore = threading.BoundedSemaphore(1)

    results: list[dict] = [None, None]  # type: ignore[list-item]

    def _worker(call_index: int) -> None:
        dcids = [f"call{call_index}/dcid/{i}" for i in range(3)]  # far under chunk_size
        results[call_index] = client.node_arcs_batch(dcids)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung past the join timeout"

    for call_index, result in enumerate(results):
        assert set(result.keys()) == {f"call{call_index}/dcid/{i}" for i in range(3)}
    assert max_in_flight == 1


def test_one_chunk_halving_retry_holds_a_slot_at_total_concurrency_1(monkeypatch) -> None:
    """Two public single-chunk reads share a one-slot semaphore. One read splits into 3 POSTs.

    The other reads once, and no POSTs overlap during the halving retry."""
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0
    calls_a: list[list[str]] = []

    dcids_a = [f"A/dcid/{i}" for i in range(4)]
    dcids_b = [f"B/dcid/{i}" for i in range(3)]

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal in_flight, max_in_flight
        nodes = list(payload["nodes"])
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.1)
        with lock:
            in_flight -= 1
        if nodes and nodes[0].startswith("A/"):
            with lock:
                calls_a.append(nodes)
            if len(nodes) == len(dcids_a):
                # dcids_a's own initial POST over its full chunk -- forces
                # the halving split below.
                return {"data": {}, "nextToken": "tok"}
        return _fake_post_body(nodes)

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")
    client._node_arc_semaphore = threading.BoundedSemaphore(1)

    results: dict[str, dict] = {}

    def _run_a() -> None:
        results["a"] = client.node_arcs_batch(dcids_a)

    def _run_b() -> None:
        results["b"] = client.node_arcs_batch(dcids_b)

    threads = [threading.Thread(target=_run_a), threading.Thread(target=_run_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung past the join timeout"

    assert set(results["a"].keys()) == set(dcids_a)
    assert set(results["b"].keys()) == set(dcids_b)
    # The continuation-reporting initial POST plus one POST per half.
    assert len(calls_a) == 3
    assert max_in_flight == 1


def test_labels_and_types_batch_take_no_node_arc_slot(monkeypatch) -> None:
    """Label and type batches complete while the only node-arc semaphore slot is held."""

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        prop = payload["property"]

        def _arcs(d: str) -> dict:
            if prop == "->name":
                return {"name": {"nodes": [{"value": f"label/{d}"}]}}
            return {"typeOf": {"nodes": [{"dcid": f"type/{d}"}]}}

        return {"data": {d: {"arcs": _arcs(d)} for d in nodes}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")
    client._node_arc_semaphore = threading.BoundedSemaphore(1)
    held = client._node_arc_semaphore.acquire()
    assert held

    results: dict[str, dict] = {}

    def _run_labels() -> None:
        results["labels"] = client.node_labels_batch(["dcid/0", "dcid/1"])

    def _run_types() -> None:
        results["types"] = client.node_types_batch(["dcid/2", "dcid/3"])

    threads = [threading.Thread(target=_run_labels), threading.Thread(target=_run_types)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung waiting for the held node-arc slot"

    assert results["labels"] == {"dcid/0": "label/dcid/0", "dcid/1": "label/dcid/1"}
    assert results["types"] == {"dcid/2": ("type/dcid/2",), "dcid/3": ("type/dcid/3",)}


@pytest.mark.parametrize("arc", ["name", "typeOf"])
def test_label_and_type_chunks_post_concurrently(monkeypatch, arc) -> None:
    """Three chunks meet at a three-party barrier, which a one-at-a-time read never fills. Read
    one at a time, the labels and types of a 5,173-place scope took 21 s on dev."""
    cap = 3
    barrier = threading.Barrier(cap, timeout=5.0)

    def fake_post(self, path, payload, *, deadline=None):
        barrier.wait()
        field = "value" if arc == "name" else "dcid"
        return {
            "data": {
                d: {"arcs": {arc: {"nodes": [{field: f"{arc}/{d}"}]}}} for d in payload["nodes"]
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api",
        node_arc_call_concurrency=cap,
        single_arc_batch_chunk_size=2,
    )
    dcids = [f"dcid/{i}" for i in range(2 * cap)]

    if arc == "name":
        assert client.node_labels_batch(dcids) == {d: f"{arc}/{d}" for d in dcids}
    else:
        assert client.node_types_batch(dcids) == {d: (f"{arc}/{d}",) for d in dcids}


def test_multi_chunk_call_completes_at_total_concurrency_1(monkeypatch) -> None:
    """Three chunks finish with the shared semaphore at capacity 1.

    Each POST acquires the slot once. An executor-level acquire would deadlock."""
    dcids = [f"dcid/{i}" for i in range(3 * _CHUNK_SIZE)]

    def fake_post(self, path, payload, *, deadline=None):
        return _fake_post_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")
    client._node_arc_semaphore = threading.BoundedSemaphore(1)

    result_holder: dict[str, dict] = {}

    def _run() -> None:
        result_holder["result"] = client.node_arcs_batch(dcids)

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=10.0)
    assert not thread.is_alive(), "multi-chunk call deadlocked at total concurrency 1"

    assert set(result_holder["result"].keys()) == set(dcids)
