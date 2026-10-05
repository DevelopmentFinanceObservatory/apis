"""Concurrent cold reads share one POST per chunk and respect each caller's deadline.
The fake transport keeps these tests offline.
"""

from __future__ import annotations

import threading
import time

from dc_graph import GraphClient, declared_role_refresh_deadline
from dc_graph.client import _DEFAULT_NODE_ARC_BATCH_CHUNK_SIZE

_CHUNK_SIZE = _DEFAULT_NODE_ARC_BATCH_CHUNK_SIZE


def _fake_post_body(dcids: list[str]) -> dict:
    """Return one resolving arc per dcid so tests can count chunk calls."""
    return {
        "data": {
            d: {"arcs": {"observationProperties": {"nodes": [{"dcid": "observationAbout"}]}}}
            for d in dcids
        }
    }


def test_two_concurrent_cold_reads_issue_one_post_per_chunk_and_both_get_the_full_result(
    monkeypatch,
) -> None:
    """Two cold reads share 3 POSTs, one for each of the 3 chunks."""
    client = GraphClient(base_url="http://fake/core/api")

    dcids = [f"dcid/{i}" for i in range(3 * _CHUNK_SIZE)]
    posted_chunks: list[list[str]] = []
    lock = threading.Lock()
    both_may_start = threading.Barrier(2, timeout=5.0)

    def fake_post(self, path: str, payload: dict, *, deadline=None) -> dict:
        chunk = list(payload["nodes"])
        with lock:
            posted_chunks.append(chunk)
        # Keep the first fetch open long enough for the second caller to join it.
        time.sleep(0.05)
        return _fake_post_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)

    results: list[dict | None] = [None, None]

    def _call(slot: int) -> None:
        both_may_start.wait(timeout=5.0)
        results[slot] = client.node_observation_properties_batch(dcids)

    t1 = threading.Thread(target=_call, args=(0,))
    t2 = threading.Thread(target=_call, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not t1.is_alive() and not t2.is_alive(), "a concurrent call hung past its join timeout"
    assert len(posted_chunks) == 3, (
        f"expected exactly one POST per chunk (3) shared between both callers; "
        f"got {len(posted_chunks)} POST(s): {posted_chunks!r}"
    )
    first, second = results
    assert first is not None and second is not None, f"a call returned no result: {results!r}"
    assert first == second
    assert set(first.keys()) == set(dcids)
    for dcid in dcids:
        assert first[dcid] == ("observationAbout",)


def test_waiter_with_a_short_deadline_times_out_while_the_leader_completes(monkeypatch) -> None:
    """The waiter's 0.3s deadline expires without cancelling the leader's fetch."""
    client = GraphClient(base_url="http://fake/core/api")

    leader_may_finish = threading.Event()
    leader_post_started = threading.Event()

    def fake_post(self, path: str, payload: dict, *, deadline=None) -> dict:
        leader_post_started.set()
        released = leader_may_finish.wait(timeout=10.0)
        assert released, "test failed to release the leader before its own timeout"
        return _fake_post_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)

    dcids = [f"dcid/{i}" for i in range(3)]  # single chunk, well under _CHUNK_SIZE
    leader_result: dict[str, object] = {}
    waiter_result: dict[str, object] = {}

    def _leader() -> None:
        try:
            leader_result["result"] = client.node_observation_properties_batch(dcids)
        except BaseException as exc:
            leader_result["error"] = exc

    def _waiter() -> None:
        deadline = time.monotonic() + 0.3
        with declared_role_refresh_deadline(deadline):
            try:
                waiter_result["result"] = client.node_observation_properties_batch(dcids)
            except BaseException as exc:
                waiter_result["error"] = exc

    leader_thread = threading.Thread(target=_leader)
    leader_thread.start()
    assert leader_post_started.wait(timeout=5.0), "leader never started its POST"

    waiter_thread = threading.Thread(target=_waiter)
    waiter_thread.start()

    waiter_thread.join(timeout=5.0)
    assert not waiter_thread.is_alive(), "waiter hung past its own deadline"
    assert isinstance(waiter_result.get("error"), TimeoutError), (
        f"expected the waiter to time out on its own short deadline without disturbing "
        f"the leader's fetch; got {waiter_result!r}"
    )

    leader_may_finish.set()
    leader_thread.join(timeout=5.0)
    assert not leader_thread.is_alive(), "leader hung after being released"
    leader_fetch_result = leader_result.get("result")
    assert isinstance(leader_fetch_result, dict), (
        f"the leader's own fetch must still succeed, undisturbed by the waiter giving "
        f"up on its own shorter deadline; got {leader_result!r}"
    )
    for dcid in dcids:
        assert leader_fetch_result.get(dcid) == ("observationAbout",)
