"""Tests deadline-aware chunk scheduling in `node_observation_properties_batch`.

Fake clocks, barriers, and events control expiry and prove withheld reads are not posted. Calls
outside a deadline still process every chunk.
"""

from __future__ import annotations

import threading
import time as _real_time

import pytest

import dc_graph.client as graph_module
from dc_graph import GraphClient, GraphReadError, declared_role_refresh_deadline

_CHUNK_SIZE = graph_module._DEFAULT_NODE_ARC_BATCH_CHUNK_SIZE


def _fake_post_body(dcids: list[str]) -> dict:
    """Return one valid observationProperties arc for each requested dcid."""
    return {
        "data": {
            d: {"arcs": {"observationProperties": {"nodes": [{"dcid": "observationAbout"}]}}}
            for d in dcids
        }
    }


class _FakeClock:
    """A monotonic clock advanced directly by the fake POST."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now


class _PatchedTime:
    """Forward real time functions and use the fake clock for `monotonic`."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(_real_time, name)


def _client(*, call_concurrency: int | None = None) -> GraphClient:
    kwargs = {} if call_concurrency is None else {"node_arc_call_concurrency": call_concurrency}
    client = GraphClient(base_url="http://fake/core/api", **kwargs)
    # Keep the process-wide total bound wide open so only the per-call
    # scheduler (and its deadline check) is under test.
    client._node_arc_semaphore = threading.BoundedSemaphore(100)
    return client


def test_chunk_scheduling_stops_issuing_new_posts_once_the_deadline_elapses(monkeypatch) -> None:
    """A barrier starts both first-wave POSTs across 6 chunks at concurrency 2.

    After expiry, no third chunk is posted and the call raises `TimeoutError`.
    """
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(graph_module, "time", _PatchedTime(clock))
    client = _client(call_concurrency=2)
    deadline = 100.0

    dcids = [f"dcid/{i}" for i in range(6 * _CHUNK_SIZE)]
    barrier = threading.Barrier(2, timeout=5.0)
    lock = threading.Lock()
    posted_chunks: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        with lock:
            posted_chunks.append(chunk)
        # Block until both of the first wave's POSTs have started, so
        # neither can advance the clock before the scheduler has genuinely
        # dispatched both -- only the first two calls ever reach this
        # barrier, since scheduling stops once the clock trips.
        barrier.wait(timeout=5.0)
        with lock:
            clock.now = deadline + 1.0
        return _fake_post_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)

    with declared_role_refresh_deadline(deadline), pytest.raises(TimeoutError):
        client.node_observation_properties_batch(dcids)

    assert len(posted_chunks) == 2, (
        f"expected exactly the first 2-chunk wave to be dispatched before scheduling "
        f"stopped; got {len(posted_chunks)} POST(s)"
    )


def test_a_deadline_already_elapsed_issues_no_post_at_all(monkeypatch) -> None:
    """An expired deadline prevents every chunk POST."""
    clock = _FakeClock(start=200.0)
    monkeypatch.setattr(graph_module, "time", _PatchedTime(clock))
    client = _client()

    dcids = [f"dcid/{i}" for i in range(3 * _CHUNK_SIZE)]
    posted_chunks: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        posted_chunks.append(list(payload["nodes"]))
        return _fake_post_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)

    with declared_role_refresh_deadline(100.0), pytest.raises(TimeoutError):
        client.node_observation_properties_batch(dcids)

    assert posted_chunks == []


def test_no_deadline_scope_reads_to_completion_unchanged(monkeypatch) -> None:
    """Without a deadline scope, every chunk is read."""
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(graph_module, "time", _PatchedTime(clock))
    client = _client(call_concurrency=2)

    dcids = [f"dcid/{i}" for i in range(6 * _CHUNK_SIZE)]
    posted_chunks: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        with threading.Lock():
            posted_chunks.append(chunk)
            # Even a clock that reads arbitrarily far in the future never
            # trips a limit with no scope open to carry a deadline at all.
            clock.now = 10_000.0
        return _fake_post_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)

    result = client.node_observation_properties_batch(dcids)

    assert set(result.keys()) == set(dcids)
    assert len(posted_chunks) == 6


class _AnnouncingSemaphore:
    """Signals when a worker enters `acquire`, so the test can expire the deadline while it waits."""

    def __init__(self, value: int, about_to_block: threading.Event) -> None:
        self._sem = threading.Semaphore(value)
        self._about_to_block = about_to_block

    def acquire(self, *args, **kwargs):
        self._about_to_block.set()
        return self._sem.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self._sem.release(*args, **kwargs)


def test_chunk_that_clears_the_semaphore_after_the_deadline_is_never_posted(monkeypatch) -> None:
    """A worker queued on a held semaphore must check its deadline before posting.

    The fake POST records reads; the test expires the clock only after `about_to_block` fires.
    """
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(graph_module, "time", _PatchedTime(clock))
    client = GraphClient(base_url="http://fake/core/api", node_arc_call_concurrency=1)

    about_to_block = threading.Event()
    semaphore = _AnnouncingSemaphore(1, about_to_block)
    # Hold the one slot as "other requests' traffic" would, before handing
    # the semaphore to the client under test.
    semaphore.acquire()
    about_to_block.clear()
    client._node_arc_semaphore = semaphore

    deadline = 50.0
    posted_chunks: list[list[str]] = []
    lock = threading.Lock()

    def fake_post(self, path, payload, *, deadline=None):
        with lock:
            posted_chunks.append(list(payload["nodes"]))
        return _fake_post_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)

    # Two chunks at concurrency 1, so the second is never even reached once
    # the first is withheld.
    dcids = [f"dcid/{i}" for i in range(2 * _CHUNK_SIZE)]
    outcome: dict[str, object] = {}

    def _run() -> None:
        with declared_role_refresh_deadline(deadline):
            try:
                outcome["result"] = client.node_observation_properties_batch(dcids)
            except BaseException as exc:
                outcome["error"] = exc

    thread = threading.Thread(target=_run)
    thread.start()

    assert about_to_block.wait(timeout=5.0), (
        "worker never reached the semaphore queue -- test setup did not "
        "exercise the race under test"
    )
    with lock:
        clock.now = deadline + 1.0
    semaphore.release()

    thread.join(timeout=5.0)
    assert not thread.is_alive(), "worker thread hung past the join timeout"

    assert posted_chunks == [], (
        f"a chunk whose worker reached the semaphore only after the deadline "
        f"must never POST; got {posted_chunks!r}"
    )
    assert isinstance(outcome.get("error"), TimeoutError), f"expected TimeoutError, got {outcome!r}"


def test_multi_chunk_read_completes_unchanged_when_deadline_never_elapses(monkeypatch) -> None:
    """A live deadline leaves the six-chunk read complete when the fake clock stays within budget."""
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(graph_module, "time", _PatchedTime(clock))
    client = _client(call_concurrency=2)

    dcids = [f"dcid/{i}" for i in range(6 * _CHUNK_SIZE)]
    posted_chunks: list[list[str]] = []
    lock = threading.Lock()

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        with lock:
            posted_chunks.append(chunk)
        return _fake_post_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)

    with declared_role_refresh_deadline(10_000.0):
        result = client.node_observation_properties_batch(dcids)

    assert set(result.keys()) == set(dcids)
    for dcid in dcids:
        assert result[dcid] == ("observationAbout",)
    assert len(posted_chunks) == 6


def test_call_returns_without_waiting_for_a_still_running_chunk(monkeypatch) -> None:
    """The headline mechanism of the finding, not just the post-semaphore
    recheck above: once the deadline trips, node_observation_properties_batch
    must return (raising TimeoutError) without blocking on a chunk whose own
    POST is still running -- proving pool.shutdown's `wait=not
    budget_exhausted` genuinely skips the wait rather than merely avoiding a
    second raise while still waiting underneath. Verified by perturbation:
    reverting that shutdown call to an unconditional wait=True (the exact
    defect under test) makes this test fail; the fixed code passes.

    Uses a real wall-clock deadline (no _FakeClock/_PatchedTime here -- the
    thing under test is whether the call blocks in real time, so the
    deadline must be real too) and an explicit, bounded thread join, not a
    bare call: this repo has no pytest-timeout and no timeout markers, so a
    regression to unconditional wait=True would otherwise wedge the whole
    suite instead of failing this one test. One chunk's fake POST blocks on
    a threading.Event this test alone controls, held un-set through every
    assertion below; the call runs on its own thread so this test's own
    thread can bound the wait with `join(timeout=...)`. The event is
    released only at the very end, so the worker drains and no thread is
    left blocked when the test exits."""
    client = GraphClient(base_url="http://fake/core/api", node_arc_call_concurrency=2)
    client._node_arc_semaphore = threading.BoundedSemaphore(100)

    release_worker = threading.Event()
    post_started = threading.Event()
    post_finished = threading.Event()

    def fake_post(self, path, payload, *, deadline=None):
        post_started.set()
        released = release_worker.wait(timeout=10.0)
        assert released, "test failed to release the blocked worker before its own timeout"
        post_finished.set()
        return _fake_post_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)

    # Two chunks so the multi-chunk executor path (not the inline
    # single-chunk shortcut, which runs on the caller's own thread) is
    # exercised.
    dcids = [f"dcid/{i}" for i in range(2 * _CHUNK_SIZE)]
    outcome: dict[str, object] = {}
    deadline = _real_time.monotonic() + 0.1

    def _run() -> None:
        with declared_role_refresh_deadline(deadline):
            try:
                outcome["result"] = client.node_observation_properties_batch(dcids)
            except BaseException as exc:
                outcome["error"] = exc

    driver = threading.Thread(target=_run)
    driver.start()

    assert post_started.wait(timeout=5.0), "worker never started its POST"
    # The deadline (~0.1s away) trips while the worker above is still
    # blocked on release_worker; the call must return well within this join
    # timeout rather than waiting on that worker.
    driver.join(timeout=3.0)
    assert not driver.is_alive(), (
        "node_observation_properties_batch blocked on a still-running chunk "
        "instead of returning once the deadline elapsed"
    )
    assert isinstance(outcome.get("error"), TimeoutError), f"expected TimeoutError, got {outcome!r}"
    assert not post_finished.is_set(), (
        "the blocked worker had already finished -- this run never actually "
        "proved the call returned ahead of it"
    )

    release_worker.set()
    assert post_finished.wait(timeout=5.0), "blocked worker never drained after being released"


def test_single_chunk_past_deadline_raises_without_posting(monkeypatch) -> None:
    """The single-chunk path (no executor, see _fetch_observation_property_chunks'
    own docstring) checks the deadline before its one inline read too."""
    clock = _FakeClock(start=500.0)
    monkeypatch.setattr(graph_module, "time", _PatchedTime(clock))
    client = _client()

    dcids = [f"dcid/{i}" for i in range(3)]
    posted_chunks: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        posted_chunks.append(list(payload["nodes"]))
        return _fake_post_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)

    with declared_role_refresh_deadline(100.0), pytest.raises(TimeoutError):
        client.node_observation_properties_batch(dcids)

    assert posted_chunks == []


def test_single_chunk_returns_at_the_deadline_not_after_full_contention(monkeypatch) -> None:
    """A contended single-chunk read raises at the deadline instead of waiting for the held slot.

    This test uses real wall-clock time, not the fake clock.
    """
    client = GraphClient(base_url="http://fake/core/api")
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()  # simulate another request's traffic already holding the one slot
    client._node_arc_semaphore = semaphore

    hold_seconds = 2.0

    def _release_later() -> None:
        _real_time.sleep(hold_seconds)
        semaphore.release()

    threading.Thread(target=_release_later, daemon=True).start()

    posted_chunks: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        posted_chunks.append(list(payload["nodes"]))
        return _fake_post_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)

    dcids = [f"dcid/{i}" for i in range(3)]  # well under chunk size: single-chunk path
    deadline_headroom = 0.3
    deadline = _real_time.monotonic() + deadline_headroom

    start = _real_time.monotonic()
    with declared_role_refresh_deadline(deadline), pytest.raises(TimeoutError):
        client.node_observation_properties_batch(dcids)
    elapsed = _real_time.monotonic() - start

    assert posted_chunks == [], "the withheld chunk must never be POSTed"
    assert elapsed < hold_seconds * 0.5, (
        f"expected the call to return near the {deadline_headroom:.1f}s deadline, not "
        f"wait out the full {hold_seconds:.1f}s of contention; measured {elapsed:.2f}s"
    )


class _AnnouncingSemaphoreForSingleChunk:
    """Signals when a single-chunk worker enters `acquire`."""

    def __init__(self, value: int, about_to_block: threading.Event) -> None:
        self._sem = threading.Semaphore(value)
        self._about_to_block = about_to_block

    def acquire(self, *args, **kwargs):
        self._about_to_block.set()
        return self._sem.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self._sem.release(*args, **kwargs)


def test_single_chunk_that_clears_the_semaphore_after_the_deadline_raises_timeouterror(
    monkeypatch,
) -> None:
    """The inline single-chunk path translates an expired semaphore wait to `TimeoutError`."""
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(graph_module, "time", _PatchedTime(clock))
    client = GraphClient(base_url="http://fake/core/api")

    about_to_block = threading.Event()
    semaphore = _AnnouncingSemaphoreForSingleChunk(1, about_to_block)
    semaphore.acquire()
    about_to_block.clear()
    client._node_arc_semaphore = semaphore

    deadline = 50.0
    posted_chunks: list[list[str]] = []
    lock = threading.Lock()

    def fake_post(self, path, payload, *, deadline=None):
        with lock:
            posted_chunks.append(list(payload["nodes"]))
        return _fake_post_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)

    dcids = [f"dcid/{i}" for i in range(3)]  # well under chunk size: single-chunk path
    outcome: dict[str, object] = {}

    def _run() -> None:
        with declared_role_refresh_deadline(deadline):
            try:
                outcome["result"] = client.node_observation_properties_batch(dcids)
            except BaseException as exc:
                outcome["error"] = exc

    thread = threading.Thread(target=_run)
    thread.start()

    assert about_to_block.wait(timeout=5.0), (
        "worker never reached the semaphore queue -- test setup did not exercise the race"
    )
    with lock:
        clock.now = deadline + 1.0
    semaphore.release()

    thread.join(timeout=5.0)
    assert not thread.is_alive(), "worker thread hung past the join timeout"

    assert posted_chunks == [], (
        f"a single chunk whose worker reached the semaphore only after the deadline "
        f"must never POST; got {posted_chunks!r}"
    )
    assert isinstance(outcome.get("error"), TimeoutError), f"expected TimeoutError, got {outcome!r}"


def test_exceptional_path_returns_promptly_without_waiting_for_slow_siblings(monkeypatch) -> None:
    """Finding 2b: once one chunk's own read fails outright (a genuine
    GraphReadError, unrelated to any deadline), the `except Exception:`
    handler in _fetch_observation_property_chunks must not block the caller
    on still-running sibling chunks' own POSTs. Verified by perturbation:
    reverting pool.shutdown(wait=False, cancel_futures=True) there back to
    the unconditional wait=True the defect used makes this test fail.

    Uses real threading.Event synchronization and a bounded thread join,
    matching test_call_returns_without_waiting_for_a_still_running_chunk's
    own real-threading style, since a regression to wait=True would
    otherwise wedge the whole suite rather than failing only this test. No
    declared_role_refresh_deadline scope is opened -- this failure is
    unrelated to any time budget."""
    client = GraphClient(
        base_url="http://fake/core/api", node_arc_call_concurrency=4, node_arc_batch_chunk_size=1
    )
    client._node_arc_semaphore = threading.BoundedSemaphore(100)

    release_siblings = threading.Event()
    failing_chunk_started = threading.Event()

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        if chunk[0] == "dcid/0":
            failing_chunk_started.set()
            raise GraphReadError("simulated transport failure")
        released = release_siblings.wait(timeout=10.0)
        assert released, "test failed to release sibling chunks before its own timeout"
        return _fake_post_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)

    dcids = [f"dcid/{i}" for i in range(4)]
    outcome: dict[str, object] = {}

    def _run() -> None:
        try:
            outcome["result"] = client.node_observation_properties_batch(dcids)
        except BaseException as exc:
            outcome["error"] = exc

    driver = threading.Thread(target=_run)
    driver.start()

    assert failing_chunk_started.wait(timeout=5.0), "the failing chunk never started"
    driver.join(timeout=3.0)
    assert not driver.is_alive(), (
        "node_observation_properties_batch blocked on still-running siblings after "
        "another chunk had already failed"
    )
    assert isinstance(outcome.get("error"), GraphReadError), (
        f"expected GraphReadError, got {outcome!r}"
    )

    release_siblings.set()
