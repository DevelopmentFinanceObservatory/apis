"""Tests for per-pair observation fetch concurrency. Events and barriers prove
ordering and saturation without relying on elapsed-time guesses."""

from __future__ import annotations

import contextvars
import threading
import time
from unittest.mock import patch

import pytest

import search.engine.region.observations.fetch as region_fetch
from dfo_platform.request_deadline import request_deadline
from materialise import (
    MaterialisationUsageError,
)
from search.engine.region.codes import (
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_TIMED_OUT,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    build_observations_planned,
    harness_reader,
    sdmx_double,
    stub_preview_result,
)
from tests.fixtures import FakeGraph


def test_build_observations_preview_series_stay_in_pair_order_under_inverted_completion():
    """The last worker returns before the first starts. Equal values force the
    StatVar and role-tuple tie-breaks, which keep series in entity order.
    """
    entities = ["country/AAA", "country/BBB", "country/CCC"]
    later_entities_answered = threading.Event()

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        entity = next(iter(request.role_bindings["observationAbout"]))
        if entity == entities[0]:
            assert later_entities_answered.wait(timeout=5.0), (
                "the later entities never answered ahead of the first"
            )
        result = stub_preview_result(sv="Count_Person", entity=entity)
        if entity == entities[-1]:
            later_entities_answered.set()
        return result

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=harness_reader(entities),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert [s.entities[0].ref for s in observations.series] == entities


def test_build_observations_preview_concurrency_is_bounded_and_saturated(search_settings):
    """With a cap of 3 and 5 pairs, the counter stays at or below 3. A
    three-party barrier proves all workers run concurrently.
    """
    cap = 3
    search_settings(obs_preview_concurrency=cap)
    entities = [f"country/E{i}" for i in range(cap + 2)]
    barrier_entities = frozenset(entities[:cap])
    barrier = threading.Barrier(cap, timeout=5.0)
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        nonlocal in_flight, max_in_flight
        entity = next(iter(request.role_bindings["observationAbout"]))
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        if entity in barrier_entities:
            barrier.wait(timeout=5.0)
        result = stub_preview_result(sv="Count_Person", entity=entity)
        with lock:
            in_flight -= 1
        return result

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=harness_reader(entities),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert len(observations.series) == len(entities)
    assert max_in_flight <= cap


def test_build_observations_total_preview_concurrency_is_process_wide_bounded(
    monkeypatch, search_settings
):
    """Three calls each dispatch 5 pairs with a per-call cap of 8. The shared
    semaphore, set to 4, limits their combined in-flight fetches to 4.
    """
    total_cap = 4
    search_settings(obs_preview_concurrency=8)
    monkeypatch.setattr(
        region_fetch, "_OBS_PREVIEW_TOTAL_SEMAPHORE", threading.BoundedSemaphore(total_cap)
    )
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0
    never_set = threading.Event()

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        nonlocal in_flight, max_in_flight
        entity = next(iter(request.role_bindings["observationAbout"]))
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        never_set.wait(timeout=0.3)
        with lock:
            in_flight -= 1
        return stub_preview_result(sv="Count_Person", entity=entity)

    monkeypatch.setattr(region_fetch, "materialise_preview", _spy)

    results: list[tuple] = [None, None, None]  # type: ignore[list-item]

    def _worker(call_index: int) -> None:
        entities = [f"country/C{call_index}_{i}" for i in range(5)]
        results[call_index] = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=harness_reader(entities),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    # Each worker runs under its own copy of the test's context, which binds the catalogue.
    threads = [
        threading.Thread(target=contextvars.copy_context().run, args=(_worker, i)) for i in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    for i, thread in enumerate(threads):
        assert not thread.is_alive(), f"worker thread {i} hung past the join timeout"

    for observations, warn in results:
        assert warn is None
        assert observations is not None
        assert len(observations.series) == 5
    assert max_in_flight <= total_cap


def test_build_observations_single_call_fanout_not_serialized_by_total_gate(search_settings):
    """With the process-wide default of 32, one call reaches its per-call cap
    of 3 across 5 pairs. The global gate does not reduce that concurrency.
    """
    cap = 3
    search_settings(obs_preview_concurrency=cap)
    entities = [f"country/E{i}" for i in range(cap + 2)]
    barrier_entities = frozenset(entities[:cap])
    barrier = threading.Barrier(cap, timeout=5.0)
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        nonlocal in_flight, max_in_flight
        entity = next(iter(request.role_bindings["observationAbout"]))
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        if entity in barrier_entities:
            barrier.wait(timeout=5.0)
        result = stub_preview_result(sv="Count_Person", entity=entity)
        with lock:
            in_flight -= 1
        return result

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=harness_reader(entities),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert len(observations.series) == len(entities)
    assert max_in_flight == cap


def test_build_observations_pair_degrades_when_total_semaphore_wait_times_out(
    monkeypatch, search_settings
):
    """A held global permit makes the pair wait past `SEARCH_GRAPH_TIMEOUT_S`.
    The preview call is skipped and the pair counts as a hard failure.
    """
    search_settings(graph_timeout_s=0.05)
    held_semaphore = threading.BoundedSemaphore(1)
    held_semaphore.acquire()  # the process-wide permit is already held elsewhere
    monkeypatch.setattr(region_fetch, "_OBS_PREVIEW_TOTAL_SEMAPHORE", held_semaphore)

    def _unreached(
        request, *, sdmx, graph, limit, ceiling, deadline=None, rate_cache=None, precomputed=None
    ):
        raise AssertionError("materialise_preview must not run: the permit wait times out first")

    monkeypatch.setattr(region_fetch, "materialise_preview", _unreached)

    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/KEN"]},
        graph=FakeGraph(),
        materialise_graph=harness_reader(["country/KEN"]),
        materialise_sdmx=sdmx_double(),
        coverage=empty_coverage_snapshot(),
    )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED


class _HeldPermits:
    """A process-wide permit gate every permit of which is held elsewhere. It records the
    longest each acquire would wait and refuses at once, so no test waits on it."""

    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        return False

    def release(self) -> None:
        raise AssertionError("no permit was acquired")


def test_build_observations_permit_wait_ends_at_the_bound_request_deadline(
    monkeypatch, search_settings
):
    """A permit wait inside a bound request, or a fail-open scope, ends at that deadline,
    however long `SEARCH_GRAPH_TIMEOUT_S` allows. The pair counts as timed out."""
    search_settings(graph_timeout_s=30)
    permits = _HeldPermits()
    monkeypatch.setattr(region_fetch, "_OBS_PREVIEW_TOTAL_SEMAPHORE", permits)

    with request_deadline(time.monotonic() - 1):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/KEN"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/KEN"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert permits.timeouts == [0.0]
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_TIMED_OUT
    assert warn is not None
    assert warn.code == OBSERVATIONS_TIMED_OUT


def test_build_observations_materialisation_usage_error_from_pooled_worker_propagates_uncaught():
    """A worker's `MaterialisationUsageError` propagates through
    `Future.result()` instead of counting as a hard pair failure.
    """
    entities = ["country/AAA", "country/BBB", "country/CCC"]

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        entity = next(iter(request.role_bindings["observationAbout"]))
        if entity == "country/BBB":
            raise MaterialisationUsageError("malformed request")
        return stub_preview_result(sv="Count_Person", entity=entity)

    with (
        patch.object(region_fetch, "materialise_preview", _spy),
        pytest.raises(MaterialisationUsageError),
    ):
        build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=harness_reader(entities),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )


def test_build_observations_output_byte_identical_at_concurrency_one_and_eight(search_settings):
    """Changing concurrency from 1 to 8 leaves serialized output unchanged."""
    entities = [f"country/E{i}" for i in range(5)]

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        entity = next(iter(request.role_bindings["observationAbout"]))
        return stub_preview_result(sv="Count_Person", entity=entity)

    def _fetch() -> dict:
        with patch.object(region_fetch, "materialise_preview", _spy):
            observations, warn = build_observations_planned(
                ["Count_Person"],
                {"observationAbout": entities},
                graph=FakeGraph(),
                materialise_graph=harness_reader(entities),
                materialise_sdmx=sdmx_double(),
                coverage=empty_coverage_snapshot(),
            )
        assert warn is None
        assert observations is not None
        return observations.model_dump()

    search_settings(obs_preview_concurrency=1)
    sequential = _fetch()
    search_settings(obs_preview_concurrency=8)
    pooled = _fetch()

    assert sequential == pooled
