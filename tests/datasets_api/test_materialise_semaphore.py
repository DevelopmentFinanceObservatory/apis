"""Tests that the materialisation semaphore bounds concurrent mint and options-route work.
A barrier-controlled SDMX transport makes overlap observable without timing assumptions.
"""

from __future__ import annotations

import contextvars
import gc
import itertools
import math
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

import graph_release.service as release_service_module
from artifact_store import StoredArtifact, Tier
from code_index.shared import get_index, get_measure_index
from datasets_api import deps as deps_module
from datasets_api import serve as serve_module
from datasets_api.admission import (
    STALE_CHECK_SECONDS,
    MaterialisePermits,
    RebuildFlights,
    bounded_materialise_slot,
)
from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.operation_store import NullOperationStore
from datasets_api.operations.task_client import NullTaskClient
from datasets_api.versions import NullHeadWriter, SubjectHead
from dc_sdmx import SdmxClient
from dfo_platform.app import create_app
from graph_release.context import bound_release, current_release
from graph_release.service import Release, ReleaseService, ReleaseViews
from materialise.graph import Facet
from series_coverage.index import CoverageIndex
from series_coverage.schema import (
    ABSENT_ENTITY,
    FacetMetadata,
    RoleNames,
)
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.graph_release.support import build_test_release, sample_tables

from ._app_factories import offline_recipe_stores
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import offline_version_pair
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR_A = "SemaphoreGrantA"
_STAT_VAR_B = "SemaphoreGrantB"
_REF_A = f"dcid:{_STAT_VAR_A}"
_REF_B = f"dcid:{_STAT_VAR_B}"
_BASE_URL = "https://datacommons.one.org/core/api"

# Required by the composed app lifespan through `DatasetsSettings.from_env()`.
_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
}

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _csv_row(*, stat_var: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},country/KEN,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )


_CSV_BY_STAT_VAR = {
    _STAT_VAR_A: _CSV_HEADER + _csv_row(stat_var=_STAT_VAR_A),
    _STAT_VAR_B: _CSV_HEADER + _csv_row(stat_var=_STAT_VAR_B),
}

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)


class _OverlapTrackingTransport:
    """Records concurrent SDMX requests and blocks on an event to make overlap deterministic."""

    def __init__(self, *, release: threading.Event | None = None) -> None:
        self._release = release
        self._lock = threading.Lock()
        self._current = 0
        self.max_concurrent = 0
        self.requests: list[httpx.Request] = []
        self.first_entered = threading.Event()
        self.reached_two = threading.Event()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
            if self._current >= 2:
                self.reached_two.set()
        self.first_entered.set()
        try:
            if self._release is not None:
                assert self._release.wait(timeout=5), "the test never released the transport"
            variable_param = request.url.params.get("c[variableMeasured]", "")
            body = next(csv for ref, csv in _CSV_BY_STAT_VAR.items() if ref in variable_param)
            return httpx.Response(
                200,
                text=body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        finally:
            with self._lock:
                self._current -= 1


def _default_graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR_A: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            _STAT_VAR_B: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
        },
        facets={
            (_STAT_VAR_A, "country/KEN"): [_FACET],
            (_STAT_VAR_B, "country/KEN"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )


def _coverage_service() -> FakeCoverageService:
    """Coverage for both StatVars, matching `_FACET` and their 2015 rows."""
    return FakeCoverageService(_coverage_index())


def _coverage_index() -> CoverageIndex:
    return build_coverage_index(
        [
            (_STAT_VAR_A, "facet1", ABSENT_ENTITY, "country/KEN", [2015]),
            (_STAT_VAR_B, "facet1", ABSENT_ENTITY, "country/KEN", [2015]),
        ],
        role_names={
            _STAT_VAR_A: RoleNames(donor=None, recipient="recipient"),
            _STAT_VAR_B: RoleNames(donor=None, recipient="recipient"),
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )


@contextmanager
def _client(
    *,
    materialise_max_concurrency: int,
    transport: Callable[[httpx.Request], httpx.Response],
    store: FakeDatasetStore | None = None,
    graph: FakeMintGraphReader | None = None,
    head_writer=None,
    materialise_wait_seconds: float | None = None,
    artifact_store: FakeArtifactStore | None = None,
    coverage: object | None = None,
    timeout_seconds: float | None = None,
    release: ReleaseService | None = None,
):
    """Builds a client under `TestClient` lifespan with one portal for threaded requests.
    Every store is injected, so the lifespan opens no pool. `timeout_seconds` sets both the
    mint and the definition timeout. `release` is the graph release service each request
    binds from."""
    store = store if store is not None else FakeDatasetStore()
    graph = graph if graph is not None else _default_graph()
    sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(transport))
    surface = datasets_surface(
        settings=DatasetsBuildSettings(
            materialise_max_concurrency=materialise_max_concurrency,
            work_max_concurrency=4,
            **(
                {"materialise_wait_seconds": materialise_wait_seconds}
                if materialise_wait_seconds is not None
                else {}
            ),
            **(
                {
                    "mint_timeout_seconds": timeout_seconds,
                    "definition_timeout_seconds": timeout_seconds,
                }
                if timeout_seconds is not None
                else {}
            ),
        ),
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        **offline_version_pair(version_writer=None, snapshot_reader=None),
        reference_writer=FakeArtifactReferenceWriter(),
        head_writer=head_writer if head_writer is not None else NullHeadWriter(),
        operation_store=NullOperationStore(),
        task_client=NullTaskClient(),
        coverage=coverage if coverage is not None else _coverage_service(),
        release=release,
        **offline_recipe_stores(),
    )
    with (
        mock.patch.dict(os.environ, _SETTINGS_ENV),
        TestClient(create_app([surface]), raise_server_exceptions=False) as client,
    ):
        yield client


def _scope(*, statvar: str) -> dict:
    return {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [statvar],
            "scope": {
                "entities": {"recipient": ["country/KEN"]},
                "facets": {
                    "provenance": {"ref": "OECD_DAC2A"},
                    "measurementMethod": {"ref": "GrantEquivalent"},
                    "unit": {"ref": "USD"},
                },
            },
        },
        "options": {"bundle": False},
    }


def test_materialise_semaphore_of_one_serialises_two_concurrent_mints():
    release = threading.Event()
    transport = _OverlapTrackingTransport(release=release)
    results: list[int] = []

    with _client(materialise_max_concurrency=1, transport=transport) as client:

        def _mint(statvar_ref: str) -> None:
            resp = client.post("/api/datasets", json=_scope(statvar=statvar_ref))
            results.append(resp.status_code)

        first = threading.Thread(target=contextvars.copy_context().run, args=(_mint, _REF_A))
        second = threading.Thread(target=contextvars.copy_context().run, args=(_mint, _REF_B))
        first.start()
        assert transport.first_entered.wait(timeout=5)
        second.start()
        # The second mint must stay out of the transport while the first holds the slot.
        assert len(transport.requests) == 1
        assert not transport.reached_two.wait(timeout=0.3)
        assert len(transport.requests) == 1

        release.set()
        for thread in (first, second):
            thread.join(timeout=10)
        assert not first.is_alive() and not second.is_alive()

    assert results == [200, 200]
    # Both mints finish without overlapping inside the transport.
    assert transport.max_concurrent == 1
    assert len(transport.requests) > 1


def test_materialise_semaphore_of_two_lets_two_concurrent_mints_overlap():
    release = threading.Event()
    transport = _OverlapTrackingTransport(release=release)
    results: list[int] = []

    with _client(materialise_max_concurrency=2, transport=transport) as client:

        def _mint(statvar_ref: str) -> None:
            resp = client.post("/api/datasets", json=_scope(statvar=statvar_ref))
            results.append(resp.status_code)

        threads = [
            threading.Thread(target=contextvars.copy_context().run, args=(_mint, _REF_A)),
            threading.Thread(target=contextvars.copy_context().run, args=(_mint, _REF_B)),
        ]
        for thread in threads:
            thread.start()
        # Both slots let the requests reach the blocked transport together.
        assert transport.reached_two.wait(timeout=5)
        assert transport.max_concurrent == 2
        release.set()
        for thread in threads:
            thread.join(timeout=10)

    assert results == [200, 200]
    assert transport.max_concurrent == 2


def test_options_route_re_materialise_counts_against_the_semaphore():
    """Two options requests for distinct definitions share the semaphore.
    With `head_writer` unset, each request rematerialises.
    """
    store = FakeDatasetStore()
    mint_transport = _OverlapTrackingTransport()
    spec_ids: dict[str, str] = {}
    with _client(
        materialise_max_concurrency=4, transport=mint_transport, store=store
    ) as mint_client:
        for ref in (_REF_A, _REF_B):
            resp = mint_client.post("/api/datasets", json=_scope(statvar=ref))
            assert resp.status_code == 200, resp.text
            spec_ids[ref] = resp.json()["definition"]["spec_id"]

    release = threading.Event()
    transport = _OverlapTrackingTransport(release=release)
    results: list[int] = []

    with _client(materialise_max_concurrency=1, transport=transport, store=store) as client:

        def _options(spec_id: str) -> None:
            resp = client.get(f"/api/datasets/{spec_id}/options")
            results.append(resp.status_code)

        first = threading.Thread(
            target=contextvars.copy_context().run, args=(_options, spec_ids[_REF_A])
        )
        second = threading.Thread(
            target=contextvars.copy_context().run, args=(_options, spec_ids[_REF_B])
        )
        first.start()
        assert transport.first_entered.wait(timeout=5)
        second.start()
        # The second options call must stay out of the transport while the first holds the slot.
        assert len(transport.requests) == 1
        assert not transport.reached_two.wait(timeout=0.3)
        assert len(transport.requests) == 1

        release.set()
        for thread in (first, second):
            thread.join(timeout=10)
        assert not first.is_alive() and not second.is_alive()

    assert results == [200, 200]
    assert transport.max_concurrent == 1
    assert len(transport.requests) > 1


class _FakeHeadStore:
    """In-memory head store with compare-and-set writes and direct timestamp aging.
    `advance_head` updates both timestamps. `confirm_head` updates `confirmed_at` only.
    """

    def __init__(self) -> None:
        self._heads: dict[str, SubjectHead] = {}
        self._runs = itertools.count(1)

    def read_head(self, subject_id: str) -> SubjectHead | None:
        return self._heads.get(subject_id)

    def advance_head(
        self,
        *,
        materialisation,
        expected_materialisation_id: int | None,
        fence=None,
    ) -> SubjectHead | None:
        subject_id = materialisation.subject_id
        content_version = materialisation.content_version
        existing = self._heads.get(subject_id)
        stored_run = existing.materialisation_id if existing is not None else None
        if stored_run != expected_materialisation_id:
            return None
        now = datetime.now(UTC)
        written = SubjectHead(
            subject_id=subject_id,
            content_version=content_version,
            confirmed_at=now,
            materialised_at=now,
            materialisation_id=next(self._runs),
            dependencies=materialisation.dependencies,
        )
        self._heads[subject_id] = written
        return written

    def confirm_head(self, *, subject_id: str, content_version: str) -> SubjectHead | None:
        existing = self._heads.get(subject_id)
        if existing is None or existing.content_version != content_version:
            return None
        written = SubjectHead(
            subject_id=existing.subject_id,
            content_version=existing.content_version,
            confirmed_at=datetime.now(UTC),
            materialised_at=existing.materialised_at,
            materialisation_id=existing.materialisation_id,
            dependencies=existing.dependencies,
        )
        self._heads[subject_id] = written
        return written

    def age_head(
        self, subject_id: str, *, age: timedelta, materialised_age: timedelta | None = None
    ) -> None:
        resolved_materialised_age = age if materialised_age is None else materialised_age
        existing = self._heads[subject_id]
        self._heads[subject_id] = SubjectHead(
            subject_id=existing.subject_id,
            content_version=existing.content_version,
            confirmed_at=datetime.now(UTC) - age,
            materialised_at=datetime.now(UTC) - resolved_materialised_age,
            materialisation_id=existing.materialisation_id,
            dependencies=existing.dependencies,
        )


# A 900 s age triggers a facet probe. A 24 h age forces rematerialisation before snapshot loading.
# This fixture has no snapshot reader.
_PAST_INTERVAL = timedelta(hours=25)


def test_stale_head_rebuild_waits_for_a_materialise_slot_and_times_out_as_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GET of a head past its maximum age returns retryable 503 after the 0.1 s wait bound
    while spec A holds the slot. Spec B fetches after release. Role-bearing probes read
    coverage.
    """
    graph = _default_graph()
    store = FakeDatasetStore()
    head_store = _FakeHeadStore()

    setup_transport = _OverlapTrackingTransport()
    with _client(
        materialise_max_concurrency=4,
        transport=setup_transport,
        store=store,
        graph=graph,
        head_writer=head_store,
    ) as setup_client:
        resp_b = setup_client.post("/api/datasets", json=_scope(statvar=_REF_B))
        assert resp_b.status_code == 200, resp_b.text
        spec_b = resp_b.json()["definition"]["spec_id"]
    head_store.age_head(spec_b, age=_PAST_INTERVAL)

    started = threading.Event()
    hold_a = threading.Event()
    # Role-bearing probes read coverage. This count tracks spec B's fetch while A holds the slot.
    fetch_calls_b = {"n": 0}

    def _slow_transport(request: httpx.Request) -> httpx.Response:
        # Only spec A's fetch parks and signals `started`.
        variable_param = request.url.params.get("c[variableMeasured]", "")
        body = next(csv for ref, csv in _CSV_BY_STAT_VAR.items() if ref in variable_param)
        if _STAT_VAR_A in variable_param:
            started.set()
            assert hold_a.wait(timeout=5), "the test never released spec A's fetch"
        elif _STAT_VAR_B in variable_param:
            fetch_calls_b["n"] += 1
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    results: dict[str, int] = {}

    with _client(
        materialise_max_concurrency=1,
        transport=_slow_transport,
        store=store,
        graph=graph,
        head_writer=head_store,
        materialise_wait_seconds=0.1,
    ) as client:

        def _mint_a() -> None:
            resp = client.post("/api/datasets", json=_scope(statvar=_REF_A))
            results["mint_a"] = resp.status_code

        mint_thread = threading.Thread(target=contextvars.copy_context().run, args=(_mint_a,))
        mint_thread.start()
        assert started.wait(timeout=5)
        # A's probe reads coverage. B's fetch must wait for release.
        fetch_calls_while_held = fetch_calls_b["n"]

        blocked = client.get(f"/api/datasets/{spec_b}")

        # Spec A holds the slot, so the 503 marks the wait bound expiring.
        assert not hold_a.is_set()
        assert blocked.status_code == 503, blocked.text
        body = blocked.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True
        assert fetch_calls_b["n"] == fetch_calls_while_held

        hold_a.set()
        mint_thread.join(timeout=10)
        assert not mint_thread.is_alive()
        assert results["mint_a"] == 200

        third = client.get(f"/api/datasets/{spec_b}")
        assert third.status_code == 200, third.text
        assert fetch_calls_b["n"] == fetch_calls_while_held + 1


def test_materialise_wait_expires_returns_retryable_503_and_frees_the_pool_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 0.1 s materialisation wait bound returns retryable 503 while spec A holds the slot.
    After release, a third GET for spec B succeeds, showing the pool thread and permit are free.
    """
    started = threading.Event()
    hold_a = threading.Event()

    def _slow_transport(request: httpx.Request) -> httpx.Response:
        # Route rows by variable. Only spec A's fetch blocks.
        # This lets spec B's later GET use its own row.
        variable_param = request.url.params.get("c[variableMeasured]", "")
        body = next(csv for ref, csv in _CSV_BY_STAT_VAR.items() if ref in variable_param)
        if _STAT_VAR_A in variable_param:
            started.set()
            assert hold_a.wait(timeout=5), "the test never released spec A's fetch"
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    store = FakeDatasetStore()
    setup_transport = _OverlapTrackingTransport()
    with _client(
        materialise_max_concurrency=4, transport=setup_transport, store=store
    ) as setup_client:
        resp_b = setup_client.post("/api/datasets", json=_scope(statvar=_REF_B))
        assert resp_b.status_code == 200, resp_b.text
        spec_b = resp_b.json()["definition"]["spec_id"]

    results: dict[str, int] = {}

    with _client(
        materialise_max_concurrency=1,
        transport=_slow_transport,
        store=store,
        materialise_wait_seconds=0.1,
    ) as client:

        def _mint_a() -> None:
            resp = client.post("/api/datasets", json=_scope(statvar=_REF_A))
            results["mint_a"] = resp.status_code

        mint_thread = threading.Thread(target=contextvars.copy_context().run, args=(_mint_a,))
        mint_thread.start()
        assert started.wait(timeout=5)

        blocked = client.get(f"/api/datasets/{spec_b}")

        assert not hold_a.is_set()
        assert blocked.status_code == 503, blocked.text
        body = blocked.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True

        # The 503 frees the definition thread while spec A holds the slot.
        hold_a.set()
        mint_thread.join(timeout=10)
        assert not mint_thread.is_alive()
        assert results["mint_a"] == 200

        third = client.get(f"/api/datasets/{spec_b}")
        assert third.status_code == 200, third.text


def test_bounded_materialise_slot_caps_the_wait_at_the_nearer_deadline() -> None:
    """`bounded_materialise_slot` waits for the smaller of `wait_seconds`
    and the time remaining to `deadline`. With the wait setting at 10 s,
    an already-held semaphore, and a `deadline` 0.1 s out, the deadline
    bounds this call.
    """
    semaphore = MaterialisePermits(1)
    semaphore.acquire(timeout=0)
    try:
        deadline = time.monotonic() + 0.1
        start = time.monotonic()
        with (
            pytest.raises(MintBackpressure),
            bounded_materialise_slot(semaphore, wait_seconds=10, deadline=deadline),
        ):
            pass
        elapsed = time.monotonic() - start
        assert elapsed < 1.0
    finally:
        semaphore.release()


def test_bounded_materialise_slot_with_no_deadline_waits_the_full_setting() -> None:
    """`deadline=None` (the default) leaves `wait_seconds` as the only
    bound, matching this function's pre-deadline behaviour.
    """
    semaphore = MaterialisePermits(1)
    semaphore.acquire(timeout=0)
    try:
        start = time.monotonic()
        with (
            pytest.raises(MintBackpressure),
            bounded_materialise_slot(semaphore, wait_seconds=0.1, deadline=None),
        ):
            pass
        elapsed = time.monotonic() - start
        assert elapsed < 1.0
    finally:
        semaphore.release()


def test_a_release_swap_takes_every_permit_under_a_stream_of_arrivals() -> None:
    """Six callers loop on two permits, each releasing and asking again at once. A swap that
    starts among them closes the gate, so the permits frames release go to it, and arrivals
    wait at the gate within their own wait bound."""
    permits = MaterialisePermits(2)
    stop = threading.Event()

    def _arrive() -> None:
        while not stop.is_set():
            try:
                with bounded_materialise_slot(permits, wait_seconds=0.5):
                    time.sleep(0.005)
            except MintBackpressure:
                pass

    arrivals = [
        threading.Thread(target=contextvars.copy_context().run, args=(_arrive,)) for _ in range(6)
    ]
    for arrival in arrivals:
        arrival.start()
    swapped = threading.Event()

    def _swap() -> None:
        with permits.every_permit():
            swapped.set()

    swap = threading.Thread(target=_swap, daemon=True)
    try:
        time.sleep(0.05)
        swap.start()
        assert swapped.wait(timeout=3)
    finally:
        stop.set()
        for arrival in arrivals:
            arrival.join(timeout=5)
    swap.join(timeout=5)


def test_two_concurrent_drains_run_one_after_the_other() -> None:
    """Two drains take the same two permits at once. Without a drain mutex each takes one permit and waits forever for the other, which a pause after
    each acquire makes certain."""
    permits = MaterialisePermits(2)
    acquire = permits._semaphore.acquire

    def _acquire_then_pause() -> bool:
        taken = acquire()
        time.sleep(0.02)
        return taken

    done: list[int] = []

    def _drain(position: int) -> None:
        with permits.every_permit():
            done.append(position)

    with mock.patch.object(permits._semaphore, "acquire", _acquire_then_pause):
        drains = [threading.Thread(target=_drain, args=(n,), daemon=True) for n in range(2)]
        for drain in drains:
            drain.start()
        for drain in drains:
            drain.join(timeout=3)
    assert sorted(done) == [0, 1]
    assert permits.acquire(timeout=0)
    assert permits.acquire(timeout=0)


def test_a_release_swap_drains_callers_already_past_the_gate() -> None:
    """Two callers wait for the one permit past the gate when a swap closes it. Each takes the
    permit and releases it, or gives up at its own bound, so the swap takes the permit once
    they are done. A caller arriving during the drain waits at the gate and takes the permit
    only after the swap."""
    permits = MaterialisePermits(1)
    assert permits.acquire(timeout=0)
    outcomes: list[str] = []
    order: list[str] = []

    def _wait_past_the_gate() -> None:
        try:
            with bounded_materialise_slot(permits, wait_seconds=1.0):
                time.sleep(0.02)
        except MintBackpressure:
            outcomes.append("gave_up")
        else:
            outcomes.append("took")

    waiters = [
        threading.Thread(target=contextvars.copy_context().run, args=(_wait_past_the_gate,))
        for _ in range(2)
    ]
    for waiter in waiters:
        waiter.start()
    # The gate is open, so both pass it at once and wait on the held permit.
    time.sleep(0.1)
    swapped = threading.Event()

    def _swap() -> None:
        with permits.every_permit():
            order.append("swap")
            swapped.set()

    def _arrive_late() -> None:
        assert permits.acquire(timeout=3)
        order.append("late")
        permits.release()

    swap = threading.Thread(target=_swap, daemon=True)
    swap.start()
    deadline = time.monotonic() + 2
    while permits._draining == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    late = threading.Thread(target=_arrive_late)
    late.start()

    assert permits.acquire(timeout=0.1) is False
    permits.release()
    assert swapped.wait(timeout=3)
    for thread in (*waiters, late, swap):
        thread.join(timeout=3)
    assert len(outcomes) == 2
    assert order == ["swap", "late"]


def test_mint_facet_probe_waits_for_a_materialise_slot_and_times_out_as_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mint probe returns retryable 503 at the 0.1 s wait bound while spec A holds the slot.
    Spec B succeeds after release. Both role-bearing probes read coverage.
    """
    graph = _default_graph()
    store = FakeDatasetStore()

    started = threading.Event()
    hold_a = threading.Event()
    # Role-bearing probes read coverage. This counter tracks spec B's fetch while the slot is held.
    fetch_calls_b = {"n": 0}

    def _slow_transport(request: httpx.Request) -> httpx.Response:
        # Only spec A's fetch parks and signals `started`.
        variable_param = request.url.params.get("c[variableMeasured]", "")
        body = next(csv for ref, csv in _CSV_BY_STAT_VAR.items() if ref in variable_param)
        if _STAT_VAR_A in variable_param:
            started.set()
            assert hold_a.wait(timeout=5), "the test never released spec A's fetch"
        elif _STAT_VAR_B in variable_param:
            fetch_calls_b["n"] += 1
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    results: dict[str, int] = {}

    with _client(
        materialise_max_concurrency=1,
        transport=_slow_transport,
        store=store,
        graph=graph,
        materialise_wait_seconds=0.1,
    ) as client:

        def _mint_a() -> None:
            resp = client.post("/api/datasets", json=_scope(statvar=_REF_A))
            results["mint_a"] = resp.status_code

        mint_thread = threading.Thread(target=contextvars.copy_context().run, args=(_mint_a,))
        mint_thread.start()
        assert started.wait(timeout=5)
        # A's fetch holds the slot. Its probe reads coverage, so B's fetch must wait.
        fetch_calls_while_held = fetch_calls_b["n"]

        blocked = client.post("/api/datasets", json=_scope(statvar=_REF_B))

        # Spec A holds the slot, so the 503 marks the wait bound expiring.
        assert not hold_a.is_set()
        assert blocked.status_code == 503, blocked.text
        body = blocked.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True
        assert fetch_calls_b["n"] == fetch_calls_while_held

        hold_a.set()
        mint_thread.join(timeout=10)
        assert not mint_thread.is_alive()
        assert results["mint_a"] == 200

        third = client.post("/api/datasets", json=_scope(statvar=_REF_B))
        assert third.status_code == 200, third.text
        assert fetch_calls_b["n"] == fetch_calls_while_held + 1


def test_posted_definition_options_wait_for_the_slot_a_mint_holds_and_refuse_retryably() -> None:
    """B's definition is posted to the options route of a deployment that never registered it.
    While A's mint holds the only slot, the call fetches nothing and returns the retryable 503
    `MintBackpressure` maps to after the 0.1 s wait bound. After release, it answers. The slot
    refusal comes after the call registered B, so B's registry row stays, as a mint's does."""
    with _client(materialise_max_concurrency=1, transport=_OverlapTrackingTransport()) as minting:
        minted_b = minting.post("/api/datasets", json=_scope(statvar=_REF_B))
        assert minted_b.status_code == 200, minted_b.text
    definition_b = minted_b.json()["definition"]
    options_b = f"/api/datasets/{definition_b['spec_id']}/options"

    started = threading.Event()
    hold_a = threading.Event()
    fetch_calls_b = {"n": 0}

    def _slow_transport(request: httpx.Request) -> httpx.Response:
        variable_param = request.url.params.get("c[variableMeasured]", "")
        body = next(csv for ref, csv in _CSV_BY_STAT_VAR.items() if ref in variable_param)
        if _STAT_VAR_A in variable_param:
            started.set()
            assert hold_a.wait(timeout=5), "the test never released spec A's fetch"
        elif _STAT_VAR_B in variable_param:
            fetch_calls_b["n"] += 1
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    store = FakeDatasetStore()
    results: dict[str, int] = {}

    with _client(
        materialise_max_concurrency=1,
        transport=_slow_transport,
        store=store,
        materialise_wait_seconds=0.1,
    ) as client:

        def _mint_a() -> None:
            resp = client.post("/api/datasets", json=_scope(statvar=_REF_A))
            results["mint_a"] = resp.status_code

        mint_thread = threading.Thread(target=contextvars.copy_context().run, args=(_mint_a,))
        mint_thread.start()
        assert started.wait(timeout=5)

        blocked_options = client.post(options_b, json={"definition": definition_b})

        assert not hold_a.is_set()
        assert blocked_options.status_code == 503, blocked_options.text
        assert blocked_options.json()["code"] == "service_unavailable"
        assert blocked_options.json()["retryable"] is True
        assert fetch_calls_b["n"] == 0

        hold_a.set()
        mint_thread.join(timeout=10)
        assert not mint_thread.is_alive()
        assert results["mint_a"] == 200
        registered = set(store.state.spec_registry)
        assert definition_b["spec_id"] in registered

        answered = client.post(options_b, json={"definition": definition_b})
        assert answered.status_code == 200, answered.text
        assert fetch_calls_b["n"] == 1
        assert set(store.state.spec_registry) == registered


class _PermitLog:
    """One ordered log of materialise permit acquires and releases, SDMX requests, artifact
    writes and coverage loads, each with the number of permits held when it happened."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.held = 0
        self.events: list[tuple[str, int]] = []

    def record(self, kind: str, *, delta: int = 0) -> None:
        with self._lock:
            self.held += delta
            self.events.append((kind, self.held))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]

    def held_at(self, kind: str) -> set[int]:
        return {held for event, held in self.events if event == kind}


class _CountedPermits(MaterialisePermits):
    """`MaterialisePermits` that logs each permit it grants and takes back."""

    def __init__(self, permits: int, *, log: _PermitLog) -> None:
        super().__init__(permits)
        self._log = log

    def acquire(self, *, timeout: float, keep_waiting: Callable[[], bool] | None = None) -> bool:
        acquired = super().acquire(timeout=timeout, keep_waiting=keep_waiting)
        if acquired:
            self._log.record("acquire", delta=1)
        return acquired

    def release(self) -> None:
        self._log.record("release", delta=-1)
        super().release()

    @contextmanager
    def every_permit(self) -> Iterator[None]:
        with super().every_permit():
            self._log.record("acquire", delta=self.permits)
            try:
                yield
            finally:
                self._log.record("release", delta=-self.permits)


class _LoggedArtifactStore(FakeArtifactStore):
    def __init__(self, log: _PermitLog) -> None:
        super().__init__()
        self._log = log

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        self._log.record("write")
        return super().put(data, tier=tier)


class _TrackedBytes(bytearray):
    """Frame bytes a test can hold a weak reference to, which `bytes` does not allow."""


class _FrameTrackingPermits(MaterialisePermits):
    """Permits that say, at each release after a tracked frame read, whether anything still
    references the last frame read."""

    def __init__(self, permits: int) -> None:
        super().__init__(permits)
        self.read: list[weakref.ref[object]] = []
        self.freed_at_release: list[bool] = []

    def track(self, data: bytes) -> _TrackedBytes:
        """A copy of `data` this double watches, for an artifact store to return."""
        tracked = _TrackedBytes(data)
        self.watch(tracked)
        return tracked

    def watch(self, frame: object) -> None:
        """Watch `frame`, an object that holds a frame a request built."""
        self.read.append(weakref.ref(frame))

    def release(self) -> None:
        if self.read:
            gc.collect()
            self.freed_at_release.append(self.read[-1]() is None)
        super().release()


def _count_permits(monkeypatch: pytest.MonkeyPatch, log: _PermitLog) -> None:
    """Build the surface's materialise permits as `_CountedPermits` over `log`."""
    monkeypatch.setattr(
        deps_module, "MaterialisePermits", lambda permits: _CountedPermits(permits, log=log)
    )


def _logged_transport(log: _PermitLog, *, release: threading.Event | None = None):
    entered = threading.Event()

    def transport(request: httpx.Request) -> httpx.Response:
        log.record("sdmx")
        entered.set()
        if release is not None:
            assert release.wait(timeout=5), "the test never released the transport"
        variable_param = request.url.params.get("c[variableMeasured]", "")
        body = next(csv for ref, csv in _CSV_BY_STAT_VAR.items() if ref in variable_param)
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return transport, entered


def test_one_permit_is_held_from_the_first_sdmx_request_to_the_last_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mint holds one materialise permit across its whole frame life: taken before its
    first SDMX request and released only after its frame1, statistics and requested rendering
    are written."""
    log = _PermitLog()
    _count_permits(monkeypatch, log)
    transport, _ = _logged_transport(log)
    with _client(
        materialise_max_concurrency=2,
        transport=transport,
        artifact_store=_LoggedArtifactStore(log),
    ) as client:
        resp = client.post("/api/datasets", json=_scope(statvar=_REF_A))

    assert resp.status_code == 200, resp.text
    kinds = log.kinds()
    assert kinds.count("acquire") == 1
    assert kinds.count("release") == 1
    assert kinds.count("write") >= 3
    last_write = len(kinds) - 1 - kinds[::-1].index("write")
    assert kinds.index("acquire") < kinds.index("sdmx")
    assert last_write < kinds.index("release")
    assert log.held_at("sdmx") == log.held_at("write") == {1}


@pytest.mark.parametrize("route", ["mint", "get"])
def test_a_materialisation_past_its_deadline_refuses_timeout_at_the_timeout(route: str) -> None:
    """A mint or GET whose SDMX read outlasts the 1 s timeout answers the retryable `timeout`
    refusal at the timeout, with no wait added past it. The transport stays blocked for up to
    4 s, so an answer later than 3 s means the door waited past its timeout."""
    store = FakeDatasetStore()
    spec_id = ""
    if route == "get":
        with _client(
            materialise_max_concurrency=1, transport=_OverlapTrackingTransport(), store=store
        ) as minting:
            minted = minting.post("/api/datasets", json=_scope(statvar=_REF_A))
            assert minted.status_code == 200, minted.text
            spec_id = minted.json()["definition"]["spec_id"]
    release = threading.Event()

    def stalled(request: httpx.Request) -> httpx.Response:
        release.wait(timeout=4)
        variable_param = request.url.params.get("c[variableMeasured]", "")
        body = next(csv for ref, csv in _CSV_BY_STAT_VAR.items() if ref in variable_param)
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    with _client(
        materialise_max_concurrency=1,
        transport=stalled,
        store=store,
        timeout_seconds=1.0,
    ) as client:
        started = time.monotonic()
        if route == "mint":
            resp = client.post("/api/datasets", json=_scope(statvar=_REF_A))
        else:
            resp = client.get(f"/api/datasets/{spec_id}")
        elapsed = time.monotonic() - started
        release.set()

    assert resp.status_code == 504, resp.text
    assert resp.json()["code"] == "timeout"
    assert resp.json()["retryable"] is True
    assert elapsed < 3.0


class _SwapBeforePermit(_CountedPermits):
    """Permits whose next `acquire` first runs `before_acquire` once in the caller's thread,
    where a test swaps the graph release between a request's entry and its permit. It passes
    `keep_waiting` on, so the request's first look before the wait sees the swap."""

    before_acquire: Callable[[], None] | None = None

    def acquire(self, *, timeout: float, keep_waiting: Callable[[], bool] | None = None) -> bool:
        before, self.before_acquire = self.before_acquire, None
        if before is not None:
            before()
        return super().acquire(timeout=timeout, keep_waiting=keep_waiting)

    @contextmanager
    def every_permit(self) -> Iterator[None]:
        """Log the drain as `swap`, so `acquire` and `release` count requests alone."""
        with MaterialisePermits.every_permit(self):
            self._log.record("swap")
            yield


def _swapping_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> tuple[ReleaseService, Callable[[], None]]:
    """A graph release service serving the first of two published releases, and the function
    that swaps it to the second. Each release's views are the packaged code index and
    `_coverage_index()`, and each one's origin view is the session release's, so a request
    reads the same index and origin from either."""
    views = ReleaseViews(
        code_index=get_index(), measure_index=get_measure_index(), coverage=_coverage_index()
    )
    origin = current_release().origin
    monkeypatch.setattr(release_service_module, "build_release_views", lambda _loaded: views)
    monkeypatch.setattr(release_service_module, "release_origin", lambda _loaded: origin)
    root = (tmp_path / "releases").as_uri()
    build_test_release(root, tables=sample_tables())
    clock = [0.0]
    service = ReleaseService(
        uri=root,
        local_dir=tmp_path / "local",
        recheck_seconds=60,
        clock=lambda: clock[0],
    )
    second = build_test_release(root, tables=sample_tables(), dc_serving_revision="second")

    def swap() -> None:
        clock[0] += 60
        service.snapshot()
        service.close()
        assert service.snapshot().release_id == second.release_id

    return service, swap


def test_a_mint_bound_to_a_swapped_out_release_refuses_backpressure_without_a_permit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The mint bound the first release at entry, and the service swapped to the second
    before the mint's permit. The mint resolved its members from the first release's code
    index, so it cannot start again on the second: it answers the retryable backpressure
    refusal having taken no permit and fetched nothing. The next mint binds the second
    release and is served."""
    log = _PermitLog()
    permit = _SwapBeforePermit(1, log=log)
    monkeypatch.setattr(deps_module, "MaterialisePermits", lambda _permits: permit)
    release, swap = _swapping_release(monkeypatch, tmp_path)
    transport, _ = _logged_transport(log)
    with _client(materialise_max_concurrency=1, transport=transport, release=release) as client:
        permit.before_acquire = swap
        refused = client.post("/api/datasets", json=_scope(statvar=_REF_A))
        refused_kinds = log.kinds()
        served = client.post("/api/datasets", json=_scope(statvar=_REF_A))

    assert refused.status_code == 503, refused.text
    assert refused.json()["code"] == "service_unavailable"
    assert refused.json()["retryable"] is True
    assert refused_kinds == ["swap"]
    assert served.status_code == 200, served.text
    assert "acquire" in log.kinds()


def _within(condition: Callable[[], bool], *, seconds: float = 3.0) -> bool:
    """Whether `condition` came true within `seconds`, polling every hundredth of a second."""
    ends = time.monotonic() + seconds
    while not condition():
        if time.monotonic() >= ends:
            return False
        time.sleep(0.01)
    return True


class _ServedRelease:
    """Stands in for the request's bound `graph_release.service.Release`, whose `served` is all
    admission reads. Records when each look was made. With `swapped_after`, every look past
    that many finds the release swapped out."""

    def __init__(self, *, swapped_after: int | None = None) -> None:
        self.serving = True
        self.looks: list[float] = []
        self._swapped_after = swapped_after

    @property
    def served(self) -> bool:
        self.looks.append(time.monotonic())
        if self._swapped_after is not None and len(self.looks) > self._swapped_after:
            return False
        return self.serving


def _bound_to(release: _ServedRelease, work: Callable[[], None]) -> threading.Thread:
    """A started thread running `work` with `release` bound, as a request runs."""

    def run() -> None:
        with bound_release(cast(Release, release)):
            work()

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def test_a_request_waiting_for_a_permit_refuses_within_a_wait_slice_of_a_swap() -> None:
    """A request waits behind two held permits when the service swaps its bound release out.
    It refuses with the retryable backpressure refusal within a few wait slices, while both
    permits are still held, and takes none."""
    permits = MaterialisePermits(2)
    assert permits.acquire(timeout=0) and permits.acquire(timeout=0)
    release = _ServedRelease()
    outcomes: list[str] = []

    def work() -> None:
        try:
            with bounded_materialise_slot(permits, wait_seconds=10):
                outcomes.append("built")
        except MintBackpressure:
            outcomes.append("refused")

    waiter = _bound_to(release, work)
    try:
        # The second look is the one made as the wait for a permit starts.
        assert _within(lambda: len(release.looks) >= 2)
        release.serving = False

        assert _within(lambda: outcomes == ["refused"], seconds=30 * STALE_CHECK_SECONDS)
    finally:
        permits.release()
        permits.release()
        waiter.join(timeout=5)
    assert permits.acquire(timeout=0) and permits.acquire(timeout=0)


def test_a_request_waiting_its_rebuild_turn_refuses_within_a_wait_slice_of_a_swap() -> None:
    """A request waits its turn behind a rebuild of the same subject when the service swaps its
    bound release out. It refuses with the retryable backpressure refusal within a few wait
    slices, while the holder still builds."""
    flights = RebuildFlights()
    release = _ServedRelease()
    holding, may_finish = threading.Event(), threading.Event()
    outcomes: list[str] = []

    def holder() -> None:
        with flights.turn("spec_x", deadline=None, wait_seconds=10):
            holding.set()
            may_finish.wait(timeout=5)

    def waiter() -> None:
        try:
            with flights.turn("spec_x", deadline=None, wait_seconds=10):
                outcomes.append("entered")
        except MintBackpressure:
            outcomes.append("refused")

    threads = [_bound_to(_ServedRelease(), holder)]
    try:
        assert holding.wait(timeout=5)
        threads.append(_bound_to(release, waiter))
        assert _within(lambda: len(release.looks) >= 1)
        release.serving = False

        assert _within(lambda: outcomes == ["refused"], seconds=30 * STALE_CHECK_SECONDS)
    finally:
        may_finish.set()
        for thread in threads:
            thread.join(timeout=5)


def test_a_swap_that_lands_as_a_request_takes_its_permit_gives_the_permit_back() -> None:
    """The release is served at both looks that lead to the permit and swapped out by the look
    made once the permit is held. The request refuses with the retryable backpressure refusal,
    runs nothing under the permit, and gives it back."""
    permits = MaterialisePermits(1)
    release = _ServedRelease(swapped_after=2)
    ran: list[bool] = []

    with (
        bound_release(cast(Release, release)),
        pytest.raises(MintBackpressure),
        bounded_materialise_slot(permits, wait_seconds=1),
    ):
        ran.append(True)

    assert ran == []
    assert len(release.looks) == 3
    assert permits.acquire(timeout=0)


def test_a_request_at_the_gate_during_a_swap_refuses_without_taking_a_permit() -> None:
    """A request reaches the permits while a swap holds every one, and the swap replaces its
    bound release. When the swap ends the request takes no permit: it refuses with the
    retryable backpressure refusal."""
    permits = MaterialisePermits(1)
    release = _ServedRelease()
    swapping, may_end = threading.Event(), threading.Event()
    outcomes: list[str] = []

    def swap() -> None:
        with permits.every_permit():
            swapping.set()
            may_end.wait(timeout=5)

    def work() -> None:
        try:
            with bounded_materialise_slot(permits, wait_seconds=10):
                outcomes.append("built")
        except MintBackpressure:
            outcomes.append("refused")

    swapper = threading.Thread(target=swap)
    swapper.start()
    try:
        assert swapping.wait(timeout=5)
        waiter = _bound_to(release, work)
        # The look made before the gate, which the swap holds closed.
        assert _within(lambda: len(release.looks) == 1)
        release.serving = False
    finally:
        may_end.set()
        swapper.join(timeout=5)
    waiter.join(timeout=5)

    assert outcomes == ["refused"]
    assert permits.acquire(timeout=0)


def test_a_sliced_wait_ends_at_its_bound_and_looks_once_a_slice() -> None:
    """A wait lasts its bound, and asks whether the bound release is served once before it
    starts and once a slice after that."""
    permits = MaterialisePermits(1)
    assert permits.acquire(timeout=0)
    release = _ServedRelease()

    wait_seconds = 2.5 * STALE_CHECK_SECONDS
    started = time.monotonic()
    try:
        with (
            bound_release(cast(Release, release)),
            pytest.raises(MintBackpressure),
            bounded_materialise_slot(permits, wait_seconds=wait_seconds),
        ):
            pass
        elapsed = time.monotonic() - started
    finally:
        permits.release()
    assert wait_seconds <= elapsed < 5.0
    # One look before the wait and one per slice begun after it. A stalled thread begins fewer
    # slices, never more.
    assert 2 <= len(release.looks) <= math.ceil(wait_seconds / STALE_CHECK_SECONDS) + 2


@pytest.mark.parametrize("route", ["mint", "options"])
def test_a_materialisation_frees_its_frame_before_its_permit_is_released(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mint, and a definition posted to the options route of a deployment that never
    registered it, each materialise a frame under one permit. When the request gives the
    permit back, nothing references the materialisation that holds the frame."""
    with _client(materialise_max_concurrency=1, transport=_OverlapTrackingTransport()) as minting:
        minted = minting.post("/api/datasets", json=_scope(statvar=_REF_B))
        assert minted.status_code == 200, minted.text
    definition = minted.json()["definition"]
    permits = _FrameTrackingPermits(1)
    monkeypatch.setattr(deps_module, "MaterialisePermits", lambda _permits: permits)
    real_prewrite = serve_module.prewrite_definition

    def watched_prewrite(**kwargs):
        prewrite = real_prewrite(**kwargs)
        permits.watch(prewrite.result)
        return prewrite

    monkeypatch.setattr(serve_module, "prewrite_definition", watched_prewrite)
    with _client(materialise_max_concurrency=1, transport=_OverlapTrackingTransport()) as client:
        if route == "mint":
            resp = client.post("/api/datasets", json=_scope(statvar=_REF_B))
        else:
            resp = client.post(
                f"/api/datasets/{definition['spec_id']}/options", json={"definition": definition}
            )

    assert resp.status_code == 200, resp.text
    assert len(permits.read) == 1
    assert permits.freed_at_release == [True]
