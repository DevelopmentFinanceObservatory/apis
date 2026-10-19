"""Tests that the materialisation semaphore bounds concurrent mint and options-route work.
A barrier-controlled SDMX transport makes overlap observable without timing assumptions.
"""

from __future__ import annotations

import functools
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
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

from artifact_store import StoredArtifact, Tier
from code_index.shared import get_index
from datasets_api import deps as deps_module
from datasets_api import serve as serve_module
from datasets_api.admission import (
    COVERAGE_RESTARTS,
    STALE_CHECK_SECONDS,
    MaterialisePermits,
    RebuildFlights,
    bounded_materialise_slot,
    under_current_coverage,
)
from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.operation_store import NullOperationStore
from datasets_api.operations.task_client import NullTaskClient
from datasets_api.versions import NullHeadWriter, NullVersionRecordWriter, SubjectHead
from dc_sdmx import SdmxClient
from dfo_platform.app import create_app
from materialise.graph import Facet
from series_coverage import shared as coverage_shared
from series_coverage.index import CoverageIndex
from series_coverage.loader import LoadedCoverage
from series_coverage.schema import (
    ABSENT_ENTITY,
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
)
from series_coverage.shared import CoverageService, CoverageSnapshot
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import offline_recipe_stores
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
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
):
    """Builds a client under `TestClient` lifespan with one portal for threaded requests.
    Every store is injected, so the lifespan opens no pool. `timeout_seconds` sets both the
    mint and the definition timeout."""
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
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        version_writer=NullVersionRecordWriter(),
        head_writer=head_writer if head_writer is not None else NullHeadWriter(),
        operation_store=NullOperationStore(),
        task_client=NullTaskClient(),
        coverage=coverage if coverage is not None else _coverage_service(),
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

        first = threading.Thread(target=_mint, args=(_REF_A,))
        second = threading.Thread(target=_mint, args=(_REF_B,))
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
            threading.Thread(target=_mint, args=(_REF_A,)),
            threading.Thread(target=_mint, args=(_REF_B,)),
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

        first = threading.Thread(target=_options, args=(spec_ids[_REF_A],))
        second = threading.Thread(target=_options, args=(spec_ids[_REF_B],))
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

        mint_thread = threading.Thread(target=_mint_a)
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

        mint_thread = threading.Thread(target=_mint_a)
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


def test_a_coverage_reload_takes_every_permit_under_a_stream_of_arrivals() -> None:
    """Six callers loop on two permits, each releasing and asking again at once. A reload that
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

    arrivals = [threading.Thread(target=_arrive) for _ in range(6)]
    for arrival in arrivals:
        arrival.start()
    reloaded = threading.Event()

    def _reload() -> None:
        with permits.every_permit():
            reloaded.set()

    reload = threading.Thread(target=_reload, daemon=True)
    try:
        time.sleep(0.05)
        reload.start()
        assert reloaded.wait(timeout=3)
    finally:
        stop.set()
        for arrival in arrivals:
            arrival.join(timeout=5)
    reload.join(timeout=5)


def test_two_concurrent_drains_run_one_after_the_other() -> None:
    """A coverage reload and a release swap drain the same two permits at once. Without a
    drain mutex each takes one permit and waits forever for the other, which a pause after
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


def test_a_coverage_reload_drains_callers_already_past_the_gate() -> None:
    """Two callers wait for the one permit past the gate when a reload closes it. Each takes the
    permit and releases it, or gives up at its own bound, so the reload takes the permit once
    they are done. A caller arriving during the drain waits at the gate and takes the permit
    only after the reload."""
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

    waiters = [threading.Thread(target=_wait_past_the_gate) for _ in range(2)]
    for waiter in waiters:
        waiter.start()
    # The gate is open, so both pass it at once and wait on the held permit.
    time.sleep(0.1)
    reloaded = threading.Event()

    def _reload() -> None:
        with permits.every_permit():
            order.append("reload")
            reloaded.set()

    def _arrive_late() -> None:
        assert permits.acquire(timeout=3)
        order.append("late")
        permits.release()

    reload = threading.Thread(target=_reload, daemon=True)
    reload.start()
    deadline = time.monotonic() + 2
    while permits._draining == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    late = threading.Thread(target=_arrive_late)
    late.start()

    assert permits.acquire(timeout=0.1) is False
    permits.release()
    assert reloaded.wait(timeout=3)
    for thread in (*waiters, late, reload):
        thread.join(timeout=3)
    assert len(outcomes) == 2
    assert order == ["reload", "late"]


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

        mint_thread = threading.Thread(target=_mint_a)
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

        mint_thread = threading.Thread(target=_mint_a)
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


def _manifest(build_id: str) -> CoverageManifest:
    return CoverageManifest(
        stamp=f"stamp-{build_id}",
        build_id=build_id,
        built_at="1970-01-01T00:00:00Z",
        source="fake",
        row_counts={},
        files={},
        horizon=_coverage_index().horizon,
    )


def test_a_coverage_reload_waits_for_a_running_mint_then_holds_every_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload of a moved coverage build starts its load only once the mint holding a permit
    has written its last artifact, and loads holding both of the surface's permits."""
    log = _PermitLog()
    _count_permits(monkeypatch, log)
    published = {"build_id": "build-1"}
    monkeypatch.setattr(
        coverage_shared,
        "read_published_pointer",
        lambda _uri: CoveragePointer(
            stamp=f"stamp-{published['build_id']}", build_id=published["build_id"]
        ),
    )
    reloaded = threading.Event()

    def loader(base_uri: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        log.record("load")
        if published["build_id"] != "build-1":
            reloaded.set()
        return LoadedCoverage(manifest=_manifest(published["build_id"]), index=_coverage_index())

    coverage = CoverageService(
        artifact_uri="gs://coverage-fixture/coverage",
        index=get_index(),
        index_check=False,
        recheck_seconds=0,
        loader=loader,
    )
    log.events.clear()
    release = threading.Event()
    transport, entered = _logged_transport(log, release=release)
    results: list[int] = []
    with _client(
        materialise_max_concurrency=2,
        transport=transport,
        artifact_store=_LoggedArtifactStore(log),
        coverage=coverage,
    ) as client:
        mint = threading.Thread(
            target=lambda: results.append(
                client.post("/api/datasets", json=_scope(statvar=_REF_A)).status_code
            )
        )
        mint.start()
        assert entered.wait(timeout=5)
        published["build_id"] = "build-2"
        reload = threading.Thread(target=coverage.snapshot_current)
        reload.start()
        assert not reloaded.wait(timeout=0.3)
        release.set()
        for thread in (mint, reload):
            thread.join(timeout=10)
        assert not mint.is_alive() and not reload.is_alive()

    assert results == [200]
    assert coverage.stamp == "stamp-build-2"
    kinds = log.kinds()
    last_write = len(kinds) - 1 - kinds[::-1].index("write")
    assert last_write < kinds.index("load")
    assert log.held_at("load") == {2}


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


class _CoverageBuilds:
    """A `CoverageService` over an in-memory pointer and loader. Build `n` serves a new index
    from `make_index` under the stamp `stamp-build-<n>`. `indexes` holds a weak reference to
    each build's index, so a test can tell when nothing keeps a replaced one resident. Each
    snapshot taken is recorded in `log` as `snapshot`, with the permits held at that point."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        make_index: Callable[[], CoverageIndex] = _coverage_index,
        log: _PermitLog | None = None,
    ) -> None:
        self._log = log
        self.published = 1
        self.indexes: dict[int, weakref.ref[CoverageIndex]] = {}
        self.snapshots_taken = 0
        monkeypatch.setattr(
            coverage_shared,
            "read_published_pointer",
            lambda _uri: CoveragePointer(
                stamp=f"stamp-build-{self.published}", build_id=f"build-{self.published}"
            ),
        )

        def loader(base_uri: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
            index = make_index()
            self.indexes[self.published] = weakref.ref(index)
            return LoadedCoverage(
                manifest=CoverageManifest(
                    stamp=f"stamp-build-{self.published}",
                    build_id=f"build-{self.published}",
                    built_at="1970-01-01T00:00:00Z",
                    source="fake",
                    row_counts={},
                    files={},
                    horizon=index.horizon,
                ),
                index=index,
            )

        self.service = CoverageService(
            artifact_uri="gs://coverage-fixture/coverage",
            index=get_index(),
            index_check=False,
            recheck_seconds=0,
            loader=loader,
        )
        for name in ("snapshot", "snapshot_current"):
            monkeypatch.setattr(self.service, name, self._counting(getattr(self.service, name)))

    def _counting(self, take: Callable[[], CoverageSnapshot]) -> Callable[[], CoverageSnapshot]:
        def counted() -> CoverageSnapshot:
            self.snapshots_taken += 1
            if self._log is not None:
                self._log.record("snapshot")
            return take()

        return counted

    def reload(self) -> None:
        """Publish the next build and load it in the calling thread, which holds no permit."""
        self.published += 1
        self.service.snapshot_current()

    def resident(self, build: int) -> bool:
        """Whether anything still references `build`'s index, after a collection."""
        gc.collect()
        return self.indexes[build]() is not None

    def serving(self, index: CoverageIndex) -> int:
        """The number of the build whose index `index` is."""
        return next(build for build, ref in self.indexes.items() if ref() is index)


class _ReloadBeforePermit(_CountedPermits):
    """Permits whose `acquire` first runs `before_acquire` in the caller's thread, where a test
    reloads coverage between a request's snapshot and its permit. The reload lands after the
    request's last look at its snapshot while it waits, so `keep_waiting` is not asked and the
    check made once the permit is held is the one that finds it."""

    before_acquire: Callable[[], None] | None = None

    def acquire(self, *, timeout: float, keep_waiting: Callable[[], bool] | None = None) -> bool:
        if self.before_acquire is not None:
            self.before_acquire()
        return super().acquire(timeout=timeout)

    @contextmanager
    def every_permit(self) -> Iterator[None]:
        """Log the drain as `reload`, so `acquire` and `release` count requests alone."""
        with MaterialisePermits.every_permit(self):
            self._log.record("reload")
            yield


def _reloading_permit(
    log: _PermitLog, coverage: _CoverageBuilds
) -> tuple[_ReloadBeforePermit, Callable[..., None]]:
    """One logged materialise permit and its `schedule`. After `schedule(n)`, each of the next
    `n` permit requests made once a coverage snapshot was taken meets a completed reload
    first. `schedule(n, after=m)` lets the first `m` of those requests through untouched."""
    remaining = [0]
    untouched = [0]

    def schedule(reloads: int, *, after: int = 0) -> None:
        coverage.snapshots_taken = 0
        remaining[0] = reloads
        untouched[0] = after

    def before_acquire() -> None:
        if not coverage.snapshots_taken:
            return
        if untouched[0]:
            untouched[0] -= 1
        elif remaining[0]:
            remaining[0] -= 1
            coverage.reload()

    permit = _ReloadBeforePermit(1, log=log)
    permit.before_acquire = before_acquire
    return permit, schedule


def _reload_before_permit(
    monkeypatch: pytest.MonkeyPatch, log: _PermitLog, coverage: _CoverageBuilds
) -> Callable[..., None]:
    """Build the surface's materialise permits as `_reloading_permit` and return its
    `schedule`."""
    permit, schedule = _reloading_permit(log, coverage)
    monkeypatch.setattr(deps_module, "MaterialisePermits", lambda _permits: permit)
    return schedule


def test_a_mint_whose_snapshot_a_reload_replaced_materialises_on_the_served_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mint took build 1's snapshot and a reload to build 2 completed before its permit. The
    mint gives the permit back, starts again, and fetches against build 2's index and stamp,
    after build 1's index has left memory, as its first SDMX request observes."""
    log = _PermitLog()
    coverage = _CoverageBuilds(monkeypatch)
    _reload_before_permit(monkeypatch, log, coverage)(1)
    transport, _ = _logged_transport(log)
    old_index_resident: list[bool] = []
    fetched_on_served_index: list[bool] = []

    def observing_transport(request: httpx.Request) -> httpx.Response:
        old_index_resident.append(coverage.resident(1))
        return transport(request)

    real_prewrite = serve_module.prewrite_definition

    def observing_prewrite(**kwargs):
        fetched_on_served_index.append(kwargs["coverage_index"] is coverage.indexes[2]())
        return real_prewrite(**kwargs)

    monkeypatch.setattr(serve_module, "prewrite_definition", observing_prewrite)
    with _client(
        materialise_max_concurrency=1, transport=observing_transport, coverage=coverage.service
    ) as client:
        resp = client.post("/api/datasets", json=_scope(statvar=_REF_A))

    assert resp.status_code == 200, resp.text
    assert resp.json()["series_coverage_stamp"] == "stamp-build-2"
    assert fetched_on_served_index == [True]
    assert old_index_resident and not any(old_index_resident)
    # The stale pass takes and returns its permit before anything is fetched.
    kinds = [kind for kind in log.kinds() if kind in ("acquire", "release", "sdmx")]
    assert kinds[:4] == ["acquire", "release", "acquire", "sdmx"]
    assert kinds.count("acquire") == kinds.count("release") == 2
    assert "coverage_restarts;desc=1" in resp.headers["server-timing"]


def test_a_mint_whose_snapshot_is_replaced_on_every_start_refuses_backpressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload replaces the snapshot before the permit on the first pass and on each of the
    `COVERAGE_RESTARTS` restarts. The mint then answers the retryable backpressure refusal,
    having fetched nothing and holding no permit."""
    log = _PermitLog()
    coverage = _CoverageBuilds(monkeypatch)
    _reload_before_permit(monkeypatch, log, coverage)(1 + COVERAGE_RESTARTS)
    transport, _ = _logged_transport(log)
    with _client(
        materialise_max_concurrency=1, transport=transport, coverage=coverage.service
    ) as client:
        resp = client.post("/api/datasets", json=_scope(statvar=_REF_A))

    assert resp.status_code == 503, resp.text
    assert resp.json()["code"] == "service_unavailable"
    assert resp.json()["retryable"] is True
    kinds = log.kinds()
    assert "sdmx" not in kinds
    assert kinds.count("acquire") == kinds.count("release") == 1 + COVERAGE_RESTARTS


def _within(condition: Callable[[], bool], *, seconds: float = 3.0) -> bool:
    """Whether `condition` came true within `seconds`, polling every hundredth of a second."""
    ends = time.monotonic() + seconds
    while not condition():
        if time.monotonic() >= ends:
            return False
        time.sleep(0.01)
    return True


def _slot_on(permits: MaterialisePermits, coverage: _CoverageBuilds, snapshot: CoverageSnapshot):
    """A materialise slot for a request that took `snapshot` before it."""
    return bounded_materialise_slot(
        permits,
        wait_seconds=10,
        coverage_current=functools.partial(coverage.service.serves, snapshot),
    )


def test_a_request_reaching_the_permits_drops_a_replaced_index_while_others_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request took build 1's snapshot and is still computing when a reload serves build 2
    and two requests on build 2 take both permits. Build 1's index stays resident for that
    computation. Once the request reaches the permits it drops the index within a few wait
    slices, while both permits are still held, and waits again on build 2."""
    coverage = _CoverageBuilds(monkeypatch)
    permits = MaterialisePermits(2)
    coverage.service.register_load_guard(permits.every_permit)
    computing, may_wait, may_finish = threading.Event(), threading.Event(), threading.Event()
    holding = threading.Semaphore(0)
    waiter_runs: list[str] = []
    built_on: list[tuple[str, str]] = []

    def waiter(snapshot: CoverageSnapshot) -> None:
        waiter_runs.append(snapshot.stamp)
        computing.set()
        may_wait.wait(timeout=5)
        with _slot_on(permits, coverage, snapshot):
            built_on.append(("waiter", snapshot.stamp))

    def admitted(snapshot: CoverageSnapshot) -> None:
        with _slot_on(permits, coverage, snapshot):
            holding.release()
            may_finish.wait(timeout=5)
            built_on.append(("admitted", snapshot.stamp))

    def request(work: Callable[[CoverageSnapshot], None]) -> threading.Thread:
        thread = threading.Thread(
            target=under_current_coverage, args=(coverage.service.snapshot, work)
        )
        thread.start()
        return thread

    threads = [request(waiter)]
    try:
        assert computing.wait(timeout=5)
        coverage.reload()
        threads += [request(admitted), request(admitted)]
        assert holding.acquire(timeout=5) and holding.acquire(timeout=5)
        assert coverage.resident(1)

        may_wait.set()

        assert _within(lambda: not coverage.resident(1))
        assert _within(lambda: waiter_runs == ["stamp-build-1", "stamp-build-2"])
        assert built_on == []
    finally:
        may_wait.set()
        may_finish.set()
        for thread in threads:
            thread.join(timeout=5)
    assert sorted(built_on) == [
        ("admitted", "stamp-build-2"),
        ("admitted", "stamp-build-2"),
        ("waiter", "stamp-build-2"),
    ]


def test_a_request_waiting_for_a_permit_drops_a_replaced_index_within_a_wait_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request that took build 1's snapshot waits behind two held permits when build 2 is
    served. The service here loads outside the permits, which puts the reload inside the
    request's wait. The request drops build 1's index within a few wait slices, while both
    permits are still held, and is admitted on build 2 once one is free."""
    coverage = _CoverageBuilds(monkeypatch)
    permits = MaterialisePermits(2)
    assert permits.acquire(timeout=0) and permits.acquire(timeout=0)
    held = 2
    looks: list[bool] = []
    runs: list[str] = []
    built_on: list[str] = []

    def work(snapshot: CoverageSnapshot) -> None:
        runs.append(snapshot.stamp)

        def current() -> bool:
            looks.append(coverage.service.serves(snapshot))
            return looks[-1]

        with bounded_materialise_slot(permits, wait_seconds=10, coverage_current=current):
            built_on.append(snapshot.stamp)

    waiter = threading.Thread(target=under_current_coverage, args=(coverage.service.snapshot, work))
    waiter.start()
    try:
        # The second look is the one made as the wait for a permit starts.
        assert _within(lambda: len(looks) >= 2)
        assert looks == [True] * len(looks)

        coverage.reload()

        assert _within(lambda: not coverage.resident(1), seconds=30 * STALE_CHECK_SECONDS)
        assert _within(lambda: runs == ["stamp-build-1", "stamp-build-2"])
        assert built_on == []
        permits.release()
        held -= 1
        waiter.join(timeout=5)
    finally:
        for _ in range(held):
            permits.release()
        waiter.join(timeout=5)
    assert built_on == ["stamp-build-2"]


def test_a_request_waiting_its_rebuild_turn_drops_a_replaced_index_while_the_holder_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request took build 1's snapshot and is still reading when a reload serves build 2
    and a second request for the same subject takes the rebuild turn and the permit on
    build 2. The first request then waits its turn. It drops build 1's index within a few
    wait slices, while the holder still builds, starts again on build 2, and is served by the
    holder's rebuild."""
    coverage = _CoverageBuilds(monkeypatch)
    permits = MaterialisePermits(1)
    flights = RebuildFlights()
    coverage.service.register_load_guard(permits.every_permit)
    reading, may_wait, building, may_finish = (threading.Event() for _ in range(4))
    waiter_runs: list[str] = []
    outcomes: list[tuple[str, str, bool]] = []

    def rebuild(name: str, snapshot: CoverageSnapshot) -> None:
        current = functools.partial(coverage.service.serves, snapshot)
        with flights.turn(
            "spec_x", deadline=None, wait_seconds=10, coverage_current=current
        ) as waited:
            if waited:
                outcomes.append((name, snapshot.stamp, True))
                return
            with bounded_materialise_slot(permits, wait_seconds=10, coverage_current=current):
                building.set()
                may_finish.wait(timeout=5)
            outcomes.append((name, snapshot.stamp, False))

    def waiter(snapshot: CoverageSnapshot) -> None:
        waiter_runs.append(snapshot.stamp)
        reading.set()
        may_wait.wait(timeout=5)
        rebuild("waiter", snapshot)

    def request(work: Callable[[CoverageSnapshot], None]) -> threading.Thread:
        thread = threading.Thread(
            target=under_current_coverage, args=(coverage.service.snapshot, work)
        )
        thread.start()
        return thread

    threads = [request(waiter)]
    try:
        assert reading.wait(timeout=5)
        coverage.reload()
        threads.append(request(functools.partial(rebuild, "holder")))
        assert building.wait(timeout=5)
        assert coverage.resident(1)

        may_wait.set()

        assert _within(lambda: not coverage.resident(1), seconds=30 * STALE_CHECK_SECONDS)
        assert _within(lambda: waiter_runs == ["stamp-build-1", "stamp-build-2"])
        assert outcomes == []
    finally:
        may_wait.set()
        may_finish.set()
        for thread in threads:
            thread.join(timeout=5)
    assert sorted(outcomes) == [
        ("holder", "stamp-build-2", False),
        ("waiter", "stamp-build-2", True),
    ]


def test_a_request_at_the_gate_during_a_reload_starts_again_without_taking_a_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request that took build 1's snapshot reaches the permits while a reload to build 2
    holds them. When the reload ends the request takes no permit on build 1: it starts again
    and takes its one permit on build 2."""
    log = _PermitLog()
    loading, may_load = threading.Event(), threading.Event()
    built = itertools.count(1)

    def make_index() -> CoverageIndex:
        if next(built) > 1:
            loading.set()
            may_load.wait(timeout=5)
        return _coverage_index()

    coverage = _CoverageBuilds(monkeypatch, make_index=make_index)
    permits = _CountedPermits(1, log=log)
    coverage.service.register_load_guard(permits.every_permit)
    took, may_wait = threading.Event(), threading.Event()
    looks: list[bool] = []
    built_on: list[str] = []

    def work(snapshot: CoverageSnapshot) -> None:
        took.set()
        may_wait.wait(timeout=5)

        def current() -> bool:
            looks.append(coverage.service.serves(snapshot))
            return looks[-1]

        with bounded_materialise_slot(permits, wait_seconds=10, coverage_current=current):
            built_on.append(snapshot.stamp)

    waiter = threading.Thread(target=under_current_coverage, args=(coverage.service.snapshot, work))
    reload = threading.Thread(target=coverage.reload)
    waiter.start()
    try:
        assert took.wait(timeout=5)
        reload.start()
        assert loading.wait(timeout=5)
        may_wait.set()
        # The request looked once, found build 1 still served, and went on to the gate.
        assert _within(lambda: looks == [True])
    finally:
        may_wait.set()
        may_load.set()
        for thread in (reload, waiter):
            thread.join(timeout=5)
    assert built_on == ["stamp-build-2"]
    # The reload's permit, then the request's one.
    assert log.kinds() == ["acquire", "release", "acquire", "release"]


def test_a_sliced_wait_ends_at_its_bound_and_looks_once_a_slice() -> None:
    """A wait that checks a coverage snapshot lasts as long as one that checks none, and asks
    about the snapshot once before it starts and once a slice after that."""
    permits = MaterialisePermits(1)
    assert permits.acquire(timeout=0)
    looks: list[float] = []

    def current() -> bool:
        looks.append(time.monotonic())
        return True

    wait_seconds = 2.5 * STALE_CHECK_SECONDS
    started = time.monotonic()
    try:
        with (
            pytest.raises(MintBackpressure),
            bounded_materialise_slot(permits, wait_seconds=wait_seconds, coverage_current=current),
        ):
            pass
        elapsed = time.monotonic() - started
    finally:
        permits.release()
    assert wait_seconds <= elapsed < 5.0
    # One look before the wait and one as it ends, with one per slice begun between them. A
    # stalled thread begins fewer slices, never more.
    assert 2 <= len(looks) <= math.ceil(wait_seconds / STALE_CHECK_SECONDS) + 2


def test_a_posted_definition_whose_snapshot_a_reload_replaced_materialises_on_the_served_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A definition posted to the options route of a deployment that never registered it took
    build 1's snapshot, and a reload to build 2 completed before its permit. The call gives the
    permit back, starts again, and materialises against build 2's index after build 1's has
    left memory, as its first SDMX request observes."""
    with _client(materialise_max_concurrency=1, transport=_OverlapTrackingTransport()) as minting:
        minted = minting.post("/api/datasets", json=_scope(statvar=_REF_B))
        assert minted.status_code == 200, minted.text
    definition = minted.json()["definition"]
    log = _PermitLog()
    coverage = _CoverageBuilds(monkeypatch)
    schedule = _reload_before_permit(monkeypatch, log, coverage)
    transport, _ = _logged_transport(log)
    old_index_resident: list[bool] = []
    fetched_on_served_index: list[bool] = []

    def observing_transport(request: httpx.Request) -> httpx.Response:
        old_index_resident.append(coverage.resident(1))
        return transport(request)

    real_prewrite = serve_module.prewrite_definition

    def observing_prewrite(**kwargs):
        fetched_on_served_index.append(kwargs["coverage_index"] is coverage.indexes[2]())
        return real_prewrite(**kwargs)

    monkeypatch.setattr(serve_module, "prewrite_definition", observing_prewrite)
    with _client(
        materialise_max_concurrency=1, transport=observing_transport, coverage=coverage.service
    ) as client:
        schedule(1)
        resp = client.post(
            f"/api/datasets/{definition['spec_id']}/options", json={"definition": definition}
        )

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "materialise"
    assert fetched_on_served_index == [True]
    assert old_index_resident and not any(old_index_resident)
    # The stale pass takes and returns its permit before anything is fetched.
    kinds = [kind for kind in log.kinds() if kind in ("acquire", "release", "sdmx")]
    assert kinds[:4] == ["acquire", "release", "acquire", "sdmx"]
    assert kinds.count("acquire") == kinds.count("release") == 2
    assert "coverage_restarts;desc=1" in resp.headers["server-timing"]


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
