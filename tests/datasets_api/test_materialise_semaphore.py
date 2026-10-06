"""Tests that the materialisation semaphore bounds concurrent mint and options-route work.
A barrier-controlled SDMX transport makes overlap observable without timing assumptions.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

from datasets_api.admission import bounded_materialise_slot
from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.operation_store import NullOperationStore
from datasets_api.operations.task_client import NullTaskClient
from datasets_api.revision import SentinelRevisionReader
from datasets_api.versions import NullVersionRecordWriter, SubjectHead
from dc_sdmx import SdmxClient
from dfo_platform.app import create_app
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

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
    index = build_coverage_index(
        [
            (_STAT_VAR_A, "facet1", "country/USA", "country/KEN", [2015]),
            (_STAT_VAR_B, "facet1", "country/USA", "country/KEN", [2015]),
        ],
        role_names={
            _STAT_VAR_A: RoleNames(donor="provider", recipient="recipient"),
            _STAT_VAR_B: RoleNames(donor="provider", recipient="recipient"),
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


@contextmanager
def _client(
    *,
    materialise_max_concurrency: int,
    transport: _OverlapTrackingTransport,
    store: FakeDatasetStore | None = None,
    graph: FakeMintGraphReader | None = None,
    head_writer=None,
    materialise_wait_seconds: float | None = None,
):
    """Builds a client under `TestClient` lifespan with one portal for threaded requests.
    Snapshot, pin, and meta dependencies satisfy the other doors' startup requirements.
    """
    store = store if store is not None else FakeDatasetStore()
    graph = graph if graph is not None else _default_graph()
    sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(transport))
    surface = datasets_surface(
        settings=(
            DatasetsBuildSettings(materialise_wait_seconds=materialise_wait_seconds)
            if materialise_wait_seconds is not None
            else None
        ),
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        version_writer=NullVersionRecordWriter(),
        materialise_max_concurrency=materialise_max_concurrency,
        mint_pool_size=4,
        head_writer=head_writer,
        # Keep the meta door from opening a live operation-store connection.
        operation_store=NullOperationStore(),
        # Avoid creating a live Cloud Tasks client during lifespan startup.
        task_client=NullTaskClient(),
        coverage=_coverage_service(),
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

    def read_head(self, subject_id: str) -> SubjectHead | None:
        return self._heads.get(subject_id)

    def advance_head(
        self,
        *,
        subject_id: str,
        content_version: str,
        signal,
        expected_content_version: str | None,
    ) -> SubjectHead | None:
        existing = self._heads.get(subject_id)
        stored_content_version = existing.content_version if existing is not None else None
        if stored_content_version != expected_content_version:
            return None
        now = datetime.now(UTC)
        written = SubjectHead(
            subject_id=subject_id,
            content_version=content_version,
            confirmed_at=now,
            materialised_at=now,
            signal=tuple(signal),
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
            signal=existing.signal,
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
            signal=existing.signal,
        )


# A 900 s age triggers a facet probe. A 24 h age forces rematerialisation before snapshot loading.
# This fixture has no snapshot reader.
_PAST_INTERVAL = timedelta(hours=25)


def test_stale_head_probe_waits_for_a_materialise_slot_and_times_out_as_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale-head GET returns retryable 503 after the 0.1 s wait bound while spec A holds the
    slot. Spec B fetches after release. Role-bearing probes read coverage.
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
        transport=_slow_transport,  # type: ignore[arg-type]
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
        transport=_slow_transport,  # type: ignore[arg-type]
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
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
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
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
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
        transport=_slow_transport,  # type: ignore[arg-type]
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
    `MintBackpressure` maps to after the 0.1 s wait bound. After release, it answers. B is
    never registered."""
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
        transport=_slow_transport,  # type: ignore[arg-type]
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

        answered = client.post(options_b, json={"definition": definition_b})
        assert answered.status_code == 200, answered.text
        assert fetch_calls_b["n"] == 1
        assert set(store.state.spec_registry) == registered
        assert definition_b["spec_id"] not in registered
