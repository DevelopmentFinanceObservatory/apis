"""Tests pin `Server-Timing` headers and one `datasets.timings` log per call across the
mint, definition, snapshot, register, and pin doors.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import UTC, datetime

import httpx
import pytest

from artifact_store import Tier
from datasets_api import timings
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.pin import build_pin_door
from datasets_api.doors.snapshot import build_snapshot_door
from datasets_api.kinds import spec_id as spec_id_kind
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.versions import NullHeadWriter
from dc_sdmx import SdmxClient
from materialise import finish as finish_module
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader

from ._app_factories import make_client, offline_allowances
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import pinned_identity, register_body
from ._operation_store import FakeOperationDatasetStore
from ._snapshot_doubles import (
    FakeSnapshotReader,
    FakeSpecReader,
    RecordingVersionWriter,
    _location,
    _store_and_locate,
    _version,
)
from ._snapshot_doubles import make_client_for_door as _make_snapshot_client
from .conftest import FakeDatasetStore, wire_definition_for
from .test_mint_concurrency import _BODY, _coverage_service, _default_graph
from .test_mint_primes_head import _client as _primed_client
from .test_mint_primes_head import _CountingTransport as _PrimedTransport
from .test_mint_primes_head import _graph as _primed_graph
from .test_mint_primes_head import _spec_sidecar as _primed_sidecar
from .test_pin_door import _PRINCIPAL, FakeArtifactReferenceWriter

_NOW = datetime(2026, 6, 1, tzinfo=UTC)

_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR = "MalariaGrantDisbursements"


def _sdmx_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
                "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
                f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
                "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
            ),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _mint_client(*, graph: FakeMintGraphReader | None = None):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _default_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _server_timing_metric_names(header_value: str) -> set[str]:
    """The phases the header names, leaving out its counts (`name;desc=n`)."""
    return {metric.split(";", 1)[0] for metric in header_value.split(", ") if ";dur=" in metric}


def _server_timing_counts(header_value: str) -> dict[str, int]:
    return {
        name: int(value)
        for name, _, value in (
            metric.partition(";desc=") for metric in header_value.split(", ") if ";desc=" in metric
        )
    }


def test_mint_response_carries_server_timing_with_every_expected_phase():
    client = _mint_client()

    resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    names = _server_timing_metric_names(resp.headers["server-timing"])
    assert names == {
        "total",
        "queue",
        "resolve",
        "coverage",
        "estimate",
        "materialise_wait",
        "head_read",
        "materialise",
        "profile",
        "citation",
        "render",
        "store_statistics",
        "store_rendering",
        "store_frame",
        "record",
        "head_advance",
        "retention",
        "handoff",
    }
    counts = _server_timing_counts(resp.headers["server-timing"])
    assert set(counts) == {
        "sdmx_requests",
        "graph_requests",
        "db_transactions",
        "sign_blob_calls",
        "permit_wait_ms",
    }
    assert counts["sdmx_requests"] >= 1
    assert counts["sign_blob_calls"] == 0


def test_definition_retrieval_response_carries_server_timing_and_keeps_no_store():
    client = _mint_client()
    minted = client.post("/api/datasets", json=_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    assert resp.headers["cache-control"] == "no-store"
    names = _server_timing_metric_names(resp.headers["server-timing"])
    assert names == {
        "total",
        "resolve",
        "coverage",
        "materialise_wait",
        "head_read",
        "materialise",
        "profile",
        "citation",
        "render",
        "store_statistics",
        "store_rendering",
        "store_frame",
        "record",
        "head_advance",
        "retention",
    }


class _Delay:
    """Seconds each `_Delayed` double and the SDMX transport sleep per call. Zero
    until a test sets it, so setup calls run at full speed."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def sleep(self) -> None:
        if self.seconds:
            time.sleep(self.seconds)


class _Delayed:
    """Forwards every attribute to `inner` and sleeps `delay.seconds` before each
    forwarded method call, apart from the methods named in `exempt`."""

    def __init__(self, inner, *, delay: _Delay, exempt: frozenset[str] = frozenset()) -> None:
        self._inner = inner
        self._delay = delay
        self._exempt = exempt

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if not callable(attr) or name in self._exempt:
            return attr

        def call(*args, **kwargs):
            self._delay.sleep()
            return attr(*args, **kwargs)

        return call


def _delayed_sdmx_transport(delay: _Delay) -> httpx.MockTransport:
    inner = _sdmx_transport()

    def handler(request: httpx.Request) -> httpx.Response:
        delay.sleep()
        return inner.handle_request(request)

    return httpx.MockTransport(handler)


def _spec_id_mint_client(delay: _Delay):
    """Every dependency the spec_id mint path calls sleeps `delay` per call.

    The artifact store's `open_put` is exempt: the canonical frame write calls it on its own
    thread while the requested rendering runs, and that overlapping `store_frame` span
    would offset an untimed stage in the sum. So is `deferred`, which builds a view of the
    store and calls nothing.
    """
    store = FakeDatasetStore()
    reader = FakeSnapshotReader()
    return make_client(
        registry=_Delayed(LiveSpecRegistry(store=store), delay=delay),
        spec_reader=_Delayed(LiveSpecReader(store=store), delay=delay),
        graph=_Delayed(_default_graph(), delay=delay),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_delayed_sdmx_transport(delay)),
        artifact_store=_Delayed(
            FakeArtifactStore(), delay=delay, exempt=frozenset({"open_put", "deferred"})
        ),
        snapshot_reader=_Delayed(reader, delay=delay),
        version_writer=_Delayed(RecordingVersionWriter(reader=reader), delay=delay),
        head_writer=_Delayed(NullHeadWriter(), delay=delay),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_Delayed(_coverage_service(), delay=delay),
        operation_store=_Delayed(
            LiveOperationStore(store=FakeOperationDatasetStore()), delay=delay
        ),
        task_client=_Delayed(FakeTaskClient(), delay=delay),
    )


def _mint_spec_id(
    client,
    *,
    delay: _Delay,
    seconds: float = 0.0,
    over_budget: pytest.MonkeyPatch | None = None,
) -> httpx.Response:
    """Register a definition through a statvar_set mint, then re-mint it by `spec_id`
    with every dependency sleeping `seconds` per call, over the sync budget when
    `over_budget` is given."""
    minted = client.post("/api/datasets", json=_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    if over_budget is not None:
        over_budget.setattr(spec_id_kind, "within_sync_budget", lambda estimate, **_: False)
    delay.seconds = seconds
    return client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id},
            "options": {"bundle": False},
        },
    )


def test_spec_id_mint_server_timing_names_every_stage():
    delay = _Delay()
    client = _spec_id_mint_client(delay)

    resp = _mint_spec_id(client, delay=delay)

    assert resp.status_code == 200, resp.text
    names = _server_timing_metric_names(resp.headers["server-timing"])
    assert names == {
        "total",
        "queue",
        "resolve",
        "coverage",
        "estimate",
        "materialise_wait",
        "head_read",
        "materialise",
        "profile",
        "citation",
        "render",
        "store_statistics",
        "store_rendering",
        "store_frame",
        "record",
        "head_advance",
        "retention",
        "handoff",
    }


def test_over_budget_spec_id_mint_server_timing_names_every_stage(
    monkeypatch: pytest.MonkeyPatch,
):
    delay = _Delay()
    client = _spec_id_mint_client(delay)

    resp = _mint_spec_id(client, delay=delay, over_budget=monkeypatch)

    assert resp.status_code == 202, resp.text
    names = _server_timing_metric_names(resp.headers["server-timing"])
    assert names == {
        "total",
        "queue",
        "resolve",
        "coverage",
        "head_read",
        "estimate",
        "operation",
        "dispatch",
        "handoff",
    }


_STAGE_DELAY_SECONDS = 0.1
_UNTIMED_TOLERANCE_MS = 25.0
# Phases recorded inside another phase: `profile` inside `materialise`
# (`materialise.finish`), and the connection pool's phases inside whichever
# store call borrowed the connection (`dataset_store.pool`).
_NESTED_PHASES = ("profile", "db_pool_wait", "db_connect", "db_ping")


@pytest.mark.parametrize("over_budget", [False, True], ids=["sync", "operation"])
def test_spec_id_mint_phases_sum_to_the_total(
    monkeypatch: pytest.MonkeyPatch, *, over_budget: bool
):
    """Every dependency call and the in-memory `sync_estimate` sleep 100 ms, so a
    dependency call or the estimate outside every phase leaves at least 100 ms of
    `total` unaccounted for. The sum leaves out `_NESTED_PHASES`, which are
    recorded inside other phases.
    """
    delay = _Delay()
    client = _spec_id_mint_client(delay)
    real_sync_estimate = spec_id_kind.sync_estimate

    def slow_sync_estimate(*args, **kwargs):
        delay.sleep()
        return real_sync_estimate(*args, **kwargs)

    monkeypatch.setattr(spec_id_kind, "sync_estimate", slow_sync_estimate)

    resp = _mint_spec_id(
        client,
        delay=delay,
        seconds=_STAGE_DELAY_SECONDS,
        over_budget=monkeypatch if over_budget else None,
    )

    assert resp.status_code == (202 if over_budget else 200), resp.text
    metrics = _server_timing_metrics(resp.headers["server-timing"])
    total = metrics.pop("total")
    for name in _NESTED_PHASES:
        metrics.pop(name, None)
    untimed = total - sum(metrics.values())
    assert abs(untimed) < _UNTIMED_TOLERANCE_MS, resp.headers["server-timing"]


def _server_timing_metrics(header_value: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric in header_value.split(", "):
        name, _, rest = metric.partition(";dur=")
        if rest:
            metrics[name] = float(rest)
    return metrics


def _server_timing_metric_ms(header_value: str, *, name: str) -> float:
    for metric in header_value.split(", "):
        metric_name, _, rest = metric.partition(";dur=")
        if metric_name == name:
            return float(rest)
    raise AssertionError(f"no {name!r} metric in {header_value!r}")


class _AdvancingClock:
    """Each locked read advances time by `step`, giving tests deterministic phase durations
    without sleeping."""

    def __init__(self, *, step: float) -> None:
        self.step = step
        self.now = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            current = self.now
            self.now += self.step
            return current


def test_mint_server_timing_times_each_stored_artifact_kind(monkeypatch: pytest.MonkeyPatch):
    """The requested rendering, the series statistics and the canonical frame each time their
    write under their own phase. The clock advances 0.05 s per read, so each span is at least
    one step."""
    clock = _AdvancingClock(step=0.05)
    monkeypatch.setattr(timings.time, "perf_counter", clock)
    store = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_default_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )

    mint_resp = client.post("/api/datasets", json=_BODY)
    assert mint_resp.status_code == 200, mint_resp.text
    spec_id = mint_resp.json()["definition"]["spec_id"]
    retrieval_resp = client.get(f"/api/datasets/{spec_id}")
    assert retrieval_resp.status_code == 200, retrieval_resp.text

    for resp in (mint_resp, retrieval_resp):
        for name in _STORE_PHASES:
            assert _server_timing_metric_ms(resp.headers["server-timing"], name=name) >= 49.0


_STORE_PHASES = ("store_rendering", "store_statistics", "store_frame")


def test_head_hit_mint_server_timing_names_the_version_state_read():
    """A repeat mint answers from the head. Its version-state read and the coverage snapshot
    each time under their own phase, so nothing it waits on goes untimed."""
    graph = _primed_graph()
    client = _primed_client(graph=graph, transport=_PrimedTransport())
    spec_id, sidecar = _primed_sidecar(graph=graph)
    body = {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
        "options": {"bundle": False},
    }
    first = client.post("/api/datasets", json=body)
    assert first.status_code == 200, first.text

    repeat = client.post("/api/datasets", json=body)
    retrieval = client.get(f"/api/datasets/{spec_id}?bundle=false")

    for resp in (repeat, retrieval):
        assert resp.status_code == 200, resp.text
        assert resp.json()["trace"]["served_from"] == "head"
    assert _server_timing_metric_names(repeat.headers["server-timing"]) == {
        "total",
        "queue",
        "resolve",
        "coverage",
        "head_read",
        "snapshot_state",
        "retention",
        "handoff",
    }
    assert {"coverage", "head_read", "snapshot_state"} <= _server_timing_metric_names(
        retrieval.headers["server-timing"]
    )


def test_mint_server_timing_profile_phase_measures_the_profile_call(
    monkeypatch: pytest.MonkeyPatch,
):
    """A clock advancing 0.05 s per read makes the measured `profile` phase at least 50 ms."""
    clock = _AdvancingClock(step=0.05)
    monkeypatch.setattr(finish_module.time, "perf_counter", clock)
    client = _mint_client()

    resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    assert _server_timing_metric_ms(resp.headers["server-timing"], name="profile") >= 50.0


def _snapshot_client():
    store = FakeArtifactStore()
    stored = store.put(b"snapshot-bytes", tier=Tier.working)
    location = _location(digest=stored.digest)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_snapshot_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )
    return _make_snapshot_client(door)


def test_snapshot_response_carries_server_timing_and_keeps_no_store():
    client = _snapshot_client()

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    assert resp.headers["cache-control"] == "no-store"
    names = _server_timing_metric_names(resp.headers["server-timing"])
    assert names == {"total", "retention"}


def _register_client():
    return make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))


def test_register_response_carries_server_timing():
    client = _register_client()
    identity = pinned_identity(stat_var="ServerTimingRegister")
    spec_id = recompute_identity(identity).spec_id

    resp = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )

    assert resp.status_code == 200, resp.text
    names = _server_timing_metric_names(resp.headers["server-timing"])
    assert names == {"total", "register"}
    # No artifact URL means no `no-store` header.
    assert "cache-control" not in resp.headers


def _pin_client():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-me-for-server-timing")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_pin_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=FakeArtifactReferenceWriter(),
            allowances=offline_allowances(),
            now=lambda: _NOW,
        ),
        settings=DatasetsBuildSettings.from_env(),
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    return _make_snapshot_client(door), location


def test_pin_response_carries_server_timing():
    client, location = _pin_client()

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 200, resp.text
    names = _server_timing_metric_names(resp.headers["server-timing"])
    # The first pin of an artifact that is only in the working tier runs
    # both the copy into the pinned tier and the reference write.
    assert names == {"total", "copy", "record"}
    # No artifact URL means no `no-store` header.
    assert "cache-control" not in resp.headers


def test_mint_call_logs_one_datasets_timings_line_with_trace_id_and_outcome(
    caplog,
):
    client = _mint_client()

    with caplog.at_level(logging.INFO, logger="datasets_api.doors.mint"):
        resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    timing_records = [r for r in caplog.records if r.message.startswith("datasets.timings")]
    assert len(timing_records) == 1
    message = timing_records[0].message
    assert "door=mint" in message
    assert "outcome=ok" in message
    assert "trace_id=" in message
    assert "resolve=" in message
    assert "trigger=mint served_from=materialise" in message
    assert " rebuild_reason=no_head " in message
    # The SDMX fetch runs on the client's chunk threads, which carry the door's timings. The
    # graph double sends no HTTP request.
    assert re.search(r" sdmx_requests=[1-9]\d* graph_requests=0 ", message), message


def test_snapshot_call_logs_one_datasets_timings_line_with_trace_id_and_outcome(caplog):
    client = _snapshot_client()

    with caplog.at_level(logging.INFO, logger="datasets_api.doors.snapshot"):
        resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    timing_records = [r for r in caplog.records if r.message.startswith("datasets.timings")]
    assert len(timing_records) == 1
    message = timing_records[0].message
    assert "door=snapshot" in message
    assert "outcome=ok" in message
    assert "trace_id=" in message


def test_register_call_logs_one_datasets_timings_line_with_trace_id_and_outcome(caplog):
    client = _register_client()
    identity = pinned_identity(stat_var="ServerTimingRegisterLog")
    spec_id = recompute_identity(identity).spec_id

    with caplog.at_level(logging.INFO, logger="datasets_api.doors.register"):
        resp = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )

    assert resp.status_code == 200, resp.text
    timing_records = [r for r in caplog.records if r.message.startswith("datasets.timings")]
    assert len(timing_records) == 1
    message = timing_records[0].message
    assert "door=register" in message
    assert "outcome=ok" in message
    assert "trace_id=" in message
    assert "register=" in message


def test_pin_call_logs_one_datasets_timings_line_with_trace_id_and_outcome(caplog):
    client, location = _pin_client()

    with caplog.at_level(logging.INFO, logger="datasets_api.doors.pin"):
        resp = client.post(
            "/api/datasets/spec_a/cv1/pin",
            json={"artifact_digest": location.artifact_digest},
            headers={"x-dfo-principal": _PRINCIPAL},
        )

    assert resp.status_code == 200, resp.text
    timing_records = [r for r in caplog.records if r.message.startswith("datasets.timings")]
    assert len(timing_records) == 1
    message = timing_records[0].message
    assert "door=pin" in message
    assert "outcome=ok" in message
    assert "trace_id=" in message
    assert "record=" in message
