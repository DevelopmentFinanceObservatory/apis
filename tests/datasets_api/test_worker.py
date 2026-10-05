"""Pins `run_operation` lease renewal, retries, attempt fencing, and execution-time coverage checks.
Graph and materialisation calls are stubbed. Worker state runs through `FakeOperationDatasetStore`."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset_store.snapshot import StoredArtifactLocation
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import DefinitionDeps, classify_definition_exception
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import LiveOperationStore, Operation
from datasets_api.operations import worker as worker_module
from datasets_api.operations.worker import (
    OperationSuperseded,
    OperationTransientFailure,
    run_operation,
)
from datasets_api.render import artifact_rendering
from datasets_api.serve import raw_fetch_guard
from datasets_api.version_store import rendering_key
from series_coverage.index import Estimate
from series_coverage.shared import CoverageService
from tests.series_coverage.support import build_test_artifact, index_for, pointer_path

from ._operation_store import FakeOperationDatasetStore

_SUBJECT_ID = "spec_abc123"
_COLLECTION_ID = "col_abc123"
# `run_operation` requires `lease_seconds`. This matches the settings default used by callers
# such as `build_worker_app` when they omit it.
_DEFAULT_LEASE_SECONDS = DatasetsBuildSettings().operation_lease_seconds


class _StubCoverageSnapshot:
    """Provides the `.stamp` and `.index` fields read by `run_operation`."""

    stamp = "cov_stub"
    index = object()


class _StubCoverage:
    """Provides one coverage snapshot to `run_operation`."""

    def snapshot_current(self):
        return _StubCoverageSnapshot()


# The options `_claimed_operation` stores. They make a CSV bundle, since `MintOptions.bundle`
# defaults to True.
_OPTIONS: dict[str, object] = {"format": "csv"}


def _location(options: dict[str, object]) -> StoredArtifactLocation:
    """A working-tier location stored now under the rendering `options` map to."""
    rendering = artifact_rendering(
        MintOptions.model_validate(options), site=DatasetsBuildSettings().site
    )
    now = datetime.now(UTC)
    return StoredArtifactLocation(
        artifact_digest="sha256:stub",
        rendering_key=rendering_key(rendering),
        rendering=rendering,
        media_type="application/zip" if rendering["bundle"] else "application/vnd.apache.parquet",
        size_bytes=1,
        created_at=now,
        stored_at=now,
        tier="working",
        bucket="working",
        object_name="stub",
    )


@dataclasses.dataclass(frozen=True)
class _LocationsReader:
    """Answers artifact locations stored under `(subject_id, content_version)`, and none for
    any other subject or version."""

    locations: dict[tuple[str, str], tuple[StoredArtifactLocation, ...]]

    def read_artifact_locations(self, subject_id: str, content_version: str):
        return list(self.locations.get((subject_id, content_version), ()))


def _linked(subject_id: str, content_version: str, *options: dict[str, object]) -> _LocationsReader:
    """A snapshot listing one artifact per `options` under `(subject_id, content_version)`."""
    return _LocationsReader(
        {(subject_id, content_version): tuple(_location(item) for item in options)}
    )


def _deps(snapshot_reader: _LocationsReader | None = None) -> DefinitionDeps:
    """Stub deps whose snapshot lists the claimed operation's rendering under the version
    `_stub_manifest` completes with, unless `snapshot_reader` says otherwise."""
    if snapshot_reader is None:
        snapshot_reader = _linked(_SUBJECT_ID, "cv_stub", _OPTIONS)
    return DefinitionDeps(
        spec_reader=None,  # type: ignore[arg-type]
        graph=None,  # type: ignore[arg-type]
        sdmx=None,  # type: ignore[arg-type]
        revision_reader=None,  # type: ignore[arg-type]
        artifact_store=None,  # type: ignore[arg-type]
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=4,
        enumerator=None,  # type: ignore[arg-type]
        enrichment=None,  # type: ignore[arg-type]
        code_index=None,  # type: ignore[arg-type]
        rate_cache=None,  # type: ignore[arg-type]
        coverage=_StubCoverage(),  # type: ignore[arg-type]
        snapshot_reader=snapshot_reader,  # type: ignore[arg-type]
        head_recheck_interval=timedelta(seconds=DatasetsBuildSettings().head_recheck_seconds),
        head_max_age=timedelta(seconds=DatasetsBuildSettings().head_max_age_seconds),
    )


def _operation_store(now=None) -> LiveOperationStore:
    kwargs = {"store": FakeOperationDatasetStore()}
    if now is not None:
        kwargs["now"] = now
    return LiveOperationStore(**kwargs)


def _claimed_operation(
    operation_store: LiveOperationStore,
    *,
    lease_seconds: float = 1_700,
    subject_id: str = _SUBJECT_ID,
):
    operation = operation_store.create_or_get(
        request_identity={"subject_id": subject_id},
        options=_OPTIONS,
        coverage_stamp="cov_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=200_000, requests=5),
    )
    claimed = operation_store.claim_lease(
        operation.id, submission_count=0, lease_seconds=lease_seconds
    )
    assert claimed is not None
    return claimed


def test_heartbeat_renews_the_lease_while_materialisation_is_still_running(
    monkeypatch: pytest.MonkeyPatch,
):
    operation_store = _operation_store()
    # The short initial lease makes renewal observable as a later `lease_until`.
    operation = _claimed_operation(operation_store, lease_seconds=0.3)
    original_lease_until = operation.lease_until
    assert original_lease_until is not None

    release_event = threading.Event()

    def _slow_resolve(*, spec_id, deps, seed=None):
        release_event.wait(timeout=5)
        raise RuntimeError("test stops the resolve before it returns")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _slow_resolve)

    def _run_and_swallow_transient_failure() -> None:
        # The release raises a transient failure after the heartbeat assertion.
        with contextlib.suppress(OperationTransientFailure):
            run_operation(
                operation=operation,
                deps=_deps(),
                operation_store=operation_store,
                # A short lease with a fast heartbeat tick (lease_seconds / 3,
                # worker.py's own ratio) so the test observes a renewal
                # without waiting anywhere near the default 60s lease.
                lease_seconds=0.3,
            )

    thread = threading.Thread(target=_run_and_swallow_transient_failure)
    thread.start()
    try:
        # Poll for the heartbeat's own first tick to land, well under the
        # test's own 5s timeout on release_event.
        deadline = time.monotonic() + 3
        renewed = False
        while time.monotonic() < deadline:
            current = operation_store.get(operation.id)
            assert current is not None
            if current.lease_until is not None and current.lease_until > original_lease_until:
                renewed = True
                break
            time.sleep(0.02)
        assert renewed, "heartbeat never renewed the lease while the materialisation ran"
    finally:
        release_event.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_heartbeat_reports_nonzero_rows_written_mid_run(monkeypatch: pytest.MonkeyPatch):
    """The heartbeat reports `RowProgress` while materialisation is blocked."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store, lease_seconds=0.3)

    release_event = threading.Event()

    def _stub_resolve(*, spec_id, deps, seed=None):
        return object(), []

    def _slow_serve_definition(
        *, stored, requests, options, deps, ceiling, coverage_stamp, coverage_index, progress
    ):
        progress.add(123)
        release_event.wait(timeout=5)
        raise RuntimeError("test stops serve_definition before it returns")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(worker_module, "serve_definition", _slow_serve_definition)

    def _run_and_swallow_transient_failure() -> None:
        with contextlib.suppress(OperationTransientFailure):
            run_operation(
                operation=operation,
                deps=_deps(),
                operation_store=operation_store,
                lease_seconds=0.3,
            )

    thread = threading.Thread(target=_run_and_swallow_transient_failure)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        observed = 0
        while time.monotonic() < deadline:
            current = operation_store.get(operation.id)
            assert current is not None
            if current.rows_written > 0:
                observed = current.rows_written
                break
            time.sleep(0.02)
        assert observed == 123, "heartbeat never reported the counter's own running total"
    finally:
        release_event.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_non_final_transient_failure_releases_the_lease_and_raises():
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _raise_transient(*, spec_id, deps, seed=None):
        raise ConnectionError("upstream SDMX call failed")

    def _run():
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(worker_module, "_resolve_spec_requests", _raise_transient)
            run_operation(
                operation=operation,
                deps=_deps(),
                operation_store=operation_store,
                lease_seconds=30,
                is_final_attempt=False,
            )

    with pytest.raises(OperationTransientFailure):
        _run()

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "pending"
    # The row remains detectable as claimed by dead-worker checks and is immediately reclaimable.
    assert stored.lease_until is not None


def test_final_attempt_transient_failure_marks_failed_with_internal_error():
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _raise_transient(*, spec_id, deps, seed=None):
        raise ConnectionError("upstream SDMX call failed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _raise_transient)
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            lease_seconds=30,
            is_final_attempt=True,
        )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == "internal_error"
    # The transient failure remains retryable on resubmission despite exhausting this task's attempts.
    assert stored.retryable is True


def test_run_operation_takes_one_coverage_snapshot_and_forwards_its_stamp_and_index():
    """One coverage snapshot keeps its stamp and index consistent across an artifact swap."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    deps = _deps()

    calls = 0
    real_snapshot = deps.coverage.snapshot_current

    def _counting_snapshot():
        nonlocal calls
        calls += 1
        return real_snapshot()

    deps.coverage.snapshot_current = _counting_snapshot  # type: ignore[method-assign]

    captured: dict[str, object] = {}
    stub_manifest = SimpleNamespace(
        definition=SimpleNamespace(spec_id=_SUBJECT_ID),
        collection=None,
        content_version="cv_stub",
    )

    def _capture_serve_definition(
        *, stored, requests, options, deps, ceiling, coverage_stamp, coverage_index, progress
    ):
        captured["coverage_stamp"] = coverage_stamp
        captured["coverage_index"] = coverage_index
        return stub_manifest

    def _stub_resolve(*, spec_id, deps, seed=None):
        return object(), []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
        mp.setattr(worker_module, "serve_definition", _capture_serve_definition)
        run_operation(
            operation=operation, deps=deps, operation_store=operation_store, lease_seconds=30
        )

    assert calls == 1
    snapshot = real_snapshot()
    assert captured["coverage_stamp"] == snapshot.stamp
    assert captured["coverage_index"] is snapshot.index


def test_deterministic_failure_fails_regardless_of_final_attempt():
    """A deterministic (non-retryable) classification fails the operation
    the same way whether or not this is the last Cloud Tasks attempt.
    `is_final_attempt` only changes the TRANSIENT branch.
    """
    from datasets_api.doors.definition import SpecNotFound

    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _raise_deterministic(*, spec_id, deps, seed=None):
        raise SpecNotFound(id=spec_id)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _raise_deterministic)
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            lease_seconds=30,
            is_final_attempt=False,
        )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == "spec_not_found"
    # Resubmitting the identical request repeats this deterministic failure.
    assert stored.retryable is False
    # The row stores the refusal the synchronous door classifies for the same exception.
    classified = classify_definition_exception(SpecNotFound(id=_SUBJECT_ID))
    assert classified is not None
    assert stored.refusal == classified


def _claimed_operation_with_malformed_options(operation_store: LiveOperationStore):
    """Create a stored row with options that fail `MintOptions.model_validate`."""
    operation = operation_store.create_or_get(
        request_identity={"subject_id": _SUBJECT_ID},
        options={"format": "not-a-real-format"},
        coverage_stamp="cov_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=200_000, requests=5),
    )
    claimed = operation_store.claim_lease(operation.id, submission_count=0, lease_seconds=30)
    assert claimed is not None
    return claimed


def test_preamble_failure_releases_the_lease_on_a_non_final_attempt():
    """Malformed stored options release the lease through the transient-failure path."""
    operation_store = _operation_store()
    operation = _claimed_operation_with_malformed_options(operation_store)

    with pytest.raises(OperationTransientFailure):
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            lease_seconds=30,
            is_final_attempt=False,
        )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "pending"
    assert stored.lease_until is not None


def test_preamble_failure_marks_failed_on_the_final_attempt():
    operation_store = _operation_store()
    operation = _claimed_operation_with_malformed_options(operation_store)

    run_operation(
        operation=operation,
        deps=_deps(),
        operation_store=operation_store,
        lease_seconds=30,
        is_final_attempt=True,
    )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == "internal_error"
    assert stored.retryable is True
    # A transient failure records its code alone.
    assert stored.refusal is None


def test_complete_returning_none_logs_an_error_naming_the_operation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A completed artifact is orphaned when a newer attempt reclaims the lease before `complete`."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = _operation_store(now=lambda: now_box["t"])
    operation = _claimed_operation(operation_store, lease_seconds=30)

    stub_manifest = SimpleNamespace(
        definition=SimpleNamespace(spec_id=_SUBJECT_ID),
        collection=None,
        content_version="cv_stub_orphaned",
    )

    def _stub_resolve(*, spec_id, deps, seed=None):
        return object(), []

    def _serve_and_expire_out_from_under(
        *, stored, requests, options, deps, ceiling, coverage_stamp, coverage_index, progress
    ):
        # Stands in for Cloud Tasks retrying onto another instance once
        # Attempt 2 reclaims the row after this lease lapses, before
        # this call returns its own (successful) manifest.
        now_box["t"] = now_box["t"] + timedelta(seconds=31)
        reclaimed = operation_store.claim_lease(
            operation.id, submission_count=0, lease_seconds=1_700
        )
        assert reclaimed is not None and reclaimed.attempt_count == 2
        return stub_manifest

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
        mp.setattr(worker_module, "serve_definition", _serve_and_expire_out_from_under)
        with caplog.at_level(logging.ERROR):
            run_operation(
                operation=operation,
                deps=_deps(_linked(_SUBJECT_ID, "cv_stub_orphaned", _OPTIONS)),
                operation_store=operation_store,
                lease_seconds=_DEFAULT_LEASE_SECONDS,
            )

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any(
        operation.id in r.getMessage()
        and _SUBJECT_ID in r.getMessage()
        and "cv_stub_orphaned" in r.getMessage()
        for r in errors
    )
    # The newer attempt still owns the row. `complete()` left it running.
    stored_op = operation_store.get(operation.id)
    assert stored_op is not None
    assert stored_op.state == "running"
    assert stored_op.attempt_count == 2
    assert stored_op.content_version is None


def _stub_resolve(*, spec_id, deps, seed=None):
    return object(), []


def _stub_manifest(content_version: str = "cv_stub") -> SimpleNamespace:
    return SimpleNamespace(
        definition=SimpleNamespace(spec_id=_SUBJECT_ID),
        collection=None,
        content_version=content_version,
    )


def test_heartbeat_that_loses_the_fence_cancels_the_run_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """A lost heartbeat fence cancels progress and leaves the newer attempt's row unchanged."""
    monkeypatch.setattr(worker_module, "_HEARTBEAT_MIN_INTERVAL_SECONDS", 0.02)
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = _operation_store(now=lambda: now_box["t"])
    operation = _claimed_operation(operation_store, lease_seconds=0.3)
    reclaimed_box: dict[str, object] = {}
    chunks = 0

    def _chunked_serve(
        *, stored, requests, options, deps, ceiling, coverage_stamp, coverage_index, progress
    ):
        nonlocal chunks
        # Cloud Tasks retried onto another instance once this attempt's
        # After the lease lapses, a new attempt reclaims the row.
        now_box["t"] = now_box["t"] + timedelta(seconds=1)
        reclaimed = operation_store.claim_lease(
            operation.id, submission_count=0, lease_seconds=3_600
        )
        assert reclaimed is not None
        reclaimed_box["op"] = reclaimed
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            progress.add(10)  # one finished chunk
            chunks += 1
            time.sleep(0.01)
        raise AssertionError("the heartbeat never cancelled the run")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(worker_module, "serve_definition", _chunked_serve)

    with pytest.raises(OperationSuperseded):
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            lease_seconds=0.3,
            is_final_attempt=True,
        )

    reclaimed = reclaimed_box["op"]
    assert isinstance(reclaimed, Operation)
    current = operation_store.get(operation.id)
    assert current is not None
    assert chunks > 0
    assert current.state == "running"
    assert current.attempt_count == 2
    assert current.lease_until == reclaimed.lease_until
    assert current.rows_written == reclaimed.rows_written
    assert current.refusal_code is None


def test_a_run_whose_attempt_was_reclaimed_before_it_started_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """`record_execution` is the first write of the run and is fenced: an
    attempt that no longer owns the row stops there, before any fetch."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = _operation_store(now=lambda: now_box["t"])
    operation = _claimed_operation(operation_store, lease_seconds=30)

    def _resolve_then_lose_the_lease(*, spec_id, deps, seed=None):
        now_box["t"] = now_box["t"] + timedelta(seconds=31)
        assert operation_store.claim_lease(operation.id, submission_count=0, lease_seconds=60)
        return object(), []

    def _must_not_serve(**_):
        raise AssertionError("a superseded attempt must not materialise")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _resolve_then_lose_the_lease)
    monkeypatch.setattr(worker_module, "serve_definition", _must_not_serve)

    with pytest.raises(OperationSuperseded):
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )

    current = operation_store.get(operation.id)
    assert current is not None
    assert current.state == "running"
    assert current.attempt_count == 2
    assert current.execution_coverage_stamp is None


def test_run_re_estimates_at_execution_and_sizes_the_fetch_guard_from_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """Execution uses its fresh coverage snapshot and estimate to size `raw_fetch_guard`."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    deps = _deps()
    fresh = Estimate(rows=1_500_000, requests=12)
    estimated_against: list[object] = []
    captured: dict[str, object] = {}

    def _fresh_estimate(requests, *, index, member_concurrency):
        estimated_against.append(index)
        return fresh

    def _capture_serve(
        *, stored, requests, options, deps, ceiling, coverage_stamp, coverage_index, progress
    ):
        captured["ceiling"] = ceiling
        captured["recorded"] = operation_store.get(operation.id)
        return _stub_manifest()

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(worker_module, "sync_estimate", _fresh_estimate)
    monkeypatch.setattr(worker_module, "serve_definition", _capture_serve)

    run_operation(
        operation=operation,
        deps=deps,
        operation_store=operation_store,
        lease_seconds=_DEFAULT_LEASE_SECONDS,
    )

    assert estimated_against == [_StubCoverageSnapshot.index]
    assert captured["ceiling"] == raw_fetch_guard(fresh, floor=deps.ceiling)
    assert captured["ceiling"] != raw_fetch_guard(operation.estimate, floor=deps.ceiling)
    recorded = captured["recorded"]
    assert isinstance(recorded, Operation)
    assert recorded.execution_coverage_stamp == _StubCoverageSnapshot.stamp
    assert recorded.estimate == fresh
    assert recorded.coverage_stamp == operation.coverage_stamp


def _coverage_deps(coverage: CoverageService) -> DefinitionDeps:
    return dataclasses.replace(_deps(), coverage=coverage)


def test_run_loads_a_moved_coverage_pointer_before_it_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """At run start, the worker reads a newly published pointer even when the one-hour recheck is not due."""
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1")
    coverage = CoverageService(
        artifact_uri=uri, index=index_for(uri), recheck_seconds=3_600, clock=lambda: 0.0
    )
    build_test_artifact(base, stamp="stamp-2")
    assert coverage.snapshot().stamp == "stamp-1"

    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    estimated_against: list[object] = []
    captured: dict[str, object] = {}

    def _estimate(requests, *, index, member_concurrency):
        estimated_against.append(index)
        return Estimate(rows=1, requests=1)

    def _capture_serve(
        *, stored, requests, options, deps, ceiling, coverage_stamp, coverage_index, progress
    ):
        captured["coverage_stamp"] = coverage_stamp
        captured["coverage_index"] = coverage_index
        return _stub_manifest()

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(worker_module, "sync_estimate", _estimate)
    monkeypatch.setattr(worker_module, "serve_definition", _capture_serve)

    run_operation(
        operation=operation,
        deps=_coverage_deps(coverage),
        operation_store=operation_store,
        lease_seconds=_DEFAULT_LEASE_SECONDS,
    )

    served = coverage.snapshot()
    assert served.stamp == "stamp-2"
    assert captured["coverage_stamp"] == "stamp-2"
    assert captured["coverage_index"] is served.index
    assert estimated_against == [served.index]
    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "done"
    assert stored.execution_coverage_stamp == "stamp-2"


def test_an_unreadable_coverage_pointer_releases_the_run_without_executing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An unreadable pointer prevents materialisation and releases the lease for retry."""
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1")
    coverage = CoverageService(artifact_uri=uri, index=index_for(uri), recheck_seconds=0)
    pointer_path(base).unlink()

    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _must_not_serve(**_):
        raise AssertionError("a run without a current coverage pointer must not materialise")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(worker_module, "serve_definition", _must_not_serve)

    with pytest.raises(OperationTransientFailure):
        run_operation(
            operation=operation,
            deps=_coverage_deps(coverage),
            operation_store=operation_store,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            is_final_attempt=False,
        )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "pending"
    assert stored.execution_coverage_stamp is None


def test_execution_estimate_over_the_async_ceiling_fails_over_ceiling_without_fetching(
    monkeypatch: pytest.MonkeyPatch,
):
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    over = Estimate(rows=DatasetsBuildSettings().async_max_rows + 1, requests=1)

    def _must_not_serve(**_):
        raise AssertionError("an over-ceiling run must not materialise")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(
        worker_module, "sync_estimate", lambda requests, *, index, member_concurrency: over
    )
    monkeypatch.setattr(worker_module, "serve_definition", _must_not_serve)

    # `over_ceiling` is deterministic. It fails even on a non-final attempt.
    run_operation(
        operation=operation,
        deps=_deps(),
        operation_store=operation_store,
        lease_seconds=_DEFAULT_LEASE_SECONDS,
        is_final_attempt=False,
    )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == "over_ceiling"
    assert stored.retryable is False
    assert stored.execution_coverage_stamp == _StubCoverageSnapshot.stamp
    assert stored.estimate == over


def _stub_collection_manifest() -> SimpleNamespace:
    return SimpleNamespace(
        definition=None,
        collection=SimpleNamespace(collection_id=_COLLECTION_ID),
        content_version="cv_collection",
    )


@pytest.mark.parametrize(
    ("subject_id", "linked", "is_final_attempt", "state", "refusal_code", "retryable"),
    [
        pytest.param(
            _SUBJECT_ID,
            _linked(_SUBJECT_ID, "cv_stub", _OPTIONS),
            False,
            "done",
            None,
            None,
            id="linked-completes",
        ),
        pytest.param(
            _COLLECTION_ID,
            _linked(_COLLECTION_ID, "cv_collection", _OPTIONS),
            False,
            "done",
            None,
            None,
            id="collection-linked-completes",
        ),
        pytest.param(
            _SUBJECT_ID,
            _linked(_SUBJECT_ID, "cv_stub"),
            False,
            "pending",
            None,
            None,
            id="unlinked-releases",
        ),
        pytest.param(
            _SUBJECT_ID,
            _linked(_SUBJECT_ID, "cv_stub", {"format": "parquet", "bundle": False}),
            False,
            "pending",
            None,
            None,
            id="other-rendering-only-releases",
        ),
        pytest.param(
            _SUBJECT_ID,
            _linked(_SUBJECT_ID, "cv_stub"),
            True,
            "failed",
            "internal_error",
            True,
            id="unlinked-final-fails",
        ),
    ],
)
def test_run_completes_only_when_the_requests_rendering_is_servable(
    subject_id, linked, is_final_attempt, state, refusal_code, retryable
):
    """The check reads the subject and content version the manifest completes with. A bundle
    left unlinked after a failed graph label or documentation read is transient."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store, subject_id=subject_id)
    is_collection = subject_id == _COLLECTION_ID
    completed = _stub_collection_manifest() if is_collection else _stub_manifest()

    def _serve(
        *, stored, requests, options, deps, ceiling, coverage_stamp, coverage_index, progress
    ):
        return completed

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
        mp.setattr(worker_module, "_resolve_collection_requests", lambda **_: (object(), []))
        mp.setattr(worker_module, "serve_definition", _serve)
        mp.setattr(worker_module, "serve_collection", _serve)
        released = state == "pending"
        with pytest.raises(OperationTransientFailure) if released else contextlib.nullcontext():
            run_operation(
                operation=operation,
                deps=_deps(linked),
                operation_store=operation_store,
                lease_seconds=30,
                is_final_attempt=is_final_attempt,
            )

    row = operation_store.get(operation.id)
    assert row is not None
    assert row.state == state
    assert row.refusal_code == refusal_code
    assert row.retryable == retryable
    assert row.content_version == (completed.content_version if state == "done" else None)
