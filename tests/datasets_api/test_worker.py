"""Pins `run_operation` progress writes, retries, attempt fencing, and execution-time coverage checks.
Graph and materialisation calls are stubbed. Worker state runs through `FakeOperationDatasetStore`."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from code_index.shared import get_index
from dataset_store.snapshot import SnapshotRevoked, StoredArtifactLocation
from datasets_api import current as current_module
from datasets_api import serve as serve_module
from datasets_api.adapter import MemberReadBounds, family_basis
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import (
    DefinitionDeps,
    SpecNotFound,
    classify_definition_exception,
)
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import LiveOperationStore, Operation
from datasets_api.operations import worker as worker_module
from datasets_api.operations.worker import (
    OperationSuperseded,
    OperationTransientFailure,
    run_operation,
)
from datasets_api.render import artifact_rendering
from datasets_api.serve import WorkDeadlinePassed
from datasets_api.version_store import enriched_rendering_key
from datasets_api.versions import HeadFence, HeadFenceLost, SubjectHead
from materialise.errors import VariableBreadthUnreadError
from materialise.facets import read_variable_breadths
from materialise.trace import UNKNOWN_FAMILY, MemberFamily, RecordedFamily, StatedFamily
from series_coverage.index import Estimate

from ._graph_doubles import FakeMintGraphReader
from ._operation_store import FakeOperationDatasetStore
from ._version_record_store import materialisation_record
from .test_resolve_definition_members_guards import _resolve, _value_bound_broad_family

_SUBJECT_ID = "spec_abc123"
_CONFIRMED_AT = datetime(2026, 1, 1, 12, tzinfo=UTC)


class _StubCoverageSnapshot:
    """Provides the `.stamp` and `.index` fields read by `run_operation`."""

    stamp = "cov_stub"
    index = object()


class _StubCoverage:
    """Provides one coverage snapshot to `run_operation`."""

    def snapshot(self):
        return _StubCoverageSnapshot()


# The options `_claimed_operation` stores. They make a CSV bundle, since `MintOptions.bundle`
# defaults to True.
_OPTIONS: dict[str, object] = {"format": "csv"}


def _location(
    options: dict[str, object], *, enrichment_digest: str | None = None
) -> StoredArtifactLocation:
    """A working-tier location stored now under the rendering `options` map to, rendered under
    `enrichment_digest`, which defaults to the serving code index's."""
    if enrichment_digest is None:
        enrichment_digest = get_index().enrichment_digest
    rendering = artifact_rendering(
        MintOptions.model_validate(options), site=DatasetsBuildSettings().site
    )
    now = datetime.now(UTC)
    return StoredArtifactLocation(
        artifact_digest="sha256:stub",
        rendering_key=enriched_rendering_key(rendering, enrichment_digest),
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


def _linked(
    subject_id: str,
    content_version: str,
    *options: dict[str, object],
    enrichment_digest: str | None = None,
) -> _LocationsReader:
    """A snapshot listing one artifact per `options` under `(subject_id, content_version)`,
    rendered under `enrichment_digest`."""
    return _LocationsReader(
        {
            (subject_id, content_version): tuple(
                _location(item, enrichment_digest=enrichment_digest) for item in options
            )
        }
    )


def _deps(snapshot_reader: _LocationsReader | None = None) -> DefinitionDeps:
    """Stub deps whose snapshot lists the claimed operation's rendering under the version
    `_stub_manifest` completes with, unless `snapshot_reader` says otherwise."""
    if snapshot_reader is None:
        snapshot_reader = _linked(_SUBJECT_ID, "cv_stub", _OPTIONS)
    return DefinitionDeps(
        registry=None,  # type: ignore[arg-type]
        spec_reader=None,  # type: ignore[arg-type]
        graph=None,  # type: ignore[arg-type]
        sdmx=None,  # type: ignore[arg-type]
        artifact_store=None,  # type: ignore[arg-type]
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=4,
        enumerator=None,  # type: ignore[arg-type]
        enrichment=None,  # type: ignore[arg-type]
        code_index=get_index(),
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
    lease_seconds: int = 1_500,
    subject_id: str = _SUBJECT_ID,
    member_dcids: list[str] | None = None,
):
    request_identity: dict[str, object] = {"subject_id": subject_id}
    if member_dcids is not None:
        request_identity["member_dcids"] = member_dcids
    operation = operation_store.create_or_get(
        request_identity=request_identity,
        options=_OPTIONS,
        coverage_stamp="cov_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=200_000, requests=5),
    )
    claimed = operation_store.claim(
        operation.id, submission_count=0, lease_seconds=lease_seconds, final_delivery=False
    )
    assert claimed is not None
    return claimed


def test_each_finished_chunk_writes_the_running_total_with_no_thread(
    monkeypatch: pytest.MonkeyPatch,
):
    """Progress is written from the materialiser's own `add` calls, on its own thread."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    seen: list[tuple[int, int, frozenset[int | None]]] = []

    def _chunked_serve(
        stored,
        *,
        resolve_requests,
        options,
        trigger,
        deps,
        trace_id,
        admitted,
        coverage_snapshot,
        progress,
    ):
        for rows in (100, 23):
            progress.add(rows)
            current = operation_store.get(operation.id)
            assert current is not None
            started = frozenset(t.ident for t in threading.enumerate()) - threads_before
            seen.append((current.rows_written, threading.get_ident(), started))
        return _stub_manifest()

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(worker_module, "serve_current", _chunked_serve)
    # Thread identities, not a count: a thread an earlier test left running can
    # end mid-test and lower the count without this code starting or ending one.
    threads_before = frozenset(t.ident for t in threading.enumerate())
    caller = threading.get_ident()

    run_operation(operation=operation, deps=_deps(), operation_store=operation_store)

    assert seen == [(100, caller, frozenset()), (123, caller, frozenset())]
    stored = operation_store.get(operation.id)
    assert stored is not None and stored.state == "done"


def test_non_final_transient_failure_releases_the_lease_and_raises():
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _raise_transient(*, spec_id, deps):
        raise ConnectionError("upstream SDMX call failed")

    def _run():
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(worker_module, "_resolve_spec_requests", _raise_transient)
            run_operation(
                operation=operation,
                deps=_deps(),
                operation_store=operation_store,
                is_final_attempt=False,
            )

    with pytest.raises(OperationTransientFailure):
        _run()

    stored = operation_store.get(operation.id)
    assert stored is not None
    # The released row has no lease, so the next delivery claims it at once.
    assert (stored.state, stored.claimed_at, stored.lease_seconds) == ("pending", None, None)


def test_a_run_past_its_work_deadline_releases_before_it_renders_or_uploads(
    monkeypatch: pytest.MonkeyPatch,
):
    """Materialisation returned after the deadline, so nothing is written under a lease the
    poll may already be about to end."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    monkeypatch.setattr(
        worker_module,
        "_resolve_spec_requests",
        lambda *, spec_id, deps: (SimpleNamespace(spec_id=spec_id, identity={}), []),
    )
    monkeypatch.setattr(serve_module, "prewrite_definition", lambda **_: SimpleNamespace())

    with pytest.raises(OperationTransientFailure) as excinfo:
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            deadline=time.monotonic() - 1,
        )

    assert isinstance(excinfo.value.__cause__, WorkDeadlinePassed)
    stored = operation_store.get(operation.id)
    assert stored is not None
    assert (stored.state, stored.claimed_at) == ("pending", None)


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        pytest.param(
            ConnectionError("upstream SDMX call failed"), "internal_error", id="unclassified"
        ),
        # A single-place tie its unread whole-variable breadth would decide: the run stores no
        # other facet and stays retryable.
        pytest.param(
            VariableBreadthUnreadError(["fEUR", "fUSD"]),
            "service_unavailable",
            id="breadth-unread",
        ),
    ],
)
def test_final_attempt_transient_failure_marks_failed_retryable(failure, code):
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _raise_transient(*, spec_id, deps):
        raise failure

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _raise_transient)
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            is_final_attempt=True,
        )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == code
    # The transient failure remains retryable on resubmission despite exhausting this task's attempts.
    assert stored.retryable is True


def test_run_operation_takes_one_coverage_snapshot_and_forwards_its_stamp_and_index():
    """One coverage snapshot keeps its stamp and index consistent across an artifact swap."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    deps = _deps()

    calls = 0
    real_snapshot = deps.coverage.snapshot

    def _counting_snapshot():
        nonlocal calls
        calls += 1
        return real_snapshot()

    deps.coverage.snapshot = _counting_snapshot  # type: ignore[method-assign]

    captured: dict[str, object] = {}
    stub_manifest = _stub_manifest()

    def _capture_serve_current(
        stored,
        *,
        resolve_requests,
        options,
        trigger,
        deps,
        trace_id,
        admitted,
        coverage_snapshot,
        progress,
    ):
        captured["coverage_stamp"] = coverage_snapshot.stamp
        captured["coverage_index"] = coverage_snapshot.index
        return stub_manifest

    def _stub_resolve(*, spec_id, deps):
        return _STUB_STORED, []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
        mp.setattr(worker_module, "serve_current", _capture_serve_current)
        run_operation(operation=operation, deps=deps, operation_store=operation_store)

    assert calls == 1
    snapshot = real_snapshot()
    assert captured["coverage_stamp"] == snapshot.stamp
    assert captured["coverage_index"] is snapshot.index


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        pytest.param(lambda: SpecNotFound(id=_SUBJECT_ID), "spec_not_found", id="spec-not-found"),
        # A single-place tie whose whole-variable breadth failed to build for a defect: a
        # retry would fail alike.
        pytest.param(
            lambda: VariableBreadthUnreadError(["fEUR", "fUSD"], transient=False),
            "internal_error",
            id="breadth-defect",
        ),
    ],
)
def test_deterministic_failure_fails_regardless_of_final_attempt(failure, code):
    """A deterministic (non-retryable) classification fails the operation
    the same way whether or not this is the last Cloud Tasks attempt.
    `is_final_attempt` only changes the TRANSIENT branch.
    """
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _raise_deterministic(*, spec_id, deps):
        raise failure()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _raise_deterministic)
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            is_final_attempt=False,
        )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == code
    # Resubmitting the identical request repeats this deterministic failure.
    assert stored.retryable is False
    # The row stores the refusal the synchronous door classifies for the same exception, its
    # wire fields alone: the line a classification names is logged, not stored.
    classified = classify_definition_exception(failure())
    assert classified is not None
    assert stored.refusal is not None
    assert stored.refusal.to_canonical_data() == classified.to_canonical_data()


@pytest.mark.parametrize("is_final_attempt", [False, True], ids=["early", "final"])
def test_a_failure_whose_classification_raises_fails_the_row_with_internal_error(
    monkeypatch: pytest.MonkeyPatch, is_final_attempt: bool
):
    """A registry gap makes the classifier raise `LookupError` inside the failure handler. The
    row still reaches a terminal state, retryable so an identical resubmit resets it."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _raise_revoked(*, spec_id, deps):
        raise SnapshotRevoked(id=spec_id, content_version="cv_stub", reason=None)

    def _unmapped(exc):
        raise LookupError("the code registry gives 'snapshot_revoked' the statuses []")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _raise_revoked)
    monkeypatch.setattr(worker_module, "classify_definition_exception", _unmapped)
    run_operation(
        operation=operation,
        deps=_deps(),
        operation_store=operation_store,
        is_final_attempt=is_final_attempt,
    )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert (stored.state, stored.refusal_code, stored.retryable) == (
        "failed",
        "internal_error",
        True,
    )


@pytest.mark.parametrize(
    ("resolved", "state", "refusal_code"),
    [
        (("Count_Person_Male", "Count_Person_Female"), "done", None),
        (("Count_Person",), "failed", "upstream_version_changed"),
    ],
    ids=["unchanged", "changed"],
)
def test_a_definition_whose_members_changed_since_submission_fails_retryable_at_once(
    monkeypatch: pytest.MonkeyPatch,
    resolved: tuple[str, ...],
    state: str,
    refusal_code: str | None,
):
    """The worker resolves the members again from the identity alone. Members other than the
    stored `member_dcids` fail the row on its first attempt, retryable, and serve nothing."""
    operation_store = _operation_store()
    operation = _claimed_operation(
        operation_store, member_dcids=["Count_Person_Female", "Count_Person_Male"]
    )
    served: list[object] = []

    def _resolve(*, spec_id, deps):
        return _STUB_STORED, [SimpleNamespace(stat_var=f"dcid:{dcid}") for dcid in resolved]

    def _serve(stored, **kwargs):
        served.append(kwargs["resolve_requests"]())
        return _stub_manifest()

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _resolve)
    monkeypatch.setattr(
        worker_module, "sync_estimate", lambda requests, **_: Estimate(rows=1, requests=1)
    )
    monkeypatch.setattr(worker_module, "serve_current", _serve)

    run_operation(
        operation=operation,
        deps=_deps(),
        operation_store=operation_store,
        is_final_attempt=False,
    )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert (stored.state, stored.refusal_code) == (state, refusal_code)
    assert len(served) == (1 if state == "done" else 0)
    if state == "failed":
        assert stored.retryable is True


def test_a_member_walk_past_its_deadline_fails_retryable_so_a_resubmission_walks_again():
    """The worker's walk runs in its own process with a cold cache. A walk that runs out of
    time records `timeout`, retryable, so `create_or_get` re-runs an identical submission."""
    identity, arcs = _value_bound_broad_family()
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)

    def _walk_past_deadline(*, spec_id, deps):
        expired = MemberReadBounds(subject_ceiling=20_000, deadline=time.monotonic() - 1)
        _resolve(identity, graph=FakeMintGraphReader(arcs=arcs), bounds=expired)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _walk_past_deadline)
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            is_final_attempt=True,
        )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert (stored.state, stored.refusal_code, stored.retryable) == ("failed", "timeout", True)


def _claimed_operation_with_malformed_options(operation_store: LiveOperationStore):
    """Create a stored row with options that fail `MintOptions.model_validate`."""
    operation = operation_store.create_or_get(
        request_identity={"subject_id": _SUBJECT_ID},
        options={"format": "not-a-real-format"},
        coverage_stamp="cov_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=200_000, requests=5),
    )
    claimed = operation_store.claim(
        operation.id, submission_count=0, lease_seconds=30, final_delivery=False
    )
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
            is_final_attempt=False,
        )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "pending"
    assert stored.claimed_at is None


def test_preamble_failure_marks_failed_on_the_final_attempt():
    operation_store = _operation_store()
    operation = _claimed_operation_with_malformed_options(operation_store)

    run_operation(
        operation=operation,
        deps=_deps(),
        operation_store=operation_store,
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

    stub_manifest = _stub_manifest("cv_stub_orphaned")

    def _stub_resolve(*, spec_id, deps):
        return _STUB_STORED, []

    def _serve_and_expire_out_from_under(
        stored,
        *,
        resolve_requests,
        options,
        trigger,
        deps,
        trace_id,
        admitted,
        coverage_snapshot,
        progress,
    ):
        # Stands in for Cloud Tasks retrying onto another instance once
        # Attempt 2 reclaims the row after this lease lapses, before
        # this call returns its own (successful) manifest.
        now_box["t"] = now_box["t"] + timedelta(seconds=31)
        reclaimed = operation_store.claim(
            operation.id, submission_count=0, lease_seconds=1_500, final_delivery=False
        )
        assert reclaimed is not None and reclaimed.attempt_count == 2
        return stub_manifest

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
        mp.setattr(worker_module, "serve_current", _serve_and_expire_out_from_under)
        with caplog.at_level(logging.ERROR):
            run_operation(
                operation=operation,
                deps=_deps(_linked(_SUBJECT_ID, "cv_stub_orphaned", _OPTIONS)),
                operation_store=operation_store,
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


_STUB_STORED = SimpleNamespace(identity={})
_INCLUSIVE = "provenance/ONEBilateralAndImputedMultilateral"
"""A stored definition of a plain identity, which binds no composite."""
_SERVED_FAMILY = {"family_member_digest": "b" * 64, "family_member_count": 4}


def _stub_resolve(*, spec_id, deps):
    return _STUB_STORED, []


def _stub_manifest(content_version: str = "cv_stub") -> SimpleNamespace:
    return SimpleNamespace(
        definition=SimpleNamespace(spec_id=_SUBJECT_ID),
        content_version=content_version,
        trace={
            "served_from": "materialise",
            "confirmed_at": _CONFIRMED_AT.isoformat(),
            **_SERVED_FAMILY,
        },
    )


def test_a_progress_write_that_loses_the_fence_cancels_the_run_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """A refused progress write cancels the run and leaves the newer attempt's row unchanged."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = _operation_store(now=lambda: now_box["t"])
    operation = _claimed_operation(operation_store, lease_seconds=0)
    reclaimed_box: dict[str, object] = {}
    chunks = 0

    def _chunked_serve(
        stored,
        *,
        resolve_requests,
        options,
        trigger,
        deps,
        trace_id,
        admitted,
        coverage_snapshot,
        progress,
    ):
        nonlocal chunks
        # Cloud Tasks retried onto another instance once this attempt's
        # After the lease lapses, a new attempt reclaims the row.
        now_box["t"] = now_box["t"] + timedelta(seconds=1)
        reclaimed = operation_store.claim(
            operation.id, submission_count=0, lease_seconds=1_500, final_delivery=False
        )
        assert reclaimed is not None
        reclaimed_box["op"] = reclaimed
        for _ in range(3):
            progress.add(10)  # one finished chunk
            chunks += 1
        raise AssertionError("the refused progress write never cancelled the run")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(worker_module, "serve_current", _chunked_serve)

    with pytest.raises(OperationSuperseded):
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
            is_final_attempt=True,
        )

    reclaimed = reclaimed_box["op"]
    assert isinstance(reclaimed, Operation)
    current = operation_store.get(operation.id)
    assert current is not None
    assert chunks > 0
    assert current.state == "running"
    assert current.attempt_count == 2
    assert chunks == 1
    assert current.claimed_at == reclaimed.claimed_at
    assert current.rows_written == reclaimed.rows_written
    assert current.refusal_code is None


def test_a_run_whose_attempt_was_reclaimed_before_it_started_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """`record_estimate` is the first write of the run and is fenced: an
    attempt that no longer owns the row stops there, before any fetch."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = _operation_store(now=lambda: now_box["t"])
    operation = _claimed_operation(operation_store, lease_seconds=30)

    def _resolve_then_lose_the_lease(*, spec_id, deps):
        now_box["t"] = now_box["t"] + timedelta(seconds=31)
        assert operation_store.claim(
            operation.id, submission_count=0, lease_seconds=1_500, final_delivery=False
        )
        return _STUB_STORED, []

    def _must_not_serve(*_, **__):
        raise AssertionError("a superseded attempt must not materialise")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _resolve_then_lose_the_lease)
    monkeypatch.setattr(worker_module, "serve_current", _must_not_serve)

    with pytest.raises(OperationSuperseded):
        run_operation(
            operation=operation,
            deps=_deps(),
            operation_store=operation_store,
        )

    current = operation_store.get(operation.id)
    assert current is not None
    assert current.state == "running"
    assert current.attempt_count == 2
    assert current.estimate == operation.estimate


@dataclasses.dataclass
class _FencedHeadStore:
    """In-memory head store whose fence reads the operation row, as `LiveHeadStore` reads it
    under a share lock in the head's transaction. It records every fence it receives."""

    operation_store: LiveOperationStore
    heads: dict[str, SubjectHead] = dataclasses.field(default_factory=dict)
    fences: list[HeadFence | None] = dataclasses.field(default_factory=list)

    def read_head(self, subject_id: str) -> SubjectHead | None:
        return self.heads.get(subject_id)

    def confirm_head(
        self, *, subject_id: str, content_version: str, family, basis
    ) -> SubjectHead | None:
        return None

    def advance_head(
        self, *, materialisation, family, basis, expected_materialisation_id: int | None, fence=None
    ) -> SubjectHead | None:
        self.fences.append(fence)
        if fence is not None:
            current = self.operation_store.get(fence.operation_id)
            if (
                current is None
                or current.state != "running"
                or current.attempt_count != fence.attempt
            ):
                raise HeadFenceLost(fence)
        now = datetime.now(UTC)
        head = SubjectHead(
            subject_id=materialisation.subject_id,
            content_version=materialisation.content_version,
            confirmed_at=now,
            materialised_at=now,
            materialisation_id=len(self.heads) + 1,
            dependencies=materialisation.dependencies,
            family=RecordedFamily(stated=family, confirmed_at=now, basis=basis),
        )
        self.heads[materialisation.subject_id] = head
        return head


def test_a_run_whose_attempt_is_reclaimed_after_its_version_write_leaves_the_head(
    monkeypatch: pytest.MonkeyPatch,
):
    """The lease lapses and another attempt claims the row after this run's version and artifact
    writes and before its head advance. The advance carries the run's claim, refuses, and the
    run ends superseded with no head and the newer attempt's row unchanged."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = _operation_store(now=lambda: now_box["t"])
    operation = _claimed_operation(operation_store, lease_seconds=30)
    head_store = _FencedHeadStore(operation_store)
    reclaimed_box: dict[str, Operation] = {}

    def _write_then_lose_the_claim(*, prewrite, options, deps, trace_id):
        now_box["t"] = now_box["t"] + timedelta(seconds=31)
        reclaimed = operation_store.claim(
            operation.id, submission_count=0, lease_seconds=1_500, final_delivery=False
        )
        assert reclaimed is not None
        reclaimed_box["op"] = reclaimed
        return _stub_manifest()

    monkeypatch.setattr(
        worker_module,
        "_resolve_spec_requests",
        lambda *, spec_id, deps: (SimpleNamespace(spec_id=spec_id, identity={}), []),
    )
    monkeypatch.setattr(
        serve_module,
        "prewrite_definition",
        lambda **_: SimpleNamespace(
            result=SimpleNamespace(
                content_version="cv_stub", trace=SimpleNamespace(family_statement=UNKNOWN_FAMILY)
            )
        ),
    )
    monkeypatch.setattr(serve_module, "_write_definition_artifacts", _write_then_lose_the_claim)
    monkeypatch.setattr(
        serve_module,
        "_materialisation_record",
        lambda prewrite, **_: materialisation_record(
            subject_id=_SUBJECT_ID, content_version="cv_stub", trigger="worker"
        ),
    )

    with pytest.raises(OperationSuperseded) as excinfo:
        run_operation(
            operation=operation,
            deps=dataclasses.replace(_deps(), head_store=head_store),
            operation_store=operation_store,
            is_final_attempt=True,
        )

    assert isinstance(excinfo.value.__cause__, HeadFenceLost)
    assert head_store.fences == [
        HeadFence(operation_id=operation.id, attempt=operation.attempt_count)
    ]
    assert head_store.heads == {}
    current = operation_store.get(operation.id)
    assert current == reclaimed_box["op"]


def test_run_re_estimates_at_execution_and_sizes_the_fetch_guard_from_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """Execution admits its run with its fresh coverage snapshot and estimate, which size `raw_fetch_guard`."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    deps = _deps()
    fresh = Estimate(rows=1_500_000, requests=12)
    estimated_against: list[object] = []
    captured: dict[str, object] = {}

    def _fresh_estimate(requests, *, index, member_concurrency, composite):
        estimated_against.append(index)
        return fresh

    def _capture_serve(
        stored,
        *,
        resolve_requests,
        options,
        trigger,
        deps,
        trace_id,
        admitted,
        coverage_snapshot,
        progress,
    ):
        captured["admitted"] = admitted
        captured["recorded"] = operation_store.get(operation.id)
        return _stub_manifest()

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(worker_module, "sync_estimate", _fresh_estimate)
    monkeypatch.setattr(worker_module, "serve_current", _capture_serve)

    run_operation(
        operation=operation,
        deps=deps,
        operation_store=operation_store,
    )

    assert estimated_against == [_StubCoverageSnapshot.index]
    assert captured["admitted"] == fresh
    assert captured["admitted"] != operation.estimate
    recorded = captured["recorded"]
    assert isinstance(recorded, Operation)
    assert recorded.estimate == fresh
    assert recorded.coverage_stamp == operation.coverage_stamp


def test_a_composite_run_estimates_the_composed_frame(monkeypatch: pytest.MonkeyPatch):
    """The worker admits a composite against its composed frame, zero rows included, as the
    sync door does: its estimate reads the stored identity's composite declaration."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    stored = SimpleNamespace(
        identity={"facets": {"provenance": {"state": "value", "ref": _INCLUSIVE}}}
    )
    composites: list[object] = []

    def _estimate(requests, *, index, member_concurrency, composite):
        composites.append(composite)
        return Estimate(rows=1, requests=1)

    monkeypatch.setattr(
        worker_module, "_resolve_spec_requests", lambda *, spec_id, deps: (stored, [])
    )
    monkeypatch.setattr(worker_module, "sync_estimate", _estimate)
    monkeypatch.setattr(worker_module, "serve_current", lambda *_, **__: _stub_manifest())

    run_operation(operation=operation, deps=_deps(), operation_store=operation_store)

    assert composites == [get_index().composite(_INCLUSIVE)]
    assert composites[0] is not None


def test_execution_estimate_over_the_async_ceiling_fails_over_ceiling_without_fetching(
    monkeypatch: pytest.MonkeyPatch,
):
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    over = Estimate(rows=DatasetsBuildSettings().async_max_rows + 1, requests=1)

    def _must_not_serve(*_, **__):
        raise AssertionError("an over-ceiling run must not materialise")

    monkeypatch.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
    monkeypatch.setattr(
        worker_module,
        "sync_estimate",
        lambda requests, *, index, member_concurrency, composite: over,
    )
    monkeypatch.setattr(worker_module, "serve_current", _must_not_serve)

    # `over_ceiling` is deterministic. It fails even on a non-final attempt.
    run_operation(
        operation=operation,
        deps=_deps(),
        operation_store=operation_store,
        is_final_attempt=False,
    )

    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == "over_ceiling"
    assert stored.retryable is False
    assert stored.estimate == over


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
            _linked(
                _SUBJECT_ID,
                "cv_stub",
                _OPTIONS,
                enrichment_digest=f"{get_index().enrichment_digest}-before-a-rotation",
            ),
            False,
            "pending",
            None,
            None,
            id="rendered-under-an-older-digest-releases",
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
    linked only as incomplete after a failed graph label or documentation read has another
    rendering key, so it is transient."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store, subject_id=subject_id)
    completed = _stub_manifest()

    def _serve(
        stored,
        *,
        resolve_requests,
        options,
        trigger,
        deps,
        trace_id,
        admitted,
        coverage_snapshot,
        progress,
    ):
        return completed

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _stub_resolve)
        mp.setattr(worker_module, "serve_current", _serve)
        released = state == "pending"
        with pytest.raises(OperationTransientFailure) if released else contextlib.nullcontext():
            run_operation(
                operation=operation,
                deps=_deps(linked),
                operation_store=operation_store,
                is_final_attempt=is_final_attempt,
            )

    row = operation_store.get(operation.id)
    assert row is not None
    assert row.state == state
    assert row.refusal_code == refusal_code
    assert row.retryable == retryable
    assert row.content_version == (completed.content_version if state == "done" else None)
    assert row.confirmed_at == (_CONFIRMED_AT if state == "done" else None)
    done = state == "done"
    assert row.family == (
        RecordedFamily(
            stated=StatedFamily(members=MemberFamily(digest="b" * 64, count=4)),
            confirmed_at=_CONFIRMED_AT,
            basis=family_basis(_deps(linked)),
        )
        if done
        else None
    )


def test_a_worker_run_at_one_place_reads_the_breadth_a_sync_mint_reads():
    """`serve_current`, which the worker and the mint both serve through, binds the deps'
    breadth reader around materialisation, so a probe of one place reads its variable's whole
    breadth there."""
    operation_store = _operation_store()
    operation = _claimed_operation(operation_store)
    reads: list[tuple[str, ...]] = []

    def reader(stat_vars, *, deadline):
        reads.append(tuple(stat_vars))
        return {sv: {"fUSD": 197, "fEUR": 25} for sv in stat_vars}

    deps = dataclasses.replace(_deps(), variable_breadth=reader)
    probed: dict[str, object] = {}

    def _resolve_served(*, spec_id, deps):
        return SimpleNamespace(spec_id=spec_id, identity={}), []

    def _probing_serve_definition(**_kwargs):
        # What `facets.probe_facets` reads for a subject-only request binding one place.
        pending = read_variable_breadths(
            ["dcid:Count_Person"], places=["country/FRA"], deadline=None
        )
        probed.update((sv, breadth.resolve()) for sv, breadth in pending.items())
        return SimpleNamespace(
            manifest=_stub_manifest(),
            head=SimpleNamespace(confirmed_at=datetime.now(UTC)),
            family=StatedFamily(members=MemberFamily(digest="b" * 64, count=4)),
            latest_years={},
            read_series={},
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _resolve_served)
        mp.setattr(current_module, "serve_definition", _probing_serve_definition)
        # The stub manifest carries no trace to stamp or label.
        mp.setattr(current_module, "stamp_trace", lambda manifest, **_: manifest)
        mp.setattr(current_module, "_labelled", lambda manifest, **_: manifest)
        run_operation(operation=operation, deps=deps, operation_store=operation_store)

    assert reads == [("Count_Person",)]
    assert probed == {"Count_Person": {"fUSD": 197, "fEUR": 25}}
    stored = operation_store.get(operation.id)
    assert stored is not None
    assert stored.state == "done"
