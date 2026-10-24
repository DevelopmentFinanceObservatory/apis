"""Tests `LiveOperationStore` against the SQLite-backed `FakeOperationDatasetStore`."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from datasets_api.config import DatasetsBuildSettings
from datasets_api.operation_store import (
    LiveOperationStore,
    OperationBacklogFull,
    classify_operation_store_exception,
    compute_operation_id,
    delete_terminal_operations,
)
from datasets_api.operations.lease import LEASE_SECONDS
from datasets_api.operations.task_client import DISPATCH_DEADLINE_SECONDS
from dfo_platform import Alternative, ProblemIssue, ProblemSpec
from series_coverage.index import Estimate

from ._operation_store import FakeOperationDatasetStore

_CONFIRMED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_IDENTITY = {"input": {"spec_id": "spec_abc"}}
_OPTIONS = {"format": "parquet"}
_STAMP = "cov_stamp_1"
_BUILD = "20260929T000000Z-abcd1234"
_ESTIMATE = Estimate(rows=2_500_000, requests=847)


class _RecordingTransaction:
    """Records SQL statements in transaction order."""

    def __init__(self, inner: Any, executed: list[str]) -> None:
        self._inner = inner
        self._executed = executed

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._executed.append(sql)
        self._inner.execute(sql, params)

    def fetch_one(self, sql: str, params: Sequence[object] = ()):
        self._executed.append(sql)
        return self._inner.fetch_one(sql, params)

    def fetch_all(self, sql: str, params: Sequence[object] = ()):
        self._executed.append(sql)
        return self._inner.fetch_all(sql, params)


class _RecordingOperationDatasetStore:
    """Records SQL statements across transactions."""

    def __init__(self) -> None:
        self._inner = FakeOperationDatasetStore()
        self.executed: list[str] = []

    def close(self) -> None:
        self._inner.close()

    @contextmanager
    def transaction(self) -> Generator[_RecordingTransaction, None, None]:
        with self._inner.transaction() as tx:
            yield _RecordingTransaction(tx, self.executed)


@pytest.fixture
def store():
    fake = FakeOperationDatasetStore()
    try:
        yield LiveOperationStore(store=fake, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    finally:
        fake.close()


def test_compute_operation_id_is_deterministic() -> None:
    id_a = compute_operation_id(
        request_identity=_IDENTITY,
        origin_catalogue_build_id=_BUILD,
        options=_OPTIONS,
    )
    id_b = compute_operation_id(
        request_identity=dict(_IDENTITY),
        origin_catalogue_build_id=_BUILD,
        options=dict(_OPTIONS),
    )
    assert id_a == id_b
    assert id_a.startswith("op_")


def test_a_coverage_build_leaves_the_operation_and_its_completion_in_place(
    store: LiveOperationStore,
) -> None:
    """The coverage stamp stays out of the id, so a resubmission after a coverage build finds
    the done operation. Its rebuild rejected an older head, so the completion stays."""
    first = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    assert store.claim(first.id, submission_count=0, lease_seconds=60, final_delivery=False)
    assert store.complete(
        first.id,
        attempt=1,
        subject_id="spec_abc",
        content_version="cv_1",
        confirmed_at=_CONFIRMED_AT,
    )

    after_build = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp="cov_stamp_2",
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version="cv_0",
    )

    assert (after_build.id, after_build.state, after_build.coverage_stamp) == (
        first.id,
        "done",
        _STAMP,
    )


def test_compute_operation_id_differs_by_rendering() -> None:
    """Different renderings of the same subject and coverage stamp get distinct ids."""
    id_parquet = compute_operation_id(
        request_identity=_IDENTITY,
        origin_catalogue_build_id=_BUILD,
        options={"format": "parquet"},
    )
    id_csv = compute_operation_id(
        request_identity=_IDENTITY,
        origin_catalogue_build_id=_BUILD,
        options={"format": "csv"},
    )

    assert id_parquet != id_csv


def test_compute_operation_id_differs_by_origin_catalogue_build() -> None:
    """A new catalogue build can admit different provenances, so it gets a distinct id."""
    ids = {
        compute_operation_id(
            request_identity=_IDENTITY,
            origin_catalogue_build_id=build,
            options=_OPTIONS,
        )
        for build in (_BUILD, "20261006T000000Z-ef567890")
    }

    assert len(ids) == 2


def test_a_completed_operation_is_not_reused_under_a_new_catalogue_build(
    store: LiveOperationStore,
) -> None:
    first = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    claimed = store.claim(first.id, submission_count=0, lease_seconds=60, final_delivery=False)
    assert claimed is not None
    assert store.complete(
        first.id,
        attempt=1,
        subject_id="spec_abc",
        content_version="cv_1",
        confirmed_at=_CONFIRMED_AT,
    )

    rebuilt = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id="20261006T000000Z-ef567890",
        estimate=_ESTIMATE,
        rejected_version=None,
    )

    assert rebuilt.id != first.id
    assert rebuilt.state == "pending"
    assert first.origin_catalogue_build_id == _BUILD
    assert rebuilt.origin_catalogue_build_id == "20261006T000000Z-ef567890"


def test_create_or_get_is_idempotent_on_repeat_submit_of_the_same_rendering(
    store: LiveOperationStore,
) -> None:
    first = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    assert first.state == "pending"
    assert first.rows_written == 0
    assert first.estimate == _ESTIMATE

    # A changed estimate must not overwrite the first submission's estimate.
    second = store.create_or_get(
        request_identity=_IDENTITY,
        options=dict(_OPTIONS),
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=Estimate(rows=1, requests=1),
        rejected_version=None,
    )
    assert second.id == first.id
    assert second.options == _OPTIONS
    assert second.estimate == _ESTIMATE


def test_create_or_get_starts_a_different_operation_for_a_different_rendering(
    store: LiveOperationStore,
) -> None:
    parquet = store.create_or_get(
        request_identity=_IDENTITY,
        options={"format": "parquet"},
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    csv = store.create_or_get(
        request_identity=_IDENTITY,
        options={"format": "csv"},
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )

    assert parquet.id != csv.id
    assert parquet.options == {"format": "parquet"}
    assert csv.options == {"format": "csv"}


def test_create_or_get_refuses_a_new_operation_at_the_backlog_max() -> None:
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(
            store=fake, now=lambda: datetime(2026, 1, 1, tzinfo=UTC), backlog_max=2
        )
        store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        store.create_or_get(
            request_identity={"input": {"spec_id": "spec_b"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        with pytest.raises(OperationBacklogFull) as excinfo:
            store.create_or_get(
                request_identity={"input": {"spec_id": "spec_c"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
                rejected_version=None,
            )
        assert excinfo.value.in_flight == 2
        assert excinfo.value.limit == 2

        spec = classify_operation_store_exception(excinfo.value)
        assert spec is not None
        assert spec.status == 503
        assert spec.code == "service_unavailable"
        assert spec.retryable is True
        assert spec.headers == {"Retry-After": "1"}
    finally:
        fake.close()


def test_create_or_get_at_the_backlog_max_still_returns_an_existing_operation() -> None:
    """The backlog cap applies to new operations. Existing operations remain available."""
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(
            store=fake, now=lambda: datetime(2026, 1, 1, tzinfo=UTC), backlog_max=1
        )
        first = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        second = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        assert second.id == first.id
    finally:
        fake.close()


def test_create_or_get_backlog_count_excludes_terminal_operations() -> None:
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(
            store=fake, now=lambda: datetime(2026, 1, 1, tzinfo=UTC), backlog_max=1
        )
        first = store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        store.claim(first.id, submission_count=0, lease_seconds=60, final_delivery=False)
        store.complete(
            first.id,
            attempt=1,
            subject_id="spec_a",
            content_version="cv_1",
            confirmed_at=_CONFIRMED_AT,
        )

        second = store.create_or_get(
            request_identity={"input": {"spec_id": "spec_b"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        assert second.state == "pending"
    finally:
        fake.close()


@pytest.mark.parametrize(
    ("final_delivery", "elapsed", "admitted"),
    [
        pytest.param(True, 60, False, id="final-live-lease"),
        pytest.param(True, 61, True, id="final-lease-ended"),
        pytest.param(False, 61, False, id="delivery-due-lease-ended"),
        pytest.param(False, 600, False, id="delivery-due-at-the-deadline"),
        pytest.param(False, 601, True, id="delivery-due-past-the-deadline"),
    ],
)
def test_create_or_get_backlog_count_excludes_a_run_nothing_will_finish(
    final_delivery: bool, elapsed: int, admitted: bool
) -> None:
    """A run whose lease ended holds its backlog slot while a later delivery is due."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(
            store=fake, now=lambda: now_box["t"], backlog_max=1, deadline_seconds=600
        )
        first = store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        assert store.claim(
            first.id, submission_count=0, lease_seconds=60, final_delivery=final_delivery
        )
        now_box["t"] += timedelta(seconds=elapsed)

        def _second():
            return store.create_or_get(
                request_identity={"input": {"spec_id": "spec_b"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
                rejected_version=None,
            )

        if admitted:
            assert _second().state == "pending"
        else:
            with pytest.raises(OperationBacklogFull):
                _second()
    finally:
        fake.close()


def _create(store: LiveOperationStore, spec_id: str):
    return store.create_or_get(
        request_identity={"input": {"spec_id": spec_id}},
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )


@pytest.mark.parametrize("dispatched", [True, False], ids=["dispatched", "never-dispatched"])
@pytest.mark.parametrize(("elapsed", "abandoned"), [(600, False), (601, True)])
def test_a_pending_row_past_the_operation_deadline_leaves_the_backlog(
    elapsed: int, abandoned: bool, dispatched: bool
) -> None:
    """An undispatched row is queued from its submission."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(
            store=fake, now=lambda: now_box["t"], backlog_max=1, deadline_seconds=600
        )
        first = _create(store, "spec_a")
        if dispatched:
            assert store.mark_dispatched(first.id, submission_count=0)
        now_box["t"] += timedelta(seconds=elapsed)

        if abandoned:
            assert _create(store, "spec_b").state == "pending"
        else:
            with pytest.raises(OperationBacklogFull):
                _create(store, "spec_b")
    finally:
        fake.close()


@pytest.mark.parametrize("dispatched", [True, False], ids=["dispatched", "never-dispatched"])
@pytest.mark.parametrize(("elapsed", "abandoned"), [(600, False), (601, True)])
def test_a_pending_row_past_the_operation_deadline_resets_on_resubmit(
    elapsed: int, abandoned: bool, dispatched: bool
) -> None:
    """The reset is a new submission, so its task gets a new name."""
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    now_box = {"t": t0}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"], deadline_seconds=600)
        first = _create(store, "spec_a")
        if dispatched:
            assert store.mark_dispatched(first.id, submission_count=0)
        now_box["t"] += timedelta(seconds=elapsed)

        resubmitted = _create(store, "spec_a")

        assert resubmitted.state == "pending"
        if abandoned:
            assert (resubmitted.submission_count, resubmitted.submitted_at) == (1, now_box["t"])
            assert resubmitted.dispatched_at is None
        else:
            assert (resubmitted.submission_count, resubmitted.submitted_at) == (0, t0)
    finally:
        fake.close()


@pytest.mark.parametrize(("idle", "admitted"), [(LEASE_SECONDS, False), (LEASE_SECONDS + 1, True)])
def test_a_running_row_with_no_claim_time_holds_a_backlog_slot_while_its_heartbeat_is_fresh(
    idle: int, admitted: bool
) -> None:
    """An image that renewed its lease through `updated_at` and recorded no delivery claimed
    the row."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"], backlog_max=1)
        first = _create(store, "spec_a")
        assert store.claim(first.id, submission_count=0, lease_seconds=60, final_delivery=False)
        with fake.transaction() as tx:
            tx.execute(
                "UPDATE datasets.operation SET claimed_at = NULL, lease_seconds = NULL, "
                "final_delivery = NULL WHERE id = %s",
                (first.id,),
            )
        now_box["t"] += timedelta(seconds=idle)

        if admitted:
            assert _create(store, "spec_b").state == "pending"
        else:
            with pytest.raises(OperationBacklogFull):
                _create(store, "spec_b")
    finally:
        fake.close()


def test_the_operation_deadline_default_lets_every_delivery_wait_out_a_full_backlog_and_run():
    """The queue's locals in `infra`: 3 attempts, a 1,680 s backoff (the lease plus 60 s
    skew) and 4 concurrent dispatches. A change to the backlog max, the attempts, the lease or
    the dispatch deadline moves the deadline, so this fails until the default follows."""
    settings = DatasetsBuildSettings()
    dispatch_concurrency = 4
    backoff = LEASE_SECONDS + 60
    attempts = settings.operation_max_attempts
    queue_wait = (
        math.ceil(settings.operation_backlog_max / dispatch_concurrency) * DISPATCH_DEADLINE_SECONDS
    )
    deliveries = attempts * (queue_wait + DISPATCH_DEADLINE_SECONDS)

    assert (attempts, backoff, settings.operation_backlog_max) == (3, 1_680, 20)
    assert settings.operation_deadline_seconds == deliveries + (attempts - 1) * backoff + 240


def test_create_or_get_takes_the_advisory_lock_before_counting_on_a_new_row() -> None:
    """The new-row branch takes the advisory lock before counting backlog rows."""
    recording = _RecordingOperationDatasetStore()
    try:
        store = LiveOperationStore(store=recording, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        statements = [s.strip().upper() for s in recording.executed]
        lock_index = next(
            i for i, s in enumerate(statements) if s.startswith("SELECT PG_ADVISORY_XACT_LOCK")
        )
        count_index = next(i for i, s in enumerate(statements) if "COUNT(*)" in s)
        assert lock_index < count_index
    finally:
        recording.close()


def test_create_or_get_skips_the_advisory_lock_for_an_existing_operation() -> None:
    """The advisory lock guards new-row admission."""
    recording = _RecordingOperationDatasetStore()
    try:
        store = LiveOperationStore(store=recording, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        recording.executed.clear()

        store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        assert not any(
            s.strip().upper().startswith("SELECT PG_ADVISORY_XACT_LOCK") for s in recording.executed
        )
    finally:
        recording.close()


def test_claim_moves_pending_to_running(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )

    claimed = store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)

    assert claimed is not None
    assert claimed.state == "running"
    assert claimed.attempt_count == 1
    assert (claimed.claimed_at, claimed.lease_seconds) == (
        datetime(2026, 1, 1, tzinfo=UTC),
        60,
    )


def test_claim_refuses_a_live_lease(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    store.claim(op.id, submission_count=0, lease_seconds=3600, final_delivery=False)

    second_claim = store.claim(op.id, submission_count=0, lease_seconds=3600, final_delivery=False)

    assert second_claim is None
    assert store.get(op.id).attempt_count == 1


@pytest.mark.parametrize(("elapsed", "attempt"), [(30, None), (31, 2)])
def test_claim_takes_over_a_lease_only_once_it_has_ended(elapsed: int, attempt: int | None) -> None:
    now = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    live_store = LiveOperationStore(store=FakeOperationDatasetStore(), now=lambda: now["t"])
    op = live_store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    live_store.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)

    now["t"] = now["t"] + timedelta(seconds=elapsed)
    retaken = live_store.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)

    assert (retaken.attempt_count if retaken is not None else None) == attempt


def test_claim_refuses_a_terminal_operation(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    store.complete(
        op.id,
        attempt=1,
        subject_id="spec_abc",
        content_version="cv_deadbeef",
        confirmed_at=_CONFIRMED_AT,
    )

    assert store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False) is None


def test_progress_only_applies_while_running(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )

    assert store.progress(op.id, attempt=1, rows_written=100) is None

    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    progressed = store.progress(op.id, attempt=1, rows_written=1_000_000)

    assert progressed is not None
    assert progressed.rows_written == 1_000_000


def test_complete_sets_the_manifest_reference_and_is_running_only(
    store: LiveOperationStore,
) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )

    assert (
        store.complete(
            op.id,
            attempt=1,
            subject_id="spec_abc",
            content_version="cv_deadbeef",
            confirmed_at=_CONFIRMED_AT,
        )
        is None
    )

    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    done = store.complete(
        op.id,
        attempt=1,
        subject_id="spec_abc",
        content_version="cv_deadbeef",
        confirmed_at=_CONFIRMED_AT,
    )

    assert done is not None
    assert done.state == "done"
    assert done.subject_id == "spec_abc"
    assert done.content_version == "cv_deadbeef"
    assert done.confirmed_at == _CONFIRMED_AT

    assert (
        store.complete(
            op.id,
            attempt=1,
            subject_id="spec_abc",
            content_version="cv_deadbeef",
            confirmed_at=_CONFIRMED_AT,
        )
        is None
    )


def test_fail_sets_the_refusal_code_and_is_running_only(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )

    assert store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False) is None

    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    failed = store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False)

    assert failed is not None
    assert failed.state == "failed"
    assert failed.refusal_code == "excel_limit"
    assert failed.retryable is False


_REFUSAL = ProblemSpec(
    status=422,
    code="facet_not_available",
    retryable=False,
    detail="no facet matches the requested unit",
    pointer="/input/scope/facets",
    issues=[
        ProblemIssue(
            member="dcid:sv_a",
            axis="unit",
            alternatives=[Alternative(ref="USD", label="US dollars")],
            truncated=False,
        )
    ],
)


def test_fail_stores_the_classified_refusal_beside_its_code(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    assert op.refusal is None
    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)

    failed = store.fail(
        op.id, attempt=1, refusal_code="facet_not_available", retryable=False, refusal=_REFUSAL
    )

    assert failed is not None
    assert failed.refusal == _REFUSAL
    stored = store.get(op.id)
    assert stored is not None
    assert stored.refusal == _REFUSAL


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param({"severity": "high"}, id="key_from_a_newer_writer"),
        pytest.param({"count": "many"}, id="wrong_type"),
    ],
)
def test_an_unparseable_stored_refusal_reads_as_none_and_keeps_the_code(
    corruption: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
        store.fail(
            op.id, attempt=1, refusal_code="facet_not_available", retryable=False, refusal=_REFUSAL
        )
        with fake.transaction() as tx:
            tx.execute(
                "UPDATE datasets.operation SET refusal = %s WHERE id = %s",
                ({**_REFUSAL.to_canonical_data(), **corruption}, op.id),
            )

        with caplog.at_level("WARNING", logger="datasets_api.operation_store"):
            stored = store.get(op.id)
    finally:
        fake.close()

    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == "facet_not_available"
    assert stored.refusal is None
    assert op.id in caplog.text
    assert next(iter(corruption)) in caplog.text


def test_fail_without_a_refusal_leaves_the_column_null(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)

    failed = store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

    assert failed is not None
    assert failed.refusal is None


def test_a_retryable_reset_clears_a_stored_refusal(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    store.fail(op.id, attempt=1, refusal_code="upstream_error", retryable=True, refusal=_REFUSAL)

    resubmitted = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )

    assert resubmitted.state == "pending"
    assert resubmitted.refusal_code is None
    assert resubmitted.refusal is None


def test_fail_persists_retryable_true_for_a_transient_failure(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)

    failed = store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

    assert failed is not None
    assert failed.retryable is True


def test_release_moves_running_back_to_pending_and_clears_the_claim(
    store: LiveOperationStore,
) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )

    assert store.release(op.id, attempt=1) is None

    claimed = store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=True)
    assert claimed is not None
    assert (claimed.attempt_count, claimed.final_delivery) == (1, True)

    released = store.release(op.id, attempt=1)

    assert released is not None
    assert (
        released.state,
        released.claimed_at,
        released.lease_seconds,
        released.final_delivery,
    ) == ("pending", None, None, None)
    # Release preserves the attempt count.
    assert released.attempt_count == 1

    # A released row is immediately claimable.
    reclaimed = store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    assert reclaimed is not None
    assert (reclaimed.attempt_count, reclaimed.final_delivery) == (2, False)


def test_release_is_a_no_op_once_terminal(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    store.complete(
        op.id,
        attempt=1,
        subject_id="spec_abc",
        content_version="cv_deadbeef",
        confirmed_at=_CONFIRMED_AT,
    )

    assert store.release(op.id, attempt=1) is None
    assert store.get(op.id).state == "done"


def test_fail_unservable_done_fails_only_a_done_row_at_the_polled_attempt(
    store: LiveOperationStore,
) -> None:
    """A poll that read an earlier attempt's completion leaves a re-run's completion done, even
    when the re-run repeats the content version."""
    submission = {
        "request_identity": _IDENTITY,
        "options": _OPTIONS,
        "coverage_stamp": _STAMP,
        "origin_catalogue_build_id": _BUILD,
        "estimate": _ESTIMATE,
        "rejected_version": None,
    }
    op = store.create_or_get(**submission)
    assert store.fail_unservable_done(op.id, attempt=0) is None
    assert (
        store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False) is not None
    )
    assert store.fail_unservable_done(op.id, attempt=1) is None
    running = store.get(op.id)
    assert running is not None and running.state == "running"
    assert store.complete(
        op.id, attempt=1, subject_id="spec_abc", content_version="cv_1", confirmed_at=_CONFIRMED_AT
    )

    first = store.fail_unservable_done(op.id, attempt=1)

    assert first is not None
    assert (first.state, first.refusal_code, first.retryable) == ("failed", "internal_error", True)

    # The resubmit re-runs it under a higher attempt and completes with the same version.
    resubmitted = store.create_or_get(**submission)
    assert (resubmitted.state, resubmitted.submission_count) == ("pending", 1)
    rerun = store.claim(op.id, submission_count=1, lease_seconds=60, final_delivery=False)
    assert rerun is not None and rerun.attempt_count == 2
    assert store.complete(
        op.id, attempt=2, subject_id="spec_abc", content_version="cv_1", confirmed_at=_CONFIRMED_AT
    )

    assert store.fail_unservable_done(op.id, attempt=1) is None
    done = store.get(op.id)
    assert done is not None and done.state == "done"

    second = store.fail_unservable_done(op.id, attempt=2)

    assert second is not None
    assert (second.state, second.refusal_code, second.retryable) == (
        "failed",
        "internal_error",
        True,
    )
    assert store.fail_unservable_done(op.id, attempt=2) is None
    assert store.fail_unservable_done("op_unknown", attempt=1) is None


def _reclaimed(now_box: dict[str, datetime], store: LiveOperationStore) -> str:
    """Creates an operation reclaimed for attempt 2 and returns its id."""
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    first = store.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
    assert first is not None and first.attempt_count == 1
    now_box["t"] = now_box["t"] + timedelta(seconds=31)
    second = store.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
    assert second is not None and second.attempt_count == 2
    return op.id


def test_stale_attempt_progress_is_refused_after_a_reclaim() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _reclaimed(now_box, store)
        claimed_before = store.get(id_).claimed_at

        assert store.progress(id_, attempt=1, rows_written=99) is None

        current = store.get(id_)
        assert current.rows_written == 0
        assert current.claimed_at == claimed_before
        assert store.progress(id_, attempt=2, rows_written=5) is not None
    finally:
        fake.close()


def test_stale_attempt_complete_is_refused_after_a_reclaim() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _reclaimed(now_box, store)

        assert (
            store.complete(
                id_,
                attempt=1,
                subject_id="spec_abc",
                content_version="cv_1",
                confirmed_at=_CONFIRMED_AT,
            )
            is None
        )
        assert store.get(id_).state == "running"

        done = store.complete(
            id_,
            attempt=2,
            subject_id="spec_abc",
            content_version="cv_2",
            confirmed_at=_CONFIRMED_AT,
        )
        assert done is not None and done.content_version == "cv_2"
    finally:
        fake.close()


def test_stale_attempt_fail_is_refused_after_a_reclaim() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _reclaimed(now_box, store)

        assert store.fail(id_, attempt=1, refusal_code="excel_limit", retryable=False) is None
        assert store.get(id_).state == "running"
    finally:
        fake.close()


def test_stale_attempt_release_is_refused_after_a_reclaim() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _reclaimed(now_box, store)

        assert store.release(id_, attempt=1) is None
        current = store.get(id_)
        assert current.state == "running"
        assert current.attempt_count == 2
    finally:
        fake.close()


def test_claim_refuses_a_delivery_from_an_earlier_submission(
    store: LiveOperationStore,
) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)
    reset = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    assert reset.submission_count == 1

    assert store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False) is None
    assert store.get(op.id).state == "pending"

    claimed = store.claim(op.id, submission_count=1, lease_seconds=60, final_delivery=False)
    assert claimed is not None
    # Attempt counts remain unique across submissions.
    assert claimed.attempt_count == 2


def test_get_returns_none_for_an_unknown_id(store: LiveOperationStore) -> None:
    assert store.get("op_does_not_exist") is None


def test_get_returns_a_terminal_operation(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False)

    read_back = store.get(op.id)

    assert read_back is not None
    assert read_back.state == "failed"


def test_create_or_get_returns_a_deterministic_failed_row_unchanged() -> None:
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
        store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False)

        resubmitted = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        assert resubmitted.id == op.id
        assert resubmitted.state == "failed"
        assert resubmitted.refusal_code == "excel_limit"
        assert resubmitted.retryable is False
        assert resubmitted.submission_count == 0
    finally:
        fake.close()


def test_create_or_get_resets_a_retryable_failed_row_to_pending() -> None:
    """A transient failure resets to pending on resubmission and increments `submission_count`."""
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

        resubmitted = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        assert resubmitted.id == op.id
        assert resubmitted.state == "pending"
        assert resubmitted.refusal_code is None
        assert resubmitted.retryable is None
        assert resubmitted.rows_written == 0
        assert (resubmitted.claimed_at, resubmitted.dispatched_at) == (None, None)
        # `submission_count` changes; the operation's hashed identity stays fixed.
        assert resubmitted.submission_count == 1
        assert resubmitted.estimate == _ESTIMATE

        store.claim(resubmitted.id, submission_count=1, lease_seconds=60, final_delivery=False)
        store.fail(resubmitted.id, attempt=2, refusal_code="internal_error", retryable=True)
        twice_resubmitted = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        assert twice_resubmitted.submission_count == 2
    finally:
        fake.close()


@pytest.mark.parametrize(
    ("rejected_version", "state"),
    [
        pytest.param("cv_1", "pending", id="rejected-head-is-the-done-version"),
        pytest.param(None, "pending", id="no-head"),
        pytest.param("cv_0", "done", id="done-after-the-rejected-head"),
    ],
)
def test_create_or_get_resets_a_done_row_the_rebuild_rejected(
    store: LiveOperationStore, rejected_version: str | None, state: str
) -> None:
    """A rebuild that rejected the head at the done row's version, or found no head, restarts
    the operation, so its 202 has something to wait for. A done row at another version completed
    after the caller read the head and stays."""
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=None,
    )
    assert store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
    assert store.complete(
        op.id,
        attempt=1,
        subject_id="spec_abc",
        content_version="cv_1",
        confirmed_at=_CONFIRMED_AT,
    )

    resubmitted = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
        rejected_version=rejected_version,
    )

    assert (resubmitted.id, resubmitted.state) == (op.id, state)
    assert resubmitted.submission_count == (0 if state == "done" else 1)
    if state == "pending":
        claimed = store.claim(op.id, submission_count=1, lease_seconds=60, final_delivery=False)
        assert claimed is not None
        assert claimed.attempt_count == 2


def test_create_or_get_reset_of_a_retryable_failed_row_takes_the_backlog_lock() -> None:
    """Retryable resubmissions pass the same backlog admission as new operations."""
    recording = _RecordingOperationDatasetStore()
    try:
        store = LiveOperationStore(store=recording, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)
        recording.executed.clear()

        store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        statements = [s.strip().upper() for s in recording.executed]
        lock_index = next(
            i for i, s in enumerate(statements) if s.startswith("SELECT PG_ADVISORY_XACT_LOCK")
        )
        count_index = next(i for i, s in enumerate(statements) if "COUNT(*)" in s)
        assert lock_index < count_index
    finally:
        recording.close()


def test_create_or_get_reset_survives_a_concurrent_reset_of_the_same_row() -> None:
    """A concurrent reset between the initial read and backlog lock leaves the row pending.

    The fake commits that reset at `before_backlog_lock`; the locked re-read must return it.
    """
    now = datetime(2026, 1, 1, tzinfo=UTC)
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now, backlog_max=1)
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

        def _concurrent_reset(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE datasets.operation SET state = 'pending', claimed_at = NULL, "
                "lease_seconds = NULL, "
                "refusal_code = NULL, retryable = NULL, rows_written = 0, "
                "submission_count = submission_count + 1, updated_at = ? "
                "WHERE id = ? AND state = 'failed' AND retryable",
                (now.isoformat(), op.id),
            )
            fake.before_backlog_lock = None

        fake.before_backlog_lock = _concurrent_reset

        # The pre-lock read sees failure; the concurrent reset makes this row pending at the cap.
        resubmitted = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        assert resubmitted.id == op.id
        assert resubmitted.state == "pending"
        # The concurrent reset increments submission_count once.
        assert resubmitted.submission_count == 1
    finally:
        fake.close()


def test_create_or_get_new_row_survives_a_concurrent_insert_of_the_same_id() -> None:
    """A concurrent insert between the initial read and lock is returned after the locked re-read."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now, backlog_max=1)
        id_ = compute_operation_id(
            request_identity=_IDENTITY,
            origin_catalogue_build_id=_BUILD,
            options=_OPTIONS,
        )

        def _concurrent_insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO datasets.operation "
                "(id, request_identity, options, coverage_stamp, state, estimate_rows, "
                "estimate_requests, rows_written, attempt_count, created_at, updated_at, "
                "submitted_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?, 0, 0, ?, ?, ?)",
                (
                    id_,
                    json.dumps(dict(_IDENTITY)),
                    json.dumps(dict(_OPTIONS)),
                    _STAMP,
                    _ESTIMATE.rows,
                    _ESTIMATE.requests,
                    now.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            fake.before_backlog_lock = None

        fake.before_backlog_lock = _concurrent_insert

        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        assert op.id == id_
        assert op.state == "pending"
    finally:
        fake.close()


def test_create_or_get_reset_refuses_at_the_backlog_max() -> None:
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(
            store=fake, now=lambda: datetime(2026, 1, 1, tzinfo=UTC), backlog_max=1
        )
        op = store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

        # A second pending operation fills the one-slot backlog.
        store.create_or_get(
            request_identity={"input": {"spec_id": "spec_b"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )

        with pytest.raises(OperationBacklogFull):
            store.create_or_get(
                request_identity={"input": {"spec_id": "spec_a"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
                rejected_version=None,
            )
    finally:
        fake.close()


def test_record_estimate_writes_the_estimate_for_the_owning_attempt_only() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _reclaimed(now_box, store)
        fresh = Estimate(rows=7, requests=2)

        assert store.record_estimate(id_, attempt=1, estimate=fresh) is None
        assert store.get(id_).estimate == _ESTIMATE

        recorded = store.record_estimate(id_, attempt=2, estimate=fresh)
        assert recorded is not None
        assert recorded.estimate == fresh
        assert recorded.coverage_stamp == _STAMP
    finally:
        fake.close()


def test_mark_dispatched_records_once_per_submission() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        assert op.dispatched_at is None

        first = store.mark_dispatched(op.id, submission_count=0)
        assert first is not None and first.dispatched_at == now_box["t"]
        now_box["t"] += timedelta(seconds=5)
        assert store.mark_dispatched(op.id, submission_count=0) is None
        assert store.get(op.id).dispatched_at == first.dispatched_at

        assert store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
        assert store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)
        reset = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        assert (reset.submission_count, reset.dispatched_at) == (1, None)
        # A late record from submission 0 leaves the new submission undispatched.
        assert store.mark_dispatched(op.id, submission_count=0) is None
        assert store.mark_dispatched(op.id, submission_count=1) is not None
    finally:
        fake.close()


@pytest.mark.parametrize(
    ("final_delivery", "elapsed", "abandoned"),
    [
        pytest.param(True, 60, False, id="final-live-lease"),
        pytest.param(True, 61, True, id="final-lease-ended"),
        pytest.param(False, 61, False, id="delivery-due-lease-ended"),
        pytest.param(False, 600, False, id="delivery-due-at-the-deadline"),
        pytest.param(False, 601, True, id="delivery-due-past-the-deadline"),
    ],
)
def test_abandon_fails_a_running_row_only_once_nothing_will_finish_it(
    final_delivery: bool, elapsed: int, abandoned: bool
) -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"], deadline_seconds=600)
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
            rejected_version=None,
        )
        assert store.abandon(op.id, attempt=0) is None
        assert store.claim(
            op.id, submission_count=0, lease_seconds=60, final_delivery=final_delivery
        )
        now_box["t"] += timedelta(seconds=elapsed)

        # `Operation.abandoned` and the SQL predicate decide alike.
        read = store.get(op.id)
        assert read is not None
        assert read.abandoned(now_box["t"], deadline=timedelta(seconds=600)) is abandoned
        assert store.abandon(op.id, attempt=0) is None
        result = store.abandon(op.id, attempt=1)

        if abandoned:
            assert result is not None
            assert (result.state, result.refusal_code, result.retryable) == (
                "failed",
                "internal_error",
                True,
            )
        else:
            assert result is None
            assert store.get(op.id).state == "running"
    finally:
        fake.close()


@pytest.mark.parametrize("final_delivery", [True, False], ids=["final", "delivery-due"])
@pytest.mark.parametrize(("elapsed", "reset"), [(60, False), (61, True)])
def test_create_or_get_resets_a_running_row_only_past_its_lease(
    elapsed: int, reset: bool, final_delivery: bool
) -> None:
    """A resubmit recovers a lease-ended run without waiting for a delivery still due. That
    row counts in the backlog, and its own reset takes no second slot."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"], backlog_max=1)
        submission = {
            "request_identity": _IDENTITY,
            "options": _OPTIONS,
            "coverage_stamp": _STAMP,
            "origin_catalogue_build_id": _BUILD,
            "estimate": _ESTIMATE,
            "rejected_version": None,
        }
        op = store.create_or_get(**submission)
        assert store.claim(
            op.id, submission_count=0, lease_seconds=60, final_delivery=final_delivery
        )
        now_box["t"] += timedelta(seconds=elapsed)

        resubmitted = store.create_or_get(**submission)

        expected = (
            ("pending", 1, None, None)
            if reset
            else ("running", 0, now_box["t"] - timedelta(seconds=elapsed), final_delivery)
        )
        assert (
            resubmitted.state,
            resubmitted.submission_count,
            resubmitted.claimed_at,
            resubmitted.final_delivery,
        ) == expected
        assert resubmitted.attempt_count == 1
    finally:
        fake.close()


def test_delete_terminal_removes_only_done_and_failed_rows_past_the_window() -> None:
    now = datetime(2026, 3, 1, tzinfo=UTC)
    window = timedelta(days=30)
    fake = FakeOperationDatasetStore()
    try:
        rows = [
            (f"op_{state}_{age}", state, now - age)
            for state in ("pending", "running", "done", "failed")
            for age in (window - timedelta(seconds=1), window, window + timedelta(seconds=1))
        ]
        with fake.transaction() as tx:
            for id, state, updated_at in rows:
                tx.execute(
                    "INSERT INTO datasets.operation (id, request_identity, options, "
                    "coverage_stamp, state, estimate_rows, estimate_requests, claimed_at, "
                    "lease_seconds, created_at, updated_at, submitted_at) "
                    "VALUES (%s, '{}', '{}', %s, %s, 1, 1, %s, 60, %s, %s, %s)",
                    (id, _STAMP, state, updated_at, updated_at, updated_at, updated_at),
                )
        store = LiveOperationStore(store=fake, now=lambda: now)

        assert (
            delete_terminal_operations(store=fake, older_than=window, now=lambda: now, batch_size=1)
            == 2
        )
        assert {id for id, _, _ in rows if store.get(id) is None} == {
            f"op_{state}_{window + timedelta(seconds=1)}" for state in ("done", "failed")
        }
    finally:
        fake.close()
