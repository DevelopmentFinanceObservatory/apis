"""Tests `LiveOperationStore` against SQLite-backed `FakeOperationDatasetStore` and the real
`0011_operation.sql` migration.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from datasets_api.config import DatasetsBuildSettings
from datasets_api.operation_store import (
    MAX_REDISPATCHES,
    REDISPATCH_GRACE_SECONDS,
    LiveOperationStore,
    OperationBacklogFull,
    classify_operation_store_exception,
    compute_operation_id,
    lease_stale_after_seconds,
    never_claimed_stale_after_seconds,
)
from dfo_platform import Alternative, ProblemIssue, ProblemSpec
from series_coverage.index import Estimate

from ._operation_store import FakeOperationDatasetStore

_IDENTITY = {"input": {"spec_id": "spec_abc"}}
_OPTIONS = {"format": "parquet"}
_STAMP = "cov_stamp_1"
_BUILD = "20260929T000000Z-abcd1234"
_ESTIMATE = Estimate(rows=2_500_000, requests=847)

# Default settings used by stores built without explicit limits.
_DEFAULT_SETTINGS = DatasetsBuildSettings()


def _lease_stale_after_seconds() -> float:
    return lease_stale_after_seconds(
        max_retry_duration_seconds=_DEFAULT_SETTINGS.operation_max_retry_duration_seconds
    )


def _never_claimed_stale_after_seconds(
    *,
    backlog_max: int = _DEFAULT_SETTINGS.operation_backlog_max,
    dispatch_concurrency: int = _DEFAULT_SETTINGS.operation_dispatch_concurrency,
) -> float:
    return never_claimed_stale_after_seconds(
        backlog_max=backlog_max,
        dispatch_concurrency=dispatch_concurrency,
        max_retry_duration_seconds=_DEFAULT_SETTINGS.operation_max_retry_duration_seconds,
    )


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
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        options=_OPTIONS,
    )
    id_b = compute_operation_id(
        request_identity=dict(_IDENTITY),
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        options=dict(_OPTIONS),
    )
    assert id_a == id_b
    assert id_a.startswith("op_")

    id_c = compute_operation_id(
        request_identity=_IDENTITY,
        coverage_stamp="cov_stamp_2",
        origin_catalogue_build_id=_BUILD,
        options=_OPTIONS,
    )
    assert id_c != id_a


def test_compute_operation_id_differs_by_rendering() -> None:
    """Different renderings of the same subject and coverage stamp get distinct ids."""
    id_parquet = compute_operation_id(
        request_identity=_IDENTITY,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        options={"format": "parquet"},
    )
    id_csv = compute_operation_id(
        request_identity=_IDENTITY,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        options={"format": "csv"},
    )

    assert id_parquet != id_csv


def test_compute_operation_id_differs_by_origin_catalogue_build() -> None:
    """A new catalogue build can admit different provenances, so it gets a distinct id."""
    ids = {
        compute_operation_id(
            request_identity=_IDENTITY,
            coverage_stamp=_STAMP,
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
    )
    claimed = store.claim_lease(first.id, submission_count=0, lease_seconds=60)
    assert claimed is not None
    assert store.complete(first.id, attempt=1, subject_id="spec_abc", content_version="cv_1")

    rebuilt = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id="20261006T000000Z-ef567890",
        estimate=_ESTIMATE,
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
    )
    csv = store.create_or_get(
        request_identity=_IDENTITY,
        options={"format": "csv"},
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
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
        )
        store.create_or_get(
            request_identity={"input": {"spec_id": "spec_b"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )

        with pytest.raises(OperationBacklogFull) as excinfo:
            store.create_or_get(
                request_identity={"input": {"spec_id": "spec_c"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
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
        )

        second = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
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
        )
        store.claim_lease(first.id, submission_count=0, lease_seconds=60)
        store.complete(first.id, attempt=1, subject_id="spec_a", content_version="cv_1")

        second = store.create_or_get(
            request_identity={"input": {"spec_id": "spec_b"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        assert second.state == "pending"
    finally:
        fake.close()


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
        )
        recording.executed.clear()

        store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )

        assert not any(
            s.strip().upper().startswith("SELECT PG_ADVISORY_XACT_LOCK") for s in recording.executed
        )
    finally:
        recording.close()


def test_claim_lease_moves_pending_to_running(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )

    claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)

    assert claimed is not None
    assert claimed.state == "running"
    assert claimed.attempt_count == 1
    assert claimed.lease_until is not None


def test_claim_lease_refuses_a_live_lease(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=3600)

    second_claim = store.claim_lease(op.id, submission_count=0, lease_seconds=3600)

    assert second_claim is None
    assert store.get(op.id).attempt_count == 1


def test_claim_lease_takes_over_an_expired_lease(store: LiveOperationStore) -> None:
    now = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    live_store = LiveOperationStore(store=FakeOperationDatasetStore(), now=lambda: now["t"])
    op = live_store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    live_store.claim_lease(op.id, submission_count=0, lease_seconds=30)

    now["t"] = now["t"] + timedelta(seconds=31)
    retaken = live_store.claim_lease(op.id, submission_count=0, lease_seconds=30)

    assert retaken is not None
    assert retaken.attempt_count == 2


def test_claim_lease_refuses_a_terminal_operation(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    store.complete(op.id, attempt=1, subject_id="spec_abc", content_version="cv_deadbeef")

    assert store.claim_lease(op.id, submission_count=0, lease_seconds=60) is None


def test_progress_only_applies_while_running(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )

    assert store.progress(op.id, attempt=1, rows_written=100) is None

    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    progressed = store.progress(op.id, attempt=1, rows_written=1_000_000)

    assert progressed is not None
    assert progressed.rows_written == 1_000_000


def test_progress_with_lease_seconds_also_renews_the_lease(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    assert claimed is not None and claimed.lease_until is not None
    original_lease_until = claimed.lease_until

    renewed = store.progress(op.id, attempt=1, rows_written=500, lease_seconds=120)

    assert renewed is not None
    assert renewed.rows_written == 500
    assert renewed.lease_until is not None
    assert renewed.lease_until > original_lease_until


def test_progress_without_lease_seconds_leaves_the_lease_untouched(
    store: LiveOperationStore,
) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    assert claimed is not None
    original_lease_until = claimed.lease_until

    progressed = store.progress(op.id, attempt=1, rows_written=500)

    assert progressed is not None
    assert progressed.lease_until == original_lease_until


def test_complete_sets_the_manifest_reference_and_is_running_only(
    store: LiveOperationStore,
) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )

    assert (
        store.complete(op.id, attempt=1, subject_id="spec_abc", content_version="cv_deadbeef")
        is None
    )

    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    done = store.complete(op.id, attempt=1, subject_id="spec_abc", content_version="cv_deadbeef")

    assert done is not None
    assert done.state == "done"
    assert done.subject_id == "spec_abc"
    assert done.content_version == "cv_deadbeef"
    assert done.lease_until is None

    assert (
        store.complete(op.id, attempt=1, subject_id="spec_abc", content_version="cv_deadbeef")
        is None
    )


def test_fail_sets_the_refusal_code_and_is_running_only(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )

    assert store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False) is None

    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    failed = store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False)

    assert failed is not None
    assert failed.state == "failed"
    assert failed.refusal_code == "excel_limit"
    assert failed.retryable is False
    assert failed.lease_until is None


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
    )
    assert op.refusal is None
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)

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
        )
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
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
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)

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
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    store.fail(op.id, attempt=1, refusal_code="upstream_error", retryable=True, refusal=_REFUSAL)

    resubmitted = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
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
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)

    failed = store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

    assert failed is not None
    assert failed.retryable is True


def test_release_moves_running_back_to_pending_and_sets_the_lease_to_now(
    store: LiveOperationStore,
) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )

    assert store.release(op.id, attempt=1) is None

    claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    assert claimed is not None and claimed.attempt_count == 1

    released = store.release(op.id, attempt=1)

    assert released is not None
    assert released.state == "pending"
    # `lease_until=now` marks the row as claimed for dead-worker detection.
    assert released.lease_until is not None
    # Release preserves the attempt count.
    assert released.attempt_count == 1

    # `lease_until <= now` makes the released row immediately claimable.
    reclaimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    assert reclaimed is not None
    assert reclaimed.attempt_count == 2


def test_release_is_a_no_op_once_terminal(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    store.complete(op.id, attempt=1, subject_id="spec_abc", content_version="cv_deadbeef")

    assert store.release(op.id, attempt=1) is None
    assert store.get(op.id).state == "done"


def _dead_store(
    now_box: dict[str, datetime], fake: FakeOperationDatasetStore
) -> LiveOperationStore:
    return LiveOperationStore(
        store=fake, now=lambda: now_box["t"], backlog_max=20, dispatch_concurrency=4
    )


def test_expire_if_dead_fails_a_running_row_whose_lease_is_past_the_stale_window() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = _dead_store(now_box, fake)
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        assert claimed is not None and claimed.lease_until is not None

        now_box["t"] = claimed.lease_until + timedelta(seconds=_lease_stale_after_seconds() - 1)
        assert store.expire_if_dead(op.id) is None
        assert store.get(op.id).state == "running"

        now_box["t"] = claimed.lease_until + timedelta(seconds=_lease_stale_after_seconds() + 1)
        expired = store.expire_if_dead(op.id)

        assert expired is not None
        assert expired.state == "failed"
        assert expired.refusal_code == "internal_error"
        assert expired.retryable is True
        assert expired.lease_until is None
    finally:
        fake.close()


def test_expire_if_dead_judges_an_undispatched_never_claimed_row_by_its_submission() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = _dead_store(now_box, fake)
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        assert op.submitted_at == op.created_at
        threshold = _never_claimed_stale_after_seconds()

        # Past the claimed-row window, well short of the never-claimed one.
        now_box["t"] = op.created_at + timedelta(seconds=_lease_stale_after_seconds() + 1)
        assert store.expire_if_dead(op.id) is None

        now_box["t"] = op.created_at + timedelta(seconds=threshold + 1)
        expired = store.expire_if_dead(op.id)
        assert expired is not None
        assert expired.state == "failed"
    finally:
        fake.close()


def test_expire_if_dead_measures_a_never_claimed_row_from_its_last_dispatch() -> None:
    """The never-claimed window starts at dispatch, allowing queue wait time."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = _dead_store(now_box, fake)
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        threshold = timedelta(seconds=_never_claimed_stale_after_seconds())
        now_box["t"] = op.submitted_at + timedelta(seconds=600)
        dispatched = store.claim_dispatch(op.id)
        assert dispatched is not None and dispatched.dispatched_at is not None
        store.confirm_dispatch(op.id, submission_count=0, dispatch_count=dispatched.dispatch_count)

        now_box["t"] = op.submitted_at + threshold + timedelta(seconds=1)
        assert store.expire_if_dead(op.id) is None

        now_box["t"] = dispatched.dispatched_at + threshold + timedelta(seconds=1)
        expired = store.expire_if_dead(op.id)
        assert expired is not None
        assert expired.state == "failed"
    finally:
        fake.close()


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
    }
    op = store.create_or_get(**submission)
    assert store.fail_unservable_done(op.id, attempt=0) is None
    assert store.claim_lease(op.id, submission_count=0, lease_seconds=60) is not None
    assert store.fail_unservable_done(op.id, attempt=1) is None
    running = store.get(op.id)
    assert running is not None and running.state == "running"
    assert store.complete(op.id, attempt=1, subject_id="spec_abc", content_version="cv_1")

    first = store.fail_unservable_done(op.id, attempt=1)

    assert first is not None
    assert (first.state, first.refusal_code, first.retryable) == ("failed", "internal_error", True)

    # The resubmit re-runs it under a higher attempt and completes with the same version.
    resubmitted = store.create_or_get(**submission)
    assert (resubmitted.state, resubmitted.submission_count) == ("pending", 1)
    rerun = store.claim_lease(op.id, submission_count=1, lease_seconds=60)
    assert rerun is not None and rerun.attempt_count == 2
    assert store.complete(op.id, attempt=2, subject_id="spec_abc", content_version="cv_1")

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


def test_a_resubmit_of_an_old_operation_survives_polls_and_sweeps_inside_its_new_window() -> None:
    """A retryable resubmission gets a fresh staleness window from its dispatch."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = _dead_store(now_box, fake)
        threshold = timedelta(seconds=_never_claimed_stale_after_seconds())
        first = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        assert store.claim_lease(first.id, submission_count=0, lease_seconds=60) is not None
        assert store.fail(first.id, attempt=1, refusal_code="internal_error", retryable=True)

        now_box["t"] = first.created_at + threshold + timedelta(hours=1)
        reset = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        assert reset.state == "pending"
        assert reset.submission_count == 1
        assert reset.created_at == first.created_at
        assert reset.submitted_at == now_box["t"]
        dispatched = store.claim_dispatch(reset.id)
        assert dispatched is not None and dispatched.dispatched_at is not None
        store.confirm_dispatch(
            reset.id, submission_count=1, dispatch_count=dispatched.dispatch_count
        )

        now_box["t"] = now_box["t"] + timedelta(seconds=2)
        assert store.expire_if_dead(reset.id) is None
        store.create_or_get(
            request_identity={"input": {"spec_id": "spec_other"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        pending = store.get(reset.id)
        assert pending is not None
        assert pending.state == "pending"

        now_box["t"] = dispatched.dispatched_at + threshold + timedelta(seconds=1)
        expired = store.expire_if_dead(reset.id)
        assert expired is not None
        assert expired.state == "failed"
    finally:
        fake.close()


def test_expire_if_dead_judges_a_released_row_by_its_lease_not_created_at() -> None:
    """A released row uses its lease timestamp for expiry, even while pending."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = _dead_store(now_box, fake)
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        released = store.release(op.id, attempt=1)
        assert released is not None and released.lease_until is not None

        now_box["t"] = released.lease_until + timedelta(seconds=_lease_stale_after_seconds() + 1)
        expired = store.expire_if_dead(op.id)

        assert expired is not None
        assert expired.state == "failed"
    finally:
        fake.close()


def test_expire_if_dead_refuses_a_freshly_renewed_lease() -> None:
    """Expiry rechecks the lease after a concurrent heartbeat renews it."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = _dead_store(now_box, fake)
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        assert claimed is not None and claimed.lease_until is not None
        now_box["t"] = claimed.lease_until + timedelta(seconds=_lease_stale_after_seconds() + 1)
        assert store.get(op.id).lease_until == claimed.lease_until

        renewed = store.progress(op.id, attempt=1, rows_written=10, lease_seconds=60)
        assert renewed is not None

        assert store.expire_if_dead(op.id) is None
        assert store.get(op.id).state == "running"
    finally:
        fake.close()


def test_expire_if_dead_is_a_no_op_once_terminal(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False)

    # Preserve the worker's refusal code.
    assert store.expire_if_dead(op.id) is None
    assert store.get(op.id).refusal_code == "excel_limit"


def test_expire_if_dead_returns_none_for_an_unknown_id(store: LiveOperationStore) -> None:
    assert store.expire_if_dead("op_does_not_exist") is None


def test_sweep_leaves_a_completion_that_landed_after_the_lease_went_stale() -> None:
    """The sweep preserves completion that lands after the lease goes stale."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = _dead_store(now_box, fake)
        op = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        assert claimed is not None and claimed.lease_until is not None
        now_box["t"] = claimed.lease_until + timedelta(seconds=_lease_stale_after_seconds() + 1)

        done = store.complete(op.id, attempt=1, subject_id="spec_abc", content_version="cv_1")
        assert done is not None

        store.create_or_get(
            request_identity={"input": {"spec_id": "spec_other"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )

        after = store.get(op.id)
        assert after is not None
        assert after.state == "done"
        assert after.content_version == "cv_1"
    finally:
        fake.close()


def _reclaimed(now_box: dict[str, datetime], store: LiveOperationStore) -> str:
    """Creates an operation reclaimed for attempt 2 and returns its id."""
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    first = store.claim_lease(op.id, submission_count=0, lease_seconds=30)
    assert first is not None and first.attempt_count == 1
    now_box["t"] = now_box["t"] + timedelta(seconds=31)
    second = store.claim_lease(op.id, submission_count=0, lease_seconds=30)
    assert second is not None and second.attempt_count == 2
    return op.id


def test_stale_attempt_progress_is_refused_after_a_reclaim() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _reclaimed(now_box, store)
        lease_before = store.get(id_).lease_until

        assert store.progress(id_, attempt=1, rows_written=99, lease_seconds=3_600) is None

        current = store.get(id_)
        assert current.rows_written == 0
        assert current.lease_until == lease_before
        assert store.progress(id_, attempt=2, rows_written=5) is not None
    finally:
        fake.close()


def test_stale_attempt_complete_is_refused_after_a_reclaim() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _reclaimed(now_box, store)

        assert store.complete(id_, attempt=1, subject_id="spec_abc", content_version="cv_1") is None
        assert store.get(id_).state == "running"

        done = store.complete(id_, attempt=2, subject_id="spec_abc", content_version="cv_2")
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


def test_claim_lease_refuses_a_delivery_from_an_earlier_submission(
    store: LiveOperationStore,
) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)
    reset = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    assert reset.submission_count == 1

    assert store.claim_lease(op.id, submission_count=0, lease_seconds=60) is None
    assert store.get(op.id).state == "pending"

    claimed = store.claim_lease(op.id, submission_count=1, lease_seconds=60)
    assert claimed is not None
    # Attempt counts remain unique across submissions.
    assert claimed.attempt_count == 2


def test_claim_dispatch_is_won_once_per_dispatch(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    assert op.dispatch_count == 0

    won = store.claim_dispatch(op.id)
    assert won is not None
    assert won.dispatch_count == 1
    assert won.dispatched_at is not None
    assert won.dispatch_confirmed == 0

    assert store.claim_dispatch(op.id) is None


def test_claim_dispatch_redispatches_an_unconfirmed_dispatch_after_the_grace() -> None:
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
        )
        assert store.claim_dispatch(op.id) is not None

        now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS - 1)
        assert store.claim_dispatch(op.id) is None

        now_box["t"] = now_box["t"] + timedelta(seconds=2)
        again = store.claim_dispatch(op.id)
        assert again is not None
        assert again.dispatch_count == 2
    finally:
        fake.close()


def test_claim_dispatch_never_redispatches_a_confirmed_dispatch() -> None:
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
        )
        won = store.claim_dispatch(op.id)
        assert won is not None
        store.confirm_dispatch(op.id, submission_count=0, dispatch_count=won.dispatch_count)
        assert store.get(op.id).dispatch_confirmed == 1

        # Cloud Tasks may queue a confirmed task for hours under a full backlog.
        now_box["t"] = now_box["t"] + timedelta(hours=3)
        assert store.claim_dispatch(op.id) is None
    finally:
        fake.close()


def test_claim_dispatch_never_redispatches_a_claimed_or_released_row() -> None:
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
        )
        assert store.claim_dispatch(op.id) is not None
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
        assert store.claim_dispatch(op.id) is None

        released = store.release(op.id, attempt=1)
        assert released is not None and released.state == "pending"
        now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
        assert store.claim_dispatch(op.id) is None
    finally:
        fake.close()


def test_claim_dispatch_stops_at_the_redispatch_cap() -> None:
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
        )
        counts = []
        for _ in range(MAX_REDISPATCHES + 2):
            won = store.claim_dispatch(op.id)
            if won is not None:
                counts.append(won.dispatch_count)
            now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)

        assert counts == list(range(1, MAX_REDISPATCHES + 2))
    finally:
        fake.close()


def test_reset_starts_a_fresh_dispatch_and_confirm_is_scoped_to_its_submission() -> None:
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
        )
        store.claim_dispatch(op.id)
        store.confirm_dispatch(op.id, submission_count=0, dispatch_count=1)
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

        reset = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        assert reset.submission_count == 1
        assert reset.dispatch_count == 0
        assert reset.dispatch_confirmed == 0
        assert reset.dispatched_at is None

        won = store.claim_dispatch(op.id)
        assert won is not None and won.dispatch_count == 1

        # Ignore a late confirmation from submission 0.
        store.confirm_dispatch(op.id, submission_count=0, dispatch_count=1)
        assert store.get(op.id).dispatch_confirmed == 0
        store.confirm_dispatch(op.id, submission_count=1, dispatch_count=1)
        assert store.get(op.id).dispatch_confirmed == 1
    finally:
        fake.close()


def test_get_returns_none_for_an_unknown_id(store: LiveOperationStore) -> None:
    assert store.get("op_does_not_exist") is None


def test_get_returns_a_terminal_operation(store: LiveOperationStore) -> None:
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    store.claim_lease(op.id, submission_count=0, lease_seconds=60)
    store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False)

    read_back = store.get(op.id)

    assert read_back is not None
    assert read_back.state == "failed"


def test_never_claimed_stale_after_seconds_matches_the_documented_formula() -> None:
    """Adds retry time to `ceil(backlog_max / dispatch_concurrency)` dispatch deadlines."""
    seconds = never_claimed_stale_after_seconds(
        backlog_max=20,
        dispatch_concurrency=4,
        dispatch_deadline_seconds=1_800,
        max_retry_duration_seconds=3_600,
    )
    assert seconds == 5 * 1_800 + 3_600

    # 21 items at concurrency 4 require 6 dispatch rounds.
    seconds_uneven = never_claimed_stale_after_seconds(
        backlog_max=21,
        dispatch_concurrency=4,
        dispatch_deadline_seconds=1_800,
        max_retry_duration_seconds=3_600,
    )
    assert seconds_uneven == 6 * 1_800 + 3_600


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
        )
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        store.fail(op.id, attempt=1, refusal_code="excel_limit", retryable=False)

        resubmitted = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
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
        )
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

        resubmitted = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )

        assert resubmitted.id == op.id
        assert resubmitted.state == "pending"
        assert resubmitted.refusal_code is None
        assert resubmitted.retryable is None
        assert resubmitted.rows_written == 0
        assert resubmitted.lease_until is None
        # `submission_count` changes; the operation's hashed identity stays fixed.
        assert resubmitted.submission_count == 1
        assert resubmitted.estimate == _ESTIMATE

        store.claim_lease(resubmitted.id, submission_count=1, lease_seconds=60)
        store.fail(resubmitted.id, attempt=2, refusal_code="internal_error", retryable=True)
        twice_resubmitted = store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        assert twice_resubmitted.submission_count == 2
    finally:
        fake.close()


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
        )
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)
        recording.executed.clear()

        store.create_or_get(
            request_identity=_IDENTITY,
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
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
        )
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

        def _concurrent_reset(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE datasets.operation SET state = 'pending', lease_until = NULL, "
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
            coverage_stamp=_STAMP,
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
        )
        store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        store.fail(op.id, attempt=1, refusal_code="internal_error", retryable=True)

        # A second pending operation fills the one-slot backlog.
        store.create_or_get(
            request_identity={"input": {"spec_id": "spec_b"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )

        with pytest.raises(OperationBacklogFull):
            store.create_or_get(
                request_identity={"input": {"spec_id": "spec_a"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
            )
    finally:
        fake.close()


def test_create_or_get_sweeps_stale_never_claimed_rows_before_counting_the_backlog() -> None:
    """The pre-count sweep expires 20 stale, never-claimed rows in a committed transaction."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(
            store=fake,
            now=lambda: now_box["t"],
            backlog_max=20,
            dispatch_concurrency=4,
        )
        stale_ids = []
        for i in range(20):
            op = store.create_or_get(
                request_identity={"input": {"spec_id": f"spec_stale_{i}"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
            )
            assert op.state == "pending"
            assert op.lease_until is None
            stale_ids.append(op.id)

        with pytest.raises(OperationBacklogFull):
            store.create_or_get(
                request_identity={"input": {"spec_id": "spec_new_too_soon"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
            )

        threshold = _never_claimed_stale_after_seconds()
        now_box["t"] = now_box["t"] + timedelta(seconds=threshold + 1)

        new_op = store.create_or_get(
            request_identity={"input": {"spec_id": "spec_new"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        assert new_op.state == "pending"

        for stale_id in stale_ids:
            swept = store.get(stale_id)
            assert swept is not None
            assert swept.state == "failed"
            assert swept.refusal_code == "internal_error"
            assert swept.retryable is True
    finally:
        fake.close()


def test_create_or_get_sweeps_dead_running_rows_before_counting_the_backlog() -> None:
    """The pre-count sweep expires 20 stale running rows whose workers exhausted retries."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"], backlog_max=20)
        dead_ids = []
        for i in range(20):
            op = store.create_or_get(
                request_identity={"input": {"spec_id": f"spec_dead_running_{i}"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
            )
            claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=60)
            assert claimed is not None
            dead_ids.append(op.id)

        with pytest.raises(OperationBacklogFull):
            store.create_or_get(
                request_identity={"input": {"spec_id": "spec_new_too_soon"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
            )

        # Advance past the 60-second leases and the stale window.
        now_box["t"] = now_box["t"] + timedelta(seconds=_lease_stale_after_seconds() + 61)

        new_op = store.create_or_get(
            request_identity={"input": {"spec_id": "spec_new"}},
            options=_OPTIONS,
            coverage_stamp=_STAMP,
            origin_catalogue_build_id=_BUILD,
            estimate=_ESTIMATE,
        )
        assert new_op.state == "pending"

        for dead_id in dead_ids:
            swept = store.get(dead_id)
            assert swept is not None
            assert swept.state == "failed"
            assert swept.refusal_code == "internal_error"
            assert swept.retryable is True
    finally:
        fake.close()


def test_create_or_get_dead_row_sweep_commits_even_when_the_backlog_count_still_raises() -> None:
    """The sweep commits 19 dead rows even when five live rows keep the backlog full."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        setup_store = LiveOperationStore(store=fake, now=lambda: now_box["t"], backlog_max=100)

        dead_ids = []
        for i in range(19):
            op = setup_store.create_or_get(
                request_identity={"input": {"spec_id": f"spec_dead_{i}"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
            )
            setup_store.claim_lease(op.id, submission_count=0, lease_seconds=60)
            dead_ids.append(op.id)

        for i in range(5):
            op = setup_store.create_or_get(
                request_identity={"input": {"spec_id": f"spec_active_{i}"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
            )
            setup_store.claim_lease(op.id, submission_count=0, lease_seconds=1_000_000)

        # Pass the stale window for 60-second leases.
        # Keep the 1,000,000-second leases live.
        now_box["t"] = now_box["t"] + timedelta(seconds=_lease_stale_after_seconds() + 61)

        # Five live rows fill `backlog_max=5` after the sweep.
        test_store = LiveOperationStore(store=fake, now=lambda: now_box["t"], backlog_max=5)
        with pytest.raises(OperationBacklogFull) as excinfo:
            test_store.create_or_get(
                request_identity={"input": {"spec_id": "spec_trigger"}},
                options=_OPTIONS,
                coverage_stamp=_STAMP,
                origin_catalogue_build_id=_BUILD,
                estimate=_ESTIMATE,
            )
        assert excinfo.value.in_flight == 5

        # Sweep failures commit before the backlog refusal.
        for dead_id in dead_ids:
            swept = test_store.get(dead_id)
            assert swept is not None
            assert swept.state == "failed"
    finally:
        fake.close()


def test_record_execution_writes_the_stamp_and_estimate_for_the_owning_attempt_only() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _reclaimed(now_box, store)
        fresh = Estimate(rows=7, requests=2)

        assert (
            store.record_execution(id_, attempt=1, coverage_stamp="cov_new", estimate=fresh) is None
        )
        assert store.get(id_).execution_coverage_stamp is None

        recorded = store.record_execution(id_, attempt=2, coverage_stamp="cov_new", estimate=fresh)
        assert recorded is not None
        assert recorded.execution_coverage_stamp == "cov_new"
        assert recorded.estimate == fresh
        assert recorded.coverage_stamp == _STAMP
    finally:
        fake.close()


def _exhaust_dispatches(now_box: dict[str, datetime], store: LiveOperationStore) -> str:
    """Exhausts every dispatch for a pending operation and returns its id."""
    op = store.create_or_get(
        request_identity=_IDENTITY,
        options=_OPTIONS,
        coverage_stamp=_STAMP,
        origin_catalogue_build_id=_BUILD,
        estimate=_ESTIMATE,
    )
    for _ in range(MAX_REDISPATCHES + 1):
        assert store.claim_dispatch(op.id) is not None
        now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
    assert store.claim_dispatch(op.id) is None
    return op.id


def test_expire_if_dead_fails_a_submission_that_used_every_dispatch_once_its_last_is_stale() -> (
    None
):
    """An exhausted submission stays pending through the queue window after its last dispatch."""
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
        )
        for _ in range(MAX_REDISPATCHES + 1):
            assert store.claim_dispatch(op.id) is not None
            assert store.expire_if_dead(op.id) is None
            now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
        assert store.claim_dispatch(op.id) is None
        exhausted = store.get(op.id)
        assert exhausted is not None and exhausted.dispatched_at is not None
        last_dispatched_at = exhausted.dispatched_at
        threshold = timedelta(seconds=_never_claimed_stale_after_seconds())

        now_box["t"] = last_dispatched_at + threshold
        assert store.expire_if_dead(op.id) is None
        pending = store.get(op.id)
        assert pending is not None
        assert pending.state == "pending"

        now_box["t"] = last_dispatched_at + threshold + timedelta(seconds=1)
        expired = store.expire_if_dead(op.id)
        assert expired is not None
        assert expired.state == "failed"
        assert expired.refusal_code == "internal_error"
        assert expired.retryable is True
    finally:
        fake.close()


def test_the_last_dispatch_confirmed_keeps_an_exhausted_submission_alive() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    fake = FakeOperationDatasetStore()
    try:
        store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        id_ = _exhaust_dispatches(now_box, store)
        store.confirm_dispatch(id_, submission_count=0, dispatch_count=MAX_REDISPATCHES + 1)

        assert store.expire_if_dead(id_) is None
        assert store.get(id_).state == "pending"
    finally:
        fake.close()
