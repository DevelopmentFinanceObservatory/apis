"""Tests `submit_head_check`, the freshness check a GET past the recheck window submits as an
ordinary operation, and the `x-dfo-freshness` it answers. The operation store is the real
`LiveOperationStore` over SQLite.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import override

import pytest

from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import (
    LiveOperationStore,
    NullOperationStore,
    Operation,
    OperationStore,
)
from datasets_api.operations.head_check import HeadFreshness, submit_head_check
from datasets_api.operations.task_client import FakeTaskClient, TaskClient
from datasets_api.store_errors import DatasetsStoreUnavailable
from materialise.trace import UNKNOWN_FAMILY
from series_coverage.index import Estimate

from ._operation_store import FakeOperationDatasetStore
from ._version_record_store import TEST_BASIS

_SPEC = "spec_head_check"
_BACKLOG_MAX = 20
_OPTIONS = MintOptions(format="csv", bundle=False)


class _RaisingTaskClient:
    def __init__(self) -> None:
        self.calls = 0

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        self.calls += 1
        raise RuntimeError("cloud tasks unavailable")


class _UnavailableOperationStore(NullOperationStore):
    @override
    def create_or_get(
        self,
        *,
        request_identity: Mapping[str, object],
        options: Mapping[str, object],
        coverage_stamp: str,
        origin_catalogue_build_id: str,
        estimate: Estimate,
        headroom: int = 0,
    ) -> Operation:
        raise DatasetsStoreUnavailable("creating operation failed: connection refused")


def _check(
    operation_store: OperationStore,
    task_client: TaskClient,
    *,
    backlog_max: int = _BACKLOG_MAX,
) -> HeadFreshness:
    return submit_head_check(
        subject_id=_SPEC,
        options=_OPTIONS,
        coverage_stamp="cov_stamp_1",
        operation_store=operation_store,
        task_client=task_client,
        backlog_max=backlog_max,
    )


def _store(*, backlog_max: int = _BACKLOG_MAX) -> LiveOperationStore:
    return LiveOperationStore(store=FakeOperationDatasetStore(), backlog_max=backlog_max)


def _other_operation(operation_store: LiveOperationStore, n: int) -> None:
    operation_store.create_or_get(
        request_identity={"subject_id": f"spec_other_{n}", "member_dcids": ["sv_a"]},
        options=MintOptions().model_dump(mode="json"),
        coverage_stamp="cov_stamp_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=10, requests=1),
    )


def test_stale_reads_share_one_check_with_one_task() -> None:
    operation_store = _store()
    task_client = FakeTaskClient()

    answers = [_check(operation_store, task_client), _check(operation_store, task_client)]

    assert answers == ["pending", "pending"]
    ((operation_id, submission),) = task_client.enqueued
    assert submission == 0
    stored = operation_store.get(operation_id)
    assert stored is not None
    # The marker keeps the check apart from an over-budget submission of the same rendering,
    # and with no members the worker has none to compare.
    assert stored.request_identity == {"subject_id": _SPEC, "head_check": True}
    assert stored.options == _OPTIONS.model_dump(mode="json")
    assert (stored.estimate.rows, stored.estimate.requests) == (0, 0)
    assert stored.dispatched_at is not None


def test_a_check_done_runs_again_under_a_new_submission() -> None:
    """A check that confirmed the head leaves its row done, and the next stale read resets it
    under a new submission count, so its task takes a new name."""
    operation_store = _store()
    task_client = FakeTaskClient()
    _check(operation_store, task_client)
    ((operation_id, _),) = task_client.enqueued
    operation_store.claim(operation_id, submission_count=0, lease_seconds=60, final_delivery=False)
    operation_store.complete(
        operation_id,
        attempt=1,
        subject_id=_SPEC,
        content_version="cv_1",
        confirmed_at=datetime(2026, 11, 20, tzinfo=UTC),
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
    )

    assert _check(operation_store, task_client) == "pending"
    assert task_client.enqueued == [(operation_id, 0), (operation_id, 1)]


def test_a_failed_task_create_answers_stale_and_the_next_read_creates_it() -> None:
    """Nobody polls a check, so each stale read is what retries its lost task create."""
    operation_store = _store()
    raising = _RaisingTaskClient()

    assert _check(operation_store, raising) == "stale"
    assert raising.calls == 1

    task_client = FakeTaskClient()
    assert _check(operation_store, task_client) == "pending"
    ((operation_id, submission),) = task_client.enqueued
    assert submission == 0
    stored = operation_store.get(operation_id)
    assert stored is not None and stored.dispatched_at is not None


def test_a_check_leaves_half_the_backlog_to_callers_who_wait() -> None:
    """With half the backlog live a check is refused and the read answers `stale`, while an
    over-budget submission still finds a place."""
    operation_store = _store(backlog_max=4)
    for n in range(2):
        _other_operation(operation_store, n)
    task_client = FakeTaskClient()

    assert _check(operation_store, task_client, backlog_max=4) == "stale"
    assert task_client.enqueued == []
    _other_operation(operation_store, 2)


def _unavailable() -> OperationStore:
    return _UnavailableOperationStore()


def _failed_for_good() -> OperationStore:
    operation_store = _store()
    task_client = FakeTaskClient()
    _check(operation_store, task_client)
    ((operation_id, _),) = task_client.enqueued
    operation_store.claim(operation_id, submission_count=0, lease_seconds=60, final_delivery=False)
    operation_store.fail(operation_id, attempt=1, refusal_code="over_ceiling", retryable=False)
    return operation_store


@pytest.mark.parametrize("arrange", [_unavailable, _failed_for_good])
def test_a_check_that_cannot_run_answers_stale(arrange: Callable[[], OperationStore]) -> None:
    operation_store = arrange()
    task_client = FakeTaskClient()

    assert _check(operation_store, task_client) == "stale"
    assert task_client.enqueued == []
