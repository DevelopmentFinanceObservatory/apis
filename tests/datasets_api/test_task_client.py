"""Tests task naming, deduplication, and enqueue recovery with a Cloud Tasks stub."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from google.api_core.exceptions import AlreadyExists

from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS, dispatch_if_due
from datasets_api.operations.task_client import ENQUEUE_TIMEOUT_SECONDS, LiveTaskClient
from series_coverage.index import Estimate

from ._operation_store import FakeOperationDatasetStore

_QUEUE = "projects/p/locations/l/queues/q"


class _DedupingCloudTasks:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}
        self.timeouts: list[float] = []

    def create_task(self, *, request: dict, timeout: float) -> None:
        self.timeouts.append(timeout)
        task = request["task"]
        if task["name"] in self.tasks:
            raise AlreadyExists(f"task {task['name']} already exists")
        self.tasks[task["name"]] = task


def _client(cloud_tasks: _DedupingCloudTasks) -> LiveTaskClient:
    return LiveTaskClient(
        client=cloud_tasks,
        queue=_QUEUE,
        worker_url="https://worker.example.org/",
        worker_invoker_service_account="invoker@example.iam.gserviceaccount.com",
    )


def test_a_task_is_named_per_submission_and_carries_its_submission_count() -> None:
    cloud_tasks = _DedupingCloudTasks()

    _client(cloud_tasks).enqueue(operation_id="op_abc", submission_count=2)

    task = cloud_tasks.tasks[f"{_QUEUE}/tasks/op_abc-2"]
    body = json.loads(task["http_request"]["body"])
    assert body == {"operation_id": "op_abc", "submission_count": 2}
    assert task["http_request"]["url"] == (
        "https://worker.example.org/internal/datasets/operations/op_abc/run"
    )
    assert cloud_tasks.timeouts == [ENQUEUE_TIMEOUT_SECONDS]


def test_a_recovered_create_whose_first_create_landed_dedups_to_one_task() -> None:
    """A second create after a landed timeout gets `AlreadyExists` and leaves one task."""
    cloud_tasks = _DedupingCloudTasks()
    client = _client(cloud_tasks)
    client.enqueue(operation_id="op_abc", submission_count=0)

    client.enqueue(operation_id="op_abc", submission_count=0)

    assert list(cloud_tasks.tasks) == [f"{_QUEUE}/tasks/op_abc-0"]


def test_a_new_submission_gets_a_new_task() -> None:
    cloud_tasks = _DedupingCloudTasks()
    client = _client(cloud_tasks)

    client.enqueue(operation_id="op_abc", submission_count=0)
    client.enqueue(operation_id="op_abc", submission_count=1)

    assert list(cloud_tasks.tasks) == [f"{_QUEUE}/tasks/op_abc-0", f"{_QUEUE}/tasks/op_abc-1"]


class _LandsThenTimesOut(_DedupingCloudTasks):
    """The first create reaches Cloud Tasks but the caller sees a timeout."""

    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def create_task(self, *, request: dict, timeout: float) -> None:
        super().create_task(request=request, timeout=timeout)
        if not self.failed_once:
            self.failed_once = True
            raise TimeoutError("create_task deadline exceeded")


def test_a_recovered_create_answered_already_exists_is_recorded() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = LiveOperationStore(
        store=FakeOperationDatasetStore(), now=lambda: now_box["t"]
    )
    operation = operation_store.create_or_get(
        request_identity={"subject_id": "spec_abc"},
        options={"format": "csv"},
        coverage_stamp="cov_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=200_000, requests=5),
    )
    cloud_tasks = _LandsThenTimesOut()
    client = _client(cloud_tasks)

    first = dispatch_if_due(operation, operation_store=operation_store, task_client=client)
    assert first.dispatched_at is None

    now_box["t"] = now_box["t"] + timedelta(seconds=ENQUEUE_RECOVERY_SECONDS + 1)
    again = dispatch_if_due(
        first,
        operation_store=operation_store,
        task_client=client,
        older_than=now_box["t"] - timedelta(seconds=ENQUEUE_RECOVERY_SECONDS),
    )

    assert again.dispatched_at == now_box["t"]
    assert list(cloud_tasks.tasks) == [f"{_QUEUE}/tasks/{operation.id}-0"]
