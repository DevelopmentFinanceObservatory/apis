"""Enqueues dataset operations through Cloud Tasks.

Tasks call the worker route with an OIDC token for `worker_url`. Their deterministic name and
body include `operation_id` and `submission_count`. Enqueue recovery creates a task again under
the same name, so `AlreadyExists` means an earlier create succeeded and counts as success. A
retryable resubmit increments `submission_count` and gets a new name. The 10-second enqueue
timeout is a third of `datasets_api.operations.dispatch.ENQUEUE_RECOVERY_SECONDS`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

from datasets_api.config import DatasetsConfigError

logger = logging.getLogger(__name__)

_RUN_ROUTE_TEMPLATE = "{worker_url}/internal/datasets/operations/{operation_id}/run"

# Cloud Tasks retries after its 1,800-second dispatch deadline, even if the worker may finish.
# The worker claim's lease, `datasets_api.operations.worker.LEASE_SECONDS`, sits under it.
DISPATCH_DEADLINE_SECONDS = 1_800

# Enqueue timeouts have an unknown outcome. Enqueue recovery may create the same task name again.
ENQUEUE_TIMEOUT_SECONDS = 10


class TaskClientSettings(BaseSettings):
    """No-safe-default env the mint and definition doors' own `lifespan`
    reads to build a live `TaskClient`. See `datasets_api.config
    .DatasetsSettings`'s own docstring for why this stays a separate
    settings object rather than three more fields on it.
    """

    worker_url: str = Field(min_length=1, validation_alias="DFO_WORKER_URL")
    worker_invoker_service_account: str = Field(
        min_length=1, validation_alias="DFO_WORKER_INVOKER_SERVICE_ACCOUNT"
    )
    tasks_queue: str = Field(min_length=1, validation_alias="DFO_TASKS_QUEUE")

    @classmethod
    def from_env(cls) -> TaskClientSettings:
        try:
            return cls()
        except ValidationError as exc:
            raise DatasetsConfigError(str(exc)) from exc


class TaskClient(Protocol):
    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        """Enqueues a task for this operation submission.

        The worker reads request data from the store.
        """
        ...


class NullTaskClient:
    """Raises on enqueue so missing live-client configuration surfaces immediately."""

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        raise NotImplementedError(f"NullTaskClient cannot enqueue operation {operation_id!r}")


class FakeTaskClient:
    """Records enqueues and can synchronously forward each task to a test worker."""

    def __init__(self, *, on_enqueue: Callable[[str, int], None] | None = None) -> None:
        self.enqueued: list[tuple[str, int]] = []
        self._on_enqueue = on_enqueue

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        self.enqueued.append((operation_id, submission_count))
        if self._on_enqueue is not None:
            self._on_enqueue(operation_id, submission_count)


class LiveTaskClient:
    """Uses a fully qualified Cloud Tasks queue. `worker_url` is the token audience and
    `worker_invoker_service_account` mints the token."""

    def __init__(
        self,
        *,
        # Kept as `Any` so importing the fake client does not require Google Cloud Tasks.
        client: Any,
        queue: str,
        worker_url: str,
        worker_invoker_service_account: str,
    ) -> None:
        self._client = client
        self._queue = queue
        self._worker_url = worker_url.rstrip("/")
        self._invoker = worker_invoker_service_account

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        from google.api_core.exceptions import AlreadyExists

        task_name = f"{self._queue}/tasks/{operation_id}-{submission_count}"
        task = {
            "name": task_name,
            "http_request": {
                "http_method": "POST",
                "url": _RUN_ROUTE_TEMPLATE.format(
                    worker_url=self._worker_url, operation_id=operation_id
                ),
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {"operation_id": operation_id, "submission_count": submission_count}
                ).encode("utf-8"),
                "oidc_token": {
                    "service_account_email": self._invoker,
                    "audience": self._worker_url,
                },
            },
            "dispatch_deadline": {"seconds": DISPATCH_DEADLINE_SECONDS},
        }
        try:
            self._client.create_task(
                request={"parent": self._queue, "task": task}, timeout=ENQUEUE_TIMEOUT_SECONDS
            )
        except AlreadyExists:
            # An existing or recently completed task means the earlier create succeeded.
            logger.info(
                "Datasets operations: task %r already exists on queue %r; treating this "
                "enqueue as a no-op",
                task_name,
                self._queue,
            )

    def close(self) -> None:
        """Closes the client's gRPC channel through its transport, matching `__exit__`."""
        self._client.transport.close()


def build_live_task_client(*, queue: str, worker_url: str, worker_invoker_service_account: str):
    """Builds a live client, importing Google Cloud Tasks only when needed."""
    from google.cloud import tasks_v2

    return LiveTaskClient(
        client=tasks_v2.CloudTasksClient(),
        queue=queue,
        worker_url=worker_url,
        worker_invoker_service_account=worker_invoker_service_account,
    )


__all__ = [
    "DISPATCH_DEADLINE_SECONDS",
    "ENQUEUE_TIMEOUT_SECONDS",
    "FakeTaskClient",
    "LiveTaskClient",
    "NullTaskClient",
    "TaskClient",
    "TaskClientSettings",
    "build_live_task_client",
]
