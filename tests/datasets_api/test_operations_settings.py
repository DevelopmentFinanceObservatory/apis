"""`TaskClientSettings` and `WorkerSettings` reject missing and empty `DFO_*` values."""

from __future__ import annotations

import pytest

from datasets_api.config import DatasetsConfigError
from datasets_api.operations.task_client import TaskClientSettings
from datasets_api.operations.worker_app import WorkerSettings

_TASK_CLIENT_ENV = {
    "DFO_WORKER_URL": "https://worker.example.org",
    "DFO_WORKER_INVOKER_SERVICE_ACCOUNT": "worker@example.iam.gserviceaccount.com",
    "DFO_TASKS_QUEUE": "projects/p/locations/l/queues/q",
}

_WORKER_ENV = {
    "DFO_WORKER_URL": _TASK_CLIENT_ENV["DFO_WORKER_URL"],
    "DFO_WORKER_INVOKER_SERVICE_ACCOUNT": _TASK_CLIENT_ENV["DFO_WORKER_INVOKER_SERVICE_ACCOUNT"],
}


@pytest.mark.parametrize("empty_var", sorted(_TASK_CLIENT_ENV))
def test_task_client_settings_empty_var_fails_fast(
    empty_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var, value in _TASK_CLIENT_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv(empty_var, "")

    with pytest.raises(DatasetsConfigError, match=empty_var):
        TaskClientSettings.from_env()


@pytest.mark.parametrize("empty_var", sorted(_WORKER_ENV))
def test_worker_settings_empty_var_fails_fast(
    empty_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var, value in _WORKER_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv(empty_var, "")

    with pytest.raises(DatasetsConfigError, match=empty_var):
        WorkerSettings.from_env()
