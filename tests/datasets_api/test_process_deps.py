"""I10: the Datasets surface builds one client of each kind per process, and the process's shared
facet profile cache one graph client and SDMX pair of its own.

The surface is composed with no injected dependency, as `deployed_surfaces()` composes it, over a
fake database connection, a stub storage client, a stub Cloud Tasks client and a fake graph
generation. Constructors are counted on the classes themselves, so a construction under any import
alias counts. Requests to several doors run between startup and shutdown, so a door that built a
client per request would count too.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from artifact_store import LiveArtifactStore
from artifact_store import store as artifact_store_module
from dataset_store import provider
from datasets_api import generation as generation_module
from datasets_api.app import datasets_surface
from datasets_api.operations.task_client import LiveTaskClient
from dc_sdmx import SdmxClient
from dfo_platform.app import create_app
from graph_generation.fake import FakeGeneration
from origin_catalogue.graph import UN_BASE_URL, OriginGraphClient

_SDMX_BASE = "https://sdmx.example.org"

_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": _SDMX_BASE,
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
    "DFO_WORKER_URL": "https://worker.example.org",
    "DFO_WORKER_INVOKER_SERVICE_ACCOUNT": "worker-invoker@example.iam.gserviceaccount.com",
    "DFO_TASKS_QUEUE": "projects/example/locations/us-central1/queues/datasets-worker",
}


class _FakeConnection:
    def close(self) -> None:
        pass


class _StubCloudTasksClient:
    def __init__(self) -> None:
        self.transport = SimpleNamespace(close=lambda: None)


@pytest.fixture(autouse=True)
def _reset_provider_singleton(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(provider, "_pool", None)
    monkeypatch.setattr(provider, "_connector", None)
    monkeypatch.setattr(provider, "_borrowers", 0)


def _count_constructions(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Wrap each live class's `__init__` to count constructions. `SdmxClient` counts per base
    URL, since the graph client's UN availability client addresses a different server."""
    counts: Counter[str] = Counter()

    def _counting(cls: type, key) -> None:
        real_init = cls.__init__

        def _init(self, *args, **kwargs) -> None:
            counts[key(kwargs)] += 1
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", _init)

    _counting(LiveArtifactStore, lambda _kwargs: "LiveArtifactStore")
    _counting(OriginGraphClient, lambda _kwargs: "OriginGraphClient")
    _counting(LiveTaskClient, lambda _kwargs: "LiveTaskClient")
    _counting(SdmxClient, lambda kwargs: f"SdmxClient {kwargs['base_url']}")
    return counts


def test_the_surface_builds_one_client_of_each_kind_across_a_lifespan_serving_several_doors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
    monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
    monkeypatch.setattr("dataset_store.connection.connect", lambda **_kwargs: _FakeConnection())
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(artifact_store_module.storage, "Client", lambda: object())
    monkeypatch.setattr("google.cloud.tasks_v2.CloudTasksClient", _StubCloudTasksClient)
    shared = FakeGeneration()
    monkeypatch.setattr(generation_module, "acquire_shared", lambda **_kwargs: shared)
    monkeypatch.setattr(generation_module, "release_shared", lambda _generation: None)
    counts = _count_constructions(monkeypatch)

    app = create_app([datasets_surface()])
    with TestClient(app, raise_server_exceptions=False) as client:
        # The fake connection serves no query, so store-backed doors refuse; each still runs.
        for path in (
            "/api/datasets/healthz",
            "/api/datasets/operations/op_unknown",
            "/api/datasets/spec_unknown/cv_unknown",
            "/api/datasets/spec_unknown/cv_unknown/observations",
            "/api/datasets/spec_unknown/versions",
        ):
            client.get(path)

    # The second graph client and SDMX pair are the process's shared facet profile cache's
    # (`materialise.shared_profiles`), which the surface acquires once with its own clients.
    assert counts == {
        "LiveArtifactStore": 1,
        "OriginGraphClient": 2,
        "LiveTaskClient": 1,
        f"SdmxClient {_SDMX_BASE}": 2,
        f"SdmxClient {UN_BASE_URL}": 2,
    }
