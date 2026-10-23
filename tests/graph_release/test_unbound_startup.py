"""Pins that every surface starts, and answers, with no graph release bound when it starts.

A production process starts its lifespans with nothing bound: a request binds its release at
entry, and startup is no request. A code index, measure index or coverage read with no release
bound raises `ReleaseNotBound`, so a lifespan that warms one up must bind the served release
around the read. These tests build the apis service's composed app,
every surface it mounts, and the datasets worker app over a release with views, run their
lifespans from an unbound context, and send one request through a route that reads the code
index. Live clients are injected or replaced, and every startup step that reads an index runs
as in production.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import graph_release.service as service_module
from code_index.query import CodeIndex
from datasets_api import generation as generation_module
from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations import worker_app as worker_module
from datasets_api.operations.oidc import FakeOidcVerifier
from datasets_api.operations.worker_app import RUN_ROUTE_PATH, build_worker_app
from dfo_platform.app import create_app
from graph_api.app import graph_surface
from graph_generation.fake import FakeGeneration
from graph_release.service import ReleaseService
from presentation_api.app import presentation_surface
from search.engine.app import search_surface
from series_coverage.shared import get_coverage_service
from skills_api.app import skills_surface
from tests.datasets_api._artifact_doubles import FakeArtifactStore
from tests.datasets_api._operation_store import FakeOperationDatasetStore
from tests.datasets_api.test_worker_app import _complete_run, _submit, _task_body
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM
from tests.graph_release.support import build_view_release
from tests.origin_catalogue.fake_catalogue import fake_catalogue_service

pytestmark = pytest.mark.unbound_release

_DATASETS_LIVE_DEPENDENCIES = (
    "registry",
    "spec_reader",
    "snapshot_reader",
    "version_writer",
    "head_writer",
    "version_lister",
    "operation_store",
    "recipe_store",
    "description_store",
    "reference_writer",
    "graph",
    "sdmx",
    "task_client",
)


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ReleaseService]:
    """The process's release service, over a published release with views, configured as on the
    apis service."""
    root = (tmp_path / "releases").as_uri()
    build_view_release(root)
    monkeypatch.setenv("GRAPH_RELEASE_URI", root)
    served = ReleaseService(uri=root, local_dir=tmp_path / "local")
    with service_module._singleton.poisoned_for_test(served):
        yield served
    served.close()


def test_the_composed_app_starts_and_reads_the_code_index_with_no_release_bound(
    service: ReleaseService,
):
    surfaces = [
        search_surface(
            graph=FakeGraph(),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            generation=FakeGeneration(),
        ),
        datasets_surface(
            artifact_store=FakeArtifactStore(),
            **{name: object() for name in _DATASETS_LIVE_DEPENDENCIES},
        ),
        # Capabilities read neither the graph nor the resolver.
        graph_surface(graph=object(), resolver=object(), coverage=get_coverage_service()),
        presentation_surface(revision_store=object(), frame_store=object()),
        skills_surface(),
    ]
    app = create_app(surfaces)

    with TestClient(app) as client:
        for prefix in ("/api/search", "/api/datasets", "/api/graph"):
            response = client.get(f"{prefix}/capabilities")
            assert response.status_code == 200, response.text
        assert client.get("/api/presentation/healthz").status_code == 200
        assert client.get("/api/skills/healthz").status_code == 200


def test_the_datasets_worker_starts_and_runs_on_the_served_code_index_with_no_release_bound(
    service: ReleaseService, monkeypatch: pytest.MonkeyPatch
):
    for var, value in {
        "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
        "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
        "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
        "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
    }.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(generation_module, "acquire_shared", lambda **_: FakeGeneration())
    monkeypatch.setattr(generation_module, "release_shared", lambda _generation: None)
    for name in ("shared_pool", "LiveDatasetStore", "LiveArtifactStore", "LiveArtifactInventory"):
        monkeypatch.setattr(worker_module, name, lambda **_: object())
    monkeypatch.setattr(worker_module, "close_shared_pool", lambda: None)
    # The run reads neither client.
    for name in ("build_graph_client", "SdmxClient"):
        monkeypatch.setattr(worker_module, name, lambda **_: SimpleNamespace(close=lambda: None))
    # A run's deps are the code index the route reads for it.
    monkeypatch.setattr(
        worker_module, "_live_deps_factory", lambda **_: lambda *, code_index: code_index
    )
    run_indexes: list[CodeIndex] = []

    def run(*, deps: CodeIndex, **kwargs: Any) -> None:
        run_indexes.append(deps)
        _complete_run(deps=deps, **kwargs)

    operation_store = LiveOperationStore(store=FakeOperationDatasetStore())
    operation_id = _submit(operation_store)
    app = build_worker_app(
        settings=DatasetsBuildSettings(),
        operation_store=operation_store,
        oidc_verifier=FakeOidcVerifier(),
        worker_url="https://worker.example.org",
        worker_invoker_service_account="invoker@example.iam.gserviceaccount.com",
        run_in_thread=run,
        catalogue=fake_catalogue_service(),
    )

    with TestClient(app) as client:
        response = client.post(
            RUN_ROUTE_PATH.format(operation_id=operation_id),
            json=_task_body(),
            headers={"Authorization": "Bearer t"},
        )

    assert response.status_code == 200, response.text
    views = service.snapshot().views
    assert views is not None
    assert run_indexes == [views.code_index]
