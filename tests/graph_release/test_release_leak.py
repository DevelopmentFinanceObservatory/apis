"""Pins that a swapped-out graph release is freed once the requests that bound it end.

A surface that kept the old `Release` (or anything holding it) past its request would keep
its mapped parts resident after every import, which ends in an out-of-memory kill. Each
surface is built through the process's release service, as in production, and answers its
health route with no lifespan running.

With `GRAPH_RELEASE_COVERAGE` on, the release carries its views, the surfaces read the code index
from them as they are built, and the datasets and graph surfaces answer their capabilities
routes, which read it per request. The datasets and graph surfaces also run their lifespans
across the swap, over the coverage `get_coverage_service()` serves, and the graph surface's
capabilities read that coverage per request. A per-index memo, a construction-time read or a
lifespan that kept the old release's code index or coverage would keep it alive after the swap.
"""

from __future__ import annotations

import gc
import weakref
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import graph_release.service as service_module
from datasets_api.app import datasets_surface
from dfo_platform.app import Surface, create_app
from graph_api.app import graph_surface
from graph_release.context import current_release
from graph_release.service import ReleaseService
from presentation_api.app import presentation_surface
from search.engine.app import search_surface
from series_coverage.shared import get_coverage_service
from tests.datasets_api._artifact_doubles import FakeArtifactStore
from tests.graph_release.support import build_test_release, build_view_release, sample_tables

_RECHECK = 60.0


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# The surfaces whose capabilities route reads the code index with no lifespan running.
_READS_INDEX = frozenset({"datasets", "graph"})


def _request(surface: Surface, *, release_id: str, views: bool) -> None:
    """Sends one request through `surface` and checks its request scope binds `release_id`.
    With `views`, also reads its capabilities where they read the code index."""
    assert surface.request_scope is not None
    with surface.request_scope():
        assert current_release().release_id == release_id
    client = TestClient(create_app([surface]), raise_server_exceptions=False)
    assert client.get(f"{surface.prefix}/healthz").status_code == 200
    if views and surface.name in _READS_INDEX:
        response = client.get(f"{surface.prefix}/capabilities")
        assert response.status_code == 200, response.text


def _graph_client_beside_datasets(stack: ExitStack) -> TestClient:
    """Runs the datasets and graph surfaces' lifespans until `stack` closes, every live
    dependency injected and the coverage `get_coverage_service()` serves, and returns the graph
    surface's client."""
    unused = {
        name: object()
        for name in (
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
    }
    datasets = datasets_surface(artifact_store=FakeArtifactStore(), **unused)
    stack.enter_context(TestClient(create_app([datasets])))
    # Capabilities read neither the graph nor the resolver.
    graph = graph_surface(graph=object(), resolver=object(), coverage=get_coverage_service())
    return stack.enter_context(TestClient(create_app([graph])))


def _coverage_build(client: TestClient) -> str:
    """The coverage build the graph surface's capabilities name, read per request."""
    response = client.get("/api/graph/capabilities")
    assert response.status_code == 200, response.text
    return response.json()["data_builds"]["coverage_index"]


def _publish(root: str, *, views: bool, revision: str) -> str:
    if views:
        return build_view_release(root, dc_serving_revision=revision).release_id
    return build_test_release(root, tables=sample_tables(), dc_serving_revision=revision).release_id


# Unbound, so the surfaces read the code index at construction only under the release they bind.
@pytest.mark.unbound_release
@pytest.mark.parametrize("views", [False, True], ids=["flag-off", "flag-on"])
def test_a_swapped_out_release_dies_after_a_request_through_every_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, views: bool
):
    root = (tmp_path / "releases").as_uri()
    first = _publish(root, views=views, revision="first")
    clock = _Clock()
    service = ReleaseService(
        uri=root,
        local_dir=tmp_path / "local",
        recheck_seconds=_RECHECK,
        clock=clock,
        build_views=views,
    )
    served = service.snapshot()
    old = weakref.ref(served)
    old_views = (
        [weakref.ref(served.views.code_index), weakref.ref(served.views.coverage)]
        if served.views is not None
        else []
    )
    del served
    monkeypatch.setenv("GRAPH_RELEASE_URI", root)
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1" if views else "0")

    with service_module._singleton.poisoned_for_test(service), ExitStack() as running:
        surfaces = [datasets_surface(), graph_surface(), presentation_surface(), search_surface()]
        for surface in surfaces:
            _request(surface, release_id=first, views=views)
        graph = _graph_client_beside_datasets(running) if views else None
        assert graph is None or _coverage_build(graph) == first

        second = _publish(root, views=views, revision="second")
        clock.now += _RECHECK
        service.snapshot()
        service.close()
        for surface in surfaces:
            _request(surface, release_id=second, views=views)
        assert graph is None or _coverage_build(graph) == second

        gc.collect()
        assert old() is None
        assert [ref() for ref in old_views] == [None] * len(old_views)
