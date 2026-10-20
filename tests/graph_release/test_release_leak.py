"""Pins that a swapped-out graph release is freed once the requests that bound it end.

A surface that kept the old `Release` (or anything holding it) past its request would keep
its mapped parts resident after every import, which ends in an out-of-memory kill. Each
surface is built through the process's release service, as in production, and answers its
health route with no lifespan running.

With `GRAPH_RELEASE_COVERAGE` on, the release carries its views, the surfaces read the code index
from them as they are built, and the datasets and graph surfaces answer their capabilities
routes, which read it per request. A per-index memo or a construction-time read that kept the
old release's code index would keep it alive after the swap.
"""

from __future__ import annotations

import gc
import weakref
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
    old_index = weakref.ref(served.views.code_index) if served.views is not None else None
    del served
    monkeypatch.setenv("GRAPH_RELEASE_URI", root)
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1" if views else "0")

    with service_module._singleton.poisoned_for_test(service):
        surfaces = [datasets_surface(), graph_surface(), presentation_surface(), search_surface()]
        for surface in surfaces:
            _request(surface, release_id=first, views=views)

        second = _publish(root, views=views, revision="second")
        clock.now += _RECHECK
        service.snapshot()
        service.close()
        for surface in surfaces:
            _request(surface, release_id=second, views=views)

        gc.collect()
        assert old() is None
        assert old_index is None or old_index() is None
