"""Pins that a swapped-out graph release is freed once the requests that bound it end.

A surface that kept the old `Release` (or anything holding it) past its request would keep
its mapped parts resident after every import, which ends in an out-of-memory kill. Each
surface is built through the process's release service, as in production, and answers its
health route with no lifespan running.
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
from tests.graph_release.support import build_test_release, sample_tables

_RECHECK = 60.0


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _request(surface: Surface, *, release_id: str) -> None:
    """Sends one request through `surface` and checks its request scope binds `release_id`."""
    assert surface.request_scope is not None
    with surface.request_scope():
        assert current_release().release_id == release_id
    client = TestClient(create_app([surface]), raise_server_exceptions=False)
    assert client.get(f"{surface.prefix}/healthz").status_code == 200


def test_a_swapped_out_release_dies_after_a_request_through_every_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = (tmp_path / "releases").as_uri()
    first = build_test_release(root, tables=sample_tables())
    clock = _Clock()
    service = ReleaseService(
        uri=root, local_dir=tmp_path / "local", recheck_seconds=_RECHECK, clock=clock
    )
    old = weakref.ref(service.snapshot())
    monkeypatch.setenv("GRAPH_RELEASE_URI", root)

    with service_module._singleton.poisoned_for_test(service):
        surfaces = [datasets_surface(), graph_surface(), presentation_surface(), search_surface()]
        for surface in surfaces:
            _request(surface, release_id=first.release_id)

        second = build_test_release(root, tables=sample_tables(), dc_serving_revision="second")
        clock.now += _RECHECK
        service.snapshot()
        service.close()
        for surface in surfaces:
            _request(surface, release_id=second.release_id)

        gc.collect()
        assert old() is None
