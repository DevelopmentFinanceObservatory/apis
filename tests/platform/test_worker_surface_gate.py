"""Checks that `DFO_WORKER_SURFACE` mounts only the internal run route.
The tests inspect route declarations without entering either app's lifespan.
"""

from __future__ import annotations

import pytest

import dfo_apis.app as dfo_apis_app


@pytest.fixture(autouse=True)
def _clear_surface_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "DFO_DATASETS_SURFACE",
        "DFO_GRAPH_SURFACE",
        "DFO_PRESENTATION_SURFACE",
        "DFO_SKILLS_SURFACE",
        "DFO_WORKER_SURFACE",
    ):
        monkeypatch.delenv(var, raising=False)


def _route_paths(app) -> set[str]:
    """Resolve lazily mounted routes through OpenAPI and return their paths."""
    return set(app.openapi().get("paths", {}))


def test_worker_surface_mounts_only_the_run_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DFO_WORKER_SURFACE", "1")

    app = dfo_apis_app.create_app()

    paths = _route_paths(app)
    assert "/internal/datasets/operations/{operation_id}/run" in paths
    # No /api/* route of any kind -- no Search, no public Datasets surface.
    assert not any(path.startswith("/api/") for path in paths)


@pytest.mark.parametrize("value", ["true", "TRUE", " 1 "])
def test_worker_surface_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DFO_WORKER_SURFACE", value)

    app = dfo_apis_app.create_app()

    assert "/internal/datasets/operations/{operation_id}/run" in _route_paths(app)


def test_api_app_never_mounts_internal_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the worker flag unset, Search mounts and internal routes stay absent."""
    from search.engine.app import search_surface
    from tests.engine._fake_sdmx import FakeSdmxClient
    from tests.fixtures import FakeGraph, FakeLLM

    def _fake_search_surface(**kwargs: object):
        return search_surface(
            graph=FakeGraph(), llm=FakeLLM(), sdmx=FakeSdmxClient(observations={}, availability={})
        )

    monkeypatch.setattr(dfo_apis_app, "search_surface", _fake_search_surface)

    app = dfo_apis_app.create_app()

    paths = _route_paths(app)
    assert not any(path.startswith("/internal/") for path in paths)
    assert any(path.startswith("/api/search") for path in paths)


def test_worker_surface_falsy_value_composes_the_ordinary_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from search.engine.app import search_surface
    from tests.engine._fake_sdmx import FakeSdmxClient
    from tests.fixtures import FakeGraph, FakeLLM

    def _fake_search_surface(**kwargs: object):
        return search_surface(
            graph=FakeGraph(), llm=FakeLLM(), sdmx=FakeSdmxClient(observations={}, availability={})
        )

    monkeypatch.setattr(dfo_apis_app, "search_surface", _fake_search_surface)
    monkeypatch.setenv("DFO_WORKER_SURFACE", "0")

    app = dfo_apis_app.create_app()

    paths = _route_paths(app)
    assert "/internal/datasets/operations/{operation_id}/run" not in paths
    assert any(path.startswith("/api/search") for path in paths)
