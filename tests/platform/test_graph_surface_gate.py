"""Tests the Graph surface gate, including truthy values, imports, and route mounting.

Each test clears all optional-surface flags to isolate its result from the environment.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import dfo_apis.app as dfo_apis_app
from dfo_apis.app import all_surfaces, create_app, deployed_surfaces
from search.engine.app import search_surface
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM


def _fake_search_surface(**kwargs: object):
    # Injects fake Search clients so surface-gate tests need no credentials or network.
    # Accepts other factory kwargs such as `rate_cache`.
    return search_surface(
        graph=FakeGraph(),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


@pytest.fixture(autouse=True)
def _stub_search_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dfo_apis_app, "search_surface", _fake_search_surface)


@pytest.fixture(autouse=True)
def _clear_surface_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DFO_DATASETS_SURFACE", raising=False)
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)


@pytest.mark.parametrize(
    "value",
    (
        pytest.param("1", id="one"),
        pytest.param("true", id="true"),
        pytest.param("True", id="title-true"),
        pytest.param("TRUE", id="upper-true"),
        pytest.param(" 1 ", id="padded-one"),
        pytest.param(" true ", id="padded-true"),
    ),
)
def test_truthy_value_mounts_the_graph_surface(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DFO_GRAPH_SURFACE", value)
    assert [s.name for s in deployed_surfaces()] == ["search", "graph"]


@pytest.mark.parametrize(
    "value",
    (
        pytest.param("0", id="zero"),
        pytest.param("false", id="false"),
        pytest.param("False", id="title-false"),
        pytest.param("yes", id="yes"),
        pytest.param("on", id="on"),
        pytest.param("", id="empty"),
        pytest.param(None, id="unset"),
    ),
)
def test_non_truthy_or_unset_value_omits_the_graph_surface(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    else:
        monkeypatch.setenv("DFO_GRAPH_SURFACE", value)
    assert [s.name for s in deployed_surfaces()] == ["search"]


def test_all_surfaces_always_includes_graph_regardless_of_the_flag() -> None:
    assert [s.name for s in all_surfaces()] == [
        "search",
        "datasets",
        "graph",
        "presentation",
        "skills",
    ]


def test_flag_unset_never_imports_graph_api() -> None:
    """Restore graph modules after checking the disabled surface import boundary."""
    removed = {
        name: module
        for name, module in sys.modules.items()
        if name == "graph_api" or name.startswith("graph_api.")
    }
    for name in removed:
        del sys.modules[name]

    try:
        deployed_surfaces()
        leaked = [
            name for name in sys.modules if name == "graph_api" or name.startswith("graph_api.")
        ]
        assert not leaked, f"deployed_surfaces() with the flag unset pulled in: {leaked}"
    finally:
        sys.modules.update(removed)


@pytest.mark.parametrize(
    "path",
    (
        pytest.param("/api/graph/healthz", id="healthz"),
        pytest.param("/api/graph/capabilities", id="capabilities"),
    ),
)
def test_flag_set_mounts_the_graph_route(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setenv("DFO_GRAPH_SURFACE", "1")

    # The graph client opens during lifespan, after route construction.
    app = create_app()

    assert path in app.openapi()["paths"]


def test_unset_omits_the_graph_capabilities_route():
    app = create_app()

    assert "/api/graph/capabilities" not in app.openapi()["paths"]


@pytest.mark.usefixtures("published_catalogue_uri")
def test_import_with_flag_unset_does_not_pull_in_graph_api() -> None:
    """A fresh subprocess verifies that the disabled Graph gate does not import `graph_api`."""
    script = (
        "import sys\n"
        "import dfo_apis.app as app_module\n"
        "from graph_release.context import bound_served_release\n"
        "from graph_release.service import get_release_service\n"
        "with bound_served_release(get_release_service()):\n"
        "    app_module.deployed_surfaces()\n"
        "bad = [\n"
        "    m for m in sys.modules\n"
        "    if m == 'graph_api' or m.startswith('graph_api.')\n"
        "]\n"
        "assert not bad, f'import pulled in: {bad}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
