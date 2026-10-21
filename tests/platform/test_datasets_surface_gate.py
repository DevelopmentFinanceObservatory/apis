"""Tests the Datasets surface gate, including truthy values, imports, and route mounting.

Each test clears all optional-surface flags to isolate its result from the environment.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

import dfo_apis.app as dfo_apis_app
from dfo_apis.app import all_surfaces, create_app, deployed_surfaces
from search.engine.app import search_surface
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM

_SURFACE_GATE_ENVIRONMENT_VARIABLES = (
    "DFO_DATASETS_SURFACE",
    "DFO_GRAPH_SURFACE",
    "DFO_PRESENTATION_SURFACE",
)


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
    for environment_variable in _SURFACE_GATE_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(environment_variable, raising=False)


@pytest.mark.parametrize(
    ("environment_variable", "value", "expected_surface_names"),
    (
        pytest.param("DFO_DATASETS_SURFACE", "1", ("search", "datasets"), id="datasets-one"),
        pytest.param("DFO_DATASETS_SURFACE", "true", ("search", "datasets"), id="datasets-true"),
        pytest.param(
            "DFO_DATASETS_SURFACE", "True", ("search", "datasets"), id="datasets-title-true"
        ),
        pytest.param(
            "DFO_DATASETS_SURFACE", "TRUE", ("search", "datasets"), id="datasets-upper-true"
        ),
        pytest.param(
            "DFO_DATASETS_SURFACE", " 1 ", ("search", "datasets"), id="datasets-spaced-one"
        ),
        pytest.param(
            "DFO_DATASETS_SURFACE", " true ", ("search", "datasets"), id="datasets-spaced-true"
        ),
        pytest.param("DFO_GRAPH_SURFACE", "1", ("search", "graph"), id="graph-one"),
        pytest.param("DFO_GRAPH_SURFACE", "true", ("search", "graph"), id="graph-true"),
        pytest.param("DFO_GRAPH_SURFACE", "True", ("search", "graph"), id="graph-title-true"),
        pytest.param("DFO_GRAPH_SURFACE", "TRUE", ("search", "graph"), id="graph-upper-true"),
        pytest.param("DFO_GRAPH_SURFACE", " 1 ", ("search", "graph"), id="graph-spaced-one"),
        pytest.param("DFO_GRAPH_SURFACE", " true ", ("search", "graph"), id="graph-spaced-true"),
        pytest.param(
            "DFO_PRESENTATION_SURFACE", "1", ("search", "presentation"), id="presentation-one"
        ),
        pytest.param(
            "DFO_PRESENTATION_SURFACE", "true", ("search", "presentation"), id="presentation-true"
        ),
        pytest.param(
            "DFO_PRESENTATION_SURFACE",
            "True",
            ("search", "presentation"),
            id="presentation-title-true",
        ),
        pytest.param(
            "DFO_PRESENTATION_SURFACE",
            "TRUE",
            ("search", "presentation"),
            id="presentation-upper-true",
        ),
        pytest.param(
            "DFO_PRESENTATION_SURFACE",
            " 1 ",
            ("search", "presentation"),
            id="presentation-spaced-one",
        ),
        pytest.param(
            "DFO_PRESENTATION_SURFACE",
            " true ",
            ("search", "presentation"),
            id="presentation-spaced-true",
        ),
    ),
)
def test_truthy_values_mount_the_optional_surface(
    monkeypatch: pytest.MonkeyPatch,
    environment_variable: str,
    value: str,
    expected_surface_names: tuple[str, ...],
) -> None:
    monkeypatch.setenv(environment_variable, value)
    assert [surface.name for surface in deployed_surfaces()] == list(expected_surface_names)


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
def test_non_truthy_or_unset_value_omits_the_datasets_surface(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("DFO_DATASETS_SURFACE", raising=False)
    else:
        monkeypatch.setenv("DFO_DATASETS_SURFACE", value)
    assert [s.name for s in deployed_surfaces()] == ["search"]


def test_all_surfaces_always_includes_datasets_regardless_of_the_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("DFO_DATASETS_SURFACE", raising=False)
    assert [s.name for s in all_surfaces()] == [
        "search",
        "datasets",
        "graph",
        "presentation",
        "skills",
    ]


@pytest.mark.parametrize(
    "module_prefixes",
    (
        pytest.param(("datasets_api", "dataset_store"), id="datasets"),
        pytest.param(("graph_api",), id="graph"),
        pytest.param(("presentation_api", "presentation_store"), id="presentation"),
    ),
)
def test_flags_unset_never_import_optional_surface_dependencies(
    monkeypatch: pytest.MonkeyPatch, module_prefixes: tuple[str, ...]
) -> None:
    """Restore removed optional modules after the isolation check."""
    for environment_variable in _SURFACE_GATE_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(environment_variable, raising=False)

    removed = {
        module_name: sys.modules[module_name]
        for module_name in sys.modules
        if module_name in module_prefixes
        or module_name.startswith(tuple(f"{prefix}." for prefix in module_prefixes))
    }
    for module_name in removed:
        del sys.modules[module_name]

    try:
        deployed_surfaces()

        leaked = [
            module_name
            for module_name in sys.modules
            if module_name in module_prefixes
            or module_name.startswith(tuple(f"{prefix}." for prefix in module_prefixes))
        ]
        assert not leaked, f"deployed_surfaces() with the flags unset pulled in: {leaked}"
    finally:
        sys.modules.update(removed)


def test_flag_unset_app_starts_without_dataset_store_runtime_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DFO_DATASETS_SURFACE", raising=False)
    monkeypatch.delenv("DATASET_STORE_RUNTIME_DB_USER", raising=False)

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/search/healthz")
        assert resp.status_code == 200

    assert "/api/datasets/register" not in app.openapi()["paths"]


def test_flag_set_mounts_the_register_door_route(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")

    # The store opens during lifespan, after route construction.
    app = create_app()

    assert "/api/datasets/register" in app.openapi()["paths"]


@pytest.mark.parametrize(
    "module_prefixes",
    (
        pytest.param(("datasets_api", "dataset_store"), id="datasets"),
        pytest.param(("graph_api",), id="graph"),
        pytest.param(("presentation_api", "presentation_store"), id="presentation"),
    ),
)
@pytest.mark.usefixtures("published_catalogue_uri")
def test_import_with_flags_unset_does_not_pull_in_optional_surface_dependencies(
    monkeypatch: pytest.MonkeyPatch, module_prefixes: tuple[str, ...]
) -> None:
    """A fresh subprocess verifies that disabled surfaces do not import optional packages."""
    for environment_variable in _SURFACE_GATE_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(environment_variable, raising=False)

    script = (
        "import os\n"
        "import sys\n"
        f"module_prefixes = {module_prefixes!r}\n"
        f"surface_gate_environment_variables = {_SURFACE_GATE_ENVIRONMENT_VARIABLES!r}\n"
        "for environment_variable in surface_gate_environment_variables:\n"
        "    os.environ.pop(environment_variable, None)\n"
        "import dfo_apis.app as app_module\n"
        "from graph_release.context import bound_served_release\n"
        "from graph_release.service import get_release_service\n"
        "with bound_served_release(get_release_service()):\n"
        "    app_module.deployed_surfaces()\n"
        "bad = [\n"
        "    m for m in sys.modules\n"
        "    if m in module_prefixes\n"
        "    or m.startswith(tuple(f'{prefix}.' for prefix in module_prefixes))\n"
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
