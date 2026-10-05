"""Tests the Presentation surface gate, including disabled behavior and health-route mounting.

Each test clears the other optional-surface flags to isolate its result.
"""

from __future__ import annotations

import pytest

import dfo_apis.app as dfo_apis_app
from dfo_apis.app import create_app, deployed_surfaces
from search.engine.app import search_surface
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM


def _fake_search_surface(**kwargs: object):
    # Injects fake Search clients so gate tests need no credentials or network.
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
def _clear_other_surface_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DFO_DATASETS_SURFACE", raising=False)
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)


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
def test_non_truthy_or_unset_value_omits_the_presentation_surface(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)
    else:
        monkeypatch.setenv("DFO_PRESENTATION_SURFACE", value)
    assert [s.name for s in deployed_surfaces()] == ["search"]


def test_flag_set_mounts_the_healthz_route(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DFO_PRESENTATION_SURFACE", "1")

    # The factory loads packaged catalogue data without opening the store.
    app = create_app()

    assert "/api/presentation/healthz" in app.openapi()["paths"]
