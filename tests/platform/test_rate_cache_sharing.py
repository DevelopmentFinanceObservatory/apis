"""Checks that Search and Datasets share one `RateTableCache` per surface-build call."""

from __future__ import annotations

from typing import ClassVar

import pytest

import datasets_api.app as datasets_api_app
import dfo_apis.app as dfo_apis_app
from dfo_apis.app import all_surfaces, deployed_surfaces
from materialise.rate_tables import RateTableCache
from search.engine.app import search_surface
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM

_SURFACE_GATE_ENVIRONMENT_VARIABLES = (
    "DFO_DATASETS_SURFACE",
    "DFO_GRAPH_SURFACE",
    "DFO_PRESENTATION_SURFACE",
    "DFO_SKILLS_SURFACE",
)


class _SpyRateTableCache(RateTableCache):
    """Same admission behaviour as the real cache; records every instance
    built through this constructor so a test can assert exactly one was
    built and identify which one.
    """

    instances: ClassVar[list[_SpyRateTableCache]] = []

    def __init__(self) -> None:
        super().__init__()
        _SpyRateTableCache.instances.append(self)


@pytest.fixture(autouse=True)
def _clear_surface_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for environment_variable in _SURFACE_GATE_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(environment_variable, raising=False)


@pytest.fixture
def _spy_rate_table_cache(monkeypatch: pytest.MonkeyPatch):
    _SpyRateTableCache.instances = []
    monkeypatch.setattr(dfo_apis_app, "RateTableCache", _SpyRateTableCache)
    return _SpyRateTableCache


def _fake_search_surface(**kwargs: object):
    # Same stub test_datasets_surface_gate.py uses: no live LLM/graph/SDMX
    # client, so building the app needs no GEMINI_API_KEY or network. Records
    # every kwarg deployed_surfaces()/all_surfaces() pass through.
    _captured_search_surface_kwargs.append(kwargs)
    return search_surface(
        graph=FakeGraph(),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


_captured_search_surface_kwargs: list[dict[str, object]] = []


@pytest.fixture(autouse=True)
def _stub_search_surface(monkeypatch: pytest.MonkeyPatch):
    _captured_search_surface_kwargs.clear()
    monkeypatch.setattr(dfo_apis_app, "search_surface", _fake_search_surface)
    return _captured_search_surface_kwargs


_captured_datasets_surface_kwargs: list[dict[str, object]] = []


@pytest.fixture(autouse=True)
def _stub_datasets_surface(monkeypatch: pytest.MonkeyPatch):
    # The real factory builds the Surface and its routes; clients start in door lifespans.
    real_datasets_surface = datasets_api_app.datasets_surface
    _captured_datasets_surface_kwargs.clear()

    def _fake_datasets_surface(**kwargs: object):
        _captured_datasets_surface_kwargs.append(kwargs)
        return real_datasets_surface(**kwargs)

    monkeypatch.setattr(datasets_api_app, "datasets_surface", _fake_datasets_surface)
    return _captured_datasets_surface_kwargs


def test_deployed_surfaces_hands_search_the_one_rate_table_cache_it_built(
    _spy_rate_table_cache, _stub_search_surface
):
    deployed_surfaces()

    assert len(_spy_rate_table_cache.instances) == 1
    assert _stub_search_surface[0]["rate_cache"] is _spy_rate_table_cache.instances[0]


def test_all_surfaces_hands_search_the_one_rate_table_cache_it_built(
    _spy_rate_table_cache, _stub_search_surface
):
    all_surfaces()

    assert len(_spy_rate_table_cache.instances) == 1
    assert _stub_search_surface[0]["rate_cache"] is _spy_rate_table_cache.instances[0]


def test_deployed_surfaces_hands_datasets_the_same_rate_table_cache_as_search(
    _spy_rate_table_cache,
    _stub_search_surface,
    _stub_datasets_surface,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")

    deployed_surfaces()

    assert len(_spy_rate_table_cache.instances) == 1
    assert _stub_search_surface[0]["rate_cache"] is _spy_rate_table_cache.instances[0]
    assert _stub_datasets_surface[0]["rate_cache"] is _spy_rate_table_cache.instances[0]


def test_all_surfaces_hands_datasets_the_same_rate_table_cache_as_search(
    _spy_rate_table_cache, _stub_search_surface, _stub_datasets_surface
):
    all_surfaces()

    assert len(_spy_rate_table_cache.instances) == 1
    assert _stub_search_surface[0]["rate_cache"] is _spy_rate_table_cache.instances[0]
    assert _stub_datasets_surface[0]["rate_cache"] is _spy_rate_table_cache.instances[0]
