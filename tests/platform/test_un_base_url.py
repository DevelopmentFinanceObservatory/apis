"""`UN_GRAPH_BASE_URL` sets the UN deployment for the catalogue job and every serving surface."""

from __future__ import annotations

import pytest

from datasets_api.config import DatasetsSettings
from graph_api.config import GraphSettings
from origin_catalogue.graph import UN_BASE_URL, UN_BASE_URL_ENV
from origin_catalogue.job import JobSettings
from search.engine.config import SearchSettings

_REQUIRED_ENV = {
    "ORIGIN_CATALOGUE_URI": "file:///tmp/catalogue",
    "ORIGIN_CATALOGUE_BQ_CONNECTION": "proj.region.conn",
    "DATASETS_GRAPH_BASE_URL": "https://mixer/core/api",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned",
    "GRAPH_DCP_BASE_URL": "https://mixer/core/api",
}


def _un_base_urls() -> dict[str, str]:
    return {
        "job": JobSettings().un_base_url,
        "graph": GraphSettings.from_env().un_base_url,
        "datasets": DatasetsSettings.from_env().un_graph_base_url,
        "search": SearchSettings().un_graph_base_url,
    }


@pytest.fixture(autouse=True)
def _required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(UN_BASE_URL_ENV, raising=False)


def test_every_reader_defaults_to_the_same_un_deployment() -> None:
    assert set(_un_base_urls().values()) == {UN_BASE_URL}


def test_one_variable_moves_every_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(UN_BASE_URL_ENV, "https://un.example.org/core/api")

    assert set(_un_base_urls().values()) == {"https://un.example.org/core/api"}


@pytest.mark.parametrize(
    "retired",
    [
        "ORIGIN_CATALOGUE_UN_BASE_URL",
        "GRAPH_UN_BASE_URL",
        "DATASETS_UN_GRAPH_BASE_URL",
        "SEARCH_UN_GRAPH_BASE",
    ],
)
def test_a_surface_prefixed_name_is_not_read(monkeypatch: pytest.MonkeyPatch, retired: str) -> None:
    monkeypatch.setenv(retired, "https://elsewhere/core/api")

    assert set(_un_base_urls().values()) == {UN_BASE_URL}
