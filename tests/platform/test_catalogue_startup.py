"""Every surface that reads the graph through the origin client refuses to start without a
loadable origin catalogue."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import origin_catalogue.service as catalogue_service_module
from origin_catalogue import CatalogueArtifactError, CatalogueUriMissingError


def _search() -> object:
    from search.engine.app import search_surface

    return search_surface()


def _datasets() -> object:
    from datasets_api.app import datasets_surface

    return datasets_surface()


def _graph() -> object:
    from graph_api.app import graph_surface

    return graph_surface()


def _presentation() -> object:
    from presentation_api.app import presentation_surface

    return presentation_surface()


def _worker() -> object:
    from datasets_api.operations.worker_app import build_worker_app

    return build_worker_app()


_FACTORIES = pytest.mark.parametrize(
    "build",
    [_search, _datasets, _graph, _presentation, _worker],
    ids=["search", "datasets", "graph", "presentation", "worker"],
)


@pytest.fixture(autouse=True)
def _no_catalogue_service() -> Iterator[None]:
    """Replace the session's fake service with none, so each build loads its own."""
    with catalogue_service_module._singleton.isolated():
        yield


@_FACTORIES
def test_startup_refuses_without_a_catalogue_uri(
    build: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ORIGIN_CATALOGUE_URI", raising=False)
    with pytest.raises(CatalogueUriMissingError):
        build()


@_FACTORIES
def test_startup_refuses_when_no_build_is_published(
    build: Callable[[], object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ORIGIN_CATALOGUE_URI", f"file://{tmp_path / 'empty'}")
    with pytest.raises(CatalogueArtifactError):
        build()


@_FACTORIES
@pytest.mark.usefixtures("published_catalogue_uri")
def test_startup_succeeds_with_a_published_catalogue(build: Callable[[], object]) -> None:
    """Control: the same builds start once a catalogue is published."""
    build()
