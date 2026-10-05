"""Pins how the search catalogue service loads, refreshes and keeps a build."""

from __future__ import annotations

from pathlib import Path

import pytest

from search.catalogue.artifact import CatalogueArtifact, CatalogueArtifactError, load_published
from search.catalogue.service import SearchCatalogueService, search_catalogue_service
from tests.search_catalogue.support import publish_catalogue, sample_catalogue


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_loads_the_published_build(tmp_path: Path) -> None:
    uri = tmp_path.as_uri()
    published = publish_catalogue(uri)
    service = SearchCatalogueService(artifact_uri=uri, recheck_seconds=0)
    assert service.current() == published


def test_starts_empty_when_nothing_is_published(tmp_path: Path) -> None:
    service = SearchCatalogueService(artifact_uri=tmp_path.as_uri(), recheck_seconds=0)
    assert service.current() is None


def test_picks_up_a_first_build_after_the_recheck_interval(tmp_path: Path) -> None:
    uri = tmp_path.as_uri()
    clock = _Clock()
    service = SearchCatalogueService(artifact_uri=uri, recheck_seconds=300, clock=clock)
    published = publish_catalogue(uri)

    clock.now = 299
    assert service.current() is None
    clock.now = 300
    service.current()
    service.close()
    assert service.current() == published


def test_reloads_only_when_the_pointer_moves(tmp_path: Path) -> None:
    uri = tmp_path.as_uri()
    publish_catalogue(uri)
    clock = _Clock()
    loads: list[str] = []

    def loader(base_uri: str) -> CatalogueArtifact:
        loaded = load_published(base_uri)
        loads.append(loaded.catalogue_version)
        return loaded

    service = SearchCatalogueService(
        artifact_uri=uri, recheck_seconds=300, clock=clock, loader=loader
    )
    clock.now = 300
    service.current()
    service.close()
    assert loads == ["20260930T000000Z-00000001"]

    newer = publish_catalogue(uri, sample_catalogue("20260930T001500Z-00000002"))
    clock.now = 600
    service.current()
    service.close()
    assert loads == ["20260930T000000Z-00000001", "20260930T001500Z-00000002"]
    assert service.current() == newer


def test_a_failed_reload_keeps_the_last_good_build(tmp_path: Path) -> None:
    uri = tmp_path.as_uri()
    published = publish_catalogue(uri)
    clock = _Clock()
    calls = 0

    def loader(base_uri: str) -> CatalogueArtifact:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise CatalogueArtifactError("unreadable build")
        return load_published(base_uri)

    service = SearchCatalogueService(
        artifact_uri=uri, recheck_seconds=300, clock=clock, loader=loader
    )
    publish_catalogue(uri, sample_catalogue("20260930T001500Z-00000002"))
    clock.now = 300
    service.current()
    service.close()
    assert calls == 2
    assert service.current() == published


def test_no_service_without_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARCH_CATALOGUE_URI", raising=False)
    assert search_catalogue_service() is None


def test_service_reads_the_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uri = tmp_path.as_uri()
    published = publish_catalogue(uri)
    monkeypatch.setenv("SEARCH_CATALOGUE_URI", uri)
    service = search_catalogue_service()
    assert service is not None
    assert service.current() == published
