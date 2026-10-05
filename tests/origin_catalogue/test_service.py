"""Tests catalogue loading at construction, rechecks, retained snapshots, and singleton wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import origin_catalogue.service as service_module
from origin_catalogue.artifact import (
    CatalogueArtifactError,
    CatalogueTooLargeError,
    load_catalogue,
)
from origin_catalogue.catalogue import CatalogueSnapshot
from origin_catalogue.schema import FORMAT_ROOT
from origin_catalogue.service import (
    CatalogueService,
    CatalogueUriMissingError,
    get_catalogue_service,
    reset_catalogue_service,
)
from tests.origin_catalogue.support import publish_sample


@pytest.fixture(autouse=True)
def _isolated_singleton():
    with service_module._singleton.isolated():
        yield


def _uri(tmp_path: Path) -> str:
    return f"file://{tmp_path / 'catalogue'}"


def test_service_loads_at_construction(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    service = CatalogueService(artifact_uri=uri, recheck_seconds=0)
    assert service.snapshot().build_id == "build-a"
    assert service.snapshot().un_system_enabled


def test_service_raises_at_construction_with_no_build(tmp_path: Path) -> None:
    with pytest.raises(CatalogueArtifactError):
        CatalogueService(artifact_uri=_uri(tmp_path), recheck_seconds=0)


def test_service_refuses_a_first_build_over_budget(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    with pytest.raises(CatalogueTooLargeError):
        CatalogueService(artifact_uri=uri, recheck_seconds=0, max_resident_bytes=1)


def test_service_passes_the_un_setting_to_its_snapshots(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    service = CatalogueService(artifact_uri=uri, recheck_seconds=0, un_system_enabled=False)
    assert not service.snapshot().un_system_enabled


def test_recheck_swaps_to_a_new_build_after_the_interval(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    now = [0.0]
    service = CatalogueService(artifact_uri=uri, recheck_seconds=10, clock=lambda: now[0])
    publish_sample(uri, build_id="build-b")
    now[0] = 5.0
    service.snapshot()
    service.close()
    assert service.snapshot().build_id == "build-a"
    now[0] = 11.0
    service.snapshot()
    service.close()
    assert service.snapshot().build_id == "build-b"


def test_recheck_of_an_unchanged_pointer_does_not_reload(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    loads: list[str] = []

    def loader(u: str) -> CatalogueSnapshot:
        loads.append(u)
        return load_catalogue(u)

    now = [0.0]
    service = CatalogueService(
        artifact_uri=uri, recheck_seconds=10, clock=lambda: now[0], loader=loader
    )
    now[0] = 11.0
    service.snapshot()
    service.close()
    assert len(loads) == 1


def test_failed_reload_keeps_the_last_good_snapshot(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    now = [0.0]
    service = CatalogueService(
        artifact_uri=uri, recheck_seconds=10, clock=lambda: now[0], max_resident_bytes=10**9
    )
    first = service.snapshot()
    publish_sample(uri, build_id="build-b")
    (tmp_path / "catalogue" / FORMAT_ROOT / "versions" / "build-b" / "topics.parquet").unlink()
    now[0] = 11.0
    service.snapshot()
    service.close()
    assert service.snapshot() is first
    with pytest.raises(CatalogueArtifactError):
        service.snapshot_current()
    assert service.snapshot() is first


def test_reload_over_budget_keeps_the_last_good_snapshot(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    build = publish_sample(uri)
    budget = build.manifest.resident_bytes
    now = [0.0]
    service = CatalogueService(
        artifact_uri=uri, recheck_seconds=10, clock=lambda: now[0], max_resident_bytes=budget
    )
    publish_sample(uri, build_id="build-b")
    manifest = tmp_path / "catalogue" / FORMAT_ROOT / "versions" / "build-b" / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["resident_bytes"] = budget + 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(CatalogueTooLargeError):
        service.snapshot_current()
    assert service.snapshot().build_id == "build-a"


def test_snapshot_current_waits_for_a_published_build(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    service = CatalogueService(artifact_uri=uri, recheck_seconds=0)
    publish_sample(uri, build_id="build-b")
    assert service.snapshot_current().build_id == "build-b"
    assert service.snapshot().build_id == "build-b"


def test_get_catalogue_service_requires_the_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORIGIN_CATALOGUE_URI", raising=False)
    with pytest.raises(CatalogueUriMissingError):
        get_catalogue_service()


def test_get_catalogue_service_reads_its_settings_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    monkeypatch.setenv("ORIGIN_CATALOGUE_URI", uri)
    monkeypatch.setenv("ORIGIN_CATALOGUE_RECHECK_SECONDS", "0")
    monkeypatch.setenv("ORIGIN_CATALOGUE_UN_SYSTEM_ENABLED", "false")
    service = get_catalogue_service()
    assert get_catalogue_service() is service
    assert not service.snapshot().un_system_enabled
    monkeypatch.setenv("ORIGIN_CATALOGUE_MAX_RESIDENT_BYTES", "1")
    reset_catalogue_service()
    with pytest.raises(CatalogueTooLargeError):
        get_catalogue_service()
