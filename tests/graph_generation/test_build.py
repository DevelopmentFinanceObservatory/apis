"""`build_from_env` selects a fixed or Cloud Run source from environment settings, and every
stamp carries the served catalogue's build id."""

from __future__ import annotations

import logging

import httpx
import pytest

import graph_generation.config as config_module
from graph_generation import GenerationStamp
from graph_generation.build import build_from_env
from tests.graph_generation.conftest import FakeCatalogueBuilds, FakeCredentials


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION", "")
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION_SERVICE", "")


def test_fixed_override_skips_http_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION", "local-mirror-token")
    monkeypatch.setattr(
        config_module, "DFO_GRAPH_GENERATION_SERVICE", "projects/p/locations/r/services/s"
    )

    def transport_never_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("DFO_GRAPH_GENERATION must skip HTTP entirely")

    provider = build_from_env(
        catalogue=FakeCatalogueBuilds("build-a"),
        transport=httpx.MockTransport(transport_never_called),
    )
    provider.refresh_now()
    assert provider.stamp().key == "local-mirror-token@build-a"


def test_live_service_var_builds_a_cloud_run_source(monkeypatch: pytest.MonkeyPatch) -> None:
    service = "projects/p/locations/r/services/dfo-dev-dc-datacommons-service"
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION_SERVICE", service)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"trafficStatuses": [{"revision": "rev-1", "percent": 100}], "urls": []},
        )

    provider = build_from_env(
        catalogue=FakeCatalogueBuilds("build-a"),
        transport=httpx.MockTransport(handler),
        credentials_factory=lambda: FakeCredentials(),
    )
    provider.refresh_now()
    assert provider.stamp().key == "rev-1@build-a"


def test_neither_var_set_stays_unknown_with_one_startup_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="graph_generation.build"):
        provider = build_from_env(catalogue=FakeCatalogueBuilds("build-a"))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "DFO_GRAPH_GENERATION" in warnings[0].message
    assert "DFO_GRAPH_GENERATION_SERVICE" in warnings[0].message

    provider.refresh_now()  # a no-op: there is no source to read from
    assert provider.stamp().token is None
    assert provider.stamp().key == "unknown@build-a"


def test_a_catalogue_swap_changes_the_key_and_clears_registered_caches_without_a_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION", "rev-1")
    catalogue = FakeCatalogueBuilds("build-a")
    provider = build_from_env(catalogue=catalogue)
    cleared: list[str] = []
    provider.register("cache", stats=dict, clear=lambda: cleared.append("cache"))
    provider.refresh_now()
    assert provider.stamp().key == "rev-1@build-a"

    catalogue.swap("build-b")
    assert provider.stamp().key == "rev-1@build-b"
    assert cleared == ["cache"]


def test_an_unknown_token_still_keys_on_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    catalogue = FakeCatalogueBuilds("build-a")
    provider = build_from_env(catalogue=catalogue)
    first = provider.stamp()
    catalogue.swap("build-b")
    second = provider.stamp()

    assert first.key != second.key
    assert first.max_age_s == second.max_age_s == GenerationStamp(token=None).max_age_s
