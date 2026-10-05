"""Shared-provider reference counting and recovery when ADC credentials fail."""

from __future__ import annotations

import logging

import google.auth.exceptions
import httpx
import pytest

import graph_generation.config as config_module
import graph_generation.shared as shared_module
from graph_generation.shared import acquire_shared, release_shared
from tests.graph_generation.conftest import FakeCatalogueBuilds

_SERVICE = "projects/p/locations/l/services/s"


@pytest.fixture(autouse=True)
def _isolated_singleton(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION", "fixed-test-token")
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION_SERVICE", "")
    with shared_module._singleton.isolated():
        yield


_CATALOGUE = FakeCatalogueBuilds("build-a")


def _no_http(request: httpx.Request) -> httpx.Response:
    raise AssertionError("the fixed-token override must never make an HTTP call")


def test_acquire_shared_twice_returns_the_same_object_and_reads_once() -> None:
    first = acquire_shared(catalogue=_CATALOGUE, transport=httpx.MockTransport(_no_http))
    second = acquire_shared(catalogue=_CATALOGUE, transport=httpx.MockTransport(_no_http))
    try:
        assert first is second
        assert first.stamp().key == "fixed-test-token@build-a"
    finally:
        release_shared(second)
        release_shared(first)


def test_release_shared_twice_then_acquire_builds_anew() -> None:
    first = acquire_shared(catalogue=_CATALOGUE)
    second = acquire_shared(catalogue=_CATALOGUE)
    assert first is second

    release_shared(second)
    release_shared(first)

    third = acquire_shared(catalogue=_CATALOGUE)
    try:
        assert third is not first
    finally:
        release_shared(third)


def test_release_shared_refuses_an_object_it_never_returned() -> None:
    shared = acquire_shared(catalogue=_CATALOGUE)
    try:
        with pytest.raises(RuntimeError):
            release_shared(object())  # type: ignore[arg-type]
    finally:
        release_shared(shared)


def test_release_shared_refuses_once_the_refcount_is_already_zero() -> None:
    shared = acquire_shared(catalogue=_CATALOGUE)
    release_shared(shared)
    with pytest.raises(RuntimeError):
        release_shared(shared)


def test_acquire_shared_returns_normally_when_adc_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`DFO_GRAPH_GENERATION_SERVICE` set, no ADC available: the 0-to-1
    build's `refresh_now()` must contain the auth failure, never raise out
    of `acquire_shared`, and leave the shared instance unknown.
    """
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION", "")
    monkeypatch.setattr(config_module, "DFO_GRAPH_GENERATION_SERVICE", _SERVICE)

    def factory():
        raise google.auth.exceptions.DefaultCredentialsError("no ADC found")

    def no_http(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be reached when credentials cannot be built")

    with caplog.at_level(logging.WARNING, logger="graph_generation.provider"):
        shared = acquire_shared(
            catalogue=_CATALOGUE,
            transport=httpx.MockTransport(no_http),
            credentials_factory=factory,
        )
    try:
        assert shared.stamp().token is None
        assert any("read failed" in r.message for r in caplog.records)
    finally:
        release_shared(shared)
