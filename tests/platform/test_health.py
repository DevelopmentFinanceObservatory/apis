"""Tests the shared health response model and helper."""

from __future__ import annotations

import pytest

from dfo_platform import HealthResponse, health_response
from dfo_platform.build import build_commit


def test_health_response_reads_build_from_the_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DFO_BUILD_COMMIT", "abc1234")

    response = health_response(capability_version="v-t")

    assert response.build == build_commit() == "abc1234"


def test_health_response_defaults_build_to_unknown_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DFO_BUILD_COMMIT", raising=False)

    response = health_response(capability_version="v-t")

    assert response.build == "unknown"


def test_health_response_carries_the_given_capability_version():
    response = health_response(capability_version="v-t")

    assert response.status == "ok"
    assert response.capability_version == "v-t"


def test_health_response_status_rejects_anything_but_ok():
    with pytest.raises(ValueError):
        HealthResponse(status="degraded", build="abc1234", capability_version="v-t")


def test_model_dump_key_set_is_exactly_status_build_capability_version(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DFO_BUILD_COMMIT", "abc1234")

    response = health_response(capability_version="v-t")

    assert set(response.model_dump()) == {"status", "build", "capability_version"}
