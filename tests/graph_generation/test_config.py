"""`_GraphGenerationSettings` rejects invalid values and inconsistent lifetimes.
Import-time environment error translation is covered by the API config tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

import graph_generation.config as config_module


def test_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_CACHE_MAX_AGE_SECONDS", raising=False)
    monkeypatch.delenv("GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS", raising=False)
    settings = config_module._GraphGenerationSettings()
    assert settings.cache_max_age_seconds == 86_400
    assert settings.cache_unknown_generation_max_age_seconds == 3_600


def test_rejects_unparseable_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_CACHE_MAX_AGE_SECONDS", "not-an-int")
    with pytest.raises(pydantic.ValidationError, match="not-an-int"):
        config_module._GraphGenerationSettings()


def test_rejects_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_CACHE_MAX_AGE_SECONDS", "0")
    with pytest.raises(pydantic.ValidationError, match="greater than or equal to 1"):
        config_module._GraphGenerationSettings()


def test_rejects_a_lifetime_past_one_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_CACHE_MAX_AGE_SECONDS", "86401")
    with pytest.raises(pydantic.ValidationError, match="less than or equal to 86400"):
        config_module._GraphGenerationSettings()


def test_startup_refuses_a_lifetime_past_one_day() -> None:
    """A UN answer must not outlive its UTC day, so the module refuses to load, naming the
    variable."""
    result = subprocess.run(
        [sys.executable, "-c", "import graph_generation.config"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "GRAPH_CACHE_MAX_AGE_SECONDS": "172800"},
    )
    assert result.returncode != 0
    assert (
        "invalid env var GRAPH_CACHE_MAX_AGE_SECONDS: Input should be less than or equal to 86400"
        in result.stderr
    )


def test_unknown_lifetime_over_known_lifetime_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_CACHE_MAX_AGE_SECONDS", "100")
    monkeypatch.setenv("GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS", "200")
    with pytest.raises(pydantic.ValidationError, match="must be <="):
        config_module._GraphGenerationSettings()


def test_bare_unprefixed_variable_has_no_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATION", "zzz")
    monkeypatch.setenv("GENERATION_SERVICE", "svc")
    monkeypatch.setenv("CACHE_MAX_AGE_SECONDS", "5")
    settings = config_module._GraphGenerationSettings()
    assert settings.generation == ""
    assert settings.generation_service == ""
    assert settings.cache_max_age_seconds == 86_400


def test_module_constants_satisfy_the_cross_check() -> None:
    # The process-level constants derived at import from whatever env was
    # present when this module first loaded (before any test ran, and so
    # unaffected by any test's own monkeypatched environment).
    assert config_module.GRAPH_CACHE_MAX_AGE_SECONDS == 86_400
    assert config_module.GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS == 3_600
