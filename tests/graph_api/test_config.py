"""`GraphSettings.from_env()` names missing, empty, and invalid settings."""

from __future__ import annotations

import pytest

from graph_api.config import GraphSettings


def test_dcp_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_DCP_BASE_URL", "https://dcp.example.org")
    monkeypatch.delenv("GRAPH_DCP_TOKEN", raising=False)
    settings = GraphSettings.from_env()
    assert settings.dcp_base_url == "https://dcp.example.org"
    assert settings.dcp_token == ""


def test_dcp_base_url_missing_fails_fast_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAPH_DCP_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="GRAPH_DCP_BASE_URL"):
        GraphSettings.from_env()


def test_dcp_base_url_empty_fails_fast_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPH_DCP_BASE_URL", "")
    with pytest.raises(ValueError, match="GRAPH_DCP_BASE_URL"):
        GraphSettings.from_env()
