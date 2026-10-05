"""Pin how the safe-default settings tier reads the environment."""

from __future__ import annotations

import pytest

from presentation_api.config import PresentationSettings, _PresentationDefaults


def test_a_leftover_presentation_session_header_variable_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
):
    """A deployment that sets `PRESENTATION_SESSION_HEADER` starts normally."""
    monkeypatch.setenv("PRESENTATION_SESSION_HEADER", "x-dfo-session")

    defaults = _PresentationDefaults()

    assert not hasattr(defaults, "session_header")
    assert defaults.components_max_concurrency >= 1


def test_the_frames_door_settings_do_not_require_the_public_site_url(
    monkeypatch: pytest.MonkeyPatch,
):
    for var, value in {
        "DATASETS_GRAPH_BASE_URL": "http://graph.example",
        "DATASETS_SDMX_BASE_URL": "http://sdmx.example",
        "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
        "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
    }.items():
        monkeypatch.setenv(var, value)
    monkeypatch.delenv("DFO_PUBLIC_SITE_URL", raising=False)

    settings = PresentationSettings.from_env()

    assert settings.datasets.graph_base_url == "http://graph.example"
