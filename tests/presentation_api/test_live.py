"""Checks cleanup when live frame-store construction fails."""

from __future__ import annotations

import pytest

import presentation_api.live as live
from datasets_api.config import DatasetsSettings
from presentation_api.config import PresentationSettings


def _settings() -> PresentationSettings:
    return PresentationSettings(
        datasets=DatasetsSettings(
            graph_base_url="http://graph.example",
            sdmx_base_url="http://sdmx.example",
            artifact_bucket_working="working-bucket",
            artifact_bucket_pinned="pinned-bucket",
        ),
        graph_token="",
    )


class _ExplodingLiveFrameStore:
    def __init__(self, **_kwargs: object) -> None:
        raise RuntimeError("boom")


def test_a_construction_failure_after_the_graph_is_built_closes_it_and_the_pool(monkeypatch):
    closed = {"graph": False, "pool": False}

    monkeypatch.setattr(live, "dataset_shared_pool", lambda: object())
    monkeypatch.setattr(live, "LiveDatasetStore", lambda **_kwargs: object())
    monkeypatch.setattr(live, "LiveSnapshotReader", lambda **_kwargs: object())
    monkeypatch.setattr(live, "LiveArtifactStore", lambda **_kwargs: object())
    monkeypatch.setattr(live, "LiveArtifactInventory", lambda **_kwargs: object())
    monkeypatch.setattr(live, "LiveArtifactReferenceWriter", lambda **_kwargs: object())
    monkeypatch.setattr(live, "LiveFrameStore", _ExplodingLiveFrameStore)
    monkeypatch.setattr(
        live.OriginGraphClient, "close", lambda self: closed.__setitem__("graph", True)
    )
    monkeypatch.setattr(live, "close_dataset_pool", lambda: closed.__setitem__("pool", True))

    with pytest.raises(RuntimeError, match="boom"):
        live.build_live_frame_store(settings=_settings(), allowances=None)

    assert closed == {"graph": True, "pool": True}


def test_a_construction_failure_before_the_graph_is_built_still_closes_the_pool(monkeypatch):
    closed = {"pool": False}

    monkeypatch.setattr(live, "dataset_shared_pool", lambda: object())

    def _boom(**_kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(live, "LiveDatasetStore", _boom)
    monkeypatch.setattr(live, "close_dataset_pool", lambda: closed.__setitem__("pool", True))

    with pytest.raises(RuntimeError, match="boom"):
        live.build_live_frame_store(settings=_settings(), allowances=None)

    assert closed == {"pool": True}
