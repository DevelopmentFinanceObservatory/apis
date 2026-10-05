"""Tests that mint and definition doors pass `DATASETS_SDMX_TIMEOUT_SECONDS` to
`SdmxClient`, using a recording fake with no network call.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import definition as definition_door
from datasets_api.doors import mint as mint_door
from datasets_api.doors.definition import build_definition_door
from datasets_api.doors.mint import build_mint_door
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.versions import NullVersionRecordWriter
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._artifact_doubles import FakeArtifactStore
from .conftest import FakeDatasetStore


class _RecordingSdmxClient:
    """Stands in for `SdmxClient`: captures every keyword argument its own
    constructor received, so `timeout=` is directly observable."""

    created: ClassVar[list[_RecordingSdmxClient]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        type(self).created.append(self)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_recordings():
    _RecordingSdmxClient.created.clear()
    yield


_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
}


class _RecordingGraphReader:
    """Stands in for `build_graph_client`, injected purely to let this test's
    mint/definition door lifespan complete, while its own `sdmx` dependency
    is left uninjected so `SdmxClient`'s live construction path is what
    actually runs."""

    def __init__(self, *, base_url: str, token: str | None = None, **_options: object) -> None:
        self.base_url = base_url
        self.token = token
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _enter_and_exit(door) -> None:
    async def _run() -> None:
        async with door.lifespan(None):
            pass

    asyncio.run(_run())


def test_mint_door_passes_configured_sdmx_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(mint_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(mint_door, "SdmxClient", _RecordingSdmxClient)
    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        settings=DatasetsBuildSettings(sdmx_timeout_seconds=42),
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        # graph, sdmx left uninjected: this door's own lifespan builds both.
        task_client=FakeTaskClient(),
    )

    _enter_and_exit(door)

    assert _RecordingSdmxClient.created[0].kwargs["timeout"] == 42


def test_definition_door_passes_configured_sdmx_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(definition_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(definition_door, "SdmxClient", _RecordingSdmxClient)
    door = build_definition_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        settings=DatasetsBuildSettings(sdmx_timeout_seconds=42),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        # graph, sdmx left uninjected: this door's own lifespan builds both.
        task_client=FakeTaskClient(),
    )

    _enter_and_exit(door)

    assert _RecordingSdmxClient.created[0].kwargs["timeout"] == 42
