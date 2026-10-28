"""Settings tests cover environment defaults, overrides, and fast failures for invalid or
missing required values. Each test changes the environment before constructing settings.
"""

from __future__ import annotations

import asyncio

import pytest

from datasets_api.app import datasets_surface
from datasets_api.config import (
    DatasetsBuildSettings,
    DatasetsConfigError,
    DatasetsSettings,
)
from datasets_api.operation_store import NullOperationStore
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.versions import NullHeadWriter, NullVersionRecordWriter
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._app_factories import offline_allowances, offline_recipe_stores
from ._artifact_doubles import FakeArtifactStore
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter


def test_sync_budget_rows_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATASETS_SYNC_BUDGET_ROWS", "12345")
    assert DatasetsBuildSettings().sync_budget_rows == 12345


def test_sync_budget_requests_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATASETS_SYNC_BUDGET_REQUESTS", "7")
    assert DatasetsBuildSettings().sync_budget_requests == 7


def test_sync_budget_rows_non_integer_fails_fast_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASETS_SYNC_BUDGET_ROWS", "not-an-int")
    with pytest.raises(DatasetsConfigError, match="DATASETS_SYNC_BUDGET_ROWS"):
        DatasetsBuildSettings.from_env()


def test_sync_budget_rows_below_minimum_fails_fast_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASETS_SYNC_BUDGET_ROWS", "0")
    with pytest.raises(DatasetsConfigError, match="DATASETS_SYNC_BUDGET_ROWS"):
        DatasetsBuildSettings.from_env()


def test_sync_budget_requests_below_minimum_fails_fast_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASETS_SYNC_BUDGET_REQUESTS", "0")
    with pytest.raises(DatasetsConfigError, match="DATASETS_SYNC_BUDGET_REQUESTS"):
        DatasetsBuildSettings.from_env()


def test_datasets_graph_token_defaults_to_empty() -> None:
    assert DatasetsBuildSettings().graph_token == ""


def test_datasets_graph_token_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATASETS_GRAPH_TOKEN", "shh-secret")
    assert DatasetsBuildSettings().graph_token == "shh-secret"


def test_head_recheck_at_or_past_max_age_fails_fast_naming_both_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASETS_HEAD_RECHECK_SECONDS", "900")
    monkeypatch.setenv("DATASETS_HEAD_MAX_AGE_SECONDS", "900")
    with pytest.raises(
        DatasetsConfigError,
        match=r"DATASETS_HEAD_RECHECK_SECONDS.*DATASETS_HEAD_MAX_AGE_SECONDS",
    ):
        DatasetsBuildSettings.from_env()


_REQUIRED_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "dev-fin-observatory-datasets-working",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "dev-fin-observatory-datasets-pinned",
}


def test_datasets_settings_from_env_succeeds_with_every_required_var_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in _REQUIRED_SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    settings = DatasetsSettings.from_env()
    assert settings == DatasetsSettings(
        graph_base_url=_REQUIRED_SETTINGS_ENV["DATASETS_GRAPH_BASE_URL"],
        sdmx_base_url=_REQUIRED_SETTINGS_ENV["DATASETS_SDMX_BASE_URL"],
        artifact_bucket_working=_REQUIRED_SETTINGS_ENV["DATASETS_ARTIFACT_BUCKET_WORKING"],
        artifact_bucket_pinned=_REQUIRED_SETTINGS_ENV["DATASETS_ARTIFACT_BUCKET_PINNED"],
    )


@pytest.mark.parametrize("missing_var", sorted(_REQUIRED_SETTINGS_ENV))
def test_datasets_settings_from_env_fails_fast_naming_each_missing_var(
    missing_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var, value in _REQUIRED_SETTINGS_ENV.items():
        if var != missing_var:
            monkeypatch.setenv(var, value)
    monkeypatch.delenv(missing_var, raising=False)

    with pytest.raises(DatasetsConfigError, match=missing_var):
        DatasetsSettings.from_env()


@pytest.mark.parametrize("empty_var", sorted(_REQUIRED_SETTINGS_ENV))
def test_datasets_settings_from_env_fails_fast_naming_each_empty_var(
    empty_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var, value in _REQUIRED_SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv(empty_var, "")

    with pytest.raises(DatasetsConfigError, match=empty_var):
        DatasetsSettings.from_env()


def test_datasets_settings_from_env_is_actually_invoked_by_the_surfaces_own_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves `DatasetsSettings.from_env()` is live wiring, not a call site
    that exists in prose only: a `from_env` that raises must fail the
    surface's lifespan entry once it builds a live client, the same way a
    real required setting added to `DatasetsSettings` would. Every store is
    injected, so the live graph client is the first thing built.
    """

    def _raise() -> DatasetsSettings:
        raise DatasetsConfigError("missing required env var DATASETS_SOMETHING_FUTURE")

    monkeypatch.setattr(DatasetsSettings, "from_env", staticmethod(_raise))
    store = FakeDatasetStore()
    surface = datasets_surface(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        snapshot_reader=FakeSnapshotReader(),
        version_writer=NullVersionRecordWriter(),
        head_writer=NullHeadWriter(),
        operation_store=NullOperationStore(),
        reference_writer=FakeArtifactReferenceWriter(),
        artifact_store=FakeArtifactStore(),
        task_client=FakeTaskClient(),
        allowances=offline_allowances(),
        **offline_recipe_stores(),
    )

    async def _enter_and_exit_lifespan() -> None:
        async with surface.lifespan(None):
            pass

    with pytest.raises(DatasetsConfigError, match="DATASETS_SOMETHING_FUTURE"):
        asyncio.run(_enter_and_exit_lifespan())


def test_datasets_build_settings_refuse_a_missing_public_site_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DFO_PUBLIC_SITE_URL", raising=False)

    with pytest.raises(ValueError, match="invalid env var DFO_PUBLIC_SITE_URL"):
        DatasetsBuildSettings.from_env()


def test_the_datasets_surface_refuses_to_build_without_the_public_site_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DFO_PUBLIC_SITE_URL", raising=False)

    with pytest.raises(ValueError, match="invalid env var DFO_PUBLIC_SITE_URL"):
        datasets_surface(registry=LiveSpecRegistry(store=FakeDatasetStore()))
