"""Safe-default settings load at import. Required settings load from
`PresentationSettings.from_env()` during a door lifespan. The surface reuses `DatasetsSettings`
for its graph URL and artifact buckets. That class also requires
`DATASETS_SDMX_BASE_URL` for the datasets SDMX client. `presentation_store` loads its own
connection settings.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from datasets_api.config import DatasetsSettings, FoldCeilingSettings, GroupbyFoldMaxRows
from dfo_platform.config import env_validation_error_message

__all__ = [
    "PRESENTATION_COMPONENTS_MAX_CONCURRENCY",
    "PRESENTATION_COMPONENTS_TIMEOUT_SECONDS",
    "PRESENTATION_FRAMES_BARE_MAX_CONCURRENCY",
    "PRESENTATION_FRAMES_MAX_CONCURRENCY",
    "PRESENTATION_FRAMES_TIMEOUT_SECONDS",
    "PresentationSettings",
]


class _PresentationDefaults(BaseSettings):
    """The safe-default tier: read once at import, each field validated by
    pydantic. An unparseable or below-minimum value raises
    `pydantic.ValidationError`, translated below into a `ValueError` naming
    the offending `PRESENTATION_*` variable (`dfo_platform.config
    .env_validation_error_message`, shared with `datasets_api.config`,
    `graph_api.config`, and `graph_generation.config`).
    """

    # Environment variables are read per declared field, so an unknown `PRESENTATION_*`
    # variable, such as a `PRESENTATION_SESSION_HEADER` left on a deployment, is ignored.
    model_config = SettingsConfigDict(env_prefix="PRESENTATION_")

    # One executor shared by the components door's create, revise, and read routes.
    components_max_concurrency: int = Field(default=4, ge=1)

    # Bounds the executor wait. A timeout does not stop its worker thread.
    components_timeout_seconds: int = Field(default=30, ge=1)

    # Separate capacity keeps frame reads independent of component writes.
    frames_max_concurrency: int = Field(default=4, ge=1)

    # A bare `spec_id` read can rebuild its dataset, so it has its own executor and pair reads
    # keep theirs.
    frames_bare_max_concurrency: int = Field(default=2, ge=1)

    frames_timeout_seconds: int = Field(default=30, ge=1)


try:
    _defaults = _PresentationDefaults()
except ValidationError as exc:
    raise ValueError(env_validation_error_message(exc, env_prefix="PRESENTATION_")) from exc

PRESENTATION_COMPONENTS_MAX_CONCURRENCY: int = _defaults.components_max_concurrency
PRESENTATION_COMPONENTS_TIMEOUT_SECONDS: int = _defaults.components_timeout_seconds
PRESENTATION_FRAMES_MAX_CONCURRENCY: int = _defaults.frames_max_concurrency
PRESENTATION_FRAMES_BARE_MAX_CONCURRENCY: int = _defaults.frames_bare_max_concurrency
PRESENTATION_FRAMES_TIMEOUT_SECONDS: int = _defaults.frames_timeout_seconds


class PresentationSettings(BaseModel):
    """Settings with no safe default, validated fail-fast once from a
    door's `lifespan` (never at import). See this module's docstring for
    why this delegates to `DatasetsSettings` rather than declaring its
    own connection fields.

    `graph_token` is the exception: it is read here, under the same
    `DATASETS_GRAPH_TOKEN` env var `datasets_api.config.DatasetsBuildSettings`
    reads, so this surface's `OriginGraphClient` construction reads it
    from `PresentationSettings` alone. `groupby_fold_max_rows` is the fold
    ceiling Datasets reads (`datasets_api.config.FoldCeilingSettings`), so the
    frame store bounds a frame1 document as Datasets does. Nothing below the presentation
    lifespan reads the environment directly.
    """

    model_config = ConfigDict(frozen=True)

    datasets: DatasetsSettings
    graph_token: str
    groupby_fold_max_rows: GroupbyFoldMaxRows

    @classmethod
    def from_env(cls) -> PresentationSettings:
        return cls(
            datasets=DatasetsSettings.from_env(),
            graph_token=os.environ.get("DATASETS_GRAPH_TOKEN", ""),
            groupby_fold_max_rows=FoldCeilingSettings.from_env().groupby_fold_max_rows,
        )
