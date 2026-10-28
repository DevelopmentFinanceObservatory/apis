"""Compose the API surfaces onto `dfo_platform`.

Search always mounts. `DFO_DATASETS_SURFACE`, `DFO_GRAPH_SURFACE`,
`DFO_PRESENTATION_SURFACE`, and `DFO_SKILLS_SURFACE` gate the other surfaces.
`build_presentation_surface` builds the Presentation surface for any app that composes it.
Datasets requires `DATASET_STORE_RUNTIME_DB_USER` and Graph requires
`GRAPH_DCP_BASE_URL`. Presentation requires its runtime database user and
the datasets runtime user. Skills uses Search's shared graph index.
Gated packages load only when their flags enable them, keeping optional dependencies such
as the Datasets `pg8000` driver out of engine-only installs.

When Datasets and Presentation both mount, they share the Datasets dependency cell, so a live
share slot checks for newer data through Datasets' frame refresh and analysis runner in process
(`presentation_api.newest_frames`).

`all_surfaces()` includes every surface for platform conformance tests.
`deployed_surfaces()` returns live mounts. `DFO_WORKER_SURFACE` bypasses
this composition and serves the worker's `/internal/` route from a separate
ASGI app.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dfo_platform.app import Surface
from dfo_platform.app import create_app as platform_create_app
from dfo_platform.config import parse_cors_origins
from dfo_platform.flags import lenient_bool
from materialise.rate_tables import RateTableCache
from search.engine.app import search_surface

if TYPE_CHECKING:
    from datasets_api.deps import DepsCell

_TRUTHY_VALUES = frozenset({"1", "true"})


class _SurfaceFlags(BaseSettings):
    """Read mount flags on each call. Only `1` and `true` are truthy, ignoring case and
    surrounding whitespace. Unset and unrecognised values are false. The worker flag
    selects a separate ASGI app because its route lives under `/internal/`, outside
    the platform `Surface` prefix.
    """

    model_config = SettingsConfigDict(env_prefix="DFO_")

    datasets_surface: bool = False
    graph_surface: bool = False
    presentation_surface: bool = False
    skills_surface: bool = False
    worker_surface: bool = False

    @field_validator(
        "datasets_surface",
        "graph_surface",
        "presentation_surface",
        "skills_surface",
        "worker_surface",
        mode="before",
    )
    @classmethod
    def _parse_flag(cls, value: object) -> object:
        return lenient_bool(truthy=_TRUTHY_VALUES, strip=True)(value)


@dataclass(frozen=True, slots=True)
class _Shared:
    """What the gated surfaces of one app share: Search's rate cache, and the Datasets
    dependency cell when the Datasets surface mounts."""

    rate_cache: RateTableCache
    datasets_cell: DepsCell | None


def _datasets_factory(shared: _Shared) -> Surface:
    from datasets_api.app import datasets_surface

    return datasets_surface(rate_cache=shared.rate_cache, deps_cell=shared.datasets_cell)


def _graph_factory(shared: _Shared) -> Surface:
    from graph_api.app import graph_surface

    return graph_surface()


def build_presentation_surface(
    rate_cache: RateTableCache, *, datasets_cell: DepsCell | None = None
) -> Surface:
    """Build the Presentation surface `create_app` mounts under `DFO_PRESENTATION_SURFACE`,
    with its render routes admitting the Datasets live-refresh principal from the environment.
    `datasets_cell` is the mounted Datasets surface's dependency cell, which live share slots
    check for newer data through; without one they serve their cards' own frames as stale.
    Takes the shared `rate_cache` for the gated-surface factory signature; Presentation reads
    no rate tables."""
    from datasets_api.config import DatasetsBuildSettings
    from datasets_api.live_refresh import LiveRefreshPrincipal
    from presentation_api.app import presentation_surface
    from presentation_api.config import PRESENTATION_COMPONENTS_TIMEOUT_SECONDS
    from presentation_api.newest_frames import DatasetsNewestFrames

    # The render routes admit the principal the Datasets frame refresh admits, read from the
    # same settings, whether or not the Datasets surface is mounted.
    return presentation_surface(
        live_refresh=LiveRefreshPrincipal.resolve(settings=DatasetsBuildSettings.from_env()),
        newest_frames=(
            # Half the share route's wait goes to the check, leaving the rest to draw the card.
            DatasetsNewestFrames(
                datasets_cell, budget_seconds=PRESENTATION_COMPONENTS_TIMEOUT_SECONDS / 2
            )
            if datasets_cell is not None
            else None
        ),
    )


def _presentation_factory(shared: _Shared) -> Surface:
    return build_presentation_surface(shared.rate_cache, datasets_cell=shared.datasets_cell)


def _skills_factory(shared: _Shared) -> Surface:
    from skills_api.app import skills_surface

    return skills_surface()


@dataclass(frozen=True, slots=True)
class _GatedSurface:
    """Map a `_SurfaceFlags` field to a lazily imported surface factory."""

    flag: str
    factory: Callable[[_Shared], Surface]


_GATED_SURFACES: tuple[_GatedSurface, ...] = (
    _GatedSurface("datasets_surface", _datasets_factory),
    _GatedSurface("graph_surface", _graph_factory),
    _GatedSurface("presentation_surface", _presentation_factory),
    _GatedSurface("skills_surface", _skills_factory),
)


def all_surfaces() -> list[Surface]:
    """Build every surface for platform conformance tests, regardless of deployment flags."""
    # One process-wide RateTableCache, shared by Search and the Datasets
    # surface's mint door below, so neither loads the same tables twice.
    rate_cache = RateTableCache()
    shared = _Shared(rate_cache=rate_cache, datasets_cell=_datasets_cell())
    return [
        search_surface(rate_cache=rate_cache),
        *(gated.factory(shared) for gated in _GATED_SURFACES),
    ]


def _datasets_cell() -> DepsCell:
    from datasets_api.deps import DepsCell

    return DepsCell()


def deployed_surfaces() -> list[Surface]:
    """Return enabled surfaces in mount order for `create_app()` to compose."""
    rate_cache = RateTableCache()
    flags = _SurfaceFlags()
    shared = _Shared(
        rate_cache=rate_cache, datasets_cell=_datasets_cell() if flags.datasets_surface else None
    )
    surfaces = [search_surface(rate_cache=rate_cache)]
    surfaces.extend(
        gated.factory(shared) for gated in _GATED_SURFACES if getattr(flags, gated.flag)
    )
    return surfaces


def create_app() -> FastAPI:
    """Build the service app. `DFO_WORKER_SURFACE` selects the separate worker app with
    `POST /internal/datasets/operations/{operation_id}/run`. Enabled API surfaces
    share the platform's RFC 9457 envelope, code registry, and trace id.
    """
    if _SurfaceFlags().worker_surface:
        from datasets_api.operations.worker_app import build_worker_app

        return build_worker_app()
    return platform_create_app(
        deployed_surfaces(),
        # One `DFO_CORS_ORIGINS` allowlist applies to every mounted surface.
        cors_origins=parse_cors_origins(),
        title="DFO APIs",
        version="1.0",
    )
