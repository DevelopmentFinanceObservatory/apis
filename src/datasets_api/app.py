"""Compose the Datasets API doors into one `dfo_platform.app.Surface`.

`datasets_surface()` combines each door's router, classifier, exception types, and admission
checks, and owns the one lifespan that builds the doors' shared `Deps` (`datasets_api.deps`). The
meta door mounts first. Static paths are disjoint from subject-id routes. Those routes accept only
`spec_` ids.

Routes include `GET /api/datasets/healthz`, `/api/datasets/schema`,
`/api/datasets/capabilities`, `/api/datasets/operations/{operation_id}`,
`GET /api/datasets/{spec_id}/{content_version}/observations`,
`GET /api/datasets/{spec_id}/{content_version}/files/{digest}`,
`GET /api/datasets/{spec_id}/{content_version}/file`,
`GET /api/datasets/{id}/{content_version}/description` and its `/values` page for a `spec_` or
`ops_` id, `POST /api/datasets/register`,
`POST /api/datasets`, `POST /api/datasets/frames/publish`, `POST /api/datasets/frames/refresh`,
the recipe door's
`PUT` and `GET /api/datasets/frames/{spec_id}/{content_version}/recipe` and
`GET /api/datasets/recipes/replays/{replay_key}`, `POST /api/datasets/analysis` and its
`/anchor` route, and `GET /api/datasets/{spec_id}/versions`.
Registration stores definitions for later materialisation. The mint route handles four input
kinds. Publish records derived `ops_`/`ocv_` frames, and the recipe door keeps their recipes.
The versions door mounts before the snapshot door, whose
`GET /api/datasets/{spec_id}/{content_version}` would otherwise read `versions` as a content
version. Unknown operation ids return `operation_not_found`. Other errors use RFC 9457
problem details.

Callers reach the API through the website's proxy, which answers `429` once an IP passes its
allowance, 120 calls a minute on `/api/*` by default, counted in a fixed window that every proxy
instance shares in Postgres. A call whose count fails goes through. The proxy passes
`Authorization` upstream unread. The deployment bounds request bodies, and publish also checks its
own `Content-Length` ceiling at admission, before it reads the body.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI

from artifact_store import ArtifactStore
from code_index.query import CodeIndex
from code_index.shared import get_index
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import SnapshotReader
from datasets_api.allowance import AllowanceLedger
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell, DepsOverrides, open_deps
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.doors.analysis import build_analysis_door, classify_analysis_exception
from datasets_api.doors.definition import build_definition_door, classify_definition_exception
from datasets_api.doors.description import build_description_door
from datasets_api.doors.files import build_files_door, classify_files_exception
from datasets_api.doors.meta import build_meta_door, classify_meta_exception
from datasets_api.doors.mint import build_mint_door
from datasets_api.doors.mint_errors import classify_mint_exception
from datasets_api.doors.observations import (
    build_observations_door,
    classify_observations_exception,
)
from datasets_api.doors.pin import build_pin_door, classify_pin_exception
from datasets_api.doors.publish import build_publish_door, classify_publish_exception
from datasets_api.doors.recipe import build_recipe_door, classify_recipe_exception
from datasets_api.doors.refresh import build_refresh_door, classify_refresh_exception
from datasets_api.doors.register import (
    RegistryBackpressure as RegistryBackpressure,  # Tests import this exception from this module.
)
from datasets_api.doors.register import (
    build_register_door,
    classify_register_exception,
)
from datasets_api.doors.snapshot import build_snapshot_door
from datasets_api.doors.versions import build_versions_door
from datasets_api.operation_store import OperationStore, classify_operation_store_exception
from datasets_api.operations.task_client import TaskClient
from datasets_api.recipe_store import RecipeStore
from datasets_api.reference_store import PinReferenceWriter
from datasets_api.routing import build_id_grammar_refusal_router
from datasets_api.store_errors import classify_store_exception
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.versions import HeadStore, VersionLister, VersionRecordWriter
from dc_sdmx import SdmxClient
from dfo_platform.app import Surface
from dfo_platform.doors import combine_classifiers, flatten
from graph_release.context import bound_served_release
from graph_release.service import ReleaseService, get_release_service
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import bound_catalogue
from origin_catalogue.service import CatalogueService, get_catalogue_service
from series_coverage.shared import ServedCoverage, get_coverage_service
from spec_registry import SpecRegistry
from spec_registry.errors import SpecRegistryStoreError
from spec_registry.reader import SpecReader

# Classifiers depend only on exception types, so the module-level combination needs no Surface.
# Store and operation-store exceptions span multiple doors and join the door classifiers here.
classify_datasets_exception = combine_classifiers(
    classify_meta_exception,
    classify_register_exception,
    classify_mint_exception,
    classify_definition_exception,
    classify_files_exception,
    classify_observations_exception,
    classify_pin_exception,
    classify_publish_exception,
    classify_recipe_exception,
    classify_refresh_exception,
    classify_analysis_exception,
    classify_store_exception,
    classify_operation_store_exception,
)


def datasets_surface(
    *,
    settings: DatasetsBuildSettings | None = None,
    registry: SpecRegistry | None = None,
    spec_reader: SpecReader | None = None,
    snapshot_reader: SnapshotReader | None = None,
    version_writer: VersionRecordWriter | None = None,
    head_writer: HeadStore | None = None,
    version_lister: VersionLister | None = None,
    operation_store: OperationStore | None = None,
    recipe_store: RecipeStore | None = None,
    description_store: LiveDescriptionStore | None = None,
    reference_writer: PinReferenceWriter | None = None,
    allowances: AllowanceLedger | None = None,
    graph: GraphReader | None = None,
    sdmx: SdmxClient | None = None,
    artifact_store: ArtifactStore | None = None,
    task_client: TaskClient | None = None,
    code_index: CodeIndex | None = None,
    rate_cache: RateTableCache | None = None,
    coverage: ServedCoverage | None = None,
    now: Clock = utc_now,
    pin_allowed_principals: frozenset[str] | None = None,
    publish_allowed_principals: frozenset[str] | None = None,
    catalogue: CatalogueService | None = None,
    release: ReleaseService | None = None,
) -> Surface:
    """Build the Datasets `Surface` from its doors.

    `settings` defaults to `DatasetsBuildSettings.from_env()` and sizes the read and work
    executors and the materialise semaphore. Each dependency keyword replaces the live one the
    lifespan would build (`datasets_api.deps.open_deps`); the caller owns what it passes. A
    store left out is built over one `dataset_store` pool borrow shared by every omitted store.
    `head_writer` is the head store. `now` is the clock every door reads.

    A given `code_index` serves every request. Left out, each request reads `get_index()`,
    which can change between requests. The index is read here, under the served release as a
    request reads it, so a release without one fails construction. `coverage` defaults to the
    bound release's coverage (`series_coverage.shared.ReleaseCoverage`).
    `rate_cache` defaults to a cache of this surface's own; pass Search's when composing both.

    `pin_allowed_principals` and `publish_allowed_principals` set separate allowlists,
    defaulting to `DATASETS_PIN_FIRST_PARTY_PRINCIPAL` and `DATASETS_FIRST_PARTY_PRINCIPAL`.
    Pinning has no delete route, so publish permission grants no pin permission. The recipe
    door admits the publish allowlist. The frame refresh door admits
    `settings.live_refresh_principal` alone.

    Each request binds `catalogue`'s loaded snapshot. It defaults to the process-wide service,
    so a missing or unloadable `ORIGIN_CATALOGUE_URI` fails startup. Without an injected `graph`,
    the lifespan acquires the process-wide graph generation the live graph client keys its
    caches on, and registers the default-scope memo on it.

    Each request also binds `release`'s served release, and a swap holds every materialise
    permit. `release` defaults to the process-wide service, so an unset `GRAPH_RELEASE_URI` or
    an unreadable release fails startup.
    """
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _catalogue = catalogue if catalogue is not None else get_catalogue_service()
    _release = release if release is not None else get_release_service()
    cell = DepsCell()
    # The doors read the code index as they are built, so they are built under the release a
    # request would bind, and keep none of what they read.
    with bound_served_release(_release):
        if code_index is None:
            get_index()
        doors = (
            build_meta_door(deps=cell, settings=_settings),
            build_register_door(deps=cell),
            build_mint_door(deps=cell),
            build_definition_door(deps=cell),
            build_refresh_door(deps=cell, settings=_settings),
            build_analysis_door(deps=cell),
            build_recipe_door(
                deps=cell, settings=_settings, allowed_principals=publish_allowed_principals
            ),
            build_versions_door(deps=cell),
            build_snapshot_door(deps=cell),
            build_description_door(deps=cell),
            build_files_door(deps=cell),
            build_observations_door(deps=cell),
            build_pin_door(
                deps=cell, settings=_settings, allowed_principals=pin_allowed_principals
            ),
            build_publish_door(
                deps=cell, settings=_settings, allowed_principals=publish_allowed_principals
            ),
        )
    overrides = DepsOverrides(
        coverage=coverage if coverage is not None else get_coverage_service(),
        rate_cache=rate_cache if rate_cache is not None else RateTableCache(),
        code_index=code_index,
        release=_release,
        registry=registry,
        spec_reader=spec_reader,
        snapshot_reader=snapshot_reader,
        version_writer=version_writer,
        head_store=head_writer,
        version_lister=version_lister,
        operation_store=operation_store,
        recipe_store=recipe_store,
        description_store=description_store,
        reference_writer=reference_writer,
        allowances=allowances,
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        task_client=task_client,
        now=now,
    )

    router = flatten(doors)

    # Append grammar refusals after real routes. The converter excludes valid ids.
    router.routes.extend(build_id_grammar_refusal_router().routes)

    @contextmanager
    def request_scope() -> Iterator[None]:
        with bound_catalogue(_catalogue.snapshot()), bound_served_release(_release):
            yield

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with open_deps(settings=_settings, overrides=overrides, catalogue=_catalogue) as deps:
            # The bundle stays in the cell after shutdown, so a late request meets the shut-down
            # executors and refuses with its door's backpressure refusal.
            cell.set(deps)
            yield

    return Surface(
        name="datasets",
        description="Register and mint datasets, poll their operations, and read versions, observations and files.",
        prefix="/api/datasets",
        router=router,
        capability_version=DATASETS_CAPABILITY_VERSION,
        codes=DATASETS_CODE_CATALOGUE,
        # Store classifiers cover errors shared across doors. Grammar refusals fall through to
        # the platform's `StarletteHTTPException` handling.
        classify=combine_classifiers(
            *(door.classify for door in doors),
            classify_store_exception,
            classify_operation_store_exception,
        ),
        # A spec registry store fault can reach any door that reads or writes the registry, so
        # the surface declares it once.
        exception_types=(
            *(exc_type for door in doors for exc_type in door.exception_types),
            SpecRegistryStoreError,
        ),
        exception_exemplars=(
            *(exemplar for door in doors for exemplar in door.exception_exemplars),
            SpecRegistryStoreError("exemplar"),
        ),
        lifespan=lifespan,
        middleware=tuple(check for door in doors for check in door.middleware),
        registry_endpoints={
            path: endpoint for door in doors for path, endpoint in door.registry_endpoints.items()
        },
        request_scope=request_scope,
    )
