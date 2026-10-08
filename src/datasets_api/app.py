"""Compose the Datasets API doors into one `dfo_platform.app.Surface`.

`datasets_surface()` combines each door's router, classifier, exception types, lifespan, and
admission checks. It builds the door tuple per call so tests can inject registry settings. The
meta door mounts first. Static paths are disjoint from subject-id routes. Those routes accept only
`spec_` ids.

Routes include `GET /api/datasets/healthz`, `/api/datasets/schema`,
`/api/datasets/capabilities`, `/api/datasets/operations/{operation_id}`,
`GET /api/datasets/{spec_id}/{content_version}/observations`, `POST /api/datasets/register`,
`POST /api/datasets`, `POST /api/datasets/frames/publish`, `POST /api/datasets/frames/refresh`,
and the recipe door's
`PUT` and `GET /api/datasets/frames/{spec_id}/{content_version}/recipe`,
`GET /api/datasets/recipes/replays/{replay_key}`, and `GET /api/datasets/{spec_id}/versions`.
Registration stores definitions for later materialisation. The mint route handles four input
kinds. Publish records derived `ops_`/`ocv_` frames, and the recipe door keeps their recipes. It
mounts before the snapshot door, whose `GET /api/datasets/{spec_id}/{content_version}` would
otherwise read `versions` as a content version. Unknown operation ids return `operation_not_found`. Other errors use RFC 9457
problem details.

The Data access proxy authenticates and authorises requests and enforces `401`, `413`, `415`, and
`429`. The deployment bounds request bodies, except publish, which checks its own `Content-Length`
ceiling before proxy admission.
"""

from __future__ import annotations

import threading
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from artifact_store import ArtifactStore
from code_index.query import CodeIndex
from dataset_store.snapshot import SnapshotReader
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.config import DatasetsBuildSettings, DatasetsSettings
from datasets_api.doors.definition import build_definition_door, classify_definition_exception
from datasets_api.doors.meta import build_meta_door, classify_meta_exception
from datasets_api.doors.mint import build_mint_door
from datasets_api.doors.mint_errors import classify_mint_exception
from datasets_api.doors.observations import (
    build_observations_door,
    classify_observations_exception,
)
from datasets_api.doors.pin import PinReferenceWriter, build_pin_door, classify_pin_exception
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
from datasets_api.doors.snapshot import build_snapshot_door, classify_snapshot_exception
from datasets_api.generation import (
    PLACE_SCOPE_MEMO_NAME,
    enter_shared_generation,
    register_place_scope_memo,
)
from datasets_api.operation_store import OperationStore, classify_operation_store_exception
from datasets_api.operations.task_client import TaskClient

# Raw-fetch floor for mint/definition materialisation. `raw_fetch_guard` scales it to each
# request's coverage estimate. Requests lacking a coverage row still need a bound. The floor
# reuses the `publish_models.MAX_ROWS` backstop.
from datasets_api.publish_models import MAX_ROWS as _MATERIALISE_RAW_FETCH_GUARD_ROWS
from datasets_api.recipe_store import RecipeStore
from datasets_api.routing import build_id_grammar_refusal_router
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import classify_store_exception
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.versions import HeadStore, VersionLister, VersionRecordWriter
from dc_sdmx import SdmxClient
from dfo_platform.app import Surface
from dfo_platform.build import build_commit
from dfo_platform.doors import DoorContribution, combine_classifiers, flatten
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from materialise.revision import RevisionReader
from origin_catalogue import bound_catalogue
from origin_catalogue.service import CatalogueService, get_catalogue_service
from series_coverage.shared import CoverageService
from spec_registry import SpecRegistry
from spec_registry.reader import SpecReader

# Classifiers depend only on exception types, so the module-level combination needs no Surface.
# Store and operation-store exceptions span multiple doors and join the door classifiers here.
classify_datasets_exception = combine_classifiers(
    classify_meta_exception,
    classify_register_exception,
    classify_mint_exception,
    classify_definition_exception,
    classify_snapshot_exception,
    classify_observations_exception,
    classify_pin_exception,
    classify_publish_exception,
    classify_recipe_exception,
    classify_refresh_exception,
    classify_store_exception,
    classify_operation_store_exception,
)


def _build_datasets_doors(
    *,
    settings: DatasetsBuildSettings,
    budget: ShutdownBudget,
    materialise_semaphore: threading.BoundedSemaphore,
    registry: SpecRegistry | None,
    pool_size: int | None,
    spec_reader: SpecReader | None,
    graph: GraphReader | None,
    sdmx: SdmxClient | None,
    revision_reader: RevisionReader | None,
    artifact_store: ArtifactStore | None,
    version_writer: VersionRecordWriter | None,
    head_writer: HeadStore | None,
    mint_pool_size: int | None,
    definition_pool_size: int | None,
    snapshot_reader: SnapshotReader | None,
    reference_writer: PinReferenceWriter | None,
    pin_allowed_principals: frozenset[str] | None,
    publish_allowed_principals: frozenset[str] | None,
    publish_pool_size: int | None,
    recipe_store: RecipeStore | None,
    version_lister: VersionLister | None,
    code_index: CodeIndex | None,
    rate_cache: RateTableCache | None,
    coverage: CoverageService | None,
    operation_store: OperationStore | None,
    operations_pool_size: int | None,
    task_client: TaskClient | None,
    catalogue: CatalogueService,
) -> tuple[DoorContribution, ...]:
    definition_door = build_definition_door(
        settings=settings,
        spec_reader=spec_reader,
        graph=graph,
        sdmx=sdmx,
        revision_reader=revision_reader,
        artifact_store=artifact_store,
        materialiser_build=build_commit(),
        capability_version=DATASETS_CAPABILITY_VERSION,
        # See the mint door's own `ceiling` comment below.
        ceiling=_MATERIALISE_RAW_FETCH_GUARD_ROWS,
        member_concurrency=settings.mint_member_max_concurrency,
        version_writer=version_writer,
        head_store=head_writer,
        snapshot_reader=snapshot_reader,
        materialise_semaphore=materialise_semaphore,
        pool_size=definition_pool_size
        if definition_pool_size is not None
        else settings.definition_max_concurrency,
        budget=budget,
        code_index=code_index,
        rate_cache=rate_cache,
        coverage=coverage,
        operation_store=operation_store,
        task_client=task_client,
        version_lister=version_lister,
        catalogue=catalogue,
    )
    return (
        build_meta_door(
            settings=settings,
            operation_store=operation_store,
            task_client=task_client,
            spec_reader=spec_reader,
            snapshot_reader=snapshot_reader,
            artifact_store=artifact_store,
            pool_size=operations_pool_size,
            budget=budget,
        ),
        build_register_door(registry=registry, pool_size=pool_size, budget=budget),
        build_mint_door(
            settings=settings,
            registry=registry,
            spec_reader=spec_reader,
            graph=graph,
            sdmx=sdmx,
            revision_reader=revision_reader,
            artifact_store=artifact_store,
            materialiser_build=build_commit(),
            capability_version=DATASETS_CAPABILITY_VERSION,
            ceiling=_MATERIALISE_RAW_FETCH_GUARD_ROWS,
            member_concurrency=settings.mint_member_max_concurrency,
            version_writer=version_writer,
            snapshot_reader=snapshot_reader,
            head_writer=head_writer,
            materialise_semaphore=materialise_semaphore,
            pool_size=mint_pool_size
            if mint_pool_size is not None
            else settings.mint_max_concurrency,
            budget=budget,
            code_index=code_index,
            rate_cache=rate_cache,
            coverage=coverage,
            operation_store=operation_store,
            task_client=task_client,
            catalogue=catalogue,
        ),
        definition_door,
        # Runs on the definition door's state, so both share one rebuild per dataset. Mounted
        # after it, so the refresh executor drains before the definition door closes its clients.
        build_refresh_door(retrieval=definition_door.state, settings=settings),
        build_recipe_door(
            settings=settings,
            recipe_store=recipe_store,
            version_lister=version_lister,
            allowed_principals=publish_allowed_principals,
            budget=budget,
        ),
        build_snapshot_door(
            settings=settings,
            spec_reader=spec_reader,
            snapshot_reader=snapshot_reader,
            artifact_store=artifact_store,
            budget=budget,
        ),
        build_observations_door(
            settings=settings,
            spec_reader=spec_reader,
            snapshot_reader=snapshot_reader,
            artifact_store=artifact_store,
            graph=graph,
            code_index=code_index,
            materialise_semaphore=materialise_semaphore,
            catalogue=catalogue,
            budget=budget,
        ),
        build_pin_door(
            settings=settings,
            spec_reader=spec_reader,
            snapshot_reader=snapshot_reader,
            artifact_store=artifact_store,
            reference_writer=reference_writer,
            allowed_principals=pin_allowed_principals,
            budget=budget,
        ),
        build_publish_door(
            settings=settings,
            artifact_store=artifact_store,
            snapshot_reader=snapshot_reader,
            version_writer=version_writer,
            recipe_store=recipe_store,
            allowed_principals=publish_allowed_principals,
            pool_size=publish_pool_size,
            budget=budget,
        ),
    )


def datasets_surface(
    *,
    settings: DatasetsBuildSettings | None = None,
    registry: SpecRegistry | None = None,
    pool_size: int | None = None,
    spec_reader: SpecReader | None = None,
    graph: GraphReader | None = None,
    sdmx: SdmxClient | None = None,
    revision_reader: RevisionReader | None = None,
    artifact_store: ArtifactStore | None = None,
    version_writer: VersionRecordWriter | None = None,
    head_writer: HeadStore | None = None,
    materialise_max_concurrency: int | None = None,
    mint_pool_size: int | None = None,
    definition_pool_size: int | None = None,
    snapshot_reader: SnapshotReader | None = None,
    reference_writer: PinReferenceWriter | None = None,
    pin_allowed_principals: frozenset[str] | None = None,
    publish_allowed_principals: frozenset[str] | None = None,
    publish_pool_size: int | None = None,
    recipe_store: RecipeStore | None = None,
    version_lister: VersionLister | None = None,
    code_index: CodeIndex | None = None,
    rate_cache: RateTableCache | None = None,
    coverage: CoverageService | None = None,
    operation_store: OperationStore | None = None,
    operations_pool_size: int | None = None,
    task_client: TaskClient | None = None,
    catalogue: CatalogueService | None = None,
) -> Surface:
    """Build the Datasets `Surface` from its doors.

    `settings` defaults to `DatasetsBuildSettings.from_env()`. Explicit sizes and concurrency
    arguments override its defaults. `registry` reaches register and mint. Each door builds its own
    registry if omitted. `pool_size` sizes the register executor and semaphore.

    `spec_reader`, `graph`, `sdmx`, `revision_reader`, and `artifact_store` reach the mint and
    definition doors. The snapshot, observations, and pin doors also receive the applicable store
    dependencies, and the observations door `graph` and `code_index`.
    Uninjected live dependencies are built independently in each door's lifespan. `snapshot_reader`
    also reaches definition. `reference_writer` reaches pin.

    `version_writer` reaches mint and publish. Mint builds a live writer when it opens a shared
    store, otherwise it uses a no-op writer. Pass `NullVersionRecordWriter()` to disable writes.
    Publish requires `artifact_store`, `snapshot_reader`, and `version_writer` all injected or all
    omitted. Partial injection raises `DatasetsConfigError`. Omitted values share one pool borrow.
    `head_writer` reaches mint and definition. The shared materialisation semaphore bounds rebuilds
    across both doors. `mint_pool_size` and `definition_pool_size` size independent executors.

    `pin_allowed_principals` and `publish_allowed_principals` set separate allowlists, defaulting
    to `DATASETS_PIN_FIRST_PARTY_PRINCIPAL` and `DATASETS_FIRST_PARTY_PRINCIPAL`. Pinning has no
    delete route, so publish permission grants no pin permission. `publish_pool_size` sizes its
    executor. Its row ceiling is the fixed `PUBLISH_MAX_ROWS`.

    `recipe_store` reaches the publish and recipe doors. `version_lister` reaches the recipe
    door and the definition door. The recipe door admits the publish allowlist and requires
    both injected or both omitted. The frame refresh door runs on the definition door's state
    and admits `settings.live_refresh_principal` alone.
    Inject a recipe store over the store `version_writer` writes to. Omitted, each door builds a
    live one. With the publish door's three store dependencies injected and no `recipe_store`,
    a publish carrying a recipe is refused.

    `code_index` reaches mint and definition. Each uses the process-wide index if omitted.
    `rate_cache` reaches both doors, but each creates a private cache if omitted. Pass one shared
    cache when composing with Search. `coverage` reaches both doors. Each resolves the service at
    build time if omitted, so a missing or unreadable `COVERAGE_ARTIFACT_URI` fails startup.

    `operation_store` reaches meta, mint, and definition. Each builds its own live store if omitted.
    `task_client` reaches those same doors and each builds its own Cloud Tasks client if omitted.
    `operations_pool_size` sizes the meta door's executor and semaphore.

    Each request binds `catalogue`'s loaded snapshot. It defaults to the process-wide service,
    so a missing or unloadable `ORIGIN_CATALOGUE_URI` fails startup. Without an injected `graph`,
    the lifespan acquires the process-wide graph generation that the doors' graph clients key
    their caches on, and registers the default-scope memo on it.
    """
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _catalogue = catalogue if catalogue is not None else get_catalogue_service()

    # One budget bounds teardown across all door lifespans.
    shutdown_budget = ShutdownBudget()

    # Shared by mint and definition to bound concurrent materialisation across both doors.
    materialise_semaphore = threading.BoundedSemaphore(
        materialise_max_concurrency
        if materialise_max_concurrency is not None
        else _settings.materialise_max_concurrency
    )

    doors = _build_datasets_doors(
        settings=_settings,
        budget=shutdown_budget,
        materialise_semaphore=materialise_semaphore,
        registry=registry,
        pool_size=pool_size,
        spec_reader=spec_reader,
        graph=graph,
        sdmx=sdmx,
        revision_reader=revision_reader,
        artifact_store=artifact_store,
        version_writer=version_writer,
        head_writer=head_writer,
        mint_pool_size=mint_pool_size,
        definition_pool_size=definition_pool_size,
        snapshot_reader=snapshot_reader,
        reference_writer=reference_writer,
        pin_allowed_principals=pin_allowed_principals,
        publish_allowed_principals=publish_allowed_principals,
        publish_pool_size=publish_pool_size,
        recipe_store=recipe_store,
        version_lister=version_lister,
        code_index=code_index,
        rate_cache=rate_cache,
        coverage=coverage,
        operation_store=operation_store,
        operations_pool_size=operations_pool_size,
        task_client=task_client,
        catalogue=_catalogue,
    )

    router = flatten(doors)

    # Append grammar refusals after real routes. The converter excludes valid ids.
    router.routes.extend(build_id_grammar_refusal_router().routes)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Validate required settings once at startup before requests run.
        live_settings = DatasetsSettings.from_env()
        async with AsyncExitStack() as stack:
            if graph is None:
                # The doors' live graph clients key their caches on this provider.
                generation = await enter_shared_generation(
                    stack, catalogue=_catalogue, graph_base_url=live_settings.graph_base_url
                )
                register_place_scope_memo(stack, generation, name=PLACE_SCOPE_MEMO_NAME)
            for door in doors:
                if door.lifespan is not None:
                    await stack.enter_async_context(door.lifespan(app))
            yield

    return Surface(
        name="datasets",
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
        exception_types=tuple(exc_type for door in doors for exc_type in door.exception_types),
        exception_exemplars=tuple(
            exemplar for door in doors for exemplar in door.exception_exemplars
        ),
        lifespan=lifespan,
        middleware=tuple(check for door in doors for check in door.middleware),
        request_scope=lambda: bound_catalogue(_catalogue.snapshot()),
    )
