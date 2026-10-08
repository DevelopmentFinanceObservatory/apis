"""Build datasets API apps and dependencies for tests.
`make_app` uses the platform app factory; `mint_deps` supports unit tests that bypass HTTP.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.mint_deps import MintDeps
from datasets_api.operation_store import OperationStore
from datasets_api.operations.task_client import FakeTaskClient, TaskClient
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister
from datasets_api.versions import NullVersionRecordWriter
from dfo_platform.app import create_app
from materialise.rate_tables import RateTableCache
from materialise.revision import RevisionReader
from origin_catalogue.service import CatalogueService
from spec_registry import SpecRegistry
from spec_registry.reader import SpecReader

from ._version_record_store import FakeVersionRecordDatasetStore


def offline_recipe_stores() -> dict[str, object]:
    """Return `datasets_surface`'s `recipe_store` and `version_lister` keywords over one empty
    SQLite-backed store, so the recipe door's lifespan opens no pool.
    """
    store = FakeVersionRecordDatasetStore()
    return {
        "recipe_store": LiveRecipeStore(store=store),
        "version_lister": LiveVersionLister(store=store),
    }


def make_app(
    *,
    registry: SpecRegistry,
    settings: DatasetsBuildSettings | None = None,
    pool_size: int | None = None,
    mint_pool_size: int | None = None,
    definition_pool_size: int | None = None,
    spec_reader: SpecReader | None = None,
    graph=None,
    sdmx=None,
    revision_reader: RevisionReader | None = None,
    artifact_store=None,
    version_writer=None,
    head_writer=None,
    materialise_max_concurrency: int | None = None,
    snapshot_reader=None,
    reference_writer=None,
    pin_allowed_principals: frozenset[str] | None = None,
    publish_allowed_principals: frozenset[str] | None = None,
    recipe_store=None,
    version_lister=None,
    code_index=None,
    rate_cache: RateTableCache | None = None,
    coverage=None,
    operation_store: OperationStore | None = None,
    task_client: TaskClient | None = None,
    catalogue: CatalogueService | None = None,
) -> FastAPI:
    """Compose an app from injected dependencies.

    With `artifact_store` and `snapshot_reader` but no `version_writer`, the factory adds a
    `NullVersionRecordWriter` so the surface builds. Publish verification then raises
    `PublishRecordMissing` (500). Successful publish tests must inject a writer that shares the
    snapshot reader's store.

    `task_client` defaults to `FakeTaskClient`, avoiding live task-client construction.
    `recipe_store` and `version_lister` default to `offline_recipe_stores()`. A recipe test
    injects both over the store its version writer shares.
    `catalogue` replaces the process-wide catalogue service the request scope binds.
    """
    kwargs: dict[str, object] = {"registry": registry, "pool_size": pool_size}
    if settings is not None:
        kwargs["settings"] = settings
    if mint_pool_size is not None:
        kwargs["mint_pool_size"] = mint_pool_size
    if definition_pool_size is not None:
        kwargs["definition_pool_size"] = definition_pool_size
    if spec_reader is not None:
        kwargs["spec_reader"] = spec_reader
    if graph is not None:
        kwargs["graph"] = graph
    if sdmx is not None:
        kwargs["sdmx"] = sdmx
    if revision_reader is not None:
        kwargs["revision_reader"] = revision_reader
    if artifact_store is not None:
        kwargs["artifact_store"] = artifact_store
    if version_writer is not None:
        kwargs["version_writer"] = version_writer
    elif artifact_store is not None and snapshot_reader is not None:
        kwargs["version_writer"] = NullVersionRecordWriter()
    if head_writer is not None:
        kwargs["head_writer"] = head_writer
    if materialise_max_concurrency is not None:
        kwargs["materialise_max_concurrency"] = materialise_max_concurrency
    if snapshot_reader is not None:
        kwargs["snapshot_reader"] = snapshot_reader
    if reference_writer is not None:
        kwargs["reference_writer"] = reference_writer
    if pin_allowed_principals is not None:
        kwargs["pin_allowed_principals"] = pin_allowed_principals
    if publish_allowed_principals is not None:
        kwargs["publish_allowed_principals"] = publish_allowed_principals
    if recipe_store is None and version_lister is None:
        kwargs.update(offline_recipe_stores())
    else:
        kwargs["recipe_store"] = recipe_store
        kwargs["version_lister"] = version_lister
    if code_index is not None:
        kwargs["code_index"] = code_index
    if rate_cache is not None:
        kwargs["rate_cache"] = rate_cache
    if coverage is not None:
        kwargs["coverage"] = coverage
    if operation_store is not None:
        kwargs["operation_store"] = operation_store
    if catalogue is not None:
        kwargs["catalogue"] = catalogue
    kwargs["task_client"] = task_client if task_client is not None else FakeTaskClient()
    return create_app([datasets_surface(**kwargs)])


def make_client(
    *,
    registry: SpecRegistry,
    pool_size: int | None = None,
    mint_pool_size: int | None = None,
    definition_pool_size: int | None = None,
    **kwargs,
) -> TestClient:
    return TestClient(
        make_app(
            registry=registry,
            pool_size=pool_size,
            mint_pool_size=mint_pool_size,
            definition_pool_size=definition_pool_size,
            **kwargs,
        ),
        raise_server_exceptions=False,
    )


def mint_deps(
    *,
    registry: SpecRegistry,
    spec_reader: SpecReader,
    graph,
    sdmx,
    revision_reader: RevisionReader,
    artifact_store,
    materialiser_build: str = "test-build",
    capability_version: str = "dset_test",
    ceiling: int = 50_000,
    member_concurrency: int = 4,
    version_writer=None,
    enumerator=None,
    enrichment=None,
    code_index=None,
    rate_cache=None,
    coverage=None,
) -> MintDeps:
    """Build `MintDeps` for unit tests that bypass HTTP.

    `enumerator` and `enrichment` wrap `graph`. Code index and coverage use shared services.
    Each call creates a rate cache and defaults to a null version writer. Coverage uses the recorded
    one-StatVar slice. Inject matching coverage for fake StatVars outside the slice.
    """
    from code_index.shared import get_index
    from datasets_api.versions import NullVersionRecordWriter
    from materialise import EnrichmentLookup
    from materialise.rate_tables import RateTableCache
    from series_coverage.shared import get_coverage_service
    from statvar_members import MemberEnumerator

    return MintDeps(
        registry=registry,
        spec_reader=spec_reader,
        sdmx=sdmx,
        graph=graph,
        revision_reader=revision_reader,
        artifact_store=artifact_store,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        ceiling=ceiling,
        member_concurrency=member_concurrency,
        enumerator=enumerator if enumerator is not None else MemberEnumerator(),
        enrichment=enrichment if enrichment is not None else EnrichmentLookup(graph),
        code_index=code_index if code_index is not None else get_index(),
        rate_cache=rate_cache if rate_cache is not None else RateTableCache(),
        coverage=coverage if coverage is not None else get_coverage_service(),
        version_writer=version_writer if version_writer is not None else NullVersionRecordWriter(),
    )
