"""Build datasets API apps and dependencies for tests.
`make_app` uses the platform app factory; `mint_deps` supports unit tests that bypass HTTP.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dataset_store.clock import Clock
from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.mint_deps import MintDeps
from datasets_api.operation_store import NullOperationStore, OperationStore
from datasets_api.operations.task_client import FakeTaskClient, TaskClient
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister
from datasets_api.versions import NullHeadWriter, NullVersionRecordWriter
from dfo_platform.app import create_app
from materialise.rate_tables import RateTableCache
from origin_catalogue.service import CatalogueService
from spec_registry import SpecRegistry
from spec_registry.reader import SpecReader

from ._deps_doubles import Unwired
from ._snapshot_doubles import FakeSnapshotReader
from ._version_record_store import FakeVersionRecordDatasetStore


def offline_recipe_stores() -> dict[str, object]:
    """Return `datasets_surface`'s `recipe_store`, `version_lister` and `description_store`
    keywords over one empty SQLite-backed store, so the lifespan opens no pool.
    """
    store = FakeVersionRecordDatasetStore()
    return {
        "recipe_store": LiveRecipeStore(store=store),
        "version_lister": LiveVersionLister(store=store),
        "description_store": LiveDescriptionStore(store=store),
    }


def make_app(
    *,
    registry: SpecRegistry,
    settings: DatasetsBuildSettings | None = None,
    work_max_concurrency: int | None = None,
    read_max_concurrency: int | None = None,
    materialise_max_concurrency: int | None = None,
    spec_reader: SpecReader | None = None,
    graph=None,
    sdmx=None,
    artifact_store=None,
    version_writer=None,
    head_writer=None,
    snapshot_reader=None,
    reference_writer=None,
    pin_allowed_principals: frozenset[str] | None = None,
    publish_allowed_principals: frozenset[str] | None = None,
    recipe_store=None,
    version_lister=None,
    description_store=None,
    code_index=None,
    rate_cache: RateTableCache | None = None,
    coverage=None,
    operation_store: OperationStore | None = None,
    task_client: TaskClient | None = None,
    catalogue: CatalogueService | None = None,
    now: Clock | None = None,
) -> FastAPI:
    """Compose an app from injected dependencies, with offline stand-ins for the rest, so its
    lifespan builds no live client and opens no pool.

    Omitted writers, the head store and the operation store are the null implementations, an
    omitted snapshot reader reads nothing, and `task_client` defaults to `FakeTaskClient`.
    `recipe_store` and `version_lister` default to `offline_recipe_stores()`. A recipe test
    injects both over the store its version writer shares. `description_store` defaults to one
    over an empty SQLite-backed store. Every other omitted client is
    `Unwired`. Publish verification under the null writer raises `PublishRecordMissing` (500),
    so a successful publish test injects a writer sharing the snapshot reader's store.
    The concurrency keywords replace the matching `settings` fields. `catalogue` replaces the
    process-wide catalogue service the request scope binds.
    """
    resolved = settings if settings is not None else DatasetsBuildSettings.from_env()
    sizes = {
        name: value
        for name, value in (
            ("work_max_concurrency", work_max_concurrency),
            ("read_max_concurrency", read_max_concurrency),
            ("materialise_max_concurrency", materialise_max_concurrency),
        )
        if value is not None
    }
    if sizes:
        resolved = resolved.model_copy(update=sizes)
    recipe_stores = (
        offline_recipe_stores()
        if recipe_store is None and version_lister is None
        else {"recipe_store": recipe_store, "version_lister": version_lister}
    )
    if description_store is not None or "description_store" not in recipe_stores:
        recipe_stores["description_store"] = (
            description_store
            if description_store is not None
            else LiveDescriptionStore(store=FakeVersionRecordDatasetStore())
        )
    kwargs: dict[str, object] = {
        "settings": resolved,
        "registry": registry,
        "spec_reader": spec_reader if spec_reader is not None else Unwired("spec_reader"),
        "graph": graph if graph is not None else Unwired("graph"),
        "sdmx": sdmx if sdmx is not None else Unwired("sdmx"),
        "artifact_store": artifact_store
        if artifact_store is not None
        else Unwired("artifact_store"),
        "snapshot_reader": snapshot_reader if snapshot_reader is not None else FakeSnapshotReader(),
        "version_writer": version_writer
        if version_writer is not None
        else NullVersionRecordWriter(),
        "head_writer": head_writer if head_writer is not None else NullHeadWriter(),
        "operation_store": operation_store if operation_store is not None else NullOperationStore(),
        "reference_writer": (
            reference_writer if reference_writer is not None else Unwired("reference_writer")
        ),
        "task_client": task_client if task_client is not None else FakeTaskClient(),
        **recipe_stores,
    }
    optional = (
        ("pin_allowed_principals", pin_allowed_principals),
        ("publish_allowed_principals", publish_allowed_principals),
        ("code_index", code_index),
        ("rate_cache", rate_cache),
        ("coverage", coverage),
        ("catalogue", catalogue),
        ("now", now),
    )
    kwargs.update((name, value) for name, value in optional if value is not None)
    return create_app([datasets_surface(**kwargs)])


class RunningClient(TestClient):
    """A `TestClient` whose app lifespan is already running. `with client:` adds nothing on
    entry and stops the lifespan on exit; the root conftest stops any still running after the
    test."""

    def start(self) -> RunningClient:
        TestClient.__enter__(self)
        _running_clients.append(self)
        return self

    def __enter__(self) -> RunningClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self in _running_clients:
            _running_clients.remove(self)
            TestClient.__exit__(self, *exc_info)


_running_clients: list[RunningClient] = []


def make_client(**kwargs) -> RunningClient:
    """A client over `make_app(**kwargs)` with the surface's lifespan running, which builds the
    doors' `Deps`."""
    return RunningClient(make_app(**kwargs), raise_server_exceptions=False).start()


def close_open_clients() -> None:
    """Stop every lifespan `make_client` started and no test stopped, newest first."""
    while _running_clients:
        _running_clients[-1].__exit__(None, None, None)


def mint_deps(
    *,
    registry: SpecRegistry,
    spec_reader: SpecReader,
    graph,
    sdmx,
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

    index = code_index if code_index is not None else get_index()
    return MintDeps(
        registry=registry,
        spec_reader=spec_reader,
        sdmx=sdmx,
        graph=graph,
        artifact_store=artifact_store,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        ceiling=ceiling,
        member_concurrency=member_concurrency,
        enumerator=enumerator if enumerator is not None else MemberEnumerator(),
        enrichment=(
            enrichment
            if enrichment is not None
            else EnrichmentLookup(graph, parties=lambda: index.parties)
        ),
        code_index=index,
        rate_cache=rate_cache if rate_cache is not None else RateTableCache(),
        coverage=coverage if coverage is not None else get_coverage_service(),
        version_writer=version_writer if version_writer is not None else NullVersionRecordWriter(),
    )
