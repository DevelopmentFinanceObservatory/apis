"""Offline `Deps` for tests that build one door, and the stand-in for a dependency a test leaves
unwired."""

from __future__ import annotations

from dataclasses import fields

from code_index.shared import get_index
from dataset_store.clock import utc_now
from datasets_api.admission import MaterialisePermits, RebuildFlights
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import Deps, DepsCell, build_executors
from datasets_api.operation_store import NullOperationStore
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.versions import NullHeadWriter
from materialise import EnrichmentLookup
from materialise.rate_tables import RateTableCache
from series_coverage.shared import get_coverage_service

from ._snapshot_doubles import offline_version_pair


class Unwired:
    """A dependency the test did not wire. Any attribute read fails naming it."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attribute: str) -> object:
        raise AssertionError(f"the test wired no {self._name}, and the code read .{attribute}")


def fake_deps(**overrides: object) -> Deps:
    """A `Deps` of offline doubles. The head store, the operation store and the other writers
    are the null implementations, the version writer and snapshot reader follow
    `offline_version_pair`, the task client is a `FakeTaskClient`, and every other client or
    store is `Unwired` unless `overrides` names it. A `code_index` override serves every read,
    and without one each read takes `get_index()`. No breadth reader is bound, `now` defaults to
    `utc_now`, and the executors and semaphore follow `settings`."""
    settings = overrides.pop("settings", None) or DatasetsBuildSettings.from_env()
    assert isinstance(settings, DatasetsBuildSettings)
    read_executor, work_executor = build_executors(settings)
    given_index = overrides.pop("code_index", None)
    index_source = get_index if given_index is None else lambda: given_index
    graph = overrides.get("graph", Unwired("graph"))
    defaults: dict[str, object] = {
        "settings": settings,
        "head_store": NullHeadWriter(),
        "operation_store": NullOperationStore(),
        "task_client": FakeTaskClient(),
        "coverage": get_coverage_service(),
        "rate_cache": RateTableCache(),
        "code_index_source": index_source,
        "enrichment": EnrichmentLookup(graph, parties=lambda: index_source().parties),
        "materialiser_build": "test-build",
        "now": utc_now,
        "variable_breadth": None,
        "materialise_semaphore": MaterialisePermits(settings.materialise_max_concurrency),
        "rebuilds": RebuildFlights(),
        "read_executor": read_executor,
        "work_executor": work_executor,
    }
    defaults.update(
        offline_version_pair(
            version_writer=overrides.pop("version_writer", None),
            snapshot_reader=overrides.pop("snapshot_reader", None),
        )
    )
    values = {
        field.name: overrides.pop(field.name, defaults.get(field.name, Unwired(field.name)))
        for field in fields(Deps)
    }
    assert not overrides, f"Deps has no field(s) {sorted(overrides)}"
    return Deps(**values)


def fake_deps_cell(**overrides: object) -> DepsCell:
    """A `DepsCell` already holding `fake_deps(**overrides)`."""
    return DepsCell(fake_deps(**overrides))
