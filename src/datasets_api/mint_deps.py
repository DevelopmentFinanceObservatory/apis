"""Dependencies shared by the mint door and its kind resolvers.

The mint door builds one per request from the surface's `Deps` (`datasets_api.doors.mint`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

from artifact_store import ArtifactStore
from code_index.query import CodeIndex
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import SnapshotReader
from datasets_api.admission import MaterialisePermits, RebuildFlights
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import DefinitionDeps
from datasets_api.operation_store import NullOperationStore, OperationStore
from datasets_api.operations.task_client import NullTaskClient, TaskClient
from datasets_api.versions import (
    HeadStore,
    NullHeadWriter,
    NullVersionRecordWriter,
    VersionRecordWriter,
)
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup
from materialise.facets import VariableBreadthReader
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from series_coverage.shared import CoverageService
from spec_registry import SpecRegistry
from spec_registry.reader import SpecReader
from statvar_members import MemberEnumerator


@dataclass
class MintDeps:
    registry: SpecRegistry
    spec_reader: SpecReader
    sdmx: SdmxClient
    graph: GraphReader
    artifact_store: ArtifactStore
    materialiser_build: str
    capability_version: str
    ceiling: int
    member_concurrency: int
    # The process's shared enumerator (`statvar_members.shared_enumerator`), which holds no
    # graph. Direct kind resolvers use it.
    enumerator: MemberEnumerator
    # Shares `graph` and caches country-code and label lookups for the process lifetime.
    enrichment: EnrichmentLookup
    # The process-wide loaded index decodes StatVar-declared code systems during rendering.
    code_index: CodeIndex
    # One process-wide cache shared by materialisations.
    rate_cache: RateTableCache
    # Process-wide service used by admission before SDMX calls. It stays outside `ServeDeps`.
    coverage: CoverageService
    version_writer: VersionRecordWriter = field(default_factory=NullVersionRecordWriter)
    # Passed to `datasets_api.serve.ServeDeps.snapshot_reader`.
    snapshot_reader: SnapshotReader | None = None
    # The serve path reads this before materialisation and advances it after each rebuild.
    head_store: HeadStore = field(default_factory=NullHeadWriter)
    # `None` means materialisation concurrency is unbounded.
    materialise_semaphore: MaterialisePermits | None = None
    # Absolute monotonic deadline for this request's SDMX work. The door sets
    # it on the per-request dependency copy.
    deadline: float | None = None
    # Over-budget submissions use these clients. Inline requests keep the Null defaults.
    operation_store: OperationStore = field(default_factory=NullOperationStore)
    task_client: TaskClient = field(default_factory=NullTaskClient)
    # Shared limits for admission, async ceilings, and serving. Resolvers read them here.
    settings: DatasetsBuildSettings = field(default_factory=DatasetsBuildSettings.from_env)
    # The currency clock, and the surface's one rebuild per subject at a time, shared with the
    # definition and refresh doors.
    now: Clock = utc_now
    rebuilds: RebuildFlights = field(default_factory=RebuildFlights)
    # Bound around each materialisation, so a mint binding one place breaks a facet tie on each
    # facet's places across the whole variable. None leaves the tie to the unit's name.
    variable_breadth: VariableBreadthReader | None = None

    def to_definition_deps(self) -> DefinitionDeps:
        """The `current.serve_current` view of this bundle. The head recheck interval and
        maximum age come from `settings`."""
        return DefinitionDeps(
            registry=self.registry,
            spec_reader=self.spec_reader,
            graph=self.graph,
            sdmx=self.sdmx,
            artifact_store=self.artifact_store,
            materialiser_build=self.materialiser_build,
            capability_version=self.capability_version,
            ceiling=self.ceiling,
            member_concurrency=self.member_concurrency,
            enumerator=self.enumerator,
            enrichment=self.enrichment,
            code_index=self.code_index,
            rate_cache=self.rate_cache,
            coverage=self.coverage,
            version_writer=self.version_writer,
            head_store=self.head_store,
            snapshot_reader=cast(SnapshotReader, self.snapshot_reader),
            materialise_semaphore=self.materialise_semaphore,
            head_recheck_interval=timedelta(seconds=self.settings.head_recheck_seconds),
            head_max_age=timedelta(seconds=self.settings.head_max_age_seconds),
            now=self.now,
            deadline=self.deadline,
            operation_store=self.operation_store,
            task_client=self.task_client,
            settings=self.settings,
            rebuilds=self.rebuilds,
            variable_breadth=self.variable_breadth,
        )
