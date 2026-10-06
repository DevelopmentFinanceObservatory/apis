"""Dependencies shared by the mint door and its kind resolvers.

The mutable bundle receives live implementations during the door's lifespan.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from artifact_store import ArtifactStore
from code_index.query import CodeIndex
from dataset_store.snapshot import SnapshotReader
from datasets_api.config import DatasetsBuildSettings
from datasets_api.operation_store import NullOperationStore, OperationStore
from datasets_api.operations.task_client import NullTaskClient, TaskClient
from datasets_api.serve import ServeDeps
from datasets_api.versions import (
    HeadStore,
    NullHeadWriter,
    NullVersionRecordWriter,
    VersionRecordWriter,
)
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from materialise.revision import RevisionReader
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
    revision_reader: RevisionReader
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
    # The serve path reads this before materialisation.
    # It writes the head when a resolver supplies a signal.
    head_store: HeadStore = field(default_factory=NullHeadWriter)
    # `None` means materialisation concurrency is unbounded.
    materialise_semaphore: threading.BoundedSemaphore | None = None
    # Absolute monotonic deadline for this request's SDMX work. The door sets
    # it on the per-request dependency copy.
    deadline: float | None = None
    # Over-budget submissions use these clients. Inline requests keep the Null defaults.
    operation_store: OperationStore = field(default_factory=NullOperationStore)
    task_client: TaskClient = field(default_factory=NullTaskClient)
    # Shared limits for admission, async ceilings, and serving. Resolvers read them here.
    settings: DatasetsBuildSettings = field(default_factory=DatasetsBuildSettings.from_env)

    def to_serve_deps(self) -> ServeDeps:
        """Builds `ServeDeps` with the serving fields from this bundle."""
        return ServeDeps(
            sdmx=self.sdmx,
            graph=self.graph,
            revision_reader=self.revision_reader,
            artifact_store=self.artifact_store,
            materialiser_build=self.materialiser_build,
            capability_version=self.capability_version,
            ceiling=self.ceiling,
            member_concurrency=self.member_concurrency,
            enrichment=self.enrichment,
            code_index=self.code_index,
            rate_cache=self.rate_cache,
            version_writer=self.version_writer,
            snapshot_reader=self.snapshot_reader,
            head_store=self.head_store,
            materialise_semaphore=self.materialise_semaphore,
            deadline=self.deadline,
            settings=self.settings,
        )
