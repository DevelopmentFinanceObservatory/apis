"""`serve_definition` and `serve_collection` run synchronously inside the calling door's executor.
Each prewrite half materialises resolved requests and builds a manifest with an empty artifact list.
Each write half reshapes, stores artifacts, and records the version. The definition options route
uses only the prewrite half. Every materialising door must reuse this path.

Collection members materialise across `member_concurrency` workers. The materialise timing covers
the full fan-out. Both prewrite functions record output-profile CPU in a separate `Server-Timing`
phase.

`serve_definition` streams unsigned frame1 from every frame chunk beside the requested rendering on
a worker, then joins it before recording the version. Frame1 is written only for results of at most
`FRAME1_MAX_ROWS` (250,000) rows; a larger version mints without it, and presentation answers
`missing_rendering`. A failed frame1 write is logged, does not fail the mint, and leaves
presentation unavailable until a later mint. Collections store frame1 and version records per
member because each member has its own frame.

A definition write holds up to two `dataset_store` borrows, one per artifact transaction. At four
concurrent calls each, mint and definition retrieval can need 16 borrows. Presentation frames and
components can add eight. The pool defaults to 10 connections and a 30-second borrow timeout.
Collection member persistence uses up to `member_concurrency` borrows, plus one for its requested
artifact. The semaphore bounds rematerialisation. The pool bounds borrows. Presentation has a
separate pool, while its frame and component reads share `dataset_store`. Frame1 and the requested
rendering overlap. The `store` timing includes both, and frame1 encoding runs inside it. A
primary-write failure waits for the frame1 worker to finish.

If frame1 succeeds before the requested rendering fails, its inventory object lacks a version link,
so snapshot reads ignore it. A later mint links its digest in `version_artifact`.

A supplied `signal` advances the head after the version record commits. The caller reads the head
before materialising and passes its `content_version`, or `None`, for compare-and-set. Mint priming
and currency checks share `datasets_api.currency.derive_signals`.
"""

from __future__ import annotations

import contextvars
import logging
import math
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import cast

from artifact_store import ArtifactStore, ArtifactStoreError
from code_index.query import CodeIndex
from dataset_store.clock import utc_now
from dataset_store.snapshot import SnapshotReader, select_servable_locations
from datasets_api.admission import bounded_materialise_slot
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest import (
    assemble_collection_manifest,
    assemble_manifest,
    coverage_facts_of,
    encode_version_body,
    healed_citation,
    persisted_manifest,
    record_collection_version,
    record_definition_version,
    reshape_warning_models,
)
from datasets_api.manifest_models import (
    ArtifactEntryModel,
    CitationModel,
    CitationSource,
    CollectionModel,
    Manifest,
    WarningModel,
)
from datasets_api.mint_models import MintOptions
from datasets_api.origin import (
    origin_reads,
    un_unavailable_warnings,
    withheld_warning,
    withheld_warnings,
)
from datasets_api.query_options import encode_query_options
from datasets_api.render import (
    FRAME1_MAX_ROWS,
    CollectionMemberRendering,
    artifact_rendering,
    cited_source_names,
    metadata_deadline,
    render_and_store,
    render_and_store_collection,
    render_and_store_frame1,
    render_and_store_series_statistics,
    sign_stored_artifact,
)
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import phase, record_elapsed
from datasets_api.version_store import rendering_key
from datasets_api.versions import (
    ArtifactRecord,
    HeadStore,
    MemberSignal,
    NullHeadWriter,
    NullVersionRecordWriter,
    VersionRecord,
    VersionRecordWriter,
)
from dc_sdmx import SdmxClient
from dfo_platform.config import PublicSiteSettings
from materialise import (
    CollectionFacetProbe,
    CollectionMaterialisationResult,
    EnrichmentLookup,
    MaterialisationRequest,
    MaterialisationResult,
    RowProgress,
    materialise_collection,
    materialise_group,
)
from materialise.bundle import ReproducingCall
from materialise.content_version import MintPreimage, recipe_dependency_fingerprint
from materialise.convert import ConversionRecord
from materialise.graph import GraphReader
from materialise.provenance_doc import (
    SourceCitation,
    read_source_citations,
)
from materialise.rate_tables import RateTableCache
from materialise.revision import RevisionReader
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from origin_catalogue import current_catalogue
from series_coverage.index import CoverageIndex, Estimate
from spec_registry.reader import StoredCollection, StoredDefinition

logger = logging.getLogger(__name__)

DEFINITION_ROUTE_LIVE = True
"""Whether the definition retrieval route (`GET /api/datasets/{id}`) is
mounted on this deployment. A bundle's reproducing call is only published
once that route exists to serve it. `datasets_api.doors.definition`
mounts it unconditionally, so this is `True` for every deployment that
composes `datasets_api.app.datasets_surface`."""

# Logs frame1 store failures with `spec_id` and `content_version`. The requested mint succeeds.
# Keep the marker stable for alerts. Later mints can restore frame1.
FRAME1_STORE_FAILED_MARKER = "frame1_store_failed"

# Logs member version-record failures with `spec_id` and `content_version`. The collection mint
# succeeds, and a later member mint can restore the missing record.
MEMBER_VERSION_RECORD_FAILED_MARKER = "member_version_record_failed"

# Logs a bundle returned but left unlinked because a graph metadata read failed, with the subject,
# `content_version` and cited refs. The next render of that version reads again and links on
# success.
BUNDLE_NOT_LINKED_MARKER = "bundle_not_linked"

_RAW_FETCH_GUARD_MARGIN = 1.5
"""The 1.5 margin covers rows added between the exact coverage snapshot and live SDMX fetch.
Coverage is exact for covered StatVars and overestimates `latest` windows or `ContainedIn`-bounded
roles. Admission runs before fetch. The 50% mid-fetch backstop targets genuine response growth.
"""


def raw_fetch_guard(estimate: Estimate, *, floor: int) -> int:
    """Returns `max(floor, ceil(estimate.rows * _RAW_FETCH_GUARD_MARGIN))` for the materialiser's
    mid-fetch row guard and post-fetch ceiling. `floor` covers uncovered requests and calls lacking
    an estimate. A zero estimate can mean missing coverage or a covered request with no data.
    Definition option rerenders use the floor: `refuse_inline_over_ceiling` admits only an estimate
    at or below it."""
    return max(floor, math.ceil(estimate.rows * _RAW_FETCH_GUARD_MARGIN))


def _provenance_ref(result: MaterialisationResult) -> str:
    """The `provenance` axis's own dispatched ref off an already-finished
    materialisation's trace, read back without a second probe.
    """
    return cast(str, next(f["ref"] for f in result.trace.facets if f["axis"] == "provenance"))


def refs_from_resolved_stat_vars(entries: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    """Extracts refs from live or stored `resolved_stat_vars` trace entries. Both materialised and
    stored-rendering paths use this function."""
    return tuple(cast(str, entry["ref"]) for entry in entries)


def _resolved_stat_var_refs(result: MaterialisationResult) -> tuple[str, ...]:
    """Returns every resolved `dcid:` StatVar, including zero-row members, from the materialisation
    trace. Groupby uses these refs for population checks."""
    return refs_from_resolved_stat_vars(result.trace.resolved_stat_vars)


def _groupby_collapse_preimage(result: MaterialisationResult) -> MintPreimage:
    """Returns the exact preimage used by the base `mint_version` call. Trace facets use a 7-key
    wire shape. The version preimage uses grammar facets `{axis, ref, basis}`, including the
    converted-unit basis and ref. `_reshape` reads it only for full collapses."""
    return result.mint_preimage


def _rate_provenance_refresh(
    records: Iterable[ConversionRecord | None],
) -> tuple[tuple[str, str | None], ...]:
    """Returns sorted, deduplicated `(table_provenance_id, table_refresh_date)` pairs.
    `table_refresh_date` is the `lastDataRefreshDate` each member's tables were admitted under,
    which `currency.recheck_rate_provenance` compares with the live date. The table digests stay
    in the content version. Group members share table provenance, so their records collapse to one pair.
    Returns `()` when no conversion record exists."""
    pairs: dict[str, str | None] = {}
    for record in records:
        if record is not None:
            pairs[record.table_provenance_id] = record.table_refresh_date
    return tuple(sorted(pairs.items()))


def _recipe_dependency_fingerprint(
    records: Iterable[ConversionRecord | None],
) -> tuple[str, str] | None:
    """Returns the group's `(recipe_id, digest)` fingerprint from its first conversion record.
    Group members share one recipe. Returns `None` when no conversion record exists."""
    for record in records:
        if record is not None:
            return recipe_dependency_fingerprint(record.recipe_id)
    return None


def _signal_with_rate_provenance(
    signal: Sequence[MemberSignal], records: Iterable[ConversionRecord | None]
) -> tuple[MemberSignal, ...]:
    """Adds rate-provenance refresh and recipe fingerprint fields to each signal from this
    materialisation's conversion records. It reads `records` into a tuple because both helpers
    iterate it. Probe-derived signals lack conversion data. This function adds the fields before
    `HeadWriter.advance_head`."""
    records = tuple(records)
    refresh = _rate_provenance_refresh(records)
    fingerprint = _recipe_dependency_fingerprint(records)
    return tuple(
        replace(member, rate_provenance_refresh=refresh, recipe_dependency_fingerprint=fingerprint)
        for member in signal
    )


@dataclass(frozen=True)
class ServeDeps:
    sdmx: SdmxClient
    graph: GraphReader
    revision_reader: RevisionReader
    artifact_store: ArtifactStore
    materialiser_build: str
    capability_version: str
    # Default mid-fetch row guard. `raw_fetch_guard` scales it to each coverage estimate.
    ceiling: int
    member_concurrency: int
    enrichment: EnrichmentLookup
    # Process-wide index used by rendering and manifest assembly.
    code_index: CodeIndex
    # One process-wide rate-table cache shared by materialisations.
    rate_cache: RateTableCache
    version_writer: VersionRecordWriter = field(default_factory=NullVersionRecordWriter)
    # Reads the version record `version_writer` persists. A write reads it before rendering, so a
    # re-mint renders its bundle from the stored citation, and again after its record insert finds
    # the record already there, so the render dates and cites as the stored record does
    # (`_persisted_body`). `None` reads nothing.
    snapshot_reader: SnapshotReader | None = None
    # Callers read before materialising and advance after writes when given a signal. The default
    # writer reads nothing and treats writes as no-ops.
    head_store: HeadStore = field(default_factory=NullHeadWriter)
    # Bounds rematerialisation across doors sharing the semaphore. Mint and definition share one.
    # `None` disables the bound.
    materialise_semaphore: threading.BoundedSemaphore | None = None
    # Absolute monotonic deadline for queue waits, SDMX fetches, and the citation read before the
    # write. A bundle's two metadata reads each run until this deadline or
    # `settings.metadata_read_seconds` from their start, whichever ends later: the documentation
    # read (at a version's first mint, or the live read that fills a stored citation's empty
    # fields) and the code label read. Member persistence logs if late but attempts every write.
    # `None` means no deadline.
    deadline: float | None = None
    # Supplies queue-wait and pivot-cell caps plus capability limits embedded in manifests. The
    # The application passes it explicitly. The default serves direct construction.
    settings: DatasetsBuildSettings = field(default_factory=DatasetsBuildSettings.from_env)


def _persisted_body(
    snapshot_reader: SnapshotReader | None, *, subject_id: str, content_version: str
) -> Mapping[str, object] | None:
    """The stored `immutable_body` of the version record, or `None` if absent or unreadable."""
    if snapshot_reader is None:
        return None
    with phase("snapshot_read"):
        stored = snapshot_reader.read_version(subject_id, content_version)
    return stored.immutable_body if stored is not None else None


def _render_citation(
    persisted_body: Mapping[str, object] | None, *, assembled: CitationModel
) -> CitationModel:
    """The stored record's citation when the version was already on record, else `assembled`.

    A re-mint then cites, names and dates its bundle as the first mint did.
    """
    if persisted_body is None:
        return assembled
    return CitationModel.model_validate(persisted_body["citation"])


def _render_prepared_at(
    persisted_body: Mapping[str, object] | None, *, assembled: datetime
) -> datetime:
    """The stored record's `prepared_at` when the version was already on record, else
    `assembled`, as `_render_citation` chooses the citation."""
    if persisted_body is None:
        return assembled
    return datetime.fromisoformat(cast(str, persisted_body["prepared_at"]))


def _render_member_refs(
    persisted_body: Mapping[str, object] | None,
    *,
    per_member: Sequence[tuple[StoredDefinition, MaterialisationResult]],
) -> tuple[tuple[str, ...], ...]:
    """Each collection member's cited refs for a render, parallel to `per_member`: the stored
    record's member `provenance_refs` when the version was already on record, else each
    member's live provenance ref, as `_render_citation` chooses the citation.

    A snapshot read names the collection bundle, its member files and its bare member artifacts
    from the stored entries, so a render of a version on record names them the same way. They
    feed naming only. The reshape reads each member's live ref.
    """
    if persisted_body is None:
        return tuple((_provenance_ref(result),) for _stored, result in per_member)
    collection = cast("Mapping[str, list[Mapping[str, list[str]]]]", persisted_body["collection"])
    return tuple(tuple(member["provenance_refs"]) for member in collection["members"])


def _render_member_prepared_at(
    persisted_body: Mapping[str, object] | None, *, assembled: Sequence[datetime]
) -> tuple[datetime, ...]:
    """Each collection member's `prepared_at` for a render, parallel to `assembled`: the stored
    record's member entries when the version was already on record, else `assembled`, as
    `_render_member_refs` chooses the refs. A snapshot read dates a bare member artifact from
    the stored entry, so a render of a version on record dates it the same way."""
    if persisted_body is None:
        return tuple(assembled)
    collection = cast("Mapping[str, list[Mapping[str, str]]]", persisted_body["collection"])
    return tuple(datetime.fromisoformat(member["prepared_at"]) for member in collection["members"])


def _prepared_now() -> datetime:
    """The `prepared_at` a version gets at its first mint: now in UTC, to the second."""
    return utc_now().replace(microsecond=0)


def persisted_response(
    snapshot_reader: SnapshotReader | None,
    *,
    subject_id: str,
    content_version: str,
    body: Mapping[str, object] | None,
    served: Manifest,
    call_warnings: Sequence[WarningModel],
    artifact_store: ArtifactStore,
    now: datetime,
) -> Manifest:
    """What a write returns when its version record already existed: the stored version.

    The response then equals a snapshot read of that version (`persisted_manifest`). `body` is
    the stored record the write rendered from; when it is `None` the record is read here.
    `served` returns unchanged only without a `snapshot_reader`.

    The record stays as stored. The write still links this call's linkable artifacts, each
    under its own `rendering_key`; an existing key keeps its artifact while that one is
    servable, and the response lists whichever the store links (`with_linked_artifacts`).
    """
    if body is None:
        body = _persisted_body(
            snapshot_reader, subject_id=subject_id, content_version=content_version
        )
    if body is None:
        return served
    return with_linked_artifacts(
        persisted_manifest(body=body, served=served, call_warnings=call_warnings),
        subject_id=subject_id,
        snapshot_reader=snapshot_reader,
        artifact_store=artifact_store,
        now=now,
    )


def with_linked_artifacts(
    manifest: Manifest,
    *,
    subject_id: str,
    snapshot_reader: SnapshotReader | None,
    artifact_store: ArtifactStore,
    now: datetime,
) -> Manifest:
    """`manifest` with each artifact replaced by the one linked under its rendering key, when
    that one has a servable location (`select_servable_locations`), signed under the name a
    snapshot read gives it. An artifact whose key has no servable link stays as this call
    rendered it.

    A linked artifact stays linked while it is servable, so every mint, re-mint and head hit of
    a version lists the same bytes under one key, whatever this call's render read from the
    graph; once it is not, the next linked render replaces it (`version_store`). Every write
    renders from the stored record, so a linked artifact carries the record's `prepared_at` and
    citation whichever writer linked it. A bundle left unlinked because its graph metadata reads
    failed stays in the response only while no servable artifact is linked under its key.
    Without a `snapshot_reader`, `manifest` returns unchanged.
    """
    if snapshot_reader is None or not manifest.artifacts:
        return manifest
    with phase("snapshot_read"):
        locations = snapshot_reader.read_artifact_locations(subject_id, manifest.content_version)
    servable = {
        location.rendering_key: location
        for location in select_servable_locations(locations, now=now)
    }
    artifacts: list[ArtifactEntryModel] = []
    for artifact in manifest.artifacts:
        location = servable.get(rendering_key(artifact.rendering))
        artifacts.append(
            artifact
            if location is None
            else sign_stored_artifact(
                location, manifest=manifest, artifact_store=artifact_store, read_at=now
            )
        )
    return manifest.model_copy(update={"artifacts": artifacts})


def bundle_linked(
    snapshot_reader: SnapshotReader | None,
    *,
    subject_id: str,
    content_version: str,
    options: MintOptions,
    site: PublicSiteSettings,
    now: datetime,
) -> bool:
    """Whether the version holds a servable artifact linked under the bundle rendering
    `options` asks for. The response then lists that artifact (`with_linked_artifacts`), so
    this call's render is discarded and needs no live documentation read. `False` for a bare
    rendering or without a `snapshot_reader`."""
    if snapshot_reader is None or not options.bundle:
        return False
    key = rendering_key(artifact_rendering(options, site=site))
    with phase("snapshot_read"):
        locations = snapshot_reader.read_artifact_locations(subject_id, content_version)
    return any(
        location.rendering_key == key for location in select_servable_locations(locations, now=now)
    )


def bundle_citations(
    citation: CitationModel,
    *,
    options: MintOptions,
    on_record: bool,
    linked: bool,
    documentation_complete: bool,
    graph: GraphReader,
    deadline: float | None,
    metadata_read_seconds: float,
    coverage_index: CoverageIndex | None,
) -> tuple[dict[str, CitationSource], bool]:
    """The sources a render cites by ref, and whether the graph reads behind them answered.

    Only a bundle carries documentation and licences. For a version already on record
    (`on_record`) whose bundle is not yet linked (`linked`, from `bundle_linked`) they come
    from `healed_citation`, given at least `metadata_read_seconds` whatever is left of
    `deadline`. A linked bundle is what the response lists, so the render cites the stored
    citation as it is. A first mint renders what it just read, and `documentation_complete`
    says whether those reads answered.
    """
    sources = citation.sources_by_ref()
    if not options.bundle or linked:
        return sources, True
    if not on_record:
        return sources, documentation_complete
    healed, complete = healed_citation(
        citation,
        graph=graph,
        deadline=metadata_deadline(deadline, floor_seconds=metadata_read_seconds),
        coverage_index=coverage_index,
    )
    return healed.sources_by_ref(), complete


def log_bundle_not_linked(
    *,
    subject_id: str,
    content_version: str,
    refs: Sequence[str],
    documentation_complete: bool,
    labels_complete: bool,
) -> None:
    logger.warning(
        "%s subject_id=%s content_version=%s refs=%s documentation_read_complete=%s "
        "label_read_complete=%s",
        BUNDLE_NOT_LINKED_MARKER,
        subject_id,
        content_version,
        list(refs),
        documentation_complete,
        labels_complete,
    )


def _link_artifacts(
    *,
    subject_id: str,
    content_version: str,
    artifacts: Iterable[ArtifactEntryModel | ArtifactRecord],
    version_writer: VersionRecordWriter,
    trace_id: str | None,
) -> None:
    """Links each artifact to a version already on record, one `record_artifact` call each. A
    key whose linked artifact is still servable keeps it (`version_store`)."""
    with phase("record"):
        for artifact in artifacts:
            version_writer.record_artifact(
                subject_id=subject_id,
                content_version=content_version,
                artifact=(
                    artifact
                    if isinstance(artifact, ArtifactRecord)
                    else ArtifactRecord(
                        artifact_digest=artifact.artifact_digest,
                        rendering=artifact.rendering,
                        media_type=artifact.media_type,
                    )
                ),
                trace_id=trace_id,
            )


@dataclass(frozen=True)
class DefinitionPrewrite:
    """Carries the resolved materialisation and empty-artifact manifest into the write half.
    `manifest` serves `GET /api/datasets/{id}/options`. The write half adds rendering and warnings.
    `manifest.prepared_at` is this call's instant, which the write half keeps only when the
    version is not yet on record.
    """

    stored: StoredDefinition
    result: MaterialisationResult
    manifest: Manifest
    # `read_source_citations` over this call's provenance ref, with documentation only when the
    # version is not yet on record. The write half builds every manifest from this map.
    source_citations: Mapping[str, SourceCitation]
    # `datasets_api.origin`'s `withheld_by_registry` and `un_system_unavailable` warnings for this
    # call, found once here and joined to every manifest the write half builds.
    origin_warnings: tuple[WarningModel, ...]
    provenance_ref: str
    spec_dict: dict[str, object]
    # All requests share a role signature, so rendering uses the first request's roles.
    entity_columns: tuple[str, ...]
    # Carried into final manifest assembly so both halves use the same stamp.
    coverage_stamp: str | None = None
    # The version's stored body, read once materialisation fixed its `content_version`. `None`
    # when the version is not yet on record or no snapshot reader was given.
    persisted_body: Mapping[str, object] | None = None
    # `False` when a documentation read in `source_citations` failed.
    documentation_complete: bool = True
    # The coverage snapshot materialisation read, for the documentation a bundle render reads.
    coverage_index: CoverageIndex | None = None


def _documentation_complete(source_citations: Mapping[str, SourceCitation]) -> bool:
    return not any(citation.documentation_read_failed for citation in source_citations.values())


def _prewrite_citations(
    graph: GraphReader,
    refs: Sequence[str],
    *,
    documented: bool,
    deadline: float | None,
    settings: DatasetsBuildSettings,
    coverage_index: CoverageIndex | None,
) -> dict[str, SourceCitation]:
    """A prewrite's `read_source_citations` over `refs`. A read that takes each source's
    documentation is a bundle metadata read, so it runs until `deadline` or
    `settings.metadata_read_seconds` from its start, whichever ends later. A mint that spent its
    deadline materialising still documents its first bundle."""
    return read_source_citations(
        graph,
        refs,
        catalogue=current_catalogue(),
        deadline=(
            metadata_deadline(deadline, floor_seconds=settings.metadata_read_seconds)
            if documented
            else deadline
        ),
        documented=documented,
        coverage_facts_of=coverage_facts_of(coverage_index),
    )


def prewrite_definition(
    *,
    stored: StoredDefinition,
    requests: Sequence[MaterialisationRequest],
    sdmx: SdmxClient,
    graph: GraphReader,
    revision_reader: RevisionReader,
    ceiling: int,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
    code_index: CodeIndex,
    rate_cache: RateTableCache,
    member_concurrency: int,
    materialise_wait_seconds: float,
    settings: DatasetsBuildSettings,
    materialise_semaphore: threading.BoundedSemaphore | None = None,
    precomputed_probe: CollectionFacetProbe | None = None,
    deadline: float | None = None,
    coverage_stamp: str | None = None,
    coverage_index: CoverageIndex | None = None,
    progress: RowProgress | None = None,
    snapshot_reader: SnapshotReader | None = None,
    documented: bool = True,
) -> DefinitionPrewrite:
    """Materialises one pinned request or the resolved members of a described-arm spec into one
    frame, version, and trace. It returns before artifact writes. The options route uses its
    manifest directly.

    The semaphore bounds rematerialisation across shared callers. `None` disables it. A supplied
    `precomputed_probe` avoids repeating the caller's currency probe. The absolute deadline bounds
    both semaphore wait and SDMX fetches. `member_concurrency` limits concurrent member fetches.

    The same `code_index` builds the manifest and renders output. `coverage_stamp` adds lineage.
    Materialisation determines identity. Role-bearing requests require `coverage_index`. Its absence
    raises `MaterialisationUsageError` before fetching. Subject-only requests may omit it.
    `progress` forwards worker progress. Profile CPU is recorded separately from materialisation.

    The citation read takes each source's documentation only when `documented` and the
    version is not on record in `snapshot_reader`. A re-mint renders from the stored citation
    (`bundle_citations`), and the options route passes `documented=False`.
    """
    with (
        origin_reads(graph, availability=sdmx, deadline=deadline) as un_blocked,
        bounded_materialise_slot(
            materialise_semaphore, wait_seconds=materialise_wait_seconds, deadline=deadline
        ),
        phase("materialise"),
    ):
        result = materialise_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            revision_reader=revision_reader,
            ceiling=ceiling,
            materialiser_build=materialiser_build,
            capability_version=capability_version,
            enrichment=enrichment,
            member_concurrency=member_concurrency,
            rate_cache=rate_cache,
            coverage=coverage_index,
            precomputed_probe=precomputed_probe,
            deadline=deadline,
            progress=progress,
        )
    record_elapsed("profile", result.output_profile_seconds)
    provenance_ref = _provenance_ref(result)
    persisted_body = _persisted_body(
        snapshot_reader, subject_id=stored.spec_id, content_version=result.content_version
    )
    with phase("citation"):
        source_citations = _prewrite_citations(
            graph,
            [provenance_ref],
            documented=documented and persisted_body is None,
            deadline=deadline,
            settings=settings,
            coverage_index=coverage_index,
        )
    origin_warnings = withheld_warnings(
        graph,
        [(request, result.coverage) for request in requests],
        default_scope=result.default_scope,
        deadline=deadline,
    ) + un_unavailable_warnings(un_blocked)
    spec_dict: dict[str, object] = {
        "schema_version": stored.schema_version,
        "spec_id": stored.spec_id,
        "identity": stored.identity,
    }
    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=[provenance_ref],
        source_citations=source_citations,
        prepared_at=_prepared_now(),
        capability_version=capability_version,
        materialiser_build=materialiser_build,
        code_index=code_index,
        settings=settings,
        extra_warnings=origin_warnings,
        coverage_stamp=coverage_stamp,
    )
    return DefinitionPrewrite(
        stored=stored,
        result=result,
        manifest=manifest,
        source_citations=source_citations,
        origin_warnings=origin_warnings,
        provenance_ref=provenance_ref,
        spec_dict=spec_dict,
        entity_columns=requests[0].declared_roles,
        coverage_stamp=coverage_stamp,
        persisted_body=persisted_body,
        documentation_complete=_documentation_complete(source_citations),
        coverage_index=coverage_index,
    )


def serve_definition(
    *,
    stored: StoredDefinition,
    requests: Sequence[MaterialisationRequest],
    options: MintOptions,
    deps: ServeDeps,
    trace_id: str | None = None,
    signal: Sequence[MemberSignal] | None = None,
    expected_content_version: str | None = None,
    precomputed_probe: CollectionFacetProbe | None = None,
    ceiling: int | None = None,
    coverage_stamp: str | None = None,
    coverage_index: CoverageIndex | None = None,
    progress: RowProgress | None = None,
) -> Manifest:
    """Materialises with `prewrite_definition`, then writes artifacts and the version record.
    A supplied `signal` advances the head after commit. The caller reads the head and passes its
    version for compare-and-set. `precomputed_probe` and `progress` are forwarded.

    `ceiling` overrides the raw-fetch guard. `None` uses `deps.ceiling`. `coverage_stamp` adds
    manifest lineage, while `coverage_index` feeds materialisation.
    """
    prewrite = prewrite_definition(
        stored=stored,
        requests=requests,
        sdmx=deps.sdmx,
        graph=deps.graph,
        revision_reader=deps.revision_reader,
        ceiling=ceiling if ceiling is not None else deps.ceiling,
        materialiser_build=deps.materialiser_build,
        capability_version=deps.capability_version,
        enrichment=deps.enrichment,
        code_index=deps.code_index,
        rate_cache=deps.rate_cache,
        member_concurrency=deps.member_concurrency,
        materialise_wait_seconds=deps.settings.materialise_wait_seconds,
        settings=deps.settings,
        materialise_semaphore=deps.materialise_semaphore,
        precomputed_probe=precomputed_probe,
        deadline=deps.deadline,
        coverage_stamp=coverage_stamp,
        coverage_index=coverage_index,
        progress=progress,
        snapshot_reader=deps.snapshot_reader,
    )
    return _write_definition_artifacts(
        prewrite=prewrite,
        options=options,
        deps=deps,
        trace_id=trace_id,
        signal=signal,
        expected_content_version=expected_content_version,
    )


def _write_definition_artifacts(
    *,
    prewrite: DefinitionPrewrite,
    options: MintOptions,
    deps: ServeDeps,
    trace_id: str | None,
    signal: Sequence[MemberSignal] | None = None,
    expected_content_version: str | None = None,
) -> Manifest:
    """Writes the version record, links its series statistics, then renders frame1 and the
    requested rendering, links them, and builds a fresh manifest. It keeps the wire warning
    order as materialisation, reshape, citation, then revision/facet defaults. Manifest assembly
    builds its citation from `prewrite.source_citations` and makes no graph read. A supplied
    signal advances the head only after the version record commits.

    A version not yet on record is recorded with no artifacts and the `prewrite` manifest before
    anything renders. When a concurrent first mint wins that insert, its stored body is read
    back, and the render, statistics and response carry its `prepared_at` and citation. The
    statistics and frame1 depend on no option, so they are linked before the requested
    rendering can fail, and a version on record always lists its statistics. A
    multi-contributor collapse is recorded after the render with its artifacts, dated and cited
    from the base record."""
    stored = prewrite.stored
    result = prewrite.result
    source_citations = prewrite.source_citations
    provenance_ref = prewrite.provenance_ref
    spec_dict = prewrite.spec_dict
    entity_columns = prewrite.entity_columns
    persisted_body = prewrite.persisted_body
    # Record before render. A version not yet on record is written with no artifacts, and when a
    # concurrent first mint wins the insert its stored body is read back, so every artifact below
    # renders from the record's `prepared_at` and citation and is linked after.
    created = False
    if persisted_body is None:
        created = record_definition_version(
            manifest=prewrite.manifest,
            stored=stored,
            materialisation=result,
            artifacts=[],
            version_writer=deps.version_writer,
            trace_id=trace_id,
        )
        if not created:
            persisted_body = _persisted_body(
                deps.snapshot_reader,
                subject_id=stored.spec_id,
                content_version=result.content_version,
            )
    render_citation = _render_citation(persisted_body, assembled=prewrite.manifest.citation)
    render_prepared_at = _render_prepared_at(
        persisted_body, assembled=prewrite.manifest.prepared_at
    )
    # Copy the caller's timing and trace context before submission. Workers use separate contexts.
    frame1_context = contextvars.copy_context()
    # The caller releases its store connection before joining. The frame1 worker overlaps the
    # statistics and the requested rendering, and each may borrow a connection concurrently.
    # Frame1 is skipped above its row ceiling. The version is always minted.
    write_frame1 = result.coverage.rows <= FRAME1_MAX_ROWS
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="datasets-frame1") as frame1_pool:
        frame1_future = (
            frame1_pool.submit(
                frame1_context.run,
                render_and_store_frame1,
                frame_chunks=result.frame_chunks(),
                subject_id=stored.spec_id,
                content_version=result.content_version,
                artifact_store=deps.artifact_store,
            )
            if write_frame1
            else None
        )
        try:
            # The statistics, like frame1, depend on no option, so they are linked before the
            # requested rendering, which can refuse or fail.
            statistics_artifact = render_and_store_series_statistics(
                data=result.series_statistics.data,
                subject_id=stored.spec_id,
                content_version=result.content_version,
                source_names=cited_source_names([provenance_ref], render_citation.sources_by_ref()),
                prepared_at=render_prepared_at,
                artifact_store=deps.artifact_store,
            )
            _link_artifacts(
                subject_id=stored.spec_id,
                content_version=result.content_version,
                artifacts=[statistics_artifact],
                version_writer=deps.version_writer,
                trace_id=trace_id,
            )
            citations, documentation_complete = bundle_citations(
                render_citation,
                options=options,
                on_record=persisted_body is not None,
                linked=persisted_body is not None
                and bundle_linked(
                    deps.snapshot_reader,
                    subject_id=stored.spec_id,
                    content_version=result.content_version,
                    options=options,
                    site=deps.settings.site,
                    now=utc_now(),
                ),
                documentation_complete=prewrite.documentation_complete,
                graph=deps.graph,
                deadline=deps.deadline,
                metadata_read_seconds=deps.settings.metadata_read_seconds,
                coverage_index=prewrite.coverage_index,
            )
            rendered = render_and_store(
                frame_chunks=result.frame_chunks(),
                options=options,
                spec=spec_dict,
                provenance_refs=[provenance_ref],
                citations=citations,
                reproducing_call=(
                    ReproducingCall(
                        path=f"/api/datasets/{stored.spec_id}",
                        options=encode_query_options(options),
                    )
                    if DEFINITION_ROUTE_LIVE
                    else None
                ),
                content_version=result.content_version,
                prepared_at=render_prepared_at,
                site=deps.settings.site,
                artifact_store=deps.artifact_store,
                entity_columns=entity_columns,
                enrichment=deps.enrichment,
                pivot_max_output_cells=deps.settings.pivot_max_output_cells,
                code_index=deps.code_index,
                resolved_stat_var_refs=_resolved_stat_var_refs(result),
                groupby_preimage=_groupby_collapse_preimage(result),
                deadline=deps.deadline,
                metadata_read_seconds=deps.settings.metadata_read_seconds,
            )
            artifact, reshape_warnings, collapse = (
                rendered.artifact,
                rendered.warnings,
                rendered.collapse,
            )
        except BaseException:
            # Executor shutdown joins the worker but discards its exception. Retrieve the result
            # and link a stored frame1, which depends on no option, so a failed rendering leaves
            # it on record. Log `ArtifactStoreError` as best-effort and every other worker or
            # link failure, and preserve the primary failure.
            if frame1_future is not None:
                try:
                    _link_artifacts(
                        subject_id=stored.spec_id,
                        content_version=result.content_version,
                        artifacts=[cast(ArtifactRecord, frame1_future.result())],
                        version_writer=deps.version_writer,
                        trace_id=trace_id,
                    )
                except ArtifactStoreError:
                    logger.error(
                        "%s spec_id=%s content_version=%s",
                        FRAME1_STORE_FAILED_MARKER,
                        stored.spec_id,
                        result.content_version,
                        exc_info=True,
                    )
                except Exception:
                    logger.exception(
                        "frame1 write or link also raised while the primary write failed "
                        "spec_id=%s content_version=%s",
                        stored.spec_id,
                        result.content_version,
                    )
            raise
        if frame1_future is not None:
            try:
                # `Context.run` hides the worker's return type from static analysis.
                frame1_record = cast(ArtifactRecord, frame1_future.result())
            except ArtifactStoreError:
                logger.error(
                    "%s spec_id=%s content_version=%s",
                    FRAME1_STORE_FAILED_MARKER,
                    stored.spec_id,
                    result.content_version,
                    exc_info=True,
                )
            else:
                _link_artifacts(
                    subject_id=stored.spec_id,
                    content_version=result.content_version,
                    artifacts=[frame1_record],
                    version_writer=deps.version_writer,
                    trace_id=trace_id,
                )
    # A render whose graph metadata reads failed is returned but never linked, so the next
    # render of this version reads again.
    linked = documentation_complete and rendered.labels_complete
    if not linked:
        log_bundle_not_linked(
            subject_id=stored.spec_id,
            content_version=(
                collapse.content_version
                if collapse is not None and collapse.content_version is not None
                else result.content_version
            ),
            refs=[provenance_ref],
            documentation_complete=documentation_complete,
            labels_complete=rendered.labels_complete,
        )
    linked_artifacts = [artifact] if linked else []
    call_warnings = prewrite.origin_warnings + reshape_warning_models(reshape_warnings)
    # Multi-contributor collapses have a distinct version. Record the base version separately so a
    # plain GET can still resolve it. A sole-contributor collapse shares the base version. The head
    # stays on the base materialisation, and definition retrieval derives the collapsed version on a
    # head hit.
    if collapse is not None and collapse.content_version not in (None, result.content_version):
        collapsed_statistics_sink = SeriesStatisticsSink()
        collapsed_statistics_sink.write(compute_series_statistics_table(collapse.frame))
        collapsed_result = replace(
            result,
            content_version=collapse.content_version,
            coverage=replace(result.coverage, rows=len(collapse.frame.rows)),
            series_statistics=collapsed_statistics_sink.finish(),
        )
        collapsed_frame1_record: ArtifactRecord | None = None
        if collapsed_result.coverage.rows <= FRAME1_MAX_ROWS:
            try:
                collapsed_frame1_record = render_and_store_frame1(
                    frame_chunks=(collapse.frame,),
                    subject_id=stored.spec_id,
                    content_version=collapse.content_version,
                    artifact_store=deps.artifact_store,
                )
            except ArtifactStoreError:
                logger.error(
                    "%s spec_id=%s content_version=%s",
                    FRAME1_STORE_FAILED_MARKER,
                    stored.spec_id,
                    collapse.content_version,
                    exc_info=True,
                )
        collapsed_statistics_artifact = render_and_store_series_statistics(
            data=collapsed_result.series_statistics.data,
            subject_id=stored.spec_id,
            content_version=collapsed_result.content_version,
            source_names=cited_source_names([provenance_ref], render_citation.sources_by_ref()),
            prepared_at=render_prepared_at,
            artifact_store=deps.artifact_store,
        )
        # The collapsed identity cites and dates as its base record does, and so does every
        # concurrent writer of it, so its record is written with its artifacts in one call.
        # `collapsed_head_manifest` does the same.
        manifest = assemble_manifest(
            stored=stored,
            materialisation=collapsed_result,
            artifacts=[artifact, collapsed_statistics_artifact],
            provenance_refs=[provenance_ref],
            source_citations=source_citations,
            prepared_at=render_prepared_at,
            capability_version=deps.capability_version,
            materialiser_build=deps.materialiser_build,
            code_index=deps.code_index,
            settings=deps.settings,
            extra_warnings=call_warnings,
            coverage_stamp=prewrite.coverage_stamp,
        ).model_copy(update={"citation": render_citation})
        response_content_version = collapse.content_version
        response_created = record_definition_version(
            manifest=manifest,
            stored=stored,
            materialisation=collapsed_result,
            artifacts=[*linked_artifacts, collapsed_statistics_artifact],
            record_only_artifacts=(
                [collapsed_frame1_record] if collapsed_frame1_record is not None else []
            ),
            version_writer=deps.version_writer,
            trace_id=trace_id,
            base_content_version=result.content_version,
        )
        response_body = None
    else:
        manifest = assemble_manifest(
            stored=stored,
            materialisation=result,
            artifacts=[artifact, statistics_artifact],
            provenance_refs=[provenance_ref],
            source_citations=source_citations,
            prepared_at=render_prepared_at,
            capability_version=deps.capability_version,
            materialiser_build=deps.materialiser_build,
            code_index=deps.code_index,
            settings=deps.settings,
            extra_warnings=call_warnings,
            coverage_stamp=prewrite.coverage_stamp,
        )
        _link_artifacts(
            subject_id=stored.spec_id,
            content_version=result.content_version,
            artifacts=linked_artifacts,
            version_writer=deps.version_writer,
            trace_id=trace_id,
        )
        response_content_version = result.content_version
        response_created = created
        response_body = persisted_body
    # A concurrent first mint that lost the record insert links its own rendering, possibly before
    # this call does, so even a created version lists what the store links.
    if response_created:
        manifest = with_linked_artifacts(
            manifest,
            subject_id=stored.spec_id,
            snapshot_reader=deps.snapshot_reader,
            artifact_store=deps.artifact_store,
            now=utc_now(),
        )
    else:
        manifest = persisted_response(
            deps.snapshot_reader,
            subject_id=stored.spec_id,
            content_version=response_content_version,
            body=response_body,
            served=manifest,
            call_warnings=call_warnings,
            artifact_store=deps.artifact_store,
            now=utc_now(),
        )
    if signal is not None:
        with phase("head_advance"):
            written = deps.head_store.advance_head(
                subject_id=stored.spec_id,
                content_version=result.content_version,
                signal=_signal_with_rate_provenance(
                    signal, result.member_conversion_records.values()
                ),
                expected_content_version=expected_content_version,
            )
        if written is None:
            logger.debug(
                "head advance lost the race spec_id=%s expected_content_version=%s "
                "content_version=%s",
                stored.spec_id,
                expected_content_version,
                result.content_version,
            )
    return manifest


@dataclass(frozen=True)
class CollectionPrewrite:
    """Carries member materialisations and an empty-artifact manifest into the write half."""

    stored: StoredCollection
    per_member: list[tuple[StoredDefinition, MaterialisationResult]]
    per_member_entity_columns: list[tuple[str, ...]]
    collection_result: CollectionMaterialisationResult
    manifest: Manifest
    # `read_source_citations` over every member's provenance ref. The collection manifest and every
    # member's version-record manifest build their citations from this map.
    source_citations: Mapping[str, SourceCitation]
    provenance_refs: list[str]
    spec_dict: dict[str, object]
    # Keep caller warnings separate to avoid duplicating assembled warnings in the write half.
    extra_warnings: Sequence[WarningModel]
    # Parallel to `per_member`: each member's own `withheld_by_registry` and
    # `un_system_unavailable` warnings, for its version record.
    member_origin_warnings: tuple[tuple[WarningModel, ...], ...]
    # Parallel to `per_member`: each member version's stored `prepared_at`, or this call's
    # instant for a member not yet on record.
    member_prepared_at: tuple[datetime, ...]
    # Parallel to `per_member`: whether each member version was on record when read here.
    member_on_record: tuple[bool, ...]
    # Carried into the final manifest assembly, as in `DefinitionPrewrite`.
    coverage_stamp: str | None = None
    # The collection version's stored body, as in `DefinitionPrewrite`.
    persisted_body: Mapping[str, object] | None = None
    # `False` when a documentation read in `source_citations` failed.
    documentation_complete: bool = True
    # The coverage snapshot materialisation read, as in `DefinitionPrewrite`.
    coverage_index: CoverageIndex | None = None


def prewrite_collection(
    *,
    stored: StoredCollection,
    requests: Sequence[tuple[StoredDefinition, MaterialisationRequest]],
    sdmx: SdmxClient,
    graph: GraphReader,
    revision_reader: RevisionReader,
    ceiling: int,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
    member_concurrency: int,
    code_index: CodeIndex,
    rate_cache: RateTableCache,
    materialise_wait_seconds: float,
    settings: DatasetsBuildSettings,
    extra_warnings: Sequence[WarningModel] = (),
    materialise_semaphore: threading.BoundedSemaphore | None = None,
    precomputed_probe: CollectionFacetProbe | None = None,
    deadline: float | None = None,
    coverage_stamp: str | None = None,
    coverage_index: CoverageIndex | None = None,
    progress: RowProgress | None = None,
    snapshot_reader: SnapshotReader | None = None,
    documented: bool = True,
) -> CollectionPrewrite:
    """Materialises collection members and returns a manifest with an empty artifact list. Caller
    warnings join that manifest. The write half adds reshape warnings.

    The semaphore and deadline bound rematerialisation, queue wait, and SDMX fetches. A supplied
    `precomputed_probe` reuses the caller's facet probe. The shared `code_index` builds the
    manifest.
    `coverage_index` feeds materialisation. `coverage_stamp` records lineage. Profile CPU is summed
    across members and recorded separately. `progress` is forwarded to each fetch.
    `snapshot_reader` supplies each member version's stored `prepared_at`. A member not on record,
    and every member when it is `None`, takes this call's instant. It also supplies the
    collection version's stored body, and the citation read takes documentation only when
    `documented` and that version is not on record, as in `prewrite_definition`.
    """
    with (
        origin_reads(graph, availability=sdmx, deadline=deadline) as un_blocked,
        bounded_materialise_slot(
            materialise_semaphore, wait_seconds=materialise_wait_seconds, deadline=deadline
        ),
        phase("materialise"),
    ):
        collection_result = materialise_collection(
            [(member_stored.spec_id, request) for member_stored, request in requests],
            sdmx=sdmx,
            graph=graph,
            revision_reader=revision_reader,
            ceiling=ceiling,
            materialiser_build=materialiser_build,
            capability_version=capability_version,
            enrichment=enrichment,
            member_concurrency=member_concurrency,
            rate_cache=rate_cache,
            coverage=coverage_index,
            precomputed_probe=precomputed_probe,
            deadline=deadline,
            progress=progress,
        )
    record_elapsed(
        "profile",
        sum(result.output_profile_seconds for _spec_id, result in collection_result.members),
    )
    per_member = [
        (member_stored, result)
        for (member_stored, _request), (_spec_id, result) in zip(
            requests, collection_result.members, strict=True
        )
    ]
    # Parallel to `per_member`, preserving each request's declared role columns for rendering.
    per_member_entity_columns = [request.declared_roles for _member_stored, request in requests]
    provenance_refs = [_provenance_ref(result) for _stored, result in per_member]
    persisted_body = _persisted_body(
        snapshot_reader,
        subject_id=stored.collection_id,
        content_version=collection_result.content_version,
    )
    with phase("citation"):
        source_citations = _prewrite_citations(
            graph,
            provenance_refs,
            documented=documented and persisted_body is None,
            deadline=deadline,
            settings=settings,
            coverage_index=coverage_index,
        )
    prepared_at = _prepared_now()
    member_bodies = [
        _persisted_body(
            snapshot_reader,
            subject_id=member_stored.spec_id,
            content_version=result.content_version,
        )
        for member_stored, result in per_member
    ]
    member_prepared_at = tuple(
        _render_prepared_at(body, assembled=prepared_at) for body in member_bodies
    )
    withheld = withheld_warning(
        graph,
        [
            (request, result.coverage)
            for (_member_stored, request), (_stored, result) in zip(
                requests, per_member, strict=True
            )
        ],
        default_scope=collection_result.default_scope,
        deadline=deadline,
    )
    withheld_by_index = dict([withheld] if withheld is not None else [])
    member_origin_warnings = tuple(
        (
            *((withheld_by_index[index],) if index in withheld_by_index else ()),
            *un_unavailable_warnings(un_blocked, request=request),
        )
        for index, (_member_stored, request) in enumerate(requests)
    )
    extra_warnings = [
        *extra_warnings,
        *withheld_by_index.values(),
        *un_unavailable_warnings(un_blocked),
    ]

    spec_dict: dict[str, object] = {
        "collection_id": stored.collection_id,
        "members": stored.member_spec_ids,
    }
    manifest = assemble_collection_manifest(
        stored=stored,
        per_member=per_member,
        collection_result=collection_result,
        artifacts=[],
        provenance_refs=provenance_refs,
        source_citations=source_citations,
        prepared_at=prepared_at,
        member_prepared_at=member_prepared_at,
        capability_version=capability_version,
        materialiser_build=materialiser_build,
        code_index=code_index,
        settings=settings,
        extra_warnings=extra_warnings,
        coverage_stamp=coverage_stamp,
    )
    return CollectionPrewrite(
        stored=stored,
        per_member=per_member,
        per_member_entity_columns=per_member_entity_columns,
        collection_result=collection_result,
        manifest=manifest,
        source_citations=source_citations,
        provenance_refs=provenance_refs,
        spec_dict=spec_dict,
        extra_warnings=extra_warnings,
        member_origin_warnings=member_origin_warnings,
        member_prepared_at=member_prepared_at,
        member_on_record=tuple(body is not None for body in member_bodies),
        coverage_stamp=coverage_stamp,
        persisted_body=persisted_body,
        documentation_complete=_documentation_complete(source_citations),
        coverage_index=coverage_index,
    )


def serve_collection(
    *,
    stored: StoredCollection,
    requests: Sequence[tuple[StoredDefinition, MaterialisationRequest]],
    options: MintOptions,
    deps: ServeDeps,
    trace_id: str | None = None,
    extra_warnings: Sequence[WarningModel] = (),
    signal: Sequence[MemberSignal] | None = None,
    expected_content_version: str | None = None,
    precomputed_probe: CollectionFacetProbe | None = None,
    ceiling: int | None = None,
    coverage_stamp: str | None = None,
    coverage_index: CoverageIndex | None = None,
    progress: RowProgress | None = None,
) -> Manifest:
    """Materialises collection members, then writes artifacts and the version record. A supplied
    `signal` advances the collection head after commit using the caller's expected version. Probe,
    ceiling, coverage, and progress arguments pass to `prewrite_collection`."""
    prewrite = prewrite_collection(
        stored=stored,
        requests=requests,
        sdmx=deps.sdmx,
        graph=deps.graph,
        revision_reader=deps.revision_reader,
        ceiling=ceiling if ceiling is not None else deps.ceiling,
        materialiser_build=deps.materialiser_build,
        capability_version=deps.capability_version,
        enrichment=deps.enrichment,
        member_concurrency=deps.member_concurrency,
        code_index=deps.code_index,
        rate_cache=deps.rate_cache,
        materialise_wait_seconds=deps.settings.materialise_wait_seconds,
        settings=deps.settings,
        extra_warnings=extra_warnings,
        materialise_semaphore=deps.materialise_semaphore,
        precomputed_probe=precomputed_probe,
        deadline=deps.deadline,
        coverage_stamp=coverage_stamp,
        coverage_index=coverage_index,
        progress=progress,
        snapshot_reader=deps.snapshot_reader,
    )
    return _write_collection_artifacts(
        prewrite=prewrite,
        options=options,
        deps=deps,
        trace_id=trace_id,
        signal=signal,
        expected_content_version=expected_content_version,
    )


def _collection_member_renderings(
    per_member: Sequence[tuple[StoredDefinition, MaterialisationResult]],
    per_member_entity_columns: Sequence[tuple[str, ...]],
    member_prepared_at: Sequence[datetime],
    member_refs: Sequence[Sequence[str]],
) -> Iterator[CollectionMemberRendering]:
    """Yields renderings in member order. The renderer calls each `frame_chunks` factory when it
    reaches that member, keeping later member rows lazy. `member_refs` comes from
    `_render_member_refs` and names the member's files. The reshape takes the member's live
    provenance ref."""
    for (member_stored, result), entity_columns, prepared_at, refs in zip(
        per_member, per_member_entity_columns, member_prepared_at, member_refs, strict=True
    ):
        yield CollectionMemberRendering(
            spec_id=member_stored.spec_id,
            frame_chunks=result.frame_chunks,
            provenance_refs=[_provenance_ref(result)],
            cited_refs=list(refs),
            content_version=result.content_version,
            prepared_at=prepared_at,
            entity_columns=entity_columns,
            resolved_stat_var_refs=_resolved_stat_var_refs(result),
        )


def _write_collection_artifacts(
    *,
    prewrite: CollectionPrewrite,
    options: MintOptions,
    deps: ServeDeps,
    trace_id: str | None,
    signal: Sequence[MemberSignal] | None = None,
    expected_content_version: str | None = None,
) -> Manifest:
    """Writes each member's frame1 and version record and the collection's version record, then
    renders and links the per-member statistics and then the requested artifact from the stored
    values, and builds a fresh collection manifest."""
    stored = prewrite.stored
    per_member = prewrite.per_member
    per_member_entity_columns = prewrite.per_member_entity_columns
    collection_result = prewrite.collection_result
    source_citations = prewrite.source_citations
    provenance_refs = prewrite.provenance_refs
    spec_dict = prewrite.spec_dict
    extra_warnings = prewrite.extra_warnings
    persisted_body = prewrite.persisted_body
    # Record before render, as in `_write_definition_artifacts`. Member records come first: a
    # collection not yet on record dates each member entry as that member's record is, read back
    # when a concurrent mint of the member won its insert. The collection record follows with no
    # artifacts, and a lost insert reads back the winner's body. Every artifact below renders from
    # the stored values and is linked after. Member records keyed by `(spec_id, content_version)`
    # are idempotent across collection and direct definition mints.
    member_prepared_at = _record_member_versions(
        collection_id=stored.collection_id,
        per_member=per_member,
        member_origin_warnings=prewrite.member_origin_warnings,
        member_prepared_at=_render_member_prepared_at(
            persisted_body, assembled=prewrite.member_prepared_at
        ),
        member_on_record=prewrite.member_on_record,
        source_citations=source_citations,
        deps=deps,
        trace_id=trace_id,
    )
    created = False
    if persisted_body is None:
        created = record_collection_version(
            manifest=_with_member_prepared_at(prewrite.manifest, member_prepared_at),
            stored=stored,
            per_member=per_member,
            collection_result=collection_result,
            artifacts=[],
            version_writer=deps.version_writer,
            trace_id=trace_id,
        )
        if not created:
            persisted_body = _persisted_body(
                deps.snapshot_reader,
                subject_id=stored.collection_id,
                content_version=collection_result.content_version,
            )
    render_citation = _render_citation(persisted_body, assembled=prewrite.manifest.citation)
    render_prepared_at = _render_prepared_at(
        persisted_body, assembled=prewrite.manifest.prepared_at
    )
    member_refs = _render_member_refs(persisted_body, per_member=per_member)
    render_member_prepared_at = _render_member_prepared_at(
        persisted_body, assembled=member_prepared_at
    )
    # Store one statistics artifact per member, independent of format or bundle. The member key
    # keeps each artifact distinct under the shared collection version. They depend on no option,
    # so they are linked before the requested rendering, which can refuse or fail. Names come from
    # the stored labels, which a bundle's citation fill never changes.
    statistics_artifacts = [
        render_and_store_series_statistics(
            data=result.series_statistics.data,
            subject_id=member_stored.spec_id,
            content_version=result.content_version,
            source_names=cited_source_names(refs, render_citation.sources_by_ref()),
            prepared_at=prepared_at,
            artifact_store=deps.artifact_store,
            member=member_stored.spec_id,
        )
        for (member_stored, result), prepared_at, refs in zip(
            per_member, render_member_prepared_at, member_refs, strict=True
        )
    ]
    _link_artifacts(
        subject_id=stored.collection_id,
        content_version=collection_result.content_version,
        artifacts=statistics_artifacts,
        version_writer=deps.version_writer,
        trace_id=trace_id,
    )
    citations_by_ref, documentation_complete = bundle_citations(
        render_citation,
        options=options,
        on_record=persisted_body is not None,
        linked=persisted_body is not None
        and bundle_linked(
            deps.snapshot_reader,
            subject_id=stored.collection_id,
            content_version=collection_result.content_version,
            options=options,
            site=deps.settings.site,
            now=utc_now(),
        ),
        documentation_complete=prewrite.documentation_complete,
        graph=deps.graph,
        deadline=deps.deadline,
        metadata_read_seconds=deps.settings.metadata_read_seconds,
        coverage_index=prewrite.coverage_index,
    )

    rendered = render_and_store_collection(
        members=_collection_member_renderings(
            per_member, per_member_entity_columns, render_member_prepared_at, member_refs
        ),
        options=options,
        spec=spec_dict,
        citations=citations_by_ref,
        reproducing_call=(
            ReproducingCall(
                path=f"/api/datasets/{stored.collection_id}",
                options=encode_query_options(options),
            )
            if DEFINITION_ROUTE_LIVE
            else None
        ),
        content_version=collection_result.content_version,
        prepared_at=render_prepared_at,
        site=deps.settings.site,
        artifact_store=deps.artifact_store,
        enrichment=deps.enrichment,
        pivot_max_output_cells=deps.settings.pivot_max_output_cells,
        code_index=deps.code_index,
        deadline=deps.deadline,
        metadata_read_seconds=deps.settings.metadata_read_seconds,
    )
    call_warnings = list(extra_warnings) + list(reshape_warning_models(rendered.warnings))
    # A render whose graph metadata reads failed is returned but never linked, as in
    # `_write_definition_artifacts`.
    linked = documentation_complete and rendered.labels_complete
    if not linked:
        log_bundle_not_linked(
            subject_id=stored.collection_id,
            content_version=collection_result.content_version,
            refs=provenance_refs,
            documentation_complete=documentation_complete,
            labels_complete=rendered.labels_complete,
        )
    _link_artifacts(
        subject_id=stored.collection_id,
        content_version=collection_result.content_version,
        artifacts=rendered.artifacts if linked else [],
        version_writer=deps.version_writer,
        trace_id=trace_id,
    )
    manifest = assemble_collection_manifest(
        stored=stored,
        per_member=per_member,
        collection_result=collection_result,
        artifacts=rendered.artifacts + statistics_artifacts,
        provenance_refs=provenance_refs,
        source_citations=source_citations,
        prepared_at=render_prepared_at,
        member_prepared_at=render_member_prepared_at,
        capability_version=deps.capability_version,
        materialiser_build=deps.materialiser_build,
        code_index=deps.code_index,
        settings=deps.settings,
        extra_warnings=call_warnings,
        coverage_stamp=prewrite.coverage_stamp,
    )
    # As in `_write_definition_artifacts`, a created version lists what the store links.
    if created:
        manifest = with_linked_artifacts(
            manifest,
            subject_id=stored.collection_id,
            snapshot_reader=deps.snapshot_reader,
            artifact_store=deps.artifact_store,
            now=utc_now(),
        )
    else:
        manifest = persisted_response(
            deps.snapshot_reader,
            subject_id=stored.collection_id,
            content_version=collection_result.content_version,
            body=persisted_body,
            served=manifest,
            call_warnings=call_warnings,
            artifact_store=deps.artifact_store,
            now=utc_now(),
        )
    if signal is not None:
        # Per-member (`combine=False`): each `signal` entry's own rate
        # provenance comes from that same-position member's own
        # `result.conversion_record` alone, off `per_member`, in the same
        # `requests` order both `signal` and `per_member` share
        # (`prewrite_collection`'s own docstring). This differs from the
        # one shared group entry `_write_definition_artifacts` fills in.
        signal_with_rate_provenance = tuple(
            _signal_with_rate_provenance((member_signal,), (result.conversion_record,))[0]
            for member_signal, (_member_stored, result) in zip(signal, per_member, strict=True)
        )
        with phase("head_advance"):
            written = deps.head_store.advance_head(
                subject_id=stored.collection_id,
                content_version=collection_result.content_version,
                signal=signal_with_rate_provenance,
                expected_content_version=expected_content_version,
            )
        if written is None:
            logger.debug(
                "head advance lost the race collection_id=%s expected_content_version=%s "
                "content_version=%s",
                stored.collection_id,
                expected_content_version,
                collection_result.content_version,
            )
    return manifest


def _with_member_prepared_at(
    manifest: Manifest, member_prepared_at: Sequence[datetime]
) -> Manifest:
    """`manifest`, a collection's, with each member entry dated from `member_prepared_at`,
    parallel to its members."""
    collection = cast(CollectionModel, manifest.collection)
    members = [
        member.model_copy(update={"prepared_at": prepared_at})
        for member, prepared_at in zip(collection.members, member_prepared_at, strict=True)
    ]
    return manifest.model_copy(
        update={"collection": collection.model_copy(update={"members": members})}
    )


def _record_member_versions(
    *,
    collection_id: str,
    per_member: Sequence[tuple[StoredDefinition, MaterialisationResult]],
    member_origin_warnings: Sequence[Sequence[WarningModel]],
    member_prepared_at: Sequence[datetime],
    member_on_record: Sequence[bool],
    source_citations: Mapping[str, SourceCitation],
    deps: ServeDeps,
    trace_id: str | None = None,
) -> tuple[datetime, ...]:
    """Stores frame1 and a version record for each member, whose manifest carries that member's
    entries in `member_origin_warnings` and `member_prepared_at`. Returns, parallel to
    `per_member`, the `prepared_at` each member's record carries: the stored one, read back,
    for a member not on record before (`member_on_record`) whose insert a concurrent mint won,
    else the one given.
    Frame1 writes are best-effort. Failures are logged and later mints can heal them. Each member
    gets its own `(spec_id, content_version)` record so collection-only members remain loadable.
    Uses a sliding pool of `member_concurrency` workers. Each task gets a fresh context because
    one `Context` cannot run in two threads at once.

    Version-record failures are logged and never fail the collection mint. A member with a missing
    record is unavailable until a later mint. Its absence leaves the collection record and manifest
    unchanged. Member writes use the existing `record` timing phase.

    Every member gets its write attempts after the deadline, keeping persisted head state complete.
    Late completion logs the collection, member count, and delay. Remaining work is bounded by
    `MAX_COLLECTION_MEMBERS / member_concurrency` rounds. Each member record contains frame1. The
    collection rendering belongs to the collection subject.
    """

    def _persist_one(
        member_stored: StoredDefinition,
        result: MaterialisationResult,
        origin_warnings: Sequence[WarningModel],
        prepared_at: datetime,
        on_record: bool,
    ) -> datetime:
        if result.coverage.rows > FRAME1_MAX_ROWS:
            member_artifacts: tuple[ArtifactRecord, ...] = ()
        else:
            try:
                frame1_record = render_and_store_frame1(
                    frame_chunks=result.frame_chunks(),
                    subject_id=member_stored.spec_id,
                    content_version=result.content_version,
                    artifact_store=deps.artifact_store,
                )
            except ArtifactStoreError:
                logger.error(
                    "%s spec_id=%s content_version=%s",
                    FRAME1_STORE_FAILED_MARKER,
                    member_stored.spec_id,
                    result.content_version,
                    exc_info=True,
                )
                member_artifacts = ()
            else:
                member_artifacts = (frame1_record,)
        member_manifest = assemble_manifest(
            stored=member_stored,
            materialisation=result,
            artifacts=[],
            provenance_refs=[_provenance_ref(result)],
            source_citations=source_citations,
            prepared_at=prepared_at,
            capability_version=deps.capability_version,
            materialiser_build=deps.materialiser_build,
            code_index=deps.code_index,
            settings=deps.settings,
            extra_warnings=origin_warnings,
        )
        with phase("record"):
            try:
                created = deps.version_writer.record_version(
                    VersionRecord(
                        subject_id=member_stored.spec_id,
                        subject_kind="spec",
                        content_version=result.content_version,
                        schema_version=member_stored.schema_version,
                        immutable_body=encode_version_body(
                            member_manifest,
                            output_profile=result.output_profile,
                            series_conversions=result.series_conversions,
                            default_scope=result.default_scope,
                        ),
                        artifacts=member_artifacts,
                    ),
                    trace_id=trace_id,
                )
            except DatasetsStoreUnavailable:
                logger.error(
                    "%s spec_id=%s content_version=%s",
                    MEMBER_VERSION_RECORD_FAILED_MARKER,
                    member_stored.spec_id,
                    result.content_version,
                    exc_info=True,
                )
                return prepared_at
        if created or on_record:
            return prepared_at
        return _render_prepared_at(
            _persisted_body(
                deps.snapshot_reader,
                subject_id=member_stored.spec_id,
                content_version=result.content_version,
            ),
            assembled=prepared_at,
        )

    def _submit(pool: ThreadPoolExecutor, index: int) -> Future:
        # Copy context per task because a `Context` cannot run in two threads at once. `pool.submit`
        # hides `_persist_one`'s return type.
        context = contextvars.copy_context()
        member_stored, result = per_member[index]
        return pool.submit(
            context.run,
            _persist_one,
            member_stored,
            result,
            member_origin_warnings[index],
            member_prepared_at[index],
            member_on_record[index],
        )

    max_workers = max(1, min(deps.member_concurrency, len(per_member)))
    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="datasets-member-persist")
    try:
        pending = [_submit(pool, index) for index in range(min(max_workers, len(per_member)))]
        next_index = len(pending)
        stored_prepared_at: list[datetime] = []
        for _member in per_member:
            stored_prepared_at.append(cast(datetime, pending.pop(0).result()))
            if next_index < len(per_member):
                pending.append(_submit(pool, next_index))
                next_index += 1
    finally:
        # Cancel queued tasks after a failure, but wait for every submitted task to finish.
        pool.shutdown(wait=True, cancel_futures=True)

    if deps.deadline is not None:
        seconds_past_deadline = time.monotonic() - deps.deadline
        if seconds_past_deadline > 0:
            logger.warning(
                "member persistence finished past the deadline collection_id=%s "
                "member_count=%d seconds_past_deadline=%.3f",
                collection_id,
                len(per_member),
                seconds_past_deadline,
            )
    return tuple(stored_prepared_at)
