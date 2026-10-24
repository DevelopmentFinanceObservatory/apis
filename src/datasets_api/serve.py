"""`serve_definition` runs synchronously on the Datasets surface's work executor, or in the worker.
Its prewrite half materialises resolved requests and builds a manifest with an empty artifact list.
Its write half stores artifacts, renders the requested rendering, records the version, and
advances the head. `datasets_api.current.serve_current` calls it for every materialising entry
point.

Group members materialise across `member_concurrency` workers. The materialise timing covers
the full fan-out. The prewrite half records output-profile CPU in a separate `Server-Timing`
phase.

`serve_definition` streams the canonical frame from every frame chunk beside the requested
rendering on a worker, at every size, and joins it before recording the version. A failed
canonical frame write fails the serve before the head advances, so a head names a version only
once its canonical frame, statistics and materialisation row are durable. A later serve of the
version writes the frame again.

A serve's store work is a fixed set of transactions, each holding one `dataset_store` borrow at
a time: the head read its caller makes, a read of the stored version only when that head names
the result's `content_version`, one write transaction for the version record, every artifact's
inventory row and link and any collapsed version (`VersionRecordWriter.write_version`), and one
for the materialisation row and head advance. The write transaction also writes each version's
description record (`version_descriptions.mint_describers`), whose labels one batched graph read
gathers just before it. Artifacts are stored through a `deferred` view
of the artifact store, whose writes add their inventory rows to an `InventoryBatch` for the
write transaction and read no inventory for new bytes, so the frame worker opens no
transaction beside the request thread. The canonical frame and the requested rendering overlap.
Each times under its own phase, `store_frame` and `store_rendering`, beside `store_statistics` for
the series statistics, and frame encoding runs inside `store_frame`.

Every run records one `datasets.materialisation` row (`MaterialisationRecord`) after the version
record commits: the trigger, release, code index, catalogue, coverage and graph generation it ran
under, its dependency fingerprint, selection and conversion. The head advances in the same
transaction as that insert, so the head names the run that produced it and carries its
fingerprint for `current.serve_current`'s freshness check. A worker run's operation claim
(`ServeDeps.head_fence`) is checked under a row lock in that transaction, and a claim another
attempt has taken raises `HeadFenceLost` with neither row written. The caller reads the head before
materialising and passes its `content_version`, or `None`, for compare-and-set, and
`serve_definition` returns the head it wrote (`ServedDefinition`).

`serve_definition` holds one materialise permit (`ServeDeps.materialise_semaphore`) from before
its first SDMX request to its last artifact write, since the frame stays alive through citation,
canonical frame, statistics and rendering. The permits bound the frames alive at once, and so
peak memory.

`ServeDeps.deadline` bounds every SDMX, graph and metadata read of a serve; a bundle whose
metadata reads run out of time is linked as incomplete (`incomplete_artifact`). `ServeDeps.work_deadline` bounds a whole
worker run. The write half checks it before the version record and before each artifact render
and upload, and raises `WorkDeadlinePassed` once it has passed, so no write starts after it.
"""

from __future__ import annotations

import contextvars
import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import cast

from artifact_store import ArtifactStore, ArtifactStoreError
from artifact_store.inventory import InventoryBatch, InventoryEntry
from code_index.query import CodeIndex
from dataset_store.clock import utc_now
from dataset_store.snapshot import (
    ManifestFacts,
    SnapshotReader,
    StoredArtifactLocation,
    StoredVersion,
    VersionState,
    select_servable_locations,
)
from datasets_api.adapter import pinned_provenance_scope
from datasets_api.admission import MaterialisePermits, bounded_materialise_slot
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest import (
    assemble_manifest,
    collapsed_citation,
    coverage_facts_of,
    definition_version_record,
    healed_citation,
    persisted_manifest,
    provenance_refs_of,
    reshape_warning_models,
    stored_manifest_facts,
    year_coverage,
)
from datasets_api.manifest_models import (
    ArtifactEntryModel,
    CitationModel,
    CitationSource,
    Manifest,
    WarningModel,
    year_coverage_counted,
)
from datasets_api.mint_models import MintOptions
from datasets_api.origin import (
    catalogue_build_id,
    refuse_unserved_empty_probe,
    refuse_unserved_selection,
    un_unavailable_gaps,
    un_unavailable_warnings,
    withheld_finding,
)
from datasets_api.query_options import encode_query_options
from datasets_api.render import (
    DerivedVersion,
    artifact_rendering_keys,
    complete_rendering,
    incomplete_rendering,
    render_and_store,
    render_and_store_canonical_frame,
    render_and_store_series_statistics,
    stored_artifact_entry,
)
from datasets_api.timings import phase, record_elapsed
from datasets_api.version_descriptions import MintDescribers, mint_describers
from datasets_api.version_store import current_rendering_keys
from datasets_api.versions import (
    ArtifactRecord,
    HeadFence,
    HeadStore,
    MaterialisationRecord,
    NullHeadWriter,
    NullVersionRecordWriter,
    SubjectHead,
    Trigger,
    VersionDescriber,
    VersionRecord,
    VersionRecordWriter,
    VersionWrite,
    WrittenVersion,
    link_digest,
)
from dc_sdmx import SdmxClient
from dfo_platform.config import PublicSiteSettings
from materialise import (
    CollectionFacetProbe,
    EnrichmentLookup,
    MaterialisationRequest,
    MaterialisationResult,
    RowProgress,
    materialise_group,
)
from materialise.bundle import ReproducingCall
from materialise.continued_series import ContinuedSeries, continued_series_of_trace
from materialise.errors import DefaultFacetUndefinedError
from materialise.graph import GraphReader
from materialise.profile import OutputProfile, compute_output_profile_from_chunks
from materialise.provenance_doc import (
    SourceCitation,
    read_source_citations,
)
from materialise.rate_tables import RateTableCache
from materialise.statistics import (
    SeriesFigures,
    SeriesStatisticsFile,
    SeriesStatisticsSink,
    compute_series_statistics_table,
)
from materialise.window import Coverage
from origin_catalogue import current_catalogue, recording_un_blocked_places
from series_coverage.index import CoverageIndex, Estimate
from spec_registry.reader import StoredDefinition

logger = logging.getLogger(__name__)

DEFINITION_ROUTE_LIVE = True
"""Whether the definition retrieval route (`GET /api/datasets/{id}`) is
mounted on this deployment. A bundle's reproducing call is only published
once that route exists to serve it. `datasets_api.doors.definition`
mounts it unconditionally, so this is `True` for every deployment that
composes `datasets_api.app.datasets_surface`."""

# Logs every canonical frame store failure once, with `spec_id` and `content_version`, whether it
# fails the serve alone or beside another write's failure. The value is the name the alert matches.
CANONICAL_FRAME_STORE_FAILED_MARKER = "frame1_store_failed"

# Logs an artifact linked as incomplete because a graph metadata read failed, with the subject,
# `content_version` and cited refs. The next render of that version reads again and links a
# complete artifact beside it on success.
BUNDLE_LINKED_INCOMPLETE_MARKER = "bundle_linked_incomplete"

_RAW_FETCH_GUARD_MARGIN = 1.5
"""The 1.5 margin covers rows added between the exact coverage snapshot and live SDMX fetch.
Coverage is exact for covered StatVars and overestimates `latest` windows or `ContainedIn`-bounded
roles. Admission runs before fetch. The 50% mid-fetch backstop targets genuine response growth.
"""


class WorkDeadlinePassed(RuntimeError):
    """The run's work deadline passed before a write could start. It is unclassified, so the
    worker takes its transient path."""


def _check_work_deadline(deps: ServeDeps, step: str) -> None:
    """Raise `WorkDeadlinePassed` when `deps.work_deadline` has passed before `step`."""
    if deps.work_deadline is not None and time.monotonic() >= deps.work_deadline:
        raise WorkDeadlinePassed(f"the run's work deadline passed before {step}")


def raw_fetch_guard(estimate: Estimate, *, floor: int) -> int:
    """Returns `max(floor, ceil(estimate.rows * _RAW_FETCH_GUARD_MARGIN))` for the materialiser's
    mid-fetch row guard and post-fetch ceiling. `floor` covers uncovered requests and calls lacking
    an estimate. A zero estimate can mean missing coverage or a covered request with no data.
    Definition option rerenders use the floor: `refuse_inline_over_ceiling` admits only an estimate
    at or below it."""
    return max(floor, math.ceil(estimate.rows * _RAW_FETCH_GUARD_MARGIN))


def refs_from_resolved_stat_vars(entries: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    """Extracts refs from live or stored `resolved_stat_vars` trace entries. Both materialised and
    stored-rendering paths use this function."""
    return tuple(cast(str, entry["ref"]) for entry in entries)


def _resolved_stat_var_refs(result: MaterialisationResult) -> tuple[str, ...]:
    """Returns every resolved `dcid:` StatVar, including zero-row members, from the materialisation
    trace. Groupby uses these refs for population checks."""
    return refs_from_resolved_stat_vars(result.trace.resolved_stat_vars)


@dataclass(frozen=True)
class ServeDeps:
    sdmx: SdmxClient
    graph: GraphReader
    artifact_store: ArtifactStore
    materialiser_build: str
    capability_version: str
    # Default mid-fetch row guard. `raw_fetch_guard` scales it to each coverage estimate.
    ceiling: int
    member_concurrency: int
    enrichment: EnrichmentLookup
    # The code index this request reads, used by rendering and manifest assembly.
    code_index: CodeIndex
    # One process-wide rate-table cache shared by materialisations.
    rate_cache: RateTableCache
    version_writer: VersionRecordWriter = field(default_factory=NullVersionRecordWriter)
    # Reads the version record `version_writer` persists, before rendering, when the caller's
    # head names the result's version, so a re-mint renders its bundle from the stored citation
    # (`_persisted_state`). `None` reads nothing.
    snapshot_reader: SnapshotReader | None = None
    # Records each run and advances the head to it. Callers read the head before materialising.
    # The default writer reads nothing and treats writes as no-ops.
    head_store: HeadStore = field(default_factory=NullHeadWriter)
    # Bounds the frames alive at once across every door sharing the semaphore. A serve holds one
    # permit for its frame's whole life. `None` disables the bound.
    materialise_semaphore: MaterialisePermits | None = None
    # Absolute monotonic deadline for the permit wait, SDMX fetches, and every graph read: the
    # citation read, the documentation read, and the code label read. `None` means no deadline.
    deadline: float | None = None
    # Absolute monotonic deadline of a whole worker run, checked before the version record and
    # before each artifact render and upload. `None` checks nothing.
    work_deadline: float | None = None
    # A worker run's operation claim, checked in the head advance's own transaction
    # (`HeadWriter.advance_head`). `None` fences nothing.
    head_fence: HeadFence | None = None
    # Supplies queue-wait and pivot-cell caps plus capability limits embedded in manifests. The
    # application passes it explicitly. The default serves direct construction.
    settings: DatasetsBuildSettings = field(default_factory=DatasetsBuildSettings.from_env)


def _persisted_state(
    snapshot_reader: SnapshotReader | None,
    *,
    subject_id: str,
    content_version: str,
    head_content_version: str | None,
) -> VersionState | None:
    """The stored version and its linked locations when the head names `content_version`, so
    the version is on record, else `None` with no read.

    A version on record that the head does not name, such as one a concurrent first mint just
    recorded, is found by the write transaction instead (`VersionRecordWriter.write_version`),
    and the write half renders again from it.
    """
    if snapshot_reader is None or content_version != head_content_version:
        return None
    with phase("snapshot_read"):
        return snapshot_reader.read_version_state(subject_id, content_version)


def _render_citation(persisted: StoredVersion | None, *, assembled: CitationModel) -> CitationModel:
    """The stored record's citation when the version was already on record, else `assembled`.

    A re-mint then cites, names and dates its bundle as the first mint did.
    """
    if persisted is None:
        return assembled
    return CitationModel.model_validate(stored_manifest_facts(persisted).citation)


def _render_prepared_at(persisted: StoredVersion | None, *, assembled: datetime) -> datetime:
    """The stored record's `prepared_at` when the version was already on record, else
    `assembled`, as `_render_citation` chooses the citation."""
    if persisted is None:
        return assembled
    return stored_manifest_facts(persisted).prepared_at


def _prepared_now() -> datetime:
    """The `prepared_at` a version gets at its first mint: now in UTC, to the second."""
    return utc_now().replace(microsecond=0)


def persisted_response(
    *,
    subject_id: str,
    persisted: StoredVersion,
    locations: Sequence[StoredArtifactLocation],
    served: Manifest,
    call_warnings: Sequence[WarningModel],
    now: datetime,
    enrichment_digest: str | None,
) -> Manifest:
    """What a write returns when its version record already existed: the stored version.

    The response then equals a snapshot read of that version (`persisted_manifest`).
    `persisted` is the stored record and `locations` the locations linked to it, read after
    this write's links.

    The record stays as stored. The write still links this call's artifacts, each
    under its own `rendering_key`; an existing key keeps its artifact while that one is
    servable, and the response lists whichever the store links (`with_linked_artifacts`).
    `enrichment_digest` is `with_linked_artifacts`'s.
    """
    return with_linked_artifacts(
        persisted_manifest(stored=persisted, served=served, call_warnings=call_warnings),
        subject_id=subject_id,
        locations=locations,
        now=now,
        enrichment_digest=enrichment_digest,
    )


def with_linked_artifacts(
    manifest: Manifest,
    *,
    subject_id: str,
    locations: Sequence[StoredArtifactLocation],
    now: datetime,
    enrichment_digest: str | None,
) -> Manifest:
    """`manifest` with each artifact replaced by the one linked under its rendering key in
    `locations`, the version's linked locations, when that one is servable
    (`select_servable_locations`), listed as a snapshot read lists it. An artifact whose key
    has no servable link stays as this call rendered it. An incomplete artifact
    (`incomplete_artifact`) gives way to a servable artifact linked under its complete
    rendering, and otherwise to the one linked under its own.

    `enrichment_digest` is the serving code index's (`CodeIndex.enrichment_digest`). A file
    with party-kind cells is looked up under the keys it salts
    (`version_store.current_rendering_keys`), so one linked under an older digest is not
    listed: this call's render of it, linked under the current one, is. Each listed file
    carries the digest as its `enrichment_stamp` when it holds party-kind cells read under it
    (`render.stored_artifact_entry`), as this call's render does. With no digest the keys are
    unsalted and no file carries a stamp.

    A linked artifact stays linked while it is servable, so every mint, re-mint and head hit of
    a version lists the same bytes under one key, whatever this call's render read from the
    graph; once it is not, the next linked render replaces it (`version_store`). Every write
    renders from the stored record, so a linked artifact carries the record's `prepared_at` and
    citation whichever writer linked it.
    """
    if not manifest.artifacts:
        return manifest
    servable = {
        location.rendering_key: location
        for location in select_servable_locations(locations, now=now)
    }
    artifacts: list[ArtifactEntryModel] = []
    for artifact in manifest.artifacts:
        location = next(
            (
                servable[key]
                for rendering in (complete_rendering(artifact.rendering), artifact.rendering)
                for key in current_rendering_keys(rendering, enrichment_digest)
                if key in servable
            ),
            None,
        )
        artifacts.append(
            artifact
            if location is None
            else stored_artifact_entry(
                location,
                subject_id=subject_id,
                content_version=manifest.content_version,
                enrichment_digest=enrichment_digest,
            )
        )
    return manifest.model_copy(update={"artifacts": artifacts})


def bundle_linked(
    locations: Sequence[StoredArtifactLocation],
    *,
    options: MintOptions,
    site: PublicSiteSettings,
    now: datetime,
    enrichment_digest: str | None,
) -> bool:
    """Whether `locations`, a version's linked locations, hold a servable artifact under the
    bundle rendering `options` asks for, keyed under `enrichment_digest`
    (`render.artifact_rendering_keys`). The response then lists that artifact
    (`with_linked_artifacts`), so this call's render is discarded and needs no live
    documentation read. `False` for a bare rendering."""
    if not options.bundle:
        return False
    keys = artifact_rendering_keys(options, site=site, enrichment_digest=enrichment_digest)
    return any(
        location.rendering_key in keys for location in select_servable_locations(locations, now=now)
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
    coverage_index: CoverageIndex | None,
) -> tuple[dict[str, CitationSource], bool]:
    """The sources a render cites by ref, and whether the graph reads behind them answered.

    Only a bundle carries documentation and licences. For a version already on record (`on_record`)
    whose bundle is not yet linked (`linked`, from `bundle_linked`) they come from
    `healed_citation`, read within `deadline`. A linked bundle is what the response lists, so the
    render cites the stored citation as it is. A first mint renders what it just read, and
    `documentation_complete` says whether those reads answered.
    """
    sources = citation.sources_by_ref()
    if not options.bundle or linked:
        return sources, True
    if not on_record:
        return sources, documentation_complete
    healed, complete = healed_citation(
        citation,
        graph=graph,
        deadline=deadline,
        coverage_index=coverage_index,
    )
    return healed.sources_by_ref(), complete


def incomplete_artifact(
    artifact: ArtifactEntryModel,
    *,
    subject_id: str,
    content_version: str,
    refs: Sequence[str],
    documentation_complete: bool,
    labels_complete: bool,
) -> ArtifactEntryModel:
    """`artifact`, rendered with a failed graph metadata read, under its incomplete rendering
    (`render.incomplete_rendering`), with a `BUNDLE_LINKED_INCOMPLETE_MARKER` warning logged.

    The caller links it under that rendering, so the download link the response lists
    resolves and the digest can be pinned. No request's rendering key matches it, so the next
    render of the version reads the graph again and links a complete artifact beside it.
    """
    logger.warning(
        "%s subject_id=%s content_version=%s refs=%s documentation_read_complete=%s "
        "label_read_complete=%s",
        BUNDLE_LINKED_INCOMPLETE_MARKER,
        subject_id,
        content_version,
        list(refs),
        documentation_complete,
        labels_complete,
    )
    return artifact.model_copy(update={"rendering": incomplete_rendering(artifact.rendering)})


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
    # The provenance refs of this call's trace (`provenance_refs_of`).
    provenance_refs: list[str]
    spec_dict: dict[str, object]
    # All requests share a role signature, so rendering uses the first request's roles.
    entity_columns: tuple[str, ...]
    # Carried into final manifest assembly so both halves use the same stamp.
    coverage_stamp: str | None = None
    # The stored version and its linked locations when the caller's head names this call's
    # `content_version` (`_persisted_state`), else `None`.
    persisted: VersionState | None = None
    # `False` when a documentation read in `source_citations` failed.
    documentation_complete: bool = True
    # The coverage snapshot materialisation read, for the documentation a bundle render reads.
    coverage_index: CoverageIndex | None = None
    # The year members of the version's `coverage` (`manifest.year_coverage`), counted here
    # from the index the run elected over and joined to every manifest the write half builds.
    year_coverage: Mapping[str, object] = field(default_factory=dict)


def _documentation_complete(source_citations: Mapping[str, SourceCitation]) -> bool:
    return not any(citation.documentation_read_failed for citation in source_citations.values())


def _prewrite_citations(
    graph: GraphReader,
    refs: Sequence[str],
    *,
    documented: bool,
    deadline: float | None,
    coverage_index: CoverageIndex | None,
) -> dict[str, SourceCitation]:
    """A prewrite's `read_source_citations` over `refs`, within `deadline`. A documentation read
    that fails links the bundle as incomplete (`incomplete_artifact`), so a later render reads
    again."""
    return read_source_citations(
        graph,
        refs,
        catalogue=current_catalogue(),
        deadline=deadline,
        documented=documented,
        coverage_facts_of=coverage_facts_of(coverage_index),
    )


def prewrite_definition(
    *,
    stored: StoredDefinition,
    requests: Sequence[MaterialisationRequest],
    sdmx: SdmxClient,
    graph: GraphReader,
    ceiling: int,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
    code_index: CodeIndex,
    rate_cache: RateTableCache,
    member_concurrency: int,
    settings: DatasetsBuildSettings,
    precomputed_probe: CollectionFacetProbe | None = None,
    deadline: float | None = None,
    coverage_stamp: str | None = None,
    coverage_index: CoverageIndex | None = None,
    progress: RowProgress | None = None,
    snapshot_reader: SnapshotReader | None = None,
    head_content_version: str | None = None,
    documented: bool = True,
) -> DefinitionPrewrite:
    """Materialises one pinned request or the resolved members of a described-arm spec into one
    frame, version, and trace. It returns before artifact writes. The options route answers with
    its manifest.

    The caller holds a materialise permit across this call and every write of the frame it
    returns (`serve_definition`). A supplied `precomputed_probe` reuses a facet probe the caller
    already made. The absolute deadline bounds SDMX fetches and graph reads, the citation read
    included. `member_concurrency` limits concurrent member fetches.

    The same `code_index` builds the manifest and renders output. `coverage_stamp` adds lineage.
    Materialisation determines identity. Role-bearing requests require `coverage_index`. Its absence
    raises `MaterialisationUsageError` before fetching. Subject-only requests may omit it.
    `progress` forwards worker progress. Profile CPU is recorded separately from materialisation.

    A result whose facet probe was empty (`MaterialisationResult.probe_empty`) refuses here,
    before anything is written, when the UN could not serve its places, the registry
    withholds their data, or the read that would tell failed
    (`origin.refuse_unserved_empty_probe`). Otherwise it is the empty result, with zero rows,
    `empty_result`, and a citation with no source. A member's `default_facet_undefined`
    refusal becomes the UN or registry refusal by the same two checks
    (`origin.refuse_unserved_selection`).

    The citation read takes each source's documentation only when `documented` and the
    version is not on record. `snapshot_reader` is read, in one transaction, only when the
    caller's head names the result's `content_version` (`head_content_version`). A re-mint
    renders from the stored citation (`bundle_citations`).

    Every read runs under the identity's pinned provenances
    (`adapter.pinned_provenance_scope`), so a pinned source is admitted before DFO
    precedence and the withheld re-reads name no source the pin leaves out.
    """
    with pinned_provenance_scope(stored.identity):
        with recording_un_blocked_places() as un_blocked:
            try:
                with phase("materialise"):
                    result = materialise_group(
                        requests,
                        sdmx=sdmx,
                        graph=graph,
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
            except DefaultFacetUndefinedError as exc:
                if exc.request is not None:
                    refuse_unserved_selection(
                        un_blocked, exc.request, graph, availability=sdmx, deadline=deadline
                    )
                raise
            if result.probe_empty:
                refuse_unserved_empty_probe(
                    un_blocked, requests, graph, availability=sdmx, deadline=deadline
                )
        record_elapsed("profile", result.output_profile_seconds)
        provenance_refs = provenance_refs_of(result.trace.facets)
        persisted = _persisted_state(
            snapshot_reader,
            subject_id=stored.spec_id,
            content_version=result.content_version,
            head_content_version=head_content_version,
        )
        with phase("citation"):
            source_citations = _prewrite_citations(
                graph,
                provenance_refs,
                documented=documented and persisted is None,
                deadline=deadline,
                coverage_index=coverage_index,
            )
        # `refuse_unserved_empty_probe` already made the withheld re-read an empty probe gets.
        withheld = (
            None
            if result.probe_empty
            else withheld_finding(
                graph,
                [(request, result.coverage) for request in requests],
                default_scope=result.default_scope,
                deadline=deadline,
            )
        )
        origin_warnings = (() if withheld is None else (withheld[1],)) + un_unavailable_warnings(
            un_blocked
        )
        # Every manifest built from `result` records the places those warnings name.
        origin_gaps = (() if withheld is None else (withheld[2],)) + un_unavailable_gaps(un_blocked)
        result = replace(
            result,
            trace=replace(
                result.trace, origin_gaps=tuple(gap.as_trace_entry() for gap in origin_gaps)
            ),
        )
        spec_dict: dict[str, object] = {
            "schema_version": stored.schema_version,
            "spec_id": stored.spec_id,
            "identity": stored.identity,
        }
        years = year_coverage(result, requests=requests, index=coverage_index)
        manifest = assemble_manifest(
            stored=stored,
            materialisation=result,
            artifacts=[],
            provenance_refs=provenance_refs,
            source_citations=source_citations,
            prepared_at=_prepared_now(),
            capability_version=capability_version,
            materialiser_build=materialiser_build,
            code_index=code_index,
            settings=settings,
            extra_warnings=origin_warnings,
            coverage_stamp=coverage_stamp,
            year_coverage=years,
        )
        return DefinitionPrewrite(
            stored=stored,
            result=result,
            manifest=manifest,
            source_citations=source_citations,
            origin_warnings=origin_warnings,
            provenance_refs=provenance_refs,
            spec_dict=spec_dict,
            entity_columns=requests[0].declared_roles,
            coverage_stamp=coverage_stamp,
            persisted=persisted,
            documentation_complete=_documentation_complete(source_citations),
            coverage_index=coverage_index,
            year_coverage=years,
        )


@dataclass(frozen=True)
class ServedDefinition:
    """A serve's response and the head its run advanced to, `None` when a concurrent writer
    advanced it first."""

    manifest: Manifest
    head: SubjectHead | None


def serve_definition(
    *,
    stored: StoredDefinition,
    requests: Sequence[MaterialisationRequest],
    options: MintOptions | None,
    deps: ServeDeps,
    trigger: Trigger,
    trace_id: str | None = None,
    expected_head: SubjectHead | None = None,
    precomputed_probe: CollectionFacetProbe | None = None,
    ceiling: int | None = None,
    coverage_stamp: str | None = None,
    coverage_index: CoverageIndex | None = None,
    progress: RowProgress | None = None,
) -> ServedDefinition:
    """Materialises with `prewrite_definition`, then writes artifacts and the version record,
    and records the run as a materialisation under `trigger`, advancing the head in the same
    transaction as that record. The caller reads the head and passes it as `expected_head`
    (`None` for no head). The head advances only while it names the run `expected_head` names,
    and the prewrite reads the stored version only when that head names the result.
    `precomputed_probe` and `progress` are forwarded. `options` `None` records the version, its
    canonical frame and statistics, renders nothing, and returns the prewrite manifest,
    with no artifacts.

    `ceiling` overrides the raw-fetch guard. `None` uses `deps.ceiling`. `coverage_stamp` adds
    manifest lineage, while `coverage_index` feeds materialisation.

    One materialise permit is held from before materialisation to after the last artifact write,
    waiting up to the earlier of `settings.materialise_wait_seconds` and `deps.deadline`. A full
    semaphore, or a swap that replaced the request's bound release, raises retryable
    `MintBackpressure` (503) before anything is fetched. The prewrite and its frame are bound
    only in the function the slot calls, so they are freed before the permit is released.
    """

    def _materialise_and_record() -> ServedDefinition:
        prewrite = prewrite_definition(
            stored=stored,
            requests=requests,
            sdmx=deps.sdmx,
            graph=deps.graph,
            ceiling=ceiling if ceiling is not None else deps.ceiling,
            materialiser_build=deps.materialiser_build,
            capability_version=deps.capability_version,
            enrichment=deps.enrichment,
            code_index=deps.code_index,
            rate_cache=deps.rate_cache,
            member_concurrency=deps.member_concurrency,
            settings=deps.settings,
            precomputed_probe=precomputed_probe,
            deadline=deps.deadline,
            coverage_stamp=coverage_stamp,
            coverage_index=coverage_index,
            progress=progress,
            snapshot_reader=deps.snapshot_reader,
            head_content_version=(
                expected_head.content_version if expected_head is not None else None
            ),
        )
        manifest = _write_definition_artifacts(
            prewrite=prewrite, options=options, deps=deps, trace_id=trace_id
        )
        head = _record_run(
            prewrite,
            deps=deps,
            trigger=trigger,
            trace_id=trace_id,
            expected_materialisation_id=(
                expected_head.materialisation_id if expected_head is not None else None
            ),
        )
        return ServedDefinition(manifest=manifest, head=head)

    with bounded_materialise_slot(
        deps.materialise_semaphore,
        wait_seconds=deps.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        return _materialise_and_record()


def _base_record(
    prewrite: DefinitionPrewrite,
    *,
    persisted: VersionState | None,
    artifacts: Sequence[ArtifactRecord],
    describer: VersionDescriber | None = None,
) -> VersionRecord:
    """The version record of `prewrite`'s materialisation, linking `artifacts` and described by
    `describer`. It takes the stored body when the version is on record (`persisted`), else
    this call's. A stored body whose year coverage was never counted
    (`year_coverage_counted`) takes this call's `coverage`, which the write stores on the
    version (`VersionRecordWriter.write_version`)."""
    record = definition_version_record(
        manifest=prewrite.manifest,
        stored=prewrite.stored,
        materialisation=prewrite.result,
        artifacts=[],
        record_only_artifacts=artifacts,
    )
    record = replace(record, describer=describer)
    if persisted is None:
        return record
    facts = persisted.version.facts
    if isinstance(facts, ManifestFacts) and not year_coverage_counted(facts.coverage):
        assert isinstance(record.facts, ManifestFacts)
        facts = replace(facts, coverage=record.facts.coverage)
    return replace(record, facts=facts)


def _describers(
    prewrite: DefinitionPrewrite, *, deps: ServeDeps, rendered: _RenderPass | None
) -> MintDescribers:
    """The describers of the versions this serve writes: its own and the derived version
    `rendered` holds. Their labels are read here, before the write transaction."""
    collapsed = None
    if rendered is not None and rendered.collapsed is not None:
        assert rendered.collapsed_figures is not None
        collapsed = (rendered.collapsed, rendered.collapsed_figures)
    return mint_describers(
        _base_record(prewrite, persisted=prewrite.persisted, artifacts=()),
        figures=prewrite.result.series_statistics.figures,
        collapsed=collapsed,
        collapsed_summed=rendered is None or rendered.collapsed_summed,
        code_index=deps.code_index,
        enrichment=deps.enrichment,
        deadline=deps.deadline,
    )


def _frame_result(
    frame_future: Future[ArtifactRecord], prewrite: DefinitionPrewrite
) -> ArtifactRecord:
    """The stored canonical frame, joined from its worker. An `ArtifactStoreError` from its
    write is logged as `CANONICAL_FRAME_STORE_FAILED_MARKER` before it propagates."""
    try:
        return frame_future.result()
    except ArtifactStoreError:
        logger.error(
            "%s spec_id=%s content_version=%s",
            CANONICAL_FRAME_STORE_FAILED_MARKER,
            prewrite.stored.spec_id,
            prewrite.result.content_version,
            exc_info=True,
        )
        raise


def _failed_frame_result(
    frame_future: Future[ArtifactRecord], prewrite: DefinitionPrewrite
) -> ArtifactRecord | None:
    """The stored canonical frame, joined from its worker while a primary failure propagates,
    or `None` when its write failed too. Every frame failure is logged (`_frame_result` logs a
    store failure), so the primary one is raised."""
    try:
        return _frame_result(frame_future, prewrite)
    except ArtifactStoreError:
        return None
    except Exception:
        logger.exception(
            "canonical frame write also raised while the primary write failed "
            "spec_id=%s content_version=%s",
            prewrite.stored.spec_id,
            prewrite.result.content_version,
        )
    return None


def _write_definition_artifacts(
    *,
    prewrite: DefinitionPrewrite,
    options: MintOptions | None,
    deps: ServeDeps,
    trace_id: str | None,
) -> Manifest:
    """Stores the canonical frame, the series statistics and the requested rendering, then writes
    the version record, every artifact's inventory row and link, and any collapsed version in one
    transaction (`VersionRecordWriter.write_version`), and builds the response from what that
    transaction read back. Artifacts are stored through a `deferred` view of the artifact store,
    so no artifact write opens a transaction, and the frame worker runs beside the statistics
    and the requested rendering. The wire warning order is materialisation, reshape,
    citation, then facet defaults. Manifest assembly builds its citation from
    `prewrite.source_citations` and makes no graph read.

    The render dates and cites as the stored record when the prewrite found the version on
    record, else as this call's manifest. When a concurrent first mint recorded the version in
    between, the transaction links only the canonical frame and statistics, and the requested
    rendering renders again from the stored record and links in a second transaction, so every
    linked artifact dates and cites as its record. When the requested rendering raises and the
    canonical frame is stored, the version is recorded with its canonical frame and statistics
    before the failure propagates. A version is recorded only with its canonical frame and
    statistics. A failed canonical frame write records nothing, logs
    `CANONICAL_FRAME_STORE_FAILED_MARKER` and leaves the head where it was. `options` `None`
    renders nothing and returns `prewrite.manifest`."""
    _check_work_deadline(deps, "rendering the canonical frame and the series statistics")
    stored = prewrite.stored
    result = prewrite.result
    batch = InventoryBatch()
    artifact_store = deps.artifact_store.deferred(batch)
    # Copy the caller's timing and trace context before submission. Workers use separate contexts.
    frame_context = contextvars.copy_context()
    statistics_artifact: ArtifactEntryModel | None = None
    rendered: _RenderPass | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="datasets-frame") as frame_pool:
        frame_future = frame_pool.submit(
            frame_context.run,
            render_and_store_canonical_frame,
            frame_chunks=result.frame_chunks(),
            subject_id=stored.spec_id,
            content_version=result.content_version,
            artifact_store=artifact_store,
        )
        try:
            statistics_artifact = render_and_store_series_statistics(
                data=result.series_statistics.data,
                subject_id=stored.spec_id,
                content_version=result.content_version,
                artifact_store=artifact_store,
            )
            if options is not None:
                rendered = _render_pass(
                    prewrite,
                    options=options,
                    deps=deps,
                    artifact_store=artifact_store,
                    persisted=prewrite.persisted,
                    statistics_artifact=statistics_artifact,
                )
        except BaseException:
            frame_record = _failed_frame_result(
                cast("Future[ArtifactRecord]", frame_future), prewrite
            )
            if statistics_artifact is not None and frame_record is not None:
                try:
                    _check_work_deadline(deps, "recording the version")
                    describers = _describers(prewrite, deps=deps, rendered=None)
                    with phase("record"):
                        deps.version_writer.write_version(
                            VersionWrite(
                                record=_base_record(
                                    prewrite,
                                    persisted=prewrite.persisted,
                                    artifacts=_option_free(statistics_artifact, frame_record),
                                    describer=describers.version,
                                ),
                                inventory=batch.entries,
                            ),
                            trace_id=trace_id,
                        )
                except Exception:
                    logger.exception(
                        "recording the version also raised while the primary write failed "
                        "spec_id=%s content_version=%s",
                        stored.spec_id,
                        result.content_version,
                    )
            raise
        # A failed frame write raises here, before the version is recorded or the head advances.
        # `Context.run` hides the worker's return type from static analysis.
        frame_record = _frame_result(cast("Future[ArtifactRecord]", frame_future), prewrite)
    assert statistics_artifact is not None
    option_free = _option_free(statistics_artifact, frame_record)
    _check_work_deadline(deps, "recording the version")
    describers = _describers(prewrite, deps=deps, rendered=rendered)
    if rendered is None:
        with phase("record"):
            deps.version_writer.write_version(
                VersionWrite(
                    record=_base_record(
                        prewrite,
                        persisted=prewrite.persisted,
                        artifacts=option_free,
                        describer=describers.version,
                    ),
                    inventory=batch.entries,
                ),
                trace_id=trace_id,
            )
        return prewrite.manifest
    assert options is not None
    with phase("record"):
        written = deps.version_writer.write_version(
            rendered.write(option_free, inventory=batch.entries, describers=describers),
            trace_id=trace_id,
        )
    if not written.rendered:
        # A concurrent first mint recorded the version first. Render again from its record.
        rebatch = InventoryBatch()
        rendered = _render_pass(
            prewrite,
            options=options,
            deps=deps,
            artifact_store=deps.artifact_store.deferred(rebatch),
            persisted=written.state,
            statistics_artifact=statistics_artifact,
        )
        _check_work_deadline(deps, "recording the rendering")
        with phase("record"):
            written = deps.version_writer.write_version(
                rendered.write((), inventory=rebatch.entries, describers=describers),
                trace_id=trace_id,
            )
    return rendered.response(written, now=utc_now())


def _option_free(
    statistics_artifact: ArtifactEntryModel, frame_record: ArtifactRecord
) -> tuple[ArtifactRecord, ...]:
    """The artifacts every write links, whoever recorded the version. They are the series statistics
    and the canonical frame."""
    return (_artifact_record(statistics_artifact), frame_record)


def _artifact_record(artifact: ArtifactEntryModel) -> ArtifactRecord:
    """The link of `artifact`, rendered by this call, under the key its `enrichment_stamp`
    salts (`versions.link_digest`)."""
    return ArtifactRecord(
        artifact_digest=artifact.artifact_digest,
        rendering=artifact.rendering,
        media_type=artifact.media_type,
        enrichment_digest=link_digest(artifact.rendering, stamp=artifact.enrichment_stamp),
    )


@dataclass(frozen=True)
class _RenderPass:
    """The requested rendering of a version, rendered from one body (`persisted`, else the
    prewrite manifest), with the manifest it answers and what linking it writes."""

    prewrite: DefinitionPrewrite
    persisted: VersionState | None
    manifest: Manifest
    call_warnings: tuple[WarningModel, ...]
    # `CodeIndex.enrichment_digest`: the salt of the keys the response looks its links up under.
    enrichment_digest: str
    # Linked to the base version: the rendering, unless it belongs to a derived version.
    links: tuple[ArtifactRecord, ...]
    # A derived version (`render.DerivedVersion`), recorded with its artifacts, the series
    # figures of its statistics, and whether its frame sums the definition's variables.
    collapsed: VersionRecord | None
    collapsed_figures: SeriesFigures | None
    collapsed_summed: bool = True

    def write(
        self,
        option_free: Sequence[ArtifactRecord],
        *,
        inventory: Sequence[InventoryEntry],
        describers: MintDescribers,
    ) -> VersionWrite:
        return VersionWrite(
            record=_base_record(
                self.prewrite,
                persisted=self.persisted,
                artifacts=option_free,
                describer=describers.version,
            ),
            inventory=tuple(inventory),
            rendered=self.links,
            collapsed=None
            if self.collapsed is None
            else replace(self.collapsed, describer=describers.collapsed),
        )

    def response(self, written: WrittenVersion, *, now: datetime) -> Manifest:
        """The manifest this call answers. It is the call's own, with the store's links, when it
        recorded the served version, else the stored version (`persisted_response`)."""
        subject_id = self.prewrite.stored.spec_id
        locations = written.state.locations
        if written.created:
            return with_linked_artifacts(
                self.manifest,
                subject_id=subject_id,
                locations=locations,
                now=now,
                enrichment_digest=self.enrichment_digest,
            )
        return persisted_response(
            subject_id=subject_id,
            persisted=written.state.version,
            locations=locations,
            served=self.manifest,
            call_warnings=self.call_warnings,
            now=now,
            enrichment_digest=self.enrichment_digest,
        )


def _render_pass(
    prewrite: DefinitionPrewrite,
    *,
    options: MintOptions,
    deps: ServeDeps,
    artifact_store: ArtifactStore,
    persisted: VersionState | None,
    statistics_artifact: ArtifactEntryModel,
) -> _RenderPass:
    """Render and store the rendering `options` asks for from `persisted`'s facts when the
    version is on record, else from the prewrite manifest, and assemble the manifest it
    answers. A render whose graph metadata reads failed is linked as incomplete
    (`incomplete_artifact`), so the next render of this version reads again."""
    stored = prewrite.stored
    result = prewrite.result
    persisted_version = persisted.version if persisted is not None else None
    render_citation = _render_citation(persisted_version, assembled=prewrite.manifest.citation)
    render_prepared_at = _render_prepared_at(
        persisted_version, assembled=prewrite.manifest.prepared_at
    )
    citations, documentation_complete = bundle_citations(
        render_citation,
        options=options,
        on_record=persisted is not None,
        linked=persisted is not None
        and bundle_linked(
            persisted.locations,
            options=options,
            site=deps.settings.site,
            now=utc_now(),
            enrichment_digest=deps.code_index.enrichment_digest,
        ),
        documentation_complete=prewrite.documentation_complete,
        graph=deps.graph,
        deadline=deps.deadline,
        coverage_index=prewrite.coverage_index,
    )
    _check_work_deadline(deps, "rendering the requested rendering")
    rendered = render_and_store(
        frame_chunks=result.frame_chunks,
        options=options,
        spec=prewrite.spec_dict,
        provenance_refs=prewrite.provenance_refs,
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
        artifact_store=artifact_store,
        entity_columns=prewrite.entity_columns,
        enrichment=deps.enrichment,
        pivot_max_output_cells=deps.settings.pivot_max_output_cells,
        code_index=deps.code_index,
        resolved_stat_var_refs=_resolved_stat_var_refs(result),
        output_profile=result.output_profile,
        deadline=deps.deadline,
    )
    artifact, reshape_warnings, derived = rendered.artifact, rendered.warnings, rendered.derived
    if not (documentation_complete and rendered.labels_complete):
        artifact = incomplete_artifact(
            artifact,
            subject_id=stored.spec_id,
            content_version=(
                derived.content_version if derived is not None else result.content_version
            ),
            refs=prewrite.provenance_refs,
            documentation_complete=documentation_complete,
            labels_complete=rendered.labels_complete,
        )
    call_warnings = prewrite.origin_warnings + reshape_warning_models(reshape_warnings)
    # A derived version (a multi-contributor collapse, a party-totals cut) is a version of its
    # own. Record the base version separately so a plain GET can still resolve it. A rendering
    # of the base frame shares the base version. The head stays on the base materialisation,
    # and definition retrieval derives the derived version on a head hit.
    if derived is None:
        manifest = assemble_manifest(
            stored=stored,
            materialisation=result,
            artifacts=[artifact, statistics_artifact],
            provenance_refs=prewrite.provenance_refs,
            source_citations=prewrite.source_citations,
            prepared_at=render_prepared_at,
            capability_version=deps.capability_version,
            materialiser_build=deps.materialiser_build,
            code_index=deps.code_index,
            settings=deps.settings,
            extra_warnings=call_warnings,
            coverage_stamp=prewrite.coverage_stamp,
            year_coverage=prewrite.year_coverage,
        )
        return _RenderPass(
            prewrite=prewrite,
            persisted=persisted,
            manifest=manifest,
            call_warnings=call_warnings,
            enrichment_digest=deps.code_index.enrichment_digest,
            links=(_artifact_record(artifact),),
            collapsed=None,
            collapsed_figures=None,
        )
    derived_statistics = derived_statistics_file(
        derived,
        continuing=continued_series_of_trace(
            {"facets": result.trace.facets, "continuations": result.trace.continuations}
        ),
    )
    derived_result = replace(
        result,
        frame_sha256=derived.frame_sha256,
        content_version=derived.content_version,
        trace=replace(result.trace, party_totals_left_out=derived.party_totals_left_out),
        coverage=derived_coverage(result.coverage, derived),
        series_statistics=derived_statistics,
        output_profile=derived_output_profile(derived, enrichment=deps.enrichment),
    )
    _check_work_deadline(deps, "rendering the derived version's canonical frame")
    derived_frame_record = render_and_store_canonical_frame(
        frame_chunks=derived.frame_chunks(),
        subject_id=stored.spec_id,
        content_version=derived.content_version,
        artifact_store=artifact_store,
    )
    _check_work_deadline(deps, "rendering the derived version's series statistics")
    derived_statistics_artifact = render_and_store_series_statistics(
        data=derived_statistics.data,
        subject_id=stored.spec_id,
        content_version=derived.content_version,
        artifact_store=artifact_store,
    )
    # The derived identity cites and dates as its base record does, and so does every
    # concurrent writer of it, so its record is written with its artifacts.
    # `collapsed_head_manifest` does the same.
    manifest = assemble_manifest(
        stored=stored,
        materialisation=derived_result,
        artifacts=[artifact, derived_statistics_artifact],
        provenance_refs=prewrite.provenance_refs,
        source_citations=prewrite.source_citations,
        prepared_at=render_prepared_at,
        capability_version=deps.capability_version,
        materialiser_build=deps.materialiser_build,
        code_index=deps.code_index,
        settings=deps.settings,
        extra_warnings=call_warnings,
        coverage_stamp=prewrite.coverage_stamp,
        year_coverage=prewrite.year_coverage,
    )
    manifest = manifest.model_copy(
        update={
            "citation": collapsed_citation(render_citation, base=manifest)
            if derived.summed
            else render_citation
        }
    )
    return _RenderPass(
        prewrite=prewrite,
        persisted=persisted,
        manifest=manifest,
        call_warnings=call_warnings,
        enrichment_digest=deps.code_index.enrichment_digest,
        links=(),
        collapsed=definition_version_record(
            manifest=manifest,
            stored=stored,
            materialisation=derived_result,
            artifacts=[artifact, derived_statistics_artifact],
            record_only_artifacts=[derived_frame_record],
            base_content_version=result.content_version,
        ),
        collapsed_figures=derived_statistics.figures,
        collapsed_summed=derived.summed,
    )


def derived_statistics_file(
    derived: DerivedVersion, *, continuing: Sequence[ContinuedSeries]
) -> SeriesStatisticsFile:
    """The `series-statistics/1` file of `derived`'s frame, one statistics table per chunk,
    with `continuing` the continuations of its base's series."""
    sink = SeriesStatisticsSink(continuing=continuing)
    for chunk in derived.frame_chunks():
        sink.write(compute_series_statistics_table(chunk))
    return sink.finish()


def derived_output_profile(
    derived: DerivedVersion, *, enrichment: EnrichmentLookup
) -> OutputProfile:
    """The output profile of `derived`'s frame over its role columns."""
    return compute_output_profile_from_chunks(
        derived.frame_chunks, role_columns=derived.role_columns, enrichment=enrichment
    )


def derived_coverage(coverage: Coverage, derived: DerivedVersion) -> Coverage:
    """`coverage`, its base's, with `derived`'s row count and each dimension `derived`
    counts anew (`DerivedVersion.distinct`)."""
    return replace(
        coverage,
        rows=derived.rows,
        dimensions=tuple(
            {
                **dimension,
                "count": derived.distinct.get(str(dimension["label"]), dimension["count"]),
            }
            for dimension in coverage.dimensions
        ),
    )


def _materialisation_record(
    prewrite: DefinitionPrewrite, *, deps: ServeDeps, trigger: Trigger, trace_id: str | None
) -> MaterialisationRecord:
    """The `datasets.materialisation` row of the run `prewrite` holds."""
    result = prewrite.result
    trace = result.trace
    return MaterialisationRecord(
        subject_id=prewrite.stored.spec_id,
        content_version=result.content_version,
        trigger=trigger,
        materialiser_build=deps.materialiser_build,
        frame_grammar_version=trace.frame_grammar_version,
        code_index_stamp=deps.code_index.stamp.content_hash,
        origin_catalogue_build_id=catalogue_build_id(),
        coverage_stamp=prewrite.coverage_stamp,
        graph_generation=trace.graph_revision,
        dependencies=result.dependencies.as_json(),
        selection={"facets": list(trace.facets)},
        conversion={"entries": list(trace.conversion)} if trace.conversion is not None else None,
        trace_id=trace_id,
    )


def _record_run(
    prewrite: DefinitionPrewrite,
    *,
    deps: ServeDeps,
    trigger: Trigger,
    trace_id: str | None,
    expected_materialisation_id: int | None,
) -> SubjectHead | None:
    """Record the run and advance the subject's head to its version by compare-and-set on the
    observed run in the same transaction, returning the head written. A lost race keeps the run's
    row, leaves the head the winner wrote, and returns `None`. A worker run's `deps.head_fence`
    that no longer holds raises `HeadFenceLost` and writes neither."""
    result = prewrite.result
    with phase("head_advance"):
        materialisation = _materialisation_record(
            prewrite, deps=deps, trigger=trigger, trace_id=trace_id
        )
        written = deps.head_store.advance_head(
            materialisation=materialisation,
            expected_materialisation_id=expected_materialisation_id,
            fence=deps.head_fence,
        )
    if written is None:
        logger.debug(
            "head advance lost the race spec_id=%s expected_materialisation_id=%s "
            "content_version=%s",
            materialisation.subject_id,
            expected_materialisation_id,
            result.content_version,
        )
    return written
