"""`serve_current` serves the current version of a served definition. Every entry point that
can materialise calls it: the mint, `GET /api/datasets/{id}`, `/options`, the frame refresh,
and the operations worker.

`serve_current` owns the head rule. It reads the subject's head, decides whether the head is
current, and either serves the stored version the head names or materialises, records the
version and the run (`versions.MaterialisationRecord`), and advances the head by
compare-and-set in the run's transaction (`HeadStore.advance_head`). Every head,
version, and operation is keyed on the served definition's `spec_id`, the one
`kinds.spec_id.resolve_served_definition` returns.

A head is current for `head_recheck_interval` after it was confirmed, with no read. Past that
window, a head whose `materialised_at` is older than `head_max_age` rebuilds. Otherwise one
graph read of the `lastDataRefreshDate` of every source and conversion source the head's
dependency fingerprint names builds the fingerprint a run would record now
(`materialise.dependencies.refreshed_dependencies`), and `dependency_changes` compares it with
the head's source by source: a match confirms and serves the head, a difference rebuilds. A
head with no fingerprint rebuilds once. The fingerprint of a `latest` head also holds the year
the coverage index elected for each member, and the check elects again from the call's
coverage snapshot over the series the run counted, so the head rebuilds when the index elects
another year. A `latest` head whose fingerprint records no year rebuilds once when the index
elects one for a member, and holds when it elects none, as for a member outside the index. A
current head whose version is a frame of one year stored before versions held year coverage
(`manifest_models.year_coverage_counted`) rebuilds once too, inside the recheck window or past
it, and the rebuild stores that coverage on the version when it makes the same one
(`version_store.LiveVersionRecordWriter`). The
graph serving revision is never compared, since it moves on every graph restart. A rebuild
materialises the definition (`serve.serve_definition`), sharing one rebuild per subject in this
process (`admission.RebuildFlights`), and breaks a single-place facet tie on the process's
whole-variable breadth (`DefinitionDeps.variable_breadth`).

A call with no rendering (`options` `None`, the `/options` route) loads the head's version
without its artifacts, and a rebuild records the version, its canonical frame and statistics, and
advances the head, rendering nothing. A head hit whose requested rendering is not linked
renders it from the stored frame (`render.servable_frame_location`) under a materialise slot.
`serve_fresh_head` answers from a current head alone and never renders, so it runs on the read
executor and hands every other call to the work executor.

Each serve sets the `trigger` and `served_from` labels of the call's `datasets.timings` line, the
`rebuild_reason` label of a rebuild, and starts the counts the clients, the store, the signer and
the permit wait add to (`dfo_platform.timings.count`): `sdmx_requests`, `graph_requests`,
`db_transactions`, `sign_blob_calls` and `permit_wait_ms`. The reason is the fingerprint members
that differed, joined by commas (`dependency_changes`), `schema` for a head with no readable
fingerprint, `max_age`, `no_head`, or `unservable` for a current head that cannot answer the call.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import cast

from artifact_store import ArtifactStore, Tier
from artifact_store.errors import ArtifactIntegrityError, ArtifactNotFoundError
from code_index.query import CodeIndex
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import SnapshotReader, StoredArtifactLocation, StoredVersion
from datasets_api.adapter import definition_read_scope, identity_composite
from datasets_api.admission import (
    MaterialisePermits,
    RebuildFlights,
    bounded_materialise_slot,
    elected_provenances,
    identity_role_keys,
    refuse_excel_over_estimate,
    refuse_excel_over_stored_rows,
    refuse_inline_over_ceiling,
    sync_estimate,
    within_sync_budget,
    xlsx_splits_totals,
)
from datasets_api.capabilities import build_capabilities
from datasets_api.codes_materialise import ExcludedByRegistry
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.snapshot import read_snapshot_currency, read_snapshot_manifest
from datasets_api.manifest import (
    collapsed_head_manifest,
    provenance_refs_of,
    record_collapsed_head_version,
    reshape_warning_models,
    source_refs_of,
    stored_manifest_facts,
    stored_output_profile,
)
from datasets_api.manifest_models import ArtifactEntryModel, Manifest, year_coverage_counted
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import (
    NullOperationStore,
    OperationBacklogFull,
    OperationStore,
)
from datasets_api.operations.submit import submit_over_budget
from datasets_api.operations.task_client import NullTaskClient, TaskClient
from datasets_api.output_options import derive_output_options
from datasets_api.query_options import encode_query_options
from datasets_api.recorded_rendering import serving_frames
from datasets_api.render import (
    DerivedVersion,
    FrameMode,
    artifact_rendering_keys,
    entity_roles,
    is_incomplete_rendering,
    is_statistics_rendering,
    refuse_pivot_over_stored_caps,
    render_and_store,
    render_and_store_canonical_frame,
    render_and_store_series_statistics,
    servable_frame_location,
    stored_frame_sha256,
)
from datasets_api.serve import (
    DEFINITION_ROUTE_LIVE,
    ServedDefinition,
    ServeDeps,
    bundle_citations,
    derived_output_profile,
    derived_statistics_file,
    incomplete_artifact,
    persisted_response,
    raw_fetch_guard,
    refs_from_resolved_stat_vars,
    serve_definition,
    with_linked_artifacts,
)
from datasets_api.timings import count, label, phase
from datasets_api.version_descriptions import collapse_describer
from datasets_api.versions import (
    ArtifactRecord,
    HeadFence,
    HeadStore,
    NullHeadWriter,
    NullVersionRecordWriter,
    SubjectHead,
    Trigger,
    VersionLister,
    VersionRecordWriter,
    link_digest,
)
from dc_sdmx import SdmxClient
from dfo_platform.problem import Refusal
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    RowProgress,
)
from materialise.bundle import ReproducingCall
from materialise.codes import MaterialisationWarning
from materialise.continued_series import continued_series_of_trace
from materialise.dependencies import (
    decode_dependencies,
    dependency_changes,
    refreshed_dependencies,
)
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from materialise.facets import VariableBreadthReader, bound_variable_breadth
from materialise.formats import (
    Frame,
    frame_entity_columns,
    read_stored_frame,
)
from materialise.graph import GraphReader
from materialise.profile import OutputProfile
from materialise.projection import totals_axes
from materialise.rate_tables import RateTableCache
from series_coverage.index import CoverageIndex, Estimate
from series_coverage.shared import CoverageSnapshot, ServedCoverage
from spec_registry import SpecRegistry
from spec_registry.reader import SpecReader, StoredDefinition
from statvar_members import MemberEnumerator

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DefinitionDeps:
    """What `serve_current` and the definition, refresh, and worker paths read and write.

    Serving updates `head_store` and `version_writer`, and `registry` registers a served
    identity `kinds.spec_id.resolve_served_definition` finds unregistered.
    `doors.definition.definition_deps` builds one per request from the surface's `Deps`, the
    mint builds one from its `MintDeps`, and the worker builds its own.
    """

    registry: SpecRegistry
    spec_reader: SpecReader
    graph: GraphReader
    sdmx: SdmxClient
    artifact_store: ArtifactStore
    materialiser_build: str
    capability_version: str
    ceiling: int
    member_concurrency: int
    # The process's shared member enumerator (`statvar_members.shared_enumerator`).
    enumerator: MemberEnumerator
    # Process-wide country-code and label cache, built with the graph and passed to `ServeDeps`.
    enrichment: EnrichmentLookup
    # The code index this request or run reads, passed to `ServeDeps`.
    code_index: CodeIndex
    # Shared by re-materialisations and passed to `ServeDeps`.
    rate_cache: RateTableCache
    # Used for coverage estimates before any SDMX call.
    coverage: ServedCoverage
    version_writer: VersionRecordWriter = field(default_factory=NullVersionRecordWriter)
    # Freshness reads and confirms the head, and a rebuild advances it. Re-materialisation receives
    # the observed head as `expected_head` to prevent overwriting a concurrent update. The null
    # default makes every request re-materialise.
    head_store: HeadStore = field(default_factory=NullHeadWriter)
    # Loads head responses through the snapshot route's loader, preserving manifest and
    # artifact-selection behaviour.
    snapshot_reader: SnapshotReader = field(default_factory=lambda: cast(SnapshotReader, None))
    # Bounds re-materialisations across doors sharing this semaphore. `None` means unbounded.
    materialise_semaphore: MaterialisePermits | None = None
    # How long a confirmed head serves with no read, and the age past which it rebuilds
    # regardless, from resolved settings, converted to `timedelta` at door construction.
    head_recheck_interval: timedelta
    head_max_age: timedelta
    # Injectable clock for freshness age comparisons.
    now: Clock = utc_now
    # Per-request monotonic deadline forwarded to SDMX. The earlier of it and the transport
    # timeout applies.
    deadline: float | None = None
    # A worker run's absolute monotonic work deadline, passed to `ServeDeps.work_deadline` and
    # checked before every write. `None` checks nothing.
    work_deadline: float | None = None
    # A worker run's operation claim, passed to `ServeDeps.head_fence` so the head advance
    # refuses once another attempt owns the operation. `None` fences nothing.
    head_fence: HeadFence | None = None
    # Records over-budget operations. The task client dispatches them.
    operation_store: OperationStore = field(default_factory=NullOperationStore)
    task_client: TaskClient = field(default_factory=NullTaskClient)
    # Supplies sync/async budgets, semaphore wait time, and pivot cell cap. Door construction
    # injects it. The default serves direct `DefinitionDeps` callers.
    settings: DatasetsBuildSettings = field(default_factory=DatasetsBuildSettings.from_env)
    # One head-advancing rebuild per subject at a time in this process (`serve_current`).
    # Per-request copies of these deps share it.
    rebuilds: RebuildFlights = field(default_factory=RebuildFlights)
    # Lists a subject's stored versions for the frame refresh, which looks up the collapsed
    # version already recorded under a head. `None` lists nothing.
    version_lister: VersionLister | None = None
    # Bound around each materialisation, so a request binding one place breaks a facet tie on
    # each facet's places across the whole variable (`datasets_api.profile_breadth`). None
    # leaves the tie to the unit's name.
    variable_breadth: VariableBreadthReader | None = None


# Loads the stored manifest that answers a call from the head at the given content version.
# `None` means the head cannot answer it, and the caller re-materialises.
HeadLoader = Callable[[str, CoverageSnapshot], Manifest | None]


def serve_deps(deps: DefinitionDeps) -> ServeDeps:
    """The `serve.serve_definition` view of `deps`."""
    return ServeDeps(
        sdmx=deps.sdmx,
        graph=deps.graph,
        artifact_store=deps.artifact_store,
        materialiser_build=deps.materialiser_build,
        capability_version=deps.capability_version,
        ceiling=deps.ceiling,
        member_concurrency=deps.member_concurrency,
        enrichment=deps.enrichment,
        code_index=deps.code_index,
        rate_cache=deps.rate_cache,
        version_writer=deps.version_writer,
        snapshot_reader=deps.snapshot_reader,
        head_store=deps.head_store,
        materialise_semaphore=deps.materialise_semaphore,
        deadline=deps.deadline,
        work_deadline=deps.work_deadline,
        head_fence=deps.head_fence,
        settings=deps.settings,
    )


class StoredTraceMissingResolvedStatVars(Refusal):
    """Older traces may omit `resolved_stat_vars`. A render from the stored frame uses it for
    `groupby` completeness, so `plan_stored_render` raises this error before any frame is
    read, which prevents publication of a partial total.

    It maps to non-retryable `internal_error`, 500.
    """

    code = "internal_error"

    def __init__(self, *, subject_id: str, content_version: str) -> None:
        self.subject_id = subject_id
        self.content_version = content_version
        super().__init__(
            f"stored manifest trace for {subject_id!r}@{content_version!r} carries no "
            "resolved_stat_vars key"
        )


def without_pointers[E: Refusal](exc: E) -> E:
    """Mark a materialisation refusal raised on a retrieval route, whose request has no body for
    its pointers to name."""
    exc.strip_pointer = True
    return exc


# The refusals a rebuild's admission and materialisation raise that point into a mint body.
_BODY_POINTING_REFUSALS = (DefaultFacetUndefinedError, FacetNotAvailableError, ExcludedByRegistry)


def stamp_trace(
    manifest: Manifest, *, served_from: str, confirmed_at: datetime, coverage_stamp: str | None
) -> Manifest:
    """Stamp the serving outcome, confirmation time, and current coverage snapshot onto a fresh
    copy.

    Head responses need this step because their stored manifest carries a prior coverage stamp.
    Materialised responses already carry the same snapshot's stamp. This stamp records the request's
    admission estimate. Artifact records retain their write-time metadata.
    """
    trace = served_trace(manifest.trace, served_from=served_from, confirmed_at=confirmed_at)
    return manifest.model_copy(update={"trace": trace, "series_coverage_stamp": coverage_stamp})


def served_trace(
    trace: Mapping[str, object], *, served_from: str, confirmed_at: datetime
) -> dict[str, object]:
    """`trace` with the currency pair a mint, a definition retrieval and a `done` operation poll
    report: `served_from` and `confirmed_at` as RFC 3339 text."""
    return {**trace, "served_from": served_from, "confirmed_at": confirmed_at.isoformat()}


def trace_confirmed_at(trace: Mapping[str, object]) -> datetime:
    """The `confirmed_at` instant `served_trace` wrote into `trace`."""
    return datetime.fromisoformat(cast(str, trace["confirmed_at"]))


def load_stored(
    *,
    subject_id: str,
    content_version: str,
    deps: DefinitionDeps,
    include_artifacts: bool,
    rendering_keys: tuple[str, ...] | None = None,
    revoked_under: Sequence[str] = (),
) -> Manifest:
    """Load a stored version of the served definition through the snapshot loaders.

    The caller has read the definition, so the loaders read no registry row. Both check the
    version and its revocation through `read_servable_version`, in one transaction, under
    `subject_id` and every id in `revoked_under`: a request naming the served id honours
    revocations under it, and a request naming a posted id passes that id, so it honours
    both.
    `include_artifacts=True` uses `read_snapshot_manifest`, which lists servable artifacts under
    their download links and refuses an expired version. Optional `rendering_keys` narrow
    artifacts to the requested rendering. `False` uses `read_snapshot_currency`, which returns
    no artifacts. A fresh head relies on this as its only existence check. Both time under
    `snapshot_state`.

    Its artifacts hold the party-kind files linked under the code index's digest only, each
    carrying it as `enrichment_stamp` (`read_snapshot_manifest`).
    """
    with phase("snapshot_state"):
        if not include_artifacts:
            return read_snapshot_currency(
                id=subject_id,
                content_version=content_version,
                spec_reader=None,
                snapshot_reader=deps.snapshot_reader,
                now=deps.now,
                settings=deps.settings,
                revoked_under=revoked_under,
            )
        return read_snapshot_manifest(
            id=subject_id,
            content_version=content_version,
            spec_reader=None,
            snapshot_reader=deps.snapshot_reader,
            now=deps.now,
            settings=deps.settings,
            rendering_keys=rendering_keys,
            revoked_under=revoked_under,
            enrichment_digest=deps.code_index.enrichment_digest,
        )


def _record_head_hit_derived(
    *,
    manifest: Manifest,
    derived: DerivedVersion,
    artifact: ArtifactEntryModel,
    reshape_warnings: tuple[MaterialisationWarning, ...],
    subject_id: str,
    schema_version: int,
    base_content_version: str,
    deps: DefinitionDeps,
    trace_id: str | None,
) -> Manifest:
    """Record and return a head-hit derived version (`render.DerivedVersion`): a groupby
    collapse, a party-totals cut, or a collapse of the cut.

    Store the artifact under `derived.content_version` to match the derived identity, and
    link it under its rendering, which is an incomplete one when its graph metadata reads
    failed (`incomplete_artifact`). Recording the same pair again is harmless. When the
    derived version is already on record, return it as a repeat mint does
    (`persisted_response`): its stored body with the artifacts linked for it, so its stored
    warnings stand. An incomplete render gives way to a complete artifact a concurrent call
    linked under the derived version (`with_linked_artifacts`). The base record precedes
    the render, so every writer dates and cites the derived version as the base does. The
    derived version's description record derives from the base version's in the transaction
    that records it (`version_descriptions.collapse_describer`).
    """
    derived_frame_record = render_and_store_canonical_frame(
        frame_chunks=derived.frame_chunks(),
        subject_id=subject_id,
        content_version=derived.content_version,
        artifact_store=deps.artifact_store,
    )
    derived_statistics = derived_statistics_file(
        derived, continuing=continued_series_of_trace(manifest.trace)
    )
    derived_statistics_artifact = render_and_store_series_statistics(
        data=derived_statistics.data,
        subject_id=subject_id,
        content_version=derived.content_version,
        artifact_store=deps.artifact_store,
    )
    call_warnings = reshape_warning_models(reshape_warnings)
    output_profile = derived_output_profile(derived, enrichment=deps.enrichment)
    capabilities = build_capabilities(
        capability_version=deps.capability_version,
        build=deps.materialiser_build,
        code_index_stamp=deps.code_index.stamp,
        declarations=deps.code_index.declarations,
        settings=deps.settings,
    )
    derived_manifest = collapsed_head_manifest(
        manifest=manifest,
        derived=derived,
        artifact=artifact,
        statistics_artifact=derived_statistics_artifact,
        series_count=derived_statistics.series_count,
        output_options=derive_output_options(
            output_profile,
            identity=manifest.definition.identity,
            capabilities=capabilities,
            parties=deps.code_index.parties,
            datasets=source_refs_of(manifest.trace),
        ),
        extra_warnings=call_warnings,
    )
    created = record_collapsed_head_version(
        manifest=derived_manifest,
        subject_id=subject_id,
        schema_version=schema_version,
        base_content_version=base_content_version,
        output_profile=output_profile,
        frame_sha256=derived.frame_sha256,
        frame_summed=derived.summed,
        record_only_artifacts=[derived_frame_record],
        version_writer=deps.version_writer,
        trace_id=trace_id,
        describer_for=functools.partial(
            collapse_describer,
            figures=derived_statistics.figures,
            code_index=deps.code_index,
            enrichment=deps.enrichment,
            deadline=deps.deadline,
            summed=derived.summed,
        ),
    )
    if created and not is_incomplete_rendering(artifact.rendering):
        return derived_manifest
    with phase("snapshot_state"):
        state = deps.snapshot_reader.read_version_state(subject_id, derived.content_version)
    if state is None:
        return derived_manifest
    if created:
        return with_linked_artifacts(
            derived_manifest,
            subject_id=subject_id,
            locations=state.locations,
            now=deps.now(),
            enrichment_digest=deps.code_index.enrichment_digest,
        )
    return persisted_response(
        subject_id=subject_id,
        persisted=state.version,
        locations=state.locations,
        served=derived_manifest,
        call_warnings=call_warnings,
        now=deps.now(),
        enrichment_digest=deps.code_index.enrichment_digest,
    )


@dataclass(frozen=True)
class StoredFrame:
    """A version's stored frame (`render.servable_frame_location`) and its verified bytes."""

    location: StoredArtifactLocation
    data: bytes


@dataclass(frozen=True)
class StoredRenderPlan:
    """What `plan_stored_render` read from a stored version before its frame is read: the
    stored output profile a pivot reads, the recorded frame hash a groupby compares its
    collapse with (`StoredVersion.frame_sha256`), and how a render reads the version's own
    stored frame (`render.FrameMode`), as the version's serving frames say
    (`recorded_rendering.serving_frames`). `frame_mode` is `None` when the version's own frame
    does not serve it, so the version renders from its base version's frame alone."""

    output_profile: OutputProfile | None
    frame_sha256: str | None
    frame_mode: FrameMode | None = FrameMode.RAW


def plan_stored_render(
    manifest: Manifest,
    *,
    options: MintOptions,
    subject_id: str,
    content_version: str,
    read_version: Callable[[], StoredVersion | None],
    read_locations: Callable[[], Iterable[StoredArtifactLocation]],
    deps: DefinitionDeps,
) -> StoredRenderPlan:
    """Decide from the stored version alone whether its stored frame can render `options`, so
    a refusal is made before any frame is read, re-uploaded or decoded and before a
    materialise permit is taken.

    Raise `StoredTraceMissingResolvedStatVars` for a trace without `resolved_stat_vars`,
    which a render needs for the groupby population. A pivot, a groupby, a party-totals cut, or
    a plain long `xlsx` whose identity has an axis to split reads the version record once
    (`read_version`); a groupby or a cut reads whether the version is a derived one, and a
    derived one reads its links (`read_locations`) for its serving frames. With
    no `groupby`, no `columns` and no party-totals cut, a pivot refuses its caps from the
    stored pivot facts (`refuse_pivot_over_stored_caps`). A plain long `xlsx` (no `groupby`,
    no `pivot`, no `columns`, no party-totals cut) refuses `excel_limit` from the stored row
    count (`refuse_excel_over_stored_rows`), since the stored count is then the count the
    writer writes. Where totals can take a second sheet (`_stored_xlsx_splits_totals`), only a
    count past two sheets' room refuses and the writer judges each sheet; a groupby, a cut or a
    pivot changes that count, `columns` refusals come first, and the render decides those.
    """
    if manifest.trace.get("resolved_stat_vars") is None:
        raise StoredTraceMissingResolvedStatVars(
            subject_id=subject_id, content_version=content_version
        )
    stored = (
        read_version()
        if options.pivot is not None or options.groupby is not None or options.without_party_totals
        else None
    )
    output_profile: OutputProfile | None = None
    # A groupby, `columns` or a party-totals cut changes what the stored facts count.
    reshaped = (
        options.groupby is not None or options.columns is not None or options.without_party_totals
    )
    if options.pivot is not None:
        output_profile = (
            stored_output_profile(stored_manifest_facts(stored)) if stored is not None else None
        )
        if not reshaped:
            refuse_pivot_over_stored_caps(
                options,
                output_profile=output_profile,
                pivot_max_output_cells=deps.settings.pivot_max_output_cells,
            )
    elif not reshaped and options.format == "xlsx":
        # The writer refuses at the first row past a sheet's limit and names that row.
        refuse_excel_over_stored_rows(
            manifest.coverage.rows,
            splits_totals=_stored_xlsx_splits_totals(
                manifest, options=options, read_version=read_version, deps=deps
            ),
        )
    frame_mode: FrameMode | None = FrameMode.RAW
    if stored is not None and stored.base_content_version is not None:
        frames = serving_frames(
            stored, renderings=[location.rendering for location in read_locations()]
        )
        frame_mode = next(
            (frame.mode for frame in frames if frame.content_version == content_version), None
        )
    return StoredRenderPlan(
        output_profile=output_profile,
        frame_sha256=stored.frame_sha256 if stored is not None else None,
        frame_mode=frame_mode,
    )


def _stored_xlsx_splits_totals(
    manifest: Manifest,
    *,
    options: MintOptions,
    read_version: Callable[[], StoredVersion | None],
    deps: DefinitionDeps,
) -> bool:
    """`xlsx_splits_totals` for a render of the stored version `manifest`, with the inputs the
    render splits by: the trace's provenances and the stored profile's role columns
    (`stored_output_profile`). A version minted before versions stored a profile takes the
    identity's role keys (`identity_role_keys`), which name the same axes. An identity with no
    axis to split keeps every row on one sheet without reading the version."""
    identity = manifest.definition.identity
    if not totals_axes(identity):
        return False
    stored = read_version()
    profile = stored_output_profile(stored_manifest_facts(stored)) if stored is not None else None
    role_columns = (
        [column.name for column in profile.columns if column.kind == "role"]
        if profile is not None
        else identity_role_keys(identity)
    )
    return xlsx_splits_totals(
        options,
        identity=identity,
        role_columns=role_columns,
        parties=deps.code_index.parties,
        datasets=source_refs_of(manifest.trace),
    )


def _render_frame(
    *,
    subject_id: str,
    content_version: str,
    options: MintOptions,
    manifest: Manifest,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_index: CoverageIndex,
    frame: StoredFrame,
    plan: StoredRenderPlan,
    locations: Sequence[StoredArtifactLocation] = (),
) -> Manifest | None:
    """Render a missing request format from the stored frame `frame` on a head hit. The caller
    holds a materialise permit from before it read `frame` to after this returns.
    `locations` is the version's links when the caller read them to find `frame`.

    Use the same provenance, definition, resolved-member population, and `ReproducingCall` as
    fresh rendering. The full population keeps `groupby` completeness checks consistent,
    including zero-row members. `plan` is what `plan_stored_render` read from the stored
    version: a pivot reads its stored output profile, a groupby compares its collapse with the
    recorded frame hash, and its frame mode says how the render reads `frame`.

    Return the derived version's manifest when groupby or a party-totals cut changes the frame
    (`render.DerivedVersion`). A derived version's own frame (`StoredRenderPlan.frame_mode`
    other than `RAW`) renders under the version itself. A render of it that gives another
    version records nothing and returns `None`, so the caller falls back to the base frame.
    That happens when the code index changed since the mint, and recording it would let a
    version derived from a derived one escape its raw version's revocation. A groupby that leaves the frame
    unchanged is the base version, whatever frame grammar minted its token
    (`render_and_store`). Record its artifact, and every other one, under the base version
    and add it to the manifest. A party-totals cut
    reads a canonical frame, so a version whose stored frame is a frame1 document returns
    `None` and the caller rebuilds. Render errors propagate to the shared classifier.

    A bundle cites the stored citation with its empty fields filled by a live read
    (`bundle_citations`). A render whose graph metadata reads failed is linked as incomplete
    (`incomplete_artifact`). The response lists the artifact linked under the request's
    rendering key when a concurrent call linked one first (`with_linked_artifacts`).
    """
    # `_render_stored_rendering` reads no frame of a version its own frame does not serve.
    assert plan.frame_mode is not None
    table = read_stored_frame(frame.data, media_type=frame.location.media_type)
    if (
        options.without_party_totals
        and plan.frame_mode is FrameMode.RAW
        and not isinstance(table, Frame)
    ):
        return None
    # A collapse's kept code axes render as the decoded codes its mint wrote them as.
    entity_columns = entity_roles(frame_entity_columns(table.columns))
    facets = cast("list[dict[str, object]]", manifest.trace.get("facets", []))
    provenance_refs = provenance_refs_of(facets)
    # `plan_stored_render` refused a trace without `resolved_stat_vars`.
    resolved_stat_var_refs = refs_from_resolved_stat_vars(
        cast("list[dict[str, object]]", manifest.trace["resolved_stat_vars"])
    )
    spec_dict: dict[str, object] = {
        "schema_version": manifest.definition.schema_version,
        "spec_id": manifest.definition.spec_id,
        "identity": manifest.definition.identity,
    }
    citations, documentation_complete = bundle_citations(
        manifest.citation,
        options=options,
        on_record=True,
        # A head hit renders only when no artifact is linked under the request's key.
        linked=False,
        documentation_complete=True,
        graph=deps.graph,
        deadline=deps.deadline,
        coverage_index=coverage_index,
    )
    rendered = render_and_store(
        frame_chunks=lambda: iter((table,)),
        options=options,
        spec=spec_dict,
        provenance_refs=provenance_refs,
        citation_refs=source_refs_of(manifest.trace),
        citations=citations,
        reproducing_call=(
            ReproducingCall(
                path=f"/api/datasets/{subject_id}",
                options=encode_query_options(options),
            )
            if DEFINITION_ROUTE_LIVE
            else None
        ),
        content_version=content_version,
        prepared_at=manifest.prepared_at,
        site=deps.settings.site,
        artifact_store=deps.artifact_store,
        entity_columns=entity_columns,
        enrichment=deps.enrichment,
        pivot_max_output_cells=deps.settings.pivot_max_output_cells,
        code_index=deps.code_index,
        resolved_stat_var_refs=resolved_stat_var_refs,
        # A pivot reads the stored pivot facts. They check its caps before the scan and let a
        # non-unique refusal name the varying columns.
        output_profile=plan.output_profile,
        deadline=deps.deadline,
        # Only a groupby of a frame it has not collapsed yields a collapse frame to compare
        # with the stored one.
        base_frame_sha256=(
            stored_frame_sha256(
                frame.location,
                table,
                recorded=plan.frame_sha256,
                locations=locations,
            )
            if options.groupby is not None and plan.frame_mode is not FrameMode.SUMMED
            else None
        ),
        frame_mode=plan.frame_mode,
    )
    artifact, reshape_warnings, derived = (
        rendered.artifact,
        rendered.warnings,
        rendered.derived,
    )
    if derived is not None and plan.frame_mode is not FrameMode.RAW:
        logger.warning(
            "stored frame render gave another version; falling back to the base frame "
            "subject_id=%s content_version=%s derived_content_version=%s rendering=%s",
            subject_id,
            content_version,
            derived.content_version,
            encode_query_options(options),
        )
        return None
    rendered_content_version = derived.content_version if derived is not None else content_version
    if not (documentation_complete and rendered.labels_complete):
        artifact = incomplete_artifact(
            artifact,
            subject_id=subject_id,
            content_version=rendered_content_version,
            refs=provenance_refs,
            documentation_complete=documentation_complete,
            labels_complete=rendered.labels_complete,
        )
    if derived is not None:
        return _record_head_hit_derived(
            manifest=manifest,
            derived=derived,
            artifact=artifact,
            reshape_warnings=reshape_warnings,
            subject_id=subject_id,
            schema_version=manifest.definition.schema_version,
            base_content_version=content_version,
            deps=deps,
            trace_id=trace_id,
        )
    deps.version_writer.record_artifact(
        subject_id=subject_id,
        content_version=content_version,
        artifact=ArtifactRecord(
            artifact_digest=artifact.artifact_digest,
            rendering=artifact.rendering,
            media_type=artifact.media_type,
            enrichment_digest=link_digest(artifact.rendering, stamp=artifact.enrichment_stamp),
        ),
        trace_id=trace_id,
    )
    # Keep any stored series-statistics artifact. This rendering shares its content version.
    rendered_manifest = manifest.model_copy(
        update={
            "artifacts": [artifact],
            "warnings": list(manifest.warnings) + list(reshape_warning_models(reshape_warnings)),
        }
    )
    linked_manifest = with_linked_artifacts(
        rendered_manifest,
        subject_id=subject_id,
        locations=deps.snapshot_reader.read_artifact_locations(subject_id, content_version),
        now=deps.now(),
        enrichment_digest=deps.code_index.enrichment_digest,
    )
    return linked_manifest.model_copy(
        update={"artifacts": [*linked_manifest.artifacts, *manifest.artifacts]}
    )


def _render_located_frame(
    location: StoredArtifactLocation,
    *,
    subject_id: str,
    content_version: str,
    options: MintOptions,
    manifest: Manifest,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_index: CoverageIndex,
    plan: StoredRenderPlan,
    locations: Sequence[StoredArtifactLocation],
) -> Manifest | None:
    """Read the stored frame at `location`, one of the version's links `locations`, and
    render it (`_render_frame`), or return `None` for missing or corrupt frame bytes. The
    caller holds a materialise permit and binds no frame itself, so the bytes are freed when
    this returns, before the permit is released."""
    try:
        stored_frame = StoredFrame(
            location=location,
            data=deps.artifact_store.get(
                artifact_digest=location.artifact_digest, tier=Tier(location.tier)
            ),
        )
    except (ArtifactNotFoundError, ArtifactIntegrityError):
        return None
    return _render_frame(
        subject_id=subject_id,
        content_version=content_version,
        options=options,
        manifest=manifest,
        deps=deps,
        trace_id=trace_id,
        coverage_index=coverage_index,
        frame=stored_frame,
        plan=plan,
        locations=locations,
    )


def _render_stored_rendering(
    *,
    subject_id: str,
    content_version: str,
    options: MintOptions,
    manifest: Manifest,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_snapshot: CoverageSnapshot,
    frame: StoredFrame | None = None,
    plan: StoredRenderPlan | None = None,
) -> Manifest | None:
    """Render a missing request format from the version's stored frame (`_render_frame`),
    under exactly one materialise permit from the frame read to the end of the render.

    With `frame` `None` this takes the permit: it reads the servable stored frame
    (`servable_frame_location`) inside a `bounded_materialise_slot`, and returns `None` for a
    missing frame location or missing or corrupt frame bytes. A full semaphore raises
    retryable `MintBackpressure` (503) once the wait runs out or once a swap replaced the
    request's bound release (`bounded_materialise_slot`).

    A `frame` argument is the stored frame the caller read and verified under a permit it
    still holds. The render then runs under that permit and takes none, since the permits
    are not reentrant, and the caller took `coverage_snapshot` under it.

    Once a servable frame location exists, and before the permit and the frame read,
    `plan_stored_render` makes every refusal the stored version decides. A version with no
    servable frame returns `None` before any plan, so its caller rebuilds as it would without
    the stored record's refusals. Whether a servable location's bytes are still in the store
    is known only by fetching them, so a refusal the plan makes also answers a version whose
    location row is servable and whose object is gone, which `datasets-maintenance sweep`
    reports as a missing object. A `plan` argument is the one the caller made from this
    version before it read `frame`. A plan whose version's own frame does not serve it
    (`StoredRenderPlan.frame_mode` `None`) returns `None` before the permit and the frame read.
    """

    def planned(read_locations: Callable[[], Iterable[StoredArtifactLocation]]) -> StoredRenderPlan:
        if plan is not None:
            return plan
        return plan_stored_render(
            manifest,
            options=options,
            subject_id=subject_id,
            content_version=content_version,
            read_version=functools.partial(
                deps.snapshot_reader.read_version, subject_id, content_version
            ),
            read_locations=read_locations,
            deps=deps,
        )

    if frame is not None:
        decided = planned(
            functools.partial(
                deps.snapshot_reader.read_artifact_locations, subject_id, content_version
            )
        )
        if decided.frame_mode is None:
            return None
        return _render_frame(
            subject_id=subject_id,
            content_version=content_version,
            options=options,
            manifest=manifest,
            deps=deps,
            trace_id=trace_id,
            coverage_index=coverage_snapshot.index,
            frame=frame,
            plan=decided,
        )
    locations = deps.snapshot_reader.read_artifact_locations(subject_id, content_version)
    location = servable_frame_location(locations, now=deps.now())
    if location is None:
        return None
    decided = planned(lambda: locations)
    if decided.frame_mode is None:
        return None
    # The frame loads whole, so the load and the render share the semaphore that bounds a
    # mint's materialisation.
    with bounded_materialise_slot(
        deps.materialise_semaphore,
        wait_seconds=deps.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        return _render_located_frame(
            location,
            subject_id=subject_id,
            content_version=content_version,
            options=options,
            manifest=manifest,
            deps=deps,
            trace_id=trace_id,
            coverage_index=coverage_snapshot.index,
            plan=decided,
            locations=locations,
        )


def head_rendering(
    *,
    subject_id: str,
    content_version: str,
    deps: DefinitionDeps,
    options: MintOptions | None,
    trace_id: str | None,
    coverage_snapshot: CoverageSnapshot,
    render: bool = True,
    frame: StoredFrame | None = None,
    plan: StoredRenderPlan | None = None,
    revoked_under: Sequence[str] = (),
) -> Manifest | None:
    """Load the stored version, narrowed to the rendering `options` names. `options` `None`
    loads the version without its artifacts. `revoked_under` is `load_stored`'s.

    A rendering not linked to the version renders from the stored frame when `render`
    (`_render_stored_rendering`). Return `None`
    when neither the stored nor that rendering can
    serve the request, or the rendering is missing and `render` is false. The caller then
    re-materialises or hands the call on. `coverage_snapshot` feeds an on-demand bundle's
    documentation read.

    The render takes one materialise permit. A `frame` argument is the stored
    frame the caller read and verified under a materialise permit it still holds: the render
    takes those bytes in place of the read, takes no permit, and trusts `coverage_snapshot`
    as taken under that permit. A `plan` argument is the `plan_stored_render` decision the
    caller made from this version, so the render does not make it again.
    """
    keys = (
        artifact_rendering_keys(
            options, site=deps.settings.site, enrichment_digest=deps.code_index.enrichment_digest
        )
        if options is not None
        else None
    )
    manifest = load_stored(
        subject_id=subject_id,
        content_version=content_version,
        deps=deps,
        include_artifacts=options is not None,
        rendering_keys=keys,
        revoked_under=revoked_under,
    )
    # The snapshot loader retains `series-statistics/1` even when the requested rendering is
    # absent, so test for a non-statistics artifact.
    if options is None or any(not is_statistics_rendering(a.rendering) for a in manifest.artifacts):
        return manifest
    if not render:
        return None
    return _render_stored_rendering(
        subject_id=subject_id,
        content_version=content_version,
        options=options,
        manifest=manifest,
        deps=deps,
        trace_id=trace_id,
        coverage_snapshot=coverage_snapshot,
        frame=frame,
        plan=plan,
    )


@dataclass(frozen=True)
class HeadRead:
    """The served definition and its head as read from the head store, `head` `None` when the
    subject has none. `verdict` is the freshness verdict a caller already reached on it
    (`serve_current_head`), so `serve_current` neither reads nor judges the head again."""

    stored: StoredDefinition
    head: SubjectHead | None
    verdict: _Freshness | None = None


def _rebuild_mark(head: SubjectHead | None) -> tuple[str, datetime] | None:
    """What a rebuild writes to a head and a confirmation leaves alone."""
    return None if head is None else (head.content_version, head.materialised_at)


def _head_is_fresh(head: SubjectHead, *, now: datetime, deps: DefinitionDeps) -> bool:
    """Whether `head` was confirmed inside the recheck window."""
    return now - head.confirmed_at < deps.head_recheck_interval


# The rebuild reasons that name no fingerprint member (`serve_current`).
_NO_HEAD = "no_head"
_MAX_AGE = "max_age"
_UNSERVABLE = "unservable"
_YEAR_COVERAGE = "year_coverage"


def _fingerprint_changes(
    head: SubjectHead,
    requests: Sequence[MaterialisationRequest],
    *,
    graph: GraphReader,
    coverage: CoverageIndex,
) -> tuple[str, ...]:
    """The members of `head`'s dependency fingerprint that differ from the one a run over
    `requests` would record now, read in one graph call and elected from `coverage`
    (`refreshed_dependencies`). A head with no readable fingerprint differs on `schema` and
    reads nothing."""
    stored = decode_dependencies(head.dependencies) if head.dependencies is not None else None
    if stored is None:
        return ("schema",)
    return dependency_changes(
        head.dependencies,
        refreshed_dependencies(stored, requests, graph=graph, coverage=coverage),
    )


@dataclass(frozen=True)
class _Freshness:
    """A head freshness verdict: the head to serve, or `None` and why the call rebuilds."""

    head: SubjectHead | None
    rebuild_reason: str | None = None


def _judge_head(
    head: SubjectHead | None,
    *,
    stored: StoredDefinition,
    requests: Callable[[], Sequence[MaterialisationRequest]],
    now: datetime,
    deps: DefinitionDeps,
    coverage: CoverageIndex,
) -> _Freshness:
    """Whether `head`, the head of the served definition `stored`, is current (`serve_current`'s
    head rule). A head past the recheck window and younger than `head_max_age` whose
    fingerprint matches is confirmed, and the verdict serves the head the confirmation wrote,
    or the head a concurrent update left. The fingerprint read and its election from
    `coverage` run under the definition's pinned provenances
    (`adapter.definition_read_scope`), as its materialisation does."""
    subject_id = stored.spec_id
    if head is None:
        return _Freshness(head=None, rebuild_reason=_NO_HEAD)
    if _head_is_fresh(head, now=now, deps=deps):
        return _Freshness(head=head)
    if now - head.materialised_at >= deps.head_max_age:
        return _Freshness(head=None, rebuild_reason=_MAX_AGE)
    with (
        definition_read_scope(stored.identity, code_index=deps.code_index),
        phase("fingerprint"),
    ):
        changes = _fingerprint_changes(head, requests(), graph=deps.graph, coverage=coverage)
    if changes:
        return _Freshness(head=None, rebuild_reason=",".join(changes))
    written = deps.head_store.confirm_head(
        subject_id=subject_id, content_version=head.content_version
    )
    # A concurrent update won. Serve its head.
    winner = written if written is not None else deps.head_store.read_head(subject_id)
    if winner is None:
        return _Freshness(head=None, rebuild_reason=_NO_HEAD)
    return _Freshness(head=winner)


def _serve_head(
    served_head: SubjectHead,
    *,
    subject_id: str,
    deps: DefinitionDeps,
    coverage_snapshot: CoverageSnapshot,
    options: MintOptions | None,
    trace_id: str | None,
    load_head: HeadLoader | None,
    render: bool = True,
    revoked_under: Sequence[str] = (),
) -> Manifest | str:
    """The stored manifest answering the call from `served_head`, stamped `served_from: head`,
    or the reason the call rebuilds: `_UNSERVABLE` when the head cannot answer it, and
    `_YEAR_COVERAGE` when its version is a frame of one year whose year coverage was never
    counted (`year_coverage_counted`). The rebuild counts it, on the stored version when it
    makes the same one (`VersionRecordWriter.write_version`), so a head rebuilds once for it.
    `load_head` replaces `head_rendering`, and `revoked_under` is `load_stored`'s."""
    if load_head is not None:
        manifest = load_head(served_head.content_version, coverage_snapshot)
    else:
        manifest = head_rendering(
            subject_id=subject_id,
            content_version=served_head.content_version,
            deps=deps,
            options=options,
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
            render=render,
            revoked_under=revoked_under,
        )
    if manifest is None:
        return _UNSERVABLE
    window = manifest.coverage.window or {}
    if (
        window.get("start_year") is not None
        and window.get("start_year") == window.get("end_year")
        and not year_coverage_counted(manifest.coverage.model_dump(mode="json"))
    ):
        return _YEAR_COVERAGE
    return stamp_trace(
        manifest,
        served_from="head",
        confirmed_at=served_head.confirmed_at,
        coverage_stamp=coverage_snapshot.stamp,
    )


# The counts every serve's timings line carries: upstream requests, store transactions,
# `signBlob` calls and the milliseconds spent waiting for a materialise permit.
_SERVE_COUNTS = (
    "sdmx_requests",
    "graph_requests",
    "db_transactions",
    "sign_blob_calls",
    "permit_wait_ms",
)


def _serve_judged_head(
    verdict: _Freshness,
    *,
    subject_id: str,
    deps: DefinitionDeps,
    coverage_snapshot: CoverageSnapshot,
    options: MintOptions | None,
    trace_id: str | None,
    load_head: HeadLoader | None,
    revoked_under: Sequence[str],
) -> Manifest | str:
    """`_serve_head` on the head `verdict` serves, or the reason the call rebuilds."""
    if verdict.head is None:
        return cast(str, verdict.rebuild_reason)
    return _serve_head(
        verdict.head,
        subject_id=subject_id,
        deps=deps,
        coverage_snapshot=coverage_snapshot,
        options=options,
        trace_id=trace_id,
        load_head=load_head,
        revoked_under=revoked_under,
    )


def _labelled(manifest: Manifest, *, trigger: Trigger) -> Manifest:
    label("trigger", trigger)
    label("served_from", cast(str, manifest.trace["served_from"]))
    # Start every count, so a serve that sent no request logs zero.
    for name in _SERVE_COUNTS:
        count(name, 0)
    return manifest


def serve_fresh_head(
    stored: StoredDefinition,
    *,
    deps: DefinitionDeps,
    options: MintOptions | None,
    trigger: Trigger,
    trace_id: str | None,
    revoked_under: Sequence[str] = (),
) -> Manifest | HeadRead:
    """The head's answer when the head is current and holds the requested rendering, or the
    head it read. `revoked_under` is `load_stored`'s.

    This is `serve_current`'s recheck-window branch without rendering. It reads no graph, renders
    nothing, and materialises nothing, so it runs on the read executor. A `HeadRead` hands the call
    to the work executor, whose `serve_current` starts from that head.
    """
    with phase("coverage"):
        coverage_snapshot = deps.coverage.snapshot()
    with phase("head_read"):
        head = deps.head_store.read_head(stored.spec_id)
    if head is None or not _head_is_fresh(head, now=deps.now(), deps=deps):
        return HeadRead(stored=stored, head=head)
    served = _serve_head(
        head,
        subject_id=stored.spec_id,
        deps=deps,
        coverage_snapshot=coverage_snapshot,
        options=options,
        trace_id=trace_id,
        load_head=None,
        render=False,
        revoked_under=revoked_under,
    )
    if isinstance(served, str):
        return HeadRead(stored=stored, head=head)
    return _labelled(served, trigger=trigger)


def serve_current_head(
    stored: StoredDefinition,
    *,
    requests: Sequence[MaterialisationRequest],
    options: MintOptions,
    trigger: Trigger,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_snapshot: CoverageSnapshot,
    revoked_under: Sequence[str] = (),
) -> Manifest | HeadRead:
    """The head's answer when the head is current by `serve_current`'s rule and links the
    rendering `options` names, or the head it read with its verdict. `revoked_under` is
    `load_stored`'s.

    A mint calls this before admission, so a repeat mint of a current head answers from it
    without estimating or submitting an operation, and passes a `HeadRead` on to
    `serve_current`. It renders and materialises nothing; past the recheck window it makes the
    one graph read the fingerprint comparison needs.
    """
    with phase("head_read"):
        head = deps.head_store.read_head(stored.spec_id)
    verdict = _judge_head(
        head,
        stored=stored,
        requests=lambda: requests,
        now=deps.now(),
        deps=deps,
        coverage=coverage_snapshot.index,
    )
    if verdict.head is None:
        return HeadRead(stored=stored, head=head, verdict=verdict)
    served = _serve_head(
        verdict.head,
        subject_id=stored.spec_id,
        deps=deps,
        coverage_snapshot=coverage_snapshot,
        options=options,
        trace_id=trace_id,
        load_head=None,
        render=False,
        revoked_under=revoked_under,
    )
    if isinstance(served, str):
        return HeadRead(stored=stored, head=head, verdict=verdict)
    return _labelled(served, trigger=trigger)


def serve_current(
    stored: StoredDefinition,
    *,
    resolve_requests: Callable[[], Sequence[MaterialisationRequest]],
    options: MintOptions | None,
    trigger: Trigger,
    deps: DefinitionDeps,
    trace_id: str | None,
    admitted: Estimate | None = None,
    coverage_snapshot: CoverageSnapshot | None = None,
    progress: RowProgress | None = None,
    load_head: HeadLoader | None = None,
    head_read: HeadRead | None = None,
    strip_pointer: bool = False,
    revoked_under: Sequence[str] = (),
) -> Manifest:
    """Serve the current version of the served definition `stored`, materialising and advancing
    the head when the head cannot answer. A head answers under `load_stored`'s
    `revoked_under`.

    One coverage snapshot serves each pass's head check, admission estimate, fetch, and response
    stamp: `coverage_snapshot` when given, else the service's. A head confirmed within
    `head_recheck_interval` serves with no read and skips rehydration; a missing rendering may
    need artifact-store reads. Past that window a head older than `head_max_age` rebuilds,
    which catches corrections no fingerprint member sees. Otherwise `resolve_requests` runs
    once and one graph read compares the head's dependency fingerprint: a match confirms the
    head and serves it, or the winner when another call advanced it first. A head with no
    fingerprint, a changed fingerprint, or a head that cannot render the request rebuilds with
    the observed head as `expected_head`, so a concurrent head update is never overwritten. The
    rebuild names its `rebuild_reason` on the timings line.

    Admission runs before any fetch. A caller that admitted the request passes its estimate as
    `admitted`, which sizes the raw-fetch guard. Otherwise an `options` call refuses an estimate
    over `deps.ceiling` (`refuse_inline_over_ceiling`), and a rendering call refuses Excel over
    its row cap and submits work over the sync budget as an operation (`submit_over_budget`).
    A submission the operation backlog refuses reads the head again and answers from it when it
    now serves the call, as it does when that operation's run completed in between, so the
    backlog refusal stands only on a head read that follows it.

    Rebuilds share one run per subject in this process (`deps.rebuilds`). Every other caller
    waits for that run until its own request deadline, then reads the head again and is served
    from it. When the run ended without advancing the head, each waiter re-materialises itself
    and answers with its own result or refusal.

    `load_head` replaces how a head answers the call: it returns the stored manifest to serve for
    the head's content version, or `None` to re-materialise. Omitted, the head serves the rendering
    `options` names (`head_rendering`). `head_read` is the head `serve_fresh_head` or
    `serve_current_head` read, with the verdict the latter reached; the first pass starts from it
    and later passes read the head. A rebuild that took its turn without waiting builds from the
    head it read, and stamps the confirmation of the head its run wrote. `progress` receives the
    materialiser's row progress. `strip_pointer` marks the refusals a rebuild's admission and
    materialisation raise (`without_pointers`) for a retrieval route, whose request has no body;
    refusals from a head hit keep their pointers.

    Returns a manifest stamped by `stamp_trace`.
    """
    while True:
        check = functools.partial(
            _check_currency,
            stored,
            resolve_requests=resolve_requests,
            options=options,
            trigger=trigger,
            deps=deps,
            trace_id=trace_id,
            admitted=admitted,
            progress=progress,
            load_head=load_head,
            head_read=head_read,
            strip_pointer=strip_pointer,
            revoked_under=revoked_under,
        )
        manifest = check(
            coverage_snapshot if coverage_snapshot is not None else deps.coverage.snapshot()
        )
        if manifest is not None:
            return _labelled(manifest, trigger=trigger)
        head_read = None


def _check_currency(
    stored: StoredDefinition,
    snapshot: CoverageSnapshot,
    *,
    resolve_requests: Callable[[], Sequence[MaterialisationRequest]],
    options: MintOptions | None,
    trigger: Trigger,
    deps: DefinitionDeps,
    trace_id: str | None,
    admitted: Estimate | None,
    progress: RowProgress | None,
    load_head: HeadLoader | None,
    head_read: HeadRead | None,
    strip_pointer: bool,
    revoked_under: Sequence[str],
) -> Manifest | None:
    """One pass of `serve_current` on the coverage snapshot `snapshot`. `None` means another
    caller rebuilt the subject while this one waited, so the head is read again."""
    subject_id = stored.spec_id
    if head_read is not None:
        head = head_read.head
    else:
        with phase("head_read"):
            head = deps.head_store.read_head(subject_id)
    now = deps.now()
    requests = functools.cache(resolve_requests)

    verdict = (
        head_read.verdict
        if head_read is not None and head_read.verdict is not None
        else _judge_head(
            head, stored=stored, requests=requests, now=now, deps=deps, coverage=snapshot.index
        )
    )
    answer_from_head = functools.partial(
        _serve_judged_head,
        subject_id=subject_id,
        deps=deps,
        coverage_snapshot=snapshot,
        options=options,
        trace_id=trace_id,
        load_head=load_head,
        revoked_under=revoked_under,
    )
    rebuild_reason = answer_from_head(verdict)
    if not isinstance(rebuild_reason, str):
        return rebuild_reason
    rebuild_requests = requests()
    with deps.rebuilds.turn(
        subject_id,
        deadline=deps.deadline,
        wait_seconds=deps.settings.materialise_wait_seconds,
    ) as waited:
        # A rebuild that ended while this caller waited advanced the head. One that failed
        # left it, and this caller then rebuilds as it would have alone.
        if waited:
            with phase("head_read"):
                current = deps.head_store.read_head(subject_id)
            if _rebuild_mark(current) != _rebuild_mark(head):
                return None
        label("rebuild_reason", rebuild_reason)
        try:
            served = _materialise_and_record(
                stored,
                requests=rebuild_requests,
                options=options,
                trigger=trigger,
                deps=deps,
                trace_id=trace_id,
                expected_head=head,
                coverage_snapshot=snapshot,
                admitted=admitted,
                progress=progress,
                strip_pointer=strip_pointer,
            )
        except OperationBacklogFull:
            # The refusal stands only on a head read that follows it.
            with phase("head_read"):
                current = deps.head_store.read_head(subject_id)
            recheck = answer_from_head(
                _judge_head(
                    current,
                    stored=stored,
                    requests=requests,
                    now=deps.now(),
                    deps=deps,
                    coverage=snapshot.index,
                )
            )
            if isinstance(recheck, str):
                raise
            return recheck
    written = served.head
    if written is None:
        # A concurrent update won the head. Stamp its confirmation.
        with phase("head_read"):
            written = deps.head_store.read_head(subject_id)
    return stamp_trace(
        served.manifest,
        served_from="materialise",
        confirmed_at=written.confirmed_at if written is not None else now,
        coverage_stamp=snapshot.stamp,
    )


def _admitted_ceiling(
    stored: StoredDefinition,
    *,
    requests: Sequence[MaterialisationRequest],
    options: MintOptions | None,
    deps: DefinitionDeps,
    coverage_snapshot: CoverageSnapshot,
    admitted: Estimate | None,
) -> int:
    """Admit a materialisation and return its raw-fetch guard (`serve_current`)."""
    if admitted is not None:
        return raw_fetch_guard(admitted, floor=deps.ceiling)
    estimate = sync_estimate(
        requests,
        index=coverage_snapshot.index,
        member_concurrency=deps.member_concurrency,
        composite=identity_composite(stored.identity, code_index=deps.code_index),
    )
    if options is None:
        refuse_inline_over_ceiling(estimate, ceiling=deps.ceiling)
        return deps.ceiling
    refuse_excel_over_estimate(
        estimate,
        format=options.format,
        splits_totals=xlsx_splits_totals(
            options,
            identity=stored.identity,
            role_columns=identity_role_keys(stored.identity),
            parties=deps.code_index.parties,
            datasets=elected_provenances(requests),
        ),
    )
    submit_over_budget(
        subject_id=stored.spec_id,
        options=options,
        estimate=estimate,
        coverage_stamp=coverage_snapshot.stamp,
        operation_store=deps.operation_store,
        task_client=deps.task_client,
        within_budget=within_sync_budget(
            estimate,
            sync_budget_rows=deps.settings.sync_budget_rows,
            sync_budget_requests=deps.settings.sync_budget_requests,
        ),
        async_max_rows=deps.settings.async_max_rows,
        async_max_requests=deps.settings.async_max_requests,
        members=[request.stat_var for request in requests],
    )
    return raw_fetch_guard(estimate, floor=deps.ceiling)


def _materialise_and_record(
    stored: StoredDefinition,
    *,
    requests: Sequence[MaterialisationRequest],
    options: MintOptions | None,
    trigger: Trigger,
    deps: DefinitionDeps,
    trace_id: str | None,
    expected_head: SubjectHead | None,
    coverage_snapshot: CoverageSnapshot,
    admitted: Estimate | None,
    progress: RowProgress | None,
    strip_pointer: bool,
) -> ServedDefinition:
    """Admit, materialise, record the version and the run under `trigger`, and advance the head
    from `expected_head`. With `strip_pointer` a body-pointing refusal is marked
    `without_pointers`.

    The materialisation reads under `deps.variable_breadth` with `refuse_unread`, since the
    version stores its election: a single-place tie the breadth would decide refuses
    `VariableBreadthUnreadError` when the breadth cannot be read, so every trigger of one
    definition elects one facet."""
    try:
        ceiling = _admitted_ceiling(
            stored,
            requests=requests,
            options=options,
            deps=deps,
            coverage_snapshot=coverage_snapshot,
            admitted=admitted,
        )
        with bound_variable_breadth(deps.variable_breadth, refuse_unread=True):
            return serve_definition(
                stored=stored,
                requests=requests,
                options=options,
                deps=serve_deps(deps),
                trigger=trigger,
                trace_id=trace_id,
                expected_head=expected_head,
                ceiling=ceiling,
                coverage_stamp=coverage_snapshot.stamp,
                coverage_index=coverage_snapshot.index,
                progress=progress,
            )
    except _BODY_POINTING_REFUSALS as exc:
        if strip_pointer:
            without_pointers(exc)
        raise


__all__ = [
    "DefinitionDeps",
    "HeadLoader",
    "HeadRead",
    "StoredRenderPlan",
    "StoredTraceMissingResolvedStatVars",
    "Trigger",
    "head_rendering",
    "load_stored",
    "plan_stored_render",
    "serve_current",
    "serve_current_head",
    "serve_deps",
    "serve_fresh_head",
    "stamp_trace",
    "without_pointers",
]
