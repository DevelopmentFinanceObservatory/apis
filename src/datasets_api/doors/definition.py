"""Retrieve stored definitions with `GET /api/datasets/{id}` or its write-free `/options` route.

The route accepts `spec_` and `col_` ids. It reads definitions from the registry and re-materialises
against the graph when needed. The register door owns identity writes. Grammar-valid ids absent from
the registry return `spec_not_found`. Malformed ids reach the surface's 404 refusal route. Rendering
options use dotted query keys, and unsupported options are refused.

`/options` is mounted before the snapshot version route, so the literal `options` suffix reaches
this handler. It rejects every query key with `request_invalid` and an `/options/<key>` pointer. It
resolves and materialises, then returns `OptionsEnvelope` before render, reshape, or writes. Resolve
refusals and `over_ceiling` match retrieval behaviour. `POST /options` carries the spec's canonical
definition (`DefinitionOptionsRequest`), so it also answers for a spec no mint or register call
has stored. The definition is admitted as a mint admits a sidecar, and nothing is registered.

Retrieval delegates shared materialisation, registry, and snapshot errors to their classifiers.
This door adds registry-miss, backpressure, and stored-data errors. Materialisation errors whose
mint classification points into a POST body are wrapped without a pointer. Conversion refusals carry
no request pointer because their binding comes from the stored identity.

Rehydration checks stored identities against the current graph. It can refuse unresolvable members,
non-exportable definitions, role mismatches, over-ceiling enumerations, and non-reproducible
identities. `MemberNotResolvable` is wrapped so retrieval issues omit the mint body's
`/input/statvars` pointer. Collections aggregate member-resolution failures. Other member errors
abort retrieval immediately.

A dedicated executor and semaphore isolate retrieval from mint work. Admitted worker calls finish
after a client timeout, and shutdown drains permits within its budget. This door builds uninjected
graph, SDMX, registry, and artifact-store clients in its lifespan, independently of mint's clients.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import cast

from dfo_spec_kit.identity import IdentityError
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from artifact_store import (
    ArtifactSigningError,
    ArtifactStore,
    ArtifactStoreError,
    LiveArtifactInventory,
    LiveArtifactStore,
    Tier,
)
from artifact_store.errors import ArtifactIntegrityError, ArtifactNotFoundError
from code_index.query import CodeIndex
from code_index.shared import get_index
from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import (
    LiveSnapshotReader,
    SnapshotReader,
    SnapshotRevoked,
    servable_location_of_media_type,
)

# registers the `dfo_subject_id` convertor
from datasets_api import routing  # noqa: F401
from datasets_api.adapter import (
    check_reproducible,
    pinned_member_ref,
    rehydrate_definition,
    to_materialisation_request,
    verify_sidecar_definition,
)
from datasets_api.admission import (
    bounded_materialise_slot,
    refuse_excel_over_estimate,
    refuse_inline_over_ceiling,
    sync_estimate,
    within_sync_budget,
)
from datasets_api.codes_materialise import (
    DefinitionNotReproducible,
    ExcludedByRegistry,
    MalformedSidecarIdentity,
    MemberNotResolvable,
    MemberSelectionMismatch,
    RenderedArtifactSigningFailed,
    RoleSignatureIncompatible,
    SpecNotExportable,
)
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    SnapshotNotFound,
    SnapshotSigningFailed,
)
from datasets_api.codes_snapshot import SpecNotFound as SnapshotSpecNotFound
from datasets_api.config import DatasetsBuildSettings, DatasetsSettings
from datasets_api.currency import (
    clock_moved,
    derive_signals,
    recheck_rate_provenance,
    recheck_recipe_fingerprint,
)
from datasets_api.door_executor import DoorExecutor, build_door_executor
from datasets_api.doors.mint_errors import classify_mint_exception
from datasets_api.doors.register import classify_register_exception
from datasets_api.doors.snapshot import (
    classify_snapshot_exception,
    read_snapshot_currency,
    read_snapshot_manifest,
)
from datasets_api.generation import enter_shared_generation
from datasets_api.headers import no_store_headers
from datasets_api.kinds.spec_id import admit_sidecar
from datasets_api.manifest import (
    collapsed_head_manifest,
    record_collapsed_head_version,
    reshape_warning_models,
)
from datasets_api.manifest_models import ArtifactEntryModel, Manifest, OptionsEnvelope
from datasets_api.mint_models import DefinitionOptionsRequest, MintOptions
from datasets_api.operation_store import LiveOperationStore, NullOperationStore, OperationStore
from datasets_api.operations.submit import (
    OperationSubmitted,
    operation_submitted_resource,
    submit_over_budget,
)
from datasets_api.operations.task_client import (
    LiveTaskClient,
    NullTaskClient,
    TaskClient,
    TaskClientSettings,
    build_live_task_client,
)
from datasets_api.query_options import (
    QueryOptionsInvalid,
    encode_query_options,
    parse_query_options,
    pointer_for_key,
)
from datasets_api.render import (
    GroupbyCollapseRendering,
    artifact_rendering,
    check_options_supported,
    cited_source_names,
    is_statistics_rendering,
    render_and_store,
    render_and_store_frame1,
    render_and_store_series_statistics,
)
from datasets_api.revision import SentinelRevisionReader
from datasets_api.schema import OperationResource
from datasets_api.serve import (
    DEFINITION_ROUTE_LIVE,
    FRAME1_STORE_FAILED_MARKER,
    ServeDeps,
    bundle_citations,
    log_bundle_not_linked,
    persisted_response,
    prewrite_collection,
    prewrite_definition,
    raw_fetch_guard,
    refs_from_resolved_stat_vars,
    serve_collection,
    serve_definition,
    with_linked_artifacts,
)
from datasets_api.shape import resolve_shapes
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import PhaseTimings, door_call, phase, run_with_timings
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter, rendering_key
from datasets_api.versions import (
    ArtifactRecord,
    HeadStore,
    MemberSignal,
    NullVersionRecordWriter,
    VersionRecordWriter,
)
from datasets_api.versions import NullHeadWriter as NullHeadStore
from dc_sdmx import SdmxClient
from dfo_platform import ProblemIssue, ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.trace import get_trace_id
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from materialise import (
    CollectionFacetProbe,
    EnrichmentLookup,
    MaterialisationRequest,
    probe_collection_facets,
)
from materialise.bundle import ReproducingCall
from materialise.codes import MaterialisationWarning
from materialise.content_version import MintPreimage, mint_preimage_from_trace
from materialise.errors import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    OverCeilingError,
)
from materialise.formats import FRAME1_MEDIA_TYPE, frame1_entity_columns, read_frame1_table
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache, RateTableInvalidError
from materialise.recipes import NonCurrencyUnitError, RateCoverageGapError, RecipeNotDeclaredError
from materialise.revision import RevisionReader
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from origin_catalogue.service import CatalogueService
from series_coverage.index import CoverageIndex
from series_coverage.shared import CoverageService, CoverageSnapshot, get_coverage_service
from spec_registry import SpecIdMismatch
from spec_registry.reader import LiveSpecReader, SpecReader, StoredCollection, StoredDefinition
from statvar_members import MemberEnumerator

logger = logging.getLogger(__name__)

# `DefinitionOptionsRequest` body paths the options route's refusals point at.
_DEFINITION_POINTER = "/definition"
_SELECTION_POINTER = "/resolved_stat_var_dcids"


def _sdmx_deadline(timeout: float) -> float:
    """Reserve up to one second, or 10% of the timeout, for `wait_for` to unwind."""
    grace = min(1.0, timeout * 0.1)
    return time.monotonic() + max(0.0, timeout - grace)


class SpecNotFound(Exception):
    """A valid subject id is absent from the registry. It maps to `spec_not_found`, 404."""

    def __init__(self, *, id: str) -> None:
        self.id = id
        super().__init__(f"no definition or collection is registered at id {id!r}")


class DefinitionBackpressure(Exception):
    """The retrieval semaphore is full. It maps to retryable `service_unavailable`, 503."""


class DefinitionMemberNotResolvable(Exception):
    """Unresolvable members on retrieval.

    It maps to `member_not_resolvable`, 422, without a pointer.

    `truncated` is true when the member list was capped.
    """

    def __init__(self, *, members: tuple[str, ...], truncated: bool = False) -> None:
        self.members = members
        self.truncated = truncated
        detail = f"member(s) did not resolve to exactly one registered shape: {members!r}"
        if truncated:
            detail += "; more members disagreed than are named here"
        super().__init__(detail)


class RegistryInconsistent(Exception):
    """The collection lists a member missing from the same registry read.

    It maps to non-retryable `internal_error`, 500.
    """

    def __init__(self, *, collection_id: str, member_spec_id: str) -> None:
        self.collection_id = collection_id
        self.member_spec_id = member_spec_id
        super().__init__(
            f"collection {collection_id!r} names member {member_spec_id!r}, which its own "
            "registry has no definition for"
        )


class StoredTraceMissingResolvedStatVars(Exception):
    """Older traces may omit `resolved_stat_vars`. `_render_stored_rendering` uses it for
    `groupby` completeness, so this error prevents publication of a partial total.

    It maps to non-retryable `internal_error`, 500.
    """

    def __init__(self, *, subject_id: str, content_version: str) -> None:
        self.subject_id = subject_id
        self.content_version = content_version
        super().__init__(
            f"stored manifest trace for {subject_id!r}@{content_version!r} carries no "
            "resolved_stat_vars key"
        )


class DefinitionMaterialiseRefused(Exception):
    """A materialisation refusal during retrieval. Its `code`, `status`, `detail`, and `issues`
    pass through, but its mint-body pointer is removed for this GET route.
    """

    def __init__(
        self,
        *,
        wrapped: DefaultFacetUndefinedError | FacetNotAvailableError | ExcludedByRegistry,
    ) -> None:
        self.code = wrapped.code
        self.status = wrapped.status
        self.detail = wrapped.detail
        self.issues = wrapped.issues
        super().__init__(str(wrapped))


def classify_definition_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify retrieval errors, then delegate registry, materialisation, and snapshot errors
    to their door classifiers. Rehydration refusals use the mint classifier.
    """
    if isinstance(exc, SpecNotFound):
        return ProblemSpec(status=404, code="spec_not_found", detail=str(exc), retryable=False)
    if isinstance(exc, DefinitionMemberNotResolvable):
        issues = [ProblemIssue(member=member) for member in exc.members]
        return ProblemSpec(
            status=422,
            code="member_not_resolvable",
            detail=str(exc),
            retryable=False,
            truncated=exc.truncated,
            issues=issues,
        )
    if isinstance(exc, RegistryInconsistent):
        logger.error(
            "Datasets definition door's collection read was internally inconsistent: %s", exc
        )
        return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)
    if isinstance(exc, StoredTraceMissingResolvedStatVars):
        logger.error(
            "Datasets definition door's stored manifest trace was missing resolved_stat_vars: %s",
            exc,
        )
        return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)
    if isinstance(exc, DefinitionMaterialiseRefused):
        issues = (
            [issue.model_copy(update={"pointer": None}) for issue in exc.issues]
            if exc.issues
            else None
        )
        return ProblemSpec(
            status=exc.status,
            code=exc.code,
            detail=exc.detail or str(exc),
            retryable=False,
            issues=issues,
        )
    if isinstance(exc, QueryOptionsInvalid):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=str(exc),
            pointer=exc.pointer,
            retryable=False,
        )
    if isinstance(exc, DefinitionBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the definition retrieval door is at its concurrent-materialisation "
            "capacity right now; retry",
            retryable=True,
        )
    spec = classify_register_exception(exc)
    if spec is not None:
        return spec
    spec = classify_mint_exception(exc)
    if spec is not None:
        return spec
    return classify_snapshot_exception(exc)


@dataclass(kw_only=True)
class DefinitionDeps:
    """Retrieval updates `head_store` and `version_writer`. The register door owns identity writes.
    Its lifespan resolves uninjected live dependencies.
    """

    spec_reader: SpecReader
    graph: GraphReader
    sdmx: SdmxClient
    revision_reader: RevisionReader
    artifact_store: ArtifactStore
    materialiser_build: str
    capability_version: str
    ceiling: int
    member_concurrency: int
    # Direct `_serve_*` callers use this graph-bound enumerator. Routes build one per request.
    enumerator: MemberEnumerator
    # Process-wide country-code and label cache, built with the graph and passed to `ServeDeps`.
    enrichment: EnrichmentLookup
    # Process-wide code index passed to `ServeDeps`.
    code_index: CodeIndex
    # Shared by this door's re-materialisations and passed to `ServeDeps`.
    rate_cache: RateTableCache
    # Used for coverage estimates before any SDMX call.
    coverage: CoverageService
    version_writer: VersionRecordWriter = field(default_factory=NullVersionRecordWriter)
    # Currency checks read and confirm/advance the head. Re-materialisation receives the observed
    # version as `expected_content_version` to prevent overwriting a concurrent update. The null
    # default makes every request re-materialise.
    head_store: HeadStore = field(default_factory=NullHeadStore)
    # Loads head responses through the snapshot route's loader, preserving manifest, signing,
    # and artifact-selection behaviour.
    snapshot_reader: SnapshotReader = field(default_factory=lambda: cast(SnapshotReader, None))
    # Bounds re-materialisations across doors sharing this semaphore. `None` means unbounded.
    materialise_semaphore: threading.BoundedSemaphore | None = None
    # Probe interval and forced-rebuild age from resolved settings, converted to `timedelta` at
    # door construction.
    head_recheck_interval: timedelta
    head_max_age: timedelta
    # Injectable clock for currency timestamps and age comparisons.
    now: Clock = utc_now
    # Per-request monotonic deadline forwarded to SDMX. The earlier of it and the transport
    # timeout applies.
    deadline: float | None = None
    # Records over-budget operations. The task client dispatches them.
    operation_store: OperationStore = field(default_factory=NullOperationStore)
    task_client: TaskClient = field(default_factory=NullTaskClient)
    # Supplies sync/async budgets, semaphore wait time, and pivot cell cap. Door construction
    # injects it. The default serves direct `DefinitionDeps` callers.
    settings: DatasetsBuildSettings = field(default_factory=DatasetsBuildSettings.from_env)


def _serve_deps(deps: DefinitionDeps) -> ServeDeps:
    return ServeDeps(
        sdmx=deps.sdmx,
        graph=deps.graph,
        revision_reader=deps.revision_reader,
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
        settings=deps.settings,
    )


def _seed_from_head(spec_id: str, *, deps: DefinitionDeps) -> tuple[str, ...] | None:
    """Recover the resolution seed recorded on the subject's current head.

    The identity omits `resolved_stat_var_dcids`. The trace's `resolved_stat_vars` list stores it.
    Load the manifest without artifacts and strip `dcid:` prefixes for `resolve_definition_members`.

    Return `None` when the head or version is unavailable or revoked, or an older trace lacks the
    field. Resolution then runs unseeded, proving completeness from its walked window or refusing
    `over_ceiling`. This helper returns a seed or `None` for every input.

    The recorded list preserves the population resolved by a truncated-but-proven-complete mint.
    It adds candidates and keeps every member the walk would find, so later reads use the same
    population.
    """
    head = deps.head_store.read_head(spec_id)
    if head is None:
        return None
    try:
        manifest = _load_stored(
            subject_id=spec_id,
            content_version=head.content_version,
            deps=deps,
            include_artifacts=False,
        )
    except (SnapshotNotFound, SnapshotRevoked, SnapshotSpecNotFound):
        return None
    resolved_stat_vars = manifest.trace.get("resolved_stat_vars")
    if resolved_stat_vars is None:
        return None
    refs = refs_from_resolved_stat_vars(cast("list[dict[str, object]]", resolved_stat_vars))
    return tuple(ref.removeprefix("dcid:") for ref in refs)


def _resolve_spec_requests(
    *, spec_id: str, deps: DefinitionDeps, seed: Sequence[str] | None = None
) -> tuple[StoredDefinition, Sequence[MaterialisationRequest]]:
    """Read and rehydrate a stored spec, seeding described-arm resolution with `seed` or, when
    the caller holds none, from the spec's head when possible.

    The worker passes the members its operation stored, which the submitting door resolved.
    A first mint has no head, so those members are the only seed that reproduces a family a
    posted root proved complete.

    Wrap `MemberNotResolvable` so the GET route uses retrieval's pointer-free refusal.
    """
    with phase("resolve"):
        stored = deps.spec_reader.read_definition(spec_id)
        if stored is None:
            raise SpecNotFound(id=spec_id)
        return stored, _rehydrate_stored(stored, deps=deps, seed=seed)


def _rehydrate_stored(
    stored: StoredDefinition, *, deps: DefinitionDeps, seed: Sequence[str] | None
) -> Sequence[MaterialisationRequest]:
    """Rehydrate a registered spec, seeded with `seed` or, when that is `None`, its head's."""
    if seed is None:
        seed = _seed_from_head(stored.spec_id, deps=deps)
    try:
        _shapes, requests = rehydrate_definition(
            stored,
            graph=deps.graph,
            enumerator=deps.enumerator,
            code_index=deps.code_index,
            seed=seed,
        )
    except MemberNotResolvable as exc:
        raise DefinitionMemberNotResolvable(members=exc.members, truncated=exc.truncated) from exc
    return requests


def _verify_posted_definition(*, spec_id: str, body: DefinitionOptionsRequest) -> None:
    """Refuse a posted definition whose identity does not hash to `spec_id` or whose
    `definition.spec_id` names another id."""
    try:
        verify_sidecar_definition(body.definition, spec_id=spec_id)
    except IdentityError as exc:
        raise MalformedSidecarIdentity(detail=str(exc), pointer=_DEFINITION_POINTER) from exc
    if body.definition.spec_id != spec_id:
        # The identity hashes to the path id, so the definition's own id is the one at fault.
        raise SpecIdMismatch(supplied=body.definition.spec_id, computed=spec_id)


def _resolve_posted_spec_requests(
    *, spec_id: str, body: DefinitionOptionsRequest, deps: DefinitionDeps
) -> tuple[StoredDefinition, Sequence[MaterialisationRequest]]:
    """Resolve the definition `POST /options` carries for `spec_id`, writing nothing.

    `_verify_posted_definition` has already checked the body against `spec_id`. A registered id
    resolves as the GET route resolves it. An unregistered one is admitted as
    `preview_spec_id` admits a sidecar, seeded with the posted dcids, and becomes the
    `StoredDefinition` a registration of it would read back, so its materialisation and envelope
    are the ones a registered id gets. Member refusals keep the GET route's pointer-free form. A
    posted dcid outside the family points into this body.
    """
    with phase("resolve"):
        stored = deps.spec_reader.read_definition(spec_id)
        if stored is not None:
            return stored, _rehydrate_stored(stored, deps=deps, seed=None)
        try:
            recomputed, shapes = admit_sidecar(
                body.definition,
                spec_id=spec_id,
                scope=None,
                deps=deps,
                selection=body.resolved_stat_var_dcids,
            )
        except MemberNotResolvable as exc:
            raise DefinitionMemberNotResolvable(
                members=exc.members, truncated=exc.truncated
            ) from exc
        except MemberSelectionMismatch as exc:
            exc.pointer = _SELECTION_POINTER
            raise
        identity = recomputed.identity
        requests = tuple(
            to_materialisation_request(identity, shape=shape, code_index=deps.code_index)
            for shape in shapes
        )
    stored = StoredDefinition(
        spec_id=spec_id,
        schema_version=body.definition.schema_version,
        identity=identity,
        identity_bytes=recomputed.identity_bytes,
        identity_digest=recomputed.identity_digest,
    )
    return stored, requests


def _stamp_trace(
    manifest: Manifest, *, served_from: str, confirmed_at: datetime, coverage_stamp: str | None
) -> Manifest:
    """Stamp the serving outcome, confirmation time, and current coverage snapshot onto a fresh
    copy.

    Head responses need this step because their stored manifest carries a prior coverage stamp.
    Materialised responses already carry the same snapshot's stamp. This stamp records the request's
    admission estimate. Artifact records retain their write-time metadata.
    """
    trace = dict(manifest.trace)
    trace["served_from"] = served_from
    trace["confirmed_at"] = confirmed_at.isoformat()
    return manifest.model_copy(update={"trace": trace, "series_coverage_stamp": coverage_stamp})


def _load_stored(
    *,
    subject_id: str,
    content_version: str,
    deps: DefinitionDeps,
    include_artifacts: bool,
    rendering_key: str | None = None,
) -> Manifest:
    """Load a stored version through the snapshot loaders.

    `include_artifacts=True` uses `read_snapshot_manifest`, with signing and expiry checks. An
    optional `rendering_key` narrows artifacts to the requested rendering. `False` uses
    `read_snapshot_currency`, which checks the version and revocation without reading artifacts or
    calling the artifact store. A fresh head relies on this as its only existence check.
    """
    if not include_artifacts:
        return read_snapshot_currency(
            id=subject_id,
            content_version=content_version,
            spec_reader=deps.spec_reader,
            snapshot_reader=deps.snapshot_reader,
            settings=deps.settings,
        )
    return read_snapshot_manifest(
        id=subject_id,
        content_version=content_version,
        spec_reader=deps.spec_reader,
        snapshot_reader=deps.snapshot_reader,
        artifact_store=deps.artifact_store,
        now=deps.now,
        settings=deps.settings,
        rendering_key=rendering_key,
    )


def _record_head_hit_collapse(
    *,
    manifest: Manifest,
    collapse: GroupbyCollapseRendering,
    artifact: ArtifactEntryModel,
    reshape_warnings: tuple[MaterialisationWarning, ...],
    subject_id: str,
    schema_version: int,
    base_content_version: str,
    linked: bool,
    deps: DefinitionDeps,
    trace_id: str | None,
) -> Manifest:
    """Record and return a head-hit groupby collapse under its collapsed content version.

    Store the artifact under `collapse.content_version` to match the collapsed identity, and
    link it only when `linked` (its graph metadata reads answered). Recording the same pair
    again is harmless. When the collapsed version is already on record, return it as a repeat
    mint does (`persisted_response`): its stored body with the artifacts linked for it, so its
    stored warnings stand. An unlinked render is listed only while no servable artifact is
    linked under its key (`with_linked_artifacts`). The base record precedes the render, so
    every writer dates and cites the collapsed version as the base does.
    """
    collapsed_frame1_record: ArtifactRecord | None = None
    try:
        collapsed_frame1_record = render_and_store_frame1(
            frame_chunks=(collapse.frame,),
            subject_id=subject_id,
            content_version=cast(str, collapse.content_version),
            artifact_store=deps.artifact_store,
        )
    except ArtifactStoreError:
        logger.error(
            "%s spec_id=%s content_version=%s",
            FRAME1_STORE_FAILED_MARKER,
            subject_id,
            collapse.content_version,
            exc_info=True,
        )
    collapsed_statistics_sink = SeriesStatisticsSink()
    collapsed_statistics_sink.write(compute_series_statistics_table(collapse.frame))
    collapsed_statistics_file = collapsed_statistics_sink.finish()
    collapsed_statistics_artifact = render_and_store_series_statistics(
        data=collapsed_statistics_file.data,
        subject_id=subject_id,
        content_version=cast(str, collapse.content_version),
        source_names=cited_source_names(
            [source.ref for source in manifest.citation.sources],
            manifest.citation.sources_by_ref(),
        ),
        prepared_at=manifest.prepared_at,
        artifact_store=deps.artifact_store,
    )
    call_warnings = reshape_warning_models(reshape_warnings)
    collapsed_manifest = collapsed_head_manifest(
        manifest=manifest,
        collapse_frame=collapse.frame,
        artifact=artifact,
        statistics_artifact=collapsed_statistics_artifact,
        series_count=collapsed_statistics_file.series_count,
        content_version=cast(str, collapse.content_version),
        extra_warnings=call_warnings,
    )
    created = record_collapsed_head_version(
        # The body leaves artifacts out, so the copy changes only which artifacts are linked.
        manifest=(
            collapsed_manifest
            if linked
            else collapsed_manifest.model_copy(
                update={"artifacts": [collapsed_statistics_artifact]}
            )
        ),
        subject_id=subject_id,
        schema_version=schema_version,
        base_content_version=base_content_version,
        record_only_artifacts=(
            [collapsed_frame1_record] if collapsed_frame1_record is not None else []
        ),
        version_writer=deps.version_writer,
        trace_id=trace_id,
    )
    if created:
        if linked:
            return collapsed_manifest
        return with_linked_artifacts(
            collapsed_manifest,
            subject_id=subject_id,
            snapshot_reader=deps.snapshot_reader,
            artifact_store=deps.artifact_store,
            now=deps.now(),
        )
    return persisted_response(
        deps.snapshot_reader,
        subject_id=subject_id,
        content_version=cast(str, collapse.content_version),
        body=None,
        served=collapsed_manifest,
        call_warnings=call_warnings,
        artifact_store=deps.artifact_store,
        now=deps.now(),
    )


def _render_stored_rendering(
    *,
    subject_id: str,
    content_version: str,
    options: MintOptions,
    manifest: Manifest,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_index: CoverageIndex | None,
) -> Manifest | None:
    """Render a missing request format from the stored frame1 artifact on a head hit.

    Use the servable frame1 location and the same provenance, definition, resolved-member
    population, and `ReproducingCall` as fresh rendering. The full population keeps `groupby`
    completeness checks consistent, including zero-row members. A groupby request also needs the
    stored trace's `mint_facets` preimage. If absent, return `None` so fresh materialisation
    can assign the correct collapsed content version.

    Return the collapsed manifest when groupby changes the content version. Otherwise record the
    artifact under the base version and add it to the manifest. Return `None` for collections, a
    missing frame1 location, missing or corrupt frame1 bytes, or a missing groupby preimage. Raise
    `StoredTraceMissingResolvedStatVars` when the groupby population is incomplete. Other render
    errors propagate to the shared classifier.

    The frame1 read, decode, and render hold a `bounded_materialise_slot`. A full semaphore
    raises retryable `MintBackpressure` (503) once the wait runs out.

    A bundle cites the stored citation with its empty fields filled by a live read
    (`bundle_citations`). A render whose graph metadata reads failed is returned unlinked.
    The response lists the artifact linked under the request's rendering key when a
    concurrent call linked one first (`with_linked_artifacts`).
    """
    if manifest.definition is None:
        return None
    locations = deps.snapshot_reader.read_artifact_locations(subject_id, content_version)
    location = servable_location_of_media_type(
        locations, media_type=FRAME1_MEDIA_TYPE, now=deps.now()
    )
    if location is None:
        return None
    # A frame1 holds up to `FRAME1_MAX_ROWS` rows and decodes whole, so the load and the render
    # share the semaphore that bounds a mint's materialisation.
    with bounded_materialise_slot(
        deps.materialise_semaphore,
        wait_seconds=deps.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        try:
            data = deps.artifact_store.get(
                artifact_digest=location.artifact_digest, tier=Tier(location.tier)
            )
        except (ArtifactNotFoundError, ArtifactIntegrityError):
            return None
        table = read_frame1_table(data)
        entity_columns = frame1_entity_columns(table)
        facets = cast("list[dict[str, object]]", manifest.trace.get("facets", []))
        provenance_ref = cast(str, next(f["ref"] for f in facets if f["axis"] == "provenance"))
        resolved_stat_vars = manifest.trace.get("resolved_stat_vars")
        if resolved_stat_vars is None:
            raise StoredTraceMissingResolvedStatVars(
                subject_id=subject_id, content_version=content_version
            )
        resolved_stat_var_refs = refs_from_resolved_stat_vars(
            cast("list[dict[str, object]]", resolved_stat_vars)
        )
        spec_dict: dict[str, object] = {
            "schema_version": manifest.definition.schema_version,
            "spec_id": manifest.definition.spec_id,
            "identity": manifest.definition.identity,
        }
        groupby_preimage: MintPreimage | None = None
        if options.groupby is not None:
            groupby_preimage = mint_preimage_from_trace(manifest.trace)
            if groupby_preimage is None:
                return None
        citations, documentation_complete = bundle_citations(
            manifest.citation,
            options=options,
            on_record=True,
            # A head hit renders only when no artifact is linked under the request's key.
            linked=False,
            documentation_complete=True,
            graph=deps.graph,
            deadline=deps.deadline,
            metadata_read_seconds=deps.settings.metadata_read_seconds,
            coverage_index=coverage_index,
        )
        rendered = render_and_store(
            frame_chunks=(table,),
            options=options,
            spec=spec_dict,
            provenance_refs=[provenance_ref],
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
            groupby_preimage=groupby_preimage,
            deadline=deps.deadline,
            metadata_read_seconds=deps.settings.metadata_read_seconds,
        )
        artifact, reshape_warnings, collapse = (
            rendered.artifact,
            rendered.warnings,
            rendered.collapse,
        )
        rendered_content_version = (
            collapse.content_version
            if collapse is not None and collapse.content_version is not None
            else content_version
        )
        linked = documentation_complete and rendered.labels_complete
        if not linked:
            log_bundle_not_linked(
                subject_id=subject_id,
                content_version=rendered_content_version,
                refs=[provenance_ref],
                documentation_complete=documentation_complete,
                labels_complete=rendered.labels_complete,
            )
        if collapse is not None and rendered_content_version != content_version:
            return _record_head_hit_collapse(
                manifest=manifest,
                collapse=collapse,
                artifact=artifact,
                reshape_warnings=reshape_warnings,
                subject_id=subject_id,
                schema_version=manifest.definition.schema_version,
                base_content_version=content_version,
                linked=linked,
                deps=deps,
                trace_id=trace_id,
            )
        if linked:
            deps.version_writer.record_artifact(
                subject_id=subject_id,
                content_version=content_version,
                artifact=ArtifactRecord(
                    artifact_digest=artifact.artifact_digest,
                    rendering=artifact.rendering,
                    media_type=artifact.media_type,
                ),
                trace_id=trace_id,
            )
        # Keep any stored series-statistics artifact. This rendering shares its content version.
        rendered_manifest = manifest.model_copy(
            update={
                "artifacts": [artifact],
                "warnings": list(manifest.warnings)
                + list(reshape_warning_models(reshape_warnings)),
            }
        )
        linked_manifest = with_linked_artifacts(
            rendered_manifest,
            subject_id=subject_id,
            snapshot_reader=deps.snapshot_reader,
            artifact_store=deps.artifact_store,
            now=deps.now(),
        )
        return linked_manifest.model_copy(
            update={"artifacts": [*linked_manifest.artifacts, *manifest.artifacts]}
        )


def _serve_head(
    *,
    subject_id: str,
    content_version: str,
    confirmed_at: datetime,
    deps: DefinitionDeps,
    write_confirmation: bool,
    key: str | None,
    options: MintOptions | None,
    trace_id: str | None,
    coverage_snapshot: CoverageSnapshot,
) -> Manifest | None:
    """Load and stamp the stored version, narrowed to the requested rendering when supplied.

    Return `None` when neither stored nor on-demand rendering can serve the request. The caller
    then re-materialises. `coverage_snapshot` is the currency check's shared snapshot, which
    stamps the response and feeds an on-demand bundle's documentation read.
    """
    manifest = _load_stored(
        subject_id=subject_id,
        content_version=content_version,
        deps=deps,
        include_artifacts=write_confirmation,
        rendering_key=key,
    )
    # The snapshot loader retains `series-statistics/1` even when the requested rendering is
    # absent, so test for a non-statistics artifact.
    if key is not None and not any(
        not is_statistics_rendering(a.rendering) for a in manifest.artifacts
    ):
        rendered = _render_stored_rendering(
            subject_id=subject_id,
            content_version=content_version,
            options=cast(MintOptions, options),
            manifest=manifest,
            deps=deps,
            trace_id=trace_id,
            coverage_index=coverage_snapshot.index,
        )
        if rendered is None:
            return None
        manifest = rendered
    return _stamp_trace(
        manifest,
        served_from="head",
        confirmed_at=confirmed_at,
        coverage_stamp=coverage_snapshot.stamp,
    )


def _resolve_currency(
    *,
    subject_id: str,
    resolve_probe_requests: Callable[[], Sequence[MaterialisationRequest]],
    combine: bool,
    deps: DefinitionDeps,
    materialise_fresh: Callable[
        [Sequence[MemberSignal] | None, CollectionFacetProbe, str | None, CoverageSnapshot],
        Manifest,
    ],
    write_confirmation: bool,
    options: MintOptions | None,
    trace_id: str | None,
) -> Manifest:
    """Serve a current head or re-materialise after checking its currency.

    Read one coverage snapshot for the probe, admission estimate, fetch, and response stamp. A
    head confirmed within `head_recheck_interval` skips rehydration, facet probing, and SDMX fetches
    when the requested rendering is stored, unless a window its definition left open now ends at
    another year (`currency.clock_moved`). A missing format may need artifact-store reads.
    Otherwise resolve once, probe facets, derive signals, and recheck stored rate provenance and
    recipe fingerprints.

    If the signal matches and `materialised_at` is younger than `head_max_age`, serve the head. A
    GET confirms it. If another call advanced it first, read and serve that winner. Maximum age
    forces a rebuild for corrections that leave counts and dates unchanged. Render missing formats
    from stored frame1. If that fails or the signal changes, re-materialise using the same probe
    and observed content version as `expected_content_version`, preventing a concurrent head update
    from being overwritten.

    `/options` records no version and confirms no head. One exception: when the clock moved the
    head and the re-materialisation reproduces its content version, `/options` advances the head
    to the signal it derived, so the head is re-read once whichever route reads it first. A
    re-read yielding a new content version leaves the head as it is, and every `/options` call
    re-reads it until a GET or mint records that version. Its `confirmed_at` is the computation
    time. GET responses use the head store's timestamp.
    Return a manifest stamped by `_stamp_trace`.
    """
    coverage_snapshot = deps.coverage.snapshot()
    head = deps.head_store.read_head(subject_id)
    now = deps.now()
    key = (
        rendering_key(artifact_rendering(options, site=deps.settings.site))
        if options is not None and write_confirmation
        else None
    )

    if (
        head is not None
        and now - head.confirmed_at < deps.head_recheck_interval
        and not clock_moved(head.signal)
    ):
        served = _serve_head(
            subject_id=subject_id,
            content_version=head.content_version,
            confirmed_at=head.confirmed_at,
            deps=deps,
            write_confirmation=write_confirmation,
            key=key,
            options=options,
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )
        if served is not None:
            return served

    probe_requests = resolve_probe_requests()
    # Share the rebuild semaphore. Wait only until the configured limit or request deadline, then
    # release the slot after probing so a rebuild can acquire its own.
    with bounded_materialise_slot(
        deps.materialise_semaphore,
        wait_seconds=deps.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        probe = probe_collection_facets(
            probe_requests,
            graph=deps.graph,
            availability=deps.sdmx,
            settle=True,
            revision_reader=deps.revision_reader,
            coverage=coverage_snapshot.index,
            deadline=deps.deadline,
        )
    signals = derive_signals(probe_requests, probe, graph=deps.graph, combine=combine)
    if signals is not None and head is not None:
        signals = recheck_rate_provenance(
            signals,
            head.signal,
            graph=deps.graph,
            provenance_refresh=probe.provenance_refresh,
        )
        signals = recheck_recipe_fingerprint(signals, head.signal)

    if (
        head is not None
        and signals is not None
        and signals == head.signal
        and now - head.materialised_at < deps.head_max_age
    ):
        if write_confirmation:
            written = deps.head_store.confirm_head(
                subject_id=subject_id,
                content_version=head.content_version,
            )
            # A concurrent update won. Read and serve its head.
            winner = written if written is not None else deps.head_store.read_head(subject_id)
        else:
            winner = head
        if winner is not None:
            served = _serve_head(
                subject_id=subject_id,
                content_version=winner.content_version,
                confirmed_at=winner.confirmed_at,
                deps=deps,
                write_confirmation=write_confirmation,
                key=key,
                options=options,
                trace_id=trace_id,
                coverage_snapshot=coverage_snapshot,
            )
            if served is not None:
                return served
        # A missing winner or unrenderable head falls through to fresh materialisation.

    manifest = materialise_fresh(
        signals if write_confirmation else None,
        probe,
        head.content_version if head is not None else None,
        coverage_snapshot,
    )
    if write_confirmation and signals is not None:
        # Re-read the head only when materialisation received a signal and wrote it.
        refreshed = deps.head_store.read_head(subject_id)
        confirmed_at = refreshed.confirmed_at if refreshed is not None else now
    else:
        if (
            signals is not None
            and head is not None
            and clock_moved(head.signal)
            and manifest.content_version == head.content_version
        ):
            # The stored version still holds, so the head takes the window this read derived.
            deps.head_store.advance_head(
                subject_id=subject_id,
                content_version=head.content_version,
                signal=signals,
                expected_content_version=head.content_version,
            )
        confirmed_at = now
    return _stamp_trace(
        manifest,
        served_from="materialise",
        confirmed_at=confirmed_at,
        coverage_stamp=coverage_snapshot.stamp,
    )


def _serve_spec(
    *, spec_id: str, options: MintOptions, deps: DefinitionDeps, trace_id: str | None
) -> Manifest:
    """Memoise registry lookup and rehydration so the currency check resolves at most once."""

    @functools.cache
    def _resolve() -> tuple[StoredDefinition, Sequence[MaterialisationRequest]]:
        return _resolve_spec_requests(spec_id=spec_id, deps=deps)

    def _materialise_fresh(
        signal: Sequence[MemberSignal] | None,
        probe: CollectionFacetProbe,
        expected_content_version: str | None,
        coverage_snapshot: CoverageSnapshot,
    ) -> Manifest:
        stored, requests = _resolve()
        try:
            # Estimate before network calls. Refuse Excel over the row cap or submit over-budget
            # work as an operation. Use the currency check's shared coverage snapshot.
            estimate = sync_estimate(
                requests, index=coverage_snapshot.index, member_concurrency=deps.member_concurrency
            )
            refuse_excel_over_estimate(estimate, format=options.format)
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
            manifest = serve_definition(
                stored=stored,
                requests=requests,
                options=options,
                deps=_serve_deps(deps),
                trace_id=trace_id,
                signal=signal,
                expected_content_version=expected_content_version,
                precomputed_probe=probe,
                ceiling=raw_fetch_guard(estimate, floor=deps.ceiling),
                coverage_stamp=coverage_snapshot.stamp,
                coverage_index=coverage_snapshot.index,
            )
        except (
            DefaultFacetUndefinedError,
            FacetNotAvailableError,
            ExcludedByRegistry,
        ) as exc:
            raise DefinitionMaterialiseRefused(wrapped=exc) from exc
        return manifest

    return _resolve_currency(
        subject_id=spec_id,
        resolve_probe_requests=lambda: _resolve()[1],
        combine=True,
        deps=deps,
        materialise_fresh=_materialise_fresh,
        write_confirmation=True,
        options=options,
        trace_id=trace_id,
    )


def _serve_spec_options(
    *,
    spec_id: str,
    deps: DefinitionDeps,
    trace_id: str | None,
    body: DefinitionOptionsRequest | None = None,
) -> OptionsEnvelope:
    """Run `_serve_spec`'s currency check as a prewrite that records no version, and advance
    the head only for a clock-moved head the rebuild reproduces (`_resolve_currency`).

    A posted `body` is verified against `spec_id` before the head is read, since a fresh head
    answers without resolving. It then resolves through `_resolve_posted_spec_requests`. An
    unregistered spec has no head, so it always rebuilds, and the probe and rebuild each hold a
    `bounded_materialise_slot` as a mint's do.

    A rebuild runs inline under `deps.ceiling`, so `refuse_inline_over_ceiling` refuses an
    estimate above it before any fetch.
    """
    if body is not None:
        with phase("resolve"):
            _verify_posted_definition(spec_id=spec_id, body=body)

    @functools.cache
    def _resolve() -> tuple[StoredDefinition, Sequence[MaterialisationRequest]]:
        if body is None:
            return _resolve_spec_requests(spec_id=spec_id, deps=deps)
        return _resolve_posted_spec_requests(spec_id=spec_id, body=body, deps=deps)

    def _materialise_fresh(
        _signal: Sequence[MemberSignal] | None,
        probe: CollectionFacetProbe,
        _expected_content_version: str | None,
        coverage_snapshot: CoverageSnapshot,
    ) -> Manifest:
        stored, requests = _resolve()
        try:
            estimate = sync_estimate(
                requests, index=coverage_snapshot.index, member_concurrency=deps.member_concurrency
            )
            refuse_inline_over_ceiling(estimate, ceiling=deps.ceiling)
            prewrite = prewrite_definition(
                stored=stored,
                requests=requests,
                sdmx=deps.sdmx,
                graph=deps.graph,
                revision_reader=deps.revision_reader,
                ceiling=deps.ceiling,
                materialiser_build=deps.materialiser_build,
                capability_version=deps.capability_version,
                enrichment=deps.enrichment,
                code_index=deps.code_index,
                rate_cache=deps.rate_cache,
                member_concurrency=deps.member_concurrency,
                materialise_wait_seconds=deps.settings.materialise_wait_seconds,
                settings=deps.settings,
                materialise_semaphore=deps.materialise_semaphore,
                precomputed_probe=probe,
                coverage_index=coverage_snapshot.index,
                deadline=deps.deadline,
                documented=False,
            )
        except (
            DefaultFacetUndefinedError,
            FacetNotAvailableError,
            ExcludedByRegistry,
        ) as exc:
            raise DefinitionMaterialiseRefused(wrapped=exc) from exc
        return prewrite.manifest

    manifest = _resolve_currency(
        subject_id=spec_id,
        resolve_probe_requests=lambda: _resolve()[1],
        combine=True,
        deps=deps,
        materialise_fresh=_materialise_fresh,
        write_confirmation=False,
        options=None,
        trace_id=trace_id,
    )
    return _to_options_envelope(manifest)


def _resolve_collection_requests(
    *, collection_id: str, deps: DefinitionDeps
) -> tuple[StoredCollection, list[tuple[StoredDefinition, MaterialisationRequest]]]:
    """Resolve collection members in stored order before materialisation.

    Pinned members share one batched shape read. Described members resolve individually. Only
    `MemberNotResolvable` failures aggregate. Other errors abort at the failing member, preserving
    stored-order precedence.
    """
    with phase("resolve"):
        stored_collection = deps.spec_reader.read_collection(collection_id)
        if stored_collection is None:
            raise SpecNotFound(id=collection_id)

        stored_members = deps.spec_reader.read_definitions(stored_collection.member_spec_ids)

        pinned_refs: dict[str, str] = {}
        for member_spec_id in stored_collection.member_spec_ids:
            member_stored = stored_members.get(member_spec_id)
            if member_stored is None:
                continue
            ref = pinned_member_ref(member_stored.identity)
            if ref is not None:
                pinned_refs[member_spec_id] = ref
        shapes_by_ref = resolve_shapes(tuple(pinned_refs.values()), graph=deps.graph)

        failing: list[str] = []
        requests: list[tuple[StoredDefinition, MaterialisationRequest]] = []
        for member_spec_id in stored_collection.member_spec_ids:
            member_stored = stored_members.get(member_spec_id)
            if member_stored is None:
                raise RegistryInconsistent(
                    collection_id=collection_id, member_spec_id=member_spec_id
                )
            pinned_ref = pinned_refs.get(member_spec_id)
            if pinned_ref is not None:
                shape = shapes_by_ref.get(pinned_ref)
                if shape is None:
                    failing.append(member_spec_id)
                    continue
                check_reproducible(member_stored.identity, spec_id=member_spec_id, shape=shape)
                requests.append(
                    (
                        member_stored,
                        to_materialisation_request(
                            member_stored.identity, shape=shape, code_index=deps.code_index
                        ),
                    )
                )
                continue
            try:
                _member_shapes, member_requests = rehydrate_definition(
                    member_stored,
                    graph=deps.graph,
                    enumerator=deps.enumerator,
                    code_index=deps.code_index,
                    accepts_seed=False,
                )
            except MemberNotResolvable as exc:
                failing.extend(exc.members)
                continue
            if len(member_requests) > 1:
                # Collections require each described-arm member to resolve to one StatVar series.
                raise OverCeilingError(
                    f"collection member {member_spec_id!r} enumerates "
                    f"{len(member_requests)} StatVar series against the graph, which a "
                    "collection cannot express as a single member",
                    ceiling_kind="member_cardinality",
                    count=len(member_requests),
                    cap=1,
                )
            requests.append((member_stored, member_requests[0]))
        if failing:
            raise DefinitionMemberNotResolvable(members=tuple(failing))
    return stored_collection, requests


def _serve_collection(
    *, collection_id: str, options: MintOptions, deps: DefinitionDeps, trace_id: str | None
) -> Manifest:
    """Memoise collection resolution so the currency check resolves at most once."""

    @functools.cache
    def _resolve() -> tuple[
        StoredCollection, list[tuple[StoredDefinition, MaterialisationRequest]]
    ]:
        return _resolve_collection_requests(collection_id=collection_id, deps=deps)

    def _probe_requests() -> list[MaterialisationRequest]:
        _stored_collection, requests = _resolve()
        return [request for _member_stored, request in requests]

    def _materialise_fresh(
        signal: Sequence[MemberSignal] | None,
        probe: CollectionFacetProbe,
        expected_content_version: str | None,
        coverage_snapshot: CoverageSnapshot,
    ) -> Manifest:
        stored_collection, requests = _resolve()
        try:
            estimate = sync_estimate(
                [request for _member_stored, request in requests],
                index=coverage_snapshot.index,
                member_concurrency=deps.member_concurrency,
            )
            refuse_excel_over_estimate(estimate, format=options.format)
            submit_over_budget(
                subject_id=stored_collection.collection_id,
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
            )
            manifest = serve_collection(
                stored=stored_collection,
                requests=requests,
                options=options,
                deps=_serve_deps(deps),
                trace_id=trace_id,
                signal=signal,
                expected_content_version=expected_content_version,
                precomputed_probe=probe,
                ceiling=raw_fetch_guard(estimate, floor=deps.ceiling),
                coverage_stamp=coverage_snapshot.stamp,
                coverage_index=coverage_snapshot.index,
            )
        except (
            DefaultFacetUndefinedError,
            FacetNotAvailableError,
            ExcludedByRegistry,
        ) as exc:
            raise DefinitionMaterialiseRefused(wrapped=exc) from exc
        return manifest

    return _resolve_currency(
        subject_id=collection_id,
        resolve_probe_requests=_probe_requests,
        combine=False,
        deps=deps,
        materialise_fresh=_materialise_fresh,
        write_confirmation=True,
        options=options,
        trace_id=trace_id,
    )


def _serve_collection_options(
    *, collection_id: str, deps: DefinitionDeps, trace_id: str | None
) -> OptionsEnvelope:
    """Run `_serve_collection`'s currency check as a prewrite that records no version, and
    advance the head only for a clock-moved head the rebuild reproduces (`_resolve_currency`).

    A rebuild runs inline under `deps.ceiling`, so `refuse_inline_over_ceiling` refuses an
    estimate above it before any fetch.
    """

    @functools.cache
    def _resolve() -> tuple[
        StoredCollection, list[tuple[StoredDefinition, MaterialisationRequest]]
    ]:
        return _resolve_collection_requests(collection_id=collection_id, deps=deps)

    def _probe_requests() -> list[MaterialisationRequest]:
        _stored_collection, requests = _resolve()
        return [request for _member_stored, request in requests]

    def _materialise_fresh(
        _signal: Sequence[MemberSignal] | None,
        probe: CollectionFacetProbe,
        _expected_content_version: str | None,
        coverage_snapshot: CoverageSnapshot,
    ) -> Manifest:
        stored_collection, requests = _resolve()
        try:
            estimate = sync_estimate(
                [request for _member_stored, request in requests],
                index=coverage_snapshot.index,
                member_concurrency=deps.member_concurrency,
            )
            refuse_inline_over_ceiling(estimate, ceiling=deps.ceiling)
            prewrite = prewrite_collection(
                stored=stored_collection,
                requests=requests,
                sdmx=deps.sdmx,
                graph=deps.graph,
                revision_reader=deps.revision_reader,
                ceiling=deps.ceiling,
                materialiser_build=deps.materialiser_build,
                capability_version=deps.capability_version,
                enrichment=deps.enrichment,
                member_concurrency=deps.member_concurrency,
                code_index=deps.code_index,
                rate_cache=deps.rate_cache,
                materialise_wait_seconds=deps.settings.materialise_wait_seconds,
                settings=deps.settings,
                materialise_semaphore=deps.materialise_semaphore,
                precomputed_probe=probe,
                coverage_index=coverage_snapshot.index,
                deadline=deps.deadline,
                snapshot_reader=deps.snapshot_reader,
                documented=False,
            )
        except (
            DefaultFacetUndefinedError,
            FacetNotAvailableError,
            ExcludedByRegistry,
        ) as exc:
            raise DefinitionMaterialiseRefused(wrapped=exc) from exc
        return prewrite.manifest

    manifest = _resolve_currency(
        subject_id=collection_id,
        resolve_probe_requests=_probe_requests,
        combine=False,
        deps=deps,
        materialise_fresh=_materialise_fresh,
        write_confirmation=False,
        options=None,
        trace_id=trace_id,
    )
    return _to_options_envelope(manifest)


def _to_options_envelope(manifest: Manifest) -> OptionsEnvelope:
    """Return only the manifest fields exposed by the write-free `/options` route."""
    return OptionsEnvelope(
        schema_version=manifest.schema_version,
        definition=manifest.definition,
        collection=manifest.collection,
        content_version=manifest.content_version,
        coverage=manifest.coverage,
        output_options=manifest.output_options,
        warnings=manifest.warnings,
        trace=manifest.trace,
    )


@dataclass(frozen=True)
class _DefinitionInjected:
    """Records which dependencies the caller injected."""

    spec_reader: bool
    graph: bool
    sdmx: bool
    artifact_store: bool
    task_client: bool
    version_writer: bool
    head_store: bool
    operation_store: bool
    snapshot_reader: bool


@dataclass(frozen=True)
class _DefinitionDoorState:
    """State shared by the lifespan, retrieval runner, and routes."""

    deps: DefinitionDeps
    settings: DatasetsBuildSettings
    budget: ShutdownBudget
    door_executor: DoorExecutor
    injected: _DefinitionInjected
    catalogue: CatalogueService | None


@dataclass
class _DefinitionStoreBorrow:
    """Tracks the pool borrow before store construction can raise, so cleanup closes it."""

    store: LiveDatasetStore | None = None
    opened_here: bool = False


def _build_definition_deps(
    *,
    spec_reader: SpecReader | None,
    graph: GraphReader | None,
    sdmx: SdmxClient | None,
    revision_reader: RevisionReader | None,
    artifact_store: ArtifactStore | None,
    materialiser_build: str,
    capability_version: str,
    ceiling: int,
    member_concurrency: int,
    version_writer: VersionRecordWriter | None,
    head_store: HeadStore | None,
    snapshot_reader: SnapshotReader | None,
    materialise_semaphore: threading.BoundedSemaphore | None,
    head_recheck_seconds: int,
    head_max_age_seconds: int,
    now: Clock,
    code_index: CodeIndex | None,
    rate_cache: RateTableCache,
    coverage: CoverageService,
    operation_store: OperationStore | None,
    task_client: TaskClient | None,
    settings: DatasetsBuildSettings,
) -> DefinitionDeps:
    """Build non-client dependencies at door construction. The lifespan creates live clients.

    The live snapshot reader shares the live spec reader's store. An injected `spec_reader` leaves
    `snapshot_reader` unset unless the caller supplies it.
    """
    return DefinitionDeps(
        spec_reader=cast(SpecReader, spec_reader),
        graph=cast(GraphReader, graph),
        sdmx=cast(SdmxClient, sdmx),
        revision_reader=(
            revision_reader if revision_reader is not None else SentinelRevisionReader()
        ),
        artifact_store=cast(ArtifactStore, artifact_store),
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        ceiling=ceiling,
        member_concurrency=member_concurrency,
        # This is only the base instance a caller that skips
        # `get_definition` and invokes `_serve_spec`/`_serve_collection`
        # directly sees. `get_definition` builds its own throwaway
        # `MemberEnumerator` per request instead (mirrors
        # `datasets_api.doors.mint.mint_endpoint`'s identical guard).
        enumerator=MemberEnumerator(graph=graph)
        if graph is not None
        else cast(MemberEnumerator, None),
        enrichment=EnrichmentLookup(graph) if graph is not None else cast(EnrichmentLookup, None),
        code_index=code_index if code_index is not None else get_index(),
        rate_cache=rate_cache,
        coverage=coverage,
        version_writer=version_writer if version_writer is not None else NullVersionRecordWriter(),
        head_store=head_store if head_store is not None else NullHeadStore(),
        snapshot_reader=cast(SnapshotReader, snapshot_reader),
        materialise_semaphore=materialise_semaphore,
        head_recheck_interval=timedelta(seconds=head_recheck_seconds),
        head_max_age=timedelta(seconds=head_max_age_seconds),
        now=now,
        operation_store=operation_store if operation_store is not None else NullOperationStore(),
        task_client=task_client if task_client is not None else NullTaskClient(),
        settings=settings,
    )


def _definition_close_live_clients_built_here(
    state: _DefinitionDoorState, *, pool_opened_here: bool
) -> None:
    deps = state.deps
    if not state.injected.graph and deps.graph is not None:
        cast(OriginGraphClient, deps.graph).close()
    if not state.injected.sdmx and deps.sdmx is not None:
        deps.sdmx.close()
    if not state.injected.task_client and isinstance(deps.task_client, LiveTaskClient):
        deps.task_client.close()
    if pool_opened_here:
        close_shared_pool()


def _definition_open_spec_reader_and_snapshot(
    state: _DefinitionDoorState, borrow: _DefinitionStoreBorrow
) -> None:
    """Build the live spec reader and share its store with the snapshot reader when both are
    uninjected.
    """
    if state.injected.spec_reader:
        return
    connections = shared_pool()
    # Mark the borrow before store construction, which can raise.
    borrow.opened_here = True
    borrow.store = LiveDatasetStore(connections=connections)
    state.deps.spec_reader = LiveSpecReader(store=borrow.store)
    if not state.injected.snapshot_reader:
        state.deps.snapshot_reader = LiveSnapshotReader(store=borrow.store)


def _definition_open_graph_sdmx_and_artifact_store(
    state: _DefinitionDoorState, borrow: _DefinitionStoreBorrow, *, generation: GenerationProvider
) -> None:
    """Build missing clients. A live graph client keys its caches on `generation`."""
    if state.injected.graph and state.injected.sdmx and state.injected.artifact_store:
        return
    live_settings = DatasetsSettings.from_env()
    if not state.injected.graph:
        state.deps.graph = build_graph_client(
            base_url=live_settings.graph_base_url,
            token=state.settings.graph_token,
            generation=generation,
            un_base_url=live_settings.un_graph_base_url,
            un_availability=SdmxClient(
                base_url=live_settings.un_graph_base_url,
                timeout=state.settings.sdmx_timeout_seconds,
            ),
        )
        # Build graph-dependent helpers after resolving the live graph.
        state.deps.enumerator = MemberEnumerator(graph=state.deps.graph)
        state.deps.enrichment = EnrichmentLookup(state.deps.graph)
    if not state.injected.sdmx:
        state.deps.sdmx = SdmxClient(
            base_url=live_settings.sdmx_base_url,
            token=state.settings.graph_token,
            timeout=state.settings.sdmx_timeout_seconds,
        )
    if state.injected.artifact_store:
        return
    if borrow.store is None:
        connections = shared_pool()
        borrow.opened_here = True
        borrow.store = LiveDatasetStore(connections=connections)
    inventory = LiveArtifactInventory(store=borrow.store)
    state.deps.artifact_store = LiveArtifactStore(
        buckets={
            Tier.working: live_settings.artifact_bucket_working,
            Tier.pinned: live_settings.artifact_bucket_pinned,
        },
        inventory=inventory,
    )


def _definition_open_task_client(state: _DefinitionDoorState) -> None:
    if state.injected.task_client:
        return
    task_settings = TaskClientSettings.from_env()
    state.deps.task_client = build_live_task_client(
        queue=task_settings.tasks_queue,
        worker_url=task_settings.worker_url,
        worker_invoker_service_account=task_settings.worker_invoker_service_account,
    )


def _definition_wire_shared_store_dependents(
    state: _DefinitionDoorState, store: LiveDatasetStore | None
) -> None:
    if store is None:
        return
    if not state.injected.version_writer:
        state.deps.version_writer = LiveVersionRecordWriter(store=store)
    if not state.injected.head_store:
        state.deps.head_store = LiveHeadStore(store=store)
    if not state.injected.operation_store:
        state.deps.operation_store = LiveOperationStore(store=store, settings=state.settings)


@asynccontextmanager
async def _definition_lifespan(
    _app: FastAPI, *, state: _DefinitionDoorState
) -> AsyncIterator[None]:
    """Create uninjected dependencies, then drain admitted calls before closing live clients.

    Shutdown waits for semaphore permits within the shared budget.
    """
    borrow = _DefinitionStoreBorrow()
    async with AsyncExitStack() as stack:
        generation = UNKNOWN_GENERATION
        try:
            if not state.injected.graph:
                generation = await enter_shared_generation(
                    stack,
                    catalogue=state.catalogue,
                    graph_base_url=DatasetsSettings.from_env().graph_base_url,
                )
            _definition_open_spec_reader_and_snapshot(state, borrow)
            _definition_open_graph_sdmx_and_artifact_store(state, borrow, generation=generation)
            _definition_open_task_client(state)
        except BaseException:
            _definition_close_live_clients_built_here(state, pool_opened_here=borrow.opened_here)
            raise
        _definition_wire_shared_store_dependents(state, borrow.store)
        try:
            yield
        finally:
            state.door_executor.executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(state.door_executor.drain, state.budget)
            _definition_close_live_clients_built_here(state, pool_opened_here=borrow.opened_here)


async def _run_retrieval[T](
    state: _DefinitionDoorState,
    fn: Callable[..., T],
    /,
    *,
    timings: PhaseTimings,
    **kwargs: object,
) -> T:
    """Run retrieval in its dedicated worker. The worker finishes after a client timeout."""
    return await asyncio.wait_for(
        state.door_executor.run(
            functools.partial(run_with_timings, fn, timings, **kwargs),
            refuse=DefinitionBackpressure,
        ),
        timeout=state.settings.executor_wait_seconds(state.settings.definition_timeout_seconds),
    )


async def _get_definition_impl(
    id: str, request: Request, *, state: _DefinitionDoorState
) -> JSONResponse:
    """Build request-scoped graph and deadline state, then dispatch by subject-id prefix."""
    options = parse_query_options(request.query_params.multi_items())
    check_options_supported(options)
    trace_id = get_trace_id()
    request_deps = replace(
        state.deps,
        enumerator=MemberEnumerator(graph=state.deps.graph),
        deadline=_sdmx_deadline(state.settings.definition_timeout_seconds),
    )
    submitted_resource: OperationResource | None = None
    with door_call("definition", logger=logger) as call:
        try:
            if id.startswith("spec_"):
                manifest = await _run_retrieval(
                    state,
                    _serve_spec,
                    timings=call.timings,
                    spec_id=id,
                    options=options,
                    deps=request_deps,
                    trace_id=trace_id,
                )
            elif id.startswith("col_"):
                manifest = await _run_retrieval(
                    state,
                    _serve_collection,
                    timings=call.timings,
                    collection_id=id,
                    options=options,
                    deps=request_deps,
                    trace_id=trace_id,
                )
        except OperationSubmitted as submitted:
            # Treat a 202 submission as a successful `door_call` outcome.
            submitted_resource = operation_submitted_resource(submitted)
    if submitted_resource is not None:
        assert submitted_resource.operation.poll is not None
        return JSONResponse(
            submitted_resource.model_dump(mode="json", exclude_none=True),
            status_code=202,
            headers={
                **no_store_headers(server_timing=call.server_timing),
                "Location": submitted_resource.operation.poll,
            },
        )
    return JSONResponse(
        manifest.model_dump(mode="json", exclude_none=True),
        headers=no_store_headers(server_timing=call.server_timing),
    )


async def _definition_options_impl(
    id: str, request: Request, *, state: _DefinitionDoorState, body: DefinitionOptionsRequest | None
) -> JSONResponse:
    query_items = request.query_params.multi_items()
    if query_items:
        key, _value = query_items[0]
        raise QueryOptionsInvalid(
            pointer=pointer_for_key(key),
            detail=(f"the options route accepts no query parameters; remove {key!r}"),
        )
    trace_id = get_trace_id()
    # Keep the member-group cache request-scoped.
    request_deps = replace(
        state.deps,
        enumerator=MemberEnumerator(graph=state.deps.graph),
        deadline=_sdmx_deadline(state.settings.definition_timeout_seconds),
    )
    with door_call("definition_options", logger=logger) as call:
        # A posted definition is a spec's. Under a `col_` id its hash check refuses.
        if id.startswith("spec_") or body is not None:
            envelope = await _run_retrieval(
                state,
                _serve_spec_options,
                timings=call.timings,
                spec_id=id,
                deps=request_deps,
                trace_id=trace_id,
                body=body,
            )
        elif id.startswith("col_"):
            envelope = await _run_retrieval(
                state,
                _serve_collection_options,
                timings=call.timings,
                collection_id=id,
                deps=request_deps,
                trace_id=trace_id,
            )
    return JSONResponse(
        envelope.model_dump(mode="json", exclude_none=True),
        headers=no_store_headers(server_timing=call.server_timing),
    )


def build_definition_door(
    *,
    settings: DatasetsBuildSettings | None = None,
    spec_reader: SpecReader | None = None,
    graph: GraphReader | None = None,
    sdmx: SdmxClient | None = None,
    revision_reader: RevisionReader | None = None,
    artifact_store: ArtifactStore | None = None,
    materialiser_build: str,
    capability_version: str,
    ceiling: int,
    member_concurrency: int,
    version_writer: VersionRecordWriter | None = None,
    head_store: HeadStore | None = None,
    snapshot_reader: SnapshotReader | None = None,
    materialise_semaphore: threading.BoundedSemaphore | None = None,
    head_recheck_seconds: int | None = None,
    head_max_age_seconds: int | None = None,
    now: Clock = utc_now,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
    code_index: CodeIndex | None = None,
    rate_cache: RateTableCache | None = None,
    coverage: CoverageService | None = None,
    operation_store: OperationStore | None = None,
    task_client: TaskClient | None = None,
    catalogue: CatalogueService | None = None,
) -> DoorContribution:
    """Build the retrieval door. The lifespan creates uninjected live dependencies.

    `settings` supplies concurrency, timeout, head-age, and graph-token defaults. Explicit
    `head_recheck_seconds` and `head_max_age_seconds` override the corresponding settings.
    `materialise_semaphore` can be shared with mint to cap rebuilds across both doors. `now` is an
    injectable currency-check clock. Omitted `budget` and `rate_cache` use door-local defaults.
    A live graph client keys its caches on the shared graph generation, acquired against
    `catalogue`, which defaults to the process-wide service.
    """
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _budget = budget if budget is not None else ShutdownBudget()
    effective_pool_size = (
        pool_size if pool_size is not None else _settings.definition_max_concurrency
    )
    _head_recheck_seconds = (
        head_recheck_seconds if head_recheck_seconds is not None else _settings.head_recheck_seconds
    )
    _head_max_age_seconds = (
        head_max_age_seconds if head_max_age_seconds is not None else _settings.head_max_age_seconds
    )
    door_executor = build_door_executor(effective_pool_size, "definition")
    rate_cache = rate_cache if rate_cache is not None else RateTableCache()
    coverage = coverage if coverage is not None else get_coverage_service()

    deps = _build_definition_deps(
        spec_reader=spec_reader,
        graph=graph,
        sdmx=sdmx,
        revision_reader=revision_reader,
        artifact_store=artifact_store,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        ceiling=ceiling,
        member_concurrency=member_concurrency,
        version_writer=version_writer,
        head_store=head_store,
        snapshot_reader=snapshot_reader,
        materialise_semaphore=materialise_semaphore,
        head_recheck_seconds=_head_recheck_seconds,
        head_max_age_seconds=_head_max_age_seconds,
        now=now,
        code_index=code_index,
        rate_cache=rate_cache,
        coverage=coverage,
        operation_store=operation_store,
        task_client=task_client,
        settings=_settings,
    )
    state = _DefinitionDoorState(
        deps=deps,
        settings=_settings,
        budget=_budget,
        door_executor=door_executor,
        injected=_DefinitionInjected(
            spec_reader=spec_reader is not None,
            graph=graph is not None,
            sdmx=sdmx is not None,
            artifact_store=artifact_store is not None,
            task_client=task_client is not None,
            version_writer=version_writer is not None,
            head_store=head_store is not None,
            operation_store=operation_store is not None,
            snapshot_reader=snapshot_reader is not None,
        ),
        catalogue=catalogue,
    )

    router = APIRouter()

    @router.get("/api/datasets/{id:dfo_subject_id}", response_model=Manifest)
    async def get_definition(id: str, request: Request) -> JSONResponse:
        return await _get_definition_impl(id, request, state=state)

    @router.get("/api/datasets/{id:dfo_subject_id}/options", response_model=OptionsEnvelope)
    async def get_definition_options(id: str, request: Request) -> JSONResponse:
        """Write-free materialisation. Runs the pre-write half of
        `serve_definition`/`serve_collection` alone, reported as
        `output_options`. This route reports what the data honours,
        stopping short of render. Any query key at all refuses
        `request_invalid` with an `/options/<key>` pointer, the same
        `QueryOptionsInvalid` this door's `get_definition` raises for a
        query key outside its own namespace, through the same
        `query_options.pointer_for_key` mapping.
        """
        return await _definition_options_impl(id, request, state=state, body=None)

    @router.post("/api/datasets/{id:dfo_subject_id}/options", response_model=OptionsEnvelope)
    async def post_definition_options(
        id: str, body: DefinitionOptionsRequest, request: Request
    ) -> JSONResponse:
        """The GET route's envelope for a spec id whose canonical `definition` the body
        carries, registered or not. An unregistered definition is admitted as a mint admits a
        sidecar and materialised write-free, so browsing a search result registers nothing.
        Query keys refuse as on the GET route.
        """
        return await _definition_options_impl(id, request, state=state, body=body)

    return DoorContribution(
        router=router,
        classify=classify_definition_exception,
        exception_types=(
            SpecNotFound,
            DefinitionMemberNotResolvable,
            RegistryInconsistent,
            StoredTraceMissingResolvedStatVars,
            DefinitionMaterialiseRefused,
            QueryOptionsInvalid,
            DefinitionBackpressure,
            # Admission of a posted definition raises these. The mint and register classifiers
            # map them.
            MalformedSidecarIdentity,
            MemberSelectionMismatch,
            SpecIdMismatch,
            DefinitionNotReproducible,
            SpecNotExportable,
            RoleSignatureIncompatible,
            OverCeilingError,
            RenderedArtifactSigningFailed,
            # Conversion errors use the mint classifier. List them here because this door's
            # materialisation calls can raise them.
            RecipeNotDeclaredError,
            RateCoverageGapError,
            NonCurrencyUnitError,
            RateTableInvalidError,
            # The surface composes the store classifier. Version and head writes in this door can
            # raise this type.
            DatasetsStoreUnavailable,
            # The currency check loads stored records through the snapshot loader. Resolution
            # confirms registration first, so this door reaches only version-level snapshot errors.
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            SnapshotSigningFailed,
            # The snapshot loader wraps this error as `SnapshotSigningFailed`. Its source type
            # remains listed because the currency-check call can raise it.
            ArtifactSigningError,
        ),
        exception_exemplars=(
            SpecNotFound(id="spec_exemplar"),
            DefinitionMemberNotResolvable(members=("spec_exemplar",)),
            RegistryInconsistent(collection_id="col_exemplar", member_spec_id="spec_exemplar"),
            StoredTraceMissingResolvedStatVars(
                subject_id="spec_exemplar", content_version="cv_exemplar"
            ),
            DefinitionMaterialiseRefused(
                wrapped=FacetNotAvailableError("exemplar", pointer="/facets/exemplar")
            ),
            DefinitionMaterialiseRefused(
                wrapped=ExcludedByRegistry(
                    variables=("Exemplar",), provenances=("dc/base/Exemplar",)
                )
            ),
            QueryOptionsInvalid(pointer="/options/exemplar", detail="exemplar"),
            DefinitionBackpressure(),
            MalformedSidecarIdentity(detail="exemplar", pointer=_DEFINITION_POINTER),
            MemberSelectionMismatch(foreign=((0, "exemplar"),)),
            SpecIdMismatch(supplied="spec_exemplar", computed="spec_exemplar_computed"),
            DefinitionNotReproducible(spec_ids=("spec_exemplar",), detail="exemplar"),
            SpecNotExportable(spec_id="spec_exemplar", detail="exemplar"),
            RoleSignatureIncompatible(
                spec_id="spec_exemplar", roles_by_stat_var={"dcid:exemplar": ("recipient",)}
            ),
            OverCeilingError("exemplar", ceiling_kind="rows", count=1, cap=1),
            RecipeNotDeclaredError("exemplar", cause="recipe_undeclared"),
            RateCoverageGapError(
                "exemplar", cause="currency_unmapped", entities=("CurrencyExemplar",)
            ),
            NonCurrencyUnitError("exemplar", unit="Exemplar"),
            RateTableInvalidError("exemplar"),
            RenderedArtifactSigningFailed(
                bucket="exemplar-bucket",
                object_name="sha256:exemplar",
                digest="sha256:exemplar",
                error_class_name=None,
                status_code=None,
            ),
            DatasetsStoreUnavailable("exemplar"),
            SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotSigningFailed(
                bucket="exemplar-bucket",
                object_name="sha256:exemplar",
                digest="sha256:exemplar",
                error_class_name=None,
                status_code=None,
            ),
            ArtifactSigningError(
                bucket="exemplar-bucket", object_name="sha256:exemplar", digest="sha256:exemplar"
            ),
        ),
        lifespan=functools.partial(_definition_lifespan, state=state),
    )
