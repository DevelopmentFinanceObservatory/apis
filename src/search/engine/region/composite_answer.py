"""The observation preview of a composite answer, read per component as the Datasets mint
reads a composite.

A composite answer is resolved for its table identity and restated as the composite
(`scope.composite.assemble_composite`), so the one table that resolve read previews nothing of
the composite. Its preview reads each component identity
(`code_index.composites.component_identities`) with `resolve_spec_resubmit`, which elects
its members as the mint does and scopes its reads to the provenance it binds, in worker threads
that run together. Each component read embeds its preview
(`include_observations`) and reads no slot or place options, and on a dataset_resubmit it goes
through the part memo under its own key. A component serving no year of the composite is not
read. It reads through the request's catalogue narrowed to
its own pin alone (`ComponentReads.catalogue`), as the mint reads a component
(`datasets_api.serve`), whatever source scope the part's own reads ran under. The previews are
composed under the composite's rules (`scope.composite.compose_observations`).

A streamed request reads the preview in a trailing task (`CompositeRelaunch`),
launched once the answer is restated, so it adds nothing before the `response` event. A
buffered request reads it inline (`with_composite_observations`).

A same-shape merged answer previews the read of its merged identity
(`same_shape._with_merged_preview`). A merge of composite answers is restated as the
composite over that identity (`as_merged_composite`) and reads its components
(`CompositeRelaunch`). Any other merge whose read plan the merge rebuilds in memory reads that
plan (`PlanPreview`), so its preview and its download come from one plan, and the rest take
one resubmit of the merged identity (`IdentityPreview`). A component read and a resubmitted
merged read are both `_resubmit_preview`, so they elect, cap and fail alike.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, get_args

import dc_graph
from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.roles import role_key_axis
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import Catalogue, bound_catalogue, recording_un_blocked_places
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.concurrency import (
    bounded_semaphore,
    bounded_thread_permits,
    create_owned_task,
)
from search.engine.config import settings
from search.engine.degraded_reads import (
    LOST_READ_ERRORS,
    note_degraded_read,
    note_lost_read,
    recording_degraded_reads,
)
from search.engine.departures import FactsByWarning, WarningKey, warning_key
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError
from search.engine.events import ObservationsEvent
from search.engine.identity_edit import composite_table_identity
from search.engine.region.codes import (
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_CLIENTS_UNAVAILABLE,
    OBSERVATIONS_NO_DISPATCHABLE_FACET,
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_NO_POINTS,
    OBSERVATIONS_NOTHING_TO_PROBE,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_SCOPE_PARTIAL,
    OBSERVATIONS_SERIES_TRUNCATED,
    OBSERVATIONS_TIMED_OUT,
    UN_SYSTEM_UNAVAILABLE,
    _ObsProbeResult,
    _OptionProbeResult,
)
from search.engine.region.observations.dispatch import (
    PreviewRead,
    PreviewReader,
    _attach_observations,
    _deflator_projected_warning,
)
from search.engine.region.resubmit import resolve_spec_resubmit
from search.engine.region.result import (
    RegionResult,
    recounted_scope,
    with_composite_answer,
    with_restated_keys,
)
from search.engine.region.scope_decision import (
    ScopeDecision,
    bilateral_withheld_warning,
    decide_guarded,
)
from search.engine.region.state import EngineOptions
from search.engine.region.un_system import un_unavailable_warning
from search.engine.response_cache_keys import is_region_storable
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    Component,
    EntityGroup,
    EntityRoleKey,
    NoDataReason,
    Operation,
    ResubmitIdentityPayload,
    Spec,
    SpecObservations,
    Warning,
    omitted_observations,
)
from search.observations import (
    OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED,
    OBSERVATIONS_PREVIEW_TOO_LARGE,
    OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
)
from search.scope.composite import (
    ServedComponent,
    assemble_composite,
    compose_observations,
    served_components,
)
from search.scope.provider import bilateral_default_reading, elect_latest_window
from search.scope.region import region_groups
from search.scope.state import ScopeState
from search.scope.tables import ScopeFacts
from series_coverage.shared import CoverageSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from code_index.scope_conventions import RolePopulation
    from search.engine.response_cache_flow import PartMemoScope
    from search.heading.departure_fact import DepartureFact

logger = logging.getLogger(__name__)

PREVIEW_WARNING_CODES = frozenset(
    {
        OBSERVATIONS_BULK_FETCH_FAILED,
        OBSERVATIONS_CLIENTS_UNAVAILABLE,
        OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED,
        OBSERVATIONS_NO_DISPATCHABLE_FACET,
        OBSERVATIONS_NO_MATCHING_SOURCE,
        OBSERVATIONS_NO_POINTS,
        OBSERVATIONS_NOTHING_TO_PROBE,
        OBSERVATIONS_PREVIEW_TOO_LARGE,
        OBSERVATIONS_PROBE_FAILED,
        OBSERVATIONS_SCOPE_PARTIAL,
        OBSERVATIONS_SERIES_TRUNCATED,
        OBSERVATIONS_TIMED_OUT,
        OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
        UN_SYSTEM_UNAVAILABLE,
    }
)
"""The codes a component's preview read warns under, which the composite's preview carries.
A component answer's other warnings describe its own resolve, which the composite's answer
states through its table identity's. A converted preview's `deflator_projected` warning is
stated once over the composed preview (`dispatch._deflator_projected_warning`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentReads:
    """The request's clients, coverage snapshot, catalogue and deadline that every resubmit
    preview runs on (`_resubmit_preview`), a composite's component read and a merged
    answer's read alike, and the dataset_resubmit's part memo (None on a fresh search). Each
    door builds them once for its request and passes them to `conjoin.combine_regions` for a
    merge's read. No memoised result holds them.

    `catalogue` is the request's catalogue before any part narrows it to its source scope
    (`origin_catalogue.scoped_reads`). The composite's served years (`composite_served`) and
    its component reads read through it, so a component's own pin
    (`resubmit_request.pinned_provenance_scope`) is the only scope its reads take."""

    graph: EngineGraphClient
    sdmx: EngineSdmxClient
    coverage: CoverageSnapshot
    materialise_graph: GraphReader | None
    materialise_sdmx: MaterialiseSdmxClient | None
    rate_cache: RateTableCache | None
    deadline: float | None
    catalogue: Catalogue
    memo: PartMemoScope | None = None

    @property
    def options(self) -> EngineOptions:
        """The options a component resolves under: its preview embedded, no slot or place
        options."""
        return EngineOptions(
            include_observations=True,
            materialise_graph=self.materialise_graph,
            materialise_sdmx=self.materialise_sdmx,
            rate_cache=self.rate_cache,
            deadline=self.deadline,
        )


def composite_served(identity: BaseIdentity, *, reads: ComponentReads) -> list[ServedComponent]:
    """Each component of composite `identity` with the years it serves
    (`scope.composite.served_components`), read through `reads.catalogue` with facts of its
    own, so no source scope a part's own reads ran under narrows them. The answer the
    composite is restated as (`assemble_composite`) and its preview read these same
    components."""
    facts = ScopeFacts(index=get_index(), coverage=reads.coverage)
    document = identity.model_dump(mode="json", exclude_none=True)
    with bound_catalogue(reads.catalogue):
        return served_components(document, facts=facts)


async def read_composite_preview(
    *,
    served: Sequence[ServedComponent],
    entity_groups: Mapping[str, EntityGroup] | None,
    reads: ComponentReads,
    semaphore: asyncio.Semaphore | None,
) -> tuple[SpecObservations, tuple[Warning, ...]]:
    """The preview of the composite whose components are `served` (`composite_served`),
    composed from one read of each component serving a year, each posting `entity_groups`,
    the groups the composite's answer names, as a resubmit of that answer would, and the
    warnings those reads raised under `PREVIEW_WARNING_CODES`, in component order. A
    component serving no year is not read and raises nothing.

    The reads run together, each holding a permit of `semaphore` when one is given.
    Cancelling the caller cancels every read still running. A component whose read is lost
    (`_resubmit_preview`) leaves its years out (`compose_observations`) and raises a warning
    naming its role (`_read_component`), as a single answer's preview read turns a failed
    read into an omission (`observations.fetch`). A component whose answer stands on less
    than a later request may read (`RegionResult.degraded`) notes a degraded read. Any other
    error of a read propagates and cancels the others."""
    # A Spec keys its groups by the role keys a resubmit posts them under.
    groups = cast("Mapping[EntityRoleKey, EntityGroup] | None", entity_groups)
    read = [index for index, item in enumerate(served) if item.served]
    tasks = [
        asyncio.create_task(
            _read_component(served[index], groups, reads=reads, semaphore=semaphore)
        )
        for index in read
    ]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
    previews: list[SpecObservations | None] = [None] * len(served)
    for index, (observations, _) in zip(read, results, strict=True):
        previews[index] = observations
    composed = compose_observations(served, previews)
    return composed, tuple(warning for _, raised in results for warning in raised)


async def _read_component(
    item: ServedComponent,
    groups: Mapping[EntityRoleKey, EntityGroup] | None,
    *,
    reads: ComponentReads,
    semaphore: asyncio.Semaphore | None,
) -> tuple[SpecObservations | None, tuple[Warning, ...]]:
    """The preview `item`'s component answer embeds and its preview warnings
    (`_resubmit_preview`), or None and a warning naming the component's role when its read
    was lost (`_missing`)."""
    identity = BaseIdentity.model_validate(dict(item.component.identity))
    read = await _resubmit_preview(
        identity,
        groups,
        reads=reads,
        semaphore=semaphore,
        about=f"component {item.component.role}",
        populations=item.component.populations,
    )
    if isinstance(read, _LostRead):
        return None, (_missing(item, read),)
    return read


@dataclass(frozen=True, slots=True)
class _LostRead:
    """A resubmit preview that read nothing, `timed_out` when the read crossed its deadline,
    with `cause` saying why."""

    timed_out: bool
    cause: str

    def warning(self, message: str) -> Warning:
        """The warning stating `message` under `observations_timed_out` for a read that
        crossed its deadline, else `observations_probe_failed`."""
        if self.timed_out:
            return Warning(code=OBSERVATIONS_TIMED_OUT, severity="warn", message=message)
        return Warning(code=OBSERVATIONS_PROBE_FAILED, severity="warn", message=message)

    def omission(self, message: str) -> tuple[SpecObservations, tuple[Warning, ...]]:
        """The preview omitted for the lost read, stating `message`, and its warning."""
        reason = OBSERVATIONS_TIMED_OUT if self.timed_out else OBSERVATIONS_PROBE_FAILED
        return omitted_observations(reason, message), (self.warning(message),)


async def _resubmit_preview(
    identity: BaseIdentity,
    groups: Mapping[EntityRoleKey, EntityGroup] | None,
    *,
    reads: ComponentReads,
    semaphore: asyncio.Semaphore | None,
    about: str,
    populations: Mapping[str, RolePopulation] | None = None,
) -> tuple[SpecObservations, tuple[Warning, ...]] | _LostRead:
    """The preview a resubmit of `identity` posting `groups` embeds, read with
    `resolve_spec_resubmit` in a worker thread through `reads.catalogue` under `reads.options`,
    holding a permit of `semaphore` when one is given, and the answer's warnings under
    `PREVIEW_WARNING_CODES`. A part memo hit reads nothing, and a miss stores the answer.
    Each role `populations` lists, the roles a component binds to a population
    (`ComponentIdentity.populations`), reads as open within it
    (`ResubmitIdentityPayload.populations`).

    A read lost to `degraded_reads.LOST_READ_ERRORS` is noted (`note_lost_read`, so a read cut
    by its enclosing budget is noted as cut and leaves the request unstorable), logged under `about`, and is lost
    `observations_timed_out` when it crossed the deadline (`TimeoutError`,
    `GraphDeadlineExceededError`), else `observations_probe_failed`. A resolve that refuses
    `identity` (`EngineInputError`) or answers no definite Spec with a preview is lost
    `observations_probe_failed`. An answer the response cache would not store
    (`is_region_storable` under the request catalogue's refused places), degraded or carrying
    a warning such as `members_absent_from_graph`, notes a degraded read
    (`note_degraded_read`), so no cache stores what it previews. Any other error propagates."""
    payload = ResubmitIdentityPayload(
        identity=identity,
        entity_groups=None
        if groups is None
        else region_groups(groups, identity, index=get_index()),
        populations=dict(populations or {}),
    )
    options = reads.options
    memo = reads.memo if reads.memo is not None and reads.memo.active else None
    key = memo.key(payload, options) if memo is not None else None
    async with semaphore if semaphore is not None else nullcontext():
        try:
            # The worker thread copies this context, so the resolve reads through the
            # request's catalogue, which only `identity`'s own pin narrows.
            with bound_catalogue(reads.catalogue):
                region, stored = await asyncio.to_thread(
                    _resolve_component, payload, reads=reads, options=options, key=key
                )
        except EngineInputError:
            return _LostRead(timed_out=False, cause="refused its identity")
        except LOST_READ_ERRORS as exc:
            note_lost_read(f"resubmit preview: {about}", exc)
            timed_out = isinstance(exc, TimeoutError | dc_graph.GraphDeadlineExceededError)
            logger.warning(
                "resubmit preview: %s identity=%r %s",
                about,
                identity.model_dump(mode="json", exclude_none=True),
                "crossed the request deadline" if timed_out else "failed",
                exc_info=True,
            )
            if timed_out:
                return _LostRead(timed_out=True, cause="read crossed its deadline")
            return _LostRead(timed_out=False, cause="read failed")
    if memo is not None and key is not None and not stored:
        memo.store(key, region)
    if not is_region_storable(region, refused_at_build=reads.catalogue.un_blocked_places):
        note_degraded_read(f"resubmit preview: {about}")
    if region.status != "definite":
        return _LostRead(timed_out=False, cause=f"answered {region.status}")
    observations = region.spec.resolved.observations
    if observations is None:
        return _LostRead(timed_out=False, cause="embedded no preview")
    raised = tuple(warning for warning in region.warnings if warning.code in PREVIEW_WARNING_CODES)
    return observations, raised


def _missing(item: ServedComponent, lost: _LostRead) -> Warning:
    """The warning that `item`'s component, whose read was `lost`, left its years out of the
    composite's preview, under the lost read's code."""
    return lost.warning(
        f"the composite's {item.component.role} component {lost.cause}; its years are missing "
        "from the preview."
    )


def _resolve_component(
    payload: ResubmitIdentityPayload,
    *,
    reads: ComponentReads,
    options: EngineOptions,
    key: str | None,
) -> tuple[RegionResult, bool]:
    """The part memo's region for `payload` under `key`, else a fresh resolve of it, and
    whether the memo held it."""
    stored = reads.memo.lookup(key) if reads.memo is not None and key is not None else None
    if stored is not None:
        return stored, True
    region = resolve_spec_resubmit(
        inp=payload,
        graph=reads.graph,
        sdmx=reads.sdmx,
        coverage=reads.coverage,
        options=options,
    )
    return region, False


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeRelaunch:
    """The relaunch closure of composite `spec`'s preview (`RegionResult.observations_relaunch`):
    called with a spec_id, it launches `read_composite_preview` as a request-owned task on the
    event loop, under the request semaphore. The task's result is one `ObservationsEvent`
    under that spec_id and the component reads' warnings, with `deflator_projected` for the
    composed preview, each keyed to it.

    It reads `spec`'s identity and entity groups alone, so an answer the part memo held
    builds it as a fresh resolve does. `served` holds the components the answer was restated
    over (`composite_served`), so the preview filters by the years the answer states. With
    `served` None the preview reads `spec`'s components itself."""

    spec: Spec
    reads: ComponentReads
    served: Sequence[ServedComponent] | None = None

    def __call__(self, spec_id: str) -> asyncio.Task[_ObsProbeResult]:
        return create_owned_task(_preview_events(self, spec_id))

    async def read(
        self, *, semaphore: asyncio.Semaphore | None
    ) -> tuple[SpecObservations, tuple[Warning, ...]]:
        """The composed preview and its warnings (`read_composite_preview`)."""
        served = self.served
        if served is None:
            served = await asyncio.to_thread(
                composite_served, self.spec.definition.identity, reads=self.reads
            )
        return await read_composite_preview(
            served=served,
            entity_groups=self.spec.resolved.entity_groups,
            reads=self.reads,
            semaphore=semaphore,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityPreview:
    """The relaunch closure of single-table `spec`'s preview (`RegionResult.observations_relaunch`),
    read as a resubmit of its identity and entity groups (`_resubmit_preview`). Called with a
    spec_id, it launches the read as a request-owned task on the event loop, under the
    request semaphore. The task's result is one `ObservationsEvent` under that spec_id and
    the read's warnings, with `deflator_projected` for the preview, each keyed to it, as
    `CompositeRelaunch` returns. A lost read previews its omission under the lost read's code
    and warns under it."""

    spec: Spec
    reads: ComponentReads

    def __call__(self, spec_id: str) -> asyncio.Task[_ObsProbeResult]:
        return create_owned_task(_preview_events(self, spec_id))

    async def read(
        self, *, semaphore: asyncio.Semaphore | None
    ) -> tuple[SpecObservations, tuple[Warning, ...]]:
        """The preview a resubmit of the Spec's identity embeds and its warnings."""
        groups = cast(
            "Mapping[EntityRoleKey, EntityGroup] | None", self.spec.resolved.entity_groups
        )
        read = await _resubmit_preview(
            self.spec.definition.identity,
            groups,
            reads=self.reads,
            semaphore=semaphore,
            about="merged answer",
        )
        if not isinstance(read, _LostRead):
            return read
        return read.omission(f"the merged answer's preview {read.cause}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanPreview:
    """The relaunch closure of a single-table merged answer's preview
    (`RegionResult.observations_relaunch`): `preview`, what a read of the merged answer's own
    read plan reads (`same_shape._merged_preview_read`), read as one answer's preview is
    (`dispatch.PreviewReader`) over `reads`, the request's clients, with no resolve. Called
    with a spec_id, it launches the read as a request-owned task on the event loop, under the
    request semaphore, with the result `IdentityPreview` returns. The read's warnings are its
    own (`fetch._build_observations`) and `un_system_unavailable` for the UN places it was
    refused, naming the variables from `variable_labels`. A read lost to
    `degraded_reads.LOST_READ_ERRORS` is noted and previews its omission under the lost read's
    code, as `IdentityPreview`'s does."""

    preview: PreviewRead
    reads: ComponentReads
    variable_labels: Mapping[str, str]

    def __call__(self, spec_id: str) -> asyncio.Task[_ObsProbeResult]:
        return create_owned_task(_preview_events(self, spec_id))

    async def read(
        self, *, semaphore: asyncio.Semaphore | None
    ) -> tuple[SpecObservations, tuple[Warning, ...]]:
        """The preview the plan reads and its warnings, read in a worker thread holding a
        permit of `semaphore` when one is given."""
        reads = self.reads
        reader = PreviewReader(
            read=self.preview,
            graph=reads.graph,
            sdmx=reads.sdmx,
            materialise_graph=reads.materialise_graph,
            materialise_sdmx=reads.materialise_sdmx,
            rate_cache=reads.rate_cache,
            deadline=reads.deadline,
            coverage=reads.coverage,
        )
        async with semaphore if semaphore is not None else nullcontext():
            try:
                # The worker thread copies this context, so the read goes through the
                # request's catalogue and records the UN places it is refused.
                with (
                    bound_catalogue(reads.catalogue),
                    recording_un_blocked_places() as un_blocked,
                ):
                    observations, warning = await asyncio.to_thread(reader.observations)
            except LOST_READ_ERRORS as exc:
                note_lost_read("merged preview", exc)
                timed_out = isinstance(exc, TimeoutError | dc_graph.GraphDeadlineExceededError)
                logger.warning(
                    "merged preview: plan read %s",
                    "crossed the request deadline" if timed_out else "failed",
                    exc_info=True,
                )
                lost = _LostRead(
                    timed_out=timed_out,
                    cause="read crossed its deadline" if timed_out else "read failed",
                )
                return lost.omission(f"the merged answer's preview {lost.cause}.")
        refused = un_unavailable_warning(un_blocked, labels=self.variable_labels)
        return observations, tuple(w for w in (warning, refused) if w is not None)


def cut_merged_preview() -> tuple[SpecObservations, tuple[Warning, ...]]:
    """The preview of a merged answer whose read was cut at its deadline
    (`same_shape._read_merged_previews`): omitted `observations_timed_out`, with its warning."""
    return _LostRead(timed_out=True, cause="ran past its deadline").omission(
        "the merged answer's preview ran past its deadline."
    )


async def _preview_events(
    relaunch: CompositeRelaunch | IdentityPreview | PlanPreview, spec_id: str
) -> _ObsProbeResult:
    """The trailing preview `relaunch` reads, as one event under `spec_id` with its warnings
    keyed to it and `deflator_projected` for the preview. The event states whether the read
    swallowed a read (`ObservationsEvent.degraded`), which the task's own record collects,
    since the request's resolutions have ended by the time it runs."""
    with recording_degraded_reads() as lost:
        observations, raised = await relaunch.read(semaphore=bounded_semaphore())
    warnings = _keyed(raised, spec_id)
    projected = _deflator_projected_warning(spec_id, observations)
    if projected is not None:
        warnings.append(projected)
    event = ObservationsEvent(spec_id=spec_id, observations=observations)
    return (event.stating_degraded(lost.degraded),), tuple(warnings)


def population_table_answer(
    spec: Spec, identity: BaseIdentity, *, reads: ComponentReads
) -> RegionResult | None:
    """The answer a resubmit of composite `identity` resolves for its table identity, where
    that read binds a population (`identity_edit.composite_table_identity`,
    `TableRead.populations`) or the population's total in place of a destination `identity`
    leaves out (`RolePopulation.total`): `spec`, an answer resolved over the whole table,
    read again with each population role open within its members (`resolve_spec_resubmit`),
    posting `spec`'s entity groups, through `reads.catalogue`. A fresh answer discovers its
    open roles before its scope settles on the composite, and a merge rebuilds its Spec from
    its members', so its members, scope, headline, coverage and warnings count every party
    the table holds; this read counts the composite's population alone, as the resubmit door
    does (`dataset_resubmit`). It answers whatever the resubmit answers, no_data included,
    where the population holds no row in the window.

    Where the composite refuses `identity`, as it does a population that is unavailable,
    the read answers no_data, `facet_not_available`, and logs the refusal at ERROR: a fresh
    search does not refuse its scope, and the whole table's read would state every party
    under the composite's spec_id. A read that fails answers no_data too
    (`_failed_population_read`), so no error of a read the request did not post reaches
    its caller. None where the table read binds neither, so the caller reads `spec` as the
    one table it resolved."""
    try:
        table = composite_table_identity(
            identity, pointer="/input/identity", coverage=reads.coverage
        )
    except EngineInputError as exc:
        logger.error(
            "population read of %s as %s: the composite refuses it (%s): %s",
            spec.spec_id,
            identity_spec_id(identity),
            exc.code,
            exc,
        )
        return _unread_population("facet_not_available")
    if not table.populations and not _binds_left_out_role(identity, table.identity):
        return None
    groups = cast("Mapping[EntityRoleKey, EntityGroup] | None", spec.resolved.entity_groups)
    payload = ResubmitIdentityPayload(
        identity=table.identity,
        entity_groups=None
        if groups is None
        else region_groups(groups, table.identity, index=get_index()),
        populations=dict(table.populations),
    )
    try:
        with bound_catalogue(reads.catalogue):
            return resolve_spec_resubmit(
                inp=payload,
                graph=reads.graph,
                sdmx=reads.sdmx,
                coverage=reads.coverage,
                options=EngineOptions(
                    materialise_graph=reads.materialise_graph,
                    materialise_sdmx=reads.materialise_sdmx,
                    rate_cache=reads.rate_cache,
                    deadline=reads.deadline,
                ),
            )
    except (EngineInputError, *LOST_READ_ERRORS) as exc:
        return _failed_population_read(spec, identity, exc)


def _failed_population_read(spec: Spec, identity: BaseIdentity, exc: BaseException) -> RegionResult:
    """The no_data answer of `spec`'s population read as composite `identity` where the read
    raised `exc`. A refusal of the posted read (`EngineInputError`) answers its own code where
    that code is a no_data reason, else `facet_not_available`: the request never posted the
    identity the refusal points into. A lost read (`degraded_reads.LOST_READ_ERRORS`) is noted
    (`note_lost_read`) and answers degraded, `timed_out` where it crossed its deadline
    (`TimeoutError`, `GraphDeadlineExceededError`), else `no_observations`."""
    about = f"population read of {spec.spec_id} as {identity_spec_id(identity)}"
    if isinstance(exc, EngineInputError):
        logger.warning("%s refused (%s): %s", about, exc.code, exc)
        reason = exc.code if exc.code in _NO_DATA_REASONS else "facet_not_available"
        return _unread_population(cast("NoDataReason", reason))
    note_lost_read(about, exc)
    timed_out = isinstance(exc, TimeoutError | dc_graph.GraphDeadlineExceededError)
    logger.warning("%s lost: %s: %s", about, type(exc).__name__, exc, exc_info=not timed_out)
    return _unread_population("timed_out" if timed_out else "no_observations", degraded=True)


_NO_DATA_REASONS: frozenset[str] = frozenset(get_args(NoDataReason))


def _unread_population(reason: NoDataReason, *, degraded: bool = False) -> RegionResult:
    """A population read answering no_data for `reason`, `degraded` where it lost a read."""
    return RegionResult(
        variable_text=None,
        status="no_data",
        specs=(),
        no_data_reason=reason,
        warnings=(),
        timing_by_step={},
        degraded=degraded,
    )


def _binds_left_out_role(identity: BaseIdentity, read: BaseIdentity) -> bool:
    """Whether `read`, the table identity of composite `identity`, binds a flow axis that
    `identity` neither names nor opens: the population's total a component binds in place
    of a destination the identity leaves out (`RolePopulation.total`)."""
    asked = {role_key_axis(key) for key in identity.entities} | set(identity.open_roles or ())
    return any(role_key_axis(key) not in asked for key in read.entities)


def with_population_read(
    fresh: RegionResult, answered: RegionResult, *, request_warnings: Sequence[Warning]
) -> RegionResult:
    """`answered`, the population read of `fresh`'s definite answer
    (`population_table_answer`), definite or not, carrying what `fresh` holds from the
    request rather than from the rows it read, so every figure, member, plan and warning the
    read derives is `answered`'s, and a field this leaves out costs wording, not figures:

    - `variable_text`, `earliest_index` and `llm_usage`: the request's words for the part,
      its position among the request's parts, and the model calls its words took;
    - `timing_by_step`: the steps the request's pass ran, the population read included;
    - `warnings`: `request_warnings`, those the pass raised before it interpreted its read
      (`ResolveState.request_warnings`), then `answered`'s, every one about the answer keyed
      to `answered`'s Spec where `fresh`'s were keyed to its own (`_grafted_warnings`);
    - `departure_facts`: those of `request_warnings`, then `answered`'s;
    - `reads_deadline` and `promote_relaunch`: the request door's own deadline, and the
      relaunch facts only a resubmit door's part memo binds, which a fresh part has none of;
    - `degraded`: also set where the pass swallowed a read before the population read.

    Every field the request's pass sets once its phases return (its words, entity words,
    judged requirements, scope) is set on this result by that pass. `fresh`'s trailing tasks
    read the whole table, so the caller cancels them (`result.cancel_dropped_tasks`)."""
    kept = {warning_key(warning) for warning in request_warnings}
    reading = fresh.spec.spec_id
    warnings = _grafted_warnings(
        fresh.warnings,
        answered,
        reading=reading,
        keyed_to=answered.spec.spec_id if answered.status == "definite" else reading,
        request_warnings=request_warnings,
        own=True,
    )
    return dataclasses.replace(
        answered,
        variable_text=fresh.variable_text,
        earliest_index=fresh.earliest_index,
        llm_usage=fresh.llm_usage,
        timing_by_step=fresh.timing_by_step,
        warnings=warnings,
        departure_facts={
            **{key: fact for key, fact in fresh.departure_facts.items() if key in kept},
            **answered.departure_facts,
        },
        reads_deadline=fresh.reads_deadline,
        promote_relaunch=fresh.promote_relaunch,
        degraded=fresh.degraded or answered.degraded,
    )


def _grafted_warnings(
    held: Sequence[Warning],
    answered: RegionResult,
    *,
    reading: str,
    keyed_to: str,
    request_warnings: Sequence[Warning],
    own: bool,
) -> tuple[Warning, ...]:
    """`held`, a fresh region's warnings, with those about the reading under spec_id
    `reading` replaced by `answered`'s, its read within its population, and every warning
    about the reading keyed to `keyed_to`, so none names a Spec the response does not hold.

    A warning of `held` is about the reading when keyed to `reading`, or keyed to none where
    the reading is the region's `own`, its definite answer or first candidate. Of those, the
    ones in `request_warnings`, raised before the pass interpreted its read
    (`ResolveState.request_warnings`), stand; the rest came from the whole table's rows. A
    warning of `answered` is about the reading when keyed to the Spec it answers, whose
    spec_id is its own table identity's and may differ from `reading`, or to `reading`, or to
    none. A warning keyed to none on the region's own reading stays keyed to none."""
    request = set(map(warning_key, request_warnings))
    read = answered.spec.spec_id if answered.status == "definite" else None

    def about(warning: Warning) -> bool:
        return warning.spec_id == reading or (own and warning.spec_id is None)

    def keyed(warning: Warning) -> Warning:
        if warning.spec_id is None and own:
            return warning
        if warning.spec_id not in (None, reading, read) or warning.spec_id == keyed_to:
            return warning
        return warning.model_copy(update={"spec_id": keyed_to})

    stood = (
        keyed(warning) if about(warning) else warning
        for warning in held
        if not about(warning) or warning_key(warning) in request
    )
    return (*stood, *map(keyed, answered.warnings))


@dataclass(frozen=True, slots=True, kw_only=True)
class RestatedReading:
    """A reading restated as the composite it is served as (`restated_reading`)."""

    composite: Spec
    # The composite's components (`CompiledScope.components`).
    components: tuple[Component, ...]
    # The closure that reads the composite's preview per component.
    relaunch: CompositeRelaunch


def restated_reading(
    table: Spec,
    identity: BaseIdentity,
    spec_id: str,
    *,
    facts: ScopeFacts,
    reads: ComponentReads,
) -> RestatedReading:
    """`table`, a reading resolved for composite `identity`'s table identity, restated as the
    composite under `spec_id` (`scope.composite.assemble_composite`), over the components
    `identity` serves (`composite_served`), read through `reads.catalogue` with the part's
    `facts`, with the closure that reads its preview per component (`CompositeRelaunch`).

    Every reading a part shows is restated here: its definite answer (`with_decided_scope`),
    each reading a candidate set or refusal shows (`with_bilateral_readings`), a set left
    with one (`_collapsed`), a same-shape merge (`as_merged_composite`) and an identity
    entry (`as_composite_answer`). A reading whose composite binds a population is read
    within it first (`population_table_answer`), so `table` counts the population alone."""
    served = composite_served(identity, reads=reads)
    with bound_catalogue(reads.catalogue):
        composite, components = assemble_composite(
            table, identity, spec_id, served=served, facts=facts
        )
    return RestatedReading(
        composite=composite,
        components=tuple(components),
        relaunch=CompositeRelaunch(spec=composite, reads=reads, served=served),
    )


def with_decided_scope(
    result: RegionResult, decision: ScopeDecision, *, reads: ComponentReads
) -> RegionResult:
    """`result`, a definite answer, stating the scope `decision` decided for it
    (`RegionResult.scope_choices`) and carrying the decision's warning once
    (`ScopeDecision.warning`). Where the decision compiles a composite, the answer is
    restated as it (`restated_reading`, `result.with_composite_answer`) over `reads`, the
    compiled scope listing the composite's components. A fresh definite answer
    (`pipeline._with_scope_choices`) and a candidate set left with one reading (`_collapsed`)
    both state their scope here."""
    if decision.warning is not None and warning_key(decision.warning) not in set(
        map(warning_key, result.warnings)
    ):
        result = dataclasses.replace(result, warnings=(*result.warnings, decision.warning))
    built = decision.scope
    if built is None:
        return result
    if not decision.composite:
        return dataclasses.replace(result, scope_choices=built)
    restated = restated_reading(
        result.spec,
        built.compiled.identity,
        built.compiled.spec_id,
        facts=decision.facts,
        reads=reads,
    )
    compiled = built.compiled.model_copy(update={"components": list(restated.components)})
    return dataclasses.replace(
        with_composite_answer(result, restated.composite, observations_relaunch=restated.relaunch),
        scope_choices=built.model_copy(update={"compiled": compiled}),
    )


def as_composite_answer(
    region: RegionResult,
    identity: BaseIdentity,
    spec_id: str,
    *,
    facts: ScopeFacts,
    reads: ComponentReads,
) -> RegionResult:
    """`region`, a definite answer resolved for composite `identity`'s table identity,
    answering the composite under `spec_id` (`restated_reading`)."""
    restated = restated_reading(region.spec, identity, spec_id, facts=facts, reads=reads)
    return with_composite_answer(
        region, restated.composite, observations_relaunch=restated.relaunch
    )


def with_bilateral_readings(
    region: RegionResult,
    *,
    reads: ComponentReads,
    selection: ScopeState,
    facts: ScopeFacts,
    operations: Sequence[Operation] = (),
    request_warnings: Sequence[Warning] = (),
) -> RegionResult:
    """`region`, a fresh search's candidate set or refusal, with each reading it shows that
    is not its definite answer read as a fresh definite answer is: a candidate, the alternate
    a refusal offers in its place, each flagged match and each nearest real suggestion,
    restated as the composite of DAC2a's bilateral destinations where the default settles it
    there under the part's provider entries `selection` holds
    (`scope.provider.bilateral_default_reading`), so a stated or carried
    `provider:contributions` keeps total ODA, over the one year it serves for a `latest`
    window (`scope.provider.elect_latest_window`). A client can fetch any of them by its
    spec_id, so none serves the agencies the definite answer leaves out. `facts` are the
    part's (`ScopeFacts`), which every reading's rules and restatement share. Every reading
    is restated by `restated_reading`, through `reads.catalogue`, the request's catalogue
    before any part's source scope narrowed it, as the definite answer is.
    A reading whose composite binds a population is read again within it first
    (`population_table_answer`), once per spec_id, the readings concurrently
    (`_population_reads`), so its scope, headline, coverage and warnings count the
    population alone: of the warnings about it, those in `request_warnings`
    (`ResolveState.request_warnings`) stand and the read's replace the rest, keyed to the
    reading (`_grafted_warnings`), and its `role_scope_defaulted` warning counts the
    composite's parties (`result.recounted_scope`). A reading the default reaches that reads
    total ODA because the default could not apply, on a part with no provider axis, carries
    the `bilateral_reading_unavailable` warning a definite answer would
    (`scope_decision.bilateral_withheld_warning`). A reading whose population holds no row in
    the window, or whose composite refuses it, is dropped, never restated over the whole
    table. A set left with no candidate refuses with the reason its first such read gives,
    and a set left with one answers as a definite answer of that reading does (`_collapsed`).

    Each restated reading takes the composite's spec_id, and the warnings, unpublished-whole
    facts and assumption echoes keyed to the table's spec_id are keyed to it
    (`result.with_restated_keys`). A candidate set keeps its trailing slot options task, whose
    events for a restated reading the caller withholds on the event loop
    (`without_restated_options`). Where the region's own reading (`RegionResult.spec`) is
    restated, the closures that would launch its slot options, place options and preview
    again read the table under the composite's spec_id, so the region carries none of them,
    nor the facts a part memo hit binds them from (`promote_relaunch`), as a composite answer
    carries none (`result.with_composite_answer`). `region`, with any such warning, where no
    reading is restated."""
    identities: dict[str, BaseIdentity | None] = {}
    renamed: dict[str, Spec] = {}
    tables: dict[str, Spec] = {}

    def served_as(spec: Spec) -> BaseIdentity | None:
        """The composite identity `spec` is served as, None where it is served as itself."""
        if spec.spec_id not in identities:
            identity, _ = bilateral_default_reading(spec.definition.identity, selection, facts)
            identities[spec.spec_id] = (
                None
                if identity is spec.definition.identity
                else elect_latest_window(identity, facts)
            )
        return identities[spec.spec_id]

    shown: list[Spec] = [
        *region.specs,
        *([region.alternate_match] if region.alternate_match is not None else ()),
        *(region.flagged_match or ()),
        *(region.nearest_real or ()),
    ]
    with bound_catalogue(reads.catalogue):
        pending = {
            spec.spec_id: (spec, identity)
            for spec in shown
            if (identity := served_as(spec)) is not None
        }
        withheld = {
            spec.spec_id: warning
            for spec in shown
            if served_as(spec) is None
            and (
                warning := bilateral_withheld_warning(
                    spec.spec_id, spec.definition.identity, selection=selection, facts=facts
                )
            )
            is not None
        }
    if withheld:
        region = dataclasses.replace(region, warnings=(*region.warnings, *withheld.values()))
    answered = _population_reads(pending, reads=reads)

    def restated[S: Spec](spec: S) -> S | None:
        identity = served_as(spec)
        if identity is None:
            return spec
        reading = spec.spec_id
        read = answered[reading]
        if read is not None:
            if read.status != "definite":
                return None
            # The reading keeps its own class (`NearestRealSpec`) and the facts it carries.
            spec = spec.model_copy(
                update={"definition": read.spec.definition, "resolved": read.spec.resolved}
            )
        composite = restated_reading(
            spec, identity, identity_spec_id(identity), facts=facts, reads=reads
        ).composite
        renamed[reading] = composite
        tables[reading] = spec
        return cast("S", composite)

    def each[S: Spec](specs: Sequence[S]) -> tuple[S, ...]:
        return tuple(kept for spec in specs if (kept := restated(spec)) is not None)

    if region.status == "candidates" and len(region.specs) > 1:
        survivors = [
            spec
            for spec in region.specs
            if (read := answered.get(spec.spec_id)) is None or read.status == "definite"
        ]
        if len(survivors) == 1:
            (survivor,) = survivors
            return _collapsed(
                region,
                survivor,
                served_as(survivor),
                answered.get(survivor.spec_id),
                reads=reads,
                selection=selection,
                operations=operations,
                request_warnings=request_warnings,
            )
    specs = each(region.specs)
    alternate = region.alternate_match
    if alternate is not None:
        alternate = restated(alternate)
    flagged = region.flagged_match
    if flagged is not None:
        flagged = each(flagged) or None
    nearest = region.nearest_real
    if nearest is not None:
        nearest = each(nearest) or None
    unread = [read for read in answered.values() if read is not None and read.status != "definite"]
    if not renamed and not unread:
        return region
    dropped = frozenset(reading for reading, read in answered.items() if read in unread)
    warnings, departure_facts = _without_dropped(region.warnings, region.departure_facts, dropped)
    for reading, read in answered.items():
        if read is not None and reading not in dropped:
            warnings = _grafted_warnings(
                warnings,
                read,
                reading=reading,
                keyed_to=reading,
                request_warnings=request_warnings,
                own=bool(region.specs) and region.spec.spec_id == reading,
            )
            departure_facts.update(read.departure_facts)
    for reading, composite in renamed.items():
        own = bool(region.specs) and region.spec.spec_id == reading
        warnings = recounted_scope(warnings, tables[reading], composite, own=own)
    read_again = dataclasses.replace(region, warnings=warnings, departure_facts=departure_facts)
    restated_region = dataclasses.replace(
        with_restated_keys(read_again, renamed),
        specs=specs,
        alternate_match=alternate,
        flagged_match=flagged,
        nearest_real=nearest,
    )
    if region.specs and not specs:
        # Every reading the set offered holds no row within its population.
        restated_region = dataclasses.replace(
            restated_region, status="no_data", no_data_reason=unread[0].no_data_reason
        )
    if not region.specs or region.spec.spec_id not in renamed.keys() | answered.keys():
        return restated_region
    return dataclasses.replace(
        restated_region,
        slot_options_relaunch=None,
        place_options_relaunch=None,
        observations_relaunch=None,
        promote_relaunch=None,
    )


def _without_dropped(
    warnings: Sequence[Warning],
    departure_facts: FactsByWarning,
    dropped: frozenset[str],
) -> tuple[tuple[Warning, ...], dict[WarningKey, DepartureFact]]:
    """`warnings` without those keyed to a reading in `dropped`, and `departure_facts`
    without the facts only those warnings disclosed (`departures.warning_key`)."""
    kept = tuple(warning for warning in warnings if warning.spec_id not in dropped)
    held = set(map(warning_key, kept))
    gone = {warning_key(w) for w in warnings if w.spec_id in dropped} - held
    return kept, {key: fact for key, fact in departure_facts.items() if key not in gone}


def _population_reads(
    pending: Mapping[str, tuple[Spec, BaseIdentity]], *, reads: ComponentReads
) -> dict[str, RegionResult | None]:
    """The population read of each reading `pending` maps by spec_id to its table answer and
    the composite identity it is served as (`population_table_answer`). Two or more run
    concurrently, each in a worker thread over a copy of the caller's context, and every
    part's reads together take at most the request's concurrency limit
    (`SearchSettings.max_variable_concurrency`) at once (`concurrency.bounded_thread_permits`).
    They take no permit of the request semaphore, which the part's own leg holds one of while
    it resolves (`fanout`), so a request whose legs held every permit would wait on itself."""
    if len(pending) < 2:
        return {
            reading: population_table_answer(spec, identity, reads=reads)
            for reading, (spec, identity) in pending.items()
        }
    permits = bounded_thread_permits()

    def read(
        context: contextvars.Context, spec: Spec, identity: BaseIdentity
    ) -> RegionResult | None:
        with permits:
            return context.run(lambda: population_table_answer(spec, identity, reads=reads))

    workers = min(len(pending), settings().max_variable_concurrency)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            reading: pool.submit(read, contextvars.copy_context(), spec, identity)
            for reading, (spec, identity) in pending.items()
        }
        return {reading: future.result() for reading, future in futures.items()}


def _collapsed(
    region: RegionResult,
    survivor: Spec,
    served: BaseIdentity | None,
    read: RegionResult | None,
    *,
    reads: ComponentReads,
    selection: ScopeState,
    operations: Sequence[Operation],
    request_warnings: Sequence[Warning],
) -> RegionResult:
    """`region`, a candidate set left with one reading, `survivor`, once the others' reads
    held no row within their populations (`with_bilateral_readings`), answering as a fresh
    definite answer of `survivor` does: its scope decided on its own identity under the
    part's `selection` and `operations` (`scope_decision.decide_guarded`), and, where that
    decision compiles a composite, read within the composite's population
    (`with_population_read`) and restated as it (`with_decided_scope`). `read` is the
    population read of the composite `served` the survivor was shown as, which serves where
    the decision compiles that composite; any other composite is read again. A decision
    compiling no composite states its scope on `survivor` itself. The warnings keyed to the
    dropped readings go with them."""
    warnings, departure_facts = _without_dropped(
        region.warnings,
        region.departure_facts,
        frozenset(spec.spec_id for spec in region.specs) - {survivor.spec_id},
    )
    definite = dataclasses.replace(
        region,
        status="definite",
        specs=(survivor,),
        alternate_match=None,
        flagged_match=None,
        nearest_real=None,
        warnings=warnings,
        departure_facts=departure_facts,
    )
    decision = decide_guarded(
        survivor, selection=selection, operations=operations, coverage=reads.coverage
    )
    built = decision.scope
    if decision.composite and built is not None:
        identity = built.compiled.identity
        if served != identity:
            read = population_table_answer(survivor, identity, reads=reads)
        if read is not None:
            definite = with_population_read(definite, read, request_warnings=request_warnings)
            if definite.status != "definite":
                return definite
            decision = dataclasses.replace(decision, spec_id=definite.spec.spec_id)
    return with_decided_scope(definite, decision, reads=reads)


def without_restated_options(before: RegionResult, after: RegionResult) -> RegionResult:
    """`after`, the restatement of `before` (`with_bilateral_readings`), with the trailing
    slot options task `before` launched read without the events and warnings of each reading
    it restated: they read the table the reading resolved, under a spec_id the set does not
    show, and a composite reading carries none, as a composite answer carries none
    (`result.with_composite_answer`). The other readings keep theirs. The task is wrapped on
    the event loop, so the caller runs this there. `after` itself where it restated no
    reading or holds another task."""
    task = after.slot_options_task
    if task is None or task is not before.slot_options_task:
        return after
    withheld = frozenset(spec.spec_id for spec in before.specs) - {
        spec.spec_id for spec in after.specs
    }
    if not withheld:
        return after
    return dataclasses.replace(
        after, slot_options_task=create_owned_task(_without_readings(task, withheld))
    )


async def _without_readings(
    task: asyncio.Task[_OptionProbeResult], withheld: frozenset[str]
) -> _OptionProbeResult:
    """The events and warnings `task` yields, without those of the spec_ids `withheld`."""
    events, warnings = await task
    return (
        tuple(event for event in events if event.spec_id not in withheld),
        tuple(warning for warning in warnings if warning.spec_id not in withheld),
    )


def as_merged_composite(
    region: RegionResult, group: Sequence[RegionResult], *, reads: ComponentReads
) -> RegionResult:
    """`region`, the same-shape merge of `group`, composite answers each, restated as the
    composite over the merged identity, as a resubmit of that composite answers it
    (`dataset_resubmit._as_composite`).

    The merge rebuilds the merged Spec from its members' slots, which bind no composite, so
    its identity is the table identity over every member's entities. The composite's
    identity binds there the members' facets, the composite provenance among them, which the
    merge gate holds every member to. Its served components (`composite_served`), years,
    counts, headline and download facts (`assemble_composite`) and its preview
    (`CompositeRelaunch`) read that identity over `reads`, the request's component reads,
    to the later of their deadline and the latest any member's reads ran to
    (`RegionResult.reads_deadline`). The headline names the request's window as the members'
    do, since the gate holds every member to that window. The warnings the merge keyed to
    the table identity's spec_id are keyed to the composite's; the caller has dropped the
    members' own preview warnings (`same_shape._with_merged_preview`). The observation task the merge carried over from its first member is left to
    the combiner's reconciliation (`enrichment._reconcile_after_collapse`), which cancels
    the members' tasks and launches the composite's where the request streams
    observations. A composite binding a population restates the merged table identity read
    within it (`population_table_answer`), as a resubmit of the merged composite reads it,
    since the merged Spec is rebuilt from its members' slots and plans. Where that read
    holds no row or refuses, the merge answers its no_data, as a reading the read drops
    does (`with_bilateral_readings`), without the warnings keyed to the merged Spec."""
    member = group[0].spec
    deadlines = [
        deadline
        for deadline in (reads.deadline, *(r.reads_deadline for r in group))
        if deadline is not None
    ]
    if deadlines:
        reads = dataclasses.replace(reads, deadline=max(deadlines))
    merged = region.spec
    identity = merged.definition.identity.model_copy(
        update={"facets": member.definition.identity.facets}
    )
    spec_id = identity_spec_id(identity)
    answered = population_table_answer(merged, identity, reads=reads)
    if answered is not None and answered.status != "definite":
        # Each member was read within the population, so a merged read holding no row there
        # contradicts them; the merge answers it, never the whole table restated.
        logger.warning(
            "merged population read of %s answered %s (%s)",
            merged.spec_id,
            answered.status,
            answered.no_data_reason,
        )
        warnings, departure_facts = _without_dropped(
            region.warnings, region.departure_facts, frozenset({merged.spec_id})
        )
        return dataclasses.replace(
            region,
            status="no_data",
            specs=(),
            no_data_reason=answered.no_data_reason,
            warnings=warnings,
            departure_facts=departure_facts,
            degraded=region.degraded or answered.degraded,
            slot_options_task=None,
            slot_options_relaunch=None,
            place_options_task=None,
            place_options_relaunch=None,
            observations_task=None,
            observations_relaunch=None,
        )
    if answered is not None:
        table = answered.spec
    else:
        table = merged.model_copy(
            update={
                "resolved": merged.resolved.model_copy(
                    update={"headline": member.resolved.headline}
                )
            }
        )
    restated = restated_reading(
        table,
        identity,
        spec_id,
        facts=ScopeFacts(index=get_index(), coverage=reads.coverage),
        reads=reads,
    )
    warnings = tuple(
        warning.model_copy(update={"spec_id": spec_id})
        if warning.spec_id == merged.spec_id
        else warning
        for warning in region.warnings
    )
    return with_composite_answer(
        dataclasses.replace(region, warnings=warnings, observations_task=None),
        restated.composite,
        observations_relaunch=restated.relaunch,
    )


async def with_composite_observations(
    region: RegionResult, *, semaphore: asyncio.Semaphore | None
) -> RegionResult:
    """`region`, a composite answer, embedding the preview its relaunch closure reads
    (`CompositeRelaunch.read`) as a buffered answer embeds one
    (`dispatch._attach_observations`), with the component reads' warnings keyed to its
    spec_id and its relaunch closure dropped. A read the preview swallowed
    (`note_degraded_read`) marks it `degraded`, so no cache stores it. The read is timed as
    the `composite_preview` step."""
    started = now_ms()
    relaunch = region.observations_relaunch
    assert isinstance(relaunch, CompositeRelaunch)
    spec = region.spec
    with recording_degraded_reads() as lost:
        observations, raised = await relaunch.read(semaphore=semaphore)
    warnings = list(region.warnings)
    warnings.extend(warning for warning in _keyed(raised, spec.spec_id) if warning not in warnings)
    spec = _attach_observations(spec, (observations, None), warnings)
    return dataclasses.replace(
        region,
        specs=(spec, *region.specs[1:]),
        warnings=tuple(warnings),
        degraded=region.degraded or lost.degraded,
        observations_relaunch=None,
        timing_by_step={**region.timing_by_step, "composite_preview": now_ms() - started},
    )


def _keyed(warnings: Sequence[Warning], spec_id: str) -> list[Warning]:
    """`warnings` keyed to `spec_id`, each once."""
    keyed: list[Warning] = []
    for warning in warnings:
        restated = warning.model_copy(update={"spec_id": spec_id})
        if restated not in keyed:
            keyed.append(restated)
    return keyed
