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
(`CompositeRelaunch`), and any other merge takes one resubmit of it (`IdentityPreview`). A
component read and a merged read are both `_resubmit_preview`, so they elect, cap and fail
alike.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import dc_graph
from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import Catalogue, bound_catalogue
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.concurrency import bounded_semaphore, create_owned_task
from search.engine.degraded_reads import (
    LOST_READ_ERRORS,
    note_degraded_read,
    note_lost_read,
    recording_degraded_reads,
)
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError
from search.engine.events import ObservationsEvent
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
)
from search.engine.region.observations.dispatch import (
    _attach_observations,
    _deflator_projected_warning,
)
from search.engine.region.resubmit import resolve_spec_resubmit
from search.engine.region.result import RegionResult, with_composite_answer
from search.engine.region.state import EngineOptions
from search.engine.response_cache_keys import is_region_storable
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    EntityGroup,
    EntityRoleKey,
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
from search.scope.region import region_groups
from search.scope.tables import ScopeFacts
from series_coverage.shared import CoverageSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from search.engine.response_cache_flow import PartMemoScope

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
        identity, groups, reads=reads, semaphore=semaphore, about=f"component {item.component.role}"
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
) -> tuple[SpecObservations, tuple[Warning, ...]] | _LostRead:
    """The preview a resubmit of `identity` posting `groups` embeds, read with
    `resolve_spec_resubmit` in a worker thread through `reads.catalogue` under `reads.options`,
    holding a permit of `semaphore` when one is given, and the answer's warnings under
    `PREVIEW_WARNING_CODES`. A part memo hit reads nothing, and a miss stores the answer.

    A read lost to `degraded_reads.LOST_READ_ERRORS` is noted (`note_lost_read`, so a read cut
    by its enclosing budget notes nothing), logged under `about`, and is lost
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


def cut_merged_preview() -> tuple[SpecObservations, tuple[Warning, ...]]:
    """The preview of a merged answer whose read was cut before it ended
    (`same_shape._read_merged_previews`): omitted `observations_timed_out`, with its warning."""
    return _LostRead(timed_out=True, cause="ran past its budget").omission(
        "the merged answer's preview ran past its budget."
    )


async def _preview_events(
    relaunch: CompositeRelaunch | IdentityPreview, spec_id: str
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
    observations."""
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
    served = composite_served(identity, reads=reads)
    table = merged.model_copy(
        update={
            "resolved": merged.resolved.model_copy(update={"headline": member.resolved.headline})
        }
    )
    with bound_catalogue(reads.catalogue):
        spec, _ = assemble_composite(
            table,
            identity,
            spec_id,
            served=served,
            facts=ScopeFacts(index=get_index(), coverage=reads.coverage),
        )
    warnings = tuple(
        warning.model_copy(update={"spec_id": spec_id})
        if warning.spec_id == merged.spec_id
        else warning
        for warning in region.warnings
    )
    return with_composite_answer(
        dataclasses.replace(region, warnings=warnings, observations_task=None),
        spec,
        observations_relaunch=CompositeRelaunch(spec=spec, reads=reads, served=served),
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
