"""The observation preview of a composite answer, read per component as the Datasets mint
reads a composite.

A composite answer is minted as the composite (`spec_build.build_spec(composite=)`,
`scope.composite.CompositeMint`) over the read its elected reading names, so the one table that
read resolved previews nothing of the composite. Its preview reads each component identity
(`code_index.composites.component_identities`) with `resolve_spec_resubmit`, which elects
its members as the mint does and scopes its reads to the provenance it binds, in worker threads
that run together. Each component read embeds its preview
(`include_observations`), elects under the component's own window and reads rows in the
years the component serves (`scope.composite.component_preview`), and reads no slot or place
options, and on a dataset_resubmit it goes through the part memo under its own key and
those years. A component previewing no year of the composite
(`ServedComponent.preview_years`) is not read. It reads through the request's catalogue narrowed to
its own pin alone (`ComponentReads.catalogue`), as the mint reads a component
(`datasets_api.serve`), whatever source scope the part's own reads ran under. The previews are
composed under the composite's rules (`scope.composite.compose_observations`).

A streamed request reads the preview in a trailing task (`CompositeRelaunch`),
launched once the answer is minted, so it adds nothing before the `response` event. A
buffered request reads it inline (`with_composite_observations`). The composite's continuation
(`modelled_values`) is then the reads' (`SpecObservations.continued`). A request reading no
preview states the continuation the components' read plans serve in the years each serves
(`with_composite_continuation`): a component that reads the table the answer read takes the
answer's own plan, and each other resolves its plan with no row read, as its preview's
resolve elects it.

A same-shape merged answer previews the read of its merged identity
(`same_shape._with_merged_preview`). A merge of composite answers is minted as the
composite over that identity and reads its components (`CompositeRelaunch`). Any other merge
whose read plan the merge rebuilds in memory reads that plan (`PlanPreview`), so its preview and
its download come from one plan, and the rest take one resubmit of the merged identity
(`IdentityPreview`). A component read and a resubmitted
merged read are both `_resubmit_preview`, so they elect, cap and fail alike.
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
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import Catalogue, bound_catalogue, recording_un_blocked_places
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.concurrency import (
    bounded_semaphore,
    create_owned_task,
)
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
    PreviewRead,
    PreviewReader,
    _attach_observations,
    _deflator_projected_warning,
)
from search.engine.region.resubmit import resolve_spec_resubmit
from search.engine.region.result import (
    RegionResult,
)
from search.engine.region.state import EngineOptions
from search.engine.region.un_system import un_unavailable_warning
from search.engine.response_cache_keys import is_region_storable
from search.models import (
    BaseIdentity,
    EntityGroup,
    EntityRoleKey,
    ResolvedSpec,
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
    component_preview,
    component_spec_id,
    compose_observations,
    served_components,
)
from search.scope.region import region_groups
from search.scope.tables import ScopeFacts
from series_coverage.shared import CoverageSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from code_index.scope_conventions import RolePopulation
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
    composite is minted as (`scope.composite.CompositeMint`) and its preview read these same
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
    component previewing no year (`ServedComponent.preview_years`) is not read and raises
    nothing.

    The reads run together, each holding a permit of `semaphore` when one is given.
    Cancelling the caller cancels every read still running. A component whose read is lost
    (`_resubmit_preview`) leaves its years out (`compose_observations`) and raises a warning
    naming its role (`_read_component`), as a single answer's preview read turns a failed
    read into an omission (`observations.fetch`). A component whose answer stands on less
    than a later request may read (`RegionResult.degraded`) notes a degraded read. Any other
    error of a read propagates and cancels the others."""
    # A Spec keys its groups by the role keys a resubmit posts them under.
    groups = cast("Mapping[EntityRoleKey, EntityGroup] | None", entity_groups)
    read = [index for index, item in enumerate(served) if item.preview_years]
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
    (`_resubmit_preview`), elected under the identity and read in the years
    `component_preview` names, or None and a warning naming the component's role when its
    read was lost (`_missing`)."""
    document, years = component_preview(item)
    read = await _resubmit_preview(
        BaseIdentity.model_validate(document),
        groups,
        reads=reads,
        semaphore=semaphore,
        about=f"component {item.component.role}",
        populations=item.component.populations,
        years=years,
    )
    if isinstance(read, _LostRead):
        return None, (_missing(item, read),)
    return read


async def read_composite_continuation(
    *,
    served: Sequence[ServedComponent],
    resolved: ResolvedSpec,
    reads: ComponentReads,
    semaphore: asyncio.Semaphore | None,
) -> bool | None:
    """Whether the composite whose components are `served` and whose answer states
    `resolved` serves a continuation in the years they serve, settled with no preview read.
    A component previewing a year (`ServedComponent.preview_years`) whose preview resolves
    the table the answer read states the continuation the answer's own plan serves in those
    years (`ResolvedSpec.component_continued`). Each other one resolves its read plan as its
    preview's resolve elects it (`_component_continuation`). The composite serves one when
    any plan narrowed to its component's years does (`read_plan.serves_continuation`), none
    when every plan serves none, and states nothing (None) when no plan serves one and a
    component's resolve states none, as `compose_observations` settles it from the reads.
    A resolve that was lost marks the answer degraded, so no cache stores it. A resolve that
    refused the identity or answered no definite Spec states no claim and leaves the answer
    cacheable (`_component_continuation`).

    The resolves run together, each holding a permit of `semaphore` when one is given, and
    read no row. Cancelling the caller cancels every resolve still running."""
    groups = cast("Mapping[EntityRoleKey, EntityGroup] | None", resolved.entity_groups)
    previewing = [item for item in served if item.preview_years]
    if resolved.states_component_continued:
        # Hashing each component's identity is CPU work, kept off the event loop.
        stated, resolving = await asyncio.to_thread(_held_continuations, previewing, resolved)
    else:
        stated, resolving = [], previewing
    tasks = [
        asyncio.create_task(_component_continuation(item, groups, reads=reads, semaphore=semaphore))
        for item in resolving
    ]
    try:
        stated.extend(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            task.cancel()
    if any(stated):
        return True
    return False if all(one is False for one in stated) else None


def _held_continuations(
    previewing: Sequence[ServedComponent], resolved: ResolvedSpec
) -> tuple[list[bool | None], list[ServedComponent]]:
    """The continuation `resolved` states for each of `previewing` whose preview resolves the
    table the answer read (`ResolvedSpec.component_continued`), and the others, which are
    resolved."""
    stated: list[bool | None] = []
    resolving: list[ServedComponent] = []
    for item in previewing:
        held = resolved.component_continued(component_spec_id(item), item.preview_years)
        if held is None:
            resolving.append(item)
        else:
            stated.append(held)
    return stated, resolving


async def _component_continuation(
    item: ServedComponent,
    groups: Mapping[EntityRoleKey, EntityGroup] | None,
    *,
    reads: ComponentReads,
    semaphore: asyncio.Semaphore | None,
) -> bool | None:
    """Whether `item`'s component answer, resolved with no preview over the identity and
    years `component_preview` names (`_resubmit`), serves a continuation in those years
    (`ResolvedSpec.continued`), or None when its resolve states none.

    A resolve lost to `degraded_reads.LOST_READ_ERRORS` is transient. `_resubmit` notes and
    logs it, so the answer is degraded and no cache stores it. A resolve that refuses the
    identity (`EngineInputError`) or answers no definite Spec is deterministic, so it states
    no continuation and leaves the answer cacheable. Each of these is logged at INFO with the
    component's role, its identity and the cause, so a component whose resolve keeps
    refusing shows in the logs while its answers lose their modelled-value claims. Any other
    error is a code fault and propagates, as on the preview path."""
    document, years = component_preview(item)
    region = await _resubmit(
        BaseIdentity.model_validate(document),
        groups,
        reads=reads,
        semaphore=semaphore,
        about=f"component {item.component.role} plan",
        populations=item.component.populations,
        years=years,
        observations=False,
    )
    if isinstance(region, _LostRead):
        logger.info(
            "composite continuation: component %s states none: %s",
            item.component.role,
            region.cause,
            extra={
                JSON_FIELDS_ATTR: {
                    "component_role": item.component.role,
                    "component_identity": document,
                    "continuation_cause": region.cause,
                }
            },
        )
        return None
    return region.spec.resolved.continued


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
    years: frozenset[int] | None = None,
) -> tuple[SpecObservations, tuple[Warning, ...]] | _LostRead:
    """The preview a resubmit of `identity` posting `groups` embeds (`_resubmit`) and the
    answer's warnings under `PREVIEW_WARNING_CODES`. The preview reads rows in `years`, its
    members elected under `identity`'s own window (`EngineOptions.preview_years`), and in
    the years its read plan elected under where None. A preview read in `years`, a
    composite component's, states whether the answer serves a continuation in them
    (`SpecObservations.continued`), which the composite's own Spec cannot state. An answer
    that embeds no preview is lost `observations_probe_failed`."""
    region = await _resubmit(
        identity,
        groups,
        reads=reads,
        semaphore=semaphore,
        about=about,
        populations=populations,
        years=years,
        observations=True,
    )
    if isinstance(region, _LostRead):
        return region
    observations = region.spec.resolved.observations
    if observations is None:
        return _LostRead(timed_out=False, cause="embedded no preview")
    if years is not None:
        # The answer's Spec states the continuation its plan serves in `years` alone
        # (`spec_build.build_spec`'s `preview_years`), which the composite answer states.
        observations = observations.stating_continued(region.spec.resolved.continued)
    raised = tuple(warning for warning in region.warnings if warning.code in PREVIEW_WARNING_CODES)
    return observations, raised


async def _resubmit(
    identity: BaseIdentity,
    groups: Mapping[EntityRoleKey, EntityGroup] | None,
    *,
    reads: ComponentReads,
    semaphore: asyncio.Semaphore | None,
    about: str,
    populations: Mapping[str, RolePopulation] | None,
    years: frozenset[int] | None,
    observations: bool,
) -> RegionResult | _LostRead:
    """The definite answer a resubmit of `identity` posting `groups` resolves, built, keyed
    and read with `resolve_spec_resubmit` in a worker thread (`_resolve_component`) through
    `reads.catalogue` under `reads.options`, embedding its preview where `observations` is set and reading no row
    otherwise, holding a permit of `semaphore` when one is given. Its Spec states the
    continuation its read plan serves in `years` (`EngineOptions.preview_years`) where set.
    A part memo hit reads nothing, and a miss stores the answer. Each role `populations`
    lists, the roles a component binds to a population (`ComponentIdentity.populations`),
    reads as open within it (`ResubmitIdentityPayload.populations`).

    A read lost to `degraded_reads.LOST_READ_ERRORS` is noted (`note_lost_read`, so a read cut
    by its enclosing budget is noted as cut and leaves the request unstorable), logged under
    `about`, and is lost `observations_timed_out` when it crossed the deadline
    (`TimeoutError`, `GraphDeadlineExceededError`), else `observations_probe_failed`. A
    resolve that refuses `identity` (`EngineInputError`) or answers no definite Spec is lost
    `observations_probe_failed`. An answer the response cache would not store
    (`is_region_storable` under the request catalogue's refused places), degraded or carrying
    a warning such as `members_absent_from_graph`, notes a degraded read
    (`note_degraded_read`), so no cache stores what it states. Any other error propagates."""
    options = dataclasses.replace(
        reads.options, include_observations=observations, preview_years=years
    )
    async with semaphore if semaphore is not None else nullcontext():
        try:
            # The worker thread copies this context, so the resolve reads through the
            # request's catalogue, which only `identity`'s own pin narrows.
            with bound_catalogue(reads.catalogue):
                region, stored, key = await asyncio.to_thread(
                    _resolve_component,
                    identity,
                    groups,
                    populations=populations,
                    reads=reads,
                    options=options,
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
    if reads.memo is not None and key is not None and not stored:
        reads.memo.store(key, region)
    if not is_region_storable(region, refused_at_build=reads.catalogue.un_blocked_places):
        note_degraded_read(f"resubmit preview: {about}")
    if region.status != "definite":
        return _LostRead(timed_out=False, cause=f"answered {region.status}")
    return region


def _missing(item: ServedComponent, lost: _LostRead) -> Warning:
    """The warning that `item`'s component, whose read was `lost`, left its years out of the
    composite's preview, under the lost read's code."""
    return lost.warning(
        f"the composite's {item.component.role} component {lost.cause}; its years are missing "
        "from the preview."
    )


def _resolve_component(
    identity: BaseIdentity,
    groups: Mapping[EntityRoleKey, EntityGroup] | None,
    *,
    populations: Mapping[str, RolePopulation] | None,
    reads: ComponentReads,
    options: EngineOptions,
) -> tuple[RegionResult, bool, str | None]:
    """The resubmit of `identity` posting `groups` (`_resubmit`): the active part memo's
    region for it, else a fresh resolve, whether the memo held it, and its memo key, None
    without an active memo. It runs in a worker thread, so building and hashing the payload
    keeps off the event loop."""
    payload = ResubmitIdentityPayload(
        identity=identity,
        entity_groups=None
        if groups is None
        else region_groups(groups, identity, index=get_index()),
        populations=dict(populations or {}),
    )
    memo = reads.memo if reads.memo is not None and reads.memo.active else None
    key = memo.key(payload, options) if memo is not None else None
    stored = memo.lookup(key) if memo is not None and key is not None else None
    if stored is not None:
        return stored, True, key
    region = resolve_spec_resubmit(
        inp=payload,
        graph=reads.graph,
        sdmx=reads.sdmx,
        coverage=reads.coverage,
        options=options,
    )
    return region, False, key


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeRelaunch:
    """The relaunch closure of composite `spec`'s preview (`RegionResult.observations_relaunch`):
    called with a spec_id, it launches `read_composite_preview` as a request-owned task on the
    event loop, under the request semaphore. The task's result is one `ObservationsEvent`
    under that spec_id and the component reads' warnings, with `deflator_projected` for the
    composed preview, each keyed to it.

    It reads `spec`'s identity and entity groups alone, so an answer the part memo held
    builds it as a fresh resolve does. `served` holds the components the answer was minted
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

    async def continuation(self, *, semaphore: asyncio.Semaphore | None) -> bool | None:
        """Whether the composite serves a continuation, settled from its components' read
        plans with no preview read (`read_composite_continuation`)."""
        served = self.served
        if served is None:
            served = await asyncio.to_thread(
                composite_served, self.spec.definition.identity, reads=self.reads
            )
        return await read_composite_continuation(
            served=served, resolved=self.spec.resolved, reads=self.reads, semaphore=semaphore
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


async def with_composite_continuation(
    region: RegionResult, *, semaphore: asyncio.Semaphore | None
) -> RegionResult:
    """`region`, a composite answer whose request reads no preview, stating whether it serves
    a continuation (`ResolvedSpec.continued`) as its components' read plans settle it
    (`CompositeRelaunch.continuation`): the answer's own plan for a component whose preview
    resolves the read the answer made, a resolve with no preview for each other one. Its
    relaunch closure is dropped. A composite whose
    plans settle nothing states none. A resolve that was lost or swallowed a read
    (`note_degraded_read`) marks it `degraded`, so no cache stores it. The resolves are timed
    as the `composite_continuation` step."""
    started = now_ms()
    relaunch = region.observations_relaunch
    assert isinstance(relaunch, CompositeRelaunch)
    with recording_degraded_reads() as lost:
        continued = await relaunch.continuation(semaphore=semaphore)
    spec = region.spec
    spec = spec.model_copy(update={"resolved": spec.resolved.stating_continued(continued is True)})
    return dataclasses.replace(
        region,
        specs=(spec, *region.specs[1:]),
        degraded=region.degraded or lost.degraded,
        observations_relaunch=None,
        timing_by_step={**region.timing_by_step, "composite_continuation": now_ms() - started},
    )


def _keyed(warnings: Sequence[Warning], spec_id: str) -> list[Warning]:
    """`warnings` keyed to `spec_id`, each once."""
    keyed: list[Warning] = []
    for warning in warnings:
        restated = warning.model_copy(update={"spec_id": spec_id})
        if restated not in keyed:
            keyed.append(restated)
    return keyed
