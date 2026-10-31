"""The observation preview of a composite answer, read per component as the Datasets mint
reads a composite.

A composite answer is resolved for its table identity and restated as the composite
(`scope.composite.assemble_composite`), so the one table that resolve read previews nothing of
the composite. Its preview reads each component identity
(`code_index.composites.component_identities`) with `resolve_spec_resubmit`, which elects
its members as the mint does and scopes its reads to the provenance it binds, in worker threads
that run together. Each component read embeds its preview
(`include_observations`) and reads no slot or place options, and on a dataset_resubmit it goes
through the part memo under its own key. It reads through the request's catalogue narrowed to
its own pin alone (`ComponentReads.catalogue`), as the mint reads a component
(`datasets_api.serve`), whatever source scope the part's own reads ran under. The previews are
composed under the composite's rules (`scope.composite.compose_observations`).

A streamed request reads the preview in a trailing task (`CompositeRelaunch`),
launched once the answer is restated, so it adds nothing before the `response` event. A
buffered request reads it inline (`with_composite_observations`).
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
from search.engine.envelope import now_ms
from search.engine.errors import UPSTREAM_READ_ERRORS, EngineInputError, GraphInfraError
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
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.models import (
    BaseIdentity,
    EntityGroup,
    EntityRoleKey,
    ResubmitIdentityPayload,
    Spec,
    SpecObservations,
    Warning,
)
from search.observations import (
    OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED,
    OBSERVATIONS_PREVIEW_TOO_LARGE,
    OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
)
from search.scope.composite import ServedComponent, compose_observations, served_components
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
    """The request's clients, coverage snapshot, catalogue and deadline every component read
    of a composite runs on, and the dataset_resubmit's part memo (None on a fresh search).

    `catalogue` is the request's catalogue before any part narrows it to its source scope
    (`origin_catalogue.scoped_reads`). The composite's served years and its component reads
    read through it (`read_composite_preview`), so a component's own pin
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


async def read_composite_preview(
    identity: BaseIdentity,
    *,
    entity_groups: Mapping[str, EntityGroup] | None,
    reads: ComponentReads,
    semaphore: asyncio.Semaphore | None,
) -> tuple[SpecObservations, tuple[Warning, ...]]:
    """The preview of composite `identity`, composed from one read of each component it
    serves (`scope.composite.served_components`), each posting `entity_groups`, the groups
    the composite's answer names, as a resubmit of that answer would, and the warnings those
    reads raised under `PREVIEW_WARNING_CODES`, in component order.

    The reads run together, each holding a permit of `semaphore` when one is given.
    Cancelling the caller cancels every read still running. A component with no read, one
    whose resolve refused its identity, answered no definite Spec, failed upstream or crossed
    the deadline, leaves its years out (`compose_observations`) and raises a warning naming
    its role (`_read_component`), as a single answer's preview read turns a failed read into
    an omission (`observations.fetch`). Any other error of a read propagates and cancels the
    others."""
    facts = ScopeFacts(index=get_index(), coverage=reads.coverage)
    document = identity.model_dump(mode="json", exclude_none=True)
    # Coverage admits through the bound catalogue's source scope, so the served years and
    # every component read, each in a thread or task that copies this context, read through
    # the request's catalogue.
    with bound_catalogue(reads.catalogue):
        served = await asyncio.to_thread(served_components, document, facts=facts)
        # A Spec keys its groups by the role keys a resubmit posts them under.
        groups = cast("Mapping[EntityRoleKey, EntityGroup] | None", entity_groups)
        tasks = [
            asyncio.create_task(_read_component(item, groups, reads=reads, semaphore=semaphore))
            for item in served
        ]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
    composed = compose_observations(served, [observations for observations, _ in results])
    return composed, tuple(warning for _, raised in results for warning in raised)


async def _read_component(
    item: ServedComponent,
    groups: Mapping[EntityRoleKey, EntityGroup] | None,
    *,
    reads: ComponentReads,
    semaphore: asyncio.Semaphore | None,
) -> tuple[SpecObservations | None, tuple[Warning, ...]]:
    """The preview `item`'s component answer embeds and its preview warnings, or None when it
    has none. A part memo hit reads nothing, and a miss stores the component's answer.

    A read that fails upstream (`GraphInfraError`, `UPSTREAM_READ_ERRORS`) is logged and
    warns `observations_probe_failed`, and one past the deadline (`TimeoutError`,
    `GraphDeadlineExceededError`) warns `observations_timed_out`. A component whose resolve
    refuses its identity (`EngineInputError`) or answers no definite Spec warns
    `observations_probe_failed` where it serves a year (`ServedComponent.served`), since
    the preview then lacks years the composite serves. Each warning names the component's
    role."""
    identity = BaseIdentity.model_validate(dict(item.component.identity))
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
            region, stored = await asyncio.to_thread(
                _resolve_component, payload, reads=reads, options=options, key=key
            )
        except EngineInputError:
            return None, _unanswered(item, "refused its identity")
        except (GraphInfraError, *UPSTREAM_READ_ERRORS, TimeoutError) as exc:
            timed_out = isinstance(exc, TimeoutError | dc_graph.GraphDeadlineExceededError)
            cause = "read crossed the request deadline" if timed_out else "read failed"
            logger.warning(
                "composite preview: component role=%s identity=%r %s",
                item.component.role,
                identity.model_dump(mode="json", exclude_none=True),
                cause,
                exc_info=True,
            )
            return None, (_missing(item, cause, timed_out=timed_out),)
    if memo is not None and key is not None and not stored:
        memo.store(key, region)
    if region.status != "definite":
        return None, _unanswered(item, f"answered {region.status}")
    raised = tuple(warning for warning in region.warnings if warning.code in PREVIEW_WARNING_CODES)
    return region.spec.resolved.observations, raised


def _missing(item: ServedComponent, cause: str, *, timed_out: bool = False) -> Warning:
    """The warning that `item`'s component, whose read `cause` states, left its years out of
    the composite's preview: `observations_timed_out` when the read crossed the deadline,
    else `observations_probe_failed`."""
    message = (
        f"the composite's {item.component.role} component {cause}; its years are missing "
        "from the preview."
    )
    if timed_out:
        return Warning(code=OBSERVATIONS_TIMED_OUT, severity="warn", message=message)
    return Warning(code=OBSERVATIONS_PROBE_FAILED, severity="warn", message=message)


def _unanswered(item: ServedComponent, cause: str) -> tuple[Warning, ...]:
    """`observations_probe_failed` for `item`'s component, which `cause` states answered
    nothing, where it serves a year; nothing where it serves none."""
    if not item.served:
        return ()
    return (_missing(item, cause),)


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
    builds it as a fresh resolve does, and a same-shape merge rebuilds it over the merged
    Spec (`over`), whose identity holds every member's entities."""

    spec: Spec
    reads: ComponentReads

    def __call__(self, spec_id: str) -> asyncio.Task[_ObsProbeResult]:
        return create_owned_task(self._events(spec_id))

    def over(self, spec: Spec) -> CompositeRelaunch:
        """This relaunch reading composite `spec` in place of its own."""
        return dataclasses.replace(self, spec=spec)

    async def _events(self, spec_id: str) -> _ObsProbeResult:
        observations, raised = await read_composite_preview(
            self.spec.definition.identity,
            entity_groups=self.spec.resolved.entity_groups,
            reads=self.reads,
            semaphore=bounded_semaphore(),
        )
        warnings = _keyed(raised, spec_id)
        projected = _deflator_projected_warning(spec_id, observations)
        if projected is not None:
            warnings.append(projected)
        return (ObservationsEvent(spec_id=spec_id, observations=observations),), tuple(warnings)


async def with_composite_observations(
    region: RegionResult, *, reads: ComponentReads, semaphore: asyncio.Semaphore | None
) -> RegionResult:
    """`region`, a composite answer, embedding its preview (`read_composite_preview`) as a
    buffered answer embeds one (`dispatch._attach_observations`), with the component reads'
    warnings keyed to its spec_id and its relaunch closure dropped. The read is timed as the
    `composite_preview` step."""
    started = now_ms()
    spec = region.spec
    observations, raised = await read_composite_preview(
        spec.definition.identity,
        entity_groups=spec.resolved.entity_groups,
        reads=reads,
        semaphore=semaphore,
    )
    warnings = list(region.warnings)
    warnings.extend(warning for warning in _keyed(raised, spec.spec_id) if warning not in warnings)
    spec = _attach_observations(spec, (observations, None), warnings)
    return dataclasses.replace(
        region,
        specs=(spec, *region.specs[1:]),
        warnings=tuple(warnings),
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
