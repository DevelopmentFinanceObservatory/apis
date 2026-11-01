"""Dispatch previews as buffered Spec data or trailing SSE events.

Shared helpers attach warnings, resolved facets, and coverage evidence to buffered results.

A preview reads the plan resolve recorded (`plan`, see `retrieval.read_plan`). Under
`latest` each StatVar serves one period, on every route, and only the series holding it, as
a mint of the same Spec does: in the year a role-bearing plan elected for it
(`ReadPlan.elected_years`), or without one the newest period any of its served series holds
(`materialise.window.latest_period`). Coverage dates each route the same way
(`retrieval.coverage`):

- Named series within the pair cap fetch their latest rows with SDMX `LATEST`, and a
  StatVar that elected its year the rows in that year when a series ends after it. Once
  admission and currency omissions have run, Search selects the StatVar's one period
  across its reads, the newest in the elected year when a read holds a row there and else
  the newest any read holds, and converts the rows holding it. Coverage dates remain
  index-derived, and returned points outside those dates are logged
  (`_served_outside_planned_years`).
- A bulk role preview, a population above the cap, serves each member its one year alone
  (`BulkPlan.member_year`), read in that year for a member that elected it and with SDMX
  `LATEST` for any other. The headline names the newest year any member serves and counts
  the members serving it. A pair whose read holds no point in its member's year is dropped
  and logged.

A scope-defaulted bulk preview (`bulk_scope`) reads each place's latest point of its one
facet and serves each variable the places holding its one period, so coverage names the span
those periods fall in, as the per-pair route does.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import GraphReader
from materialise.facets import FacetSelection
from materialise.finish import deflator_projected_examples, deflator_projected_warning
from materialise.rate_tables import RateTableCache
from origin_catalogue import recording_un_blocked_places
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.concurrency import create_owned_task
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.events import ObservationsEvent
from search.engine.region.codes import DEFLATOR_PROJECTED, _ObsProbeResult
from search.engine.region.observations.bulk_scope import BulkScope
from search.engine.region.observations.fetch import (
    _build_observations,
)
from search.engine.region.result import FacetAxisFilter
from search.engine.region.un_system import un_unavailable_warning
from search.engine.retrieval.read_plan import ReadPlan
from search.models import (
    CoverageExact,
    IdentityUnitRequest,
    Spec,
    SpecObservations,
    Warning,
    in_window,
)

if TYPE_CHECKING:
    from search.engine.retrieval.member_facts import MemberFacts
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)


def _observations_events(
    spec_id: str,
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    graph: EngineGraphClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None = None,
    scope: BulkScope | None = None,
    sdmx: EngineSdmxClient | None = None,
    deadline: float | None = None,
    member_facts: Mapping[str, MemberFacts] | None = None,
    rate_cache: RateTableCache | None = None,
    unit_request: IdentityUnitRequest | None = None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None = None,
    coverage: CoverageSnapshot,
    plan: ReadPlan | None = None,
    variable_labels: Mapping[str, str],
) -> _ObsProbeResult:
    """Build one trailing event. `_build_observations` encodes empty previews in its payload.

    Fetch options pass unchanged to `_build_observations`. Missing optional inputs disable
    their branches. The event runs after the answer's own warnings are built, so places the
    UN front door refused during this read carry their own `un_system_unavailable` warning,
    its variables named from `variable_labels`, the names of the Spec's members. The event
    states whether the read swallowed a read (`ObservationsEvent.degraded`).
    """
    # The task runs once the answer's resolution has ended, so it collects the reads it
    # swallows on a record of its own.
    with recording_un_blocked_places() as un_blocked, recording_degraded_reads() as lost:
        observations, warn = _build_observations(
            plan,
            sv_dcids,
            obs_entities,
            graph=graph,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            facet_filter=facet_filter,
            scope=scope,
            sdmx=sdmx,
            deadline=deadline,
            member_facts=member_facts,
            rate_cache=rate_cache,
            unit_request=unit_request,
            elected=elected,
            coverage=coverage,
        )
    # Empty previews carry an omission reason in the payload. The event fires.
    event = ObservationsEvent(spec_id=spec_id, observations=observations)
    events = (event.stating_degraded(lost.degraded),)
    warns: list[Warning] = []
    if warn is not None:
        warns.append(warn.model_copy(update={"spec_id": spec_id}))
    # Use the same projected-conversion warning as the buffered path.
    extra = _deflator_projected_warning(spec_id, observations)
    if extra is not None:
        warns.append(extra)
    refused = un_unavailable_warning(un_blocked, labels=variable_labels)
    if refused is not None:
        warns.append(refused.model_copy(update={"spec_id": spec_id}))
    return events, tuple(warns)


def _launch_observations_task(
    spec_id: str,
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    graph: EngineGraphClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None = None,
    scope: BulkScope | None = None,
    sdmx: EngineSdmxClient | None = None,
    deadline: float | None = None,
    member_facts: Mapping[str, MemberFacts] | None = None,
    rate_cache: RateTableCache | None = None,
    unit_request: IdentityUnitRequest | None = None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None = None,
    coverage: CoverageSnapshot,
    plan: ReadPlan | None = None,
    variable_labels: Mapping[str, str],
) -> asyncio.Task[_ObsProbeResult]:
    """Schedule `_observations_events` in a worker thread and return its owned task."""
    return create_owned_task(
        asyncio.to_thread(
            _observations_events,
            spec_id,
            sv_dcids,
            obs_entities,
            graph=graph,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            facet_filter=facet_filter,
            scope=scope,
            sdmx=sdmx,
            deadline=deadline,
            member_facts=member_facts,
            rate_cache=rate_cache,
            unit_request=unit_request,
            elected=elected,
            coverage=coverage,
            plan=plan,
            variable_labels=variable_labels,
        )
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreviewRead:
    """What one answer's observation preview reads, apart from the request's clients.

    Every input of `_build_observations` but the clients, the coverage snapshot and the
    deadline: the served members `sv_dcids`, the entities they are read at by role
    (`obs_entities`), the read plan, the facet filter, the places a scope-defaulted bulk read
    names (`bulk_places`, None for any other read), the members' facts, which give each
    series its declared roles and `axes` (`fetch._build_observations`), the posted
    request-form unit and the elected unit selections. Every field is required, so a path
    that builds one cannot leave an input out. A fresh answer builds it once
    (`phase_answer._start_assemble_definite`), a resubmit once (`promote_finalize`), and the
    part memo stores it with the region (`RegionResult.preview_read`). `PreviewReader`
    binds it to one request's clients."""

    sv_dcids: tuple[str, ...]
    obs_entities: Mapping[str, Sequence[str]]
    plan: ReadPlan | None
    facet_filter: Mapping[str, FacetAxisFilter] | None
    bulk_places: tuple[str, ...] | None
    member_facts: Mapping[str, MemberFacts] | None
    unit_request: IdentityUnitRequest | None
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class PreviewReader:
    """`read` bound to one request's clients, coverage snapshot, conversion memo and
    deadline. A buffered answer reads it in a worker thread (`observations`), and a
    streamed one launches it as a trailing task (`launch`, `relaunch`), so both read the
    same inputs and preview the same."""

    read: PreviewRead
    graph: EngineGraphClient
    sdmx: EngineSdmxClient | None
    materialise_graph: GraphReader | None
    materialise_sdmx: MaterialiseSdmxClient | None
    rate_cache: RateTableCache | None
    deadline: float | None
    coverage: CoverageSnapshot

    def observations(self) -> tuple[SpecObservations, Warning | None]:
        """The preview, read in the calling thread, with the read's own warning."""
        read = self.read
        return _build_observations(
            read.plan,
            list(read.sv_dcids),
            read.obs_entities,
            graph=self.graph,
            materialise_graph=self.materialise_graph,
            materialise_sdmx=self.materialise_sdmx,
            facet_filter=read.facet_filter,
            scope=self._scope(),
            sdmx=self.sdmx,
            deadline=self.deadline,
            member_facts=read.member_facts,
            rate_cache=self.rate_cache,
            unit_request=read.unit_request,
            elected=read.elected,
            coverage=self.coverage,
        )

    def launch(
        self, spec_id: str, *, variable_labels: Mapping[str, str]
    ) -> asyncio.Task[_ObsProbeResult]:
        """The preview as a request-owned trailing task under `spec_id`
        (`_observations_events`), naming the UN variables of a refused read from
        `variable_labels`. Only the event loop's thread may call it."""
        read = self.read
        return _launch_observations_task(
            spec_id,
            list(read.sv_dcids),
            read.obs_entities,
            graph=self.graph,
            materialise_graph=self.materialise_graph,
            materialise_sdmx=self.materialise_sdmx,
            facet_filter=read.facet_filter,
            scope=self._scope(),
            sdmx=self.sdmx,
            deadline=self.deadline,
            member_facts=read.member_facts,
            rate_cache=self.rate_cache,
            unit_request=read.unit_request,
            elected=read.elected,
            coverage=self.coverage,
            plan=read.plan,
            variable_labels=variable_labels,
        )

    def relaunch(
        self, variable_labels: Mapping[str, str]
    ) -> Callable[[str], asyncio.Task[_ObsProbeResult]]:
        """`launch` for any spec_id, a merged one included (`RegionResult.observations_relaunch`).
        The labels name the Spec's members, which a fresh answer knows only once its Spec is
        built, after its buffered read starts, so they bind here, outside `PreviewRead`."""
        return functools.partial(self.launch, variable_labels=variable_labels)

    def _scope(self) -> BulkScope | None:
        if self.read.bulk_places is None:
            return None
        assert self.sdmx is not None, "a scope-defaulted bulk preview reads through SDMX"
        return BulkScope(places=self.read.bulk_places, sdmx=self.sdmx)


def _deflator_projected_warning(spec_id: str, observations: SpecObservations) -> Warning | None:
    """Return a warning when any converted series has projected points."""
    projected = [
        series.conversion
        for series in observations.series
        if series.conversion is not None and series.conversion.projected_count > 0
    ]
    if not projected:
        return None
    rows = sum(record.projected_count for record in projected)
    built = deflator_projected_warning(rows)
    return Warning(
        code=DEFLATOR_PROJECTED,
        severity=built["severity"],
        message=built["message"],
        spec_id=spec_id,
        params={"count": rows, "examples": deflator_projected_examples(projected)},
    )


def _served_outside_planned_years(spec: Spec, observations: SpecObservations) -> int:
    """How many delivered points fall outside the years coverage states.

    Coverage dates a `latest` answer from coverage-index dates (`read_plan.series_dates`),
    and the per-series preview serves each StatVar's actual latest period from SDMX.
    The two agree while the index matches SDMX. A point earlier or later than the index
    states counts here."""
    window = spec.resolved.coverage.window
    if window is None:
        return 0
    return sum(
        1
        for series in observations.series
        for point in series.points
        if not in_window(point.date, window)
    )


def _attach_observations(
    spec: Spec,
    result: tuple[SpecObservations, Warning | None],
    warnings: list[Warning],
    *,
    include_points: bool = True,
) -> Spec:
    """Attach observations and warnings to a Spec.

    `resolved.facets` stays as grounding built it from the read plan
    (`facets.answer_facets`), so the preview never changes it. With `include_points=False`,
    attach warnings only and omit the projected-conversion warning. Restamp
    `CoverageExact.evidence` as `observed` only when points are included and at least one
    series was delivered. Failed or empty previews leave coverage unchanged.
    """
    observations, warn = result
    if warn is not None:
        warnings.append(warn.model_copy(update={"spec_id": spec.spec_id}))
    if include_points and (outside := _served_outside_planned_years(spec, observations)):
        logger.warning(
            "spec %s delivered %d point(s) outside its coverage window %r: the coverage "
            "index and the observations read disagree",
            spec.spec_id,
            outside,
            spec.resolved.coverage.window,
        )
    if include_points:
        extra = _deflator_projected_warning(spec.spec_id, observations)
        if extra is not None:
            warnings.append(extra)
    # Observations belongs to ResolvedSpec. model_copy(update=...) does not validate field names.
    # `_build_observations` always returns a payload, including for empty previews.
    update: dict[str, object] = {}
    # Coverage is observed only when this payload includes at least one series.
    ships_observations = include_points
    if ships_observations:
        update["observations"] = observations
    if (
        ships_observations
        and observations.series
        and isinstance(spec.resolved.coverage, CoverageExact)
    ):
        update["coverage"] = spec.resolved.coverage.model_copy(update={"evidence": "observed"})
    spec = spec.model_copy(update={"resolved": spec.resolved.model_copy(update=update)})
    return spec
