"""Fan-out: the N-variable streaming fan-out engine. Queues each variable
leg's region events under a semaphore and a soft deadline, interleaving them
as they arrive across the pipeline's queue, plus the deadline-bounded entity
pre-resolution that runs ahead of the per-variable fan-out.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import get_args

from pydantic import ValidationError

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from entity_resolution import Admission, EntityResolver, Resolution
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from search.engine.assemble import restatement_reason_for
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.concurrency import bounded_semaphore
from search.engine.deadline import _await_bounded, _cancel_trailing_tasks
from search.engine.errors import (
    RATE_TABLE_REFUSALS,
    DeclaredDefaultUnsupportedError,
    EngineInfraError,
)
from search.engine.events import Event
from search.engine.extract import DateRequest
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.qualifier_match import CodedQualifier, recall_text
from search.engine.region import RegionResult, resolve_variable
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.region.result import with_runnable_operations
from search.engine.region.state import EngineOptions
from search.engine.role_text import DoorRole, role_text_for_axis
from search.models import (
    Operation,
    ParsedEntity,
    ParsedPart,
    PartRestatementReason,
    PipelineStep,
    RoleAxis,
)
from series_coverage.shared import CoverageSnapshot

# The directional role axes ParsedPart's role_text-less generic echo cannot
# express -- derived from RoleAxis itself (currently "origin"/"destination")
# rather than hardcoded, so a sibling axis added to that vocabulary is picked
# up here automatically. RoleAxis is a PEP 695 `type` alias (dfo_platform),
# so its Literal members sit behind __value__ -- get_args(RoleAxis) itself
# returns (). See _leg_restatement.
_DIRECTIONAL_ROLES: frozenset[str] = frozenset(get_args(RoleAxis.__value__))


@dataclass(frozen=True, slots=True)
class _RegionDone:
    """Terminal marker: one variable leg finished with a RegionResult.

    `degraded` is True when `region` is a `_placeholder_no_data` produced by
    swallowing that leg's own exception (see `_FanOutResult.degraded_leg`),
    False for every other completion including a genuine no_data result the
    leg itself returned.
    """

    index: int
    region: RegionResult
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class _TaskFailed:
    """Terminal marker: one variable leg raised (infra error, or any error on N=1)."""

    index: int
    exc: BaseException


@dataclass
class _FanOutResult:
    """Out-parameter for the async generator's final state.

    The generator assigns these fields only on normal completion. `regions`
    contains one result per input leg, including deadline backfills;
    `timed_out` and `degraded_leg` distinguish deadline and error backfills.
    The generator cancels trailing tasks if it exits abnormally.
    """

    regions: list[RegionResult] = field(default_factory=list)
    timed_out: bool = False
    # True when at least one leg's exception, other than an EngineInfraError or a
    # RATE_TABLE_REFUSALS member, was swallowed into a `_placeholder_no_data`
    # result under `degrade_on_error`.
    # That leg's response part looks like a genuine no_data answer, so a
    # consumer that needs to tell a transient upstream failure apart from a
    # settled negative one, such as the response cache deciding what is safe
    # to store, reads this flag instead of the response body. It is also True
    # when a leg's answer states `alternates_unevaluated`
    # (`RegionResult.alternates_left_out`): an alternate went unevaluated for
    # want of a presence read, and when its presence read left a StatVar's
    # default scope unsettled (`RegionResult.scope_unsettled`). It does not
    # cover the deadline backfill, which `timed_out` reports.
    degraded_leg: bool = False


def _date_request_to_time_text(date_request: DateRequest | None) -> str | None:
    """Encode the date request in the parsed door's `time_text` grammar.

    Returns `latest`, a year/range/`since` phrase, or `None` when the request
    is dateless, open-start, or reversed. The latter windows have no phrase
    that round-trips through the parser.
    """
    if date_request is None:
        return None
    if date_request.latest:
        return "latest"
    window = date_request.window
    if window is None or window.start_year is None:
        return None
    if window.end_year is None:
        return f"since {window.start_year}"
    if window.start_year > window.end_year:
        return None
    if window.start_year == window.end_year:
        return str(window.start_year)
    return f"{window.start_year} to {window.end_year}"


def _region_has_directional_entity(region: RegionResult) -> bool | None:
    """True when some Spec this leg produced bound an entity onto a
    directional role axis (_DIRECTIONAL_ROLES) rather than the subject axis;
    False when it inspected at least one Spec and none bound one.

    Scans every Spec-shaped field the RegionResult can carry, not just
    `specs` -- a no_data outcome's flagged_match/alternate_match/nearest_real
    each carry a Spec resolved against the same entities, and a directional
    binding can hide there just as it can in a definite/candidate answer.

    None (not False) when none of those fields carries any Spec at all: with
    no binding anywhere to inspect, this function has no evidence either way
    -- the leg could just as easily have failed *because* an entity would
    have landed on a directional axis (e.g. an unresolvable region name on
    one side of an "X to Y" query) as for a reason unrelated to direction.
    Callers must not treat that absence of evidence as proof of "not
    directional"; see `_success_directional`, the only caller, for the
    conservative value it falls back to instead.
    """
    specs: list = list(region.specs)
    if region.flagged_match:
        specs.extend(region.flagged_match)
    if region.alternate_match is not None:
        specs.append(region.alternate_match)
    if region.nearest_real:
        specs.extend(region.nearest_real)
    if not specs:
        return None
    return any(
        entity.role in _DIRECTIONAL_ROLES for spec in specs for entity in spec.resolved.entities
    )


def _success_directional(
    region: RegionResult,
    *,
    role_query: str | None,
    entities: Sequence[str] = (),
    entity_roles: Mapping[str, list[DoorRole]] | None = None,
) -> bool:
    """Infer whether a successful leg needs directional restatement.

    A definite answer whose members declare one role (`RegionResult.answered_roles`)
    places every entity on that role by structure, so a bare echo resolves the same and
    the leg is not directional. Otherwise use resolved Specs when available. For
    `entity_ambiguous` without a Spec, use the door's per-entity axis map. Other cases
    without Spec evidence use `role_query`; a non-empty query marks direction as possible.
    """
    if region.status == "definite" and len(region.answered_roles or ()) == 1:
        return False
    found = _region_has_directional_entity(region)
    if found is None and region.no_data_reason == "entity_ambiguous":
        roles = entity_roles or {}
        return any(roles.get(name) for name in entities)
    if found is not None:
        return found
    return bool(role_query)


def _leg_parsed_form(
    variable_text: str,
    *,
    entities: Sequence[str],
    date_request: DateRequest | None,
    unit_text: str | None,
    source_text: str | None,
    entity_roles: Mapping[str, list[DoorRole]] | None = None,
    operations: Sequence[Operation] = (),
) -> ParsedPart | None:
    """Build the leg's inputs as one parsed part, the form `data_query` pins into.

    Each entity appears once per distinct axis the door gave it, with that axis as
    `role_text`, as the parsed door reads a name repeated on opposite axes. An entity
    without an axis appears bare. Return `None` when `ParsedPart` bounds reject any
    input.
    """
    roles = entity_roles or {}
    try:
        return ParsedPart(
            variable_text=variable_text,
            entities=[
                ParsedEntity(
                    text=e,
                    role_text=role_text_for_axis(axis) if axis is not None else None,
                )
                for e in entities
                for axis in (dict.fromkeys(role.axis for role in roles.get(e) or ()) or [None])
            ]
            or None,
            time_text=_date_request_to_time_text(date_request),
            unit_text=unit_text,
            source_text=source_text,
            operations=list(operations),
        )
    except ValidationError:
        return None


def _leg_restatement(
    variable_text: str,
    *,
    entities: Sequence[str],
    date_request: DateRequest | None,
    unit_text: str | None,
    source_text: str | None,
    directional: bool = False,
    entity_roles: Mapping[str, list[DoorRole]] | None = None,
    operations: Sequence[Operation] = (),
) -> tuple[ParsedPart | None, PartRestatementReason]:
    """Build the parsed-door echo and its reason when no echo is safe.

    The echo is the leg's parsed form (`_leg_parsed_form`) when every direction is
    named. If one name occurs on both axes, or direction is known but no entity has
    an axis, return `None` with `direction_unnameable`; an entity-free directional
    leg can still echo. If `ParsedPart` bounds reject any input, return `None` with
    `inputs_exceed_caps` rather than emit a truncated echo.
    """
    roles = entity_roles or {}
    axes: set[RoleAxis] = set()
    for name in entities:
        occurrences = roles.get(name) or ()
        if len({role.axis for role in occurrences}) > 1:
            return None, "direction_unnameable"
        if occurrences:
            axes.add(occurrences[0].axis)
    if entities and directional and not axes:
        return None, "direction_unnameable"
    part = _leg_parsed_form(
        variable_text,
        entities=entities,
        date_request=date_request,
        unit_text=unit_text,
        source_text=source_text,
        entity_roles=roles,
        operations=operations,
    )
    if part is None:
        return None, "inputs_exceed_caps"
    return part, "no_phrase_inputs"


def _placeholder_no_data(
    variable: str,
    index: int,
    *,
    entities: Sequence[str] = (),
    date_request: DateRequest | None = None,
    unit_text: str | None = None,
    source_text: str | None = None,
    role_query: str | None = None,
    entity_roles: Mapping[str, list[DoorRole]] | None = None,
    operations: tuple[Operation, ...] = (),
) -> RegionResult:
    """Build a `no_data` result for a leg backfilled after a soft deadline.

    Its restatement and operations use the leg's own inputs. The door's axis map supplies
    entity roles; without an axis, a non-empty `role_query` makes direction
    uncertain and suppresses the echo.
    """
    restatement, reason = _leg_restatement(
        variable,
        entities=entities,
        date_request=date_request,
        unit_text=unit_text,
        source_text=source_text,
        directional=bool(role_query),
        entity_roles=entity_roles,
        operations=operations,
    )
    return RegionResult(
        variable_text=variable,
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
        earliest_index=index,
        restatement=restatement,
        restatement_reason=restatement_reason_for(restatement, reason=reason),
        operations=operations,
    )


def _bind_phrases(
    dropped_qualifier_phrases: list[str] | None, operations: Sequence[Operation]
) -> list[str] | None:
    """Return the leg's dropped phrases followed by its operation phrases, each once.

    Bind renders, matches and validates the two alike. A lone phrase that extract
    records as an operation rather than a dropped qualifier therefore sends bind the
    same prompt, and a `requested_all` claim on it opens the slot it names.
    """
    if not operations:
        return dropped_qualifier_phrases
    return list(
        dict.fromkeys([*(dropped_qualifier_phrases or ()), *(o.phrase for o in operations)])
    )


async def _stream_one_variable(
    v: str,
    i: int,
    *,
    entities: list[str],
    date_request: DateRequest | None,
    query: str,
    pac: bool,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    base_steps: list[PipelineStep],
    base_timing: dict[str, int],
    pre_resolved: dict[str, Resolution] | None,
    deadline: float,
    queue: asyncio.Queue[Event | _RegionDone | _TaskFailed],
    landed: list[RegionResult],
    sem: asyncio.Semaphore | None,
    degrade_on_error: bool,
    include_option_coverage: bool = False,
    stream_slot_options: bool = False,
    include_observations: bool = False,
    stream_observations: bool = False,
    role_query: str | None = None,
    entity_roles: dict[str, list[DoorRole]] | None = None,
    unit_text: str | None = None,
    unit_text_best_effort: bool = False,
    source_text: str | None = None,
    dropped_qualifier_phrases: list[str] | None = None,
    operations: tuple[Operation, ...] = (),
    query_text: str | None = None,
    qualifiers: tuple[CodedQualifier, ...] | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    coverage: CoverageSnapshot,
) -> None:
    """Run one variable leg, queue its events, then send one terminal marker.

    With `degrade_on_error=False`, any error fails the response. With it enabled,
    non-infrastructure errors become `no_data`; `EngineInfraError`, a declared-default
    refusal (`DeclaredDefaultUnsupportedError`) and the conversion-table failures in
    `RATE_TABLE_REFUSALS` always fail the response. The qualifier sidecar is appended to the
    recall/restatement text while `v` stays bare for structured resolution. `entity_roles`
    also supplies the axes written into the leg's restatement. `operations` ride on the final
    region and its restatement, their phrases join `dropped_qualifier_phrases` for this leg's
    bind alone, and the region opens the slots they run over. The final region is appended to
    `landed` before its terminal marker is queued.
    """
    # `v` plus every sidecar CodedQualifier's own phrase, verbatim -- see
    # this function's own qualifiers= comment above for why every
    # user-visible surface below reads this instead of `v`.
    variable_text = recall_text(v, qualifiers)

    async def emit(event: Event) -> None:
        await queue.put(event)

    async def _run() -> RegionResult:
        return await resolve_variable(
            v,
            entities=entities,
            date_request=date_request,
            detect_query=variable_text,
            role_query=role_query if role_query is not None else query,
            entity_roles=entity_roles,
            unit_text=unit_text,
            unit_text_best_effort=unit_text_best_effort,
            source_text=source_text,
            dropped_qualifier_phrases=_bind_phrases(dropped_qualifier_phrases, operations),
            operations=operations,
            query_text=query_text,
            qualifiers=qualifiers,
            pac=pac,
            graph=graph,
            resolver=resolver,
            admission=admission,
            llm=llm,
            sdmx=sdmx,
            base_steps=base_steps,
            base_timing=base_timing,
            pre_resolved=pre_resolved,
            variable_index=i,
            emit=emit,
            coverage=coverage,
            options=EngineOptions(
                deadline=deadline,
                include_option_coverage=include_option_coverage,
                stream_slot_options=stream_slot_options,
                include_observations=include_observations,
                stream_observations=stream_observations,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=rate_cache,
                population_memo=population_memo,
                place_options_pool=place_options_pool,
            ),
        )

    degraded = False
    try:
        if sem is not None:
            async with sem:
                region = await _run()
        else:
            region = await _run()
    except Exception as exc:
        # Infrastructure, declared-default and conversion-table failures refuse the
        # request. A placeholder no_data would read as real missing data.
        if (
            isinstance(
                exc, (EngineInfraError, DeclaredDefaultUnsupportedError, *RATE_TABLE_REFUSALS)
            )
            or not degrade_on_error
        ):
            await queue.put(_TaskFailed(index=i, exc=exc))
            return
        region = _placeholder_no_data(
            variable_text,
            i,
            entities=entities,
            date_request=date_request,
            unit_text=unit_text,
            source_text=source_text,
            role_query=role_query if role_query is not None else query,
            entity_roles=entity_roles,
            operations=operations,
        )
        degraded = True
    else:
        # See RegionResult.restatement -- resolve_variable itself builds no
        # restatement (it has no ParsedPart dependency), so this leg's own
        # echo is attached here, from the exact inputs just passed to _run().
        restatement, reason = _leg_restatement(
            variable_text,
            entities=entities,
            date_request=date_request,
            unit_text=unit_text,
            source_text=source_text,
            directional=_success_directional(
                region,
                role_query=role_query if role_query is not None else query,
                entities=entities,
                entity_roles=entity_roles,
            ),
            entity_roles=entity_roles,
            operations=operations,
        )
        region = dataclasses.replace(
            region,
            variable_text=variable_text,
            restatement=restatement,
            restatement_reason=restatement_reason_for(restatement, reason=reason),
            parsed_form=_leg_parsed_form(
                variable_text,
                entities=entities,
                date_request=date_request,
                unit_text=unit_text,
                source_text=source_text,
                entity_roles=entity_roles,
                operations=operations,
            ),
        )
        region = with_runnable_operations(region, operations)
    # Ownership attaches the instant `region` is final, ahead of the marker.
    landed.append(region)
    await queue.put(_RegionDone(index=i, region=region, degraded=degraded))


@dataclass(frozen=True, slots=True)
class _LegOverrides:
    """The broadcast values and per-index overrides every leg-lookup helper
    below reads, bundled so `_run_variables_streaming` and
    `_launch_variable_tasks` pass one object instead of nine separate
    parameters. Mirrors `_run_variables_streaming`'s own
    `per_variable_*`/broadcast parameters exactly; see that function's own
    docstring for what each override means and its None (broadcast) default.
    """

    unique: list[str]
    entities: list[str]
    date_request: DateRequest | None
    query: str
    pre_resolved: dict[str, Resolution] | None
    entity_roles: dict[str, list[DoorRole]] | None
    per_variable_entities: list[list[str]] | None
    per_variable_date_requests: list[DateRequest | None] | None
    per_variable_role_queries: list[str] | None
    per_variable_entity_roles: list[dict[str, list[DoorRole]]] | None
    per_variable_unit_texts: list[str | None] | None
    unit_text_best_effort: bool
    per_variable_source_texts: list[str | None] | None
    per_variable_qualifiers: list[tuple[CodedQualifier, ...]] | None
    per_variable_pre_resolved: list[dict[str, Resolution]] | None
    per_variable_operations: list[tuple[Operation, ...]] | None


def _leg_entities(overrides: _LegOverrides, i: int) -> list[str]:
    return (
        overrides.per_variable_entities[i]
        if overrides.per_variable_entities is not None
        else overrides.entities
    )


def _leg_date_request(overrides: _LegOverrides, i: int) -> DateRequest | None:
    return (
        overrides.per_variable_date_requests[i]
        if overrides.per_variable_date_requests is not None
        else overrides.date_request
    )


def _leg_unit_text(overrides: _LegOverrides, i: int) -> str | None:
    return (
        overrides.per_variable_unit_texts[i]
        if overrides.per_variable_unit_texts is not None
        else None
    )


def _leg_source_text(overrides: _LegOverrides, i: int) -> str | None:
    return (
        overrides.per_variable_source_texts[i]
        if overrides.per_variable_source_texts is not None
        else None
    )


def _leg_qualifiers(overrides: _LegOverrides, i: int) -> tuple[CodedQualifier, ...] | None:
    return (
        overrides.per_variable_qualifiers[i]
        if overrides.per_variable_qualifiers is not None
        else None
    )


def _leg_operations(overrides: _LegOverrides, i: int) -> tuple[Operation, ...]:
    return (
        overrides.per_variable_operations[i]
        if overrides.per_variable_operations is not None
        else ()
    )


def _leg_pre_resolved(overrides: _LegOverrides, i: int) -> dict[str, Resolution] | None:
    return (
        overrides.per_variable_pre_resolved[i]
        if overrides.per_variable_pre_resolved is not None
        else overrides.pre_resolved
    )


def _leg_variable_text(overrides: _LegOverrides, i: int) -> str:
    """`unique[i]` composed with this leg's own sidecar phrases (see
    `_stream_one_variable`'s own `qualifiers=` comment) -- used for the
    soft-deadline backfill, the one `_placeholder_no_data` caller that never
    reaches `_stream_one_variable`'s own composition.
    """
    return recall_text(overrides.unique[i], _leg_qualifiers(overrides, i))


def _leg_role_query(overrides: _LegOverrides, i: int) -> str:
    override = (
        overrides.per_variable_role_queries[i]
        if overrides.per_variable_role_queries is not None
        else None
    )
    return override if override is not None else overrides.query


def _leg_entity_roles(overrides: _LegOverrides, i: int) -> dict[str, list[DoorRole]] | None:
    if overrides.per_variable_entity_roles is not None:
        return overrides.per_variable_entity_roles[i]
    return overrides.entity_roles


def _launch_variable_tasks(
    unique: list[str],
    overrides: _LegOverrides,
    *,
    pac: bool,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    base_steps: list[PipelineStep],
    base_timing: dict[str, int],
    deadline: float,
    queue: asyncio.Queue,
    landed: list[RegionResult],
    sem: asyncio.Semaphore | None,
    degrade_on_error: bool,
    include_option_coverage: bool,
    stream_slot_options: bool,
    include_observations: bool,
    stream_observations: bool,
    dropped_qualifier_phrases: list[str] | None,
    query_text: str | None,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    population_memo: OptionsPopulationMemo | None,
    place_options_pool: PlaceOptionsPool | None,
    coverage: CoverageSnapshot,
) -> list[asyncio.Task]:
    """Launch one `_stream_one_variable` task per `unique[i]`, applying every
    per-index override `overrides` carries (see `_LegOverrides` and the
    `_leg_*` lookup helpers above).

    `role_query` is passed as the raw per-index override (None when
    unnamed), distinct from `_leg_role_query`'s own fallback to
    `overrides.query`: `_stream_one_variable` applies that identical
    fallback itself when `role_query` is None (see its own docstring), so
    passing the raw override here is not a divergence from the
    soft-deadline backfill's own `_leg_role_query` call.
    """
    return [
        asyncio.create_task(
            _stream_one_variable(
                v,
                i,
                entities=_leg_entities(overrides, i),
                date_request=_leg_date_request(overrides, i),
                query=overrides.query,
                pac=pac,
                graph=graph,
                resolver=resolver,
                admission=admission,
                llm=llm,
                sdmx=sdmx,
                deadline=deadline,
                base_steps=base_steps,
                base_timing=base_timing,
                pre_resolved=_leg_pre_resolved(overrides, i),
                queue=queue,
                landed=landed,
                sem=sem,
                degrade_on_error=degrade_on_error,
                include_option_coverage=include_option_coverage,
                stream_slot_options=stream_slot_options,
                include_observations=include_observations,
                stream_observations=stream_observations,
                role_query=(
                    overrides.per_variable_role_queries[i]
                    if overrides.per_variable_role_queries is not None
                    else None
                ),
                entity_roles=_leg_entity_roles(overrides, i),
                unit_text=_leg_unit_text(overrides, i),
                unit_text_best_effort=overrides.unit_text_best_effort,
                source_text=_leg_source_text(overrides, i),
                qualifiers=_leg_qualifiers(overrides, i),
                dropped_qualifier_phrases=dropped_qualifier_phrases,
                operations=_leg_operations(overrides, i),
                query_text=query_text,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=rate_cache,
                population_memo=population_memo,
                place_options_pool=place_options_pool,
                coverage=coverage,
            )
        )
        for i, v in enumerate(unique)
    ]


@dataclass(slots=True)
class _FanOutProgress:
    """Mutable fold state `_run_variables_streaming`'s queue-draining loop
    threads through `_handle_fanout_item`: which leg indices have landed so
    far, the first hard failure any leg raised, and whether any leg
    degraded rather than failing outright.
    """

    results: dict[int, RegionResult]
    n_pending: int
    first_failure: BaseException | None = None
    degraded_leg: bool = False


def _handle_fanout_item(
    progress: _FanOutProgress, item: Event | _RegionDone | _TaskFailed
) -> Event | None:
    """Fold one queue item into `progress`, or pass an Event through
    unchanged. `_stream_one_variable` has already appended a `_RegionDone`'s
    own region to `landed`, so this fold feeds only the terminal
    `out.regions` assembly at the end of `_run_variables_streaming`.
    """
    if isinstance(item, _RegionDone):
        progress.results[item.index] = item.region
        if item.degraded or item.region.alternates_left_out or item.region.scope_unsettled:
            progress.degraded_leg = True
        progress.n_pending -= 1
        return None
    if isinstance(item, _TaskFailed):
        if progress.first_failure is None:
            progress.first_failure = item.exc
        progress.n_pending -= 1
        return None
    return item


async def _run_variables_streaming(
    unique: list[str],
    *,
    entities: list[str],
    date_request: DateRequest | None,
    query: str,
    pac: bool,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    base_steps: list[PipelineStep],
    base_timing: dict[str, int],
    pre_resolved: dict[str, Resolution] | None,
    deadline: float,
    degrade_on_error: bool,
    use_semaphore: bool,
    out: _FanOutResult,
    include_option_coverage: bool = False,
    stream_slot_options: bool = False,
    include_observations: bool = False,
    stream_observations: bool = False,
    entity_roles: dict[str, list[DoorRole]] | None = None,
    per_variable_entities: list[list[str]] | None = None,
    per_variable_date_requests: list[DateRequest | None] | None = None,
    per_variable_role_queries: list[str] | None = None,
    per_variable_entity_roles: list[dict[str, list[DoorRole]]] | None = None,
    per_variable_unit_texts: list[str | None] | None = None,
    unit_text_best_effort: bool = False,
    per_variable_source_texts: list[str | None] | None = None,
    per_variable_qualifiers: list[tuple[CodedQualifier, ...]] | None = None,
    per_variable_pre_resolved: list[dict[str, Resolution]] | None = None,
    per_variable_operations: list[tuple[Operation, ...]] | None = None,
    dropped_qualifier_phrases: list[str] | None = None,
    query_text: str | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    coverage: CoverageSnapshot,
) -> AsyncIterator[Event]:
    """Run one task per variable and yield each leg's events as they arrive.

    The deadline and caller-initiated close cancel unfinished legs. Any
    abnormal exit also cancels and awaits every trailing task owned by landed
    regions. Cancellation runs in two passes so a pending cancellation during
    the awaits cannot leave later handles running. Normal completion, including
    a soft timeout, returns live handles through `out.regions`.

    Per-variable entity, date, role-query, role, unit, source, qualifier, and
    pre-resolution values override their broadcast counterparts by index.
    `None` broadcasts the shared value. `per_variable_pre_resolved` replaces
    the shared map; parsed-door callers merge pins before passing it. Dropped
    qualifier phrases and request-level dependencies are broadcast unchanged.
    `per_variable_operations` gives each leg its own operations, which ride on its
    region and restatement and join its bind's dropped phrases; `None` gives every
    leg none. `query_text` is supplied only for a single-variable request.
    `unit_text_best_effort` marks every leg's unit text as the raw-text door's typed phrase.
    """
    overrides = _LegOverrides(
        unique=unique,
        entities=entities,
        date_request=date_request,
        query=query,
        pre_resolved=pre_resolved,
        entity_roles=entity_roles,
        per_variable_entities=per_variable_entities,
        per_variable_date_requests=per_variable_date_requests,
        per_variable_role_queries=per_variable_role_queries,
        per_variable_entity_roles=per_variable_entity_roles,
        per_variable_unit_texts=per_variable_unit_texts,
        unit_text_best_effort=unit_text_best_effort,
        per_variable_source_texts=per_variable_source_texts,
        per_variable_qualifiers=per_variable_qualifiers,
        per_variable_pre_resolved=per_variable_pre_resolved,
        per_variable_operations=per_variable_operations,
    )

    queue: asyncio.Queue = asyncio.Queue()
    landed: list[RegionResult] = []
    # The request's single shared semaphore (search.engine.concurrency),
    # not one built here: dataset_resubmit.handle's own identity/candidate
    # promotes draw from the SAME instance, so a mixed request's two tracks
    # share one ceiling instead of each reaching SEARCH_MAX_VARIABLE_CONCURRENCY
    # independently. Fail-closed -- bounded_semaphore() raises when called
    # outside core.resolve_events' own scope, rather than silently running
    # this fan-out unbounded.
    sem = bounded_semaphore() if use_semaphore else None
    tasks = _launch_variable_tasks(
        unique,
        overrides,
        pac=pac,
        graph=graph,
        resolver=resolver,
        admission=admission,
        llm=llm,
        sdmx=sdmx,
        base_steps=base_steps,
        base_timing=base_timing,
        deadline=deadline,
        queue=queue,
        landed=landed,
        sem=sem,
        degrade_on_error=degrade_on_error,
        include_option_coverage=include_option_coverage,
        stream_slot_options=stream_slot_options,
        include_observations=include_observations,
        stream_observations=stream_observations,
        dropped_qualifier_phrases=dropped_qualifier_phrases,
        query_text=query_text,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        coverage=coverage,
    )

    progress = _FanOutProgress(results={}, n_pending=len(tasks))
    timed_out = False

    try:
        try:
            while progress.n_pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                except TimeoutError:
                    timed_out = True
                    break
                ev = _handle_fanout_item(progress, item)
                if ev is not None:
                    yield ev

            # The deadline fired with legs still pending: any leg that finished
            # and enqueued its terminal marker in the instant the deadline fired
            # must not be discarded just because this generator was briefly slow
            # (backpressure), so drain the queue before giving up.
            while progress.n_pending > 0 and not queue.empty():
                ev = _handle_fanout_item(progress, queue.get_nowait())
                if ev is not None:
                    yield ev
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if progress.first_failure is not None:
            # Every leg finishes (or is cancelled above) before this fails loud,
            # matching asyncio.gather(return_exceptions=True)'s own semantics.
            raise progress.first_failure
    except BaseException:
        # `out.regions` is never assigned on this path, so these handles
        # reach no caller. The two passes stay separate: this exception can
        # be, or be shadowed by, this generator's own pending cancellation,
        # and an interleaved cancel-then-await per region gives that
        # cancellation its first landing point inside region 0's await,
        # leaking every later region. `task.cancel()` awaits nothing, so the
        # first pass runs to completion whatever happens to the second.
        await _cancel_trailing_tasks(landed)
        raise

    for i in range(len(unique)):
        if i not in progress.results:
            progress.results[i] = _placeholder_no_data(
                _leg_variable_text(overrides, i),
                i,
                entities=_leg_entities(overrides, i),
                date_request=_leg_date_request(overrides, i),
                unit_text=_leg_unit_text(overrides, i),
                source_text=_leg_source_text(overrides, i),
                role_query=_leg_role_query(overrides, i),
                entity_roles=_leg_entity_roles(overrides, i),
                operations=_leg_operations(overrides, i),
            )
    out.regions = [progress.results[i] for i in range(len(unique))]
    out.timed_out = timed_out
    out.degraded_leg = progress.degraded_leg


async def _resolve_entities_bounded(
    unique_entities: list[str],
    *,
    resolver: EntityResolver,
    deadline: float,
) -> dict[str, Resolution]:
    """Resolve the deduplicated names in one resolver call before fan-out.

    Caller-side limits cap parsed-door input at 96 entities by default
    (`SEARCH_MAX_PARTS * SEARCH_MAX_PART_ENTITIES`) and raw-text input at
    `SEARCH_MAX_ENTITIES`. `_await_bounded` bounds the wait by the shared
    monotonic deadline.
    """
    if not unique_entities:
        return {}

    resolutions = await _await_bounded(resolver.resolve(unique_entities), deadline=deadline)
    return dict(zip(unique_entities, resolutions, strict=True))
