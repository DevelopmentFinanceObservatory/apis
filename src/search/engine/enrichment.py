"""Stream trailing enrichment and reconcile its tasks after region collapse.

A streamed definite part ships before its observations are read, so when the read changes
its departures, its complete list (`late_departures`) travels on `done` beside the trailing
warnings. A folded part shares its representative's Spec and read, but its list is rebuilt
from its own inputs, so each part's late list follows the rules its buffered list did.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass

from search.engine.assemble import definite_departures
from search.engine.departures import NO_ENTITY_WORDS, surviving_facts
from search.engine.errors import GraphInfraError
from search.engine.events import Event, LateDepartures, ObservationsEvent, PlaceOptionsEvent
from search.engine.region import (
    OBSERVATIONS_TIMED_OUT,
    OPTION_COVERAGE_TIMED_OUT,
    RegionResult,
    _ObsProbeResult,
    _OptionProbeResult,
)
from search.engine.region.un_system import refused_places, refused_places_fact
from search.models import DefinitePart, Resolution, Warning, omitted_observations

logger = logging.getLogger(__name__)

PLACE_OPTIONS_BUDGET_S: float = 1.0
"""Maximum time to await place-options work after trailing streaming starts.

The budget is independent of the request deadline. Cold population-memo reads take
2.6 to 5.3 s (3.5 s in a delayed-reader test). Warm reads take about 0.12 s against
the 0.5 s `/options` target. At 1.0 s, an expired task is cancelled and its results
dropped silently. Running executor work continues to fill the memo and the options
cache. Later stages are skipped. Clients can fetch options on demand if this event does
not arrive. A response cache hit streams place options under the same budget
(`stream_place_options`), since its bundle holds none.
"""


def _place_options_task_kind_deadlines(
    regions: Sequence[RegionResult],
    *,
    deadline: float,
    place_options_budget_s: float,
) -> tuple[dict[asyncio.Task, str], dict[asyncio.Task, float]]:
    """Map place-options tasks to the earlier of the request deadline and local budget."""
    own_deadline = min(deadline, time.monotonic() + place_options_budget_s)
    kind: dict[asyncio.Task, str] = {}
    task_deadline: dict[asyncio.Task, float] = {}
    for task in (r.place_options_task for r in regions if r.place_options_task is not None):
        kind[task] = "place_options"
        task_deadline[task] = own_deadline
    return kind, task_deadline


def _launch_trailing_tasks(
    region: RegionResult,
    *,
    launch_slot_options: bool,
    launch_observations: bool,
) -> RegionResult:
    """Launch available trailing tasks on the event loop, preserving live handles.

    Resubmit regions defer task creation because resolution runs in a worker
    thread. Parsed regions may carry live tasks. Reuse their handles and launch
    only missing tasks. Place-options use the slot-options coverage gate.
    """
    if (
        launch_slot_options
        and region.slot_options_relaunch is not None
        and region.slot_options_task is None
    ):
        region = dataclasses.replace(
            region,
            slot_options_task=region.slot_options_relaunch(
                region.spec.spec_id, region.spec.resolved.slot_options, region.facet_probe
            ),
        )
    if (
        launch_slot_options
        and region.place_options_relaunch is not None
        and region.place_options_task is None
    ):
        region = dataclasses.replace(
            region,
            place_options_task=region.place_options_relaunch(region.spec),
        )
    if (
        launch_observations
        and region.observations_relaunch is not None
        and region.observations_task is None
    ):
        region = dataclasses.replace(
            region,
            observations_task=region.observations_relaunch(region.spec.spec_id),
        )
    return region


async def _stream_slot_options(
    regions: list[RegionResult],
    *,
    deadline: float,
    warnings_out: list[Warning],
    late_departures_out: list[LateDepartures],
    incomplete_out: list[str],
    response: Resolution | None,
    answered: Sequence[RegionResult],
    place_options_budget_s: float = PLACE_OPTIONS_BUDGET_S,
) -> AsyncIterator[Event]:
    """Stream trailing enrichment events as slot, observation, or place tasks finish.

    Slot and observation tasks share the request's monotonic deadline. Place tasks
    use the earlier of that deadline and `place_options_budget_s`. Expired tasks
    stop blocking siblings. Slot and observation timeouts add warnings, while
    place timeouts drop without one. The request's `done.timed_out` still reflects
    only the primary fan-out.

    Every slot-options or observations task whose result did not reach the stream in full,
    through a timeout or a graph failure, appends its kind to `incomplete_out`. The caller
    reports it to the response cache, which then stores no bundle. A place-options task
    appends nothing, since the bundle holds no place options. The list stays inside the
    process.

    The `finally` cancels and awaits unfinished tasks, including on client disconnect.
    Executor work already running may continue, but its events are dropped. Probe
    warnings are appended to `warnings_out`, and the complete departures of each definite
    part of `response` a finished observations read changes (`late_departures`) to
    `late_departures_out`, for the caller to include in `done`. `answered` holds the
    region each part of `response` was built from, in part order; a `None` response
    (a caller with no parts to restate) collects no late departures.
    """
    task_kind, task_deadline = _place_options_task_kind_deadlines(
        regions,
        deadline=deadline,
        place_options_budget_s=place_options_budget_s,
    )
    # Map observation tasks to Spec ids for timeout events. Synthetic test regions
    # may lack specs. Omit their event but retain the aggregate timeout warning.
    obs_task_spec_id: dict[asyncio.Task, str] = {}
    sources = part_sources(response, answered=answered, streaming=regions)
    obs_task_parts: dict[asyncio.Task, list[PartSource]] = {}
    for r in regions:
        if r.slot_options_task is not None:
            task_kind[r.slot_options_task] = "slot_options"
            task_deadline[r.slot_options_task] = deadline
        if r.observations_task is not None:
            task_kind[r.observations_task] = "observations"
            task_deadline[r.observations_task] = deadline
            if r.specs:
                obs_task_spec_id[r.observations_task] = r.spec.spec_id
            if parts := [src for src in sources if src.representative is r]:
                obs_task_parts[r.observations_task] = parts
    pending = set(task_kind)
    if not pending:
        return
    all_tasks = set(pending)
    try:
        while pending:
            now = time.monotonic()
            next_deadline = min(task_deadline[t] for t in pending)
            remaining = next_deadline - now
            if remaining <= 0:
                # Expire only due tasks. Siblings with later deadlines keep running.
                expired = {t for t in pending if task_deadline[t] <= now}
                pending -= expired
                for t in expired:
                    if task_kind[t] != "place_options":
                        incomplete_out.append(task_kind[t])
                    if not t.done():
                        t.cancel()
                n_slot = sum(1 for t in expired if task_kind[t] == "slot_options")
                obs_pending = [t for t in expired if task_kind[t] == "observations"]
                if n_slot:
                    warnings_out.append(
                        Warning(
                            code=OPTION_COVERAGE_TIMED_OUT,
                            severity="warn",
                            message=(
                                f"{n_slot} option-coverage probe(s) did not finish before "
                                "the request deadline; cancelled without streaming their events."
                            ),
                        )
                    )
                if obs_pending:
                    # Emit a timeout omission per task before the aggregate warning.
                    # Clients treat a missing observations event as a protocol error.
                    _obs_timeout_message = (
                        "the observation preview did not finish before the request deadline."
                    )
                    for t in obs_pending:
                        _spec_id = obs_task_spec_id.get(t)
                        if _spec_id is None:
                            # Without a Spec id, only the aggregate warning can name this timeout.
                            continue
                        yield ObservationsEvent(
                            spec_id=_spec_id,
                            observations=omitted_observations(
                                OBSERVATIONS_TIMED_OUT, _obs_timeout_message
                            ),
                        )
                    warnings_out.append(
                        Warning(
                            code=OBSERVATIONS_TIMED_OUT,
                            severity="warn",
                            message=(
                                f"{len(obs_pending)} observations fetch(es) did not finish "
                                "before the request deadline; cancelled without streaming "
                                "their events."
                            ),
                        )
                    )
                continue
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if task_kind[t] == "place_options":
                    # Probe-builder refusals are handled internally. Unexpected errors reach the route.
                    for ev in t.result():
                        yield ev
                    continue
                try:
                    events, warns = t.result()
                except GraphInfraError:
                    # A graph failure drops this probe. Unexpected errors propagate
                    # to the route as a terminal failure. The drop is reported through
                    # `incomplete_out`, so the response cache stores no bundle without it.
                    logger.warning("trailing enrichment probe task failed", exc_info=True)
                    incomplete_out.append(task_kind[t])
                    continue
                warnings_out.extend(warns)
                late_departures_out.extend(
                    late
                    for source in obs_task_parts.get(t, ())
                    if (late := late_departures(source, events=events, warnings=warns)) is not None
                )
                for ev in events:
                    yield ev
    finally:
        for t in all_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)


async def stream_place_options(
    tasks: Sequence[asyncio.Task[list[PlaceOptionsEvent]]],
    *,
    deadline: float,
    place_options_budget_s: float = PLACE_OPTIONS_BUDGET_S,
) -> AsyncGenerator[PlaceOptionsEvent]:
    """Stream the events of place-options `tasks` as each finishes, as `_stream_slot_options`
    streams a fresh run's, until the earlier of `deadline` and `place_options_budget_s`
    from now. Tasks still running then are dropped silently, and the `finally` cancels and
    awaits every unfinished task, including on client disconnect."""
    own_deadline = min(deadline, time.monotonic() + place_options_budget_s)
    pending = set(tasks)
    try:
        while pending and (remaining := own_deadline - time.monotonic()) > 0:
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                for ev in t.result():
                    yield ev
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@dataclass(frozen=True)
class PartSource:
    """A definite part of a response with the inputs its departures were built from that
    the part does not carry: the region it was built from (`region`, its own raised facts
    and entity words), the region whose Spec it answers with (`representative`, whose
    observations task reads it, and whose raised facts join the part's as
    `conjoin._build_part` joins them) and the response's envelope warnings."""

    index: int
    part: DefinitePart
    region: RegionResult
    representative: RegionResult
    envelope_warnings: Sequence[Warning]


def part_sources(
    response: Resolution | None,
    *,
    answered: Sequence[RegionResult],
    streaming: Sequence[RegionResult],
) -> list[PartSource]:
    """The definite parts of `response`, each with its source regions.

    `answered[i]` is the region part `i` was built from. Its representative is the
    streaming region that took its index, or, for a part folded into an earlier one, the
    streaming region that answers with the part's Spec."""
    if response is None:
        return []
    by_index = {r.earliest_index: r for r in streaming if r.status == "definite"}
    by_spec_id = {r.spec.spec_id: r for r in streaming if r.status == "definite" and r.specs}
    sources: list[PartSource] = []
    for index, (part, region) in enumerate(zip(response.parts, answered, strict=True)):
        if not isinstance(part, DefinitePart):
            continue
        representative = by_index.get(index) or by_spec_id.get(part.spec.spec_id)
        if representative is None:
            continue
        sources.append(
            PartSource(
                index=index,
                part=part,
                region=region,
                representative=representative,
                envelope_warnings=response.diagnostics.warnings,
            )
        )
    return sources


def late_departures(
    source: PartSource,
    *,
    events: Sequence[Event],
    warnings: Sequence[Warning],
) -> LateDepartures | None:
    """The complete `departures` of the definite part `source` once its streamed
    observations read is known, the `events` and `warnings` its representative's task
    returned, or `None` when they equal the list the part shipped.

    The list is built as the part built its own (`assemble.definite_departures`), from its
    own heading words, warnings and operations and the facts raised beside the warnings the
    part or its envelope holds, so a folded part keeps the list of a folded part. The read's
    observations join the Spec, so `modelled_values` reads modelled and projected series,
    and places the UN front door refused during the read join as raised
    `entity_unavailable` facts (`un_system.refused_places_fact`), merged into one per role
    with those the part states. A client replaces the part's list with it."""
    part = source.part
    spec = part.spec
    observed = next(
        (
            event.observations
            for event in events
            if isinstance(event, ObservationsEvent) and event.spec_id == spec.spec_id
        ),
        None,
    )
    late_warnings = [warning for warning in warnings if warning.spec_id == spec.spec_id]
    entity_words = source.region.entity_words
    words = entity_words if entity_words is not None else NO_ENTITY_WORDS
    refused = [
        fact
        for warning in late_warnings
        if (
            fact := refused_places_fact(
                (spec,), refused=refused_places(warning), entity_words=words
            )
        )
        is not None
    ]
    if observed is None and not refused:
        return None
    raised = surviving_facts(
        {**source.region.departure_facts, **source.representative.departure_facts},
        [*part.warnings, *source.envelope_warnings],
    )
    read = (
        spec
        if observed is None
        else spec.model_copy(
            update={"resolved": spec.resolved.model_copy(update={"observations": observed})}
        )
    )
    learned = definite_departures(
        read,
        variable_text=part.variable_text,
        warnings=[*part.warnings, *late_warnings],
        raised=[*raised, *refused],
        operations=part.operations,
        entity_words=entity_words,
    )
    if learned == part.departures:
        return None
    return LateDepartures(part_index=source.index, spec_id=spec.spec_id, departures=learned)


def _reconcile_after_collapse(
    original: list[RegionResult],
    effective: list[RegionResult],
) -> tuple[list[RegionResult], list[asyncio.Task]]:
    """Replace stale probe tasks after region merges or drops.

    Untouched regions retain identity. Merges create a new `RegionResult`. The
    combiner clears slot- and place-options tasks on merged regions, so their
    surviving relaunch closures signal a merge. It copies observation tasks
    without clearing them, so object identity signals when to relaunch those.
    Return effective regions with fresh tasks and old tasks the caller must cancel.
    """
    original_ids = {id(r) for r in original}
    kept = {id(r) for r in effective}
    stale_tasks: list[
        asyncio.Task[_OptionProbeResult | _ObsProbeResult | list[PlaceOptionsEvent]]
    ] = [
        r.slot_options_task
        for r in original
        if r.slot_options_task is not None and id(r) not in kept
    ]
    stale_tasks.extend(
        r.observations_task
        for r in original
        if r.observations_task is not None and id(r) not in kept
    )
    stale_tasks.extend(
        r.place_options_task
        for r in original
        if r.place_options_task is not None and id(r) not in kept
    )

    streaming_regions: list[RegionResult] = []
    for r in effective:
        updates: dict = {}
        if r.slot_options_task is None and r.slot_options_relaunch is not None:
            updates["slot_options_task"] = r.slot_options_relaunch(
                r.spec.spec_id, r.spec.resolved.slot_options, r.facet_probe
            )
        if r.place_options_task is None and r.place_options_relaunch is not None:
            updates["place_options_task"] = r.place_options_relaunch(r.spec)
        if id(r) not in original_ids and r.observations_relaunch is not None:
            updates["observations_task"] = r.observations_relaunch(r.spec.spec_id)
        streaming_regions.append(dataclasses.replace(r, **updates) if updates else r)

    return streaming_regions, stale_tasks
