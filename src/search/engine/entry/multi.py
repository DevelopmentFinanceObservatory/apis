"""Shared N>=2 entry-path combine and trailing-enrichment stream."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
from collections.abc import AsyncGenerator
from contextlib import aclosing

from dfo_platform.build import build_commit
from dfo_platform.trace import get_trace_id
from search.engine.clients.graph import EngineGraphClient
from search.engine.conjoin import combine_regions
from search.engine.deadline import _await_bounded, _cancel_and_await
from search.engine.enrichment import _reconcile_after_collapse, _stream_slot_options
from search.engine.entry.params import EntryParams
from search.engine.envelope import _sum_usage, now_ms
from search.engine.events import DoneEvent, DoneTelemetry, Event, LateDepartures
from search.engine.region import RegionResult, detect_set_ref
from search.engine.region.option_coverage import _maybe_build_facet_options
from search.engine.region.pipeline import merged_scope_choices
from search.engine.region.result import any_degraded
from search.engine.region.sentence_events import answer_events
from search.engine.response_cache_flow import CacheScope
from search.models import QueryEcho, Warning


async def _stream_combined_regions(
    regions: list[RegionResult],
    *,
    initial_llm_usage: dict | None,
    query: str,
    extra_warnings: list[Warning],
    echo: QueryEcho | None,
    timed_out: bool,
    graph: EngineGraphClient,
    start_ms: int,
    params: EntryParams,
    scope: CacheScope | None = None,
) -> AsyncGenerator[Event]:
    """Combine regions and stream trailing enrichment.

    `timed_out` and the regions' health (`any_degraded`) are the request's cache health.
    `scope` records it, and the `DoneEvent` carries `degraded_leg` to the buffered route's
    header. `is_storable`
    decides cache eligibility from this report and the response.
    """
    degraded_leg = any_degraded(regions)
    if scope is not None:
        scope.note_health(timed_out=timed_out, degraded_leg=degraded_leg)
    regions = [dataclasses.replace(region, earliest_index=i) for i, region in enumerate(regions)]
    streaming_regions = regions
    try:
        total_usage = initial_llm_usage
        for region in regions:
            total_usage = _sum_usage(total_usage, region.llm_usage)

        combine_call = asyncio.to_thread(
            combine_regions,
            regions,
            query=query,
            extra_warnings=extra_warnings,
            start_ms=start_ms,
            engine_build=build_commit(),
            max_candidates=params.max_candidates,
            include_observations=params.include_observations,
            include_skills=params.include_skills,
            skills_index=params.skills_index,
            llm_usage=total_usage,
            echo=echo,
            set_ref_for=lambda dcids: detect_set_ref(value_dcids=dcids, graph=graph),
            facet_options_for=functools.partial(
                _maybe_build_facet_options,
                graph=graph,
                coverage=params.coverage.index,
                deadline=params.deadline,
            ),
            scope_for=functools.partial(merged_scope_choices, coverage=params.coverage),
        )
        # A timed-out fan-out still combines completed regions into its partial answer.
        if timed_out:
            response, effective = await combine_call
        else:
            response, effective = await _await_bounded(combine_call, deadline=params.deadline)

        streaming_regions, stale_tasks = _reconcile_after_collapse(regions, effective)
        await asyncio.gather(
            *(_cancel_and_await(task) for task in stale_tasks), return_exceptions=True
        )

        for event in answer_events(response):
            yield event
        warnings: list[Warning] = []
        late_departures: list[LateDepartures] = []
        incomplete: list[str] = []
        async with aclosing(
            _stream_slot_options(
                streaming_regions,
                deadline=params.deadline,
                warnings_out=warnings,
                late_departures_out=late_departures,
                incomplete_out=incomplete,
                response=response,
                answered=regions,
            )
        ) as events:
            async for event in events:
                yield event
        if scope is not None:
            scope.note_health(trailing_incomplete=bool(incomplete))
        yield DoneEvent(
            elapsed_ms=now_ms() - start_ms,
            timed_out=timed_out,
            degraded_leg=degraded_leg,
            trace_id=get_trace_id(),
            telemetry=DoneTelemetry(llm_usage=total_usage),
            warnings=warnings,
            late_departures=late_departures,
        )
    finally:
        await asyncio.gather(
            *(_cancel_and_await(region.slot_options_task) for region in streaming_regions),
            *(_cancel_and_await(region.place_options_task) for region in streaming_regions),
            *(_cancel_and_await(region.observations_task) for region in streaming_regions),
            return_exceptions=True,
        )
