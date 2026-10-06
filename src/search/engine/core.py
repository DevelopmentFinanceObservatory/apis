"""Async Search pipeline orchestration.

``resolve_events`` streams typed events. ``resolve_async`` buffers them into
one response, and ``resolve`` supplies a synchronous wrapper. Per-variable
work from recall through assembly lives in ``region.resolve_variable``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import search.engine.config as config  # module-attribute access: SEARCH_ENGINE_MODEL/SEARCH_LLM_SEED are runtime-resolved
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.build import build_commit
from dfo_platform.executor import submit_with_context
from entity_resolution import Admission, EntityResolver, acquire_shared, release_shared
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import STALE_AFTER, current_catalogue
from search.engine.assemble import (
    build_no_data_part,
    skills_index_incomplete_warning,
    skills_index_unavailable_warning,
)
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.clients.llm import LLM, SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient, LiveSdmxClient
from search.engine.concurrency import scoped_concurrency
from search.engine.entry import dataset_resubmit, parsed, raw_text
from search.engine.entry.params import EntryParams
from search.engine.envelope import _terminal_events, make_diagnostics, make_response, now_ms
from search.engine.events import DoneEvent, Event, ResponseEvent
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region.codes import ORIGIN_CATALOGUE_STALE
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_flow import CacheScope, PartMemoScope
from search.engine.response_cache_memo import PartMemo
from search.models import (
    DatasetResubmitInput,
    NoDataReason,
    ParsedInput,
    QueryEcho,
    RawTextInput,
    Resolution,
    ResolveRequest,
    ResolveResponse,
    Warning,
)
from series_coverage.shared import get_coverage_service
from skills_index import SkillsIndex

logger = logging.getLogger(__name__)


@dataclass
class ResolveHealth:
    """Run health that `resolve_async` reports beside the response, outside its body.

    `degraded_leg` is true when a raw-text leg failed and was backfilled.
    """

    degraded_leg: bool = False


class SkillsIndexSource(Protocol):
    """Anything exposing the current skills index, such as ``SharedSkillsIndex``.

    ``index`` is ``None`` while no build has succeeded.
    """

    @property
    def index(self) -> SkillsIndex | None: ...


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass
class _ResolveRecord:
    """What one resolve's log line reports, filled in as its events pass.

    `cache` is the response cache outcome: `inactive`, `miss`, `hit layer1` or `hit layer2`.
    `memo_hits` and `memo_misses` count a `dataset_resubmit` request's part memo lookups,
    and stay None on every other request and when the memo is off.
    """

    cache: str = "inactive"
    memo_hits: int | None = None
    memo_misses: int | None = None
    response: ResolveResponse | None = None


def _log_resolve(
    request: ResolveRequest, record: _ResolveRecord, *, caller: str | None, transport: str
) -> None:
    """Log one INFO line per resolve, `resolve ` followed by a JSON object.

    The object carries the input kind, the caller's User-Agent, the posted `include_*` flags
    (an unposted flag is absent), the transport (`sse` or `buffered`), the response cache
    outcome, the part memo's hits and misses, the envelope's `timing_ms` total and
    `by_step`, and each part's own `by_step`. Timing is null when the request ended before
    its response event.
    """
    options = request.response_options
    posted = options.model_dump(exclude_none=True) if options is not None else {}
    response = record.response
    timing = response.diagnostics.timing_ms if response is not None else None
    parts = response.parts if isinstance(response, Resolution) else []
    logger.info(
        "resolve %s",
        json.dumps(
            {
                "kind": request.input.kind,
                "caller": caller,
                **{name: value for name, value in posted.items() if name.startswith("include_")},
                "transport": transport,
                "cache": record.cache,
                "memo_hits": record.memo_hits,
                "memo_misses": record.memo_misses,
                "total_ms": timing.total if timing is not None else None,
                "by_step": timing.by_step if timing is not None else None,
                "parts_by_step": [
                    part.timing_ms.by_step if part.timing_ms is not None else None for part in parts
                ],
            }
        ),
    )


def _build_live_graph_client() -> LiveGraphClient:
    """Build a LiveGraphClient from the process's SearchSettings.

    Used only when resolve_events/resolve_async is called with no injected
    graph client (a script or test running outside search_surface()'s
    lifespan, which builds and injects its own shared instance instead).
    """
    return LiveGraphClient.from_search_settings(config.settings())


async def resolve_events(
    request: ResolveRequest,
    *,
    graph: EngineGraphClient | None = None,
    resolver: EntityResolver | None = None,
    admission: Admission | None = None,
    llm: SupportsLLM | None = None,
    sdmx: EngineSdmxClient | None = None,
    stream_slot_options: bool = False,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    shared_skills_index: SkillsIndexSource | None = None,
    response_cache: ResponseCache | None = None,
    part_memo: PartMemo | None = None,
    generation: GenerationProvider = UNKNOWN_GENERATION,
    caller: str | None = None,
) -> AsyncIterator[Event]:
    """Stream typed events for one resolve request.

    Success ends with exactly one ``response`` event followed by ``done``.
    Failures propagate to route-level sanitisation. ``resolve_async`` drains the
    stream and returns the same response.

    Missing graph, LLM, or SDMX clients are built here and closed on exit.
    Default resolvers and admissions are request-scoped. Production injects
    shared instances from the app lifespan. A default resolver acquires and
    releases shared resolvekit. A default admission owns its per-instance cache.
    One ``RequestScope`` spans the pipeline, sharing the concurrency limit
    and cancelling and awaiting trailing tasks on exit.

    ``stream_slot_options`` enables trailing ``slot_options`` and
    ``entity_options`` events after the response for raw-text SSE requests.
    The buffered ``resolve_async`` path leaves it off.

    The app lifespan supplies ``materialise_graph``, ``materialise_sdmx``,
    ``rate_cache``, ``population_memo``, ``place_options_pool``, and
    ``shared_skills_index``. Missing materialise clients reach the preview's
    fail-open guard. A missing rate cache uses handler defaults. A missing
    population memo bypasses memo lookup. A missing place-options pool skips
    those tasks. A missing skills snapshot emits ``skills_index_unavailable``
    when requested.

    ``response_cache=None`` disables caching. Raw-text and parsed requests whose
    transport streams no trailing events use its per-request ``CacheScope``, built
    after ``max_candidates`` and PAC are resolved. ``part_memo=None`` disables the
    part memo a ``dataset_resubmit`` request's identity entries read through
    ``PartMemoScope``. ``generation`` supplies each scope's one graph stamp. The
    default stamp has key ``unknown`` and a short lifetime.

    The request reads rate tables through one ``RateTableCache.for_request``
    view, so every spec it converts cites one version of each table.

    The request ends with one ``resolve`` log line (``_log_resolve``) naming
    ``caller``, the client's User-Agent. Its transport is ``sse`` when
    ``stream_slot_options`` is set, which only the SSE route does, and
    ``buffered`` otherwise.
    """
    start_ms = now_ms()
    record = _ResolveRecord()
    request_rate_cache = rate_cache.for_request() if rate_cache is not None else None
    owns_graph = graph is None
    _graph = graph if graph is not None else _build_live_graph_client()
    _llm = llm or LLM()
    owns_sdmx = sdmx is None
    _sdmx = sdmx if sdmx is not None else LiveSdmxClient()
    if resolver is not None:
        _resolver = resolver
        _shared_resolvekit = None
    else:
        _shared_resolvekit = acquire_shared()
        _resolver = EntityResolver(graph=_graph, shared=_shared_resolvekit, generation=generation)
    _admission = (
        admission if admission is not None else Admission(graph=_graph, generation=generation)
    )

    try:
        async with scoped_concurrency():
            async for event in _resolve_pipeline_events(
                request,
                graph=_graph,
                resolver=_resolver,
                admission=_admission,
                llm=_llm,
                sdmx=_sdmx,
                start_ms=start_ms,
                stream_slot_options=stream_slot_options,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=request_rate_cache,
                population_memo=population_memo,
                place_options_pool=place_options_pool,
                shared_skills_index=shared_skills_index,
                response_cache=response_cache,
                part_memo=part_memo,
                generation=generation,
                record=record,
            ):
                event = _with_stale_catalogue_warning(event)
                if isinstance(event, ResponseEvent):
                    record.response = event.response
                yield event
    finally:
        _log_resolve(
            request,
            record,
            caller=caller,
            transport="sse" if stream_slot_options else "buffered",
        )
        if owns_graph and hasattr(_graph, "close"):
            _graph.close()  # ty: ignore[call-non-callable]  # hasattr
        if owns_sdmx and hasattr(_sdmx, "close"):
            _sdmx.close()  # ty: ignore[call-non-callable]  # hasattr
        if _shared_resolvekit is not None:
            release_shared(_shared_resolvekit)


def _with_stale_catalogue_warning(event: Event) -> Event:
    """A response event carrying `origin_catalogue_stale` in its diagnostics when the bound
    catalogue is stale. Any other event passes through unchanged. It runs after the response
    cache, so a cached answer reports the catalogue's age when it is served."""
    if not isinstance(event, ResponseEvent):
        return event
    catalogue = current_catalogue()
    if not catalogue.is_stale():
        return event
    warning = Warning(
        code=ORIGIN_CATALOGUE_STALE,
        severity="warn",
        message=(
            f"The origin catalogue is more than {STALE_AFTER.days} days old: this answer "
            f"admits data by build {catalogue.build_id!r}, built "
            f"{catalogue.built_at.date().isoformat()}."
        ),
    )
    diagnostics = event.response.diagnostics
    return ResponseEvent(
        response=event.response.model_copy(
            update={
                "diagnostics": diagnostics.model_copy(
                    update={"warnings": [*diagnostics.warnings, warning]}
                )
            }
        )
    )


async def _resolve_pipeline_events(
    request: ResolveRequest,
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    start_ms: int,
    stream_slot_options: bool = False,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    shared_skills_index: SkillsIndexSource | None = None,
    response_cache: ResponseCache | None,
    part_memo: PartMemo | None = None,
    generation: GenerationProvider = UNKNOWN_GENERATION,
    record: _ResolveRecord,
) -> AsyncIterator[Event]:
    """Compute shared request settings, then dispatch by input kind.

    SDMX, materialise clients, and the rate cache reach entry handlers and
    variable fan-out. Resolvers and admissions handle named entities.
    Resubmitted identities use posted dcids and bypass recall.

    The response ``CacheScope`` is built after ``max_candidates`` and PAC are
    resolved. The raw-text and parsed branches use it, and the resubmit branch reads
    the part memo through a ``PartMemoScope``. Their outcomes are request-local,
    while ``stats()`` reports cumulative cache counters. Each scope reads one
    generation stamp at construction. Both key on the UTC day.

    Read ``shared_skills_index`` once. Every stage uses that request snapshot.
    """
    inp = request.input

    # One absolute monotonic deadline governs every pipeline stage and thread hop.
    deadline = start_ms / 1000.0 + config.settings().request_timeout_s

    # Per-option coverage probes and entity-option labels are opt-in, for example,
    # "Kenya, 205 obs". Facet options are built either way. Over SSE with this flag set they
    # arrive in trailing events with the slot coverage.
    include_option_coverage = bool(
        request.response_options and request.response_options.include_option_coverage
    )
    _trailing = stream_slot_options and include_option_coverage
    _include_option_coverage = include_option_coverage and not _trailing

    # Previews are opt-in. include_observations records request intent for warnings.
    # True puts previews in trailing SSE events. False embeds them inline.
    include_observations = bool(
        request.response_options and request.response_options.include_observations
    )
    _obs_trailing = stream_slot_options and include_observations

    # Skills matching is opt-in and stays in the response for both transports.
    # Read one index snapshot for the whole request.
    # It is None when disabled, uninjected, or unbuilt.
    include_skills = bool(request.response_options and request.response_options.include_skills)
    skills_index: SkillsIndex | None = (
        shared_skills_index.index if shared_skills_index is not None else None
    )

    # Every door shares the server's candidate cap. A candidates part holds at least two.
    _max_cand = max(2, config.settings().max_candidates)

    # Parsed and raw-text doors use the same server-side PAC setting.
    pac: bool = config.settings().seam_default

    # Share one coverage snapshot between the cache key and downstream readers.
    # Separate reads could straddle an artifact swap and mismatch its index and stamp.
    coverage_snapshot = get_coverage_service().snapshot()

    today = datetime.now(UTC).date().isoformat()

    # The cache key includes the resolved candidate cap and PAC setting. A transport that
    # streams trailing events stores nothing, since its response event lacks them.
    scope = CacheScope(
        response_cache,
        request=request,
        start_ms=start_ms,
        deadline=deadline,
        build_commit=build_commit(),
        model=config.settings().engine_model,
        seed=config.settings().llm_seed,
        today=today,
        max_candidates=_max_cand,
        seam_default=pac,
        generation=generation,
        coverage=coverage_snapshot,
        trailing=_trailing or _obs_trailing,
    )

    params = EntryParams(
        deadline=deadline,
        max_candidates=_max_cand,
        pac=pac,
        stream_option_coverage=_trailing,
        include_option_coverage=_include_option_coverage,
        stream_observations=_obs_trailing,
        include_observations=include_observations,
        include_skills=include_skills,
        skills_index=skills_index,
        coverage=coverage_snapshot,
    )

    if inp.kind == "dataset_resubmit":
        assert isinstance(inp, DatasetResubmitInput)
        memo = PartMemoScope(
            part_memo,
            build_commit=build_commit(),
            today=today,
            coverage=coverage_snapshot,
            generation=generation,
        )
        try:
            async for ev in dataset_resubmit.handle(
                inp,
                graph=graph,
                resolver=resolver,
                admission=admission,
                llm=llm,
                sdmx=sdmx,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                start_ms=start_ms,
                params=params,
                rate_cache=rate_cache,
                population_memo=population_memo,
                place_options_pool=place_options_pool,
                memo=memo,
            ):
                yield ev
        finally:
            if memo.active:
                record.memo_hits, record.memo_misses = memo.hits, memo.misses
        return

    if inp.kind not in ("raw_text", "parsed"):
        # Defensive fallback for an unknown input kind.
        response = _quick_no_data(
            reason="variable_not_resolved",
            query="",
            engine_build=build_commit(),
            start_ms=start_ms,
            include_skills=include_skills,
            skills_index=skills_index,
        )
        for ev in _terminal_events(response, start_ms=start_ms):
            yield ev
        return

    if isinstance(inp, RawTextInput):
        # Key the first lookup by raw query string. A hit skips extraction and binding.
        cached = await scope.lookup_raw(inp.query)
        if cached is not None:
            record.cache = scope.outcome
            response_event, done_event = _terminal_events(
                cached, start_ms=start_ms, timed_out=scope.served_timed_out
            )
            yield response_event
            yield done_event
            return
        events = raw_text.handle(
            inp,
            graph=graph,
            resolver=resolver,
            admission=admission,
            llm=llm,
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            start_ms=start_ms,
            params=params,
            rate_cache=rate_cache,
            population_memo=population_memo,
            place_options_pool=place_options_pool,
            scope=scope,
        )
    else:
        # A parsed request looks up layer 2 by its whole input before any work.
        assert isinstance(inp, ParsedInput)
        events = parsed.handle(
            inp,
            graph=graph,
            resolver=resolver,
            admission=admission,
            llm=llm,
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            start_ms=start_ms,
            params=params,
            rate_cache=rate_cache,
            population_memo=population_memo,
            place_options_pool=place_options_pool,
            scope=scope,
        )
    # Cache the ResponseEvent. CacheScope decides whether it is storable.
    # finally releases any unpublished lease if the request raises.
    try:
        async for ev in events:
            if isinstance(ev, ResponseEvent):
                await scope.store(ev.response)
            elif isinstance(ev, DoneEvent) and scope.degraded_leg:
                ev = ev.model_copy(update={"degraded_leg": True})
            yield ev
    finally:
        await scope.release()
        record.cache = scope.outcome


async def resolve_async(
    request: ResolveRequest,
    *,
    graph: EngineGraphClient | None = None,
    resolver: EntityResolver | None = None,
    admission: Admission | None = None,
    llm: SupportsLLM | None = None,
    sdmx: EngineSdmxClient | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    shared_skills_index: SkillsIndexSource | None = None,
    response_cache: ResponseCache | None = None,
    part_memo: PartMemo | None = None,
    generation: GenerationProvider = UNKNOWN_GENERATION,
    health: ResolveHealth | None = None,
    caller: str | None = None,
) -> ResolveResponse:
    """Drain ``resolve_events`` into one buffered response.

    The ``response`` event precedes ``done``. Copy ``done.timed_out`` into
    diagnostics so buffered callers can detect a partial result. ``health``,
    when given, receives ``done.degraded_leg``, which the body omits. Client
    injection and lifetimes match ``resolve_events``. The buffered path emits
    no trailing SSE events, so ``place_options_pool`` launches no tasks.
    ``caller`` reaches the ``resolve`` log line.

    Returns a ``ResolveResponse``: a ``Resolution`` carries one part per requested
    variable.
    """
    response: ResolveResponse | None = None
    timed_out: bool = False
    async for event in resolve_events(
        request,
        graph=graph,
        resolver=resolver,
        admission=admission,
        llm=llm,
        sdmx=sdmx,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        shared_skills_index=shared_skills_index,
        response_cache=response_cache,
        part_memo=part_memo,
        generation=generation,
        caller=caller,
    ):
        if isinstance(event, ResponseEvent):
            response = event.response
        elif isinstance(event, DoneEvent):
            timed_out = event.timed_out
            if health is not None:
                health.degraded_leg = event.degraded_leg
    assert response is not None, "resolve_events must yield exactly one response event"
    diagnostics = response.diagnostics.model_copy(update={"timed_out": timed_out})
    return response.model_copy(update={"diagnostics": diagnostics})


def _quick_no_data(
    *,
    reason: NoDataReason,
    query: str,
    engine_build: str,
    start_ms: int,
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
) -> Resolution:
    """Build a no_data response without any LLM or graph calls."""
    total_ms = now_ms() - start_ms
    echo = QueryEcho(
        entry_path="raw_text",
        raw_query=query or None,
        normalized_query=None,
        extract_skipped=True,
    )
    skills_warnings = [
        w
        for w in (
            skills_index_unavailable_warning(
                include_skills=include_skills, skills_index=skills_index
            ),
            skills_index_incomplete_warning(
                include_skills=include_skills, skills_index=skills_index
            ),
        )
        if w is not None
    ]
    diag = make_diagnostics(engine_build, skills_warnings, {}, total_ms)
    part = build_no_data_part(
        reason,
        variable_text=query.strip() or None,
        warnings=[],
        include_skills=include_skills,
        skills_index=skills_index,
    )
    return make_response(
        echo, diag, [part], include_skills=include_skills, skills_index=skills_index
    )


def resolve(request: ResolveRequest) -> ResolveResponse:
    """Synchronous wrapper around ``resolve_async``.

    When called inside a running event loop, runs the pipeline in a worker
    thread with its own loop. Use ``resolve_async`` for dependency injection.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(resolve_async(request))
    # Serial offload (one thread, blocks caller), keeping the caller's bound catalogue.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return submit_with_context(pool, lambda: asyncio.run(resolve_async(request))).result()
