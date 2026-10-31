"""Async Search pipeline orchestration.

``resolve_events`` streams typed events. ``resolve_async`` buffers them into
one response, and ``resolve`` supplies a synchronous wrapper. Per-variable
work from recall through assembly lives in ``region.resolve_variable``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import search.engine.config as config  # module-attribute access: SEARCH_ENGINE_MODEL/SEARCH_LLM_SEED are runtime-resolved
from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform import refusal
from dfo_platform.build import build_commit
from dfo_platform.executor import submit_with_context
from dfo_platform.trace import get_trace_id
from entity_resolution import Admission, EntityResolver, acquire_shared, release_shared
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import STALE_AFTER, current_catalogue
from origin_catalogue.service import CatalogueSettings
from search.engine.artifact_checks import index_drift_warning, pair_stale_warning, with_warning
from search.engine.assemble import (
    build_no_data_part,
    skills_index_incomplete_warning,
    skills_index_unavailable_warning,
)
from search.engine.binders import Binder, Fallbacks, configured_binder, recording_fallbacks
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.clients.llm import LLM, SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient, LiveSdmxClient
from search.engine.concurrency import scoped_concurrency
from search.engine.entry import dataset_resubmit, parsed, raw_text
from search.engine.entry.params import EntryParams
from search.engine.envelope import (
    _terminal_events,
    make_diagnostics,
    make_response,
    now_ms,
    with_step_timing,
)
from search.engine.errors import served_problem
from search.engine.events import DoneEvent, Event, ResponseEvent
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region.codes import ORIGIN_CATALOGUE_STALE
from search.engine.region.place_options import PlaceOptionsPool, replayed_place_options
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_flow import CacheScope, PartMemoScope
from search.engine.response_cache_keys import normalise_query
from search.engine.response_cache_memo import PartMemo
from search.engine.retrieval.bound_values import held_family_memo
from search.models import (
    DatasetResubmitInput,
    NoDataPart,
    NoDataReason,
    ParsedInput,
    QueryEcho,
    RawTextInput,
    Resolution,
    ResolveInput,
    ResolveRequest,
    ResolveResponse,
    UnsupportedPart,
    Warning,
)
from series_coverage.shared import CoverageSnapshot, get_coverage_service
from skills_index import SkillsIndex

logger = logging.getLogger(__name__)


@dataclass
class ResolveHealth:
    """Run health that `resolve_async` reports beside the response, outside its body.

    `degraded_leg` is the `done` event's: a leg of the answer stands on less than a later
    request may read (`RegionResult.leg_degraded`).
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
    `memo_hits` and `memo_misses` count a `dataset_resubmit` request's part memo lookups, and
    stay None on every other request and when the memo is off. `response` and `done` are the
    request's terminal events, None when it ended before them. `error_code` is how a request
    that did not run to its end ended (`_note_ending`), None for one that did.
    """

    cache: str = "inactive"
    memo_hits: int | None = None
    memo_misses: int | None = None
    response: ResolveResponse | None = None
    done: DoneEvent | None = None
    error_code: str | None = None


def _note_ending(
    record: _ResolveRecord,
    exc: BaseException,
    *,
    hard_cap: asyncio.Timeout | None,
    sse: bool,
) -> None:
    """Record how `exc` ended the request. It never raises, so a classification failure
    cannot replace `exc`.

    An SSE client holds its answer once `done` is sent, so nothing after `done` ends the
    request. A buffered client holds nothing until the route returns, so a failure or the hard
    cap after `done` still decides what it receives.

    A failure reads the problem code the route serves for it. Cancellation reads `timeout`
    when the hard cap's own timer is the one cancel the task holds, as `asyncio.timeout`
    decides at its exit, and `cancelled` otherwise, such as a client disconnect, even one
    still unwinding when the cap fires.
    """
    if sse and record.done is not None:
        return
    try:
        if isinstance(exc, Exception):
            record.error_code = served_problem(exc).code
        elif _hard_cap_fired_alone(hard_cap):
            record.error_code = refusal.TIMEOUT_DEFAULT.code
        else:
            record.error_code = "cancelled"
    except Exception:
        logger.exception("classifying %s for the resolve log line failed", type(exc).__name__)


def _hard_cap_fired_alone(hard_cap: asyncio.Timeout | None) -> bool:
    """Whether the cap expired and no other cancellation is pending on this task."""
    task = asyncio.current_task()
    return (
        hard_cap is not None and hard_cap.expired() and task is not None and task.cancelling() <= 1
    )


def _query_hmac(inp: ResolveInput) -> str | None:
    """The keyed digest of a raw_text query's cache text with case folded, None for other input
    kinds and when `SEARCH_QUERY_HMAC_KEY` is unset."""
    key = config.settings().query_hmac_key.get_secret_value()
    if not key or not isinstance(inp, RawTextInput):
        return None
    text = normalise_query(inp.query).casefold()
    return hmac.new(key.encode(), text.encode(), hashlib.sha256).hexdigest()


def _log_resolve(
    request: ResolveRequest,
    record: _ResolveRecord,
    *,
    caller: str | None,
    transport: str,
) -> None:
    """Log one INFO line per resolve that reaches `resolve_events`, `resolve ` followed by a
    JSON object. A refusal at the route, ahead of it, logs none.

    The object carries the request's `trace_id`, the input kind, the caller's User-Agent, the
    posted `include_*` flags (an unposted flag is absent), the transport (`sse` or
    `buffered`), the response cache outcome, the part memo's hits and misses, the envelope's
    `timing_ms` total and `by_step`, and each part's own `by_step`.

    `outcome` is the response's `status` for a resolution and its `kind` otherwise.
    `part_statuses` lists each part's status in order, `no_data_reasons` each no_data part's
    `reason` and `unsupported_codes` each unsupported part's `code`. `llm_usage` is
    `diagnostics.llm_usage`, so it leaves out tokens spent by cut or aborted legs.

    `error_code` says how a request that did not run to its end ended (`_note_ending`): a
    problem code, or `cancelled`. `timeout` covers the hard cap and any `TimeoutError` that
    fails the request, such as a soft deadline an extraction or a `dataset_resubmit` parsed
    entry crossed. `timed_out` is true for a `timeout`, else the `done` event's flag, null
    before `done`. `degraded_leg` is the `done` event's flag, null before `done`. A leg cut at
    the soft deadline is backfilled as
    `no_data`/`timed_out`. A leg whose error was swallowed is backfilled as
    `no_data`/`variable_not_resolved`, so only `degraded_leg` tells it from a real miss.

    `query_hmac` is the HMAC-SHA-256 hex of `normalise_query(query).casefold()` for a raw_text
    query, the text the response cache keys on with case folded, under `SEARCH_QUERY_HMAC_KEY`.
    It is null for other input kinds and when the key is unset. It counts repeats within a
    deployment: digests do not compare across deployments, guessing a query offline needs the
    key, and rotating the key breaks repeat counts across the rotation. The line carries no
    query text.

    Every response-derived field is null, and every list empty, when the request ended before
    its response event.
    """
    options = request.response_options
    posted = options.model_dump(exclude_none=True) if options is not None else {}
    response = record.response
    timing = response.diagnostics.timing_ms if response is not None else None
    parts = response.parts if isinstance(response, Resolution) else []
    if response is None:
        outcome = None
    elif isinstance(response, Resolution):
        outcome = response.status
    else:
        outcome = response.kind
    inp = request.input
    done = record.done
    timed_out = done.timed_out if done is not None else None
    if record.error_code == refusal.TIMEOUT_DEFAULT.code:
        timed_out = True
    logger.info(
        "resolve %s",
        json.dumps(
            {
                "trace_id": get_trace_id(),
                "kind": inp.kind,
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
                "outcome": outcome,
                "part_statuses": [part.status for part in parts],
                "no_data_reasons": [
                    part.no_data.reason for part in parts if isinstance(part, NoDataPart)
                ],
                "unsupported_codes": [
                    part.unsupported.code for part in parts if isinstance(part, UnsupportedPart)
                ],
                "llm_usage": response.diagnostics.llm_usage if response is not None else None,
                "error_code": record.error_code,
                "timed_out": timed_out,
                "degraded_leg": done.degraded_leg if done is not None else None,
                "query_hmac": _query_hmac(inp),
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class RequestBudget:
    """One resolve request's deadlines, measured from one clock origin.

    ``soft`` is the pipeline's deadline: the fan-out stops waiting for unfinished variables
    and answers with what it has. ``hard`` caps the whole request on both transports and
    bounds every graph and SDMX read it makes (``dfo_platform.request_deadline``), so the
    partial answer has ``finalise_grace_s`` to land before it. Both are absolute
    ``time.monotonic()`` values.
    """

    start_ms: int
    soft: float
    hard: float


def request_budget(*, kind: str) -> RequestBudget:
    """The budget of a resolve request of input ``kind`` that starts now.

    A ``dataset_resubmit`` whose member walk runs its full read reads until
    ``resubmit_full_read_timeout_s`` from the start (``dataset_resubmit.full_read_deadline_for``), so
    its grace runs from then.
    """
    start_ms = now_ms()
    settings = config.settings()
    origin = start_ms / 1000.0
    soft = origin + settings.request_timeout_s
    last_read = soft
    if kind == "dataset_resubmit":
        last_read = origin + settings.resubmit_full_read_timeout_s
    return RequestBudget(start_ms=start_ms, soft=soft, hard=last_read + settings.finalise_grace_s)


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
    budget: RequestBudget | None = None,
    hard_cap: asyncio.Timeout | None = None,
    binder: Binder | None = None,
) -> AsyncGenerator[Event]:
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

    ``response_cache=None`` disables caching. Raw-text and parsed requests use its
    per-request ``CacheScope``, built after ``max_candidates`` and PAC are resolved. It
    stores the event bundle of a request that streams trailing events and the response
    of any other. ``part_memo=None`` disables the part memo a ``dataset_resubmit``
    request's identity entries read through ``PartMemoScope``. ``generation`` supplies
    each scope's one graph stamp. The default stamp has key ``unknown`` and a short
    lifetime.

    The request reads rate tables through one ``RateTableCache.for_request``
    view, so every spec it converts cites one version of each table.

    The request ends with one ``resolve`` log line (``_log_resolve``) naming
    ``caller``, the client's User-Agent. Its transport is ``sse`` when
    ``stream_slot_options`` is set, which only the SSE route does, and
    ``buffered`` otherwise.

    ``budget`` is the route's ``request_budget``, so the pipeline's soft deadline and the
    route's hard cap share one start. Without one the request starts here. ``hard_cap`` is the
    route's running timer, which reaches the pipeline as plain cancellation, so the log line
    asks it who cancelled (``_note_ending``). Without one every cancellation reads
    ``cancelled``.

    ``binder`` reads the quantity and binds every part. ``None`` takes the one the request
    routes to (``binders.configured_binder``). The cache key reads it.
    """
    if budget is None:
        budget = request_budget(kind=request.input.kind)
    start_ms = budget.start_ms
    record = _ResolveRecord()
    owns_graph = graph is None
    _graph = graph
    owns_sdmx = sdmx is None
    _sdmx = sdmx
    _shared_resolvekit = None
    try:
        request_rate_cache = rate_cache.for_request() if rate_cache is not None else None
        _graph = graph if graph is not None else _build_live_graph_client()
        _llm = llm or LLM()
        _sdmx = sdmx if sdmx is not None else LiveSdmxClient()
        if resolver is not None:
            _resolver = resolver
        else:
            _shared_resolvekit = acquire_shared()
            _resolver = EntityResolver(
                graph=_graph, shared=_shared_resolvekit, generation=generation, code_index=get_index
            )
        _admission = (
            admission if admission is not None else Admission(graph=_graph, generation=generation)
        )

        # The request binds one artifact set: the catalogue bound at entry, this coverage
        # snapshot, the process's code index and each cache scope's one generation stamp.
        # Separate coverage reads could straddle an artifact swap and mismatch its index and
        # stamp. Both reads sit inside the `try`, so the clients are released whatever raises.
        # The snapshot's time is a once-per-request step on the envelope's timing.
        coverage_t0 = now_ms()
        coverage = get_coverage_service().snapshot()
        coverage_ms = now_ms() - coverage_t0
        checks = CatalogueSettings()
        async with scoped_concurrency():
            with held_family_memo(), recording_fallbacks() as fallbacks:
                pipeline = _resolve_pipeline_events(
                    request,
                    graph=_graph,
                    resolver=_resolver,
                    admission=_admission,
                    llm=_llm,
                    sdmx=_sdmx,
                    binder=binder,
                    start_ms=start_ms,
                    deadline=budget.soft,
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
                    coverage_snapshot=coverage,
                    warn_index_drift=checks.warn_index_drift,
                    record=record,
                    fallbacks=fallbacks,
                )
                async with aclosing(pipeline):
                    async for event in pipeline:
                        event = _with_request_step(
                            _with_stale_catalogue_warning(event), step="coverage", ms=coverage_ms
                        )
                        if isinstance(event, ResponseEvent):
                            event = _with_response_warning(
                                event,
                                pair_stale_warning(
                                    current_catalogue(),
                                    coverage.manifest,
                                    warn=checks.warn_pair_stale,
                                ),
                            )
                            record.response = event.response
                        elif isinstance(event, DoneEvent):
                            record.done = event
                        yield event
    except BaseException as exc:
        _note_ending(record, exc, hard_cap=hard_cap, sse=stream_slot_options)
        raise
    finally:
        try:
            _log_resolve(
                request,
                record,
                caller=caller,
                transport="sse" if stream_slot_options else "buffered",
            )
        except Exception:
            # A logging failure must not replace the response or skip the cleanup.
            logger.exception("resolve log line failed")
        finally:
            if owns_graph and hasattr(_graph, "close"):
                _graph.close()  # ty: ignore[call-non-callable]  # hasattr
            if owns_sdmx and hasattr(_sdmx, "close"):
                _sdmx.close()  # ty: ignore[call-non-callable]  # hasattr
            if _shared_resolvekit is not None:
                release_shared(_shared_resolvekit)


def _with_request_step(event: Event, *, step: str, ms: int) -> Event:
    """A response event whose envelope `timing_ms.by_step` starts with `step` at `ms`, for a
    step that runs once per request before the pipeline. It runs after the response cache, so
    a cached answer reports this request's time. Any other event passes through unchanged."""
    if not isinstance(event, ResponseEvent):
        return event
    diagnostics = event.response.diagnostics
    timing = diagnostics.timing_ms
    if timing is None:
        return event
    stepped = timing.model_copy(update={"by_step": {step: ms, **(timing.by_step or {})}})
    return ResponseEvent(
        response=event.response.model_copy(
            update={"diagnostics": diagnostics.model_copy(update={"timing_ms": stepped})}
        )
    )


def _routing_text(inp: ResolveInput) -> str:
    """The text a request is routed to its binder on, which is what resolution reads: a
    raw-text request's query, else the canonical JSON of a parsed or resubmitted request's
    input less `raw_query`, which is for echo only."""
    if isinstance(inp, RawTextInput):
        return inp.query
    return json.dumps(
        inp.model_dump(mode="json", exclude={"raw_query"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _with_response_warning(event: ResponseEvent, warning: Warning | None) -> ResponseEvent:
    """`event` with `warning` appended to its response's diagnostics, or `event` itself
    when `warning` is `None`."""
    if warning is None:
        return event
    return ResponseEvent(response=with_warning(event.response, warning))


def _with_index_drift_warning(event: Event, *, warn: bool) -> Event:
    """A response event carrying `catalogue_index_drift` when the bound catalogue's arcs
    and the code index disagree on a StatVar it serves (`index_drift_warning`). It runs
    before the response cache stores the answer: the cache key holds the catalogue build
    and the build commit that packages the code index, so the warning holds for every
    answer served under that key. A cached answer's Specs state no members, so it is
    checked once, when computed."""
    if not isinstance(event, ResponseEvent):
        return event
    return _with_response_warning(
        event,
        index_drift_warning(
            event.response, catalogue=current_catalogue(), index=get_index(), warn=warn
        ),
    )


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
    return _with_response_warning(event, warning)


async def _resolve_pipeline_events(
    request: ResolveRequest,
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    binder: Binder | None,
    start_ms: int,
    deadline: float,
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
    coverage_snapshot: CoverageSnapshot,
    warn_index_drift: bool,
    record: _ResolveRecord,
    fallbacks: Fallbacks,
) -> AsyncGenerator[Event]:
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

    ``deadline``, the request's soft deadline, governs every pipeline stage and thread hop.
    ``coverage_snapshot`` is the request's one coverage snapshot, shared by the cache key and every
    reader. ``warn_index_drift`` is `CatalogueSettings.warn_index_drift`.

    The request takes one binder here: ``binder`` when the caller injects one, else the one
    it routes to on what resolution reads of it (`_routing_text`, `binders.configured_binder`).
    Every door's parts read their quantity and bind through it. The response cache keys on
    it, and ``fallbacks`` records each call it served from its fallback, which keeps the
    response out of the cache.
    """
    inp = request.input

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

    today = datetime.now(UTC).date().isoformat()

    # The request's binder: the caller's when it injects one, else the one the request
    # routes to. The cache key reads it and every part binds with it (`EntryParams.binder`).
    binder = binder if binder is not None else configured_binder(_routing_text(inp))

    # The cache key includes the resolved candidate cap and PAC setting. A transport that
    # streams trailing events stores the events from `response` to `done`, since the
    # response event lacks the trailing ones.
    scope = CacheScope(
        response_cache,
        request=request,
        start_ms=start_ms,
        deadline=deadline,
        build_commit=build_commit(),
        binder=binder,
        fallbacks=fallbacks,
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
        binder=binder,
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
        resubmitted = dataset_resubmit.handle(
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
        )
        try:
            async with aclosing(resubmitted):
                async for ev in resubmitted:
                    yield _with_index_drift_warning(ev, warn=warn_index_drift)
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

    # CacheScope.observe stores the answer and decides whether it is storable.
    # finally releases any unpublished lease if the request raises.
    lookup_ms: int | None = None
    try:
        if isinstance(inp, RawTextInput):
            # Key the first lookup by raw query string. A hit skips extraction and binding.
            lookup_t0 = now_ms()
            cached = await scope.lookup_raw(inp.query)
            lookup_ms = now_ms() - lookup_t0
            if cached is not None:
                async with aclosing(
                    scope.hit_events(
                        cached,
                        start_ms=start_ms,
                        place_options=replayed_place_options(
                            stream=params.stream_option_coverage,
                            graph=graph,
                            sdmx=sdmx,
                            materialise_graph=materialise_graph,
                            materialise_sdmx=materialise_sdmx,
                            pool=place_options_pool,
                            coverage=coverage_snapshot,
                            population_memo=population_memo,
                        ),
                    )
                ) as hit:
                    async for ev in hit:
                        yield ev
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
        async with aclosing(events):
            async for ev in events:
                ev = _with_index_drift_warning(ev, warn=warn_index_drift)
                await scope.observe(ev)
                if isinstance(ev, ResponseEvent) and scope.active and lookup_ms is not None:
                    ev = ResponseEvent(
                        response=with_step_timing(ev.response, "cache_lookup", lookup_ms)
                    )
                yield ev
    finally:
        record.cache = scope.outcome
        await scope.release()


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
    budget: RequestBudget | None = None,
    hard_cap: asyncio.Timeout | None = None,
    binder: Binder | None = None,
) -> ResolveResponse:
    """Drain ``resolve_events`` into one buffered response.

    The ``response`` event precedes ``done``. Copy ``done.timed_out`` into
    diagnostics so buffered callers can detect a partial result. ``health``,
    when given, receives ``done.degraded_leg``, which the body omits. Client
    injection and lifetimes match ``resolve_events``. The buffered path emits
    no trailing SSE events, so ``place_options_pool`` launches no tasks.
    ``caller`` reaches the ``resolve`` log line, and ``budget``, ``hard_cap`` and ``binder``
    reach ``resolve_events``.

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
        binder=binder,
        budget=budget,
        hard_cap=hard_cap,
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
