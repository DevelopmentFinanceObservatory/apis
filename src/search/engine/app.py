"""Build Search's `Surface` or a standalone FastAPI app. The lifespan publishes `SearchDeps`.
Requests before startup receive `service_unavailable` (503).

Resolve supports opt-in SSE. Other requests receive buffered JSON. Options requests resolve
identity and role checks before cache lookup.

The surface returns RFC 9457 refusals. Infrastructure details stay server-side. Code-less
routing failures use generic details to prevent ID probing. Query length is checked before LLM
or graph calls, and unsupported doors return 501.

Resolve has one budget per request (`request_budget`): a soft deadline `request_timeout_s` from
its start, where the fan-out answers with what it has, and a hard cap `finalise_grace_s` later
(at least `resubmit_full_read_timeout_s` for a dataset_resubmit, whose member walk may run past
the soft deadline). The hard cap ends buffered JSON with a 504 and SSE with a terminal `error`,
and bounds every graph and SDMX read the request makes. SSE ends with a partial `done` after a
soft timeout or a terminal `error` for pipeline failures.
Options shares one deadline across two worker-thread hops. Cancellation
detaches a running thread, which keeps its executor slot until it finishes.
Upstream timeouts bound stalled calls.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from typing import Annotated, TypedDict

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

import graph_generation.config as generation_config
import search.engine.clients.llm as _llm_mod
from code_index.query import CodeIndex
from code_index.shared import get_index, per_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform import Envelope, HealthResponse, ProblemSpec, health_response, refusal
from dfo_platform.app import Surface
from dfo_platform.app import create_app as platform_create_app
from dfo_platform.build import build_commit
from dfo_platform.config import parse_cors_origins
from dfo_platform.named import cut
from dfo_platform.request_deadline import request_deadline
from dfo_platform.static_response import FrozenJSON, freeze_public_json
from dfo_platform.trace import get_trace_id
from dfo_platform.validation import error_field_loc, pointer_for_error, validation_problem
from entity_resolution import (
    Admission,
    EntityResolver,
    SharedResolver,
    acquire_shared,
    release_shared,
)
from graph_generation import GenerationProvider
from graph_generation import acquire_shared as acquire_shared_generation
from graph_generation import release_shared as release_shared_generation
from graph_release.context import bound_served_release
from graph_release.service import ReleaseService, get_release_service
from materialise import GraphReader, place_scope
from materialise.errors import UpstreamVersionChangedError
from materialise.facets import VariableBreadthReader, bound_variable_breadth
from materialise.rate_tables import RateTableCache, RateTableInvalidError
from materialise.shared_profiles import (
    ProfileSource,
    acquire_shared_profiles,
    release_shared_profiles,
)
from origin_catalogue import bound_catalogue, current_catalogue
from origin_catalogue.service import CatalogueService, get_catalogue_service
from search.catalogue.response import (
    CatalogueResponse,
    catalogue_response,
    place_catalogue_response,
)
from search.catalogue.service import search_catalogue_service
from search.engine.capabilities import Capabilities, build_capabilities
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.clients.llm import LLM, SupportsLLM, build_memo
from search.engine.clients.materialise_bridge import build_materialise_sdmx_client
from search.engine.clients.sdmx import EngineSdmxClient, LiveSdmxClient
from search.engine.clients.sdmx_cache import CachingSdmxClient
from search.engine.codes import CODE_CATALOG, SEARCH_CODE_CATALOGUE
from search.engine.config import SEARCH_UNIMPLEMENTED_DOORS, settings
from search.engine.core import ResolveHealth, request_budget, resolve_async, resolve_events
from search.engine.deadline import _await_bounded
from search.engine.degraded_reads import note_degraded_read
from search.engine.deps import (
    SEARCH_DEPS_SLOT,
    SearchDeps,
    SearchDepsDep,
    SearchDocuments,
    search_deps,
)
from search.engine.envelope import make_diagnostics, now_ms
from search.engine.errors import (
    RATE_TABLE_REFUSALS,
    EngineInfraError,
    EngineInputError,
    GraphInfraError,
    LLMInfraError,
)
from search.engine.events import DoneEvent, ErrorEvent, Event, serialize_sse
from search.engine.extraction_store import build_extraction_store
from search.engine.options.options_response_cache import (
    ComputedOptions,
    OptionsResponseCache,
    options_response_cache_key,
)
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.options.resolve import resolve_options_target
from search.engine.options.surface import compute_options
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_memo import PartMemo
from search.engine.retrieval.bound_values import prepare_index_views
from search.engine.schema import schema_payload
from search.engine.unit.conversion_catalogue import (
    ConversionCatalogue,
    ConversionCatalogueService,
    bound_conversion_catalogue,
    load_conversion_catalogue,
)
from search.models import (
    OptionsRequest,
    OptionsResponse,
    ResolveRequest,
    ResolveResponse,
)
from series_coverage.shared import get_coverage_service
from skills_index import SharedSkillsIndex, acquire_shared_index, release_shared_index

logger = logging.getLogger(__name__)

# The SSE path uses this envelope for the same catalogue and capability version as JSON.
_ENVELOPE = Envelope(catalogue=CODE_CATALOG, capability_version=settings().capability_version)

# Set to "1" on a buffered resolve whose answer backfilled a failed leg.
DEGRADED_LEG_HEADER = "X-Search-Degraded-Leg"

# Static fallback used when refusal classification or envelope rendering fails.
_FALLBACK_ERROR_TITLE = "Internal Server Error"
_FALLBACK_ERROR_DETAIL = "An unexpected error occurred."


def _accepts_sse(request: Request) -> bool:
    """Return true only for an explicit `text/event-stream` token with `q>0`. Wildcards,
    missing headers, and `q=0` select buffered JSON.
    """
    raw = request.headers.get("accept")
    if not raw:
        return False
    for part in raw.split(","):
        token, _, params = part.strip().partition(";")
        media = token.strip().lower()
        if media != "text/event-stream":
            continue
        q = 1.0
        for p in params.split(";"):
            k, _, val = p.strip().partition("=")
            if k.strip().lower() == "q":
                try:
                    q = float(val)
                except ValueError:
                    q = 0.0
        if q > 0:
            return True
    return False


def classify_search_exception(exc: BaseException) -> ProblemSpec | None:
    """Return a Search `ProblemSpec` or defer to platform classification. JSON and SSE use
    this function. Platform classification handles routing 404/405 errors.
    """
    if isinstance(exc, EngineInputError):
        # Code-less routing failures keep generic code and detail to prevent ID probing.
        code = exc.code if exc.code is not None else "request_invalid"
        detail = None if exc.code is None else str(exc)
        return ProblemSpec(
            status=exc.status,
            code=code,
            detail=detail,
            pointer=exc.pointer,
            alternatives=exc.alternatives,
            cause=exc.cause,
            retryable=False,
        )
    if isinstance(exc, RATE_TABLE_REFUSALS):
        # The materialise library refuses an inadmissible rate table with 501
        # `capability_unavailable` and a table version that moved mid-load with 503, retryable.
        logger.warning("Rate table refusal %s: %s", exc.code, exc.detail)
        return ProblemSpec(
            status=exc.status, code=exc.code, detail=exc.detail, retryable=exc.retryable
        )
    if isinstance(exc, EngineInfraError):
        logger.error("Engine infrastructure error: %s", exc, exc_info=exc)
        # Upstream 4xx responses (e.g. graph returned 404) map to 502 Bad Gateway
        # so callers can distinguish client-side upstream errors from transient failures.
        upstream_status = getattr(exc, "upstream_status", None)
        if isinstance(upstream_status, int) and 400 <= upstream_status < 500:
            return ProblemSpec(status=502, code="upstream_error", retryable=True)
        return ProblemSpec(status=503, code="service_unavailable", retryable=True)
    if isinstance(exc, RequestValidationError):
        # The platform maps body errors to 400 and schema-version errors to 422. Search adds
        # a field-specific refinement below.
        problem = validation_problem(exc)
        errors = exc.errors()
        # Unsupported identity versions and recognised future facet axes also map to 422.
        # Match the nested field path or Pydantic error type. A top-level schema-version error
        # already maps to 422 in the platform.
        if problem.code == "request_invalid":
            version_error = next(
                (
                    e
                    for e in errors
                    if error_field_loc(e)[-1:] == ["identity_version"]
                    or e.get("type") == "facet_axis_higher_schema_version"
                ),
                None,
            )
            if version_error is not None:
                problem = problem._replace(
                    code="unsupported_schema_version",
                    status=422,
                    pointer=pointer_for_error(exc, version_error),
                    detail=None,
                )
        return problem
    if isinstance(exc, asyncio.TimeoutError):
        logger.error("Request hit its hard deadline")
        return ProblemSpec(status=504, code="timeout", retryable=True)
    return None


# Module-level sentinels for the SSE merge queue.
_HEARTBEAT = object()  # 10s tick, serialized as `: ping\n\n`
_SENTINEL = object()


async def _event_source(
    stream: AsyncIterator[Event], *, hard_deadline: float
) -> AsyncIterator[str]:
    """Merge events with 10-second heartbeat comments. Stop at the terminal event and cancel
    both helper tasks on completion or disconnect. `resolve_events` owns the soft deadline.

    `hard_deadline`, an absolute `time.monotonic()` value, caps the stream. The producer
    cancels the pipeline then and ends the stream with the 504 `timeout` error event. It also
    bounds every read the pipeline makes (`request_deadline`). The cap wraps the producer's
    loop, so it never fires inside a `yield` the consumer resumes.
    """
    queue: asyncio.Queue[object] = asyncio.Queue()

    async def produce() -> None:
        try:
            with request_deadline(hard_deadline):
                async with asyncio.timeout(hard_deadline - time.monotonic()):
                    async for ev in stream:
                        await queue.put(ev)
        except Exception as exc:
            # This is the sole conversion of a pipeline failure into a terminal ErrorEvent.
            # Classification logs the failure as it does on the JSON route, so an unhandled
            # error's traceback appears once. The event carries only the client detail.
            try:
                spec = classify_search_exception(exc) or refusal.classify_default(exc)
                # The envelope validates codes against the same registry as JSON responses.
                details = _ENVELOPE.details(
                    status=spec.status,
                    code=spec.code,
                    detail=spec.detail,
                    pointer=spec.pointer,
                    alternatives=spec.alternatives,
                    truncated=spec.truncated,
                    issues=spec.issues,
                    retryable=spec.retryable,
                )
                error_event = ErrorEvent(**details.model_dump())
            except Exception as classify_exc:
                # Keep the stream terminal when classification fails. Constants provide the
                # fallback without another catalogue lookup.
                logger.error(
                    "classifying %s for the SSE terminal error event failed: %s",
                    type(exc).__name__,
                    classify_exc,
                    exc_info=classify_exc,
                )
                error_event = ErrorEvent(
                    type=f"/codes/{refusal.UNHANDLED_DEFAULT.code}",
                    title=_FALLBACK_ERROR_TITLE,
                    status=refusal.UNHANDLED_DEFAULT.status,
                    detail=_FALLBACK_ERROR_DETAIL,
                    code=refusal.UNHANDLED_DEFAULT.code,
                    trace_id=get_trace_id(),
                    capability_version=settings().capability_version,
                    retryable=refusal.UNHANDLED_DEFAULT.retryable,
                )
            await queue.put(error_event)
        finally:
            await queue.put(_SENTINEL)

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(10)
            await queue.put(_HEARTBEAT)

    prod_task = asyncio.create_task(produce())
    hb_task = asyncio.create_task(heartbeat())
    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                return
            if item is _HEARTBEAT:
                yield ": ping\n\n"
                continue
            # ty cannot narrow `object` through the sentinel identity checks.
            yield serialize_sse(item)  # ty: ignore[invalid-argument-type]
            if isinstance(item, (DoneEvent, ErrorEvent)):
                return  # exactly one terminal event
    finally:
        # Cancel and await both helpers so neither outlives the response. Producer cancellation
        # unwinds the pipeline's in-flight variable fan-out.
        for t in (prod_task, hb_task):
            t.cancel()
        await asyncio.gather(prod_task, hb_task, return_exceptions=True)


def _schema_payload_response_schema() -> dict:
    """Derive the OpenAPI body schema from `schema_payload`, keeping property types aligned
    with the returned document.
    """
    document = schema_payload()
    properties: dict[str, object] = {}
    for key, value in document.items():
        if isinstance(value, dict):
            properties[key] = {"type": "object", "description": f"JSON Schema for {key}"}
        elif isinstance(value, bool):
            properties[key] = {"type": "boolean", "const": value}
        elif isinstance(value, int):
            properties[key] = {"type": "integer", "const": value}
        elif isinstance(value, str):
            properties[key] = {"type": "string"}
        else:
            raise TypeError(
                f"schema document key {key!r} holds an unsupported value kind for the "
                f"OpenAPI response schema: {type(value)!r}"
            )
    return {
        "type": "object",
        "required": list(document),
        "properties": properties,
    }


def search_documents() -> SearchDocuments:
    """Freeze the public schema, and the capability document of the index `get_index()` returns. Build
    errors propagate."""

    @per_index
    def capabilities(index: CodeIndex) -> FrozenJSON:
        return freeze_public_json(build_capabilities(index).model_dump(mode="json"), max_age=300)

    capabilities(get_index())
    return SearchDocuments(
        schema=freeze_public_json(schema_payload(), max_age=300), capabilities=capabilities
    )


_RESOLVE_RESPONSES: dict[int | str, dict[str, object]] = {
    200: {
        "description": (
            "A buffered ResolveResponse JSON document or, with "
            "Accept: text/event-stream, progressive Server-Sent Events. "
            "Successful streams emit a complete response event and end with "
            "done; pipeline failures end with error and emit neither response "
            "nor done."
        ),
        "content": {
            "text/event-stream": {
                "schema": {
                    "type": "string",
                    "format": "event-stream",
                    "description": (
                        "Progressive Server-Sent Events frames. On successful streams, "
                        "the response event contains the complete ResolveResponse body."
                    ),
                }
            }
        },
    }
}


class _PipelineDeps(TypedDict):
    """The keywords `resolve_endpoint` passes `resolve_events` and `resolve_async` from the
    app's dependencies and the request. The binder is left to `Settings.binder`."""

    graph: EngineGraphClient
    resolver: EntityResolver
    admission: Admission | None
    llm: SupportsLLM
    sdmx: EngineSdmxClient
    materialise_graph: GraphReader | None
    materialise_sdmx: MaterialiseSdmxClient | None
    rate_cache: RateTableCache | None
    population_memo: OptionsPopulationMemo | None
    place_options_pool: PlaceOptionsPool | None
    shared_skills_index: SharedSkillsIndex | None
    response_cache: ResponseCache | None
    part_memo: PartMemo | None
    generation: GenerationProvider
    caller: str | None


async def resolve_endpoint(body: ResolveRequest, request: Request, deps: SearchDepsDep) -> Response:
    # Share the unimplemented-door set with capabilities so liveness and refusals agree.
    if body.input.kind in SEARCH_UNIMPLEMENTED_DOORS:
        raise EngineInputError(
            f"This deployment does not serve the {body.input.kind!r} input door; consult "
            "GET /api/search/capabilities for per-door liveness.",
            code="door_not_implemented",
            status=501,
            pointer="/input/kind",
        )

    if body.input.kind == "raw_text":
        q = body.input.query  # type: ignore[union-attr]
        if len(q) > settings().max_query_chars:
            raise EngineInputError(
                "The query exceeds this deployment's "
                f"{settings().max_query_chars}-character limit.",
                code="query_too_long",
                status=422,
                pointer="/input/query",
            )

    # Refresh before every generation-keyed cache lookup.
    await deps.generation.refresh_if_stale()

    pipeline_deps: _PipelineDeps = {
        "graph": deps.graph,
        "resolver": deps.resolver,
        "admission": deps.admission,
        "llm": deps.llm,
        "sdmx": deps.sdmx,
        "materialise_graph": deps.materialise_graph,
        "materialise_sdmx": deps.materialise_sdmx,
        "rate_cache": deps.rate_cache,
        "population_memo": deps.population_memo,
        "place_options_pool": deps.place_options_pool,
        "shared_skills_index": deps.skills_index,
        "response_cache": deps.response_cache,
        "part_memo": deps.part_memo,
        "generation": deps.generation,
        "caller": request.headers.get("user-agent"),
    }

    # The soft deadline and the hard cap share this start.
    budget = request_budget(kind=body.input.kind)

    # JSON and SSE share the ResolveResponse body. SSE adds trailing slot-option coverage events.
    if _accepts_sse(request):
        return StreamingResponse(
            _event_source(
                resolve_events(body, stream_slot_options=True, budget=budget, **pipeline_deps),
                hard_deadline=budget.hard,
            ),
            media_type="text/event-stream",
            headers={
                # `no-transform` prevents intermediaries such as the Next dev proxy from
                # gzip-buffering progressive frames into one burst.
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # The soft deadline returns a partial response first. The hard cap, finalise_grace_s
    # later, ends a request whose unfinished work does not unwind in time.
    health = ResolveHealth()
    with request_deadline(budget.hard):
        result: ResolveResponse = await asyncio.wait_for(
            resolve_async(body, health=health, budget=budget, **pipeline_deps),
            timeout=budget.hard - time.monotonic(),
        )
    # The degraded-leg flag rides in a header so the body contract stays unchanged.
    headers = {DEGRADED_LEG_HEADER: "1"} if health.degraded_leg else None
    return JSONResponse(result.model_dump(mode="json", exclude_none=True), headers=headers)


async def options_endpoint(body: OptionsRequest, deps: SearchDepsDep) -> Response:
    """The options door: on-demand candidate enumeration for one resolved
    entity role (origin, destination, or observationAbout) of a posted
    identity, queried independently of the /resolve pipeline and fetched
    lazily when a user opens an editor.

    Composes engine.options.resolve.resolve_options_target (derive the
    posted identity's own StatVar population, typed anchors and
    role-target validity, server-side, with no resubmit or materialise
    pipeline) with engine.options.surface.compute_options (compute the
    per-role option surface), then assembles the envelope here --
    see those two modules' own docstrings for the full design, including
    the option_target_invalid (422) role check that runs inside
    resolution. Every EngineInputError either half raises propagates
    uncaught to classify_search_exception above, the same RFC 9457
    problem body every other door's refusal returns -- no bespoke
    handling here.

    resolve_options_target always runs, hit or miss on the response
    cache below: it is what runs validate_edited_identity and the role/
    anchor refusals, and none of those may be skipped just because an
    identical target was already answered once. Only build_options_
    response's own compute -- the graph/SDMX work -- is what a cache hit
    skips; see options.options_response_cache's own module docstring for
    the key and the warning-free-result caching rule.
    """
    # The options population is derived through the materialise `GraphReader`.
    materialise_graph = deps.materialise_graph
    if materialise_graph is None:
        raise EngineInfraError("materialise graph client not initialised")
    # Refresh before the generation-keyed memo lookup. Request timing excludes this wait.
    await deps.generation.refresh_if_stale()
    # Start request timing before resolution. The deadline tracks elapsed time separately.
    start_ms = now_ms()
    # Share one coverage snapshot and one origin catalogue (the request's bound build, which
    # compute_options reads too) between response computation and the cache key.
    coverage_snapshot = get_coverage_service().snapshot()
    catalogue = current_catalogue()
    # Separate worker hops provide a cancellation boundary. An elapsed deadline maps to 504.
    deadline = time.monotonic() + settings().request_timeout_s
    target = await _await_bounded(
        asyncio.to_thread(
            resolve_options_target,
            body,
            graph=deps.graph,
            materialise_graph=materialise_graph,
            population_memo=deps.population_memo,
            deadline=deadline,
        ),
        deadline=deadline,
    )

    def _compute(compute_deadline: float) -> ComputedOptions:
        return compute_options(
            body,
            target,
            graph=deps.graph,
            deadline=compute_deadline,
            materialise_graph=materialise_graph,
            materialise_sdmx=deps.materialise_sdmx,
            coverage=coverage_snapshot,
        )

    engine_build = build_commit()
    options_response_cache = deps.options_response_cache
    if options_response_cache is not None:
        key = options_response_cache_key(
            body,
            target,
            build_commit=engine_build,
            coverage=coverage_snapshot,
            catalogue=catalogue,
        )
        # A shared compute reads by its flight's deadline, so this request's deadline never
        # cuts a result others reuse. get_or_compute bounds this request's wait.
        computed = await options_response_cache.get_or_compute(key, _compute, deadline=deadline)
    else:
        computed = await _await_bounded(asyncio.to_thread(_compute, deadline), deadline=deadline)

    # Rebuild diagnostics per request so trace ID and timing describe this call.
    diagnostics = make_diagnostics(engine_build, list(computed.warnings), {}, now_ms() - start_ms)
    result = OptionsResponse(diagnostics=diagnostics, for_=body.for_, options=computed.options)
    # Keep the `for` alias and meaningful `coverage=null` for unconfirmed SSE coverage.
    return JSONResponse(result.model_dump(mode="json", by_alias=True, exclude_none=True))


async def healthz() -> HealthResponse:
    return health_response(capability_version=settings().capability_version)


async def schema(request: Request, deps: SearchDepsDep) -> Response:
    return deps.documents.schema.respond(request)


async def capabilities(request: Request, deps: SearchDepsDep) -> Response:
    # Deployment configuration can change capabilities without changing the contract schema.
    return deps.documents.capabilities(get_index()).respond(request)


class SearchCatalogueUnavailable(EngineInfraError):
    """Retryable 503 `service_unavailable` when no search catalogue build is loaded."""


# The job publishes at most every 15 minutes, so a client may reuse a body for five.
_CATALOGUE_CACHE_CONTROL = "public, max-age=300"


async def catalogue_endpoint(
    deps: SearchDepsDep,
    place: Annotated[
        str | None,
        Query(
            pattern=r"^country/\S+$",
            description="a country dcid; scopes the counts and examples to that country",
        ),
    ] = None,
) -> Response:
    """What DFO holds and example sentences to ask of it, from the published
    search catalogue. `place` scopes the counts to one country and offers the
    examples whose StatVar has data there, filled with its name."""
    service = deps.search_catalogue
    artifact = service.current() if service is not None else None
    if artifact is None:
        raise SearchCatalogueUnavailable(
            "search catalogue unavailable: "
            + ("SEARCH_CATALOGUE_URI is not set" if service is None else "no build is loaded")
        )
    if place is None:
        body = catalogue_response(artifact)
    else:
        scoped = place_catalogue_response(artifact, place)
        if scoped is None:
            raise EngineInputError(
                f"The catalogue holds no data for place {cut(place)!r}.",
                code="request_invalid",
            )
        body = scoped
    return JSONResponse(
        body.model_dump(mode="json"), headers={"Cache-Control": _CATALOGUE_CACHE_CONTROL}
    )


def _logged(message: str, release: Callable[..., object], *args: object) -> None:
    release(*args)
    logger.info(message)


def _register(
    registrations: ExitStack,
    generation: GenerationProvider,
    name: str,
    *,
    stats: Callable[[], Mapping[str, int]],
    clear: Callable[[], None],
) -> None:
    registrations.callback(generation.register(name, stats=stats, clear=clear).close)


def _build_response_cache(
    registrations: ExitStack, generation: GenerationProvider
) -> ResponseCache:
    cache = ResponseCache()
    logger.info(
        "ResponseCache built (enabled=%s max_bytes=%d max_entry_bytes=%d "
        "known_max_age_s=%.0f unknown_max_age_s=%.0f)",
        settings().response_cache_enabled,
        settings().response_cache_max_bytes,
        settings().response_cache_max_entry_bytes,
        generation_config.GRAPH_CACHE_MAX_AGE_SECONDS,
        generation_config.GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
    )
    # Refresh invokes clear from a worker thread. Schedule clear on this loop because the cache
    # lock is loop-bound.
    loop = asyncio.get_running_loop()

    def _stats() -> Mapping[str, int]:
        return dataclasses.asdict(cache.stats())

    def _clear() -> None:
        loop.call_soon_threadsafe(cache.clear_sync)

    _register(registrations, generation, "search.response_cache", stats=_stats, clear=_clear)
    return cache


def _build_part_memo(registrations: ExitStack, generation: GenerationProvider) -> PartMemo:
    memo = PartMemo()
    logger.info(
        "PartMemo built (enabled=%s max_bytes=%d max_entry_bytes=%d)",
        settings().response_cache_enabled,
        settings().response_cache_memo_max_bytes,
        settings().response_cache_memo_max_entry_bytes,
    )
    # Refresh calls stats and clear from its own thread. `stats` takes the memo's lock; clear
    # runs on this loop.
    loop = asyncio.get_running_loop()

    def _clear() -> None:
        loop.call_soon_threadsafe(memo.clear)

    _register(
        registrations, generation, "search.response_cache_memo", stats=memo.stats, clear=_clear
    )
    return memo


async def _open_search_deps(
    owned: AsyncExitStack,
    registrations: ExitStack,
    *,
    documents: SearchDocuments,
    graph: EngineGraphClient | None,
    resolver: EntityResolver | None,
    admission: Admission | None,
    llm: SupportsLLM | None,
    sdmx: EngineSdmxClient | None,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    skills_index: SharedSkillsIndex | None,
    response_cache: ResponseCache | None,
    part_memo: PartMemo | None,
    options_response_cache: OptionsResponseCache | None,
    population_memo: OptionsPopulationMemo | None,
    place_options_pool: PlaceOptionsPool | None,
    generation: GenerationProvider | None,
    catalogue: CatalogueService,
    release: ReleaseService,
    conversion_catalogue: ConversionCatalogueService | None,
) -> SearchDeps:
    """Build missing dependencies and return the bundle. The `owned` stack closes clients in
    reverse registration order. Exit generation registrations first so refresh callbacks cannot
    reach closed caches. Injected instances are neither registered nor released.
    """
    # Acquire the provider before its caches. The source read can block.
    if generation is None:
        generation = await asyncio.to_thread(
            functools.partial(
                acquire_shared_generation,
                catalogue=catalogue,
                expected_graph_base=settings().graph_base,
            )
        )
        owned.callback(
            _logged, "graph generation provider released", release_shared_generation, generation
        )
        logger.info("graph generation provider acquired (service base: %s)", settings().graph_base)

    # The code index's views are built on first read, so build them before serving, from the
    # index a request would read. `to_thread` carries the binding into the worker thread.
    with bound_served_release(release):
        await asyncio.to_thread(prepare_index_views)

    shared_resolvekit: SharedResolver | None = None
    if resolver is None:
        shared_resolvekit = acquire_shared()
        owned.callback(_logged, "EntityResolver released", release_shared, shared_resolvekit)

    if materialise_sdmx is None:
        caching_materialise_sdmx = build_materialise_sdmx_client(generation=generation)
        owned.callback(_logged, "materialise SdmxClient closed", caching_materialise_sdmx.close)
        logger.info("materialise SdmxClient started (graph: %s)", settings().graph_base)
        _register(
            registrations,
            generation,
            "search.materialise_sdmx",
            stats=caching_materialise_sdmx.stats,
            clear=caching_materialise_sdmx.clear,
        )
        materialise_sdmx = caching_materialise_sdmx

    if sdmx is None:
        # Share one process-wide SDMX cache across resolve, options, and resubmit.
        caching_sdmx = CachingSdmxClient(LiveSdmxClient(), generation=generation)
        owned.callback(_logged, "LiveSdmxClient closed", caching_sdmx.close)
        logger.info("LiveSdmxClient started (graph: %s)", settings().graph_base)
        _register(
            registrations,
            generation,
            "search.sdmx",
            stats=caching_sdmx.stats,
            clear=caching_sdmx.clear,
        )
        sdmx = caching_sdmx

    live_graph: LiveGraphClient | None = None
    if graph is None:
        live_graph = LiveGraphClient.from_search_settings(settings(), generation=generation)
        owned.callback(_logged, "LiveGraphClient closed", live_graph.close)
        # Naming the graph target at boot makes a wrong graph_base visible
        # before it surfaces as opaque 503s mid-request.
        logger.info("LiveGraphClient started (graph: %s)", settings().graph_base)
        _register(
            registrations,
            generation,
            "search.graph_client",
            stats=live_graph.stats,
            clear=live_graph.clear,
        )
        graph = live_graph

    if rate_cache is None:
        rate_cache = RateTableCache()
        logger.info("RateTableCache built")

    # Runs beside the rest of startup; waited for, within a bound, before the app serves.
    conversion_loading = _start_conversion_catalogue_load(
        owned,
        registrations,
        conversion_catalogue,
        generation=generation,
        catalogue=catalogue,
        release=release,
        graph=graph,
        reader=materialise_graph if materialise_graph is not None else live_graph,
        sdmx=materialise_sdmx,
        rate_cache=rate_cache,
    )

    # The default place scope's memo is process-wide and keys on the generation stamp, so a
    # clear only frees its entries.
    _register(
        registrations,
        generation,
        "search.default_place_scope",
        stats=place_scope.memo_stats,
        clear=place_scope.clear_memo,
    )

    if response_cache is None:
        response_cache = _build_response_cache(registrations, generation)

    if part_memo is None:
        part_memo = _build_part_memo(registrations, generation)

    if options_response_cache is None:
        options_response_cache = OptionsResponseCache(generation=generation)
        logger.info(
            "OptionsResponseCache built (enabled=%s max_entries=%d "
            "known_max_age_s=%.0f unknown_max_age_s=%.0f)",
            settings().options_response_cache_enabled,
            settings().options_response_cache_max_entries,
            generation_config.GRAPH_CACHE_MAX_AGE_SECONDS,
            generation_config.GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
        )
        _register(
            registrations,
            generation,
            "search.options_response_cache",
            stats=options_response_cache.stats,
            clear=options_response_cache.clear,
        )

    if population_memo is None:
        population_memo = OptionsPopulationMemo(generation=generation)
        logger.info(
            "OptionsPopulationMemo built (enabled=%s max_entries=%d "
            "known_max_age_s=%.0f unknown_max_age_s=%.0f)",
            settings().options_population_cache_enabled,
            settings().options_population_cache_max_entries,
            generation_config.GRAPH_CACHE_MAX_AGE_SECONDS,
            generation_config.GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
        )
        _register(
            registrations,
            generation,
            "search.options_population_memo",
            stats=population_memo.stats,
            clear=population_memo.clear,
        )

    if place_options_pool is None:
        place_options_pool = PlaceOptionsPool.build(
            max_workers=settings().place_options_workers,
            options_cache=options_response_cache,
        )
        # Running jobs can outlive `shutdown()`. Release the pool before their clients.
        owned.callback(_logged, "PlaceOptionsPool shut down", place_options_pool.shutdown)
        logger.info("PlaceOptionsPool built (max_workers=%d)", settings().place_options_workers)

    # A missing or unreadable catalogue leaves the service empty, so startup never waits on it.
    search_catalogue = await asyncio.to_thread(search_catalogue_service)
    if search_catalogue is not None:
        owned.callback(_logged, "SearchCatalogueService closed", search_catalogue.close)

    if resolver is None:
        assert shared_resolvekit is not None
        resolver = EntityResolver(
            graph=graph, shared=shared_resolvekit, generation=generation, code_index=get_index
        )
        logger.info("EntityResolver started")
        _register(
            registrations,
            generation,
            "search.entity_resolver",
            stats=resolver.stats,
            clear=resolver.clear,
        )

    if admission is None:
        admission = Admission(graph=graph, generation=generation)
        logger.info("Admission started")
        _register(
            registrations,
            generation,
            "search.admission",
            stats=admission.stats,
            clear=admission.clear,
        )

    if skills_index is None:
        # EngineGraphClient satisfies SkillsGraph structurally, but the two
        # Protocols name separately defined SubjectsPage classes.
        skills_index = await acquire_shared_index(
            graph=graph,  # ty: ignore[invalid-argument-type]
            catalogue=catalogue.snapshot,
            release=release,
        )
        # Stop the index refresh task before closing the graph it reads.
        owned.push_async_callback(_release_skills_index, skills_index)
        logger.info("SharedSkillsIndex started")

    if llm is None:
        # Check `GEMINI_API_KEY` before building the LLM. A zero memo limit disables the memo,
        # and `SEARCH_EXTRACT_STORE=off` the extraction store.
        try:
            _llm_mod.warm()
        except LLMInfraError:
            logger.critical("LLM warm-up failed; check GEMINI_API_KEY", exc_info=True)
            raise
        llm = LLM(
            memo=build_memo(),
            extraction_store=build_extraction_store(settings().extract_store),
        )
        logger.info(
            "LLM memo built (max_bytes=%d max_entry_bytes=%d ttl_s=%.0f), extraction store %s",
            settings().llm_memo_max_bytes,
            settings().llm_memo_max_entry_bytes,
            settings().llm_memo_ttl_s,
            settings().extract_store,
        )

    variable_breadth: VariableBreadthReader | None = None
    if live_graph is not None and materialise_graph is None:
        # An answer naming one place breaks a facet tie on each facet's places across the
        # whole variable, read from these profiles as a Datasets mint reads them.
        # The process's shared profile cache, shared with Graph and Datasets when they name
        # the same roots, reads Search's graph root for both graph and SDMX availability, as
        # `materialise_sdmx` does.
        api_root = f"{settings().graph_base.rstrip('/')}/core/api"
        profiles = acquire_shared_profiles(
            source=ProfileSource(
                graph_base_url=api_root,
                sdmx_base_url=api_root,
                un_base_url=settings().un_graph_base_url,
                token=settings().graph_token or None,
            ),
            generation=generation,
        )
        owned.push_async_callback(
            asyncio.to_thread,
            _logged,
            "facet profile cache released",
            release_shared_profiles,
            profiles,
        )
        variable_breadth = profiles.breadth(get_coverage_service().snapshot)

    if conversion_loading is not None:
        await _wait_for_conversion_catalogue(conversion_loading)

    return search_deps(
        graph=graph,
        resolver=resolver,
        llm=llm,
        sdmx=sdmx,
        documents=documents,
        live_graph=live_graph,
        generation=generation,
        admission=admission,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        skills_index=skills_index,
        response_cache=response_cache,
        part_memo=part_memo,
        options_response_cache=options_response_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        search_catalogue=search_catalogue,
        variable_breadth=variable_breadth,
    )


def _note_breadth_fallback() -> None:
    """Note a tie the whole-variable breadth would have decided, left to the unit's name
    because the breadth was not read, so the resolution's region is marked degraded and its
    answer is not cached (`materialise.facets.bound_variable_breadth`'s `on_fallback`)."""
    note_degraded_read("whole-variable breadth")


_CONVERSION_CATALOGUE_LOAD_S = 120.0
"""The bound on one conversion catalogue load's graph and SDMX reads."""

_CONVERSION_CATALOGUE_STARTUP_WAIT_S = 15.0
"""How long startup waits for the first conversion catalogue load to end: about eight times
a cold load's 1.9 s, so a slow upstream delays boot by 15 s at most. When the load fails or
outlasts the wait, the app serves, its currency answers warn `unit_conversion_unavailable`,
and the load goes on, retried, in the background."""


def _start_conversion_catalogue_load(
    owned: AsyncExitStack,
    registrations: ExitStack,
    service: ConversionCatalogueService | None,
    *,
    generation: GenerationProvider,
    catalogue: CatalogueService,
    release: ReleaseService,
    graph: EngineGraphClient,
    reader: GraphReader | None,
    sdmx: MaterialiseSdmxClient,
    rate_cache: RateTableCache,
) -> ConversionCatalogueService | None:
    """Start loading the conversion catalogue on the service's thread, register a reload on
    graph generation changes, and stop the thread when `owned` closes. Return the service,
    or None when there is no service to load, or no graph reader to load it with, which
    logs an ERROR: answers then state no unit_conversion."""
    if service is None:
        return None
    if reader is None:
        logger.error("no materialise graph reader; answers state no unit_conversion")
        return None

    def load() -> ConversionCatalogue:
        with bound_catalogue(catalogue.snapshot()), bound_served_release(release):
            return load_conversion_catalogue(
                graph=graph,
                reader=reader,
                sdmx=sdmx,
                cache=rate_cache,
                deadline=time.monotonic() + _CONVERSION_CATALOGUE_LOAD_S,
            )

    owned.callback(_logged, "conversion catalogue loading stopped", service.close)
    _register(
        registrations,
        generation,
        "search.conversion_catalogue",
        stats=service.stats,
        clear=service.reload_in_background,
    )
    service.start(load)
    return service


async def _wait_for_conversion_catalogue(service: ConversionCatalogueService) -> None:
    """Wait at most `_CONVERSION_CATALOGUE_STARTUP_WAIT_S` for the first catalogue load to
    end, and log a WARNING when the app starts serving without a catalogue."""
    if not await asyncio.to_thread(service.wait_first_load, _CONVERSION_CATALOGUE_STARTUP_WAIT_S):
        logger.warning(
            "conversion catalogue not loaded at startup (bound %.0f s); serving without it "
            "while it loads in the background",
            _CONVERSION_CATALOGUE_STARTUP_WAIT_S,
        )


@contextmanager
def _request_scope(
    catalogue: CatalogueService,
    release: ReleaseService,
    conversion_catalogue: ConversionCatalogueService | None,
    variable_breadth: VariableBreadthReader | None,
) -> Iterator[None]:
    """Bind the origin catalogue's and the conversion catalogue's loaded snapshots, the
    served graph release, and the process's whole-variable breadth reader, for one request."""
    conversion = conversion_catalogue.snapshot() if conversion_catalogue is not None else None
    with (
        bound_catalogue(catalogue.snapshot()),
        bound_served_release(release),
        bound_conversion_catalogue(conversion),
        # Search stores no election, so a tie an unread breadth would have decided falls
        # back to the unit's name and marks the answer degraded.
        bound_variable_breadth(
            variable_breadth, refuse_unread=False, on_fallback=_note_breadth_fallback
        ),
    ):
        yield


async def _release_skills_index(skills_index: SharedSkillsIndex) -> None:
    await release_shared_index(skills_index)
    logger.info("SharedSkillsIndex released")


def _require_jev_key() -> None:
    """Refuse to serve when `Settings.binder` is ``jev`` or ``shadow`` and no TypeSafe key
    is set, since every call routed to Jev would then fall back to Gemini."""
    config = settings()
    if config.binder in ("jev", "shadow") and not config.jev_api_key.get_secret_value():
        raise RuntimeError(f"SEARCH_BINDER={config.binder} needs TYPESAFE_API_KEY; it is unset")


def search_surface(
    *,
    graph: EngineGraphClient | None = None,
    resolver: EntityResolver | None = None,
    admission: Admission | None = None,
    llm: SupportsLLM | None = None,
    sdmx: EngineSdmxClient | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    skills_index: SharedSkillsIndex | None = None,
    response_cache: ResponseCache | None = None,
    part_memo: PartMemo | None = None,
    options_response_cache: OptionsResponseCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    generation: GenerationProvider | None = None,
    conversion_catalogue: ConversionCatalogueService | None = None,
) -> Surface:
    """Build Search's `Surface` and a lifespan that publishes `SearchDeps`.

    Arguments inject dependencies. The lifespan builds and releases missing dependencies.
    Injected instances remain caller-owned. `materialise_graph` defaults to the live graph
    client. The process-wide generation provider rejects duplicate cache registrations, so
    concurrent apps need separate injected generation providers. Each request binds the origin
    catalogue's and the conversion catalogue's loaded snapshots for its whole run, and the
    release service's served release. An injected `conversion_catalogue` is served as loaded;
    without one the lifespan loads a new one.

    Raises:
        RuntimeError: `Settings.binder` sends calls to Jev and ``TYPESAFE_API_KEY`` is
            unset (`_require_jev_key`).
    """
    _require_jev_key()
    # Fail during construction when an artifact or the graph release is missing or unreadable.
    catalogue = get_catalogue_service()
    release = get_release_service()
    # The documents read the code index as they are built, so they are built under the
    # release a request would bind.
    with bound_served_release(release):
        documents = search_documents()
    owns_conversion = conversion_catalogue is None
    conversions = conversion_catalogue or ConversionCatalogueService()
    # The lifespan sets the reader it builds, so each request binds it (`_request_scope`).
    variable_breadth: VariableBreadthReader | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal variable_breadth
        # `registrations` exits before `owned`, on startup failure and on
        # shutdown alike.
        async with AsyncExitStack() as owned:
            with ExitStack() as registrations:
                deps = await _open_search_deps(
                    owned,
                    registrations,
                    documents=documents,
                    graph=graph,
                    resolver=resolver,
                    admission=admission,
                    llm=llm,
                    sdmx=sdmx,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=rate_cache,
                    skills_index=skills_index,
                    response_cache=response_cache,
                    part_memo=part_memo,
                    options_response_cache=options_response_cache,
                    population_memo=population_memo,
                    place_options_pool=place_options_pool,
                    generation=generation,
                    catalogue=catalogue,
                    release=release,
                    conversion_catalogue=conversions if owns_conversion else None,
                )
                SEARCH_DEPS_SLOT.set(app, deps)
                variable_breadth = deps.variable_breadth
                try:
                    yield
                finally:
                    variable_breadth = None
                    SEARCH_DEPS_SLOT.clear(app)

    router = APIRouter()
    # The handler returns a pre-built Response, so response_model only
    # documents the body in /openapi.json.
    router.post(
        "/api/search/resolve", response_model=ResolveResponse, responses=_RESOLVE_RESPONSES
    )(resolve_endpoint)
    router.post("/api/search/options", response_model=OptionsResponse)(options_endpoint)
    router.get("/api/search/healthz", response_model=HealthResponse)(healthz)
    router.get(
        "/api/search/schema",
        responses={
            200: {"content": {"application/json": {"schema": _schema_payload_response_schema()}}}
        },
    )(schema)
    router.get("/api/search/capabilities", response_model=Capabilities)(capabilities)
    router.get("/api/search/catalogue", response_model=CatalogueResponse)(catalogue_endpoint)

    return Surface(
        name="search",
        prefix="/api/search",
        router=router,
        capability_version=settings().capability_version,
        codes=SEARCH_CODE_CATALOGUE,
        classify=classify_search_exception,
        exception_types=(
            EngineInfraError,
            EngineInputError,
            RequestValidationError,
            asyncio.TimeoutError,
            RateTableInvalidError,
            UpstreamVersionChangedError,
        ),
        exception_exemplars=(
            EngineInputError(
                "posted shape does not match the five-tuple of its members",
                code="shape_mismatch",
                status=422,
            ),
            GraphInfraError("simulated upstream failure", upstream_status=404),
            RequestValidationError(
                errors=[
                    {
                        "type": "string_type",
                        "loc": ("body", "exemplar_field"),
                        "msg": "Input should be a valid string",
                        "input": None,
                    }
                ],
                body={"exemplar_field": None},
            ),
            TimeoutError(),
            RateTableInvalidError("exemplar rate table has no lastDataRefreshDate"),
            UpstreamVersionChangedError("exemplar rate table changed version during its load"),
        ),
        lifespan=lifespan,
        registry_endpoints={
            "/api/search/resolve": "search.resolve",
            "/api/search/options": "search.options",
        },
        request_scope=lambda: _request_scope(catalogue, release, conversions, variable_breadth),
    )


def create_app(
    *,
    graph: EngineGraphClient | None = None,
    resolver: EntityResolver | None = None,
    admission: Admission | None = None,
    llm: SupportsLLM | None = None,
    sdmx: EngineSdmxClient | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    skills_index: SharedSkillsIndex | None = None,
    generation: GenerationProvider | None = None,
) -> FastAPI:
    """Build a standalone Search app for tests and local runs. Deployed services compose
    `search_surface()` with the other API surfaces.
    """
    return platform_create_app(
        [
            search_surface(
                graph=graph,
                resolver=resolver,
                admission=admission,
                llm=llm,
                sdmx=sdmx,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=rate_cache,
                skills_index=skills_index,
                generation=generation,
            )
        ],
        cors_origins=parse_cors_origins(),
        title="Search Engine",
        version="1.0",
    )
