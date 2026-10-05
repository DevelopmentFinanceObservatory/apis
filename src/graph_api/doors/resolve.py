"""Resolve text through `entity_resolution` and map its results to the Graph API.

The shared component handles matching, listed-name checks, group membership,
and entity confirmation. This door expands continents and subregions through
containment, using at most two batched DCP reads for the request. Group
members are already confirmed. Group results expose both the group entity
and members for the client to choose.

All six result statuses return HTTP 200. A non-blocking semaphore covers
the resolver call and region reads. A full pool returns retryable 503.
One absolute monotonic deadline covers every DCP hop. `wait_for(shield(task))`
returns 504 to a timed-out client while the worker finishes its reads. The
semaphore stays held until those reads finish. Resolvekit matching uses its
own single-worker store executor, outside the DCP deadline.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import BrokenExecutor

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.executor import submit_with_context
from dfo_platform.refusal import TIMEOUT_DEFAULT
from entity_resolution import (
    Ambiguous,
    Entity,
    EntityResolutionError,
    Group,
    NotFound,
    NotHeld,
    Resolution,
    Resolved,
)
from graph_api.config import GRAPH_RESOLVE_TIMEOUT_SECONDS
from graph_api.deps import GraphDeps, GraphDepsDep
from graph_api.reader import NodeArcsByDcid
from graph_api.resolve_models import (
    AmbiguousResult,
    EntityRef,
    NotFoundResult,
    NotHeldResult,
    RegionMember,
    ResolvedGroupResult,
    ResolvedRegionResult,
    ResolvedResult,
    ResolveRequest,
    ResolveResponse,
    ResolveResult,
)
from materialise.graph import GraphDeadlineExceededError, GraphReadError

logger = logging.getLogger(__name__)

_REGION_KINDS = frozenset({"continent", "subregion"})
_REGION_CHILD_TYPE = "Country"


class GraphResolvePoolExhausted(Exception):
    """The DCP admission semaphore is full. The door maps this to retryable 503."""


def _entity_ref(entity: Entity) -> EntityRef:
    return EntityRef(
        dcid=entity.dcid, name=entity.name, entity_type=entity.graph_type, kind=entity.kind
    )


def _member_name(names_by_dcid: NodeArcsByDcid, dcid: str) -> str:
    arcs = names_by_dcid.get(dcid) or {}
    values = arcs.get("name", ())
    return values[0] if values else dcid


async def _run_graph[T](deps: GraphDeps, fn: Callable[..., T], *args: object) -> T:
    """Submit `fn(*args)` under the semaphore `_run_resolve` holds for the whole request."""
    executor = deps.dcp_executor
    try:
        future = submit_with_context(executor, fn, *args)
    except BrokenExecutor:
        logger.error(
            "Graph resolve door's own DCP-read executor is permanently broken; "
            "every future submitted to it will fail the same way",
            exc_info=True,
        )
        raise
    except RuntimeError:
        raise GraphResolvePoolExhausted() from None
    return await asyncio.wrap_future(future)


async def _region_members(
    deps: GraphDeps, resolutions: Sequence[Resolution], *, deadline: float
) -> tuple[dict[str, list[str]], NodeArcsByDcid]:
    """Batch region children, then member names. Return region-to-member dcids and name
    arcs, or two empty maps when no region resolves. Pass the absolute deadline to both reads.
    """
    graph = deps.graph
    region_dcids: dict[str, None] = {}
    for resolution in resolutions:
        if isinstance(resolution, Resolved) and resolution.entity.kind in _REGION_KINDS:
            region_dcids.setdefault(resolution.entity.dcid, None)
    if not region_dcids:
        return {}, {}

    members_by_dcid = await _run_graph(
        deps,
        functools.partial(
            graph.child_places_batch, child_type=_REGION_CHILD_TYPE, deadline=deadline
        ),
        list(region_dcids),
    )
    member_dcids: dict[str, None] = {}
    for members in members_by_dcid.values():
        for member in members:
            member_dcids.setdefault(member, None)

    names_by_dcid: NodeArcsByDcid = {}
    if member_dcids:
        names_by_dcid = await _run_graph(
            deps,
            functools.partial(graph.node_arcs_many, properties=("name",), deadline=deadline),
            list(member_dcids),
        )
    return members_by_dcid, names_by_dcid


def _assemble(
    resolution: Resolution, members_by_dcid: dict[str, list[str]], names_by_dcid: NodeArcsByDcid
) -> ResolveResult:
    if isinstance(resolution, Resolved):
        entity = resolution.entity
        if entity.kind in _REGION_KINDS:
            members = [
                RegionMember(dcid=member, name=_member_name(names_by_dcid, member))
                for member in members_by_dcid.get(entity.dcid, [])
            ]
            return ResolvedRegionResult(region=_entity_ref(entity), members=members)
        return ResolvedResult(entity=_entity_ref(entity))
    if isinstance(resolution, Group):
        return ResolvedGroupResult(
            entity=_entity_ref(resolution.entity) if resolution.entity is not None else None,
            members=[_entity_ref(member) for member in resolution.members],
        )
    if isinstance(resolution, Ambiguous):
        return AmbiguousResult(candidates=[_entity_ref(e) for e in resolution.candidates])
    if isinstance(resolution, NotHeld):
        return NotHeldResult(name=resolution.name, dcid=resolution.dcid)
    assert isinstance(resolution, NotFound)
    return NotFoundResult(near_matches=list(resolution.near_matches))


async def _run_resolve(deps: GraphDeps, texts: list[str]) -> ResolveResponse:
    """Refresh the generation before setting the deadline or acquiring the semaphore.
    This ensures `resolver.resolve` sees a fresh stamp after an idle gap.

    Compute the absolute deadline before admission so executor queue time
    counts against it. The resolver and region reads share it. The caller
    shields this coroutine, so the `finally` releases the semaphore only
    after all submitted work finishes.
    """
    resolver = deps.resolver
    semaphore = deps.dcp_semaphore

    await deps.generation.refresh_if_stale()

    deadline = time.monotonic() + GRAPH_RESOLVE_TIMEOUT_SECONDS
    if not semaphore.acquire(blocking=False):
        raise GraphResolvePoolExhausted()
    try:
        resolutions = await resolver.resolve(texts, deadline=deadline)
        members_by_dcid, names_by_dcid = await _region_members(deps, resolutions, deadline=deadline)
    finally:
        semaphore.release()

    return ResolveResponse(
        results=[_assemble(r, members_by_dcid, names_by_dcid) for r in resolutions]
    )


# Keep shielded tasks alive. The event loop holds only weak task references.
_pending_resolves: set[asyncio.Task[ResolveResponse]] = set()


def _log_detached_failure(task: asyncio.Task[ResolveResponse]) -> None:
    """Log failures from work that continued after the client timed out. Check cancellation
    because `exception()` raises on cancelled tasks.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "Graph resolve door: a request already timed out client-side, but its own "
            "work kept running afterward and then failed: %s",
            exc,
            exc_info=exc,
        )


def classify_resolve_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify shared graph-read failures as well as resolve failures. `EntityResolutionError`
    falls through to the platform's 500 handler. `GraphDeadlineExceededError` subclasses
    `GraphReadError`, so its 504 branch must come first. Classifiers run across all doors.
    This one is registered first and handles deadline errors from node and sources too.
    """
    if isinstance(exc, GraphResolvePoolExhausted):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the resolve door is at its DCP confirmation capacity right now; retry",
            retryable=True,
        )
    if isinstance(exc, GraphDeadlineExceededError):
        # `GraphDeadlineExceededError` subclasses `GraphReadError`. Map it first to 504.
        return ProblemSpec(
            status=TIMEOUT_DEFAULT.status,
            code=TIMEOUT_DEFAULT.code,
            detail="the request's deadline expired during a graph read; retry",
            retryable=TIMEOUT_DEFAULT.retryable,
        )
    if isinstance(exc, GraphReadError):
        logger.error("Graph resolve door's DCP confirmation read failed: %s", exc, exc_info=exc)
        upstream_status = exc.upstream_status
        if isinstance(upstream_status, int) and 400 <= upstream_status < 500:
            return ProblemSpec(status=502, code="upstream_error", retryable=True)
        return ProblemSpec(status=503, code="service_unavailable", retryable=True)
    return None


async def resolve_endpoint(body: ResolveRequest, deps: GraphDepsDep) -> JSONResponse:
    # Shield work from client cancellation.
    # `_run_resolve` releases admission after its reads finish.
    task: asyncio.Task[ResolveResponse] = asyncio.ensure_future(_run_resolve(deps, body.texts))
    _pending_resolves.add(task)
    task.add_done_callback(_pending_resolves.discard)
    try:
        response = await asyncio.wait_for(
            asyncio.shield(task), timeout=GRAPH_RESOLVE_TIMEOUT_SECONDS
        )
    except TimeoutError:
        task.add_done_callback(_log_detached_failure)
        raise
    return JSONResponse(response.model_dump(mode="json"))


def build_resolve_door() -> DoorContribution:
    router = APIRouter()
    # `response_model` documents the results list. JSONResponse passes it through unchanged.
    router.add_api_route(
        "/api/graph/resolve",
        resolve_endpoint,
        methods=["POST"],
        response_model=ResolveResponse,
    )
    return DoorContribution(
        router=router,
        classify=classify_resolve_exception,
        exception_types=(
            GraphResolvePoolExhausted,
            GraphDeadlineExceededError,
            GraphReadError,
            EntityResolutionError,
        ),
        exception_exemplars=(
            GraphResolvePoolExhausted(),
            GraphDeadlineExceededError("exemplar"),
            GraphReadError("exemplar", upstream_status=502),
            EntityResolutionError("exemplar"),
        ),
    )
