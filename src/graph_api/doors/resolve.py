"""Resolve text through `entity_resolution` and map its results to the Graph API.

The shared resolver handles matching, the listed-name gate, the `kinds` filter, codes, group
membership and graph confirmation. A group comes from the membership registry, which confirms
the group and its members without a graph read; its count and inline members are the
resolver's answer under the `countries` projection. A continent or subregion, and a group
resolved as a single node beside other candidates, takes its count, link and inline members
from the membership policy the surface publishes (`NODE_MEMBERSHIP_SLOT`), never from graph
containment. It takes the policy's canonical name, as the node route labels it, and its inline
members take theirs (`Membership.member_labels`), as the members route labels them.

All four statuses return HTTP 200. The resolver's confirmation read queues on the shared
DCP-read executor. One absolute monotonic deadline covers every DCP hop and the resolvekit job. When `GRAPH_RESOLVE_TIMEOUT_SECONDS` passes, `wait_for` cancels the request
and returns 504. A read still queued is cancelled before it runs, and a running read keeps its
worker until it returns.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
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
from entity_resolution.membership import (
    DEFAULT_PROJECTION,
    Membership,
    MembershipPolicy,
    NotRecorded,
)
from graph_api.config import GRAPH_RESOLVE_TIMEOUT_SECONDS
from graph_api.deps import GraphDeps, GraphDepsDep
from graph_api.doors.node import published_membership
from graph_api.node_models import LinkRelation, members_link
from graph_api.refs import NodeRef
from graph_api.resolve_models import (
    MAX_INLINE_MEMBERS,
    EntityRef,
    Matched,
    ResolveRequest,
    ResolveResponse,
    ResolveResult,
)
from materialise.graph import GraphDeadlineExceededError, GraphReadError

logger = logging.getLogger(__name__)

# The relation a held group's or region's link names, by the kind the policy gives it.
_RELATION_BY_KIND: dict[str, LinkRelation] = {
    "group": "members",
    "continent": "countries",
    "subregion": "countries",
}


def _entity_ref(entity: Entity) -> EntityRef:
    return EntityRef(ref=entity.dcid, label=entity.name, kind=entity.kind)


def _matched(resolution: Resolved | Group) -> Matched | None:
    matched = resolution.matched
    return None if matched is None else Matched(text=matched.text, as_=matched.matched_as)


def _near_matches(resolution: NotFound) -> list[EntityRef]:
    """The candidates a `kinds` filter removed, then the gate's rejected hints, one per ref. A
    removed candidate keeps its graph-confirmed kind over a hint for the same ref."""
    unique: dict[str, Entity] = {}
    for entity in (*resolution.removed, *resolution.near_matches):
        unique.setdefault(entity.dcid, entity)
    return [_entity_ref(entity) for entity in unique.values()]


def _held_relation(entity: Entity, policy: MembershipPolicy | None) -> LinkRelation | None:
    """The relation of the group or region `entity` names when the policy holds it, else
    `None`."""
    if policy is None or entity.dcid not in policy:
        return None
    return _RELATION_BY_KIND.get(entity.kind)


def _policy_members(membership: Membership) -> list[NodeRef]:
    """A group's or region's first `MAX_INLINE_MEMBERS` members under the default projection,
    each with the policy's canonical name."""
    members = membership.members(DEFAULT_PROJECTION)
    recorded = () if isinstance(members, NotRecorded) else members
    return [
        NodeRef(ref=m, label=membership.member_labels.get(m)) for m in recorded[:MAX_INLINE_MEMBERS]
    ]


def _assemble(
    text: str, resolution: Resolution, *, policy: MembershipPolicy | None, expand: bool
) -> ResolveResult:
    if isinstance(resolution, Group):
        assert resolution.dcid is not None and resolution.label is not None, "a registry group"
        return ResolveResult(
            input=text,
            status="resolved",
            entity=EntityRef(ref=resolution.dcid, label=resolution.label, kind="group"),
            matched=_matched(resolution),
            reading=resolution.reading,
            projection=DEFAULT_PROJECTION,
            member_count=resolution.member_count,
            members=(
                [NodeRef(ref=m.dcid, label=m.name) for m in resolution.members[:MAX_INLINE_MEMBERS]]
                if expand
                else members_link(resolution.dcid, "members")
            ),
        )
    if isinstance(resolution, Resolved):
        result = ResolveResult(
            input=text,
            status="resolved",
            entity=_entity_ref(resolution.entity),
            matched=_matched(resolution),
            reading=resolution.reading,
            near_matches=[_entity_ref(e) for e in resolution.near_matches] or None,
        )
        relation = _held_relation(resolution.entity, policy)
        if relation is None:
            return result
        assert policy is not None
        membership = policy.get(resolution.entity.dcid)
        count = membership.member_count(DEFAULT_PROJECTION)
        return result.model_copy(
            update={
                "entity": _entity_ref(resolution.entity).model_copy(
                    update={"label": membership.label}
                ),
                "projection": DEFAULT_PROJECTION,
                "member_count": None if isinstance(count, NotRecorded) else count,
                "members": (
                    _policy_members(membership)
                    if expand
                    else members_link(membership.ref, relation)
                ),
            }
        )
    if isinstance(resolution, Ambiguous):
        return ResolveResult(
            input=text,
            status="ambiguous",
            candidates=[_entity_ref(e) for e in resolution.candidates],
            near_matches=[_entity_ref(e) for e in resolution.near_matches] or None,
        )
    if isinstance(resolution, NotHeld):
        return ResolveResult(
            input=text,
            status="not_held",
            entity=EntityRef(ref=resolution.dcid, label=resolution.name, kind=None),
        )
    assert isinstance(resolution, NotFound)
    return ResolveResult(
        input=text,
        status="not_found",
        near_matches=_near_matches(resolution),
    )


async def _run_resolve(
    deps: GraphDeps, body: ResolveRequest, *, policy: MembershipPolicy | None
) -> ResolveResponse:
    """Refresh the generation before setting the deadline, so `resolver.resolve` sees a fresh
    stamp after an idle gap.

    The resolver's reads share one absolute deadline, set before their first submission so
    executor queue time counts against it.
    """
    await deps.generation.refresh_if_stale()

    deadline = time.monotonic() + GRAPH_RESOLVE_TIMEOUT_SECONDS
    resolutions = await deps.resolver.resolve(
        body.texts,
        kinds=body.kinds,
        code_system=body.code_system,
        near_matches=True,
        deadline=deadline,
    )
    expand = "members" in body.expand
    return ResolveResponse(
        results=[
            _assemble(text, r, policy=policy, expand=expand)
            for text, r in zip(body.texts, resolutions, strict=True)
        ]
    )


def classify_resolve_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify the resolve route's graph-read failures. `EntityResolutionError` falls through
    to the platform's 500 handler. `GraphDeadlineExceededError` subclasses `GraphReadError`, so
    its 504 branch must come first.
    """
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


async def resolve(body: ResolveRequest, deps: GraphDepsDep, request: Request) -> JSONResponse:
    response = await asyncio.wait_for(
        _run_resolve(deps, body, policy=published_membership(request)),
        timeout=GRAPH_RESOLVE_TIMEOUT_SECONDS,
    )
    return JSONResponse(response.model_dump(mode="json"))


def build_resolve_door() -> DoorContribution:
    router = APIRouter()
    # `response_model` documents the results list. JSONResponse passes it through unchanged.
    router.add_api_route(
        "/api/graph/resolve",
        resolve,
        methods=["POST"],
        response_model=ResolveResponse,
    )
    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "graph.resolve"),
        classify=classify_resolve_exception,
        exception_types=(
            GraphDeadlineExceededError,
            GraphReadError,
            EntityResolutionError,
        ),
        exception_exemplars=(
            GraphDeadlineExceededError("exemplar"),
            GraphReadError("exemplar", upstream_status=502),
            EntityResolutionError("exemplar"),
        ),
    )
