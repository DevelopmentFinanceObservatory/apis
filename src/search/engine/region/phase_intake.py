"""Resolve recall candidates, place sets, and shape compatibility before ranking."""

from __future__ import annotations

import asyncio
from typing import cast

from entity_resolution import Ambiguous, Group, Resolved
from search.engine.clients.graph import force_full_facets
from search.engine.config import settings
from search.engine.envelope import _record_step, now_ms
from search.engine.errors import RoleSignatureError
from search.engine.interpret import Recall, recall
from search.engine.place import expand_places
from search.engine.region.codes import CHILD_PLACES_CLAMPED
from search.engine.region.result import (
    AmbiguousEntity,
    AmbiguousEntityCandidate,
    RegionResult,
    RoleNotAdmittedEntity,
)
from search.engine.region.state import ResolveState
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.topics import is_topic_candidate
from search.engine.shape import ShapeDraft, is_role_bearing
from search.models import RoleAxis, Warning

# Full-facet probing is capped at 6 candidates. This phase checks raw recall count before
# expansion. derive_shapes checks the confirmed count to bound the actual read.
_FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES = 6


async def _phase_recall(state: ResolveState) -> None:
    """Run recall and record its candidates for place-set resolution."""
    t0 = now_ms()
    rcl: Recall = await recall(
        state.variable,
        state.entities,
        graph=state.graph,
        resolver=state.resolver,
        raw_query=state.detect_query,
        pre_resolved=state.pre_resolved,
    )
    state.rcl = rcl
    state.timing["recall"] = now_ms() - t0
    await _record_step(
        state.pipeline_steps,
        state.emit,
        state.variable_index,
        "recall",
        ran=True,
        ms=state.timing["recall"],
    )


async def _phase_resolve_places(state: ResolveState) -> None:
    """Expand Groups and place sets. Leave oversized Groups unresolved.

    Snapshot recall's country DCIDs before expansion for the availability floor. Expand
    continent/subregion entities in one batch for observation probes. Other resolved kinds
    retain their ordinary binding path. Unresolved names are refused later.
    """
    # The type checker narrows locals, while state attributes need this explicit cast.
    rcl = cast(Recall, state.rcl)

    # Key place expansions by query name. Groups may have no DCID of their own.
    place_children_by_name: dict[str, list[str]] = {}
    state.place_children_by_name = place_children_by_name

    # Groups use resolver-confirmed membership and bypass child-place lookup. Bind every
    # member, bounded by ``max_child_places``. Members share a graph type, so the first
    # supplies the placeholder ``Resolved`` value for admission checks. The group's own
    # node, when the resolver returned one, is kept by name for the Spec's entity groups.
    group_member_dcids_by_name = state.group_member_dcids_by_name
    group_member_names_by_dcid = state.group_member_names_by_dcid
    for name, resolution in list(rcl.resolutions.items()):
        if not isinstance(resolution, Group):
            continue
        if not resolution.members or len(resolution.members) > settings().max_child_places:
            # Keep empty or oversized groups unresolved. Truncating members would misstate
            # the population covered by the answer.
            rcl.resolved_entity_names.pop(name, None)
            continue
        member_dcids = [member.dcid for member in resolution.members]
        rcl.resolutions[name] = Resolved(entity=resolution.members[0])
        rcl.resolved_entity_names[name] = resolution.members[0].dcid
        place_children_by_name[name] = member_dcids
        group_member_dcids_by_name[name] = member_dcids
        if resolution.entity is not None:
            state.group_dcid_by_name[name] = resolution.entity.dcid
        for member in resolution.members:
            group_member_names_by_dcid[member.dcid] = member.name
    # A group with its own node is named by the graph's label for it (`request_named_groups`).
    if group_dcids := sorted(set(state.group_dcid_by_name.values())):
        state.group_label_by_dcid = await asyncio.to_thread(
            state.graph.node_labels_batch, group_dcids
        )

    # Preserve recall's country set for derive_shapes' availability floor. Exclude Group
    # placeholders and place sets. Group members enter the later sibling probe sample.
    state.recall_only_place_dcids = (
        list(
            dict.fromkeys(
                resolution.entity.dcid
                for name, resolution in rcl.resolutions.items()
                if isinstance(resolution, Resolved)
                and resolution.entity.kind == "country"
                and name not in group_member_dcids_by_name
            )
        )
        or None
    )

    # Keep each place-set's own DCID for binding. Use expanded members only for observation
    # probes. Expand all continent/subregion sets in one batch after Groups are settled.
    to_expand = [
        (name, resolution.entity)
        for name, resolution in rcl.resolutions.items()
        if isinstance(resolution, Resolved) and resolution.entity.kind in ("continent", "subregion")
    ]
    if to_expand:
        expansions = await asyncio.to_thread(
            expand_places,
            [(entity.dcid, entity.graph_type) for _, entity in to_expand],
            graph=state.graph,
        )
        for name, entity in to_expand:
            expansion = expansions.get(entity.dcid)
            if expansion is None:
                continue
            place_children_by_name[name] = expansion.children
            if expansion.clamped:
                state.warnings.append(
                    Warning(
                        code=CHILD_PLACES_CLAMPED,
                        severity="warn",
                        message=(
                            f"{name!r} expands to {expansion.total_members} places; "
                            f"resolved the first {len(expansion.children)}."
                        ),
                    )
                )


def _sole_axis(state: ResolveState, name: str) -> RoleAxis | None:
    """The one flow axis the door gave `name`, or `None` when it gave none or both."""
    axes = {role.axis for role in state.entity_roles.get(name, ())}
    return axes.pop() if len(axes) == 1 else None


def _unresolved_entity_terminal(state: ResolveState) -> RegionResult | None:
    """Return an unresolved-entity result, preserving entity-before-variable precedence.

    Report ``entity_ambiguous`` only when every unresolved name is ambiguous. Otherwise
    report ``entity_not_resolved``. Keep names and candidates in query order.
    """
    rcl = cast(Recall, state.rcl)

    # The normal check runs after election so a place miss can retain StatVar evidence.
    # Earlier terminals call it first to preserve entity-before-variable precedence.
    # Keep unresolved names in query order for the response.
    distinct_extracted = len(set(state.entities))
    if state.entities and len(rcl.resolved_entity_names) < distinct_extracted:
        still_unresolved = set(state.entities) - set(rcl.resolved_entity_names)
        still_unresolved_ordered = tuple(
            dict.fromkeys(name for name in state.entities if name in still_unresolved)
        )
        if still_unresolved and all(
            isinstance(rcl.resolutions.get(name), Ambiguous) for name in still_unresolved
        ):
            ambiguous_entities = tuple(
                AmbiguousEntity(
                    text=name,
                    candidates=tuple(
                        AmbiguousEntityCandidate(
                            dcid=candidate.dcid, name=candidate.name, kind=candidate.kind
                        )
                        for candidate in cast(Ambiguous, rcl.resolutions[name]).candidates
                    ),
                    role=_sole_axis(state, name),
                )
                for name in still_unresolved_ordered
            )
            # A group stays unpinned: re-resolving its text keeps its members.
            resolved_entity_dcids = {
                name: resolution.entity.dcid
                for name in dict.fromkeys(state.entities)
                if isinstance(resolution := rcl.resolutions.get(name), Resolved)
            }
            return state.no_data(
                "entity_ambiguous",
                ambiguous_entities=ambiguous_entities,
                resolved_entity_dcids=resolved_entity_dcids,
            )
        return state.no_data(
            "entity_not_resolved", unresolved_entity_names=still_unresolved_ordered
        )
    return None


async def _phase_check_entities(state: ResolveState) -> RegionResult | None:
    """Check unresolved entities after election. Earlier terminals check them first."""
    return _unresolved_entity_terminal(state)


async def _prune_shapes_by_admission(
    state: ResolveState, shapes: list[ShapeDraft]
) -> tuple[list[ShapeDraft], list[RoleNotAdmittedEntity]]:
    """Keep shapes whose declared roles admit every resolved entity.

    Return rejected-entity details only when all shapes fail, in query order. The fallback
    names all entities when each is admitted somewhere but no single shape admits all.
    """
    rcl = cast(Recall, state.rcl)
    resolved: list[tuple[str, Resolved]] = [
        (name, cast(Resolved, rcl.resolutions[name]))
        for name in dict.fromkeys(state.entities)
        if name in rcl.resolved_entity_names
    ]
    if not resolved or not shapes:
        return shapes, []

    async def _admits_any(roles: tuple[str, ...], graph_type: str) -> bool:
        for role in roles:
            if await state.admission.admits(role, graph_type):
                return True
        return False

    survivors: list[ShapeDraft] = []
    admitted_anywhere: dict[str, bool] = {
        resolution.entity.dcid: False for _, resolution in resolved
    }
    for shape in shapes:
        roles = shape.declared_roles or ("observationAbout",)
        shape_admits_all = True
        for _, resolution in resolved:
            entity = resolution.entity
            if await _admits_any(roles, entity.graph_type):
                admitted_anywhere[entity.dcid] = True
            else:
                shape_admits_all = False
        if shape_admits_all:
            survivors.append(shape)

    if survivors:
        return survivors, []

    never_admitted = [
        RoleNotAdmittedEntity(
            text=name, dcid=resolution.entity.dcid, graph_type=resolution.entity.graph_type
        )
        for name, resolution in resolved
        if not admitted_anywhere[resolution.entity.dcid]
    ] or [
        RoleNotAdmittedEntity(
            text=name, dcid=resolution.entity.dcid, graph_type=resolution.entity.graph_type
        )
        for name, resolution in resolved
    ]
    return [], never_admitted


async def _phase_derive_shapes(state: ResolveState) -> RegionResult | None:
    """Confirm recall candidates and derive role-compatible shapes.

    Return ``role_signature_incompatible`` or ``variable_not_resolved`` when derivation
    fails, and ``entity_role_not_admitted`` when all shapes fail role checks. Unresolved
    entity errors take precedence. Surviving shapes proceed to ranking.
    """
    rcl = cast(Recall, state.rcl)
    pool = rcl.candidates if state.pool == "ours" else rcl.un_candidates
    pool_svs = [c.dcid for c in pool]

    # Count non-Topic recalls before derivation. Topic expansions contribute their members.
    state.n_recalled = sum(1 for c in pool if not is_topic_candidate(c.dcid, c.type_of))

    # Confirm candidates and derive shapes off the event loop.
    t0 = state.shape_t0 = now_ms()
    sv_scores_map = {c.dcid: c.score for c in pool}
    sv_types_map = {c.dcid: c.type_of for c in pool}
    try:
        # Dated observation reads use ``needs_dates=True``. This scope lets the availability
        # probe share that cache key when raw recall has at most 6 candidates. derive_shapes
        # rechecks the limit after confirmation and Topic expansion, bounding the full-history
        # read to 6 confirmed StatVars. The context propagates through ``to_thread``.
        with force_full_facets(
            state.date_request is not None
            and len(pool_svs) <= _FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES
        ):
            shapes_found: list[ShapeDraft] = await asyncio.to_thread(
                derive_shapes,
                confirmed_svs=pool_svs,
                graph=state.graph,
                sv_scores=sv_scores_map,
                sv_types=sv_types_map,
                # Use recall's country snapshot. Place sets and Group members stay off the floor.
                place_dcids=state.recall_only_place_dcids,
                full_facets_max_candidates=_FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES,
            )
            state.shapes_found = shapes_found
    except RoleSignatureError:
        # No candidate group has a consistent entity-role signature.
        state.timing["shape"] = now_ms() - t0
        await _record_step(
            state.pipeline_steps,
            state.emit,
            state.variable_index,
            "shape",
            ran=True,
            ms=state.timing["shape"],
        )
        state.ran_steps.add("shape")
        return _unresolved_entity_terminal(state) or state.no_data("role_signature_incompatible")

    # No confirmed StatVar means there is no shape to bind.
    if not shapes_found:
        state.timing["shape"] = now_ms() - t0
        await _record_step(
            state.pipeline_steps,
            state.emit,
            state.variable_index,
            "shape",
            ran=True,
            ms=state.timing["shape"],
        )
        state.ran_steps.add("shape")
        return _unresolved_entity_terminal(state) or state.no_data("variable_not_resolved")

    # Prune before ranking so every later count sees only compatible shapes. Unresolved
    # entities retain precedence over role-admission failures.
    shapes_found, never_admitted = await _prune_shapes_by_admission(state, shapes_found)
    state.shapes_found = shapes_found
    if not shapes_found:
        state.timing["shape"] = now_ms() - t0
        await _record_step(
            state.pipeline_steps,
            state.emit,
            state.variable_index,
            "shape",
            ran=True,
            ms=state.timing["shape"],
        )
        state.ran_steps.add("shape")
        return _unresolved_entity_terminal(state) or state.no_data(
            "entity_role_not_admitted", role_not_admitted_entities=tuple(never_admitted)
        )

    # Detect role-bearing families across the full set before election.
    state.role_bearing_recalled = any(is_role_bearing(s) for s in shapes_found)
    return None
