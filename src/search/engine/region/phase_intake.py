"""Resolve recall candidates, place sets, and shape compatibility before ranking."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

from entity_resolution import Ambiguous, Group, Resolved
from search.engine.clients.graph import force_full_facets
from search.engine.config import settings
from search.engine.envelope import _emit_step, now_ms
from search.engine.errors import RoleSignatureError
from search.engine.interpret import Recall, recall
from search.engine.place import expand_places
from search.engine.region.place_roles import (
    PlaceAssignment,
    SetChoice,
    bind_place_roles,
    part_sets,
    place_key,
    set_choices,
)
from search.engine.region.place_set import PlaceNode, PlaceReading, PlaceSet
from search.engine.region.result import (
    AmbiguousEntity,
    AmbiguousEntityCandidate,
    RegionResult,
    RoleNotAdmittedEntity,
)
from search.engine.region.state import ResolveState
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft, is_role_bearing, role_names_by_axis
from search.models import EntityRoleKey, RoleAxis

# Full-facet probing is capped at 6 candidates. This phase checks raw recall count before
# expansion. derive_shapes checks the confirmed count to bound the actual read.
_FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES = 6

# The resolver kinds of a place that has member countries.
_REGION_KINDS = ("continent", "subregion")


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
    await _emit_step(
        state.emit,
        state.variable_index,
        "recall",
        ms=state.timing["recall"],
    )


async def _phase_resolve_places(state: ResolveState) -> None:
    """Build the place set of every name recall resolved (`state.place_sets`).

    A group is a set of its resolver-confirmed members, with the group's own node when the
    resolver returned one. An empty or oversized group gets no set, so its name stays
    unresolved and is refused later. A continent or subregion is a set of its node and the
    members one batched expansion gives. Every other resolved name is a set of its node
    alone. The node of a group or region carries the graph's label for it, read in one
    batch beside the expansion. A set takes the members reading when the door asks for the
    name's members (`ResolveState.member_entities`) or the resolver read the name as its
    members (`names_members`). Recall's country dcids are snapshotted for the availability
    floor.
    """
    # The type checker narrows locals, while state attributes need this explicit cast.
    rcl = cast(Recall, state.rcl)

    # Truncating an oversized group's members would misstate the population the answer
    # covers. `_unresolved_entity_terminal` answers by its `Group` resolution.
    groups = {
        name: resolution
        for name, resolution in rcl.resolutions.items()
        if isinstance(resolution, Group)
        and resolution.members
        and not _is_oversized_group(resolution)
    }
    resolved = {
        name: resolution.entity
        for name, resolution in rcl.resolutions.items()
        if isinstance(resolution, Resolved)
    }
    regions = [entity for entity in resolved.values() if entity.kind in _REGION_KINDS]

    # Preserve recall's country set for derive_shapes' availability floor. Groups and
    # regions stay off it. Their nodes and members enter the later sibling probe sample.
    state.recall_only_place_dcids = (
        list(dict.fromkeys(e.dcid for e in resolved.values() if e.kind == "country")) or None
    )

    # A set served as a group is named by the graph's label for its node
    # (`request_named_groups`). All continent and subregion sets expand in one batch.
    labelled = sorted(
        {g.entity.dcid for g in groups.values() if g.entity is not None}
        | {entity.dcid for entity in regions}
    )
    labels, expansions = await asyncio.gather(
        asyncio.to_thread(state.graph.node_labels_batch, labelled) if labelled else _nothing(),
        asyncio.to_thread(
            expand_places,
            [(entity.dcid, entity.graph_type) for entity in regions],
            graph=state.graph,
        )
        if regions
        else _nothing(),
    )

    def reading(name: str) -> PlaceReading:
        resolution = cast(Resolved | Group, rcl.resolutions[name])
        asked = name in state.member_entities or resolution.names_members
        return "members" if asked else "default"

    place_sets: dict[str, PlaceSet] = {}
    for name, group in groups.items():
        node = group.entity
        place_sets[name] = PlaceSet(
            name=name,
            kind="group",
            node=None if node is None else PlaceNode(dcid=node.dcid, graph_type=node.graph_type),
            label=None if node is None else labels.get(node.dcid),
            members=tuple(member.dcid for member in group.members),
            # The resolver states no type for a member the graph gives none.
            member_type=group.members[0].graph_type or None,
            member_names={member.dcid: member.name for member in group.members},
            reading=reading(name),
        )
    for name, entity in resolved.items():
        region = entity.kind in _REGION_KINDS
        expansion = expansions.get(entity.dcid) if region else None
        place_sets[name] = PlaceSet(
            name=name,
            kind="region" if region else "place",
            node=PlaceNode(dcid=entity.dcid, graph_type=entity.graph_type),
            label=labels.get(entity.dcid) if region else None,
            members=None if expansion is None else tuple(expansion.children),
            member_type=None if expansion is None else expansion.child_type,
            reading=reading(name),
        )
    state.place_sets = place_sets


async def _nothing() -> dict:
    """The result of a batched read that names no node."""
    return {}


def _sole_axis(state: ResolveState, name: str) -> RoleAxis | None:
    """The one flow axis the door gave `name`, or `None` when it gave none or both."""
    axes = {role.axis for role in state.entity_roles.get(name, ())}
    return axes.pop() if len(axes) == 1 else None


def _is_oversized_group(resolution: object) -> bool:
    """Whether `resolution` is a Group with more members than a Spec binds
    (``max_child_places``)."""
    return isinstance(resolution, Group) and len(resolution.members) > settings().max_child_places


def _unresolved_entity_terminal(state: ResolveState) -> RegionResult | None:
    """Return an unresolved-entity result, preserving entity-before-variable precedence.

    Report ``entity_ambiguous`` only when every unresolved name is ambiguous, and
    ``entity_set_too_large`` only when every unresolved name is an oversized Group.
    Otherwise report ``entity_not_resolved``. Keep names and candidates in query order.
    """
    rcl = cast(Recall, state.rcl)

    # The normal check runs after election so a place miss can retain StatVar evidence.
    # Earlier terminals call it first to preserve entity-before-variable precedence.
    # Keep unresolved names in query order for the response.
    still_unresolved = set(state.entities) - state.place_sets.keys()
    if still_unresolved:
        still_unresolved_ordered = tuple(
            dict.fromkeys(name for name in state.entities if name in still_unresolved)
        )
        if all(isinstance(rcl.resolutions.get(name), Ambiguous) for name in still_unresolved):
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
            # A group, and a set read as its members, stay unpinned: re-resolving the text
            # keeps the members, whether an alias row or the part's flag asked for them.
            resolved_entity_dcids = {
                name: place_set.node.dcid
                for name in dict.fromkeys(state.entities)
                if (place_set := state.place_sets.get(name)) is not None
                and place_set.kind != "group"
                and place_set.reading != "members"
                and place_set.node is not None
            }
            return state.no_data(
                "entity_ambiguous",
                ambiguous_entities=ambiguous_entities,
                resolved_entity_dcids=resolved_entity_dcids,
            )
        if all(_is_oversized_group(rcl.resolutions.get(name)) for name in still_unresolved):
            return state.no_data("entity_set_too_large")
        # An oversized group resolved, so it is not among the names that did not.
        return state.no_data(
            "entity_not_resolved",
            unresolved_entity_names=tuple(
                name
                for name in still_unresolved_ordered
                if not _is_oversized_group(rcl.resolutions.get(name))
            ),
        )
    return None


async def _phase_check_entities(state: ResolveState) -> RegionResult | None:
    """Check unresolved entities after election. Earlier terminals check them first."""
    return _unresolved_entity_terminal(state)


async def _prune_shapes_by_admission(
    state: ResolveState, shapes: list[ShapeDraft]
) -> tuple[list[ShapeDraft], list[RoleNotAdmittedEntity]]:
    """Keep shapes that can serve every place set the part names, each once however many
    texts name it (`place_roles.part_sets`).

    A shape serves a set when a declared role admits the type of a reading the set can
    take: its node's or the one its members share, so a continent passes on a role that
    takes countries, and for a set asked for its members only theirs (`_admission_types`).
    Where the shape assigns the query's sets to its roles (`place_roles.bind_place_roles`),
    it must also give each set a reading every role it fills admits, by the rule settling
    applies (`place_roles.set_choices`), members asks from a rank included. A shape kept
    here then settles every set. A shape whose places cannot be assigned yet, as when an
    undirected place waits on the elected shape's axis, is judged by its declared roles
    alone.

    Every question settling can need is asked here first, in one gathered batch: each
    declared role of each shape, with the node type and the member type of each set that
    holds members (`place_roles.set_choices`). The answers are recorded by role and type
    (`state.node_admissions`), so settling reads them in hand and a member type's class
    walk runs beside a node type's. A set with no member needs no question of settling, and
    is judged as before, a role and type at a time until one admits it.

    Return rejected-entity details only when all shapes fail, in query order. The fallback
    names all entities when each is admitted somewhere but no single shape admits all.
    """
    place_sets = list(part_sets(state).values())
    if not place_sets or not shapes:
        return shapes, []

    unasked = sorted(
        {
            (role, graph_type)
            for shape in shapes
            for role in shape.declared_roles or ("observationAbout",)
            for place_set in place_sets
            if place_set.members
            for graph_type in _set_types(place_set)
        }
        - state.node_admissions.keys()
    )
    answers = await asyncio.gather(
        *[state.admission.admits(role, graph_type) for role, graph_type in unasked]
    )
    state.node_admissions.update(zip(unasked, answers, strict=True))

    async def _admits(role: str, graph_type: str) -> bool:
        asked = (role, graph_type)
        if asked not in state.node_admissions:
            state.node_admissions[asked] = await state.admission.admits(role, graph_type)
        return state.node_admissions[asked]

    async def _admits_any(roles: tuple[str, ...], place_set: PlaceSet) -> bool:
        for role in roles:
            for graph_type in _admission_types(place_set):
                if await _admits(role, graph_type):
                    return True
        return False

    def _readings(shape: ShapeDraft) -> Mapping[str, SetChoice | None]:
        """The readings `shape` gives each set it assigns (`place_roles.set_choices`), from
        the answers in hand; empty when the shape cannot assign the query's places."""
        assignment = bind_place_roles(state, shape)
        if not isinstance(assignment, PlaceAssignment):
            return {}
        names = role_names_by_axis(shape)

        def admits(key: EntityRoleKey, graph_type: str) -> bool:
            role = "observationAbout" if key == "observationAbout" else names[key]
            return state.node_admissions[(role, graph_type)]

        return set_choices(assignment, admits=admits)

    survivors: list[ShapeDraft] = []
    admitted_anywhere = {place_key(place_set): False for place_set in place_sets}
    for shape in shapes:
        roles = shape.declared_roles or ("observationAbout",)
        readings = _readings(shape)
        shape_admits_all = True
        for place_set in place_sets:
            served = readings.get(place_key(place_set), "node") is not None
            if served and await _admits_any(roles, place_set):
                admitted_anywhere[place_key(place_set)] = True
            else:
                shape_admits_all = False
        if shape_admits_all:
            survivors.append(shape)

    if survivors:
        return survivors, []

    refused = [s for s in place_sets if not admitted_anywhere[place_key(s)]] or place_sets
    return [], [not_admitted_entity(place_set) for place_set in refused]


def _set_types(place_set: PlaceSet) -> tuple[str, ...]:
    """The graph types of every reading `place_set` has: its node's, then the one its
    members share."""
    node_type = () if place_set.node is None else (place_set.node.graph_type,)
    member_type = (place_set.member_type,) if place_set.members and place_set.member_type else ()
    return tuple(dict.fromkeys((*node_type, *member_type)))


def _admission_types(place_set: PlaceSet) -> tuple[str, ...]:
    """The graph types role admission judges `place_set` by, those of the readings it can
    take: its node's, then the one its members share. A set the request asks the members
    of reads only its members, so only their type counts."""
    types = _set_types(place_set)
    if place_set.reading == "members" and place_set.members and place_set.member_type:
        return (place_set.member_type,)
    return types


def not_admitted_entity(place_set: PlaceSet) -> RoleNotAdmittedEntity:
    """The refusal's account of `place_set`: its node, or for a set with none its first
    member and the type its members share."""
    if place_set.node is not None:
        return RoleNotAdmittedEntity(
            text=place_set.name, dcid=place_set.node.dcid, graph_type=place_set.node.graph_type
        )
    return RoleNotAdmittedEntity(
        text=place_set.name,
        dcid=cast(tuple[str, ...], place_set.members)[0],
        graph_type=place_set.member_type,
    )


def _found_empty(state: ResolveState, shape: ShapeDraft) -> bool:
    """Whether `shape` holds only members of the excluded shape of its id."""
    empty = state.excluded_shapes.get(shape.shape_id)
    return empty is not None and set(shape.sv_arc_facts or {}) <= empty


async def _phase_derive_shapes(state: ResolveState) -> RegionResult | None:
    """Confirm recall candidates and derive role-compatible shapes.

    Return ``role_signature_incompatible`` or ``variable_not_resolved`` when derivation
    fails, and ``entity_role_not_admitted`` when all shapes fail role checks. Unresolved
    entity errors take precedence. Surviving shapes proceed to ranking. A shape whose id
    ``state.excluded_shapes`` names and whose members are all among the members recorded
    there is dropped as soon as it is derived: it holds nothing the first pass did not
    read and find empty.
    """
    rcl = cast(Recall, state.rcl)
    # The first pass derives from our resolve's hits under their own dcids, so the confirm
    # cap counts a superseded hit as it did while its provenance was admitted; derivation
    # reads each as its successor (`derive_shapes`).
    if state.pool == "ours":
        pool = rcl.candidates if rcl.hits is None else rcl.hits
    else:
        pool = rcl.un_candidates
    pool_svs = [c.dcid for c in pool]

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
            shapes_found = [s for s in shapes_found if not _found_empty(state, s)]
            state.shapes_found = shapes_found
    except RoleSignatureError:
        # No candidate group has a consistent entity-role signature.
        state.timing["shape"] = now_ms() - t0
        await _emit_step(
            state.emit,
            state.variable_index,
            "shape",
            ms=state.timing["shape"],
        )
        return _unresolved_entity_terminal(state) or state.no_data("role_signature_incompatible")

    # No confirmed StatVar means there is no shape to bind.
    if not shapes_found:
        state.timing["shape"] = now_ms() - t0
        await _emit_step(
            state.emit,
            state.variable_index,
            "shape",
            ms=state.timing["shape"],
        )
        return _unresolved_entity_terminal(state) or state.no_data("variable_not_resolved")

    # Prune before ranking so every later count sees only compatible shapes. Unresolved
    # entities retain precedence over role-admission failures.
    shapes_found, never_admitted = await _prune_shapes_by_admission(state, shapes_found)
    state.shapes_found = shapes_found
    if not shapes_found:
        state.timing["shape"] = now_ms() - t0
        await _emit_step(
            state.emit,
            state.variable_index,
            "shape",
            ms=state.timing["shape"],
        )
        return _unresolved_entity_terminal(state) or state.no_data(
            "entity_role_not_admitted", role_not_admitted_entities=tuple(never_admitted)
        )

    # Detect role-bearing families across the full set before election.
    state.role_bearing_recalled = any(is_role_bearing(s) for s in shapes_found)
    return None
