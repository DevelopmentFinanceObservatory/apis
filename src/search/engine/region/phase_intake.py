"""Resolve recall candidates, place sets, and shape compatibility before ranking."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final, cast

import dc_graph
from code_index.shared import get_index
from code_index.types import PartyClass
from dfo_platform.timings import PhaseTimings, collecting
from entity_resolution import Ambiguous, Entity, Group, Resolved
from entity_resolution.membership import MembershipPolicy
from origin_catalogue import note_un_unserved, recording_un_blocked_places
from search.engine.clients.graph import DetectedSV, force_full_facets
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read, recording_degraded_reads
from search.engine.envelope import _emit_step, now_ms
from search.engine.errors import EngineInfraError, RoleSignatureError
from search.engine.interpret import Recall, recall
from search.engine.place import expand_places
from search.engine.region.place_roles import (
    PlaceAssignment,
    SetChoice,
    bind_place_roles,
    held_party_words,
    part_sets,
    place_key,
    set_choices,
    stated_party_words,
    totalled_axes,
)
from search.engine.region.place_set import PlaceNode, PlaceReading, PlaceSet
from search.engine.region.result import (
    AmbiguousEntity,
    AmbiguousEntityCandidate,
    RegionResult,
    RoleNotAdmittedEntity,
)
from search.engine.region.source_choice import allows_un_system
from search.engine.region.state import ResolveState, UnLeg, UnPool
from search.engine.region.whole_totals import (
    containing_totals,
    named_whole,
    population_totals,
    side_classes,
)
from search.engine.retrieval.bound_values import declared_datasets
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft, role_names_by_axis
from search.models import EntityRoleKey, NoDataReason, RoleAxis, elect_no_data_reason
from search.render.no_data import UnlistedPopulation
from search.scope.region import oecd_region_reading

# Full-facet probing is capped at 6 candidates. This phase checks raw recall count before
# expansion. derive_shapes checks the confirmed count to bound the actual read.
_FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES = 6

# With `SEARCH_UN_ON_FIRST_MENU`, our pass merges the UN leg's shapes, waiting for them,
# only when no shape of its own has a measured recall score of at least this
# (`_holds_a_strong_shape`). A pass holding such a shape never takes the UN pool, so its
# answer does not wait on the leg.
UN_MERGE_WEAK_SCORE: Final = 0.7

logger = logging.getLogger(__name__)

# Failures of a UN shape derivation that mean an upstream read failed or hit a capacity
# limit. The part has no UN pass. Any other exception is a defect and raises.
_UPSTREAM_FAILURES: Final = (
    EngineInfraError,
    TimeoutError,
    dc_graph.GraphReadError,
    dc_graph.GraphCapExceededError,
)

_UN_LATE: Final = (
    "The search over the UN System Data Commons' variables and their reads did not finish "
    "in time; candidates come from our Data Commons only."
)
_UN_RESOLVE_FAILED: Final = (
    "The UN System Data Commons did not answer the variable search; "
    "candidates come from our Data Commons only."
)
_UN_READS_FAILED: Final = (
    "The UN System Data Commons' variables could not be read; "
    "candidates come from our Data Commons only."
)


async def _phase_recall(state: ResolveState) -> None:
    """Run recall and record its candidates for place-set resolution. Recall searches the
    part's text without each word that names a party on an axis the request leaves unnamed
    (`place_roles.stated_party_words`)."""
    t0 = now_ms()
    rcl: Recall = await recall(
        state.variable,
        state.entities,
        graph=state.graph,
        resolver=state.resolver,
        raw_query=state.detect_query,
        pre_resolved=state.pre_resolved,
        entity_surfaces=state.entity_surfaces,
        members=None if state.source_code is None else state.source_code.dcids,
        party_words=frozenset(named.word for named in stated_party_words(state)),
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
    unresolved and is refused later. A continent or subregion by the membership policy
    (`MembershipPolicy.holds_region`), whatever the node's graph type, is a set of its node and
    the countries the policy holds for it (`place.expand_places`); one the policy omits is a set
    of its node alone. A group or region a flow
    dataset publishes a total for carries that total (`PlaceSet.total`), found by the group's
    id or the region's node. Every
    other resolved name is a set of its node alone. The node of a group or region carries the
    graph's label for it, read in one batch. A set takes the members reading when the door
    asks for the name's members (`ResolveState.member_entities`) or the resolver read the name
    as its members (`reading`). Recall's country dcids are snapshotted for the availability
    floor. The party each word of the part names on an unnamed axis gets a set of its node
    (`ResolveState.party_sets`, `_party_word_sets`). A text naming a whole population ("all
    donors") gets no place set: it and each whole total the part can bind on a side it
    totals get a set of their node (`ResolveState.whole_sets`, `_whole_total_sets`). Such a
    text asked for its members, or under an operation over its side ("official donors
    giving more than 100 million"), is listed (`ResolveState.whole_listed`): it opens its side
    (`ResolveState.open_axes`) in place of naming a total there, on the shapes that list
    exactly its parties there (`_prune_shapes_by_listing`). Whether the request directs such
    a text to the other side is recorded too (`ResolveState.whole_misdirected`).
    """
    # The type checker narrows locals, while state attributes need this explicit cast.
    rcl = cast(Recall, state.rcl)

    # Truncating an oversized group's members would misstate the population the answer
    # covers. `_phase_check_entities` refuses an oversized `Group` with `entity_set_too_large`.
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
    # A text naming a whole population ("all donors") binds as a whole total
    # (`_whole_total_sets`), so it holds no place set. Asked for its members, or under an
    # operation over its side, which may filter, rank or count them, it asks for every
    # party of its side, which it leaves open and no total binds.
    whole_named = _whole_named(state, resolved)
    groups = {name: group for name, group in groups.items() if name not in whole_named}
    resolved = {name: entity for name, entity in resolved.items() if name not in whole_named}
    parties = get_index().parties
    over = {operation.over for operation in state.operations}
    sides = {name: parties[dcids[0]].axes[0] for name, dcids in whole_named.items()}
    # A population's side is the one axis the code index publishes its parties on, so a text
    # the request directs to the other side ("from France to official donors") is refused
    # `role_incompatible` however it is asked (`_phase_check_entities`).
    state.whole_misdirected = tuple(
        name
        for name, axis in sides.items()
        if any(role.stated and role.axis != axis for role in state.entity_roles.get(name) or ())
    )
    state.whole_listed = {
        name: axis for name, axis in sides.items() if name in state.member_entities or axis in over
    }
    state.open_axes |= set(state.whole_listed.values())
    whole_named = {
        name: dcids for name, dcids in whole_named.items() if name not in state.whole_listed
    }
    membership = state.resolver.membership
    regions = [entity for entity in resolved.values() if membership.holds_region(entity.dcid)]

    # Preserve recall's country set for derive_shapes' place-availability probe. Groups and
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
    labels = await asyncio.to_thread(state.graph.node_labels_batch, labelled) if labelled else {}
    expansions = (
        expand_places((entity.dcid for entity in regions), membership=membership) if regions else {}
    )

    def reading(name: str) -> PlaceReading:
        resolution = cast(Resolved | Group, rcl.resolutions[name])
        asked = name in state.member_entities or resolution.reading == "members"
        return "members" if asked else "default"

    totals = get_index().totals_by_group

    def total(group: str | None) -> str | None:
        row = None if group is None else totals.get(group)
        return None if row is None else row.dcid

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
            total=total(group.dcid or (None if node is None else node.dcid)),
            group_id=group.dcid,
        )
    for name, entity in resolved.items():
        region = membership.holds_region(entity.dcid)
        expansion = expansions.get(entity.dcid) if region else None
        place_set = PlaceSet(
            name=name,
            kind="region" if region else "place",
            node=PlaceNode(dcid=entity.dcid, graph_type=entity.graph_type),
            label=labels.get(entity.dcid) if region else None,
            members=None if expansion is None else tuple(expansion.children),
            member_type=None if expansion is None else expansion.child_type,
            reading=reading(name),
            total=total(entity.dcid) if expansion is not None else None,
        )
        place_sets[name] = replace(place_set, oecd=oecd_reading(place_set, membership=membership))
    state.place_sets = place_sets
    state.party_sets = await _party_word_sets(state)
    state.whole_named = whole_named
    state.whole_sets = _whole_total_sets(state)


def oecd_reading(place_set: PlaceSet, *, membership: MembershipPolicy) -> PlaceSet | None:
    """The OECD's reading of `place_set` (`PlaceSet.oecd`), None for a set no row of
    `scope_conventions.REGION_READINGS` names or whose DAC group the membership policy holds
    no members for: the set with the members of the group the row's DAC total sums
    (`code_index.declarations.TOTAL_GROUPS`), that total and the group's registry id."""
    if place_set.node is None or place_set.members is None:
        return None
    row = oecd_region_reading(place_set.node.dcid)
    if row is None:
        return None
    total = get_index().parties.get(row.total)
    if total is None or total.totals is None or total.totals not in membership:
        return None
    held = membership.members(total.totals).members
    if not isinstance(held, tuple) or not held:
        return None
    return replace(
        place_set,
        members=held,
        member_type="Country",
        total=total.dcid,
        group_id=total.totals,
        oecd_read=True,
    )


def _whole_named(state: ResolveState, resolved: Mapping[str, Entity]) -> dict[str, tuple[str, ...]]:
    """Each text of the part naming a whole population, with the dcids it names
    (`whole_totals.named_whole`): the phrase of a whole total set, or a name resolving to a
    whole total of every party on its side."""
    named: dict[str, tuple[str, ...]] = {}
    for name in dict.fromkeys(state.entities):
        entity = resolved.get(name)
        if (dcids := named_whole(name, None if entity is None else entity.dcid)) is not None:
            named[name] = dcids
    return named


def _whole_total_sets(state: ResolveState) -> dict[str, PlaceSet]:
    """The place set of each whole total the part can bind (`ResolveState.whole_sets`), by
    dcid: each party a text names as a whole population (`ResolveState.whole_named`), and
    the totals of every party on each side the part totals (`place_roles.totalled_axes`,
    `whole_totals.population_totals`), and the containing totals of those sides
    (`whole_totals.containing_totals`). Each is its node, named by the code index's label
    for the party.

    A total of a side the part totals binds where the code index marks it on a side its
    dataset publishes (`place_roles.whole_total_candidates`), and a named one binds as a
    party on its side (`place_roles.bind_place_roles`). Its set asks role admission nothing
    (`_prune_shapes_by_admission`): it holds no members, so it reads its node on any role
    (`place_roles.set_choices`), and its node carries no graph type, so no graph read runs
    for it."""
    named = [dcid for dcids in state.whole_named.values() for dcid in dcids]
    axes = totalled_axes(state)
    rows = get_index().parties

    def label(dcid: str) -> str:
        row = rows.get(dcid)
        return row.label if row is not None and row.label else dcid

    return {
        dcid: PlaceSet(name=label(dcid), kind="place", node=PlaceNode(dcid=dcid, graph_type=""))
        for dcid in dict.fromkeys((*named, *population_totals(axes), *containing_totals(axes)))
    }


async def _party_word_sets(state: ResolveState) -> dict[str, PlaceSet]:
    """The place set of the party of each word the part states on an axis the request leaves
    unnamed (`place_roles.stated_party_words`), by the party's dcid: its node, typed by one
    batched graph read, and named by the code index's label for the party
    (`code_index.types.PartyRow.label`). No read runs when the part states no such word.

    The index declares each party, so one the graph gives no type has left the graph since
    the index was built. It gets no set, the word then names no party on any shape, and the
    read is logged and noted as degraded (`degraded_reads.note_degraded_read`)."""
    parties = sorted({named.party for named in stated_party_words(state)})
    if not parties:
        return {}
    types = await asyncio.to_thread(state.graph.node_types_batch, parties)
    untyped = [dcid for dcid in parties if dcid not in types]
    if untyped:
        logger.warning("declared parties with no graph type: %s", ", ".join(untyped))
        note_degraded_read("party word types")
    rows = get_index().parties

    def label(dcid: str) -> str:
        row = rows.get(dcid)
        return row.label if row is not None and row.label else dcid

    return {
        dcid: PlaceSet(
            name=label(dcid), kind="place", node=PlaceNode(dcid=dcid, graph_type=types[dcid])
        )
        for dcid in parties
        if dcid in types
    }


def _sole_axis(state: ResolveState, name: str) -> RoleAxis | None:
    """The one flow axis the door gave `name`, or `None` when it gave none or both."""
    axes = {role.axis for role in state.entity_roles.get(name, ())}
    return axes.pop() if len(axes) == 1 else None


def _is_oversized_group(resolution: object) -> bool:
    """Whether `resolution` is a Group with more members than a Spec binds
    (``max_child_places``)."""
    return isinstance(resolution, Group) and len(resolution.members) > settings().max_child_places


def _ambiguous_candidate(
    candidate: Entity, *, group_nodes: frozenset[str], group_names: Mapping[str, str]
) -> AmbiguousEntityCandidate:
    """`candidate` as an ambiguous wording's candidate, marked a group node when it is one
    of `group_nodes`, the group nodes of a wording asked for its members, with its group's
    name from `group_names`."""
    group_node = candidate.dcid in group_nodes
    return AmbiguousEntityCandidate(
        dcid=candidate.dcid,
        name=candidate.name,
        kind=candidate.kind,
        group_node=group_node,
        group_name=group_names.get(candidate.dcid) if group_node else None,
    )


async def _phase_check_entities(state: ResolveState) -> RegionResult | None:
    """Refuse a part naming an entity that did not resolve, before shape derivation and
    the bind call.

    Each name with no place set that names no whole population (`ResolveState.whole_named`,
    `ResolveState.whole_listed`) mints one reason: ``entity_ambiguous`` for an Ambiguous
    resolution, ``entity_set_too_large`` for a Group with more members than a Spec binds,
    and ``entity_not_resolved`` otherwise. The part refuses with the reason
    `elect_no_data_reason` elects over them, and names, whatever the elected reason, an
    ambiguous name with its candidates (``ambiguous_entities``) and every name that resolved
    to no single node, ambiguous ones included, as unresolved (``unresolved_entity_names``).
    An oversized group resolved, so it is never among the unresolved. Both keep query
    order. The place sets are final once `_phase_resolve_places` has run,
    so every later phase sees only resolved entities.

    A part whose names all resolved but which directs a whole population to the side other
    than its own (`ResolveState.whole_misdirected`) refuses ``role_incompatible`` here,
    naming those texts as unresolved, before any shape is derived or pruned: the conflict
    is the request's own and holds on every shape, so no listing or variable refusal
    stands in front of it.
    """
    rcl = cast(Recall, state.rcl)
    # One reason per name, in query order.
    reasons: dict[str, NoDataReason] = {
        name: _entity_reason(rcl.resolutions.get(name))
        for name in state.entities
        if name not in state.place_sets
        and name not in state.whole_named
        and name not in state.whole_listed
    }
    if not reasons:
        if state.whole_misdirected:
            return state.no_data(
                "role_incompatible", unresolved_entity_names=state.whole_misdirected
            )
        return None
    ambiguous = [name for name, name_reason in reasons.items() if name_reason == "entity_ambiguous"]
    # A pin on a group's own node cannot ask for its members, so a candidate of a wording
    # asked for its members carries whether it is one, and the group's registered name for
    # its option to post.
    asks_members = any(name in state.member_entities for name in ambiguous)
    group_nodes = await state.resolver.group_node_dcids() if asks_members else frozenset()
    group_names = await state.resolver.group_node_names() if asks_members else {}
    ambiguous_entities = tuple(
        AmbiguousEntity(
            text=name,
            candidates=tuple(
                _ambiguous_candidate(
                    candidate,
                    group_nodes=group_nodes if name in state.member_entities else frozenset(),
                    group_names=group_names,
                )
                for candidate in cast(Ambiguous, rcl.resolutions[name]).candidates
            ),
            role=_sole_axis(state, name),
        )
        for name in ambiguous
    )
    # Each option of an `entity_ambiguous` refusal pins these. A group, and a set read as
    # its members, stay unpinned: re-resolving the text keeps the members, whether an alias
    # row or the part's flag asked for them.
    resolved_entity_dcids = {
        name: place_set.node.dcid
        for name in dict.fromkeys(state.entities)
        if (place_set := state.place_sets.get(name)) is not None
        and place_set.kind != "group"
        and place_set.reading != "members"
        and place_set.node is not None
    }
    reason = elect_no_data_reason(reasons.values())
    return state.no_data(
        reason,
        ambiguous_entities=ambiguous_entities,
        resolved_entity_dcids=resolved_entity_dcids if reason == "entity_ambiguous" else None,
        unresolved_entity_names=tuple(
            name for name, name_reason in reasons.items() if name_reason != "entity_set_too_large"
        ),
    )


def _entity_reason(resolution: object) -> NoDataReason:
    """The reason one entity name with no place set mints, from its resolution."""
    if isinstance(resolution, Ambiguous):
        return "entity_ambiguous"
    if _is_oversized_group(resolution):
        return "entity_set_too_large"
    return "entity_not_resolved"


async def _prune_shapes_by_admission(
    state: ResolveState, shapes: list[ShapeDraft]
) -> tuple[list[ShapeDraft], list[RoleNotAdmittedEntity]]:
    """Keep shapes that can serve every place set the part names, each once however many
    texts name it (`place_roles.part_sets`), and the set of each party a word of the part
    names on the shape (`place_roles.held_party_words`).

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
    holds members (`place_roles.set_choices`) and the type of each party a word of the part
    can name (`ResolveState.party_sets`). The answers are recorded by role and type
    (`state.node_admissions`), so settling reads them in hand and a member type's class
    walk runs beside a node type's. A set with no member needs no question of settling, and
    is judged as before, a role and type at a time until one admits it.

    Return rejected-entity details only when all shapes fail, in query order. The fallback
    names all entities when each is admitted somewhere but no single shape admits all.
    """
    place_sets = list(part_sets(state).values())
    parties = list(state.party_sets.values())
    if not place_sets and not parties:
        return shapes, []

    unasked = sorted(
        {
            (role, graph_type)
            for shape in shapes
            for role in shape.declared_roles or ("observationAbout",)
            for place_set in (*(named for named in place_sets if named.members), *parties)
            for graph_type in _set_types(place_set)
        }
        - state.node_admissions.keys()
    )
    answers = await asyncio.gather(
        *[state.admission.admits(role, graph_type) for role, graph_type in unasked]
    )
    state.node_admissions.update(zip(unasked, answers, strict=True))
    # Read once the answers are in hand: a word names its party on a shape by the kinds
    # the named places read as there (`place_roles.held_party_words`), which the roles'
    # answers decide.
    sets_of = {
        id(shape): [
            *place_sets,
            *(state.party_sets[named.party] for named in held_party_words(state, shape)),
        ]
        for shape in shapes
    }

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
    # Every set some shape must serve, by place key, and whether one does.
    judged: dict[str, PlaceSet] = {}
    admitted_anywhere: dict[str, bool] = {}
    for shape in shapes:
        roles = shape.declared_roles or ("observationAbout",)
        readings = _readings(shape)
        shape_admits_all = True
        for place_set in sets_of[id(shape)]:
            key = place_key(place_set)
            judged[key] = place_set
            served = readings.get(key, "node") is not None
            if served and await _admits_any(roles, place_set):
                admitted_anywhere[key] = True
            else:
                admitted_anywhere.setdefault(key, False)
                shape_admits_all = False
        if shape_admits_all:
            survivors.append(shape)

    if survivors:
        return survivors, []

    refused = [s for key, s in judged.items() if not admitted_anywhere[key]] or place_sets
    return [], [not_admitted_entity(place_set) for place_set in refused]


def _prune_shapes_by_listing(
    state: ResolveState, shapes: list[ShapeDraft]
) -> tuple[list[ShapeDraft], list[RoleNotAdmittedEntity]]:
    """Keep shapes on which each whole population a text asks the members of
    (`ResolveState.whole_listed`) opens its side to exactly its parties: the classes of party
    the shape's declared datasets hold there, totals aside (`whole_totals.side_classes`), are
    a subset of the ones the population's markers cover (`WholeMarker.covers`). An open side
    lists every party its dataset holds, and a Spec states no class it leaves out, so
    "official donors" asked member by member lists on no CRS shape, where private donors
    stand beside official ones. A shape whose datasets hold no class the index reads on the
    side, such as one no declared dataset publishes, lists no population, since nothing
    shows what its open side would hold.

    Return the rejected populations, each named by its first party (`UnlistedPopulation`),
    only when every shape fails. Where some population fails on every shape, those are the
    ones named, each with the classes every shape's datasets hold beside it, or with no
    account (`RoleNotAdmittedEntity.unlisted` `None`) where the shapes share no class they
    are known to hold beside it, as where one shape's classes cannot be read. Otherwise each
    population fails only beside the others, and every one some shape fails is named as
    listing alone. A population every shape lists exactly is no cause of the refusal."""
    if not state.whole_listed:
        return shapes, []
    resolved = {
        name: resolution.entity
        for name, resolution in cast(Recall, state.rcl).resolutions.items()
        if name in state.whole_listed and isinstance(resolution, Resolved)
    }
    listed = {
        name: dcids
        for name, dcids in _whole_named(state, resolved).items()
        if name in state.whole_listed
    }
    parties = get_index().parties
    covered = {
        name: {cls for dcid in dcids for marker in parties[dcid].whole for cls in marker.covers}
        for name, dcids in listed.items()
    }

    def left_out(shape: ShapeDraft, name: str) -> set[PartyClass] | None:
        datasets = [dataset.provenance for dataset in declared_datasets(shape, get_index())]
        classes = side_classes(state.whole_listed[name], datasets=datasets)
        return set(classes) - covered[name] if classes else None

    def fails(classes: set[PartyClass] | None) -> bool:
        return classes is None or bool(classes)

    left_outs = {name: [left_out(shape, name) for shape in shapes] for name in listed}
    survivors = [
        shape
        for position, shape in enumerate(shapes)
        if not any(fails(classes[position]) for classes in left_outs.values())
    ]
    if survivors:
        return survivors, []
    nowhere = [name for name, classes in left_outs.items() if all(map(fails, classes))]
    unlisted: dict[str, UnlistedPopulation | None]
    if nowhere:
        held = {name: set.intersection(*(c or set() for c in left_outs[name])) for name in nowhere}
        unlisted = {
            name: UnlistedPopulation(axis=state.whole_listed[name], left_out=tuple(sorted(classes)))
            if classes
            else None
            for name, classes in held.items()
        }
    else:
        unlisted = {
            name: UnlistedPopulation(axis=state.whole_listed[name], left_out=(), listed_alone=True)
            for name, classes in left_outs.items()
            if any(map(fails, classes))
        }
    return [], [
        RoleNotAdmittedEntity(text=name, dcid=listed[name][0], graph_type=None, unlisted=population)
        for name, population in unlisted.items()
    ]


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


async def _derive_pool(
    state: ResolveState, pool: Sequence[DetectedSV], *, timings: PhaseTimings
) -> list[ShapeDraft]:
    """The shapes `pool`'s candidates derive, off the event loop, with derivation's stages
    (`derive.derive_shapes`) recorded on `timings`. Raises `RoleSignatureError` when no
    candidate group has a consistent entity-role signature."""
    pool_svs = [c.dcid for c in pool]

    def derive() -> list[ShapeDraft]:
        with collecting(timings):
            return derive_shapes(
                confirmed_svs=pool_svs,
                graph=state.graph,
                sv_scores={c.dcid: c.score for c in pool},
                sv_types={c.dcid: c.type_of for c in pool},
                rival_svs=frozenset(c.dcid for c in pool if c.rival),
                basis_sibling_svs=frozenset(c.dcid for c in pool if c.basis_sibling),
                # Use recall's country snapshot. Place sets and Group members stay off the
                # floor.
                place_dcids=state.recall_only_place_dcids,
                full_facets_max_candidates=_FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES,
                deadline=state.deadline,
                coverage=state.coverage.index,
            )

    # Dated observation reads use ``needs_dates=True``. This scope lets the availability
    # probe share that cache key when raw recall has at most 6 candidates. derive_shapes
    # rechecks the limit after confirmation and Topic expansion, bounding the full-history
    # read to 6 confirmed StatVars. The context propagates through ``to_thread``.
    with force_full_facets(
        state.date_request is not None and len(pool_svs) <= _FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES
    ):
        return await asyncio.to_thread(derive)


async def _un_shapes(state: ResolveState, rcl: Recall) -> UnPool:
    """The shapes the UN System Data Commons' candidates derive, each stamped with its
    pool, and why the part has no UN pool to fill a gap with, `None` when it need not. A UN
    pool whose derivation finds no consistent role signature adds no shape and needs no
    reason. A failed resolve or upstream read adds none and gives its reason. The
    derivation's reads note what they could not serve under the leg's own records, which
    the pool carries (`UnPool`)."""
    await rcl.settle_un()
    if rcl.un_unavailable:
        return UnPool(shapes=[], missed=_UN_RESOLVE_FAILED)
    candidates = rcl.un_candidates
    if not candidates:
        return UnPool(shapes=[])
    started = time.monotonic()
    with (
        recording_un_blocked_places() as un_blocked,
        recording_degraded_reads(detached=True) as lost,
    ):
        try:
            # The leg runs beside our pass, so its stages stay out of the part's timing.
            shapes = await _derive_pool(state, candidates, timings=PhaseTimings())
        except RoleSignatureError:
            shapes, missed = [], None
        except _UPSTREAM_FAILURES as exc:
            logger.warning("UN System Data Commons shape derivation failed: %s", exc)
            shapes, missed = [], _UN_READS_FAILED
        else:
            shapes, missed = [replace(shape, pool="un_system") for shape in shapes], None
        finally:
            logger.info(
                "un_derive ms=%d candidates=%d",
                round((time.monotonic() - started) * 1000),
                len(candidates),
            )
    return UnPool(
        shapes=shapes, missed=missed, unserved=un_blocked.snapshot(), lost=tuple(lost.reads)
    )


async def un_pool_shapes(leg: UnLeg, *, bound: float | None) -> UnPool:
    """The UN leg's pool (`_un_shapes`), waiting for the leg until its own end
    (`UnLeg.ends`) or `bound`, whichever comes first. A leg still running then is cancelled
    and the pool's reason says it was late; its derivation thread runs on until its reads
    end."""
    ends = leg.ends if bound is None else min(leg.ends, bound)
    done, _ = await asyncio.wait({leg.task}, timeout=max(0.0, ends - time.monotonic()))
    if not done:
        leg.task.cancel()
        return UnPool(shapes=[], missed=_UN_LATE)
    return leg.task.result()


def _holds_a_strong_shape(shapes: Sequence[ShapeDraft]) -> bool:
    """Whether a shape of `shapes` has a measured recall score of at least
    `UN_MERGE_WEAK_SCORE`."""
    return any(
        score is not None and score >= UN_MERGE_WEAK_SCORE
        for score in (shape.measured_score for shape in shapes)
    )


async def _merge_un_pool(state: ResolveState, leg: UnLeg) -> None:
    """Append the UN leg's shapes to our pass's (`ResolveState.shapes_found`) when our pass
    holds no strong shape (`_holds_a_strong_shape`), so both pools meet the same admission,
    each its own score floor, and one menu.

    A part holding a strong shape never takes the pool, whenever the leg finishes, so its
    answer does not depend on the UN leg's latency; the leg stays for `pipeline._fill_gap`.
    Any other part waits for the leg (`un_pool_shapes`), and a late or failed leg leaves
    the pool's reason in `ResolveState.un_pool` for its answer to carry. The reads a merged
    pool could not serve join the part's records, since the pool reaches its menu."""
    if _holds_a_strong_shape(state.shapes_found):
        return
    state.un_pool = pool = await un_pool_shapes(leg, bound=state.deadline)
    if pool.missed is not None:
        return
    state.shapes_found = [*state.shapes_found, *pool.shapes]
    note_un_unserved(pool.unserved)
    for read in pool.lost:
        note_degraded_read(read)


async def _phase_derive_shapes(state: ResolveState) -> RegionResult | None:
    """Confirm the pass's recall candidates and derive role-compatible shapes.

    Our pass derives from our candidates, and starts the UN System Data Commons' resolve
    and shape derivation as one task beside it (`ResolveState.un_leg`) when the part's
    scope allows a UN provenance (`source_choice.allows_un_system`) and the part names no
    series code, since a code part's answer stands without a UN pass
    (`pipeline._un_pass_skip`). With `SEARCH_UN_ON_FIRST_MENU`, our pass holding no strong
    shape appends that task's shapes to its own (`_merge_un_pool`). Otherwise our pass never waits for the
    task; the UN pass, which runs when ours ends no_data (`pipeline._fill_gap`), takes its
    shapes, already in `state.shapes_found`. Return
    ``role_signature_incompatible`` or ``variable_not_resolved`` when the pass derives no
    shape, and ``entity_role_not_admitted`` when all shapes fail role checks or none lists
    a whole population asked member by member exactly (`_prune_shapes_by_listing`). A part whose
    series code named its candidates (`ResolveState.source_code`) and derives no shape from
    them refuses ``source_code_not_held`` in place of ``variable_not_resolved``
    (`ResolveState.no_data`). Surviving
    shapes proceed to ranking.

    Beside `shape`, `state.timing` gets our pass's derivation stages (`derive.derive_shapes`:
    `shape_arcs`, `shape_probe` with `shape_probe_coverage` inside it, `shape_labels`,
    `shape_groups`) and the admission prune,
    `shape_admission`. The UN leg's derivation runs concurrently and records none of them.
    """
    rcl = cast(Recall, state.rcl)
    t0 = state.shape_t0 = now_ms()
    signature_refused = False
    if state.pool == "ours":
        if state.source_code is None and allows_un_system(state.scope):
            state.un_leg = UnLeg(
                task=asyncio.create_task(_un_shapes(state, rcl)),
                ends=time.monotonic() + settings().un_first_pass_wait_s,
            )
        # Our resolve's hits under their own dcids, so the confirm cap counts a superseded
        # hit as it did while its provenance was admitted; derivation reads each as its
        # successor.
        ours = rcl.candidates if rcl.hits is None else rcl.hits
        derive_timings = PhaseTimings()
        try:
            state.shapes_found = await _derive_pool(state, ours, timings=derive_timings)
        except RoleSignatureError:
            state.shapes_found, signature_refused = [], True
        finally:
            state.timing.update({name: round(ms) for name, ms in derive_timings.snapshot().items()})
        if state.un_leg is not None and settings().un_on_first_menu:
            await _merge_un_pool(state, state.un_leg)
    shapes_found = state.shapes_found

    # No confirmed StatVar means there is no shape to bind.
    if not shapes_found:
        state.timing["shape"] = now_ms() - t0
        await _emit_step(
            state.emit,
            state.variable_index,
            "shape",
            ms=state.timing["shape"],
        )
        if signature_refused:
            # No candidate group has a consistent entity-role signature.
            return state.no_data("role_signature_incompatible")
        if state.source_code is not None:
            logger.info(
                "source code %s: no shape derives from its StatVars %s",
                state.source_code.code,
                state.source_code.dcids,
            )
        return state.no_data("variable_not_resolved")

    # Prune before ranking so every later count sees only compatible shapes.
    admission_t0 = now_ms()
    shapes_found, never_admitted = await _prune_shapes_by_admission(state, shapes_found)
    if shapes_found:
        shapes_found, never_admitted = _prune_shapes_by_listing(state, shapes_found)
    state.timing["shape_admission"] = now_ms() - admission_t0
    state.shapes_found = shapes_found
    if not shapes_found:
        state.timing["shape"] = now_ms() - t0
        await _emit_step(
            state.emit,
            state.variable_index,
            "shape",
            ms=state.timing["shape"],
        )
        return state.no_data(
            "entity_role_not_admitted", role_not_admitted_entities=tuple(never_admitted)
        )

    return None
