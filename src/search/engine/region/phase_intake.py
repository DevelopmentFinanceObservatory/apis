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
from entity_resolution import Ambiguous, Group, Resolved
from origin_catalogue import recording_un_blocked_places
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
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft, role_names_by_axis
from search.models import EntityRoleKey, NoDataReason, RoleAxis, elect_no_data_reason

# Full-facet probing is capped at 6 candidates. This phase checks raw recall count before
# expansion. derive_shapes checks the confirmed count to bound the actual read.
_FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES = 6

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
    (`ResolveState.party_sets`, `_party_word_sets`).
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
        place_sets[name] = PlaceSet(
            name=name,
            kind="region" if region else "place",
            node=PlaceNode(dcid=entity.dcid, graph_type=entity.graph_type),
            label=labels.get(entity.dcid) if region else None,
            members=None if expansion is None else tuple(expansion.children),
            member_type=None if expansion is None else expansion.child_type,
            reading=reading(name),
            total=total(entity.dcid) if expansion is not None else None,
        )
    state.place_sets = place_sets
    state.party_sets = await _party_word_sets(state)


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


async def _phase_check_entities(state: ResolveState) -> RegionResult | None:
    """Refuse a part naming an entity that did not resolve, before shape derivation and
    the bind call.

    Each name with no place set mints one reason: ``entity_ambiguous`` for an Ambiguous
    resolution, ``entity_set_too_large`` for a Group with more members than a Spec binds,
    and ``entity_not_resolved`` otherwise. The part refuses with the reason
    `elect_no_data_reason` elects over them, and names, whatever the elected reason, an
    ambiguous name with its candidates (``ambiguous_entities``) and every name that resolved
    to no single node, ambiguous ones included, as unresolved (``unresolved_entity_names``).
    An oversized group resolved, so it is never among the unresolved. Both keep query
    order. The place sets are final once `_phase_resolve_places` has run,
    so every later phase sees only resolved entities.
    """
    rcl = cast(Recall, state.rcl)
    # One reason per name, in query order.
    reasons: dict[str, NoDataReason] = {
        name: _entity_reason(rcl.resolutions.get(name))
        for name in state.entities
        if name not in state.place_sets
    }
    if not reasons:
        return None
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
        for name, name_reason in reasons.items()
        if name_reason == "entity_ambiguous"
    )
    # The part's `data_query` pins these on an `entity_ambiguous` refusal. A group, and a
    # set read as its members, stay unpinned: re-resolving the text keeps the members,
    # whether an alias row or the part's flag asked for them.
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


async def _derive_pool(state: ResolveState, pool: Sequence[DetectedSV]) -> list[ShapeDraft]:
    """The shapes `pool`'s candidates derive, off the event loop. Raises
    `RoleSignatureError` when no candidate group has a consistent entity-role signature."""
    pool_svs = [c.dcid for c in pool]
    # Dated observation reads use ``needs_dates=True``. This scope lets the availability
    # probe share that cache key when raw recall has at most 6 candidates. derive_shapes
    # rechecks the limit after confirmation and Topic expansion, bounding the full-history
    # read to 6 confirmed StatVars. The context propagates through ``to_thread``.
    with force_full_facets(
        state.date_request is not None and len(pool_svs) <= _FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES
    ):
        return await asyncio.to_thread(
            derive_shapes,
            confirmed_svs=pool_svs,
            graph=state.graph,
            sv_scores={c.dcid: c.score for c in pool},
            sv_types={c.dcid: c.type_of for c in pool},
            rival_svs=frozenset(c.dcid for c in pool if c.rival),
            # Use recall's country snapshot. Place sets and Group members stay off the floor.
            place_dcids=state.recall_only_place_dcids,
            full_facets_max_candidates=_FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES,
            deadline=state.deadline,
        )


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
            shapes = await _derive_pool(state, candidates)
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


async def _phase_derive_shapes(state: ResolveState) -> RegionResult | None:
    """Confirm the pass's recall candidates and derive role-compatible shapes.

    Our pass derives from our candidates, and starts the UN System Data Commons' resolve
    and shape derivation as one task beside it (`ResolveState.un_leg`) when the part's
    scope allows a UN provenance (`source_choice.allows_un_system`) and the part names no
    series code, since a code part's answer stands without a UN pass
    (`pipeline._un_pass_skip`). Our pass never waits
    for that task; the UN pass, which runs only when ours ends no_data
    (`pipeline._fill_gap`), takes its shapes, already in `state.shapes_found`. Return
    ``role_signature_incompatible`` or ``variable_not_resolved`` when the pass derives no
    shape, and ``entity_role_not_admitted`` when all shapes fail role checks. A part whose
    series code named its candidates (`ResolveState.source_code`) and derives no shape from
    them refuses ``source_code_not_held`` in place of ``variable_not_resolved``
    (`ResolveState.no_data`). Surviving
    shapes proceed to ranking.
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
        try:
            state.shapes_found = await _derive_pool(state, ours)
        except RoleSignatureError:
            state.shapes_found, signature_refused = [], True
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
        return state.no_data(
            "entity_role_not_admitted", role_not_admitted_entities=tuple(never_admitted)
        )

    return None
