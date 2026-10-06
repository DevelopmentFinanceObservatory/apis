"""Bind the query's capped places to one shape's roles.

The elected shape and every alternate bind through `bind_place_roles`, so an alternate's
entities, where slot and role keys follow its own role declaration. The elected shape turns
a refusal into its no-data outcome; an alternate that refuses is dropped from the candidates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from code_index.members import family_declares_roles
from code_index.shared import get_index
from materialise.roles import axis_for_role
from search.engine.bind import SlotBindingDraft
from search.engine.config import settings
from search.engine.errors import RoleEntityUnplaced
from search.engine.interpret import Recall
from search.engine.retrieval.bound_values import shape_family, shape_part_datasets
from search.engine.role_text import DoorRole, contradicts
from search.engine.shape import (
    ShapeDraft,
    entities_by_role,
    is_role_bearing,
    required_role_names,
    role_names_by_axis,
)
from search.engine.spec_build import NamedGroup
from search.models import EntityRoleKey, RoleAxis

if TYPE_CHECKING:
    # `ResolveState` holds a `PlaceContext`, which holds `PlaceRoles`.
    from search.engine.region.state import ResolveState

logger = logging.getLogger(__name__)

PlaceRoleRefusal = Literal["role_incompatible", "role_not_resolved"]


@dataclass(frozen=True)
class PlaceRoles:
    """One shape's binding of the query's capped places."""

    # Places on each flow axis, in `place_order`. A subject-only shape leaves both empty.
    destination_dcids: list[str]
    origin_dcids: list[str]
    # Entity dcids by the shape's declared role names, as `materialise` reads them.
    resolved_entities: dict[str, list[str]]
    # Resolved dcid-role pairs on the wire's role keys; a self-flow lists one dcid twice.
    roles: list[tuple[str, EntityRoleKey]]
    # The places the where slot binds: destinations on a role-bearing shape, every place
    # on a subject-only one.
    where_dcids: list[str]
    # The single place an observation probe falls back to.
    obs_destination_dcid: str | None
    # Directed query names a subject-only shape reads as its subject, with the dropped axis.
    discarded_directions: tuple[tuple[str, RoleAxis], ...] = ()
    # The names in `discarded_directions` whose direction the request's words did not state
    # (`DoorRole.stated`), so the extractor inferred it.
    inferred_directions: frozenset[str] = frozenset()
    # The flow axes the request asks every member of that no place binds, until the
    # members are elected, and then those of them the elected members declare
    # (`PlaceContext.settled`). The role keys read them (`bound_values.entity_role_keys`),
    # and a family whose members declare no role on one elects as if it were unnamed. A
    # subject-only shape leaves it empty.
    open_axes: frozenset[RoleAxis] = frozenset()


def _required_role_axis(shape: ShapeDraft) -> RoleAxis | None:
    """Return the flow axis of the shape's one required role, else None.

    None when the shape requires other than one role or that role has no flow axis.
    """
    required = required_role_names(shape)
    return axis_for_role(required[0]) if len(required) == 1 else None


def _undirected_place_axis(
    state: ResolveState, shape: ShapeDraft, dcids: Iterable[str], *, group: bool
) -> RoleAxis | None:
    """Return the flow axis the place `dcids`, named without a direction, binds under on
    `shape`. `group` says the place is a named group of places.

    It is the axis of the shape's one required role. On a shape with a role on each axis,
    an operation running across one of them leaves that axis open, and a single place takes
    the other axis: "top creditors of external debt of Kenya" reads Kenya as the debtor.
    This holds for the elected shape and every alternate alike. A group can read either way
    ("top recipients of aid in Africa" scopes the ranked recipients, "top creditors of
    external debt of Africa" names the debtors), so it skips this rule. Otherwise a shape
    that requires two roles takes
    the axis the elected shape (`state.shape_draft`) bound the place under, when it declares
    a role on that axis: an alternate reads the variable another way, not the place, so a
    debt total's recipient stays the recipient of a by-creditor alternate. That axis is
    read from the elected shape's binding as admitted (`state.elected_place_roles`), taken
    before the role-class election narrowed it, since the narrowed shape can require two
    roles.
    """
    axis = _required_role_axis(shape)
    if axis is not None:
        return axis
    open_axes = {
        operation.over
        for operation in state.operations
        if operation.over == "origin" or operation.over == "destination"
    }
    names = role_names_by_axis(shape)
    if not group and len(open_axes) == 1 and len(names) == 2:
        (open_axis,) = open_axes
        return "destination" if open_axis == "origin" else "origin"
    if state.shape_draft is None or state.shape_draft is shape:
        return None
    elected = state.elected_place_roles
    if elected is None:
        return None
    place = set(dcids)
    elected_axis: RoleAxis | None = (
        "destination"
        if place & set(elected.destination_dcids)
        else "origin"
        if place & set(elected.origin_dcids)
        else None
    )
    return elected_axis if elected_axis in role_names_by_axis(shape) else None


def subject_shapes_refused(state: ResolveState) -> bool:
    """Whether every subject-only shape refuses ``role_incompatible`` for this query.

    True when role-bearing shapes were recalled and the query's resolved places name both
    flow ends. A subject-only shape has one subject slot, so it cannot hold an origin and a
    destination at once. `bind_place_roles` refuses subject-only shapes on this predicate,
    and `_phase_prune_and_rank` drops them from election on it once a role-bearing shape
    survives the score floor.
    """
    if not state.role_bearing_recalled:
        return False
    rcl = cast(Recall, state.rcl)
    query_directions = {
        role.axis
        for name in state.entities
        if rcl.resolved_entity_names.get(name) is not None
        for role in state.entity_roles.get(name) or ()
    }
    return {"origin", "destination"} <= query_directions


def query_role_axes(state: ResolveState) -> frozenset[RoleAxis]:
    """The flow axes the query needs a role-bearing shape to declare a role on: each axis
    a resolved place is directed to ("to Kenya" needs the destination) and each flow axis
    an operation runs across ("top recipients" needs the destination). An axis the query
    asks every member of (`state.open_axes`) is a preference among the shapes that hold
    these (`without_open_axis_gaps`)."""
    resolved = state.rcl.resolved_entity_names if state.rcl is not None else {}
    directed = {
        role.axis
        for name in state.entities
        if resolved.get(name) is not None
        for role in state.entity_roles.get(name) or ()
    }
    over = {
        operation.over
        for operation in state.operations
        if operation.over == "origin" or operation.over == "destination"
    }
    return frozenset(directed | over)


def _declares_axes(shape: ShapeDraft, axes: frozenset[RoleAxis]) -> bool:
    """Whether role-bearing `shape` declares a role on each of `axes`.

    A shape the index carries declares the axes some dataset the part may answer it from
    (`shape_part_datasets`) declares in its index family (`shape_family`,
    `code_index.members.family_declares_roles`), the test the dataset election applies,
    whichever members recall confirmed: a shape of a part naming a donor-only table lacks
    the recipient though another table of the family holds it. A family no dataset
    publishes declares the axes its rows hold. Any other shape declares the axes of its
    members' roles (`role_names_by_axis`)."""
    index = get_index()
    family = shape_family(shape, index)
    if family is None:
        return axes <= role_names_by_axis(shape).keys()
    return family_declares_roles(index, family, axes, datasets=shape_part_datasets(shape, index))


def lacks_query_axis(state: ResolveState, shape: ShapeDraft) -> bool:
    """Whether role-bearing `shape` declares no role on an axis the query needs
    (`query_role_axes`, read by `_declares_axes`): a provider-only shape for "aid to
    Kenya" or "top recipients of US aid". `bind_place_roles` would refuse it
    `role_incompatible`, or its answer could not run the operation. A subject-only shape
    lacks no axis: it binds a directed place as its subject."""
    axes = query_role_axes(state)
    return bool(axes) and is_role_bearing(shape) and not _declares_axes(shape, axes)


def without_open_axis_gaps(state: ResolveState, shapes: Sequence[ShapeDraft]) -> list[ShapeDraft]:
    """`shapes` less the role-bearing shapes the index carries that declare no role on an
    axis the query asks every member of (`state.open_axes`), each axis read beside the
    axes the query needs (`query_role_axes`).

    Such a shape leaves only when a role-bearing shape of `shapes` declares the axis
    (`_declares_axes`): "aid from France to every recipient" leaves a provider-only shape
    for one that lists recipients, and with no such shape the provider-only one stays and
    answers France's total. A subject-only shape stays, and so does a role-bearing shape
    the index does not carry: its recalled members are a sample of its family, so a debt
    shape recalled by its totals alone can still hold their by-creditor twins."""
    needed = query_role_axes(state)
    kept = list(shapes)
    for axis in sorted(state.open_axes - needed):
        axes = needed | {axis}
        declares = {id(s): _declares_axes(s, axes) for s in kept if is_role_bearing(s)}
        if any(declares.values()):
            kept = [s for s in kept if declares.get(id(s), True) or shape_family(s) is None]
    return kept


def capped_place_dcids(state: ResolveState) -> tuple[list[str], int]:
    """The query's places as an answer binds them, and the count of ordinary places named.

    Ordinary places keep query order, deduplicated by dcid so synonyms for one place stay
    scalar, and are capped at `max_standard_places`. Group placeholders are left out and
    their already-capped members follow, under their own limit (`max_child_places`). The
    result is in `place_order`, as a resubmit of the answer states its places."""
    rcl = cast(Recall, state.rcl)
    group_member_dcids_by_name = state.group_member_dcids_by_name
    ordinary_place_dcids = list(
        dict.fromkeys(
            rcl.resolved_entity_names[name]
            for name in state.entities
            if name in rcl.resolved_entity_names and name not in group_member_dcids_by_name
        )
    )
    group_place_dcids = list(
        dict.fromkeys(
            dcid for name in state.entities for dcid in group_member_dcids_by_name.get(name, ())
        )
    )
    capped = ordinary_place_dcids[: settings().max_standard_places]
    return place_order(capped + group_place_dcids), len(ordinary_place_dcids)


def bind_place_roles(
    state: ResolveState, shape: ShapeDraft, *, places: Sequence[str] | None = None
) -> PlaceRoles | PlaceRoleRefusal:
    """Bind the query's capped places to `shape`'s roles, or name the refusal.

    `places` defaults to the places place binding installs (`state.standard_place_dcids`).
    A caller that runs before it passes `capped_place_dcids`.

    ``role_incompatible`` covers a subject-only shape when the query names both flow ends
    (`subject_shapes_refused`), and a place on an axis the shape declares no role on.
    ``role_not_resolved`` covers a missing or contradicted axis.

    A role-bearing shape has no subject slot: each place takes its door-resolved axis,
    coordinated members included, and origins are kept for nearest-real probes. A place
    with no axis binds to the shape's one required role, a single place beside an operation
    over one flow axis to the other axis, or on an alternate that requires two to
    the axis the elected shape binds it under (`_undirected_place_axis`). It is
    ambiguous when no such axis exists, when a second place also lacks an axis, or when a
    directed place already holds that axis. Raw text compares extractor roles with the query
    wording; parsed role text has no separate preposition channel, though wording inside
    ``variable_text`` can contradict it.

    A subject-only shape binds every place as its subject, and a single directed place
    drops its direction (`discarded_directions`).

    A role-bearing shape leaves open each axis of `state.open_axes` that no place binds. A
    bound axis wins over an open one (`PlaceRoles.open_axes`).
    """
    rcl = cast(Recall, state.rcl)
    group_member_dcids_by_name = state.group_member_dcids_by_name
    standard_place_dcids = list(places if places is not None else state.standard_place_dcids)
    first_place = standard_place_dcids[0] if standard_place_dcids else None

    if not is_role_bearing(shape) and subject_shapes_refused(state):
        return "role_incompatible"

    if is_role_bearing(shape):
        # Keep only capped places and deduplicate by ``(DCID, axis)``. Repeated names on the
        # same axis collapse. Preserve self-flows on both axes. Group members inherit the
        # group's surface name and role.
        capped_place_dcids = set(standard_place_dcids)
        named_entities: list[tuple[str, str, DoorRole | None]] = []
        seen_dcid_axis: set[tuple[str, RoleAxis | None]] = set()
        for name in state.entities:
            member_dcids = group_member_dcids_by_name.get(name)
            if member_dcids is None:
                _dcid = rcl.resolved_entity_names.get(name)
                dcids = [_dcid] if _dcid is not None else []
            else:
                dcids = member_dcids
            for dcid in dcids:
                if dcid not in capped_place_dcids:
                    continue
                for role in state.entity_roles.get(name) or (None,):
                    key = (dcid, role.axis if role is not None else None)
                    if key in seen_dcid_axis:
                        continue
                    seen_dcid_axis.add(key)
                    named_entities.append((name, dcid, role))
        # Ambiguity counts query names; group members share their group's name.
        distinct_names = {name for name, _, _ in named_entities}
        undirected_names = {name for name in distinct_names if not state.entity_roles.get(name)}
        undirected_axis: RoleAxis | None = None
        if undirected_names:
            undirected_axis = _undirected_place_axis(
                state,
                shape,
                [dcid for name, dcid, role in named_entities if role is None],
                group=undirected_names <= group_member_dcids_by_name.keys(),
            )
            directed_axes = {role.axis for _, _, role in named_entities if role is not None}
            if (
                len(undirected_names) > 1
                or undirected_axis is None
                or undirected_axis in directed_axes
            ):
                return "role_not_resolved"
        # Check each occurrence's span. Either contradictory axis refuses the query.
        if _contradicted_names(state, distinct_names):
            return "role_not_resolved"
        destination_dcids: list[str] = []
        origin_dcids: list[str] = []
        for _name, dcid, role in named_entities:
            axis = role.axis if role is not None else undirected_axis
            if axis == "origin":
                origin_dcids.append(dcid)
            else:
                destination_dcids.append(dcid)
        destination_dcids = place_order(destination_dcids)
        origin_dcids = place_order(origin_dcids)
        try:
            resolved_entities = entities_by_role(
                shape=shape, by_axis={"destination": destination_dcids, "origin": origin_dcids}
            )
        except RoleEntityUnplaced:
            return "role_incompatible"
        bound_axes = {
            axis
            for axis, dcids in (("destination", destination_dcids), ("origin", origin_dcids))
            if dcids
        }
        return PlaceRoles(
            destination_dcids=destination_dcids,
            origin_dcids=origin_dcids,
            open_axes=state.open_axes - bound_axes,
            resolved_entities=resolved_entities,
            roles=[(dcid, "destination") for dcid in destination_dcids]
            + [(dcid, "origin") for dcid in origin_dcids],
            where_dcids=destination_dcids,
            obs_destination_dcid=destination_dcids[0] if destination_dcids else None,
        )

    resolved_entities = {}
    discarded: list[tuple[str, RoleAxis]] = []
    inferred: set[str] = set()
    if standard_place_dcids:
        try:
            resolved_entities = entities_by_role(shape=shape, subject=standard_place_dcids)
        except RoleEntityUnplaced:
            return "role_incompatible"
        # Each discarded direction is stated in place order, at its name's first place.
        position = {dcid: i for i, dcid in enumerate(standard_place_dcids)}
        named: list[tuple[int, str, RoleAxis]] = []
        for name in state.entities:
            if not state.entity_roles.get(name):
                continue
            dcids = group_member_dcids_by_name.get(name)
            if dcids is None:
                _dcid = rcl.resolved_entity_names.get(name)
                dcids = [_dcid] if _dcid is not None else []
            accepted = [position[dcid] for dcid in dcids if dcid in position]
            if accepted:
                # Use the first axis when a name occurs at both flow ends.
                first = state.entity_roles[name][0]
                named.append((min(accepted), name, first.axis))
                if not first.stated:
                    inferred.add(name)
        discarded = [(name, axis) for _position, name, axis in sorted(named)]
    return PlaceRoles(
        destination_dcids=[],
        origin_dcids=[],
        open_axes=frozenset(),
        resolved_entities=resolved_entities,
        roles=[(dcid, "observationAbout") for dcid in standard_place_dcids],
        where_dcids=list(standard_place_dcids),
        obs_destination_dcid=first_place,
        discarded_directions=tuple(discarded),
        inferred_directions=frozenset(inferred),
    )


def untold_direction_names(state: ResolveState) -> tuple[str, ...]:
    """The query names whose flow direction a `role_not_resolved` refusal could not tell,
    in query order: those the query gives no direction, else those whose stated direction
    its own wording contradicts (`role_text.contradicts`). Empty when every name's
    direction is told, as when materialisation refuses the roles."""
    undirected = tuple(name for name in state.entities if not state.entity_roles.get(name))
    return undirected or _contradicted_names(state, state.entities)


def _contradicted_names(state: ResolveState, names: Iterable[str]) -> tuple[str, ...]:
    """The names in `names`, in order, one of whose stated directions the query's own
    wording contradicts (`role_text.contradicts`)."""
    return tuple(
        name
        for name in names
        if any(
            contradicts(state.role_query, span=role.span, axis=role.axis)
            for role in state.entity_roles.get(name) or ()
        )
    )


def request_named_groups(state: ResolveState) -> tuple[NamedGroup, ...]:
    """The groups the query named, each with the members its name binds.

    `build_spec` names a role by one of these when the role binds exactly its members. A group
    with its own graph node takes the graph's label for it. One without, or one the graph gives
    no label, keeps the query's name.
    """
    groups: list[NamedGroup] = []
    for name, member_dcids in state.group_member_dcids_by_name.items():
        dcid = state.group_dcid_by_name.get(name)
        label = state.group_label_by_dcid.get(dcid, name) if dcid is not None else name
        groups.append(NamedGroup(label=label, dcid=dcid, member_dcids=frozenset(member_dcids)))
    return tuple(groups)


def group_request_names(state: ResolveState) -> dict[str, str]:
    """The query's name for each group `request_named_groups` labels otherwise, keyed by that
    label ("EU" for "European Union"), so a departure asks for the group in the request's
    words (`RegionResult.entity_words`)."""
    return {
        label: name
        for name, dcid in state.group_dcid_by_name.items()
        if (label := state.group_label_by_dcid.get(dcid, name)) != name
    }


def place_order(dcids: Iterable[str]) -> list[str]:
    """The order every answer states a set of places in: distinct dcids, sorted, as the
    identity core stores them (`spec_id._entities_member`). A first answer and a resubmit of
    it, which carries no naming order, name and probe their places alike."""
    return sorted(set(dcids))


def where_binding(where_dcids: list[str]) -> SlotBindingDraft | None:
    """The where-slot binding for a shape's bound places: a value, a set, or none."""
    if not where_dcids:
        return None
    if len(where_dcids) == 1:
        return SlotBindingDraft(
            axis="where", property_dcid=None, kind="value", value_dcids=[where_dcids[0]]
        )
    return SlotBindingDraft(
        axis="where", property_dcid=None, kind="set", value_dcids=list(where_dcids)
    )


def bound_after_election(
    state: ResolveState, admitted: ShapeDraft, roles: PlaceRoles, elected: ShapeDraft
) -> PlaceRoles | PlaceRoleRefusal:
    """The binding of the query's places to `elected`, the shape the role-class election
    returned for `admitted`, whose places bound as `roles`.

    The election can narrow a role-bearing shape to a class that requires roles the query's
    places do not name: a debt shape holding a recipient-only total and its by-creditor
    variants binds an undirected Kenya as the recipient, and the by-creditor class it can
    narrow to binds Kenya to no role. So a shape that stays role-bearing, or stays
    subject-only, keeps the binding it was admitted with. A shape the mint's walk turns from
    one kind into the other binds again (`bind_place_roles`): a binding to flow roles does
    not describe a subject's places, nor the reverse."""
    if is_role_bearing(elected) == is_role_bearing(admitted):
        return roles
    return bind_place_roles(state, elected)


def bind_alternate_place_roles(
    state: ResolveState, alternates: Iterable[tuple[int, ShapeDraft]]
) -> tuple[dict[int, PlaceRoles], dict[int, str]]:
    """Bind each alternate's places, returning bound roles and refusal reasons by index.

    A shape whose places carry a direction it cannot take refuses here and is logged; the
    caller keeps it out of the candidates rather than answer under another shape's roles.
    The caller may pass shapes nobody nominated, so the log names the binding failure only.
    """
    bound: dict[int, PlaceRoles] = {}
    refused: dict[int, str] = {}
    for index, shape in alternates:
        outcome = bind_place_roles(state, shape)
        if isinstance(outcome, PlaceRoles):
            bound[index] = outcome
            continue
        refused[index] = f"places: the query's places do not bind to this shape's roles ({outcome})"
        logger.info(
            "shape %r cannot bind the query's places to its roles (%s)",
            shape.shape_id,
            outcome,
        )
    return bound, refused
