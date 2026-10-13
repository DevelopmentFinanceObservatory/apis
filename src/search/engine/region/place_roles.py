"""Assign the query's place sets to one shape's roles, and settle each set to the places
a Spec binds for it.

The elected shape and every alternate assign through `bind_place_roles`, so an alternate's
role keys follow its own role declaration. The elected shape turns a refusal into its
no-data outcome; an alternate that refuses is dropped from the candidates. An assignment
holds each set whole. `settle_place_sets` then gives each set of one Spec one reading, its
node or its members, and `settled_roles` builds the Spec's binding from the result.

A set is one place key (`place_key`): two texts of one request that name one node are one
set, on every role and in every reading.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from code_index.members import family_declares_roles
from code_index.shared import get_index
from materialise.roles import axis_for_role
from search.engine.bind import SlotBindingDraft
from search.engine.interpret import PinnedResolved
from search.engine.region.decisions import PlaceSetNotAdmittedError, PlaceSetTooLargeError
from search.engine.region.place_set import PlaceSet
from search.engine.retrieval.bound_values import (
    RoleKeys,
    phrases_within,
    shape_family,
    shape_five_tuple,
    shape_part_datasets,
)
from search.engine.role_text import contradicts
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
class PlaceAssignment:
    """One shape's assignment of the query's place sets to its roles. It holds each set
    whole and no reading of it. `settle_place_sets` chooses the places a Spec binds."""

    # The sets on each role key the shape binds a place under, in query order: the flow
    # axes of a role-bearing shape, `observationAbout` of a subject-only one. A self-flow
    # lists one set on both axes. Each place key has one `PlaceSet` here (`part_sets`).
    sets: Mapping[EntityRoleKey, tuple[PlaceSet, ...]] = field(default_factory=dict)
    # The place keys of the sets whose members the request asks for: a set named for its
    # members, and one a rank runs over (`_members_asked`).
    members_asked: frozenset[str] = frozenset()
    # The flow axes the request asks every member of that no set fills. A subject-only
    # shape leaves it empty.
    open_axes: frozenset[RoleAxis] = frozenset()
    # Each directed text a subject-only shape reads as its subject, in query order: the
    # text, the axis it drops, and the place key of the set the text names.
    discarded_directions: tuple[tuple[str, RoleAxis, str], ...] = ()
    # The texts in `discarded_directions` whose direction the request's words did not state
    # (`DoorRole.stated`), so the extractor inferred it.
    inferred_directions: frozenset[str] = frozenset()

    @property
    def role_keys(self) -> RoleKeys:
        """The role keys an identity minted for the shape binds these sets under and the
        axes it leaves open, as `bound_values.entity_role_keys` reads them off a binding."""
        bound = frozenset(key for key, sets in self.sets.items() if sets)
        return RoleKeys(bound=bound, open=frozenset(self.open_axes) - bound)

    @property
    def binds_no_place(self) -> bool:
        """Whether the query names no place for the shape."""
        return not any(self.sets.values())

    def roles_of(self, place_set: PlaceSet) -> tuple[EntityRoleKey, ...]:
        """The role keys `place_set` fills."""
        return tuple(key for key, sets in self.sets.items() if place_set in sets)

    @property
    def distinct_sets(self) -> tuple[PlaceSet, ...]:
        """Every assigned set once, in role and query order."""
        return tuple({place_key(s): s for sets in self.sets.values() for s in sets}.values())


@dataclass(frozen=True)
class PlaceRoles:
    """One Spec's binding of the query's places, once its sets have settled."""

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
    # Directed query names a subject-only shape reads as its subject, with the dropped
    # axis, in place order, each name at its first place.
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
    # Each place the request named on its own, as the role key it fills and its dcid. A
    # member of a set bound on that role is then named as the request named it
    # (`request_named_groups`).
    named_alone: frozenset[tuple[EntityRoleKey, str]] = frozenset()


def _required_role_axis(shape: ShapeDraft) -> RoleAxis | None:
    """Return the flow axis of the shape's one required role, else None.

    None when the shape requires other than one role or that role has no flow axis.
    """
    required = required_role_names(shape)
    return axis_for_role(required[0]) if len(required) == 1 else None


def _undirected_place_axis(
    state: ResolveState, shape: ShapeDraft, place_set: PlaceSet
) -> RoleAxis | None:
    """Return the flow axis `place_set`, named without a direction, fills on `shape`. A set
    that holds members is a group here: a resolver group, continent or subregion.

    It is the axis of the shape's one required role. On a shape with a role on each axis,
    an operation running across one of them leaves that axis open, and a single place takes
    the other axis: "top creditors of external debt of Kenya" reads Kenya as the debtor.
    This holds for the elected shape and every alternate alike. A group can read either way
    ("top recipients of aid in Africa" scopes the ranked recipients, "top creditors of
    external debt of Africa" names the debtors), so it skips this rule. Otherwise a shape
    that requires two roles takes
    the axis the elected shape (`state.shape_draft`) assigned the place to, when it declares
    a role on that axis: an alternate reads the variable another way, not the place, so a
    debt total's recipient stays the recipient of a by-creditor alternate. That axis is
    read from the elected shape's assignment as admitted (`state.elected_place_roles`), taken
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
    if not place_set.members and len(open_axes) == 1 and len(names) == 2:
        (open_axis,) = open_axes
        return "destination" if open_axis == "origin" else "origin"
    if state.shape_draft is None or state.shape_draft is shape:
        return None
    elected = state.elected_place_roles
    if elected is None:
        return None
    elected_axis: RoleAxis | None = next(
        (
            axis
            for axis in ("destination", "origin")
            if any(
                place_key(placed) == place_key(place_set) for placed in elected.sets.get(axis, ())
            )
        ),
        None,
    )
    return elected_axis if elected_axis in role_names_by_axis(shape) else None


def subject_shapes_refused(state: ResolveState) -> bool:
    """Whether every subject-only shape refuses ``role_incompatible`` for this query.

    True when a role-bearing eligible shape outranks the subject-only ones
    (`ResolveState.role_bearing_outranks`) and the query's resolved places name both flow
    ends. A subject-only shape has one subject slot, so it cannot hold an origin and a
    destination at once. `bind_place_roles` refuses subject-only shapes on this predicate,
    and `_phase_prune_and_rank` drops them from election on it.
    """
    if not state.role_bearing_outranks:
        return False
    query_directions = {
        role.axis
        for name in state.entities
        if name in state.place_sets
        for role in state.entity_roles.get(name) or ()
    }
    return {"origin", "destination"} <= query_directions


def query_role_axes(state: ResolveState) -> frozenset[RoleAxis]:
    """The flow axes the query needs a role-bearing shape to declare a role on: each axis
    a resolved place is directed to ("to Kenya" needs the destination) and each flow axis
    an operation runs across ("top recipients" needs the destination). An axis the query
    asks every member of (`state.open_axes`) is a preference among the shapes that hold
    these (`without_open_axis_gaps`)."""
    directed = {
        role.axis
        for name in state.entities
        if name in state.place_sets
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


def without_unasked_axes(
    state: ResolveState,
    shapes: Sequence[ShapeDraft],
    phrases_read: Callable[[ShapeDraft], AbstractSet[str]],
) -> list[ShapeDraft]:
    """`shapes` less each role-bearing shape that publishes on an axis the query does not
    ask for where another role-bearing shape of `shapes` reads as much of the query without
    it, the converse of `lacks_query_axis`.

    A shape's unasked axes are the flow axes it declares a role on (`_declares_axes`) beyond
    those the query needs (`query_role_axes`) or asks every member of (`state.open_axes`). A
    shape leaves when another declares a proper subset of its unasked axes and reads every
    query phrase it reads (`phrases_read`, compared by `bound_values.phrases_within`). "Aid
    from Japan" leaves DAC2a and CRS disbursements, which publish recipients, for DAC1's
    donor totals, since all three read "aid" alike, while "imputed multilateral ODA from
    Germany" keeps DAC2a's imputed table, whose "imputed" DAC1 does not read.

    Shapes of other five-tuples compare too, since a question that names no measure can
    recall a table of another one. "ODA from Japan" leaves CRS commitments, which publish
    recipients and read only "ODA", for DAC1's donor disbursements. A measure word reads
    only on the five-tuple or constraint value it names (`bound_values.held_codes`), so
    "ODA commitments from Japan" keeps commitments over DAC1's disbursements, and "aid
    attributed to donors via multilaterals" keeps the imputed tables, whose measure reads
    "attributed to donors".
    "Grant equivalent ODA from France" elects DAC1's grant-equivalent total over the CRS
    grant equivalents: recall adds a row of that DAC1 family beside the CRS hits
    (`clients/graph.with_rival_families`, applied by `recall_hits`), it reads the measure
    word as the CRS does, and it declares no recipient axis. A query naming no measure word
    reads none on any table, so "aid from France" leaves an imputed table of another
    five-tuple for DAC1 as it leaves CRS. A shape recall added for its five-tuple
    (`ShapeDraft.rival`) displaces only shapes of that five-tuple: the query did not recall
    it, so "adolescent birth rate", whose cut holds CRS commitments and flows, keeps the CRS
    flows beside the DAC1 commitments total added for the commitments.

    Both shapes must be ones the index carries (`shape_family`). Any other holds its
    recalled members alone, a sample of its family, as in `without_open_axis_gaps`. Its
    axes are the sample's, and it reads no index code, so its empty reading would lie within
    any other's ("debt from France" would leave a debt series for DAC1 aid). The rule
    stands down when a named place has no direction: "imputed rent Kenya" leaves the axis
    the place fills open, so no axis is unasked.
    """
    if any(
        name in state.place_sets and not state.entity_roles.get(name) for name in state.entities
    ):
        return list(shapes)
    asked = query_role_axes(state) | state.open_axes
    flow_axes: tuple[RoleAxis, ...] = ("origin", "destination")
    unasked = {
        id(shape): frozenset(
            axis
            for axis in flow_axes
            if axis not in asked and _declares_axes(shape, frozenset({axis}))
        )
        for shape in shapes
        if is_role_bearing(shape) and shape_family(shape) is not None
    }

    def leaves(shape: ShapeDraft) -> bool:
        return id(shape) in unasked and any(
            other is not shape
            and id(other) in unasked
            and unasked[id(other)] < unasked[id(shape)]
            and (not other.rival or shape_five_tuple(other) == shape_five_tuple(shape))
            and phrases_within(phrases_read(shape), phrases_read(other))
            for other in shapes
        )

    return [shape for shape in shapes if not leaves(shape)]


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


def names_a_place(state: ResolveState) -> bool:
    """Whether the query names a place that resolved."""
    return any(name in state.place_sets for name in state.entities)


def first_spellings(names: Iterable[str], place_sets: Mapping[str, PlaceSet]) -> dict[str, str]:
    """The first of `names` that spells each place a set it names holds: a group's members
    and any other set's node. A region's members and a group's node are spelled by no
    name."""
    spelled: dict[str, str] = {}
    for name in names:
        place_set = place_sets.get(name)
        if place_set is None:
            continue
        if place_set.kind == "group":
            for dcid in place_set.members or ():
                spelled.setdefault(dcid, name)
        elif place_set.node is not None:
            spelled.setdefault(place_set.node.dcid, name)
    return spelled


def place_texts(names: Iterable[str], place_sets: Mapping[str, PlaceSet]) -> dict[str, str]:
    """The request's text for each place a set named in `names` holds, at its first
    spelling (`first_spellings`): a set's own name for its node, and the resolver's name
    for a group's member. A place has one text whichever Spec binds it."""
    return {
        dcid: (
            place_sets[name].member_names.get(dcid, dcid)
            if place_sets[name].kind == "group"
            else name
        )
        for dcid, name in first_spellings(names, place_sets).items()
    }


def place_key(place_set: PlaceSet) -> str:
    """What makes two named sets one place: the node, or for a set with none its member
    dcids, sorted. A set with no node is a resolver group, which holds members, so two texts
    the resolver reads as one group are one set."""
    if place_set.node is not None:
        return place_set.node.dcid
    return "members:" + ",".join(sorted(place_set.members or ()))


def part_sets(state: ResolveState) -> dict[str, PlaceSet]:
    """The place sets the part names, by place key, in query order.

    Texts that name one place are one set, which states what any of them says of the
    place, in whatever order they come. It expands when any text's set does: it takes that
    set's kind, members and graph label, so a group's node pinned by dcid beside the
    group's name is the group. Its reading is the members when any text asked for them. Its
    name is the first text that is not a pinned dcid's key, else the graph's label for the
    node, else the first text: a pin's key names the node for the caller and no more.
    """
    texts: dict[str, list[PlaceSet]] = {}
    for name in state.entities:
        place_set = state.place_sets.get(name)
        if place_set is not None:
            texts.setdefault(place_key(place_set), []).append(place_set)
    sets: dict[str, PlaceSet] = {}
    for key, named in texts.items():
        # The first text's set that expands, else the first text's.
        whole = next((place_set for place_set in named if place_set.expands), named[0])
        asked = any(place_set.reading == "members" for place_set in named)
        first = named[0].name
        sets[key] = dataclasses.replace(
            whole,
            name=next(
                (s.name for s in named if _pin_label(state, s.name) is None),
                whole.label or _pin_label(state, first) or first,
            ),
            reading="members" if asked else whole.reading,
        )
    return sets


def _pin_label(state: ResolveState, name: str) -> str | None:
    """The graph's label for the dcid the caller pinned `name` to, or `None` when `name`
    is no pin. A pin's text is only the key the caller pinned the dcid under."""
    resolution = None if state.rcl is None else state.rcl.resolutions.get(name)
    return resolution.entity.name if isinstance(resolution, PinnedResolved) else None


def _members_asked(
    state: ResolveState,
    sets: Mapping[EntityRoleKey, tuple[PlaceSet, ...]],
    named: Mapping[str, PlaceSet],
) -> frozenset[str]:
    """The place keys of the sets in `sets` whose members the request asks for, where
    `named` is every set of the part (`part_sets`).

    A set named for its members is one. So is a set a `rank` runs over, since one place
    cannot be ranked. A rank runs over a set when it is over the axis the set fills, where
    a subject fills `place`, when it is over `place`, `origin` or `destination` on a
    subject-only shape, where all three name the subject, and when it is over `place` and
    the set is the only one of the part that expands. A rank over `category` or `year`,
    and an aggregate, ask for no members.
    """
    ranked = {
        operation.over
        for operation in state.operations
        if operation.verb == "rank" and operation.over in ("place", "origin", "destination")
    }
    expanding = {key for key, place_set in named.items() if place_set.expands}
    asked: set[str] = set()
    for role_key, placed in sets.items():
        over_axis = bool(ranked) if role_key == "observationAbout" else role_key in ranked
        for place_set in placed:
            key = place_key(place_set)
            only_set = "place" in ranked and expanding == {key}
            if place_set.reading == "members" or over_axis or only_set:
                asked.add(key)
    return frozenset(asked)


def bind_place_roles(state: ResolveState, shape: ShapeDraft) -> PlaceAssignment | PlaceRoleRefusal:
    """Assign the query's place sets to `shape`'s roles, or name the refusal.

    ``role_incompatible`` covers a subject-only shape when the query names both flow ends
    (`subject_shapes_refused`), and a place on an axis the shape declares no role on.
    ``role_not_resolved`` covers a missing or contradicted axis.

    A role-bearing shape has no subject slot: each set takes its door-resolved axis, and
    origins are kept for nearest-real probes. A set with no axis fills the shape's one
    required role, a single place beside an operation over one flow axis the other axis, or
    on an alternate that requires two the axis the elected shape assigns it
    (`_undirected_place_axis`). It is ambiguous when no such axis exists, when a second set
    also lacks an axis, or when a directed set already holds that axis. Raw text compares
    extractor roles with the query wording; parsed role text has no separate preposition
    channel, though wording inside ``variable_text`` can contradict it.

    A subject-only shape takes every set as its subject, and a directed set drops its
    direction (`discarded_directions`).

    Texts that name one place are one set (`part_sets`). On a role-bearing shape it fills
    each axis any of its texts is directed to, and is undirected only when none of them
    carries a direction. A subject-only shape drops a direction for each text that
    carries one.

    A role-bearing shape leaves open each axis of `state.open_axes` that no set fills. A
    filled axis wins over an open one (`PlaceAssignment.open_axes`).
    """
    place_sets = state.place_sets
    named = part_sets(state)

    if not is_role_bearing(shape) and subject_shapes_refused(state):
        return "role_incompatible"

    if is_role_bearing(shape):
        # The axes each set's texts direct it to, in query order. A self-flow holds both.
        axes_of: dict[str, dict[RoleAxis, None]] = {key: {} for key in named}
        for name in state.entities:
            if name not in place_sets:
                continue
            for role in state.entity_roles.get(name) or ():
                axes_of[place_key(place_sets[name])][role.axis] = None
        undirected = [key for key, axes in axes_of.items() if not axes]
        if undirected:
            if len(undirected) > 1:
                return "role_not_resolved"
            undirected_axis = _undirected_place_axis(state, shape, named[undirected[0]])
            directed_axes = {axis for axes in axes_of.values() for axis in axes}
            if undirected_axis is None or undirected_axis in directed_axes:
                return "role_not_resolved"
            axes_of[undirected[0]][undirected_axis] = None
        # Check each occurrence's span. Either contradictory axis refuses the query.
        if _contradicted_names(state, (name for name in state.entities if name in place_sets)):
            return "role_not_resolved"
        by_axis: dict[RoleAxis, list[PlaceSet]] = {"destination": [], "origin": []}
        for key, axes in axes_of.items():
            for axis in axes:
                by_axis["origin" if axis == "origin" else "destination"].append(named[key])
        declared = role_names_by_axis(shape)
        sets: dict[EntityRoleKey, tuple[PlaceSet, ...]] = {
            axis: tuple(placed) for axis, placed in by_axis.items() if placed
        }
        if any(axis not in declared for axis in sets):
            return "role_incompatible"
        return PlaceAssignment(
            sets=sets,
            members_asked=_members_asked(state, sets, named),
            open_axes=state.open_axes - frozenset(sets),
        )

    subject = tuple(named.values())
    sets = {"observationAbout": subject} if subject else {}
    discarded: list[tuple[str, RoleAxis, str]] = []
    inferred: set[str] = set()
    for name in dict.fromkeys(state.entities):
        if name not in place_sets or not state.entity_roles.get(name):
            continue
        # Use the first axis when a name occurs at both flow ends.
        first = state.entity_roles[name][0]
        discarded.append((name, first.axis, place_key(place_sets[name])))
        if not first.stated:
            inferred.add(name)
    return PlaceAssignment(
        sets=sets,
        members_asked=_members_asked(state, sets, named),
        discarded_directions=tuple(discarded),
        inferred_directions=frozenset(inferred),
    )


# The places one Spec binds on each role key, in `place_order`.
type PlaceLists = dict[EntityRoleKey, tuple[str, ...]]


def _members_reading(place_set: PlaceSet) -> tuple[str, ...]:
    """The set's members, or its node when it does not expand or holds no member."""
    return place_set.members or place_set.dcids("default")


# The readings one Spec can give a set: one of node or members, or either, which the
# Spec's own read decides (`set_choices`). A set read whole binds its node or its total.
type SetChoice = Literal["node", "members", "either"]


def _wholes(
    place_set: PlaceSet,
    roles: Sequence[EntityRoleKey],
    *,
    admits: Callable[[EntityRoleKey, str], bool],
) -> tuple[str, ...]:
    """The places that can read `place_set` whole on `roles`, in the order a Spec tries them:
    its node when every role admits the node's type, then its total (`PlaceSet.total`) when
    every role is a flow axis. A total is admitted by no type: the Spec's read decides whether
    its dataset publishes the total on the role."""
    node = place_set.node
    wholes: list[str] = []
    if node is not None and all(admits(role, node.graph_type) for role in roles):
        wholes.append(node.dcid)
    if place_set.total is not None and roles and "observationAbout" not in roles:
        wholes.append(place_set.total)
    return tuple(wholes)


def set_choices(
    assignment: PlaceAssignment, *, admits: Callable[[EntityRoleKey, str], bool]
) -> dict[str, SetChoice | None]:
    """The readings one Spec can give each set of `assignment`, by place key, where
    `admits(role key, graph type)` says whether a role admits a type.

    A reading is used only on roles that admit its type. A set the request asks the members
    of reads its members when every role it fills admits their type, and has no reading
    otherwise. Any other set with members can read them when every role it fills admits their
    type, and can read itself whole (`_wholes`) at its node when every role admits the node's
    type or at its total on flow roles: `either` when it can read both ways, its node when only
    the node is admitted, its members when it cannot read whole, and no reading when neither
    holds. A set with no member reads its node, and a set with no node its members, when
    admitted. Members that share no stated type are admitted on no role, as role admission
    judges them (`phase_intake._admission_types`).
    """
    choices: dict[str, SetChoice | None] = {}
    for place_set in assignment.distinct_sets:
        key = place_key(place_set)
        if not place_set.members:
            choices[key] = "node"
            continue
        roles = assignment.roles_of(place_set)
        node = place_set.node
        node_ok = node is not None and all(admits(role, node.graph_type) for role in roles)
        member_type = place_set.member_type
        members_ok = member_type is not None and all(admits(role, member_type) for role in roles)
        if key in assignment.members_asked:
            choices[key] = "members" if members_ok else None
        elif members_ok and _wholes(place_set, roles, admits=admits):
            choices[key] = "either"
        elif node_ok:
            choices[key] = "node"
        else:
            choices[key] = "members" if members_ok else None
    return choices


def undecided_sets(
    assignment: PlaceAssignment, *, admits: Callable[[EntityRoleKey, str], bool]
) -> tuple[PlaceSet, ...]:
    """The sets of `assignment` whose reading a Spec's own read decides, those that can
    read either whole or as their members (`set_choices`)."""
    choices = set_choices(assignment, admits=admits)
    return tuple(s for s in assignment.distinct_sets if choices[place_key(s)] == "either")


def settle_place_sets(
    assignment: PlaceAssignment,
    *,
    admits: Callable[[EntityRoleKey, str], bool],
    has_data: Callable[[PlaceLists], bool],
    bound: int | None,
) -> PlaceLists:
    """Settle each set of `assignment` to one reading for one Spec, its node, its total or its
    members, and return the places the Spec binds on each role key.

    A set with one reading (`set_choices`) takes it. A set that can read either binds itself
    whole when the Spec holds data there, trying its node, then its total (`_wholes`):
    `has_data` is the Spec's own read over the places it is given. So a group binds the total
    its dataset publishes for it ("Africa (DAC)") only when that dataset publishes the total
    on the role, and its members otherwise. When such sets fill different roles, one each,
    they are first read together whole, each combination of their wholes in order, and bind
    the first the Spec holds data at: data held between two nodes, aid from one subregion to
    another, or between two totals, G7 aid to Africa, is that pairing. Otherwise each is read
    whole on each role it fills, every other set at its members. A set on two roles is read
    whole on both and takes one reading. Two sets on one role settle apart, so the role can
    hold one set's node beside another's members. When more than one set reads whole and the
    Spec holds no data at the places they give together, every one of them binds its members.
    Every read here shares the Spec's one batch (`UnsettledPlace.settle`).

    A `has_data` that raises leaves no reading chosen. The failure is the Spec's read
    failing.

    Raises `PlaceSetNotAdmittedError` naming the sets with no reading, before any read.
    Role admission keeps only shapes that give every set a reading
    (`phase_intake._prune_shapes_by_admission`), so this is reached only by a shape whose
    assignment the prune could not build, or one the election narrowed or re-assigned.
    Raises `PlaceSetTooLargeError` when the places chosen exceed `bound`, counted as the
    distinct dcids on each role summed over the roles. The count is of the reading chosen,
    so a set with more members than `bound` binds a node or total that holds data. `None`
    checks no bound.
    """
    choices = set_choices(assignment, admits=admits)
    if refused := tuple(s for s in assignment.distinct_sets if choices[place_key(s)] is None):
        raise PlaceSetNotAdmittedError(refused)
    nodes = {
        place_key(s): (s.node.dcid,)
        for s in assignment.distinct_sets
        if choices[place_key(s)] == "node" and s.node is not None
    }
    readings = {place_key(s): _members_reading(s) for s in assignment.distinct_sets} | nodes

    def places(readings: Mapping[str, tuple[str, ...]]) -> PlaceLists:
        return {
            key: tuple(place_order(dcid for s in placed for dcid in readings[place_key(s)]))
            for key, placed in assignment.sets.items()
            if placed
        }

    members = places(readings)
    undecided = [s for s in assignment.distinct_sets if choices[place_key(s)] == "either"]
    wholes = {place_key(s): _wholes(s, assignment.roles_of(s), admits=admits) for s in undecided}
    filled = [role for s in undecided for role in assignment.roles_of(s)]
    if len(undecided) > 1 and len(filled) == len(set(filled)):
        keys = [place_key(s) for s in undecided]
        for combination in itertools.product(*(wholes[key] for key in keys)):
            paired = places(
                {
                    **readings,
                    **{key: (whole,) for key, whole in zip(keys, combination, strict=True)},
                }
            )
            if has_data(paired):
                return _within_bound(paired, bound)
    at_whole: dict[str, tuple[str, ...]] = {}
    for place_set in undecided:
        key = place_key(place_set)
        for whole in wholes[key]:
            alone = {**members, **dict.fromkeys(assignment.roles_of(place_set), (whole,))}
            if has_data(alone):
                at_whole[key] = (whole,)
                break
    settled = places({**readings, **at_whole})
    if len(at_whole) > 1 and not has_data(settled):
        settled = members
    return _within_bound(settled, bound)


def _within_bound(places: PlaceLists, bound: int | None) -> PlaceLists:
    """`places`, or `PlaceSetTooLargeError` when their distinct dcids per role, summed over
    the roles, exceed `bound`."""
    count = sum(len(dcids) for dcids in places.values())
    if bound is not None and count > bound:
        raise PlaceSetTooLargeError(count=count, bound=bound)
    return places


def settled_roles(shape: ShapeDraft, assignment: PlaceAssignment, places: PlaceLists) -> PlaceRoles:
    """The binding of `shape` to `places`, the places `settle_place_sets` chose for
    `assignment` on each role key.

    A role-bearing shape binds its destinations as the where places and keeps its origins.
    A subject-only shape binds every place as its subject, and states each direction it
    drops in place order, at the first place of the set its text names. `places` can leave
    a set no place, since the read that tests one set's node binds that node alone on its
    role. Such a set's direction is stated after the others.
    """
    named_alone = frozenset(
        (key, place_set.node.dcid)
        for key, placed in assignment.sets.items()
        for place_set in placed
        if place_set.node is not None and not place_set.expands
    )
    if is_role_bearing(shape):
        destination_dcids = list(places.get("destination", ()))
        origin_dcids = list(places.get("origin", ()))
        return PlaceRoles(
            destination_dcids=destination_dcids,
            origin_dcids=origin_dcids,
            open_axes=assignment.open_axes,
            named_alone=named_alone,
            resolved_entities=entities_by_role(
                shape=shape, by_axis={"destination": destination_dcids, "origin": origin_dcids}
            ),
            roles=[(dcid, "destination") for dcid in destination_dcids]
            + [(dcid, "origin") for dcid in origin_dcids],
            where_dcids=destination_dcids,
            obs_destination_dcid=destination_dcids[0] if destination_dcids else None,
        )
    subject = list(places.get("observationAbout", ()))
    bound = set(subject)
    by_key = {place_key(s): s for s in assignment.distinct_sets}

    def first_place(key: str) -> str | None:
        """The first place the set at `key` binds: its node when bound, else its first
        member, else `None`."""
        place_set = by_key[key]
        if place_set.node is not None and place_set.node.dcid in bound:
            return place_set.node.dcid
        return min((dcid for dcid in place_set.members or () if dcid in bound), default=None)

    def place_order_of(dropped: tuple[str, RoleAxis, str]) -> tuple[bool, str, str, str]:
        """Where a dropped direction is stated: at its set's first place, those whose set
        binds none last, then by text and axis."""
        name, axis, key = dropped
        place = first_place(key)
        return place is None, place or "", name, axis

    return PlaceRoles(
        destination_dcids=[],
        origin_dcids=[],
        open_axes=frozenset(),
        resolved_entities=entities_by_role(shape=shape, subject=subject) if subject else {},
        roles=[(dcid, "observationAbout") for dcid in subject],
        where_dcids=subject,
        obs_destination_dcid=subject[0] if subject else None,
        discarded_directions=tuple(
            (name, axis)
            for name, axis, key in sorted(assignment.discarded_directions, key=place_order_of)
        ),
        inferred_directions=assignment.inferred_directions,
        named_alone=named_alone,
    )


def untold_direction_names(state: ResolveState) -> tuple[str, ...]:
    """The query names whose flow direction a `role_not_resolved` refusal could not tell,
    in query order, one per place set: each set no text of which carries a direction,
    under its name (`part_sets`), and each name that resolved to no set and carries none.
    Else the names whose stated direction the query's own wording contradicts
    (`role_text.contradicts`), the first such text of each set. Empty when every name's
    direction is told, as when materialisation refuses the roles."""
    named = part_sets(state)
    directed = {
        place_key(state.place_sets[name])
        for name in state.entities
        if name in state.place_sets and state.entity_roles.get(name)
    }
    undirected: dict[str, str] = {}
    for name in state.entities:
        place_set = state.place_sets.get(name)
        if place_set is None:
            if not state.entity_roles.get(name):
                undirected.setdefault(name, name)
        elif (key := place_key(place_set)) not in directed:
            undirected.setdefault(key, named[key].name)
    if undirected:
        return tuple(undirected.values())
    contradicted: dict[str, str] = {}
    for name in _contradicted_names(state, state.entities):
        place_set = state.place_sets.get(name)
        contradicted.setdefault(name if place_set is None else place_key(place_set), name)
    return tuple(contradicted.values())


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


def request_named_groups(state: ResolveState, roles: PlaceRoles) -> tuple[NamedGroup, ...]:
    """The groups the query named, each with the members its name binds: every resolver
    group, continent and subregion that holds a member.

    `build_spec` names a role by one of these when the role binds exactly its members or its
    total (`PlaceSet.total`). A group with its own graph node takes that node's type and the
    graph's label for it. One without, or one the graph gives no label, keeps the query's
    name. Each states the
    members the query also named as places of their own, on the role `roles`, one Spec's
    binding, puts each of those places on (`PlaceRoles.named_alone`).
    """
    return tuple(
        NamedGroup(
            label=place_set.name if place_set.label is None else place_set.label,
            dcid=None if place_set.node is None else place_set.node.dcid,
            entity_type=None if place_set.node is None else place_set.node.graph_type,
            member_dcids=frozenset(place_set.members),
            named_alone=frozenset(
                (role, dcid) for role, dcid in roles.named_alone if dcid in place_set.members
            ),
            total=place_set.total,
        )
        for place_set in part_sets(state).values()
        if place_set.members
    )


def group_request_names(state: ResolveState) -> dict[str, str]:
    """The query's name for each group `request_named_groups` labels otherwise, keyed by that
    label ("EU" for "European Union"), so a departure asks for the group in the request's
    words (`RegionResult.entity_words`). A label the request posts as a text of its own is
    left out, so no text of the request is read as another of its texts."""
    return {
        place_set.label: place_set.name
        for place_set in part_sets(state).values()
        if place_set.members
        and place_set.label is not None
        and place_set.label != place_set.name
        and place_set.label not in state.entities
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
    state: ResolveState, admitted: ShapeDraft, roles: PlaceAssignment, elected: ShapeDraft
) -> PlaceAssignment | PlaceRoleRefusal:
    """The assignment of the query's places to `elected`, the shape the role-class election
    returned for `admitted`, whose places were assigned as `roles`.

    The election can narrow a role-bearing shape to a class that requires roles the query's
    places do not name: a debt shape holding a recipient-only total and its by-creditor
    variants takes an undirected Kenya as the recipient, and the by-creditor class it can
    narrow to gives Kenya no role. So a shape that stays role-bearing, or stays
    subject-only, keeps the assignment it was admitted with. A shape the mint's walk turns
    from one kind into the other is assigned again (`bind_place_roles`): an assignment to
    flow roles does not describe a subject's places, nor the reverse."""
    if is_role_bearing(elected) == is_role_bearing(admitted):
        return roles
    return bind_place_roles(state, elected)


def bind_alternate_place_roles(
    state: ResolveState, alternates: Iterable[tuple[int, ShapeDraft]]
) -> tuple[dict[int, PlaceAssignment], dict[int, str]]:
    """Assign each alternate's places, returning assignments and refusal reasons by index.

    A shape whose places carry a direction it cannot take refuses here and is logged; the
    caller keeps it out of the candidates rather than answer under another shape's roles.
    The caller may pass shapes nobody nominated, so the log names the binding failure only.
    """
    bound: dict[int, PlaceAssignment] = {}
    refused: dict[int, str] = {}
    for index, shape in alternates:
        outcome = bind_place_roles(state, shape)
        if isinstance(outcome, PlaceAssignment):
            bound[index] = outcome
            continue
        refused[index] = f"places: the query's places do not bind to this shape's roles ({outcome})"
        logger.info(
            "shape %r cannot bind the query's places to its roles (%s)",
            shape.shape_id,
            outcome,
        )
    return bound, refused
