"""Assign the query's place sets to one shape's roles, and settle each set to the places
a Spec binds for it.

The elected shape and every alternate assign through `bind_place_roles`, so an alternate's
role keys follow its own role declaration. Beside the places the query names, a shape binds
the party a word of the part names on an axis the query leaves unnamed (`held_party_words`).
The elected shape turns a refusal into its no-data outcome; an alternate that refuses is
dropped from the candidates. An assignment holds each set whole. `settle_place_sets` then
gives each set of one Spec one reading, its node or its members, and `settled_roles` builds
the Spec's binding from the result.

A set is one place key (`place_key`): two texts of one request that name one node are one
set, on every role and in every reading.

A shape also binds a whole total (`whole_totals`) on at most one side. A text of the part
naming a whole population ("official donors") binds its parties on its side as named parties
(`_named_wholes`). Where no text names one, binding repair offers the first side the part
totals (`totalled_axes`), origin first, where every dataset the shape answers from reads the
total whole (`whole_total_candidates`), and keeps it where the bindings settled under the
total's role keys hold its series (`holds_whole_total`).
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
from code_index.shared import get_index, get_measure_index
from code_index.tokens import Token, tokenize, word_tokens
from code_index.types import PartyWord
from entity_resolution.types import Kind, Resolved, entity_kind
from materialise.roles import axis_for_role
from search.engine.bind import SlotBindingDraft
from search.engine.bind_repair import stated_words_negated, word_stated
from search.engine.function_words import (
    AMOUNT_WORDS,
    DATA_WORDS,
    FLOW_VERBS,
    FUNCTION_WORDS,
)
from search.engine.interpret import PinnedResolved
from search.engine.qualifier_match import CONNECTIVES, read_by_vocabulary
from search.engine.region.coverage_series import holds_series
from search.engine.region.decisions import PlaceSetNotAdmittedError, PlaceSetTooLargeError
from search.engine.region.place_set import PlaceSet
from search.engine.region.whole_totals import (
    WholeReading,
    covers_unpublished,
    named_whole_label,
    requested_years,
    whole_reading,
)
from search.engine.retrieval.bound_values import (
    RoleKeys,
    declared_datasets,
    phrases_within,
    ratio_base_axis,
    shape_family,
    shape_five_tuple,
    shape_part_datasets,
)
from search.engine.role_text import contradicts
from search.engine.series_declarations import declared_side
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
    # Each directed name a subject-only shape reads as its subject, in query order: the
    # text it names a place of (`ResolveState.entity_texts`), the axis it drops, and the
    # place key of the set the name names.
    discarded_directions: tuple[tuple[str, RoleAxis, str], ...] = ()
    # The text and axis of each entry of `discarded_directions` whose direction the
    # request's words did not state (`DoorRole.stated`), so the extractor inferred it.
    inferred_directions: frozenset[tuple[str, RoleAxis]] = frozenset()
    # The side the shape reads as its whole population, whose parties' sets fill its role in
    # `sets` (`whole_total_candidates`), `None` when it reads none.
    whole: WholeReading | None = None
    # The sides no place is named on where a whole population a text names reads as left
    # unnamed (`_named_wholes`) or is listed (`ResolveState.whole_listed`), set where the
    # places are bound (`bind_place_roles`). No `role_scope_defaulted` warning says the
    # request named nothing there.
    unnamed_wholes: frozenset[RoleAxis] = frozenset()

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
    # The texts of the directed query names a subject-only shape reads as its subject, with
    # the dropped axis, in place order, each at its first place.
    discarded_directions: tuple[tuple[str, RoleAxis], ...] = ()
    # The entries of `discarded_directions` whose direction the request's words did not
    # state (`DoorRole.stated`), so the extractor inferred it.
    inferred_directions: frozenset[tuple[str, RoleAxis]] = frozenset()
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
    # The side the Spec reads as its whole population (`PlaceAssignment.whole`).
    whole: WholeReading | None = None
    # The sides a named whole population reads as left unnamed
    # (`PlaceAssignment.unnamed_wholes`).
    unnamed_wholes: frozenset[RoleAxis] = frozenset()


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
    external debt of Africa" names the debtors), so it skips this rule. A ratio whose
    datasets declare the axis whose party's own total it divides by
    (`bound_values.ratio_base_axis`) reads the place, a group too, as that party: "ODA as a
    share of GNI for Norway" reads Norway as the provider on DAC1's share of the donor's
    GNI and as the recipient on DAC2a's share of the recipient's, where Norway holds no
    data, so the menu shows that reading has none (`availability.series_at_places`) and the
    answer states the base the question named. This holds for the elected shape and every
    alternate alike. Otherwise an alternate takes the axis the elected shape
    (`state.shape_draft`) assigned the place to, when it declares a role on that axis: an alternate reads the variable another way, not
    the place, so a debt total's recipient stays the recipient of a by-creditor alternate,
    and a place the elected shape reads on the one direction with data
    (`phase_place_materialise._settle_direction`) keeps that direction on every alternate
    the status rule weighs. That axis is read from the elected shape's assignment as
    admitted (`state.elected_place_roles`), taken before the role-class election narrowed
    it, since the narrowed shape can require two roles.
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
    base = ratio_base_axis(shape, get_index())
    if base is not None and base in names:
        return base
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


def direction_assignments(
    state: ResolveState, shape: ShapeDraft
) -> tuple[tuple[RoleAxis, PlaceAssignment], ...]:
    """The readings of a direction the question leaves unstated: when `shape` refuses the
    query's places only because one set is named without a direction that no rule places
    (`_undirected_place_axis`), the assignment placing that set on each axis the shape
    declares and no directed set holds, origin first, each with its axis. Empty when the
    places bind without one, or refuse for another reason."""
    if not is_role_bearing(shape) or bind_place_roles(state, shape) != "role_not_resolved":
        return ()
    readings: list[tuple[RoleAxis, PlaceAssignment]] = []
    for axis in ("origin", "destination"):
        assignment = bind_place_roles(state, shape, direction=axis)
        if isinstance(assignment, PlaceAssignment):
            readings.append((axis, assignment))
    return tuple(readings)


def undirected_place_name(state: ResolveState) -> str | None:
    """The request's text for the one place set it names without a direction, `None` when
    it names none or several."""
    named = part_sets(state)
    directed = {
        place_key(state.place_sets[name])
        for name in state.entities
        if name in state.place_sets and state.entity_roles.get(name)
    }
    undirected = [key for key in named if key not in directed]
    return set_text(state, named[undirected[0]]) if len(undirected) == 1 else None


def subject_shapes_refused(state: ResolveState) -> bool:
    """Whether every subject-only shape refuses ``role_incompatible`` for this query.

    True when a role-bearing eligible shape outranks the subject-only ones
    (`ResolveState.role_bearing_outranks`) and the query's resolved places name both flow
    ends. A subject-only shape has one subject slot, so it cannot hold an origin and a
    destination at once. `bind_place_roles` refuses subject-only shapes on this predicate,
    and admissibility refuses them before bind (`admissibility.judge_measure_and_roles`).
    """
    if not state.role_bearing_outranks:
        return False
    query_directions = {
        role.axis
        for name in state.entities
        if name in state.place_sets
        for role in state.entity_roles.get(name) or ()
    }
    return {"origin", "destination"} <= query_directions | named_whole_axes(state)


def query_role_axes(state: ResolveState) -> frozenset[RoleAxis]:
    """The flow axes the query needs a role-bearing shape to declare a role on: each axis
    a resolved place is directed to ("to Kenya" needs the destination), the side of each
    whole population a text names ("to all recipients" needs the destination,
    `named_whole_axes`), and each flow axis an operation runs across ("top recipients" needs
    the destination). An axis the query asks every member of (`state.open_axes`) is a
    preference among the shapes that hold these (`without_open_axis_gaps`)."""
    directed = {
        role.axis
        for name in state.entities
        if name in state.place_sets
        for role in state.entity_roles.get(name) or ()
    } | named_whole_axes(state)
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
    """Whether `shape` cannot hold an axis the query needs (`query_role_axes`): a
    role-bearing shape that declares no role on it (`_declares_axes`), such as a
    provider-only shape for "aid to Kenya" or "top recipients of US aid", or a one-way
    subject-only series whose declared side (`series_declarations.declared_side`) is not
    among the query's axes, such as the series of climate finance provided for "climate
    finance to Bangladesh". A ratio whose datasets declare the axis whose party's own total
    it divides by (`bound_values.ratio_base_axis`) lacks the query's axes when that axis is
    not among them, as DAC2a's ODA as a share of the recipient's GNI does for "ODA as a
    share of GNI from Norway", which asks for Norway's own. It could not place the query's places, or its
    answer could not run the operation, so admissibility refuses it `role_incompatible`
    (`admissibility.judge_measure_and_roles`). Any other subject-only shape lacks no axis:
    it binds a directed place as its subject."""
    axes = query_role_axes(state)
    if not axes:
        return False
    if is_role_bearing(shape):
        if not _declares_axes(shape, axes):
            return True
    elif (side := declared_side(shape)) is not None and side not in axes:
        return True
    base = ratio_base_axis(shape, get_index())
    return base is not None and base not in axes


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


def place_texts(
    names: Iterable[str], place_sets: Mapping[str, PlaceSet], texts: Mapping[str, str]
) -> dict[str, str]:
    """The request's text for each place a set named in `names` holds, at its first
    spelling (`first_spellings`): the text of a set's name for its node (`texts`,
    `ResolveState.entity_texts`), and the resolver's name for a group's member. A place
    has one text whichever Spec binds it."""
    return {
        dcid: (
            place_sets[name].member_names.get(dcid, dcid)
            if place_sets[name].kind == "group"
            else texts[name]
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
    name is the first of its names that is not a pinned dcid's key, else the first, and
    `set_text` gives the request's words for it.
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
        sets[key] = dataclasses.replace(
            whole,
            name=next((s.name for s in named if _pin_label(state, s.name) is None), named[0].name),
            reading="members" if asked else whole.reading,
        )
    return sets


def set_text(state: ResolveState, place_set: PlaceSet) -> str:
    """The request's words for `place_set`. A set of `part_sets` reads as the text its name
    names a place of (`ResolveState.entity_texts`), or, when every name of it is a pinned
    dcid's key, as the graph's label for the node, since a pin's key names the node for the
    caller and no more. A party's or whole total's set, which no name of the part keys
    (`ResolveState.party_sets`, `ResolveState.whole_sets`), is named by the code index's
    label for it, which is words."""
    if place_set.name not in state.place_sets:
        return place_set.name
    if (pinned := _pin_label(state, place_set.name)) is not None:
        return place_set.label or pinned
    return state.entity_texts[place_set.name]


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


def _node_kind(state: ResolveState, place_set: PlaceSet) -> Kind | None:
    """The kind of the entity resolved to `place_set`'s node, found by dcid since a pinned
    place's set is named by its graph label and its resolution by the caller's text
    (`part_sets`), else the kind of the node's own graph type (`entity_kind`)."""
    node = place_set.node
    if node is None:
        return None
    resolutions = state.rcl.resolutions.values() if state.rcl is not None else ()
    return next(
        (
            resolution.entity.kind
            for resolution in resolutions
            if isinstance(resolution, Resolved) and resolution.entity.dcid == node.dcid
        ),
        entity_kind((node.graph_type,)) if node.graph_type else None,
    )


def place_set_kinds(
    state: ResolveState,
    place_set: PlaceSet,
    choice: SetChoice | None,
    roles: Sequence[EntityRoleKey],
    *,
    admits: Callable[[EntityRoleKey, str], bool],
) -> frozenset[Kind | None]:
    """The entity kinds (`entity_resolution.types.Entity.kind`) of every reading
    `place_set` can settle to on `roles` (`choice`, from `set_choices`), `None` for one
    nothing states.

    Members, and the total a dataset publishes for them (`_wholes`), have the kind of the
    graph type the members share (`PlaceSet.member_type`, read as the resolver reads a
    node's types). A node has its own kind (`_node_kind`). A group that can settle whole at
    a node the roles admit has both kinds, since the Spec's read decides which it binds: the
    EU's members are countries and its node an organisation. A set with no reading has no
    kind."""
    members = entity_kind((place_set.member_type,)) if place_set.member_type else None
    if choice is None:
        return frozenset()
    if choice == "members":
        return frozenset({members})
    if choice == "node":
        return frozenset({_node_kind(state, place_set)})
    node = place_set.node
    if node is not None and all(admits(role, node.graph_type) for role in roles):
        return frozenset({members, _node_kind(state, place_set)})
    return frozenset({members})


def parties_of_kind(state: ResolveState, shape: ShapeDraft, where: Mapping[RoleAxis, str]) -> bool:
    """Whether the places the query names, as `assign_named_places` assigns them to `shape`,
    fill every axis of `where` with sets whose every reading is of the named kind
    (`place_set_kinds`). An axis the part leaves open, one no set fills, and a part
    `assign_named_places` refuses do not satisfy it. A role admission the request has not
    asked (`ResolveState.node_admissions`) counts as admitted, so a reading it might allow is
    judged too."""
    assignment = assign_named_places(state, shape)
    return not isinstance(assignment, str) and _fills_with_kind(state, shape, assignment, where)


def _fills_with_kind(
    state: ResolveState,
    shape: ShapeDraft,
    assignment: PlaceAssignment,
    where: Mapping[RoleAxis, str],
) -> bool:
    """Whether `assignment` fills every axis of `where` with sets of the named kind, as
    `parties_of_kind` reads it. An empty `where` holds."""
    names = role_names_by_axis(shape)

    def admits(key: EntityRoleKey, graph_type: str) -> bool:
        role = "observationAbout" if key == "observationAbout" else names[key]
        return state.node_admissions.get((role, graph_type), True)

    choices = set_choices(assignment, admits=admits)
    return all(
        bool(sets := assignment.sets.get(axis))
        and all(
            place_set_kinds(
                state,
                place_set,
                choices[place_key(place_set)],
                assignment.roles_of(place_set),
                admits=admits,
            )
            == {kind}
            for place_set in sets
        )
        for axis, kind in where.items()
    )


def stated_party_words(state: ResolveState) -> tuple[PartyWord, ...]:
    """The words declared datasets name a party for (`DatasetDeclaration.party_words`) that
    the part states (`bind_repair.word_stated`, read in its variable and typed qualifier
    phrases) on a flow axis the request leaves unnamed: no name of the part is directed to
    it, no operation runs across it, and the part does not ask every member of it
    (`ResolveState.open_axes`), while a name is directed to each axis of the declaration's
    `where`. Each declaration once, in index order.

    Recall searches the part's text without these words as it does without a stated word
    (`interpret.recall`), and intake reads the place set of each one's party
    (`ResolveState.party_sets`). "Multilateral ODA to Kenya" states "multilateral" on the
    unnamed provider axis; "multilateral ODA from France" and "multilateral ODA to Kenya by
    donor" name or open it, and "multilateral ODA" names no recipient for the word's `where`
    to read. A part read without party words (`ResolveState.party_words_off`) states
    none."""
    if state.party_words_off:
        return ()
    index = get_index()
    directed = {role.axis for name in state.entities for role in state.entity_roles.get(name) or ()}
    taken = directed | {operation.over for operation in state.operations} | state.open_axes
    declared: list[PartyWord] = []
    for dataset in index.datasets.values():
        declared.extend(
            named
            for named in dataset.party_words
            if named.axis not in taken
            and directed.issuperset(named.where)
            and named not in declared
        )
    return tuple(named for named in declared if part_states(state, named.word))


def part_states(state: ResolveState, word: str) -> bool:
    """Whether the part states the casefolded `word`, one a declared dataset states of its
    rows or names a party for, as a modifier of its measure (`bind_repair.word_stated`),
    read in its variable and typed qualifier phrases with the declared words the verbatim
    query negates (`bind_repair.stated_words_negated`), which key the response cache."""
    texts = (state.variable, *(coded.phrase for coded in state.qualifiers or ()))
    query_text = state.query_text
    negated = stated_words_negated(query_text, index=get_index()) if query_text else ()
    return word_stated(word, texts, negated=frozenset(negated))


def held_party_words(state: ResolveState, shape: ShapeDraft) -> tuple[PartyWord, ...]:
    """The words of the part that name a party `shape` binds (`bind_place_roles`), as the
    places the query names assign to it (`assign_named_places`). Empty when they refuse, and
    with no assignment when the part states no such word."""
    if not state.party_sets:
        return ()
    assignment = assign_named_places(state, shape)
    if isinstance(assignment, str):
        return ()
    return _held_party_words(state, shape, assignment)


def _held_party_words(
    state: ResolveState, shape: ShapeDraft, assignment: PlaceAssignment
) -> tuple[PartyWord, ...]:
    """The words of the part (`stated_party_words`) that name a party on `shape` beside the
    named places `assignment` gives it.

    A word names its party on a role-bearing shape when every dataset the shape answers from
    (`bound_values.declared_datasets`) declares it for the shape's measured property, the
    named places fill the axes of its `where` with parties of the kind it names
    (`_fills_with_kind`), intake read the party's place set (`ResolveState.party_sets`), and
    the shape declares the word's axis and no place fills it there, an undirected one
    included. The part states the word only on an axis the request leaves unnamed
    (`stated_party_words`). "Multilateral ODA to Kenya" names `DACMultilaterals`, the DAC's
    total of the multilateral organisations, as DAC2a's provider, as "ODA from multilaterals
    to Kenya" does, and names no party on a CRS shape, whose datasets declare none.

    The word must also stand apart from any word the shape does not read
    (`_beside_words_read`): "multilateral development bank ODA to Kenya" names a kind of
    lender no table declares, so the word names no party there."""
    if not state.party_sets or not is_role_bearing(shape):
        return ()
    datasets = declared_datasets(shape, get_index())
    if not datasets:
        return ()
    unnamed = role_names_by_axis(shape).keys() - assignment.sets.keys()
    stated = stated_party_words(state)

    def holds(named: PartyWord) -> bool:
        return (
            named in stated
            and named.party in state.party_sets
            and named.axis in unnamed
            and named.measured_property == shape.meas_prop_dcid
            and all(named in dataset.party_words for dataset in datasets[1:])
            and _fills_with_kind(state, shape, assignment, named.where)
            and _beside_words_read(state.variable, shape, named.word)
        )

    return tuple(named for named in datasets[0].party_words if holds(named))


def _beside_words_read(variable: str, shape: ShapeDraft, word: str) -> bool:
    """Whether `shape` reads every word of `variable` next to an occurrence of the
    casefolded `word`, with no separator between them. A phrase of the shape's own code
    systems or a measure value (`qualifier_match.read_by_vocabulary`) reads the words it
    spans. A function word, a connective, an amount or data noun, a flow verb
    (`function_words`), a word of the plain name of a code system the shape declares, and a
    word that system's default reads (`DefaultDeclaration.when_named`) are read anywhere.

    "Multilateral" stands beside "ODA", a DAC2a aid type, in "net multilateral ODA" and
    beside "aid", DAC2a's "aid type", in "multilateral aid". In "multilateral development
    bank ODA" and "multilateral education aid" it stands beside a word a DAC2a shape does
    not read."""
    index = get_index()
    # `state.shape_vocabulary`, which this module cannot import.
    vocabulary = (
        frozenset(slot.property_dcid for slot in shape.slot_keys if slot.property_dcid)
        | shape.omitted_slots
    )
    (reading,) = read_by_vocabulary(
        [variable], index, get_measure_index(), vocabularies=[vocabulary]
    )
    spans = reading.by_vocabulary[vocabulary].spans
    declarations = [index.declarations[prop] for prop in vocabulary if prop in index.declarations]
    known = set(FUNCTION_WORDS | CONNECTIVES | AMOUNT_WORDS | DATA_WORDS | FLOW_VERBS)
    for declaration in declarations:
        known |= {token.text.casefold() for token in word_tokens(declaration.plain_name)}
        if declaration.default is not None:
            known |= {named.casefold() for named in declaration.default.when_named}

    def read(token: Token) -> bool:
        return token.text.casefold() in known or any(
            span.start <= token.start and token.end <= span.end for span in spans
        )

    tokens = tokenize(variable)
    return all(
        read(tokens[beside])
        for position, token in enumerate(tokens)
        if token.kind == "word" and token.text.casefold() == word
        for beside in (position - 1, position + 1)
        if 0 <= beside < len(tokens) and tokens[beside].kind == "word"
    )


def bind_place_roles(
    state: ResolveState, shape: ShapeDraft, *, direction: RoleAxis | None = None
) -> PlaceAssignment | PlaceRoleRefusal:
    """Assign the query's place sets to `shape`'s roles (`assign_named_places`), or name the
    refusal, with each whole population a text names on its side (`_named_wholes`), and on
    each flow axis they leave unnamed the party a word of the part names there
    (`_held_party_words`) as the axis's one set: "multilateral ODA to Kenya" fills the
    provider axis of a DAC2a shape with its multilaterals total. The axes keep the order
    `assign_named_places` gives them, destination first. `direction` is
    `assign_named_places`'s.

    The assignment binds no total on a side the part leaves unnamed. Binding repair offers
    one there (`whole_total_candidates`) and keeps it only where the bindings it settles
    under hold its series (`holds_whole_total`), so every reader before repair, and each
    alternate before its walk, reads the named places alone."""
    assignment = assign_named_places(state, shape, direction=direction)
    if isinstance(assignment, str):
        return assignment
    assignment = _named_wholes(state, shape, assignment)
    # A side a listed population opens was named in the request's words.
    listed = frozenset(state.whole_listed.values()) - assignment.sets.keys()
    if listed:
        assignment = dataclasses.replace(
            assignment, unnamed_wholes=assignment.unnamed_wholes | listed
        )
    parties: dict[EntityRoleKey, tuple[PlaceSet, ...]] = {
        named.axis: (state.party_sets[named.party],)
        for named in _held_party_words(state, shape, assignment)
    }
    if not parties:
        return assignment
    return _in_axis_order(dataclasses.replace(assignment, sets={**assignment.sets, **parties}))


def _in_axis_order(assignment: PlaceAssignment) -> PlaceAssignment:
    """`assignment` with its sets destination first, the order `assign_named_places` gives."""
    sets = assignment.sets
    ordered = {axis: sets[axis] for axis in ("destination", "origin") if axis in sets}
    return dataclasses.replace(assignment, sets=ordered)


def _named_wholes(
    state: ResolveState, shape: ShapeDraft, assignment: PlaceAssignment
) -> PlaceAssignment:
    """`assignment` with the sets of each whole population a text names
    (`ResolveState.whole_named`) on its side, after any place named there, on a role-bearing
    `shape` that declares that side. Such a text names its parties as any named party does:
    "official donors to Ethiopia" binds "Official Donors, Total", and "France and official
    donors to Ethiopia" binds France beside it. Where no series holds the total, the answer
    departs `entity_unavailable` for it or refuses, as for any party. A text the request's
    words direct to the side other than its population's is refused before any shape binds
    (`phase_intake._phase_check_entities`).

    A side that holds one population alone reads it whole where every dataset the shape
    answers from (`bound_values.declared_datasets`) marks it whole over the part's period
    (`whole_totals.whole_reading`), and the assignment carries that reading
    (`PlaceAssignment.whole`). It is a label: the side binds the same parties either way. A
    side no place is named on stays unnamed where, for every population named on it, no
    dataset the shape answers from publishes its total and every party they hold there is
    of a class it covers (`whole_totals.covers_unpublished`), as the CRS's recipients do for
    "all recipients": its sum over the side is every party the population holds. A side
    where one population covers and another does not ("official donors and all donors" on
    the CRS) binds both as named parties. The assignment carries the sides left unnamed
    (`PlaceAssignment.unnamed_wholes`), where each population states itself
    (`whole_totals.whole_total_unpublished_facts`) in place of a `role_scope_defaulted`
    warning that the request named nothing there."""
    if not state.whole_named or not is_role_bearing(shape):
        return assignment
    declared = role_names_by_axis(shape)
    datasets = [dataset.provenance for dataset in declared_datasets(shape, get_index())]
    parties = get_index().parties
    covered: dict[RoleAxis, bool] = {}
    for dcids in state.whole_named.values():
        axis = parties[dcids[0]].axes[0]
        if axis in declared and axis not in assignment.sets:
            covered[axis] = covered.get(axis, True) and covers_unpublished(dcids, datasets=datasets)
    unnamed = frozenset(axis for axis, covers in covered.items() if covers)
    sets = dict(assignment.sets)
    named: dict[RoleAxis, list[tuple[str, ...]]] = {}
    for dcids in dict.fromkeys(state.whole_named.values()):
        axis = parties[dcids[0]].axes[0]
        if axis not in declared or axis in unnamed:
            continue
        named.setdefault(axis, []).append(dcids)
        sets[axis] = (*sets.get(axis, ()), *(state.whole_sets[dcid] for dcid in dcids))
    if not named:
        return dataclasses.replace(assignment, unnamed_wholes=unnamed)
    years = requested_years(state.date_request)
    whole = next(
        (
            reading
            for axis, populations in named.items()
            if axis not in assignment.sets and len(populations) == 1
            if (reading := whole_reading(populations[0], datasets=datasets, years=years))
            is not None
        ),
        None,
    )
    return _in_axis_order(
        dataclasses.replace(
            assignment,
            sets=sets,
            open_axes=assignment.open_axes - named.keys(),
            whole=whole,
            unnamed_wholes=unnamed,
        )
    )


def totalled_axes(state: ResolveState) -> tuple[RoleAxis, ...]:
    """The flow axes the part totals, origin first, read from the part alone: no name of
    the part is directed to the axis, no operation runs over it, and the part does not ask
    every member of it (`ResolveState.open_axes`). An operation over an axis needs every
    member's value (`Operation.over`): it may filter, rank or count them, so the side's
    total cannot stand in for them, an `aggregate` included. "ODA to Ethiopia" and a sum
    over no axis total the donors; "ODA to Ethiopia by donor", "top donors to Ethiopia",
    a sum over the donors and a share or growth over them list them."""
    directed = {role.axis for name in state.entities for role in state.entity_roles.get(name) or ()}
    over = {operation.over for operation in state.operations}
    return tuple(
        axis
        for axis in ("origin", "destination")
        if axis not in directed and axis not in over and axis not in state.open_axes
    )


def named_whole_axes(state: ResolveState) -> frozenset[RoleAxis]:
    """The side of each whole population a text of the part names (`ResolveState.whole_named`),
    the one axis the code index publishes its parties on."""
    parties = get_index().parties
    return frozenset(parties[dcids[0]].axes[0] for dcids in state.whole_named.values())


def whole_total_candidates(
    state: ResolveState, shape: ShapeDraft, assignment: PlaceAssignment
) -> tuple[PlaceAssignment, ...]:
    """`assignment` with each whole total role-bearing `shape` offers on a side it leaves
    unnamed, in the order binding repair tries them: its parties' sets fill that side, which
    leaves `open_axes`, and its reading is `PlaceAssignment.whole`. Empty when the shape
    offers none. Each side offers at most one total, so there are at most two candidates.

    The candidates read no binding and no coverage, so the role keys each gives are fixed
    before any binding is settled under them. A side offers a total where every dataset the
    shape answers from (`bound_values.declared_datasets`) reads its parties whole over the
    part's period (`whole_totals.whole_reading`), the shape declares a role on that side,
    and `assignment` leaves it unnamed. Each side the part totals (`totalled_axes`) offers
    the total of every party there, origin first: both sides unnamed offer the donors'
    total, then the recipients' total, one at a time, since the source publishes no figure
    between two whole totals. For the same reason a side a text names a whole population on
    ("official donors") offers none, and once that population binds there (`_named_wholes`)
    neither side does. Whether a candidate binds is `holds_whole_total`'s question, asked
    under the bindings settled for its role keys; the first that holds binds."""
    named_axes = named_whole_axes(state)
    if not state.whole_sets or not is_role_bearing(shape) or named_axes & assignment.sets.keys():
        return ()
    datasets = [dataset.provenance for dataset in declared_datasets(shape, get_index())]
    declared = role_names_by_axis(shape)
    years = requested_years(state.date_request)

    def offered(dcids: Sequence[str]) -> WholeReading | None:
        reading = whole_reading(dcids, datasets=datasets, years=years)
        if reading is None or reading.axis not in declared or reading.axis in assignment.sets:
            return None
        return reading

    wholes: dict[RoleAxis, WholeReading] = {}
    for axis in totalled_axes(state):
        if axis in named_axes:
            continue
        for dcid in state.whole_sets:
            if (whole := offered((dcid,))) is not None and whole.axis == axis:
                wholes[axis] = whole
                break
    return tuple(_with_whole(state, assignment, whole) for whole in wholes.values())


def _with_whole(
    state: ResolveState, assignment: PlaceAssignment, whole: WholeReading
) -> PlaceAssignment:
    """`assignment` with the sets of `whole`'s parties on its side, which leaves
    `open_axes`, and `whole` as its reading."""
    sets = tuple(state.whole_sets[dcid] for dcid in whole.members)
    return _in_axis_order(
        dataclasses.replace(
            assignment,
            sets={**assignment.sets, whole.axis: sets},
            open_axes=assignment.open_axes - {whole.axis},
            whole=whole,
        )
    )


def holds_whole_total(
    state: ResolveState,
    shape: ShapeDraft,
    candidate: PlaceAssignment,
    bindings: Sequence[SlotBindingDraft],
) -> bool:
    """Whether the request's coverage snapshot holds a series of `shape`'s family under
    `bindings` at the whole total `candidate` binds beside its other places
    (`coverage_series.holds_series`), the bindings settled under `candidate`'s role keys. A
    total the source publishes no figure of there for the variables the bindings elect, or
    one the snapshot cannot tell of, leaves its side to its parties."""
    return bool(holds_series(state, shape, candidate.sets, bindings=bindings))


def assign_named_places(
    state: ResolveState, shape: ShapeDraft, *, direction: RoleAxis | None = None
) -> PlaceAssignment | PlaceRoleRefusal:
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

    `direction` places a set named without a direction that no rule places on that axis,
    one reading of the direction the question leaves unstated (`direction_assignments`).
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
            undirected_axis = (
                _undirected_place_axis(state, shape, named[undirected[0]]) or direction
            )
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
    inferred: set[tuple[str, RoleAxis]] = set()
    for name in dict.fromkeys(state.entities):
        if name not in place_sets or not state.entity_roles.get(name):
            continue
        # Use the first axis when a name occurs at both flow ends. A text the door named
        # apart reads once per axis, in its own words.
        first = state.entity_roles[name][0]
        text = state.entity_texts[name]
        discarded.append((text, first.axis, place_key(place_sets[name])))
        if not first.stated:
            inferred.add((text, first.axis))
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
            whole=assignment.whole,
            unnamed_wholes=assignment.unnamed_wholes,
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
    """The request texts whose flow direction a `role_not_resolved` refusal could not tell,
    in query order: every text of each untold group (`untold_direction_groups`). Empty when
    every name's direction is told, as when materialisation refuses the roles."""
    untold = {name for texts in untold_direction_groups(state) for name in texts}
    return tuple(name for name in dict.fromkeys(state.entities) if name in untold)


def untold_direction_groups(state: ResolveState) -> tuple[tuple[str, ...], ...]:
    """The request texts of each place a `role_not_resolved` refusal could not tell the
    flow direction of, one tuple per untold group (`_untold_groups`), each in query order,
    so a refusal names one place set read under several texts as one place."""
    return tuple(tuple(texts) for texts in _untold_groups(state, _text_groups(state)).values())


def _text_groups(state: ResolveState) -> dict[str, list[str]]:
    """The request texts in groups, by group key, in query order: the texts that name one
    place set under its place key, and each text that resolved to no set alone under
    the text."""
    groups: dict[str, list[str]] = {}
    for name in dict.fromkeys(state.entities):
        place_set = state.place_sets.get(name)
        groups.setdefault(name if place_set is None else place_key(place_set), []).append(name)
    return groups


def _untold_groups(state: ResolveState, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """The `groups` whose flow direction the question leaves untold, in their order: those
    no text of which carries a direction, a text that resolved to no set and names a whole
    population (`ResolveState.whole_named`, `ResolveState.whole_listed`) aside, else those
    one of whose texts states a direction the query's own wording contradicts
    (`role_text.contradicts`)."""
    undirected = {
        key: texts
        for key, texts in groups.items()
        if not any(state.entity_roles.get(name) for name in texts) and not _whole_text(state, texts)
    }
    if undirected:
        return undirected
    return {key: texts for key, texts in groups.items() if _contradicted_names(state, texts)}


def _whole_text(state: ResolveState, texts: list[str]) -> bool:
    """Whether `texts` is one text that resolved to no place set and names a whole
    population, which binds on its own side and carries no direction to tell."""
    return (
        len(texts) == 1
        and texts[0] not in state.place_sets
        and (texts[0] in state.whole_named or texts[0] in state.whole_listed)
    )


# The most readings of an unstated direction a `role_not_resolved` refusal offers. Two sets
# on a shape with a role on each axis give 2, three give 6, four 14. Past this the refusal
# offers none and its message stands.
MAX_DIRECTION_READINGS = 6


def direction_readings(state: ResolveState) -> tuple[dict[str, RoleAxis], ...]:
    """The readings a `role_not_resolved` refusal offers for the directions the question
    leaves untold, each mapping every request text of every untold set to one flow axis.

    The untold sets are the place sets among the untold groups (`_untold_groups`): those no
    text of which carries a direction, else those whose stated direction the query's
    wording contradicts. Each takes one axis on which the elected shape declares a role and
    that the part does not ask open (`ResolveState.open_axes`, or an operation's `over`); an
    open axis counts as named and takes no set. A reading is kept when every such axis then
    holds a set, a directed one or a whole population a text names (`named_whole_axes`)
    included. Readings run over the sets in query order, origin first. Empty when the shape
    declares no role, no set is untold, or more than `MAX_DIRECTION_READINGS` readings are
    kept. The count stops there, so no reading past it is built."""
    shape = state.shape_draft
    if shape is None or not is_role_bearing(shape):
        return ()
    groups = _text_groups(state)
    texts = {key: names for key, names in groups.items() if names[0] in state.place_sets}
    untold = [key for key in _untold_groups(state, groups) if key in texts]
    if not untold:
        return ()
    directed = {
        key: {role.axis for name in names for role in state.entity_roles.get(name) or ()}
        for key, names in texts.items()
    }
    over = {operation.over for operation in state.operations}
    declared = role_names_by_axis(shape)
    axes: tuple[RoleAxis, ...] = tuple(
        axis
        for axis in ("origin", "destination")
        if axis in declared and axis not in state.open_axes and axis not in over
    )
    held = {
        axis for key, told in directed.items() if key not in untold for axis in told
    } | named_whole_axes(state)
    needed = set(axes) - held
    readings: list[dict[str, RoleAxis]] = []
    for choice in itertools.product(axes, repeat=len(untold)):
        if not needed <= set(choice):
            continue
        if len(readings) == MAX_DIRECTION_READINGS:
            return ()
        readings.append(
            {text: axis for key, axis in zip(untold, choice, strict=True) for text in texts[key]}
        )
    return tuple(readings)


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
    group, continent and subregion that holds a member, then the whole population the
    binding reads on one side (`PlaceRoles.whole`, `WholeReading.named_group`).

    `build_spec` names a role by one of these when the role binds exactly its members or its
    total (`PlaceSet.total`). A group with its own graph node takes that node's type and the
    graph's label for it. One without, or one the graph gives no label, keeps the query's
    words (`set_text`), and one without names its registry id (`PlaceSet.group_id`). Each states the
    members the query also named as places of their own, on the role `roles`, one Spec's
    binding, puts each of those places on (`PlaceRoles.named_alone`).
    """
    return tuple(
        NamedGroup(
            label=set_text(state, place_set) if place_set.label is None else place_set.label,
            dcid=None if place_set.node is None else place_set.node.dcid,
            entity_type=None if place_set.node is None else place_set.node.graph_type,
            member_dcids=frozenset(place_set.members),
            named_alone=frozenset(
                (role, dcid) for role, dcid in roles.named_alone if dcid in place_set.members
            ),
            total=place_set.total,
            group_id=place_set.group_id if place_set.node is None else None,
        )
        for place_set in part_sets(state).values()
        if place_set.members
    ) + (() if roles.whole is None else (roles.whole.named_group(),))


def group_request_names(state: ResolveState) -> dict[str, str]:
    """The query's name for each group `request_named_groups` labels otherwise, keyed by that
    label ("EU" for "European Union"), so a departure asks for the group in the request's
    words (`RegionResult.entity_words`). A label the request posts as a text of its own,
    and that of a set named only by pins (`set_text`), are left out, so no text of the
    request is read as another of its texts. A whole population
    the query names ("all donors", `ResolveState.whole_named`) is keyed by the label the side
    reading it carries (`whole_totals.named_whole_label`)."""
    groups = {
        place_set.label: place_set.name
        for place_set in part_sets(state).values()
        if place_set.members
        and place_set.label is not None
        and place_set.label != set_text(state, place_set)
        and place_set.label not in state.entity_texts.values()
    }
    wholes = {
        label: text
        for text, dcids in state.whole_named.items()
        if (label := named_whole_label(dcids)) is not None
    }
    return {**groups, **wholes}


def single_node_dcids(state: ResolveState) -> dict[str, str]:
    """The node each entity name of the part resolved to alone, in query order: a set with
    a node that is neither a group nor read as its members. A group, a set read as its
    members, and an ambiguous or unresolved name are left out, since re-resolving such a
    text keeps its members or its candidates."""
    return {
        name: place_set.node.dcid
        for name in dict.fromkeys(state.entities)
        if (place_set := state.place_sets.get(name)) is not None
        and place_set.kind != "group"
        and place_set.reading != "members"
        and place_set.node is not None
    }


def bound_entity_refs(state: ResolveState) -> dict[str, tuple[str, ...]]:
    """The places the part's settled binding (`ResolveState.place`) holds for each entity
    text, in `place_order` (`RegionResult.entity_refs`).

    A text's set binds one reading (`settle_place_sets`): its node when the binding holds
    it, else its total (`PlaceSet.total`), else the members the binding holds. A set bound
    as its members states its own ref first, so a client matches the set it named: its
    node, else its total, else its registry id (`PlaceSet.group_id`). A text naming a whole
    population ("all donors", `ResolveState.whole_named`) binds the parties it names that
    the binding holds. A text with neither, ambiguous or unresolved, is left out.

    A part refused before a shape's places are installed (`ResolveState.places_bound`),
    whether before election or after it, states the node each text names alone
    (`single_node_dcids`).
    """
    if not state.places_bound:
        return {text: (dcid,) for text, dcid in single_node_dcids(state).items()}
    bound = {dcid for dcid, _role in state.place.roles.roles}
    refs: dict[str, tuple[str, ...]] = {}
    for text, place_set in state.place_sets.items():
        node = None if place_set.node is None else place_set.node.dcid
        whole = next((d for d in (node, place_set.total) if d is not None and d in bound), None)
        if whole is not None:
            refs[text] = (whole,)
        elif held := place_order(d for d in place_set.members or () if d in bound):
            own = next((d for d in (node, place_set.total, place_set.group_id) if d), None)
            refs[text] = tuple(held) if own is None else (own, *held)
    for text, dcids in state.whole_named.items():
        if held := place_order(d for d in dcids if d in bound):
            refs[text] = tuple(held)
    return refs


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
