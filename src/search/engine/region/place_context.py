"""One shape's place context: how it binds the places, where it is probed and read, and the
place warnings it carries.

`UnsettledPlace` holds a shape's places before its place sets settle, and `settle_sets`
reads the Spec once per reading it has to compare and returns the context it settles to.
`place_context` builds the context from a shape's bound places and makes no graph request.
`PlaceContext.settled` gives the context an answer is read over once materialisation has
run, and `settled_shape` the shape it is read under. `default_scope_stat_vars` chooses the
StatVars a default-scope presence read covers.
`stamp_offered_place_warnings` stamps each offered spec with the place warnings of its own
context, and `install_place_warnings` makes one context's the answer's own.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from materialise.place_scope import DefaultPlaceScope
from materialise.roles import axis_noun
from search.engine.config import settings
from search.engine.departures import WarningKey, direction_discarded, warning_key
from search.engine.errors import DefaultScopeIndexSourcedError
from search.engine.place import ConsultedPresence
from search.engine.region.codes import DIRECTION_DISCARDED, PLACE_SCOPE_DEFAULTED
from search.engine.region.convertibility import survivor_declared_roles
from search.engine.region.decisions import place_scope_defaulted_warning
from search.engine.region.place_roles import (
    PlaceAssignment,
    PlaceLists,
    PlaceRoles,
    place_order,
    settle_place_sets,
    settled_roles,
    undecided_sets,
    where_binding,
)
from search.engine.region.place_set import PlaceNode
from search.engine.retrieval.bound_values import shape_member_facts, shape_query
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.standard import FacetBatch
from search.engine.shape import ShapeDraft, is_role_bearing, role_names_by_axis
from search.heading.departure_fact import DepartureFact
from search.models import EntityRoleKey, Warning

if TYPE_CHECKING:
    from code_index.query import CodeIndex
    from search.engine.bind import SlotBindingDraft

# Where a context's probe places come from: the shape's bound places, the members a
# role-bearing shape's materialisation discovered, or a default scope, which is every place
# with data (`"every_place"`) or the countries fallback (`DefaultPlaceScope.every_place`).
PlaceMode = Literal["bound", "discovered", "every_place", "countries_fallback"]

# The codes of the place warnings a context carries (`PlaceContext.warnings`).
PLACE_WARNING_CODES = frozenset({DIRECTION_DISCARDED, PLACE_SCOPE_DEFAULTED})


@dataclass(frozen=True)
class PlaceContext:
    """One shape's places and the scope it is probed and read over."""

    # The shape's binding of the places: flow axes, role entities, wire roles, where places,
    # the observation fallback place and dropped directions.
    roles: PlaceRoles
    # Where the shape is probed and read: its where places when it binds several, a
    # discovered scope's members, or a default scope's places. `None` probes its one bound
    # place, or no place on a role-bearing shape with no destination.
    probe_dcids: tuple[str, ...] | None
    mode: PlaceMode
    # `PLACE_SCOPE_DEFAULTED` on a default scope, else `None`.
    scope_warning: Warning | None

    @property
    def where(self) -> SlotBindingDraft | None:
        """The where-slot binding of the shape's bound places."""
        return where_binding(self.roles.where_dcids)

    @property
    def probe_places(self) -> list[str] | None:
        """`probe_dcids` as the list materialisation and the observation helpers take."""
        return None if self.probe_dcids is None else list(self.probe_dcids)

    @property
    def default_scope(self) -> DefaultPlaceScope | None:
        """The default scope the shape is probed over, or `None` when its places are bound or
        discovered."""
        if self.mode == "bound" or self.mode == "discovered":
            return None
        return DefaultPlaceScope(
            places=self.probe_dcids or (), every_place=self.mode == "every_place"
        )

    @property
    def warnings(self) -> tuple[Warning, ...]:
        """The context's place warnings: `DIRECTION_DISCARDED` per dropped direction in place
        order, then the scope warning."""
        directions = direction_discarded_warnings(self.roles)
        return directions if self.scope_warning is None else (*directions, self.scope_warning)

    @property
    def departure_facts(self) -> dict[WarningKey, DepartureFact]:
        """The `direction_discarded` departure of each dropped direction the request's words
        stated, under the key of its `DIRECTION_DISCARDED` warning. A direction the
        extractor inferred (`PlaceRoles.inferred_directions`) keeps its warning and writes
        no departure, since the request named no role to depart from."""
        words = self.roles.direction_words
        return {
            warning_key(warning): direction_discarded(words.get(name, name), axis)
            for warning, (name, axis) in zip(
                direction_discarded_warnings(self.roles),
                self.roles.discarded_directions,
                strict=True,
            )
            if name not in self.roles.inferred_directions
        }

    def presence_for(self, presence: ConsultedPresence | None) -> ConsultedPresence | None:
        """The presence read materialisation derives this context's default scope from:
        `presence` on a default scope, else `None`. A default scope with no `presence`
        raises `AssertionError`."""
        if self.default_scope is None:
            return None
        if presence is None:
            raise AssertionError("a default-scoped place context needs its presence read")
        return presence

    def settled(self, mat: Materialised) -> PlaceContext:
        """The context the answer is read over.

        The open axes narrow to those the elected members declare (`Materialised.open_axes`),
        so every reader after the election reads the axes the answer serves. A discovered
        role scope's members replace the probe places, so the preview and later probes bind
        every member the open axis holds data for. A default scope takes the scope `mat` was
        settled on (`Materialised.default_scope`): the places of the StatVars the answer
        serves, with the scope warning saying which kind it is. Otherwise the probe places
        and scope stand.
        """
        place = self
        if not self.roles.open_axes <= mat.open_axes:
            place = dataclasses.replace(
                self,
                roles=dataclasses.replace(
                    self.roles, open_axes=self.roles.open_axes & mat.open_axes
                ),
            )
        if mat.discovered_scope is not None:
            return dataclasses.replace(
                place, probe_dcids=tuple(mat.discovered_scope.member_dcids), mode="discovered"
            )
        if mat.default_scope is not None:
            return dataclasses.replace(
                place,
                probe_dcids=mat.default_scope.places,
                mode=_default_scope_mode(mat.default_scope),
                scope_warning=place_scope_defaulted_warning(mat.default_scope),
            )
        return place


def _default_scope_mode(scope: DefaultPlaceScope) -> PlaceMode:
    """The mode of a context probed over `scope`."""
    return "every_place" if scope.every_place else "countries_fallback"


def settled_shape(shape: ShapeDraft, mat: Materialised) -> ShapeDraft:
    """`shape` declaring the roles of the members `mat` elects, the shape the answer is read
    under beside `PlaceContext.settled`.

    A role-bearing shape declares the union of its recalled members' roles, and for a
    family the code index decides the election keeps the recalled shape
    (`bound_values.elected_role_class_shape`), so its roles can lack an axis the elected
    members declare. The settled shape declares the survivors' shared signature
    (`convertibility.survivor_declared_roles`) with no optional role, so the roles the
    answer states (`result.answered_roles`) are the elected members'. `shape` stands when
    it is subject-only, when `mat` names no member or carries no member facts, when the
    survivors declare no flow role, and when it declares those roles already.
    """
    if not is_role_bearing(shape) or not mat.sv_dcids or mat.member_facts is None:
        return shape
    roles = survivor_declared_roles(mat)
    if roles == ("observationAbout",) or (
        frozenset(roles) == frozenset(shape.declared_roles or ()) and not shape.optional_roles
    ):
        return shape
    return dataclasses.replace(shape, declared_roles=roles, optional_roles=frozenset())


def settle_place(place: PlaceContext, mat: Materialised, warnings: list[Warning]) -> PlaceContext:
    """`place.settled(mat)`, replacing in `warnings`, where it stands, a scope warning whose
    message the settling changes."""
    settled = place.settled(mat)
    before, after = place.scope_warning, settled.scope_warning
    if before is not None and after is not None and before != after:
        warnings[:] = [after if w is before else w for w in warnings]
    return settled


def stamp_offered_place_warnings(
    warnings: list[Warning],
    offered: Iterable[tuple[str, PlaceContext]],
    *,
    departure_facts: dict[WarningKey, DepartureFact],
) -> None:
    """Append each offered spec's place warnings that the shape in play's do not hold.

    The place warnings of the shape in play stand unstamped in `warnings`. Each `(spec_id,
    place)` in `offered`, in order, appends every warning of its own context
    (`PlaceContext.warnings`) whose code and message none of those holds, stamped with its
    `spec_id`, so a spec that drops a direction or defaults its scope differently says so.
    Each context's departure facts join `departure_facts`.
    """
    in_play = {
        (w.code, w.message) for w in warnings if w.code in PLACE_WARNING_CODES and w.spec_id is None
    }
    for spec_id, place in offered:
        departure_facts.update(place.departure_facts)
        warnings.extend(
            w.model_copy(update={"spec_id": spec_id})
            for w in place.warnings
            if (w.code, w.message) not in in_play
        )


def install_place_warnings(
    warnings: list[Warning],
    place: PlaceContext,
    *,
    departure_facts: dict[WarningKey, DepartureFact],
) -> None:
    """Make `place`'s warnings the place warnings in `warnings`: every place warning is
    removed and `place.warnings` appended, directions in place order, then the scope
    warning. `place`'s departure facts join `departure_facts`. A fact whose warning was
    removed stays there and reaches no part."""
    warnings[:] = [w for w in warnings if w.code not in PLACE_WARNING_CODES]
    warnings.extend(place.warnings)
    departure_facts.update(place.departure_facts)


def empty_place_context() -> PlaceContext:
    """The context of a resolve state no shape has been installed in: no roles and no probe
    places."""
    return PlaceContext(
        roles=PlaceRoles(
            destination_dcids=[],
            origin_dcids=[],
            open_axes=frozenset(),
            resolved_entities={},
            roles=[],
            where_dcids=[],
            obs_destination_dcid=None,
        ),
        probe_dcids=None,
        mode="bound",
        scope_warning=None,
    )


def takes_default_scope(shape: ShapeDraft, roles: PlaceRoles | PlaceAssignment) -> bool:
    """Whether `shape` is probed over a default scope: it is subject-only and `roles`, its
    binding of the query's places or its assignment of them, names no place."""
    named = roles.where_dcids if isinstance(roles, PlaceRoles) else roles.sets
    return not named and not is_role_bearing(shape)


def direction_discarded_warnings(roles: PlaceRoles) -> tuple[Warning, ...]:
    """One `DIRECTION_DISCARDED` warning per directed place a subject-only shape reads as its
    subject, in place order, each name at its first place (`place_roles.bind_place_roles`)
    and in the request's words (`PlaceRoles.direction_words`)."""
    words = roles.direction_words
    return tuple(
        Warning(
            code=DIRECTION_DISCARDED,
            severity="warn",
            message=(
                f"{words.get(name, name)} was named as the {axis_noun(axis)}; this shape "
                "reports the place as its own subject, so the direction was dropped."
            ),
        )
        for name, axis in roles.discarded_directions
    )


def cannot_take_default_scope(shape: ShapeDraft) -> bool:
    """Whether `shape` is subject-only with a population the code index sources. Such a shape
    is probed over members its `sv_arc_facts` may not hold, which a default-scope presence
    read would not cover, so it takes no default scope. A shape standing for a family the
    catalogue holds (`ShapeDraft.held_family`) does take one: the read covers the members a
    mint serves for it under its bindings (`phase_place_materialise._group_stat_vars`)."""
    # Whether the index decides the population reads no binding and no role key.
    return not is_role_bearing(shape) and shape_member_facts(shape, shape_query(shape, (), ()))[1]


def default_scope_stat_vars(shapes: Iterable[ShapeDraft]) -> list[str]:
    """The StatVars a default-scope presence read covers for `shapes`: those of the
    subject-only ones, deduplicated in order. Role-bearing shapes take no default scope.

    `shapes` holds the shape being answered and the alternates that can take a default
    scope. A shape that cannot (`cannot_take_default_scope`) raises
    `DefaultScopeIndexSourcedError`, so the shape being answered refuses the request before
    any read. The callers leave such an alternate out of `shapes` and out of the offer.
    """
    stat_vars: dict[str, None] = {}
    for shape in shapes:
        if is_role_bearing(shape):
            continue
        if cannot_take_default_scope(shape):
            raise DefaultScopeIndexSourcedError(
                f"subject-only shape {shape.shape_id!r} is index-sourced; a default-scope "
                "presence read covers only its sv_arc_facts"
            )
        stat_vars.update(dict.fromkeys(shape.sv_arc_facts or {}))
    return list(stat_vars)


def place_context(
    shape: ShapeDraft,
    roles: PlaceRoles,
    *,
    stat_vars: Sequence[str],
    presence: ConsultedPresence | None,
) -> PlaceContext:
    """Build `shape`'s place context from its bound places.

    A subject-only shape that binds no place (`takes_default_scope`) is probed over the union
    of the default scopes of `stat_vars` (`ConsultedPresence.consulted_union`, which notes a
    countries fallback this request alone took), the StatVars a
    Datasets mint serves for it (`state.served_stat_vars`), each of which is read at its own
    scope; the context carries a `PLACE_SCOPE_DEFAULTED` warning. It needs `presence`, and
    raises `AssertionError` without it. Materialisation narrows it to the scopes of the
    StatVars the answer serves (`settled`). Any other shape is probed at the where places
    it binds. A role-bearing shape with no bound destination probes no place and takes no
    default scope.
    """
    where = roles.where_dcids
    if takes_default_scope(shape, roles):
        if presence is None:
            raise AssertionError(
                f"shape {shape.shape_id!r} binds no place and no default-scope presence was read"
            )
        scope = presence.consulted_union(stat_vars)
        return PlaceContext(
            roles=roles,
            probe_dcids=scope.places,
            mode=_default_scope_mode(scope),
            scope_warning=place_scope_defaulted_warning(scope),
        )
    probe_dcids = None if len(where) <= 1 else tuple(where)
    return PlaceContext(roles=roles, probe_dcids=probe_dcids, mode="bound", scope_warning=None)


@dataclass(frozen=True)
class UnsettledPlace:
    """One shape's places before its place sets settle: its assignment of the query's sets
    to its roles, and what the context of a settled reading is built from."""

    shape: ShapeDraft
    assignment: PlaceAssignment
    # The code index whose parties say which kinds each flow role takes
    # (`place_roles.set_choices`).
    index: CodeIndex
    # The role name and graph type pairs role admission admits, among
    # `admission_questions`.
    admitted: frozenset[tuple[str, str]] = frozenset()
    # The StatVars and presence read a default scope is built from (`place_context`).
    stat_vars: tuple[str, ...] = ()
    presence: ConsultedPresence | None = None

    def _role_name(self, key: EntityRoleKey) -> str | None:
        if key == "observationAbout":
            return key
        return role_names_by_axis(self.shape).get(key)

    @property
    def admission_questions(self) -> set[tuple[str, str]]:
        """The role name and graph type pairs settling asks role admission about: each role
        a set that holds members fills, with each of the node's types and the members' type
        (`place_roles.set_choices`)."""
        return {
            (role, graph_type)
            for place_set in self.assignment.distinct_sets
            if place_set.members
            for graph_type in (
                *(() if place_set.node is None else place_set.node.graph_types),
                *(() if place_set.member_type is None else (place_set.member_type,)),
            )
            for key in self.assignment.roles_of(place_set)
            if (role := self._role_name(key)) is not None
        }

    def _admits(self, key: EntityRoleKey, graph_type: str) -> bool:
        return (self._role_name(key), graph_type) in self.admitted

    @property
    def decided(self) -> bool:
        """Whether every set has one reading before the Spec is read
        (`place_roles.undecided_sets`)."""
        return not undecided_sets(self.assignment, admits=self._admits, index=self.index)

    def context(self, places: PlaceLists) -> PlaceContext:
        """The context of the shape bound to `places`."""
        return place_context(
            self.shape,
            settled_roles(self.shape, self.assignment, places),
            stat_vars=self.stat_vars,
            presence=self.presence,
        )

    def unread(self) -> PlaceContext:
        """The context before the Spec is read, with each set the read decides at its
        members. When the read decides none (`decided`) this is the settled context, and
        places over the bound on the places a Spec binds raise `PlaceSetTooLargeError`."""
        return self.context(
            settle_place_sets(
                self.assignment,
                admits=self._admits,
                index=self.index,
                has_data=lambda _places: False,
                bound=settings().max_child_places if self.decided else None,
            )
        )

    @property
    def read_places(self) -> tuple[str, ...]:
        """The places one facet read covers for a subject-only shape whose read decides a
        set: the node and the members of each of its sets. Empty when no read decides a
        set, on a role-bearing shape, whose data the coverage index holds, and when the
        members exceed the bound on the places a Spec binds, so that a node holding no data
        refuses the Spec with no read of them."""
        if is_role_bearing(self.shape) or self.decided:
            return ()
        members = self.unread().roles.where_dcids
        if len(members) > settings().max_child_places:
            return ()
        nodes = [
            cast(PlaceNode, place_set.node).dcid
            for place_set in undecided_sets(self.assignment, admits=self._admits, index=self.index)
        ]
        return tuple(place_order([*nodes, *members]))

    def settle[T](
        self, read: Callable[[PlaceContext, FacetBatch | None], T]
    ) -> tuple[PlaceContext, T]:
        """Settle the shape's sets (`place_roles.settle_place_sets`) and return the context
        it settles to with the Spec's read there.

        `read` is the Spec's own read in a context, which holds data when `holds_data` says
        so. Every read shares one `FacetBatch` over `read_places`, so comparing a node
        with its members adds no request. A read that raises propagates, and places over
        the bound on the places a Spec binds raise `PlaceSetTooLargeError`.
        """
        read_places = self.read_places
        batch = FacetBatch(read_places) if read_places else None
        reads: dict[tuple[tuple[str, tuple[str, ...]], ...], tuple[PlaceContext, T]] = {}

        def read_at(places: PlaceLists) -> tuple[PlaceContext, T]:
            key = tuple(sorted(places.items()))
            if key not in reads:
                context = self.context(places)
                reads[key] = (context, read(context, batch))
            return reads[key]

        places = settle_place_sets(
            self.assignment,
            admits=self._admits,
            index=self.index,
            has_data=lambda places: holds_data(read_at(places)[1]),
            bound=settings().max_child_places,
        )
        return read_at(places)


def unread_context(place: PlaceContext | UnsettledPlace) -> PlaceContext:
    """The context `place` has before its Spec is read: `place` itself once settled, else
    `UnsettledPlace.unread`."""
    return place.unread() if isinstance(place, UnsettledPlace) else place


def holds_data(result: object) -> bool:
    """Whether a Spec's read is a materialisation holding data in the years read, the one
    test a set's node passes to bind (`UnsettledPlace.settle`)."""
    return isinstance(result, Materialised) and result.has_data


def settle_sets[T](
    place: PlaceContext | UnsettledPlace, read: Callable[[PlaceContext, FacetBatch | None], T]
) -> tuple[PlaceContext, T]:
    """The context `place` settles to and the Spec's read there (`UnsettledPlace.settle`).
    A context whose sets are settled is read as it stands."""
    if isinstance(place, UnsettledPlace):
        return place.settle(read)
    return place, read(place, None)


def with_where(bindings: Sequence[SlotBindingDraft], place: PlaceContext) -> list[SlotBindingDraft]:
    """`bindings` with the where slot binding `place`'s bound places. A where binding
    without a property gives way to them."""
    kept = [b for b in bindings if not (b.axis == "where" and b.property_dcid is None)]
    return kept if place.where is None else [*kept, place.where]
