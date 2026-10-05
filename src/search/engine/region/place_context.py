"""One shape's place context: how it binds the places, where it is probed and read, and the
place warnings it carries.

`place_context` builds the context from a shape's bound places and makes no graph request.
`PlaceContext.settled` gives the context an answer is read over once materialisation has
run. `default_scope_stat_vars` chooses the StatVars a default-scope presence read covers.
`stamp_offered_place_warnings` stamps each offered spec with the place warnings of its own
context, and `install_place_warnings` makes one context's the answer's own.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from materialise.place_scope import DefaultPlaceScope, DefaultPresence
from materialise.roles import axis_noun
from search.engine.departures import WarningKey, direction_discarded, warning_key
from search.engine.errors import DefaultScopeIndexSourcedError
from search.engine.region.codes import (
    DIRECTION_DISCARDED,
    PLACE_SCOPE_DEFAULTED,
    STANDARD_PROBE_PLACES_CLAMPED,
)
from search.engine.region.decisions import place_scope_defaulted_warning, standard_probe_union
from search.engine.region.place_roles import PlaceRoles, where_binding
from search.engine.retrieval.bound_values import shape_member_facts, shape_query
from search.engine.shape import ShapeDraft, is_role_bearing
from search.heading.departure_fact import DepartureFact
from search.models import Warning

if TYPE_CHECKING:
    from search.engine.bind import SlotBindingDraft
    from search.engine.retrieval.retrieve import Materialised

# Where a context's probe places come from: the shape's bound places, the members a
# role-bearing shape's materialisation discovered, or a default scope, which is every place
# with data (`"every_place"`) or the countries fallback (`DefaultPlaceScope.every_place`).
PlaceMode = Literal["bound", "discovered", "every_place", "countries_fallback"]

# The codes of the place warnings a context carries (`PlaceContext.warnings`).
PLACE_WARNING_CODES = frozenset(
    {DIRECTION_DISCARDED, STANDARD_PROBE_PLACES_CLAMPED, PLACE_SCOPE_DEFAULTED}
)


@dataclass(frozen=True)
class PlaceContext:
    """One shape's places and the scope it is probed and read over."""

    # The shape's binding of the places: flow axes, role entities, wire roles, where places,
    # the observation fallback place and dropped directions.
    roles: PlaceRoles
    # Where the shape is probed and read. `None` probes its one bound place unexpanded, or
    # no place on a role-bearing shape with no destination.
    probe_dcids: tuple[str, ...] | None
    mode: PlaceMode
    # `PLACE_SCOPE_DEFAULTED` on a default scope, `STANDARD_PROBE_PLACES_CLAMPED` on a
    # clamped probe union, else `None`.
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
        return {
            warning_key(warning): direction_discarded(name, axis)
            for warning, (name, axis) in zip(
                direction_discarded_warnings(self.roles),
                self.roles.discarded_directions,
                strict=True,
            )
            if name not in self.roles.inferred_directions
        }

    def presence_for(self, presence: DefaultPresence | None) -> DefaultPresence | None:
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

        A discovered role scope's members replace the probe places, so the preview and later
        probes bind every member the open axis holds data for. A default scope takes the
        scope `mat` was settled on (`Materialised.default_scope`): the places of the
        StatVars the answer serves, with the scope warning saying which kind it is.
        Otherwise the context stands.
        """
        if mat.discovered_scope is not None:
            return dataclasses.replace(
                self, probe_dcids=tuple(mat.discovered_scope.member_dcids), mode="discovered"
            )
        if mat.default_scope is not None:
            return dataclasses.replace(
                self,
                probe_dcids=mat.default_scope.places,
                mode=_default_scope_mode(mat.default_scope),
                scope_warning=place_scope_defaulted_warning(mat.default_scope),
            )
        return self


def _default_scope_mode(scope: DefaultPlaceScope) -> PlaceMode:
    """The mode of a context probed over `scope`."""
    return "every_place" if scope.every_place else "countries_fallback"


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
            resolved_entities={},
            roles=[],
            where_dcids=[],
            obs_destination_dcid=None,
        ),
        probe_dcids=None,
        mode="bound",
        scope_warning=None,
    )


def takes_default_scope(shape: ShapeDraft, roles: PlaceRoles) -> bool:
    """Whether `shape` is probed over a default scope: it is subject-only and binds no place."""
    return not roles.where_dcids and not is_role_bearing(shape)


def direction_discarded_warnings(roles: PlaceRoles) -> tuple[Warning, ...]:
    """One `DIRECTION_DISCARDED` warning per directed place a subject-only shape reads as its
    subject, in place order, each name at its first place (`place_roles.bind_place_roles`)."""
    return tuple(
        Warning(
            code=DIRECTION_DISCARDED,
            severity="warn",
            message=(
                f"{name} was named as the {axis_noun(axis)}; this shape reports the place as "
                "its own subject, so the direction was dropped."
            ),
        )
        for name, axis in roles.discarded_directions
    )


def cannot_take_default_scope(shape: ShapeDraft) -> bool:
    """Whether `shape` is subject-only with a population the code index sources. Such a shape
    is probed over members its `sv_arc_facts` may not hold, which a default-scope presence
    read would not cover, so it takes no default scope."""
    return not is_role_bearing(shape) and shape_member_facts(shape, shape_query(shape, ()))[1]


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
    children: Mapping[str, list[str]],
    presence: DefaultPresence | None,
    max_probe_places: int,
) -> PlaceContext:
    """Build `shape`'s place context from its bound places.

    A subject-only shape that binds no place (`takes_default_scope`) is probed over the union
    of the default scopes of `stat_vars` (`DefaultPresence.union_scope`), the StatVars a
    Datasets mint serves for it (`state.served_stat_vars`), each of which is read at its own
    scope; the context carries a `PLACE_SCOPE_DEFAULTED` warning. It needs `presence`, and
    raises `AssertionError` without it. Materialisation narrows it to the scopes of the
    StatVars the answer serves (`settled`). One bound place probes its expanded members
    from `children`, or itself unexpanded when it has none.
    Several probe the deduplicated union of their expansions, capped at `max_probe_places`
    with a `STANDARD_PROBE_PLACES_CLAMPED` warning (`standard_probe_union`). A role-bearing
    shape with no bound destination probes no place and takes no default scope.

    `children` maps a place dcid to its expanded members.
    """
    where = roles.where_dcids
    if takes_default_scope(shape, roles):
        if presence is None:
            raise AssertionError(
                f"shape {shape.shape_id!r} binds no place and no default-scope presence was read"
            )
        scope = presence.union_scope(stat_vars)
        return PlaceContext(
            roles=roles,
            probe_dcids=scope.places,
            mode=_default_scope_mode(scope),
            scope_warning=place_scope_defaulted_warning(scope),
        )
    if len(where) <= 1:
        members = children.get(where[0]) if where else None
        return PlaceContext(
            roles=roles,
            probe_dcids=None if members is None else tuple(members),
            mode="bound",
            scope_warning=None,
        )
    dcids, clamped = standard_probe_union(where, children, max_probe_places=max_probe_places)
    return PlaceContext(
        roles=roles,
        probe_dcids=tuple(dcids),
        mode="bound",
        scope_warning=clamped[0] if clamped else None,
    )
