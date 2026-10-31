"""Expand the continents and subregions the membership policy holds to member countries.

A region's members are the present-day countries the membership policy holds for it
(`entity_resolution.membership`, `countries` projection), the answer the Graph API's resolve
and members doors give. Graph containment answers a different question (places with data) and
misplaces sovereign states in several regions, so it stays with `materialise.place_scope`. A
query that names no place takes the default place scope the datasets mint takes
(`materialise.place_scope`). Graph reads are synchronous. Async callers use
`asyncio.to_thread`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from entity_resolution.membership import MembershipPolicy, NotRecorded
from materialise import place_scope
from materialise.place_scope import (
    DefaultPlaceScope,
    DefaultPresence,
    DefaultScopeUnknownError,
    PlaceScopeReadError,
    is_worldwide,
)
from search.engine.degraded_reads import note_unsettled_scope
from search.engine.errors import GraphReadFailure

if TYPE_CHECKING:
    from search.engine.clients.graph import EngineGraphClient
    from search.engine.clients.sdmx import EngineSdmxClient

# The graph type every member of a region's `countries` projection has.
REGION_MEMBER_TYPE = "Country"


@dataclass
class PlaceExpansion:
    """A region's member country dcids, sorted, and the graph type they share."""

    children: list[str]
    child_type: str


def expand_places(
    dcids: Iterable[str], *, membership: MembershipPolicy
) -> dict[str, PlaceExpansion | None]:
    """The members of each region in `dcids` under the policy's default `countries` projection.

    Every dcid is a region by the policy (`MembershipPolicy.holds_region`). One the policy omits
    (`omitted_regions`, such as Antarctica, which holds no country) maps to `None`, so its set
    reads as its node alone; graph containment is not read in its place.

    Raises:
        UnknownMembership: a dcid the policy neither holds nor omits.
    """
    results: dict[str, PlaceExpansion | None] = {}
    for dcid in dcids:
        if dcid in membership.omitted_regions:
            results[dcid] = None
            continue
        members = membership.members(dcid).members
        # A region's rows are countries, so the policy always records its `countries`.
        assert not isinstance(members, NotRecorded), dcid
        results[dcid] = PlaceExpansion(children=list(members), child_type=REGION_MEMBER_TYPE)
    return results


def is_worldwide_scope(place_dcids: Sequence[str] | None) -> bool:
    """Check whether the place scope is Earth alone, a worldwide total. A single named country has
    a different dcid, so callers can distinguish worldwide coverage from a bound place.
    """
    return place_dcids is not None and is_worldwide(place_dcids)


def default_place_scope_message(scope: DefaultPlaceScope) -> str:
    """Return the shared `PLACE_SCOPE_DEFAULTED` message: a worldwide total for Earth alone,
    every place with data, or the countries with data when any StatVar's full list was too
    large or unavailable."""
    if is_worldwide(scope.places):
        return "no place was named; the answer was resolved as a worldwide total."
    if scope.every_place:
        return "no place was named; the answer covers every place with data, at every level."
    return (
        "no place was named; the answer covers the countries with data, because the full list "
        "of places was too large or unavailable."
    )


class ConsultedPresence:
    """A default-scope presence read (`materialise.place_scope.DefaultPresence`) as Search
    reads it.

    A decision reads a StatVar's scope only through `consulted_scope`, `consulted_union` and
    `consulted_any_places`, which note a scope that is a countries fallback this request
    alone took (`degraded_reads.note_unsettled_scope`), so the answer the decision shapes
    stays out of the response cache. A cache warm reads through `warming_places`, which
    notes nothing, and `covers` reads no scope. A presence no decision consults notes
    nothing. The read itself stays private, so Search cannot read a scope around them."""

    __slots__ = ("_presence",)

    def __init__(self, presence: DefaultPresence) -> None:
        self._presence = presence

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConsultedPresence) and self._presence == other._presence

    def __hash__(self) -> int:
        return hash(self._presence)

    def __repr__(self) -> str:
        return f"ConsultedPresence({self._presence!r})"

    def read_more(self, read: Callable[[DefaultPresence], DefaultPresence]) -> ConsultedPresence:
        """The presence `read` gives over this one, this one itself when it adds nothing."""
        presence = read(self._presence)
        return self if presence is self._presence else ConsultedPresence(presence)

    def consulted_scope(self, stat_var: str) -> DefaultPlaceScope:
        """The default scope of `stat_var`, for a decision that reads it."""
        scope = self._presence.scope(stat_var)
        if not scope.settled:
            note_unsettled_scope(stat_var)
        return scope

    def consulted_union(self, stat_vars: Iterable[str]) -> DefaultPlaceScope:
        """The union of the default scopes of `stat_vars` (`DefaultPresence.union_scope`), each
        read through `consulted_scope`."""
        stat_vars = list(dict.fromkeys(stat_vars))
        for stat_var in stat_vars:
            self.consulted_scope(stat_var)
        return self._presence.union_scope(stat_vars)

    def consulted_any_places(self, stat_vars: Iterable[str]) -> bool:
        """Whether the default scope of any of `stat_vars` holds a place. A settled scope
        holding one answers alone; otherwise the answer rests on every unsettled scope, each
        consulted. Neither the answer nor the note depends on the order of `stat_vars`."""
        scopes = {sv: self._presence.scope(sv) for sv in stat_vars}
        if any(scope.places for scope in scopes.values() if scope.settled):
            return True
        unsettled = [sv for sv, scope in scopes.items() if not scope.settled]
        consulted = [self.consulted_scope(sv) for sv in unsettled]
        return any(scope.places for scope in consulted)

    def covers(self, stat_var: str) -> bool:
        """Whether the read holds a default scope for `stat_var`."""
        return stat_var in self._presence.scopes

    def warming_places(
        self, members: Iterable[str], *, own: Collection[str]
    ) -> dict[str, tuple[str, ...]]:
        """The places a cache warm reads each of `members` at: each of them in `own`, the
        StatVars the resolve reads at their own scope (`standard._member_places`), at that
        scope. A warm decides nothing, so it notes nothing."""
        return {sv: self._presence.scope(sv).places for sv in members if sv in own}

    def merged(self, other: ConsultedPresence) -> ConsultedPresence:
        """This read with the StatVars `other` covers and it does not
        (`DefaultPresence.merged`)."""
        return ConsultedPresence(self._presence.merged(other._presence))


def read_default_presence(
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    stat_vars: Iterable[str],
    known: ConsultedPresence | None = None,
    deadline: float | None = None,
) -> ConsultedPresence:
    """The default place scopes (`materialise.place_scope.read_default_presence`) of
    `stat_vars` over Search's graph client, under the request's `deadline`. `sdmx` reads our
    variables' availability; `graph` reads the UN deployment's.

    An availability error the countries fallback does not cover
    (`place_scope.PlaceScopeReadError`), and a listing of Earth that names no country, raise
    `GraphReadFailure` with the upstream status, so Search refuses either as a retryable
    infrastructure failure. The read may be one the datasets door's
    reader started, so the error carries no SDMX client's type. A read the deadline
    ends raises `place_scope.PlaceScopeDeadlineError`, a `TimeoutError`, which Search refuses
    as a timeout."""

    def read(raw: DefaultPresence | None) -> DefaultPresence:
        try:
            return place_scope.read_default_presence(
                graph=graph, availability=sdmx, stat_vars=stat_vars, known=raw, deadline=deadline
            )
        except (DefaultScopeUnknownError, PlaceScopeReadError) as exc:
            raise GraphReadFailure(str(exc), upstream_status=exc.upstream_status) from exc

    return ConsultedPresence(read(None)) if known is None else known.read_more(read)


def earth_countries(
    *, graph: EngineGraphClient, deadline: float | None = None
) -> ConsultedPresence:
    """Earth's countries (`materialise.place_scope.earth_countries`) as a presence that covers
    no StatVar yet, which a later `read_default_presence` takes as `known`. A listing that
    names no country raises `GraphReadFailure`."""
    try:
        countries = place_scope.earth_countries(graph, deadline=deadline)
    except DefaultScopeUnknownError as exc:
        raise GraphReadFailure(str(exc)) from exc
    return ConsultedPresence(DefaultPresence(countries=countries, scopes={}))
