"""Every spec a first answer offers resubmits to the same place facts.

Each case places data for the StatVars of one route on a fixture graph, by
`itertools.product` over a table of placements (nowhere, Kenya, Tanzania, both, Earth only,
Earth and Kenya), under a date window and, on some routes, an availability listing that disagrees
with the facets in one of two ways (`_LIES`). Country facets run 2015 to 2022 and Earth's 2000 to 2010, so a window can
hold one and not the other.

The routes are a first election with one shape of two members (unbound, and with gender
bound to female), a nominated alternate (candidates, a lone alternate, a flagged no-data
match), a nominated alternate of three members that the query's wording does not rival, a
zero-observation re-election that adopts a sibling, a sibling menu, a sibling menu beside a
nominated alternate, close-scoring readings of other measures the model does not nominate,
a role-bearing sibling beside a subject sibling, a role-bearing elected shape beside a
subject alternate, and two named controls: income from Norway, whose by-creditor series
discovers its recipients beside a subject alternate with data in Norway, and income of
Ghana from Norway, with the subject alternate's data at both.

For every case the test checks:

- the answer against an oracle computed from the placement table alone: the status, the
  StatVars offered, and for each subject-only spec its scope, coverage label and places,
  and its scope warning;
- for every offered spec (`specs`, `flagged_match`, `alternate_match`, `nearest_real`), a
  resubmit of it on the same graph: equal coverage, `resolved.entities`, `resolved.scope`,
  resolved sources, scope warnings (attributed to the spec from the engine's place
  contexts where the wire leaves a warning unstamped), the entities its observation reads
  are built over, and its SDMX read path (bulk or per pair). A nearest-real suggestion
  states no `resolved.scope`, so its scope is left out;
- the StatVars whose availability each door reads (`_FIRST_ANSWER_READ` for a first answer),
  no presence read, since every scope is every place with data and none falls back to
  countries, and one listing of Earth's countries by a first answer that reads a default
  scope, none by a resubmit.

Across the cases each route must reach the multi-spec answers it implies.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import itertools
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from materialise.place_scope import DefaultPlaceScope, scope_of_places
from search.engine.bind import SlotBindingDraft
from search.engine.extract import DateRequest
from search.engine.place import default_place_scope_message
from search.engine.quantity_read import QuantityOutput
from search.engine.region import (
    candidate_grounding,
    grounding,
    phase_answer,
    phase_place_materialise,
    promote_finalize,
    promote_ground,
    resolve_spec_resubmit,
)
from search.engine.region.codes import PLACE_SCOPE_DEFAULTED
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.shape import ShapeDraft
from search.models import NearestRealSpec, ResubmitIdentityPayload, Spec, TimeWindow
from tests.engine import test_first_answer_place_scenarios as scenarios
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._scope_availability import ScopeAvailability
from tests.materialise.conftest import FakeGraphReader

_KEN, _GHA, _TZA, _EARTH = scenarios._KEN, scenarios._GHA, scenarios._TZA, scenarios._EARTH
# Earth's countries (`scenarios._CHILDREN`).
_COUNTRIES = tuple(scenarios._CHILDREN[_EARTH]["Country"])
_SCOPE_CODES = (PLACE_SCOPE_DEFAULTED,)

# One shape of deaths: of females, of males, and in total.
_FEMALE = "Count_DeathEventScenario_Female"
_MALE = "Count_DeathEventScenario_Male"
_TOTAL = "Count_DeathEventScenario"


def _death_arcs(gender: str | None) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "DeathEventScenario"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if gender is not None:
        arcs["constraintProperties"] = {"nodes": [{"dcid": "gender"}]}
        arcs["gender"] = {"nodes": [{"dcid": gender}]}
    return arcs


_NODES = {
    **scenarios._NODES,
    _FEMALE: scenarios._sv("Count of deaths, female", _death_arcs("Female")),
    _MALE: scenarios._sv("Count of deaths, male", _death_arcs("Male")),
    _TOTAL: scenarios._sv("Count of deaths", _death_arcs(None)),
    "Female": {"label": "Female", "type": "GenderType"},
    "Male": {"label": "Male", "type": "GenderType"},
}

# Where a StatVar's data sits, by name.
_PLACEMENTS: dict[str, tuple[str, ...]] = {
    "none": (),
    "KEN": (_KEN,),
    "TZA": (_TZA,),
    "KEN+TZA": (_KEN, _TZA),
    "Earth": (_EARTH,),
    "Earth+KEN": (_EARTH, _KEN),
}
_SPANS = {_EARTH: (2000, 2010)}
_COUNTRY_SPAN = (2015, 2022)
_ROLE_YEARS = [2020, 2021, 2022]

_WINDOWS: dict[str, tuple[int, int] | None] = {
    "any-date": None,
    "recent": (2016, 2020),
    "early": (2003, 2006),
}


def _span(place: str) -> tuple[int, int]:
    return _SPANS.get(place, _COUNTRY_SPAN)


def _facets(place: str) -> list[dict]:
    start, end = _span(place)
    return [
        {
            "earliestDate": str(start),
            "latestDate": str(end),
            "obsCount": 5,
            "provenanceId": scenarios._SOURCE,
        }
    ]


def _in_window(start: int, end: int, window: tuple[int, int] | None) -> bool:
    return window is None or (start <= window[1] and window[0] <= end)


@dataclass(frozen=True)
class _Route:
    """One path through the engine: the query, the recalled StatVars, the bind step's
    election, and which StatVars take a placement."""

    name: str
    query: str
    svs: tuple[str, ...]
    scores: tuple[float, ...]
    placed: tuple[str, ...]
    chosen: int = 0
    also_plausible: tuple[int, ...] = ()
    bindings: tuple[SlotBindingDraft, ...] = ()
    # Data every case of the route holds besides the placed StatVars.
    fixed_obs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Role coverage rows: (variable, facet_id, donor, recipient, years).
    coverage_rows: tuple[tuple[str, str, str, str, list[int]], ...] = ()
    entities: tuple[str, ...] = ()
    directed: bool = False
    # Placements a StatVar is held to on this route, where the route needs it.
    held: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Whether cases also run with each availability listing that disagrees with the facets
    # (`_LIES`).
    lying: bool = False
    # Answers the route implies and must reach across its cases (`_reached`).
    reaches: tuple[str, ...] = ()


def _menu(*stat_vars: str) -> str:
    """The label of a candidates answer offering `stat_vars` (`_reached`)."""
    return "candidates:" + "+".join(sorted(stat_vars))


_M, _EST, _PROJ, _FAR, _LIVE = (
    scenarios._MORT,
    scenarios._MORT_EST,
    scenarios._MORT_PROJ,
    scenarios._FAR,
    scenarios._LIVESTOCK,
)
_IDS, _IDS_BC, _PPP, _ATLAS = scenarios._IDS, scenarios._IDS_BC, scenarios._PPP, scenarios._ATLAS
_NOR, _USA = scenarios._NOR, scenarios._USA
_FEMALE_BINDING = SlotBindingDraft(
    axis="what", property_dcid="gender", kind="value", value_dcids=["Female"]
)
_IDS_AT_USA = ((_IDS, "f_ids", "", _USA, _ROLE_YEARS),)

_ROUTES = (
    _Route(
        "two-member-shape",
        "death count",
        (_FEMALE, _MALE),
        (0.9, 0.89),
        placed=(_FEMALE, _MALE),
        lying=True,
        reaches=("definite", "flagged_match", "no_data"),
    ),
    _Route(
        "two-member-shape-bound-female",
        "death count of females",
        (_FEMALE, _MALE),
        (0.9, 0.89),
        placed=(_FEMALE, _MALE),
        bindings=(_FEMALE_BINDING,),
        reaches=("definite", "flagged_match", "no_data"),
    ),
    _Route(
        "flagged-alternate",
        scenarios._MORTALITY,
        (_M, _FAR),
        (0.90, 0.84),
        placed=(_M, _FAR),
        also_plausible=(1,),
        lying=True,
        reaches=(_menu(_M, _FAR), "definite", "alternate_match", "flagged_match", "no_data"),
    ),
    _Route(
        "flagged-three-member-alternate",
        scenarios._MORTALITY,
        (_M, _TOTAL, _FEMALE, _MALE),
        (0.90, 0.84, 0.835, 0.83),
        placed=(_TOTAL, _FEMALE, _MALE),
        also_plausible=(1,),
        reaches=(_menu(_M, _TOTAL), "definite", "alternate_match"),
    ),
    _Route(
        "adopted-sibling",
        scenarios._MORTALITY,
        scenarios._SIBLINGS[:2],
        scenarios._SIBLING_SCORES[:2],
        placed=(_M, _EST),
        reaches=("definite", "flagged_match", "no_data"),
    ),
    _Route(
        "sibling-menu",
        scenarios._MORTALITY,
        scenarios._SIBLINGS,
        scenarios._SIBLING_SCORES,
        placed=(_EST, _PROJ),
        reaches=(_menu(_EST, _PROJ), "definite", "flagged_match"),
    ),
    _Route(
        "sibling-menu-beside-flagged-alternate",
        scenarios._MORTALITY,
        (*scenarios._SIBLINGS, _FAR),
        (*scenarios._SIBLING_SCORES, 0.80),
        placed=(_EST, _PROJ, _FAR),
        also_plausible=(3,),
        reaches=(
            _menu(_EST, _PROJ),
            _menu(_EST, _FAR),
            "definite",
            "alternate_match",
            "flagged_match",
        ),
    ),
    _Route(
        "unnominated-other-measures",
        scenarios._MORTALITY,
        (_M, _LIVE, _FAR),
        (0.90, 0.89, 0.84),
        placed=(_M, _LIVE),
        reaches=("definite", "flagged_match", "no_data"),
    ),
    _Route(
        "role-bearing-sibling",
        scenarios._GNI,
        (_ATLAS, _PPP, _IDS),
        (0.85, 0.84, 0.83),
        placed=(_PPP,),
        coverage_rows=_IDS_AT_USA,
        reaches=(_menu(_PPP, _IDS), "definite", "flagged_match"),
    ),
    _Route(
        "role-bearing-elected-subject-alternate",
        scenarios._GNI,
        (_IDS, _PPP),
        (0.85, 0.84),
        placed=(_PPP,),
        also_plausible=(1,),
        coverage_rows=_IDS_AT_USA,
        reaches=(_menu(_IDS, _PPP), "definite", "alternate_match", "no_data"),
    ),
    _Route(
        "named-n1",
        "gross national income from Norway",
        (_IDS_BC, _PPP),
        (0.85, 0.84),
        placed=(),
        also_plausible=(1,),
        fixed_obs={_PPP: (_NOR,)},
        coverage_rows=(
            (_IDS_BC, "f_ids", _NOR, _GHA, _ROLE_YEARS),
            (_IDS_BC, "f_ids", _NOR, _USA, _ROLE_YEARS),
        ),
        entities=("Norway",),
        directed=True,
        reaches=("candidates", "no_data"),
    ),
    _Route(
        "named-k1",
        "gross national income of Ghana from Norway",
        (_IDS, _IDS_BC, _PPP),
        (0.85, 0.85, 0.84),
        placed=(),
        also_plausible=(1,),
        fixed_obs={_PPP: (_GHA, _NOR)},
        coverage_rows=(
            (_IDS, "f_ids", "", _GHA, _ROLE_YEARS),
            (_IDS_BC, "f_ids", _NOR, _GHA, _ROLE_YEARS),
        ),
        entities=("Ghana", "Norway"),
        directed=True,
        reaches=("candidates", "no_data"),
    ),
)
_ROUTES_BY_NAME = {route.name: route for route in _ROUTES}
# The sibling menus hold the elected count to no data and the role-bearing sibling route its
# elected Atlas reading; the three-member alternate's route holds the elected count to data
# in Kenya.
_HELD = {
    "flagged-three-member-alternate": {_M: (_KEN,)},
    "sibling-menu": {_M: ()},
    "sibling-menu-beside-flagged-alternate": {_M: ()},
    "role-bearing-sibling": {_ATLAS: ()},
}


@dataclass(frozen=True)
class _Case:
    route: _Route
    placements: Mapping[str, str]
    window: str
    # The way the availability listing disagrees with the facets (`_LIES`), or `None`.
    lie: str | None

    @property
    def id(self) -> str:
        placed = ",".join(f"{sv.rsplit('_', 1)[-1]}={p}" for sv, p in self.placements.items())
        lie = "" if self.lie is None else f"-lying-{self.lie}"
        return f"{self.route.name}[{placed}]-{self.window}{lie}"

    @property
    def facts(self) -> dict[str, tuple[str, ...]]:
        """Where each StatVar holds facets."""
        return {
            **{sv: tuple(places) for sv, places in self.route.fixed_obs.items()},
            **_HELD.get(self.route.name, {}),
            **{sv: _PLACEMENTS[name] for sv, name in self.placements.items()},
        }

    @property
    def window_years(self) -> tuple[int, int] | None:
        return _WINDOWS[self.window]

    @property
    def date_request(self) -> DateRequest | None:
        years = self.window_years
        if years is None:
            return None
        return DateRequest(window=TimeWindow(start_year=years[0], end_year=years[1]), latest=False)


def _cases() -> list[_Case]:
    return [
        _Case(route, dict(zip(placed, combo, strict=True)), window, lie)
        for route in _ROUTES
        for placed in [[sv for sv in route.placed if sv not in _HELD.get(route.name, {})]]
        for combo in itertools.product(_PLACEMENTS, repeat=len(placed))
        for window in _WINDOWS
        for lie in ((None, *_LIES) if route.lying else (None,))
    ]


# The places a lying availability listing reports for the route's first placed StatVar, from the places it
# has facets at. `ghana` adds Ghana, where it has none, and drops Earth, where it may have
# some. `earth` reports Earth alone, so a StatVar with facets in a country reads as
# worldwide only; one with facets nowhere still reads as nowhere.
_LIES: dict[str, Callable[[tuple[str, ...]], frozenset[str]]] = {
    "ghana": lambda places: frozenset({*places, _GHA} - {_EARTH}) if places else frozenset(),
    "earth": lambda places: frozenset({_EARTH}) if places else frozenset(),
}

_CASES = _cases()


def _presence(case: _Case) -> dict[str, frozenset[str]]:
    """Where availability lists each StatVar's data."""
    presence = {sv: frozenset(places) for sv, places in case.facts.items()}
    if case.lie is not None:
        first = case.route.placed[0]
        presence[first] = _LIES[case.lie](case.facts.get(first, ()))
    return presence


# The oracle: what the answer must be, from the placement table alone.


@dataclass(frozen=True)
class _Expected:
    """One subject-only reading's answer under the placements: the StatVars it serves (or
    the one it names when it has none), whether it has data, whether data sat at its probe
    places, its settled scope, its coverage label and the places its coverage counts."""

    stat_vars: tuple[str, ...]
    has_data: bool
    held: bool
    scope: DefaultPlaceScope
    label: str | None
    served: tuple[str, ...]

    @property
    def stat_var(self) -> str:
        """The first StatVar, which keys the reading."""
        return self.stat_vars[0]


def _scope(presence: Mapping[str, frozenset[str]], stat_vars: Sequence[str]) -> DefaultPlaceScope:
    present = set().union(*(presence.get(sv, frozenset()) for sv in stat_vars))
    return scope_of_places(present, every_place=True)


def _points(case: _Case, stat_var: str, place: str) -> bool:
    return place in case.facts.get(stat_var, ()) and _in_window(*_span(place), case.window_years)


def _reading(case: _Case, members: Sequence[str]) -> _Expected:
    """A subject-only reading of `members`, the members the mint's family election keeps.
    Each member is read at its own default scope, and the reading serves every member with
    a point there, as a mint of it does, over the union of their scopes. Without one, it
    names the first member holding data at its scope in other years, else the first member,
    and states the scope of the member it names."""
    presence = _presence(case)
    own = {sv: _scope(presence, [sv]) for sv in members}
    holding = [sv for sv in members if any(p in case.facts.get(sv, ()) for p in own[sv].places)]
    served = tuple(sv for sv in members if any(_points(case, sv, p) for p in own[sv].places))
    if not served:
        named = holding[0] if holding else members[0]
        return _Expected((named,), False, bool(holding), own[named], None, ())
    probe = tuple(dict.fromkeys(p for sv in served for p in own[sv].places))
    label = "worldwide" if probe == (_EARTH,) else "places" if len(probe) > 1 else "sources"
    served_places = tuple(
        p for p in probe if any(_points(case, sv, p) for sv in served if p in own[sv].places)
    )
    return _Expected(served, True, True, _scope(presence, served), label, served_places)


def _role_has_data(case: _Case) -> bool:
    return _in_window(_ROLE_YEARS[0], _ROLE_YEARS[-1], case.window_years)


@dataclass(frozen=True)
class _Answer:
    """The oracle's answer: its kind (`candidates`, `definite`, `alternate_match`,
    `flagged_match` or `no_data`), the StatVars offered with data, the subject-only
    readings whose place facts it states, and the readings it suggests as nearest real."""

    kind: str
    offered: frozenset[str]
    readings: Mapping[str, _Expected] = field(default_factory=dict)
    nearest: Mapping[str, _Expected] = field(default_factory=dict)


def _answer(case: _Case) -> _Answer | None:
    """The answer the placements imply for `case`, or `None` on a named control, whose
    oracle is fixed (`_NAMED`)."""
    name = case.route.name
    if name.startswith("named-"):
        return None
    if name == "two-member-shape":
        return _single(_reading(case, [_FEMALE, _MALE]))
    if name == "two-member-shape-bound-female":
        return _with_nearest(_single(_reading(case, [_FEMALE])), _reading(case, [_FEMALE, _MALE]))
    if name == "flagged-alternate":
        # The nominated alternate is a rival (status rule 2(b)) and rides on a refusal
        # (rule 1).
        far = _reading(case, [_FAR])
        return _with_alternates(_reading(case, [_M]), [far], riders=[far])
    if name == "flagged-three-member-alternate":
        # The family election keeps the total, which every gendered member's constraints
        # extend, so the alternate answers through the total alone. The nominated total is
        # a rival (status rule 2(b)) and rides on a refusal (rule 1).
        total = _reading(case, [_TOTAL])
        return _with_alternates(_reading(case, [_M]), [total], riders=[total])
    if name == "adopted-sibling":
        return _reelection(_reading(case, [_M]), [_reading(case, [_EST])])
    if name == "sibling-menu":
        return _reelection(_reading(case, [_M]), [_reading(case, [_EST]), _reading(case, [_PROJ])])
    if name == "sibling-menu-beside-flagged-alternate":
        far = _reading(case, [_FAR])
        return _reelection(
            _reading(case, [_M]),
            [_reading(case, [_EST]), _reading(case, [_PROJ])],
            rivals=[far],
            riders=[far],
        )
    if name == "unnominated-other-measures":
        # Readings of other measures the model does not nominate are never offered,
        # however close their scores (status rules 2 and 5).
        return _single(_reading(case, [_M]))
    if name == "role-bearing-sibling":
        # The elected Atlas reading has no data, so its one qualifier-only sibling, GNI PPP,
        # answers in its place (status rule 1). PPP's price basis is not the declared
        # default, so the IDS series, differing from it on that open axis, is its rival
        # (rule 2(a)). The IDS series is no stand-in for the Atlas reading, being
        # role-bearing, and the model nominates nothing to ride on the refusal.
        ppp = _reading(case, [_PPP])
        if not ppp.has_data:
            return _Answer("flagged_match", frozenset())
        readings = {_PPP: ppp}
        if _role_has_data(case):
            return _Answer("candidates", frozenset({_PPP, _IDS}), readings)
        return _Answer("definite", frozenset({_PPP}), readings)
    if name == "role-bearing-elected-subject-alternate":
        ppp = _reading(case, [_PPP])
        readings = {_PPP: ppp} if ppp.has_data else {}
        if _role_has_data(case):
            kind = "candidates" if ppp.has_data else "definite"
            return _Answer(kind, frozenset({_IDS, *readings}), readings)
        if ppp.has_data:
            return _Answer("alternate_match", frozenset({_PPP}), readings)
        return _Answer("no_data", frozenset())
    raise AssertionError(f"no oracle for route {name!r}")


def _with_nearest(answer: _Answer, relaxed: _Expected) -> _Answer:
    """`answer`, suggesting `relaxed`, the reading with the bound value dropped, as nearest
    real when the elected reading has no data and the relaxed one has."""
    if answer.kind not in ("flagged_match", "no_data") or not relaxed.has_data:
        return answer
    return dataclasses.replace(answer, nearest={relaxed.stat_var: relaxed})


def _single(elected: _Expected) -> _Answer:
    return _with_alternates(elected, [])


def _with_alternates(
    elected: _Expected, rivals: Sequence[_Expected], *, riders: Sequence[_Expected] = ()
) -> _Answer:
    """The elected reading beside its rivals and the nominated readings that ride on a
    refusal (status rules 1 to 3). An elected reading with data is offered with every rival
    with data. One without data offers the first rider with data as the alternate match.
    With none, a reading whose data sits outside the years read answers no data for the
    window, and one with no data at its places offers its representative as the flagged
    match."""
    if elected.has_data:
        with_data = [r for r in (elected, *rivals) if r.has_data]
        readings = {r.stat_var: r for r in with_data}
        kind = "definite" if len(with_data) == 1 else "candidates"
        return _Answer(kind, frozenset(sv for r in with_data for sv in r.stat_vars), readings)
    rider = next((r for r in riders if r.has_data), None)
    if rider is not None:
        return _Answer("alternate_match", frozenset(rider.stat_vars), {rider.stat_var: rider})
    kind = "no_data" if elected.held else "flagged_match"
    return _Answer(kind, frozenset(), {elected.stat_var: elected})


def _reelection(
    elected: _Expected,
    siblings: Sequence[_Expected],
    *,
    rivals: Sequence[_Expected] = (),
    riders: Sequence[_Expected] = (),
) -> _Answer:
    """A zero-observation re-election (status rule 1): an elected reading with data at none
    of its probe places tries its siblings that differ from it only on the measurement
    qualifier. One sibling with data answers as the shape in play, beside `rivals`; several
    are offered as candidates without the elected reading or its rivals. With none, the
    elected reading answers, a rider with data riding on its refusal."""
    if elected.has_data or elected.held:
        return _with_alternates(elected, rivals, riders=riders)
    found = [s for s in siblings if s.has_data]
    if len(found) == 1:
        return _with_alternates(found[0], rivals)
    if not found:
        return _with_alternates(elected, [], riders=riders)
    return _Answer(
        "candidates",
        frozenset(sv for r in found for sv in r.stat_vars),
        {r.stat_var: r for r in found},
    )


# The doors, with the engine state each records.


class _Graph(FakeGraph):
    """The fixture graph. Records every availability read, presence read and listing of
    Earth's countries, and answers availability and presence reads from `_presence` in
    place of the facets."""

    def __init__(self, case: _Case) -> None:
        super().__init__(
            nodes=_NODES,
            obs={
                f"{sv}|{place}": _facets(place)
                for sv, places in case.facts.items()
                for place in places
            },
            detect={
                case.route.query: {
                    "svs": list(case.route.svs),
                    "cosine_scores": list(case.route.scores),
                }
            },
            resolve=scenarios._RESOLVE,
            places=scenarios._PLACES,
            children=scenarios._CHILDREN,
        )
        self._presence = _presence(case)
        self._availability = ScopeAvailability(
            data=[(sv, place) for sv, places in self._presence.items() for place in places]
        )
        self.presence_reads: list[frozenset[str]] = []
        self.country_listings = 0
        self._lock = threading.Lock()

    @property
    def scope_availability(self) -> ScopeAvailability:
        """Availability answered from the presence, so a lie reaches the default scope."""
        return self._availability

    @property
    def availability_reads(self) -> frozenset[str]:
        """The StatVars whose default scope availability was read."""
        return frozenset(sv for read in self._availability.reads for sv in read)

    def entities_with_data_by_variable(self, variable_dcids, entity_dcids, *, deadline=None):
        with self._lock:
            self.presence_reads.append(frozenset(variable_dcids))
        return {
            sv: [e for e in entity_dcids if e in self._presence.get(sv, frozenset())]
            for sv in variable_dcids
        }

    def child_places_batch(self, parent_dcids, child_type, *, deadline=None):
        if list(parent_dcids) == [_EARTH] and child_type == "Country":
            with self._lock:
                self.country_listings += 1
        return super().child_places_batch(parent_dcids, child_type, deadline=deadline)


class _Bind:
    """Elects the route's chosen shape with its bindings and flags its alternates."""

    def __init__(self, route: _Route) -> None:
        self._route = route

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return elected_reply(
            self._route.chosen,
            bindings=list(self._route.bindings),
            also_plausible=list(self._route.also_plausible),
        ), None


def _shape_key(shape: ShapeDraft) -> tuple[str | None, ...]:
    return (
        shape.pop_type_dcid,
        shape.meas_prop_dcid,
        shape.stat_type_dcid,
        shape.meas_qual_dcid,
        shape.meas_denom_dcid,
    )


@dataclass
class _DoorRecord:
    """What one door's engine state held per spec: the entities each observation read was
    built over, by shape, the place context and bulk read each answer was read with, by the
    StatVars it serves, and for each nearest-real suggestion, by its spec id, the place
    context it was read over and the entities its observation read was built over."""

    observation_entities: dict[tuple[str | None, ...], list[Any]] = field(default_factory=dict)
    places: dict[tuple[str, ...], Any] = field(default_factory=dict)
    bulk: dict[tuple[str, ...], Any] = field(default_factory=dict)
    nearest_places: dict[str, Any] = field(default_factory=dict)
    nearest_observation_entities: dict[str, Any] = field(default_factory=dict)


def _record_door(monkeypatch: pytest.MonkeyPatch, modules: Sequence[Any]) -> _DoorRecord:
    record = _DoorRecord()
    lock = threading.Lock()
    for module in modules:
        if hasattr(module, "_observation_entities"):
            real_entities = module._observation_entities

            @functools.wraps(real_entities)
            def entities(shape, *, _real=real_entities, **kwargs):
                built, warning = _real(shape, **kwargs)
                with lock:
                    seen = record.observation_entities.setdefault(_shape_key(shape), [])
                    if built not in seen:
                        seen.append(built)
                return built, warning

            monkeypatch.setattr(module, "_observation_entities", entities)
        if hasattr(module, "bulk_scope_for"):
            real_bulk = module.bulk_scope_for

            @functools.wraps(real_bulk)
            def bulk(place, mat, *, _real=real_bulk, **kwargs):
                scope = _real(place, mat, **kwargs)
                with lock:
                    record.places[tuple(mat.sv_dcids)] = place
                    record.bulk[tuple(mat.sv_dcids)] = None if scope is None else scope.places
                return scope

            monkeypatch.setattr(module, "bulk_scope_for", bulk)
    return record


def _record_nearest_places(monkeypatch: pytest.MonkeyPatch, record: _DoorRecord) -> None:
    """Record, for each nearest-real suggestion of the first door, the place context it is
    read over and the entities its observation read is built over. The preview builds one
    entity map per suggestion, in the order the suggestions are offered."""
    real_suggest = phase_place_materialise._suggest_nearest_real
    real_entities = phase_place_materialise._observation_entities
    pending: list[str] = []

    @functools.wraps(real_suggest)
    def suggest(**kwargs):
        suggestions, warnings, reads, presence = real_suggest(**kwargs)
        record.nearest_places.update((spec_id, read.place) for spec_id, read in reads.items())
        pending.extend(spec.spec_id for spec in suggestions)
        return suggestions, warnings, reads, presence

    @functools.wraps(real_entities)
    def entities(shape, **kwargs):
        built, warning = real_entities(shape, **kwargs)
        record.nearest_observation_entities[pending.pop(0)] = built
        return built, warning

    monkeypatch.setattr(phase_place_materialise, "_suggest_nearest_real", suggest)
    monkeypatch.setattr(phase_place_materialise, "_observation_entities", entities)


def _coverage(case: _Case) -> FakeCoverageService:
    return FakeCoverageService(
        build_coverage_index(list(case.route.coverage_rows), role_names=scenarios._ROLE_NAMES)
    )


def _options(walked: Mapping[str, Mapping[str, tuple[str, ...]]]) -> EngineOptions:
    """Both doors read observations and probe option coverage. The materialise graph holds
    the StatVars of `walked` alone, so the mint's member walk settles exactly them: none on
    a first answer, which answers from recall, and on a resubmit the members its spec
    serves, as the walk over the live graph settles the family a first answer served."""
    return EngineOptions(
        include_observations=True,
        include_option_coverage=True,
        materialise_graph=FakeGraphReader(arcs=walked),
    )


def _first_answer(case: _Case, graph: _Graph) -> RegionResult:
    entities = list(case.route.entities)
    return asyncio.run(
        resolve_variable(
            case.route.query,
            entities=entities,
            date_request=case.date_request,
            detect_query=case.route.query,
            role_query=case.route.query,
            entity_roles=(
                door_roles_for_test_query(case.route.query, entities)
                if case.route.directed
                else None
            ),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_Bind(case.route),
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            base_timing={},
            coverage=_coverage(case).snapshot(),
            options=_options({}),
        )
    )


def _resubmit(
    case: _Case, spec: Spec, graph: _Graph, stat_vars: Sequence[str] | None = None
) -> RegionResult:
    """`spec` resubmitted over a graph whose family holds its StatVars, or `stat_vars` when
    given."""
    identity = spec.definition.identity
    arcs = graph.member_arcs()
    walked = {sv: arcs[sv] for sv in stat_vars or _stat_vars(spec)}
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity,
        ),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=_coverage(case).snapshot(),
        options=_options(walked),
    )


def _offered(result: RegionResult) -> list[tuple[str, Spec]]:
    return [
        *(("specs", spec) for spec in result.specs),
        *(("flagged_match", spec) for spec in result.flagged_match or ()),
        *([("alternate_match", result.alternate_match)] if result.alternate_match else []),
        *(("nearest_real", spec) for spec in result.nearest_real or ()),
    ]


def _kind(result: RegionResult) -> str:
    if result.status in ("candidates", "definite"):
        return result.status
    if result.alternate_match is not None:
        return "alternate_match"
    if result.flagged_match:
        return "flagged_match"
    return result.status


def _stat_vars(spec: Spec) -> tuple[str, ...]:
    return tuple(sv.ref.ref for sv in spec.resolved.members)


def _answered(result: RegionResult) -> Spec:
    """The one spec a resubmit answers with."""
    offered = [spec for where, spec in _offered(result) if where != "nearest_real"]
    assert len(offered) == 1, result
    return offered[0]


def _place_facts(spec: Spec) -> dict[str, Any]:
    """The place facts a spec states. A nearest-real suggestion states no scope, so its
    scope, and its resubmit's, are left out."""
    resolved = spec.resolved
    facts = {
        "coverage": resolved.coverage.model_dump(mode="json"),
        "entities": [e.model_dump(mode="json") for e in resolved.entities],
        "scope": None if resolved.scope is None else resolved.scope.model_dump(mode="json"),
        "sources": [s.model_dump(mode="json") for s in resolved.trace.resolved_sources],
        "facets": {k: v.model_dump(mode="json") for k, v in resolved.facets.items()},
    }
    if isinstance(spec, NearestRealSpec):
        del facts["scope"]
    return facts


def _scope_warnings(warnings: Sequence[Any]) -> list[tuple[str, str]]:
    return [(w.code, w.message) for w in warnings if w.code in _SCOPE_CODES]


def _attributed_scope_warnings(
    result: RegionResult, spec: Spec, record: _DoorRecord
) -> list[tuple[str, str]]:
    """The scope warnings that belong to `spec`. A spec read in a place context the engine
    recorded takes that context's warnings, each of which the wire must carry stamped with
    the spec's id or unstamped in the block of the shape in play. A flagged no-data match,
    read in no context, takes the unstamped block."""
    place = (
        record.nearest_places[spec.spec_id]
        if isinstance(spec, NearestRealSpec)
        else record.places.get(_stat_vars(spec))
    )
    if place is None:
        return _scope_warnings([w for w in result.warnings if w.spec_id is None])
    own = _scope_warnings(place.warnings)
    on_wire = {
        (w.code, w.message)
        for w in result.warnings
        if w.code in _SCOPE_CODES and w.spec_id in (None, spec.spec_id)
    }
    assert set(own) <= on_wire, (own, result.warnings)
    stamped = _scope_warnings([w for w in result.warnings if w.spec_id == spec.spec_id])
    assert set(stamped) <= set(own), (stamped, own)
    return own


def _expected_warning(scope: DefaultPlaceScope) -> list[tuple[str, str]]:
    return [(PLACE_SCOPE_DEFAULTED, default_place_scope_message(scope))]


# The StatVars whose availability a first answer reads, by route. A subject-only
# elected shape's read covers its rivals, the nominated readings that ride on a refusal
# and its re-election siblings, whether or not the answer reaches them, and no reading the
# status rule never offers. A role-bearing elected shape makes none; its subject-only
# alternate's read covers the alternate alone.
_FIRST_ANSWER_READ: dict[str, frozenset[str]] = {
    "two-member-shape": frozenset({_FEMALE, _MALE}),
    "two-member-shape-bound-female": frozenset({_FEMALE, _MALE}),
    "flagged-alternate": frozenset({_M, _FAR}),
    "flagged-three-member-alternate": frozenset({_M, _TOTAL, _FEMALE, _MALE}),
    "adopted-sibling": frozenset({_M, _EST}),
    "sibling-menu": frozenset({_M, _EST, _PROJ}),
    "sibling-menu-beside-flagged-alternate": frozenset({_M, _EST, _PROJ, _FAR}),
    "unnominated-other-measures": frozenset({_M}),
    "role-bearing-sibling": frozenset({_ATLAS, _PPP}),
    "role-bearing-elected-subject-alternate": frozenset({_PPP}),
}


def _check_case(case: _Case, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run `case` through both doors and check it; return the answer's kind."""
    with monkeypatch.context() as patch:
        first_record = _record_door(patch, [phase_answer, candidate_grounding])
        _record_nearest_places(patch, first_record)
        graph = _Graph(case)
        result = _first_answer(case, graph)
    first_reads = graph.availability_reads

    answer = _answer(case)
    offered = _offered(result)
    assert graph.presence_reads == [], case.id
    assert graph.country_listings == (1 if first_reads else 0), (case.id, graph.country_listings)
    if answer is not None:
        with_data = {
            sv
            for where, spec in offered
            if where != "nearest_real" and spec.resolved.coverage.has_data
            for sv in _stat_vars(spec)
        }
        observed = (_kind(result), sorted(with_data))
        assert observed == (answer.kind, sorted(answer.offered)), (case.id, observed)
        nearest = [sv for spec in result.nearest_real or () for sv in _stat_vars(spec)]
        assert sorted(nearest) == sorted(answer.nearest), (case.id, nearest)
        assert first_reads == _FIRST_ANSWER_READ[case.route.name], (
            case.id,
            first_reads,
        )
    else:
        named = _NAMED[case.route.name] if case.window != "early" else "no_data"
        assert _kind(result) == named, (case.id, _kind(result))
        assert not first_reads, case.id

    for where, spec in offered:
        readings = (
            {} if answer is None else answer.nearest if where == "nearest_real" else answer.readings
        )
        expected = readings.get(_stat_vars(spec)[0])
        mine = _attributed_scope_warnings(result, spec, first_record)
        if expected is not None:
            _check_reading(case, spec, expected, mine)
            if where == "nearest_real":
                read_over = first_record.nearest_places[spec.spec_id].probe_places
                assert read_over == list(expected.scope.places), (case.id, read_over)

        with monkeypatch.context() as patch:
            resubmit_record = _record_door(patch, [promote_ground, promote_finalize])
            resubmit_graph = _Graph(case)
            replay = _resubmit(case, spec, resubmit_graph)
        replayed = _answered(replay)
        context = f"{case.id} {where} {_stat_vars(spec)}"
        assert _stat_vars(replayed) == _stat_vars(spec), context
        replayed_facts = _place_facts(replayed)
        if where == "nearest_real":
            del replayed_facts["scope"]
        assert replayed_facts == _place_facts(spec), context
        assert _scope_warnings(replay.warnings) == mine, context
        if isinstance(spec, NearestRealSpec):
            built = first_record.nearest_observation_entities.get(spec.spec_id)
            assert resubmit_record.observation_entities.get(_shape_key_from(spec)) == (
                None if built is None else [built]
            ), context
        elif (key := _shape_key_of(spec, first_record)) is not None:
            assert (
                resubmit_record.observation_entities.get(key)
                == (first_record.observation_entities[key])
            ), context
        sv_key = _stat_vars(spec)
        if sv_key in first_record.bulk:
            assert resubmit_record.bulk.get(sv_key) == first_record.bulk[sv_key], context
        subject_placeless = not case.route.entities and spec.resolved.role_axis_facts is None
        expected_reads = frozenset(sv_key) if subject_placeless else frozenset()
        assert resubmit_graph.availability_reads == expected_reads, context
        assert (resubmit_graph.presence_reads, resubmit_graph.country_listings) == ([], 0), context
    return _kind(result)


def _shape_key_from(spec: Spec) -> tuple[str | None, ...]:
    """The five-tuple of `spec`'s shape, as `_shape_key` keys a draft."""
    shape = spec.resolved.shape
    return (
        shape.population_type.ref if shape.population_type else None,
        shape.measured_property.ref if shape.measured_property else None,
        shape.stat_type.ref if shape.stat_type else None,
        shape.measurement_qualifier.ref if shape.measurement_qualifier else None,
        shape.measurement_denominator.ref if shape.measurement_denominator else None,
    )


def _shape_key_of(spec: Spec, record: _DoorRecord) -> tuple[str | None, ...] | None:
    """The shape key the first door recorded observation reads under for `spec`, matched by
    its StatVars' five-tuple."""
    key = _shape_key_from(spec)
    return key if key in record.observation_entities else None


def _check_reading(
    case: _Case, spec: Spec, expected: _Expected, warnings: list[tuple[str, str]]
) -> None:
    """A subject-only spec states the oracle's scope, coverage label, places and warning. A
    nearest-real suggestion states no scope, so its places go unchecked here."""
    context = f"{case.id} {_stat_vars(spec)}"
    assert warnings == _expected_warning(expected.scope), context
    if not expected.has_data:
        return
    labels = {dim.label: dim.count for dim in spec.resolved.coverage.dimensions or []}
    assert expected.label in labels, (context, labels)
    if expected.label == "places":
        assert labels["places"] == len(expected.served), (context, labels)
    if isinstance(spec, NearestRealSpec):
        return
    scope = spec.resolved.scope
    assert scope is not None, context
    assert sorted(m.ref for m in scope.members) == sorted(expected.served), context


# The named controls' answers with data in the years read.
_NAMED = {"named-n1": "candidates", "named-k1": "candidates"}


def _reached(case: _Case) -> str:
    """The oracle's answer to `case` as a label: its kind, and for candidates the StatVars
    offered. Each case asserts the engine gives the oracle's answer."""
    answer = _answer(case)
    if answer is None:
        return _NAMED[case.route.name] if case.window != "early" else "no_data"
    return _menu(*answer.offered) if answer.kind == "candidates" else answer.kind


@pytest.mark.parametrize("case", _CASES, ids=[case.id for case in _CASES])
def test_every_offered_spec_resubmits_to_the_same_place_facts(case, monkeypatch) -> None:
    _check_case(case, monkeypatch)


def test_each_route_reaches_the_answers_it_implies(monkeypatch) -> None:
    """Across its placements each route reaches every answer it implies, the multi-spec ones
    included, so no route passes by answering one way only."""
    reached: dict[str, set[str]] = {route.name: set() for route in _ROUTES}
    for case in _CASES:
        reached[case.route.name].add(_reached(case))
    for route in _ROUTES:
        assert set(route.reaches) <= reached[route.name], (route.name, reached[route.name])


def test_a_resubmit_settles_on_the_members_it_serves() -> None:
    """Female has data in Kenya and male in Tanzania. A resubmit whose family holds both
    reads both countries and serves both, so it settles on Kenya and Tanzania; one whose
    family holds female alone serves female and settles on Kenya."""
    case = next(
        c
        for c in _CASES
        if c.route.name == "two-member-shape"
        and c.placements == {_FEMALE: "KEN", _MALE: "TZA"}
        and c.window == "any-date"
        and c.lie is None
    )
    (spec,) = _first_answer(case, _Graph(case)).specs

    both = _answered(_resubmit(case, spec, _Graph(case), [_FEMALE, _MALE]))
    female = _answered(_resubmit(case, spec, _Graph(case), [_FEMALE]))

    assert _stat_vars(both) == (_FEMALE, _MALE)
    assert sorted(m.ref for m in both.resolved.scope.members) == sorted([_KEN, _TZA])
    assert _stat_vars(female) == (_FEMALE,)
    assert [m.ref for m in female.resolved.scope.members] == [_KEN]


def test_a_placeless_nearest_real_suggestion_is_read_over_its_own_places(monkeypatch) -> None:
    """ "death count of females" with female deaths nowhere and male deaths in Kenya and
    Tanzania: the bound reading's scope is empty, so the answer suggests male deaths as
    nearest real. The suggestion's preview and its option-coverage probe read Kenya and
    Tanzania, the places its own StatVars settle on, as its resubmit does."""
    case = next(
        c
        for c in _CASES
        if c.route.name == "two-member-shape-bound-female"
        and c.placements == {_FEMALE: "none", _MALE: "KEN+TZA"}
        and c.window == "any-date"
    )
    probed: list[list[str] | None] = []
    real_probe = grounding.standard_option_coverage

    def option_coverage(**kwargs):
        probed.append(kwargs["place_dcids"])
        return real_probe(**kwargs)

    monkeypatch.setattr(grounding, "standard_option_coverage", option_coverage)
    record = _DoorRecord()
    _record_nearest_places(monkeypatch, record)
    result = _first_answer(case, _Graph(case))

    assert (result.status, result.no_data_reason) == ("no_data", "no_observations")
    (suggestion,) = result.nearest_real or ()
    assert _stat_vars(suggestion) == (_MALE,)
    labels = {dim.label: dim.count for dim in suggestion.resolved.coverage.dimensions or []}
    assert labels["places"] == 2
    assert record.nearest_places[suggestion.spec_id].probe_places == [_KEN, _TZA]
    assert record.nearest_observation_entities[suggestion.spec_id] == {
        "observationAbout": [_KEN, _TZA]
    }
    assert probed == [[_KEN, _TZA]]
