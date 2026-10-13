"""One facet group serves a member at every place or role tuple its request binds, so every
election leads with the series each facet holds a point at in the window
(`materialise.facets.rank_candidates`' `breadth`): a mint's probe counts them
(`MemberProbe.breadth`), and member, unit-request and group selection rank by them.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from materialise.facet_profile import ProfileBreadth, _profile
from materialise.facets import (
    FacetBinding,
    SeriesCounts,
    UnitRequest,
    coverage_probe,
    merge_facets_by_series,
    probe_facets,
    rank_candidates,
    series_breadth,
    subject_probe,
    summed_counts,
)
from materialise.generation import _select_member_facets
from materialise.graph import Facet
from materialise.group_facets import group_breadth, select_group_facets
from materialise.pipeline import MaterialisationRequest
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._scope_availability import NO_SCOPE_READ
from .conftest import FakeGraphReader

_SV = "TESTFIXTURE/SeriesBreadth/Amount"
_PLACES = ["country/KEN", "country/GHA", "country/TGO"]
# Each facet's source, unit and the years it holds wherever a place holds it. `fLong` holds
# the most observations and `fWide` the fewest.
_FACETS = {
    "fLong": ("dc/base/Long", "CurrencyUSD", range(1990, 2021)),
    "fMid": ("dc/base/Mid", "CurrencyUSD", range(2010, 2023)),
    "fWide": ("dc/base/Wide", "CurrencyEUR", range(2018, 2022)),
}
_WINDOWS = {
    "all": RequestedWindow(kind="all"),
    "latest": RequestedWindow(kind="latest"),
    "2019_to_2021": RequestedWindow(kind="range", start_year=2019, end_year=2021),
    "2021_to_2022": RequestedWindow(kind="range", start_year=2021, end_year=2022),
    "2030_on": RequestedWindow(kind="range", start_year=2030, end_year=None),
}
_HOLDINGS = [
    held
    for held in itertools.product(
        [(), ("fLong",), ("fWide",), ("fLong", "fMid"), ("fLong", "fMid", "fWide")], repeat=3
    )
    if any(held)
]


def _facet(facet_id: str) -> Facet:
    provenance, unit, years = _FACETS[facet_id]
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        unit=unit,
        obs_count=len(years),
        earliest_date=str(years[0]),
        latest_date=str(years[-1]),
    )


def _in_window(facet_id: str, window: RequestedWindow) -> bool:
    years = _FACETS[facet_id][2]
    return window.kind != "range" or any(
        (window.start_year is None or year >= window.start_year)
        and (window.end_year is None or year <= window.end_year)
        for year in years
    )


def _request(window: RequestedWindow, **facet_bindings: FacetBinding) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{_SV}",
        role_bindings={"observationAbout": _PLACES},
        declared_roles=("observationAbout",),
        facet_bindings=facet_bindings,
        window=window,
    )


def _places_covered(held: tuple[tuple[str, ...], ...], window: RequestedWindow) -> dict[str, int]:
    """Each facet's count of the places holding it with a point in `window`, stated from
    the holdings alone."""
    return {
        facet_id: count
        for facet_id in _FACETS
        if (count := sum(1 for place in held if facet_id in place) * _in_window(facet_id, window))
    }


@pytest.mark.parametrize("window", list(_WINDOWS))
@pytest.mark.parametrize("held", _HOLDINGS, ids=lambda held: "|".join(",".join(p) for p in held))
def test_a_mint_serves_the_facet_covering_the_most_places_in_its_window(
    held: tuple[tuple[str, ...], ...], window: str
) -> None:
    """Whatever three places hold, the probe counts each facet's places with a point in the
    window, and the facet a member selects covers as many of them as any candidate does: a
    longer series at fewer places never outranks it."""
    requested = _WINDOWS[window]
    graph = FakeGraphReader(
        facets={
            (_SV, place): [_facet(facet_id) for facet_id in facets]
            for place, facets in zip(_PLACES, held, strict=True)
        }
    )
    probe = probe_facets(
        availability=NO_SCOPE_READ, stat_var=_SV, bound_entities=_PLACES, settle=True, graph=graph
    )
    covered = _places_covered(held, requested)

    assert probe.breadth_in(requested) == covered
    selections = _select_member_facets(
        _request(requested),
        probe.candidates,
        graph=graph,
        breadth=probe.breadth_in(requested),
    ).selections
    elected = selections["provenance"].dispatch_facet_id
    assert covered.get(elected, 0) == max(covered.values(), default=0)
    # Every axis is resolved from that one facet.
    assert {s.dispatch_facet_id for s in selections.values()} == {elected}


_DOLLAR_HOLDINGS = [
    held
    for held in _HOLDINGS
    if any(_FACETS[facet_id][1] == "CurrencyUSD" for place in held for facet_id in place)
]


@pytest.mark.parametrize("window", list(_WINDOWS))
@pytest.mark.parametrize(
    "held", _DOLLAR_HOLDINGS, ids=lambda held: "|".join(",".join(p) for p in held)
)
def test_a_unit_request_ranks_its_currency_candidates_by_the_places_they_cover(
    held: tuple[tuple[str, ...], ...], window: str
) -> None:
    """A US dollar request prefers the dollar facets, and among them the one covering the
    most places in the window, as a member selection ranks them."""
    requested = _WINDOWS[window]
    graph = FakeGraphReader(
        facets={
            (_SV, place): [_facet(facet_id) for facet_id in facets]
            for place, facets in zip(_PLACES, held, strict=True)
        }
    )
    probe = probe_facets(
        availability=NO_SCOPE_READ, stat_var=_SV, bound_entities=_PLACES, settle=True, graph=graph
    )
    covered = {
        facet_id: count
        for facet_id, count in _places_covered(held, requested).items()
        if _FACETS[facet_id][1] == "CurrencyUSD"
    }

    selections = _select_member_facets(
        _request(
            requested,
            unit=FacetBinding(
                state="request", request=UnitRequest(currency="USD", prices="current")
            ),
        ),
        probe.candidates,
        graph=graph,
        breadth=probe.breadth_in(requested),
    ).selections

    assert selections["unit"].ref == "CurrencyUSD"
    elected = selections["unit"].dispatch_facet_id
    assert covered.get(elected, 0) == max(covered.values(), default=0)


def test_a_facet_covering_every_place_equally_ranks_as_it_does_at_one_place() -> None:
    """Counts that tie leave the order to the years and the observations: with every facet
    at one place the ranking equals the ranking with no count at all."""
    candidates = [_facet(facet_id) for facet_id in _FACETS]
    for window in _WINDOWS.values():
        one_place = series_breadth({"country/KEN": candidates}, window=window)
        assert rank_candidates(candidates, window=window, breadth=one_place) == rank_candidates(
            candidates, window=window
        )


_CREDITORS = ["country/NOR", "country/FRA", "country/DEU"]
_ROLE_NAMES = {_SV: RoleNames(donor="creditor", recipient="debtor")}


@pytest.mark.parametrize("window", list(_WINDOWS))
@pytest.mark.parametrize("held", _HOLDINGS, ids=lambda held: "|".join(",".join(p) for p in held))
def test_a_role_bearing_mint_counts_and_elects_by_role_pairs(
    held: tuple[tuple[str, ...], ...], window: str
) -> None:
    """A role-bearing member's probe comes from coverage: each facet counts the creditor and
    debtor pairs holding it with a point in the window, the same probe a request binding the
    debtor reads, and the facet selected covers as many pairs as any candidate."""
    requested = _WINDOWS[window]
    coverage = build_coverage_index(
        [
            (_SV, facet_id, creditor, "country/KEN", list(_FACETS[facet_id][2]))
            for creditor, facets in zip(_CREDITORS, held, strict=True)
            for facet_id in facets
        ],
        role_names=_ROLE_NAMES,
        facet_metadata={
            facet_id: FacetMetadata(provenance_id=provenance, unit=unit)
            for facet_id, (provenance, unit, _years) in _FACETS.items()
        },
    )
    request = MaterialisationRequest(
        stat_var=f"dcid:{_SV}",
        role_bindings={"debtor": ["country/KEN"]},
        declared_roles=("creditor", "debtor"),
        facet_bindings={},
        window=requested,
    )
    probe = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var=_SV,
        bound_entities=["country/KEN"],
        settle=True,
        graph=FakeGraphReader(),
        declared_roles=request.declared_roles,
        role_bindings=request.role_bindings,
        coverage=coverage,
    )
    covered = _places_covered(held, requested)
    if requested.kind == "range" and requested.start_year == 2030:
        # Coverage holds no year from 2030, so no pair holds a point there.
        covered = {}

    assert probe.breadth_in(requested) == covered
    direct = coverage_probe(
        stat_var=_SV, bound={"recipient": frozenset({"country/KEN"})}, coverage=coverage
    )
    assert direct.candidates == probe.candidates
    assert direct.breadth_in(requested) == covered
    selections = _select_member_facets(
        request, probe.candidates, graph=FakeGraphReader(), breadth=probe.breadth_in(requested)
    ).selections
    elected = selections["provenance"].dispatch_facet_id
    assert covered.get(elected, 0) == max(covered.values(), default=0)


def test_a_group_ranks_its_shared_facets_by_the_places_they_cover_across_its_members() -> None:
    """Two members share `fLong` and `fMid`. `fLong` holds more observations, and `fMid`
    covers three places across the members against two, so the group selects `fMid`."""
    members = [f"{_SV}/A", f"{_SV}/B"]
    graph = FakeGraphReader(
        facets={
            (members[0], "country/KEN"): [_facet("fLong"), _facet("fMid")],
            (members[0], "country/GHA"): [_facet("fMid")],
            (members[1], "country/KEN"): [_facet("fLong"), _facet("fMid")],
        }
    )
    window = RequestedWindow(kind="all")
    requests = [
        MaterialisationRequest(
            stat_var=f"dcid:{member}",
            role_bindings={"observationAbout": ["country/KEN", "country/GHA"]},
            declared_roles=("observationAbout",),
            facet_bindings={},
            window=window,
        )
        for member in members
    ]
    probes = [
        probe_facets(
            availability=NO_SCOPE_READ,
            stat_var=member,
            bound_entities=["country/KEN", "country/GHA"],
            settle=True,
            graph=graph,
        )
        for member in members
    ]

    breadth = group_breadth(probes, window=window)
    assert breadth == {"fLong": 2, "fMid": 3}
    selections = select_group_facets(
        requests, [probe.candidates for probe in probes], graph=graph, breadth=breadth
    ).selections
    assert selections["provenance"].dispatch_facet_id == "fMid"
    without = select_group_facets(
        requests, [probe.candidates for probe in probes], graph=graph
    ).selections
    assert without["provenance"].dispatch_facet_id == "fLong"


# A recorded series at two places and the `Estimated` series continuing it at two others,
# under one provenance and unit, beside another source's series at three of the four places.
_JOIN_PLACES = ["country/KEN", "country/GHA", "country/TGO", "country/NGA"]
_JOIN_FACETS = {
    "fRec": ("dc/base/Rec", None, range(2010, 2021), _JOIN_PLACES[:2]),
    "fEst": ("dc/base/Rec", "Estimated", range(2021, 2031), _JOIN_PLACES[2:]),
    "fOther": ("dc/base/Other", None, range(2010, 2026), _JOIN_PLACES[:3]),
}
_JOIN_WINDOW = RequestedWindow(kind="range", start_year=2015, end_year=2028)


def _join_facet(facet_id: str) -> Facet:
    provenance, method, years, _places = _JOIN_FACETS[facet_id]
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method=method,
        unit="CurrencyUSD",
        obs_count=len(years),
        earliest_date=str(years[0]),
        latest_date=str(years[-1]),
    )


def _join_by_place(places: list[str]) -> dict[str, list[Facet]]:
    return {
        place: [_join_facet(f) for f, (*_, held) in _JOIN_FACETS.items() if place in held]
        for place in places
    }


def _join_coverage():
    return build_coverage_index(
        [
            (_SV, facet_id, creditor, "country/KEN", list(years))
            for facet_id, (_provenance, _method, years, creditors) in _JOIN_FACETS.items()
            for creditor in creditors
        ],
        role_names=_ROLE_NAMES,
        facet_metadata={
            facet_id: FacetMetadata(
                provenance_id=provenance,
                measurement_method=method,
                unit="CurrencyUSD",
                observation_period="P1Y",
            )
            for facet_id, (provenance, method, _years, _places) in _JOIN_FACETS.items()
        },
    )


def _join_probe(path: str) -> tuple[tuple[Facet, ...], Mapping[str, int] | None]:
    if path == "role":
        probe = coverage_probe(
            stat_var=_SV, bound={"recipient": frozenset({"country/KEN"})}, coverage=_join_coverage()
        )
        return probe.candidates, probe.breadth_in(_JOIN_WINDOW)
    if path == "group":
        # Two members split the places, so the group's union sums each member's.
        probes = [
            subject_probe(_join_by_place(places), places)
            for places in (_JOIN_PLACES[::2], _JOIN_PLACES[1::2])
        ]
        candidates = subject_probe(_join_by_place(_JOIN_PLACES), _JOIN_PLACES).candidates
        return candidates, group_breadth(probes, window=_JOIN_WINDOW)
    probe = subject_probe(_join_by_place(_JOIN_PLACES), _JOIN_PLACES)
    return probe.candidates, probe.breadth_in(_JOIN_WINDOW)


@pytest.mark.parametrize("path", ["subject", "role", "group"])
def test_a_joined_candidate_counts_the_places_any_of_its_facets_holds(path: str) -> None:
    """Under a 2015 to 2028 window `fRec` and the estimates continuing it rank as one
    candidate. Each holds two places and `fOther` three, but joined they serve all four, so
    the joined candidate leads."""
    candidates, breadth = _join_probe(path)

    assert breadth == {"fRec": 2, "fEst": 2, "fOther": 3}
    ranked = rank_candidates(candidates, window=_JOIN_WINDOW, breadth=breadth)
    assert [f.facet_id for f in ranked] == ["fRec", "fEst", "fOther"]


@pytest.mark.parametrize(
    ("window", "reads"),
    [
        pytest.param(_JOIN_WINDOW, [frozenset({"fRec", "fEst"})], id="join-reads-its-facets-once"),
        pytest.param(RequestedWindow(kind="all"), [], id="no-join-reads-nothing"),
    ],
)
def test_a_role_bearing_union_is_read_only_for_a_join_and_once(
    monkeypatch: pytest.MonkeyPatch, window: RequestedWindow, reads: list[frozenset[str]]
) -> None:
    """Coverage counts the pairs a joined candidate's facets hold together when the ranking
    forms the join, once however often it ranks, and never for a ranking that forms none."""
    read: list[frozenset[str]] = []
    union_series_count = CoverageIndex.union_series_count

    def counting(self: CoverageIndex, **kwargs: Any) -> int:
        read.append(frozenset(kwargs["facet_ids"]))
        return union_series_count(self, **kwargs)

    monkeypatch.setattr(CoverageIndex, "union_series_count", counting)
    probe = coverage_probe(
        stat_var=_SV, bound={"recipient": frozenset({"country/KEN"})}, coverage=_join_coverage()
    )
    breadth = probe.breadth_in(window)
    for _ in range(2):
        rank_candidates(probe.candidates, window=window, breadth=breadth)

    assert read == reads


# A profile holds each facet's own count and no union. A ranking that joins facets, at one
# bound place or across a group's merged members, counts their union once per read.
_JOIN_TO_2030 = RequestedWindow(kind="range", start_year=2020, end_year=2030)


def _dated(facet_id: str, method: str | None, first: int, last: int) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id="dc/base/Rec",
        measurement_method=method,
        unit="CurrencyUSD",
        obs_count=last - first + 1,
        earliest_date=str(first),
        latest_date=str(last),
    )


class _CountingPlaces(dict[str, frozenset[str]]):
    """A profile's `places_by_facet` counting the facets a union reads."""

    reads: int = 0

    def get(self, key, default=None):
        self.reads += 1
        return super().get(key, default)


class _Profiles:
    """A `FacetProfileCache` stand-in serving fixed profiles."""

    def __init__(self, profiles: Mapping[str, Any]) -> None:
        self._profiles = profiles

    def profiles(self, stat_vars, *, catalogue, coverage, deadline):
        return {sv: self._profiles[sv] for sv in stat_vars}


def _profile_of(stat_var: str, by_place: dict[str, list[Facet]]):
    profile = _profile(stat_var, subject_probe(by_place, list(by_place)))
    return replace(profile, places_by_facet=_CountingPlaces(profile.places_by_facet))


def _variable_breadths(
    profiles: Mapping[str, Any], coverage: CoverageIndex | None = None
) -> dict[str, Any]:
    reader = ProfileBreadth(
        profiles=_Profiles(profiles),
        coverage=FakeCoverageService(coverage or build_coverage_index([], role_names={})).snapshot,
    )
    return reader(list(profiles), deadline=None)


def test_a_group_join_counts_each_members_places_once() -> None:
    """One member's estimates continue its reported series; the other's end before its
    reported series does. Merged, the group's estimates continue its reported series, and
    each member counts the places either facet holds, reading each facet's places once."""
    first = [_dated("fRec", None, 2010, 2020), _dated("fEst", "Estimated", 2021, 2025)]
    second = [_dated("fRec", None, 2010, 2024), _dated("fEst", "Estimated", 2015, 2023)]
    profiles = {
        "sv/First": _profile_of("sv/First", {"country/KEN": first}),
        "sv/Second": _profile_of("sv/Second", {"country/KEN": second}),
    }
    breadths = _variable_breadths(profiles)
    merged = merge_facets_by_series({"first": first, "second": second}, ["first", "second"])

    ranked = rank_candidates(
        merged,
        window=_JOIN_TO_2030,
        breadth=None,
        variable_breadth=summed_counts(list(breadths.values())),
    )

    assert [f.facet_id for f in ranked] == ["fRec", "fEst"]
    assert (
        profiles["sv/First"].places_by_facet.reads,
        profiles["sv/Second"].places_by_facet.reads,
    ) == (2, 2)


# At Kenya the reported series ends in 2024 and the estimates continue it to 2030. Ghana, or
# France's pairs, hold only the reported series and Togo, or Germany's, only the estimates,
# so each facet holds two places or pairs and the two together three.
_KENYA = [_dated("fRec", None, 2010, 2024), _dated("fEst", "Estimated", 2025, 2030)]


def _kenya_profile(path: str):
    if path == "subject":
        by_place = {
            "country/KEN": _KENYA,
            "country/GHA": _KENYA[:1],
            "country/TGO": _KENYA[1:],
        }
        return _profile(_SV, subject_probe(by_place, list(by_place))), None
    coverage = build_coverage_index(
        [
            (_SV, "fRec", "country/NOR", "country/KEN", list(range(2010, 2025))),
            (_SV, "fEst", "country/NOR", "country/KEN", list(range(2025, 2031))),
            (_SV, "fRec", "country/FRA", "country/KEN", list(range(2010, 2025))),
            (_SV, "fEst", "country/DEU", "country/KEN", list(range(2025, 2031))),
        ],
        role_names=_ROLE_NAMES,
        facet_metadata={
            "fRec": FacetMetadata(provenance_id="dc/base/Rec", unit="CurrencyUSD"),
            "fEst": FacetMetadata(
                provenance_id="dc/base/Rec", measurement_method="Estimated", unit="CurrencyUSD"
            ),
        },
    )
    return _profile(_SV, coverage_probe(stat_var=_SV, bound={}, coverage=coverage)), coverage


@pytest.mark.parametrize("path", ["subject", "role"])
@pytest.mark.parametrize(
    ("window", "reads"),
    [
        pytest.param(_JOIN_TO_2030, [frozenset({"fRec", "fEst"})], id="join-reads-it-once"),
        pytest.param(RequestedWindow(kind="all"), [], id="no-join-reads-none"),
    ],
)
def test_a_profile_reads_no_union_and_a_ranking_reads_one_per_join(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    window: RequestedWindow,
    reads: list[frozenset[str]],
) -> None:
    """Building a profile counts no union. A ranking at Kenya that joins the reported series
    and its estimates counts the places, or pairs, either holds once over two rankings, and a
    ranking that forms no join counts none."""
    read: list[frozenset[str]] = []
    union = SeriesCounts.union

    def counting(self: SeriesCounts, facet_ids: frozenset[str]) -> int:
        if facet_ids not in self._unions:
            read.append(facet_ids)
        return union(self, facet_ids)

    monkeypatch.setattr(SeriesCounts, "union", counting)
    profile, coverage = _kenya_profile(path)

    assert read == []
    variable_breadth = _variable_breadths({_SV: profile}, coverage)[_SV]
    for _ in range(2):
        ranked = rank_candidates(
            _KENYA, window=window, breadth=None, variable_breadth=variable_breadth
        )

    assert [f.facet_id for f in ranked] == ["fRec", "fEst"]
    assert read == reads
    assert variable_breadth.union(frozenset({"fRec", "fEst"})) == 3
