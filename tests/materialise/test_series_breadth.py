"""One facet group serves a member at every place or role tuple its request binds, so every
election leads with the series each facet holds a point at in the window
(`materialise.facets.rank_candidates`' `breadth`): a mint's probe counts them
(`MemberProbe.breadth`), and member, unit-request and group selection rank by them.
"""

from __future__ import annotations

import itertools

import pytest

from materialise.facets import (
    FacetBinding,
    UnitRequest,
    coverage_probe,
    probe_facets,
    rank_candidates,
    series_breadth,
)
from materialise.generation import _select_member_facets
from materialise.graph import Facet
from materialise.group_facets import group_breadth, select_group_facets
from materialise.pipeline import MaterialisationRequest
from materialise.window import RequestedWindow
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

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
