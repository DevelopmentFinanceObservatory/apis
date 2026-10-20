"""The window answer on every route: a population holding data at the place with no point in
the years read answers `no_observations`, and its part departs as `year_unavailable`, naming
the years read and the span the series runs (`read_plan.window_gap`,
`engine.departures.window_gap_fact`). Engine tests
pin today to 2026, so an open end reads to 2026."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.clients.graph import Facet
from search.engine.conjoin import _build_part
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.retrieval.read_plan import plan_standard
from search.engine.retrieval.retrieve import MemberPopulation
from search.engine.retrieval.role_resolve import _zero_observation_carrier
from search.engine.retrieval.standard import _no_data_in_years
from search.models import Part, TimeWindow
from search.render.plan import render_sentence
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine import test_place_scope_default as place_scope_default
from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_PERSON = "Count_Person"
_HOUSEHOLD = "Count_Household"
_ZETA = "country/ZZZ"


def _since(year: int) -> DateRequest:
    return DateRequest(window=TimeWindow(start_year=year, end_year=None), latest=False)


def _between(start: int, end: int) -> DateRequest:
    return DateRequest(window=TimeWindow(start_year=start, end_year=end), latest=False)


def _part(result: RegionResult) -> Part:
    return _build_part(
        result,
        representative=result,
        folded=False,
        max_candidates=5,
        include_observations=False,
        include_skills=False,
        skills_index=None,
        envelope_warnings=[],
    )


def _sentence(result: RegionResult) -> str:
    sentence = _part(result).rendered_sentence
    assert sentence is not None
    return sentence


def _assert_window_answer(result: RegionResult, asked: str, data_years: tuple[int, int]) -> None:
    """The answer is `no_observations` for the window, and its part's one departure is the
    `year_unavailable` fact naming the years asked and the span the series runs."""
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.gate is None
    fact = result.window_gap_fact
    assert fact is not None
    assert (fact.code, fact.asked, fact.data_years) == ("year_unavailable", asked, data_years)
    part = _part(result)
    assert [(d.code, d.binding_ref, d.asked) for d in part.departures] == [
        ("year_unavailable", "when:", asked)
    ]
    assert all(token.kind != "note" for token in part.sentence_plan or [])


class _BindLLM:
    """Elects shape 0 with no bindings, naming `also` as also plausible."""

    def __init__(self, also: list[int] | None = None) -> None:
        self.also = also

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return elected_reply(0, also_plausible=self.also), None


def _count_arcs(population_type: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": population_type}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


def _facet(facet_id: str, years: range, method: str | None = None) -> dict:
    facet = {
        "facetId": facet_id,
        "provenanceId": "dc/base/WDI",
        "earliestDate": str(years[0]),
        "latestDate": str(years[-1]),
        "obsCount": len(years),
        "dates": [str(y) for y in years],
        "values": [1.0] * len(years),
    }
    if method:
        facet["measurementMethod"] = method
    return facet


def _standard_answer(
    person: list[dict],
    date_request: DateRequest,
    *,
    household: list[dict] | None = None,
    also: list[int] | None = None,
) -> RegionResult:
    """Resolve "population of zeta" over the Count_Person shape, and the Count_Household
    shape too when `household` is given."""
    nodes = {
        _PERSON: {
            "label": "Population",
            "type": "StatisticalVariable",
            "arcs": _count_arcs("Person"),
        },
        _ZETA: {"label": "Zeta", "type": "Country"},
    }
    obs = {f"{_PERSON}|{_ZETA}": person}
    svs = [_PERSON]
    if household is not None:
        nodes[_HOUSEHOLD] = {
            "label": "Households",
            "type": "StatisticalVariable",
            "arcs": _count_arcs("Household"),
        }
        obs[f"{_HOUSEHOLD}|{_ZETA}"] = household
        svs.append(_HOUSEHOLD)
    graph = FakeGraph(
        nodes=nodes, obs=obs, detect={"population": {"svs": svs}}, resolve={"Zeta": _ZETA}
    )
    return asyncio.run(
        resolve_variable(
            "population of zeta",
            entities=["Zeta"],
            date_request=date_request,
            detect_query="population",
            role_query="population of zeta",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(also),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )


@pytest.mark.parametrize(
    ("date_request", "years"),
    [(_since(2020), "2020 to 2026"), (_between(2020, 2025), "2020 to 2025")],
    ids=["since", "closed"],
)
def test_standard_years_inside_the_span_name_the_span(date_request: DateRequest, years: str):
    """Recorded to 2019 and projected from 2028, the place holds no point in the years read.
    The note names the span of the series a dateless answer serves, the recorded facet's
    2010 to 2019, which outranks the projection."""
    projected = [_facet("fA", range(2010, 2020)), _facet("fC", range(2028, 2031), "Estimated")]
    result = _standard_answer(projected, date_request)
    _assert_window_answer(result, years, (2010, 2019))
    dateless = _standard_answer(projected, None)
    assert ", 2010 to 2019" in render_sentence(dateless.specs[0])


@pytest.mark.parametrize("also", [None, [1]], ids=["tie", "also_plausible"])
def test_a_second_shape_with_no_data_leaves_the_window_answer_standing(also: list[int] | None):
    """Two shapes recalled: Households, elected, holds 2000 to 2004 at Zeta and Population
    holds nothing there, so no alternate survives and the elected shape answers for the
    window."""
    result = _standard_answer(
        [], _since(2025), household=[_facet("fH", range(2000, 2005))], also=also
    )
    _assert_window_answer(result, "2025 to 2026", (2000, 2004))


def test_a_window_answer_showing_an_alternate_with_data_states_no_year_departure():
    """Households, elected, holds 2000 to 2004 at Zeta and Population, nominated, holds 2025
    and 2026: the answer shows Population as the reading riding on the refusal, which holds
    the years asked, so the part carries no `year_unavailable` departure for the elected
    series."""
    result = _standard_answer(
        [_facet("fP", range(2025, 2027))],
        _since(2025),
        household=[_facet("fH", range(2000, 2005))],
        also=[1],
    )
    assert result.no_data_reason == "no_observations"
    assert result.gate == "only_alternate_has_data"
    assert result.window_gap_fact is not None
    part = _part(result)
    assert [d.code for d in part.departures] == []
    assert _sentence(result) == (
        'No data for "population of zeta". Data exists for Zeta\'s population, 2025 to 2026. '
        "Check that it measures what you asked for."
    )


_FLOW = role_bearing._SV_DCID
_ALAND = "country/AAA"
_CREDITOR = "country/CRA"
_FLOW_FACETS = {"f1": ("dc/base/Ids", None), "f3": ("dc/base/Ids", "Estimated")}


def _role_answer(
    monkeypatch,
    search_settings,
    held: dict[str, range],
    date_request: DateRequest,
    *,
    bulk: bool,
    discovered: bool,
) -> RegionResult:
    """Resolve a creditor's flows to Aland, named or discovered, per pair or bulk, over
    coverage rows holding each facet in `held` for its years."""
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    index = patch_coverage(
        monkeypatch,
        [(_FLOW, fid, _CREDITOR, _ALAND, list(years)) for fid, years in held.items()],
        role_names={_FLOW: RoleNames(donor="creditor", recipient="debtor")},
        facet_metadata={
            fid: FacetMetadata(provenance_id=p, measurement_method=m)
            for fid, (p, m) in _FLOW_FACETS.items()
        },
    )
    return role_bearing._run(
        entities=[] if discovered else ["Aland"],
        sv_dcid=_FLOW,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Aland": _ALAND},
        sv_score=0.9,
        role_query="debt flows" if discovered else "debt flows to Aland",
        sdmx=FakeSdmxClient(observations={}, availability={}),
        provenance_labels={"dc/base/Ids": "IDS"},
        date_request=date_request,
        coverage=FakeCoverageService(index).snapshot(),
    )


@pytest.mark.parametrize(
    ("date_request", "years"),
    [(_since(2021), "2021 to 2026"), (_between(2021, 2025), "2021 to 2025")],
    ids=["since", "closed"],
)
@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_role_years_inside_the_span_name_the_span(
    monkeypatch, search_settings, date_request: DateRequest, years: str, bulk, discovered
):
    """A creditor's flows to Aland are recorded 2010 to 2020 and projected 2028 to 2030.
    Named or discovered, per pair or bulk, the years read hold no point. The note names the
    span of the series a dateless answer serves, the recorded facet's 2010 to 2020."""
    held = {"f1": range(2010, 2021), "f3": range(2028, 2031)}
    result = _role_answer(
        monkeypatch, search_settings, held, date_request, bulk=bulk, discovered=discovered
    )
    _assert_window_answer(result, years, (2010, 2020))
    dateless = _role_answer(
        monkeypatch, search_settings, held, None, bulk=bulk, discovered=discovered
    )
    assert ", 2010 to 2020" in render_sentence(dateless.specs[0])


_BLAND = "country/BBB"
_ALPHA, _BETA = "dc/base/Alphaware", "dc/base/Betaware"
# What Aland and Bland hold from two sources, as (source, years) per place.
_TWO_PLACE_HOLDINGS = {
    "overlap": {
        _ALAND: [(_ALPHA, range(2010, 2021)), (_BETA, range(2015, 2025))],
        _BLAND: [(_ALPHA, range(2012, 2019)), (_BETA, range(2016, 2025))],
    },
    "split": {
        _ALAND: [(_ALPHA, range(2005, 2023)), (_BETA, range(2019, 2022))],
        _BLAND: [(_BETA, range(2000, 2024)), (_ALPHA, range(2018, 2021))],
    },
    "one_each": {_ALAND: [(_ALPHA, range(2010, 2021))], _BLAND: [(_BETA, range(2000, 2016))]},
}


def _two_place_answer(
    monkeypatch, search_settings, holdings: str, date_request: DateRequest | None, *, route: str
) -> RegionResult:
    """Aland and Bland on the standard route, or as a creditor's debtors, named or
    discovered, each read per pair."""
    held = _TWO_PLACE_HOLDINGS[holdings]
    if route == "standard":
        graph = FakeGraph(
            nodes={
                _PERSON: {
                    "label": "Population",
                    "type": "StatisticalVariable",
                    "arcs": _count_arcs("Person"),
                },
                _ALAND: {"label": "Aland", "type": "Country"},
                _BLAND: {"label": "Bland", "type": "Country"},
                _ALPHA: {"label": "Alphaware"},
                _BETA: {"label": "Betaware"},
            },
            obs={
                f"{_PERSON}|{place}": [
                    {**_facet(source, years), "provenanceId": source} for source, years in facets
                ]
                for place, facets in held.items()
            },
            detect={"population": {"svs": [_PERSON]}},
            resolve={"Aland": _ALAND, "Bland": _BLAND},
        )
        return asyncio.run(
            resolve_variable(
                "population of Aland and Bland",
                entities=["Aland", "Bland"],
                date_request=date_request,
                detect_query="population",
                role_query="population of Aland and Bland",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_BindLLM(),
                base_timing={},
                sdmx=FakeSdmxClient(
                    observations={}, availability={}, scope_availability=graph.scope_availability
                ),
                coverage=empty_coverage_snapshot(),
            )
        )
    search_settings(obs_preview_max_pairs=120)
    index = patch_coverage(
        monkeypatch,
        [
            (_FLOW, source, _CREDITOR, place, list(years))
            for place, facets in held.items()
            for source, years in facets
        ],
        role_names={_FLOW: RoleNames(donor="creditor", recipient="debtor")},
        facet_metadata={source: FacetMetadata(provenance_id=source) for source in (_ALPHA, _BETA)},
    )
    discovered = route == "discovered"
    return role_bearing._run(
        entities=[] if discovered else ["Aland", "Bland"],
        sv_dcid=_FLOW,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Aland": _ALAND, "Bland": _BLAND},
        sv_score=0.9,
        role_query="debt flows" if discovered else "debt flows to Aland and Bland",
        sdmx=FakeSdmxClient(observations={}, availability={}),
        provenance_labels={_ALPHA: "Alphaware", _BETA: "Betaware"},
        date_request=date_request,
        coverage=FakeCoverageService(index).snapshot(),
    )


@pytest.mark.parametrize("holdings", list(_TWO_PLACE_HOLDINGS))
@pytest.mark.parametrize("route", ["standard", "named", "discovered"])
def test_at_two_places_the_window_note_names_the_span_the_dateless_answer_states(
    monkeypatch, search_settings, route: str, holdings: str
):
    """Whatever two places hold from two sources, on every route, a window past all their
    data answers `no_observations` naming the span the same answer states with the window
    removed: the years of the one source the variable serves across both places."""
    dateless = _two_place_answer(monkeypatch, search_settings, holdings, None, route=route)
    assert dateless.status == "definite", dateless.no_data_reason
    (stat_var,) = dateless.specs[0].resolved.members
    assert stat_var.data_date_range is not None
    start, end = stat_var.data_date_range.start, stat_var.data_date_range.end

    windowed = _two_place_answer(monkeypatch, search_settings, holdings, _since(2028), route=route)

    _assert_window_answer(windowed, "2028 or later", (int(start), int(end)))


# Rows to 2030 give the coverage horizon two mask words, 1960 to 2087, as production sizes it.
_ASKED_PAST_THE_HORIZON = {
    "since_2040": (_since(2040), "2040 or later"),
    "until_2005": (DateRequest(window=TimeWindow(end_year=2005), latest=False), "2005 or earlier"),
    "in_1950": (_between(1950, 1950), "1950"),
    "in_2090": (_between(2090, 2090), "2090"),
}


@pytest.mark.parametrize("asked", list(_ASKED_PAST_THE_HORIZON))
@pytest.mark.parametrize("route", ["standard", "named", "discovered", "named_bulk"])
def test_every_route_names_the_years_asked_not_the_coverage_horizon(
    monkeypatch, search_settings, asked: str, route: str
):
    """A series running 2010 to 2030: the note names the years asked, open bounds open, on
    every route. A role read's years are floored and capped at the coverage horizon, which
    the note never states, and years wholly outside it still answer for the window."""
    date_request, years = _ASKED_PAST_THE_HORIZON[asked]
    if route == "standard":
        result = _standard_answer([_facet("fA", range(2010, 2031))], date_request)
    else:
        result = _role_answer(
            monkeypatch,
            search_settings,
            {"f1": range(2010, 2031)},
            date_request,
            bulk=route == "named_bulk",
            discovered=route == "discovered",
        )
    _assert_window_answer(result, years, (2010, 2030))


_LATEST = DateRequest(window=None, latest=True)


@pytest.mark.parametrize("date_request", [_LATEST, None], ids=["latest", "dateless"])
def test_held_data_under_a_latest_or_dateless_read_makes_no_window_answer(
    date_request: DateRequest | None,
):
    """Only a range leaves data outside the years read. Handed held facets under a latest or
    every-date read, each carrier is the flagged-unavailable one, with no gap to state."""
    held = [Facet(obs_count=3, earliest_date="2010", latest_date="2012")]
    role_carrier = _zero_observation_carrier(
        surviving_svs=[_FLOW],
        member_facts={},
        bound_values={},
        date_request=date_request,
        open_origin_dcids=None,
        roles_unverified=False,
        bindings_admitted=1,
        bindings_admitted_dcids=(_FLOW,),
        member_population_exact=True,
        held_outside_years={_FLOW: held},
        dateless_plan=None,
        open_axes=frozenset(),
    )
    standard_carrier = _no_data_in_years(
        (_PERSON, held, None),
        dateless_plan=plan_standard({}, members=[_PERSON], years=None),
        date_request=date_request,
        default_scope_probe=False,
        member_facts={},
        population=MemberPopulation(admitted=1, admitted_dcids=(_PERSON,), confirmed=1, exact=True),
        family_total_absent=None,
    )
    for carrier in (role_carrier, standard_carrier):
        assert carrier.available_at_place is False
        assert carrier.window_gap is None


def test_a_scope_defaulted_answer_names_the_span_its_places_run():
    """With no place named, the scope defaults to every country. Two countries hold data to
    2019 and 2015, so 2022 to 2024 holds no point at any of them."""

    def facet(years: range) -> dict:
        return {
            "facetId": "fW",
            "provenanceId": place_scope_default._OBS_PROVENANCE,
            "earliestDate": str(years[0]),
            "latestDate": str(years[-1]),
            "obsCount": len(years),
            "dates": [str(y) for y in years],
            "values": [1.0] * len(years),
        }

    result = place_scope_default._run(
        graph=place_scope_default._make_graph(
            obs={
                f"{place_scope_default._SV}|country/A1": [facet(range(2010, 2020))],
                f"{place_scope_default._SV}|country/A2": [facet(range(2010, 2016))],
            }
        ),
        date_request=_between(2022, 2024),
    )

    _assert_window_answer(result, "2022 to 2024", (2010, 2019))
