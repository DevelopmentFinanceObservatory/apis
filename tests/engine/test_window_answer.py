"""The window answer on every route: a population holding data at the place with no point in
the years read answers `no_observations`, and its part departs as `year_unavailable`, naming
the years read and the span the series runs (`read_plan.window_gap`,
`engine.departures.window_gap_fact`). Engine tests
pin today to 2026, so an open end reads to 2026."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import search.engine.region.related_read as related_read_module
import search.engine.region.trailing as trailing_module
import search.scope_window
from search.engine.binders import Binder
from search.engine.clients.graph import Facet
from search.engine.conjoin import _build_part
from search.engine.departures import DepartureFact
from search.engine.extract import DateRequest, dates_to_request
from search.engine.quantity_read import QuantityOutput
from search.engine.region import phase_answer, resolve_variable
from search.engine.region.facets import FacetAdmission, FilteredAnswer
from search.engine.region.result import RegionResult
from search.engine.region.trailing import settle_trailing
from search.engine.retrieval.read_plan import PairPlan, PlanSeries, plan_standard
from search.engine.retrieval.retrieve import Materialised, MemberPopulation
from search.engine.retrieval.role_resolve import _zero_observation_carrier
from search.engine.retrieval.standard import _no_data_in_years
from search.engine.time_bind import parse_time_text
from search.engine.unit.conversion import ConversionPeriod
from search.models import CoverageBare, Part, TimeWindow
from search.render.plan import render_sentence
from search.scope_window import ScopeWindowSettings
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine import test_place_scope_default as place_scope_default
from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
from tests.engine._binders import ReplyBinder, elected_reply
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
    binder: Binder | None = None,
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
            binder=binder,
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
        binder=ReplyBinder(),
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
_OTHER_CREDITOR = "country/CRB"
_FLOW_FACETS = {"f1": ("dc/base/Ids", None), "f3": ("dc/base/Ids", "Estimated")}


def _role_answer(
    monkeypatch,
    search_settings,
    held: dict[str, range],
    date_request: DateRequest,
    *,
    bulk: bool,
    discovered: bool,
    other_creditor: range = range(0),
) -> RegionResult:
    """Resolve a creditor's flows to Aland, named or discovered, per pair or bulk, over
    coverage rows holding each facet in `held` for its years, and a second creditor's
    flows under `f1` in `other_creditor`."""
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    rows = [(_FLOW, fid, _CREDITOR, _ALAND, list(years)) for fid, years in held.items()]
    if other_creditor:
        rows.append((_FLOW, "f1", _OTHER_CREDITOR, _ALAND, list(other_creditor)))
    index = patch_coverage(
        monkeypatch,
        rows,
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


def _trailing(text: str) -> DateRequest:
    """The request a span counted back from 2026 reads as (`time_bind`, `dates_to_request`)."""
    dates, _warning = parse_time_text(text, current_year=2026)
    request = dates_to_request(dates)
    assert request is not None and request.trailing_years is not None
    return request


def _trailing_answer(monkeypatch, search_settings, route: str, years: range, text: str):
    """Resolve `text` over a series holding `years`, on the standard route or a named
    creditor's flows per pair or in bulk."""
    if route == "standard":
        return _standard_answer([_facet("fA", years)], _trailing(text))
    return _role_answer(
        monkeypatch,
        search_settings,
        {"f1": years},
        _trailing(text),
        bulk=route == "named_bulk",
        discovered=False,
    )


def _read_years(result: RegionResult) -> tuple[int | None, int | None]:
    """The range the answer's Spec states and the years its StatVar serves."""
    window = result.specs[0].definition.identity.requested_window
    assert window.kind == "range"
    served = result.specs[0].resolved.members[0].data_date_range
    assert served is not None and served.start is not None and served.end is not None
    assert (int(served.start[:4]), int(served.end[:4])) == (window.start_year, window.end_year)
    return window.start_year, window.end_year


@pytest.mark.parametrize("route", ["standard", "named", "named_bulk"])
@pytest.mark.parametrize(
    ("text", "years"),
    [
        ("the last five years", (2020, 2024)),
        ("the past year", (2024, 2024)),
        ("the last 12 months", (2024, 2024)),
    ],
)
def test_a_trailing_span_reads_the_newest_years_the_data_holds(
    monkeypatch, search_settings, route: str, text: str, years: tuple[int, int]
):
    """A series ending in 2024 answers "the last five years" with 2020 to 2024, and "the
    past year" and "the last 12 months" with 2024: the Spec states the years read as a plain
    range its sentence names, and the part departs nothing."""
    result = _trailing_answer(monkeypatch, search_settings, route, range(2000, 2025), text)
    assert result.status == "definite"
    assert _read_years(result) == years
    first, last = years
    named = f" in {last}." if first == last else f", {first} to {last}."
    assert named in render_sentence(result.specs[0])
    assert _part(result).departures == []


@pytest.mark.parametrize("route", ["standard", "named", "named_bulk"])
def test_a_stale_series_answers_a_trailing_span_with_a_departure_naming_its_newest_year(
    monkeypatch, search_settings, route: str
):
    """A series ending in 2022, four years before 2026, answers "the last five years" with
    its newest five years and departs `year_unavailable`, naming 2022."""
    result = _trailing_answer(
        monkeypatch, search_settings, route, range(2000, 2023), "the last five years"
    )
    assert result.status == "definite"
    assert _read_years(result) == (2018, 2022)
    departures = _part(result).departures
    assert [(d.code, d.binding_ref, d.asked) for d in departures] == [
        ("year_unavailable", "when:", "2022 to 2026")
    ]
    assert "ends in 2022" in departures[0].text


@pytest.mark.parametrize("route", ["standard", "named", "named_bulk"])
@pytest.mark.parametrize("last", [2010, 2016, 2018])
def test_a_series_that_ended_long_ago_holds_no_trailing_span(
    monkeypatch, search_settings, route: str, last: int
):
    """A series whose newest year lies before 2019 holds no point in "the last five years" or
    the three years before them, so the answer is the window answer, naming the years the
    words cover. One ending in 2016 or 2018 lies inside the wider years the request reads to
    confirm its members, and answers as one ending in 2010 does."""
    result = _trailing_answer(
        monkeypatch, search_settings, route, range(2000, last + 1), "the last five years"
    )
    _assert_window_answer(result, "2022 to 2026", (2000, last))


@pytest.mark.parametrize("route", ["named", "named_bulk"])
def test_the_year_cap_counts_the_years_a_trailing_span_asks_for(
    monkeypatch, search_settings, route: str
):
    """Under a five-year cap, "the last five years" reads eight years before the answer
    settles and answers its five newest."""
    monkeypatch.setattr(
        search.scope_window, "_SETTINGS", ScopeWindowSettings(scope_max_window_years=5)
    )
    result = _trailing_answer(
        monkeypatch, search_settings, route, range(2000, 2025), "the last five years"
    )
    assert result.status == "definite"
    assert _read_years(result) == (2020, 2024)


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_stale_trailing_answer_on_discovered_creditors_serves_what_its_range_serves(
    monkeypatch, search_settings, bulk: bool
):
    """Creditor A's flows to Aland run 2000 to 2022 and creditor B's hold 2018 alone.
    "The last five years" in 2026 confirms creditors over 2015 to 2026, every span ending in
    2019 to 2026, so the answer settles on 2018 to 2022 serving both. The plain range its
    Spec states resolves to the same Spec, with the same creditors, series and rows."""
    trailing = _role_answer(
        monkeypatch,
        search_settings,
        {"f1": range(2000, 2023)},
        _trailing("the last five years"),
        bulk=bulk,
        discovered=True,
        other_creditor=range(2018, 2019),
    )
    window = trailing.specs[0].definition.identity.requested_window
    plain = _role_answer(
        monkeypatch,
        search_settings,
        {"f1": range(2000, 2023)},
        DateRequest(
            window=TimeWindow(start_year=window.start_year, end_year=window.end_year),
            latest=False,
        ),
        bulk=bulk,
        discovered=True,
        other_creditor=range(2018, 2019),
    )

    assert trailing.specs[0] == plain.specs[0]
    assert (window.start_year, window.end_year) == (2018, 2022)
    assert trailing.specs[0].resolved.download is not None
    assert trailing.specs[0].resolved.download.series == 2
    assert [(d.code, d.asked, d.shown) for d in _part(trailing).departures] == [
        ("year_unavailable", "2022 to 2026", "2018 to 2022")
    ]


@pytest.mark.parametrize("bulk", [False, True], ids=["named", "named_bulk"])
def test_a_trailing_span_reaching_below_the_role_read_floor_starts_at_it(
    monkeypatch, search_settings, bulk: bool
):
    """A creditor's flows to Aland run 1961 to 1995. "The last 40 years" in 2026 settles on
    1995, whose 40 years reach back to 1956, below the 1960 a role read starts from, so the
    answer reads 1960 to 1995: the range a resubmit of its Spec reads, which resolves to the
    same Spec."""
    trailing = _role_answer(
        monkeypatch,
        search_settings,
        {"f1": range(1961, 1996)},
        _trailing("the last 40 years"),
        bulk=bulk,
        discovered=False,
    )
    assert trailing.status == "definite", (trailing.status, trailing.no_data_reason)
    window = trailing.specs[0].definition.identity.requested_window
    assert (window.start_year, window.end_year) == (1960, 1995)
    plain = _role_answer(
        monkeypatch,
        search_settings,
        {"f1": range(1961, 1996)},
        DateRequest(window=TimeWindow(start_year=1960, end_year=1995), latest=False),
        bulk=bulk,
        discovered=False,
    )
    assert trailing.specs[0] == plain.specs[0]


def test_a_trailing_answer_reads_its_related_measures_over_its_settled_years(monkeypatch):
    """The answer's related measures are read once, over the five years it settles on, and
    never over the wider years it read before settling."""
    windows: list[object] = []
    read = related_read_module.read_related

    def recording(**kwargs):
        windows.append(kwargs["window"])
        return read(**kwargs)

    monkeypatch.setattr(related_read_module, "read_related", recording)
    result = _standard_answer([_facet("fA", range(2000, 2025))], _trailing("the last five years"))

    assert result.status == "definite"
    assert windows == [TimeWindow(start_year=2020, end_year=2024)]


_FLOW_B = "TEST/Flow_B"


def _yearly(years: range) -> Facet:
    return Facet(
        obs_count=len(years),
        earliest_date=str(years[0]),
        latest_date=str(years[-1]),
        facet_id="f1",
        provenance_id="dc/base/Ids",
        unit="CurrencyEUR",
        dates=tuple(str(y) for y in years),
    )


def test_a_trailing_role_answer_keeps_a_member_its_unit_admission_empties():
    """Two flows from a creditor to Aland, each held 2000 to 2024. Unit admission leaves the
    second flow out of the answered plan. Settling "the last five years" on 2020 to 2024
    keeps both flows and counts both, as a role read of 2020 to 2024 confirms both before
    admission and lists them."""
    roles = (("creditor", _CREDITOR), ("debtor", _ALAND))
    plan = PairPlan(
        members=(_FLOW, _FLOW_B),
        years=TimeWindow(start_year=2000, end_year=2026),
        series=tuple(
            PlanSeries(
                stat_var=sv,
                role_tuple=roles,
                candidates=(_yearly(range(2000, 2025)),),
                group=(_yearly(range(2000, 2025)),),
            )
            for sv in (_FLOW, _FLOW_B)
        ),
    )
    mat = Materialised(
        sv_dcids=[_FLOW, _FLOW_B],
        has_data=True,
        coverage=CoverageBare(has_data=True),
        read_plan=plan,
        population=MemberPopulation(
            admitted=2, admitted_dcids=(_FLOW, _FLOW_B), confirmed=2, exact=True
        ),
    )
    # As `facets.filter_answer` narrows a plan to a unit elected for the first flow alone.
    admitted = plan.narrowed(lambda sv, _facet: sv == _FLOW).leaving_out({_FLOW_B})
    filtered = FilteredAnswer(mat.coverage, None, admitted)
    request = _trailing("the last five years")

    settled = settle_trailing(
        mat, filtered, date_request=request, coverage_index=empty_coverage_snapshot().index
    )

    assert settled is not None
    assert settled.date_request.window == TimeWindow(start_year=2020, end_year=2024)
    assert settled.mat.sv_dcids == [_FLOW, _FLOW_B]
    assert settled.mat.population is not None and settled.mat.population.confirmed == 2


def _two_flows(years: range) -> Materialised:
    """Two flows from a creditor to Aland, each held in `years`, read over 2000 to 2026."""
    roles = (("creditor", _CREDITOR), ("debtor", _ALAND))
    plan = PairPlan(
        members=(_FLOW, _FLOW_B),
        years=TimeWindow(start_year=2000, end_year=2026),
        series=tuple(
            PlanSeries(
                stat_var=sv, role_tuple=roles, candidates=(_yearly(years),), group=(_yearly(years),)
            )
            for sv in (_FLOW, _FLOW_B)
        ),
    )
    return Materialised(
        sv_dcids=[_FLOW, _FLOW_B],
        has_data=True,
        coverage=CoverageBare(has_data=True),
        read_plan=plan,
        population=MemberPopulation(
            admitted=2, admitted_dcids=(_FLOW, _FLOW_B), confirmed=2, exact=True
        ),
    )


@pytest.mark.parametrize(
    ("held", "deflators_end", "shown", "limited_by"),
    [
        (range(2000, 2025), 2023, "2020 to 2023", "deflators"),
        (range(2000, 2022), 2020, "2017 to 2020", "deflators"),
        (range(2000, 2022), None, "2017 to 2021", None),
    ],
    ids=["converted", "stale_and_converted", "stale"],
)
def test_a_trailing_answer_states_the_years_its_words_cover_once(
    monkeypatch, held: range, deflators_end: int | None, shown: str, limited_by: str | None
):
    """ "The last five years" in constant dollars, in 2026, settles on the newest five years
    the flows hold. Deflators ending before them narrow the answer again, and its one year
    fact names 2022 to 2026, the years the words cover, as asked: for flows to 2024 and
    deflators to 2023, and for stale flows to 2021 and deflators to 2020, whose fact the
    conversion's replaces. Stale flows the deflators cover keep the stale fact alone."""
    monkeypatch.setattr(trailing_module, "current_year", lambda: 2026)
    mat = _two_flows(held)
    assert mat.read_plan is not None
    request = _trailing("the last five years")
    settled = settle_trailing(
        mat,
        FilteredAnswer(mat.coverage, None, mat.read_plan),
        date_request=request,
        coverage_index=empty_coverage_snapshot().index,
    )
    assert settled is not None
    first, last = settled.date_request.window.start_year, settled.date_request.window.end_year
    assert first is not None and last is not None
    period = (
        None
        if deflators_end is None
        else ConversionPeriod(
            served=(first, last), years=(first, deflators_end), limited_by="deflators"
        )
    )
    monkeypatch.setattr(phase_answer, "current_conversion_catalogue", lambda: None)
    monkeypatch.setattr(phase_answer, "survivor_declared_roles", lambda _mat: None)
    monkeypatch.setattr(phase_answer, "unit_request_period", lambda *_a, **_k: period)
    monkeypatch.setattr(phase_answer, "deflator_omission_departure", lambda *_a, **_k: None)
    facts: list[DepartureFact] = []
    state = SimpleNamespace(
        unit_request=object(),
        unit_elected_selections={},
        date_request=settled.date_request,
        mat_result=settled.mat,
        coverage=SimpleNamespace(index=empty_coverage_snapshot().index),
        depart=lambda _warning, fact: facts.append(fact),
    )

    phase_answer._over_conversion_period(
        state, settled.mat, settled.filtered, FacetAdmission(), settled=settled
    )

    assert [(f.code, f.requested_years, f.shown, f.limited_by) for f in facts] == [
        ("year_unavailable", (2022, 2026), shown, limited_by)
    ]
