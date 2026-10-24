"""Search reads a UN candidate's facts from the catalogue's UN tables: derivation sends the UN
deployment nothing, a shape holds the variables recall found, and admission follows the
tables."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from concurrent.futures import wait

import httpx
import pytest

from materialise import place_scope
from origin_catalogue import BASE, DFO, UN_SYSTEM
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import DetectedSV, DetectLegs, admit_legs
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.config import settings
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import PLACE_SCOPE_DEFAULTED
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import read_constraints, read_five_tuple
from search.engine.retrieval.standard import _admit_standard_members, _AdmittedStandardMembers
from search.engine.stated_words_read import StatedWordsOutput
from search.models import TimeWindow
from statvar_members import member_digest
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_OURS_HOST = "ours.example.org"
_UN_HOST = "un.example.org"

_CRS = "provenance/OECDCreditorReportingSystem"
_WDI = "dc/base/WorldDevelopmentIndicators"
_BASE_MORTALITY = "Count_Death_Upto5Years_AsFractionOf_Count_BirthEvent_LiveBirth"
_BASE_SDG = "sdg/SH_DYN_MORT"

_SERIES = "undata/who/SERIES-MORT_U5_RATE"
_HEADLINE = "undata/who/MORT_U5_RATE"
_FEMALE = "undata/who/MORT_U5_RATE.SEX--F"
_SEX = "undataSex"
_TOPIC = "undata/topic/who/MORT_U5_RATE"
_AID = "undata/who/ODA_1_3"
_NOT_IN_THE_BUILD = "undata/who/NOT_IN_THE_BUILD"


def _catalogue():
    """The UN tables hold a WHO series of two variables, the headline and its by-sex variant,
    and a topic reaching both, and exclude `_AID` as superseded by the CRS. `_BASE_SDG` holds
    data only under a base provenance the registry excludes."""
    return restrictive_catalogue(
        provenances=[
            provenance_row(_CRS, DFO),
            provenance_row(_WDI, BASE),
            provenance_row("undata/p/WHO", UN_SYSTEM, segment="who"),
        ],
        base_variables={_BASE_MORTALITY: ([_WDI], [])},
        withheld_variables=[_BASE_SDG],
        un_mirror=un_mirror(
            {
                _HEADLINE: un_variable(_SERIES, name="Under-five mortality rate"),
                _FEMALE: un_variable(
                    _SERIES,
                    name="Under-five mortality rate [Sex=Female]",
                    constraints={_SEX: ("Sex", "undata/SEX-F", "Female")},
                ),
            },
            excluded={_AID: "undata/who/SERIES-ODA_1_3"},
            topics={_TOPIC: (_HEADLINE, _FEMALE)},
            superseded_by=_CRS,
        ),
    )


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Every HTTP request the process sends, each answered as an empty graph read."""
    requests: list[httpx.Request] = []

    def send(self: httpx.Client, request: httpx.Request, **_kwargs: object) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": {}}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.fixture
def client() -> LiveGraphClient:
    return LiveGraphClient(base=f"https://{_OURS_HOST}", un_base_url=f"https://{_UN_HOST}/core/api")


def test_deriving_un_candidates_sends_the_un_deployment_nothing(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    (shape,) = derive_shapes(
        confirmed_svs=[_TOPIC, _HEADLINE],
        graph=client,
        sv_scores={_TOPIC: 0.9, _HEADLINE: 0.8},
        sv_types={_TOPIC: "Topic", _HEADLINE: "StatisticalVariable"},
    )

    assert set(shape.sv_arc_facts or {}) == {_HEADLINE, _FEMALE}
    assert _SEX in {slot.property_dcid for slot in shape.slot_keys}
    assert shape.pop_type_name == f"{_SERIES} name"
    assert [request.url.host for request in sent] == []


def test_a_series_members_arcs_come_from_the_tables_with_no_un_request(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    """The member walk reads a series' members and their arcs; the tables answer both."""
    catalogue = _catalogue()
    bind_catalogue(catalogue)

    headline = client.node_arcs_batch([_HEADLINE])[_HEADLINE]
    series = read_five_tuple(headline or {}).pop_type_dcid
    members = catalogue.un_series_members(series)
    arcs = client.node_arcs_batch(list(members))

    assert members == (_HEADLINE, _FEMALE)
    assert read_constraints(headline or {}) == {}
    assert {prop for member in members for prop in read_constraints(arcs[member] or {})} == {_SEX}
    assert sent == []


def test_a_un_shape_holds_the_variables_recall_found(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    """Recall returns the headline alone. The shape's members are what recall found, and
    its slot taxonomy is the whole series' the tables hold, its by-sex member's included.
    The shape stands for that series, so binding the by-sex value its taxonomy offers
    answers from the by-sex member the tables hold, exactly, with no graph read."""
    bind_catalogue(_catalogue())

    (shape,) = derive_shapes(confirmed_svs=[_HEADLINE], graph=client, sv_scores={_HEADLINE: 0.8})
    female = SlotBindingDraft(
        axis="how", property_dcid=_SEX, kind="value", value_dcids=["undata/SEX-F"]
    )
    admitted = _admit_standard_members(shape, [female], ())

    assert list(shape.sv_arc_facts or {}) == [_HEADLINE]
    assert shape.slot_taxonomy == {f"how:{_SEX}": ["undata/SEX-F"]}
    assert shape.held_family is not None
    assert shape.held_family.population_type == _SERIES
    assert isinstance(admitted, _AdmittedStandardMembers)
    assert admitted.family.members == (_FEMALE,)
    assert admitted.exact
    assert sent == []


@pytest.mark.parametrize("refused", [_AID, _NOT_IN_THE_BUILD, _BASE_SDG])
def test_recall_refuses_a_candidate_the_catalogue_does_not_serve_on_either_leg(
    refused: str,
) -> None:
    def scored(*dcids: str) -> tuple[DetectedSV, ...]:
        return tuple(DetectedSV(dcid=d, score=0.9, type_of=None, sentence=None) for d in dcids)

    legs = DetectLegs(ours=scored(refused, _BASE_MORTALITY), un=scored(refused, _HEADLINE))

    detection = admit_legs(legs, catalogue=_catalogue(), detect_k=80)

    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]
    assert [c.dcid for c in detection.un.result() or ()] == [_HEADLINE]


def _member_node(name: str, constraints: dict[str, str]) -> dict:
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": name}]},
        "populationType": {"nodes": [{"dcid": _SERIES}]},
        "measuredProperty": {"nodes": [{"dcid": "value"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }
    return {"label": name, "type": "StatisticalVariable", "arcs": arcs}


class _BindFemaleLLM:
    """Elects the first shape and binds its sex slot to female."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        female = SlotBindingDraft(
            axis="how", property_dcid=_SEX, kind="value", value_dcids=["undata/SEX-F"]
        )
        return elected_reply(0, bindings=[female]), None


def test_a_question_naming_no_place_answers_a_held_member_at_its_default_scope(
    bind_catalogue,
) -> None:
    """Recall finds the headline alone and no place is named. Binding the by-sex value the
    taxonomy offers answers from the by-sex member the tables hold, over the default scope
    its own presence read gives, where the headline has no data. The tables hold the series
    whole, so its membership is settled with no walk, and the answer states the member count
    and digest the mint's walk of the series would state."""
    bind_catalogue(_catalogue())
    query = "female under-five mortality"
    graph = FakeGraph(
        nodes={
            "country/KEN": {"label": "Kenya", "type": "Country"},
            _SERIES: {"label": "Under-five mortality rate"},
            _SEX: {"label": "Sex"},
            "undata/SEX-F": {"label": "Female"},
            _HEADLINE: _member_node("Under-five mortality rate", {}),
            _FEMALE: _member_node("Under-five mortality rate [Sex=Female]", {_SEX: "undata/SEX-F"}),
        },
        obs={
            f"{_FEMALE}|country/KEN": [
                {"earliestDate": "2015", "latestDate": "2020", "obsCount": 5}
            ]
        },
        detect={query: {"svs": [_HEADLINE], "cosine_scores": [0.9]}},
        resolve={},
    )

    result = asyncio.run(
        resolve_variable(
            query,
            entities=[],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_BindFemaleLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )

    (answer,) = result.specs
    assert [sv.ref.ref for sv in answer.resolved.members] == [_FEMALE]
    assert "membership_unverified" not in {warning.code for warning in result.warnings}
    assert answer.resolved.stat_vars.served_count == 1
    assert answer.resolved.shape.member_digest == member_digest([_FEMALE])


_MALE = "undata/who/MORT_U5_RATE.SEX--M"


class _BindNothingLLM:
    """Elects the first shape and binds none of its slots."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        return elected_reply(0), None


_Q5 = "undata/who/MORT_U5_RATE.WEALTH_QUANTILE--Q5"
_WEALTH = "undataWealthQuantile"


_TWO_BREAKDOWNS = {
    _Q5: ("Under-five mortality rate [Wealth quantile=Highest quintile]", _WEALTH, "Q5"),
    _FEMALE: ("Under-five mortality rate [Sex=Female]", _SEX, "F"),
    _MALE: ("Under-five mortality rate [Sex=Male]", _SEX, "M"),
}


def _two_breakdowns_graph(
    bind_catalogue,
    *,
    with_data: frozenset[str],
    members: Mapping[str, tuple[str, str, str]] = _TWO_BREAKDOWNS,
) -> FakeGraph:
    """Bind tables holding a WHO series of `members`, by default a highest-quintile member
    and two by-sex members, each asserting one slot, so none is its total, and return the
    graph in which each member in `with_data` holds five points at Kenya."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("undata/p/WHO", UN_SYSTEM, segment="who")],
            un_mirror=un_mirror(
                {
                    sv: un_variable(
                        _SERIES,
                        name=name,
                        constraints={prop: (prop, f"undata/{prop}-{value}", value)},
                    )
                    for sv, (name, prop, value) in members.items()
                }
            ),
        )
    )
    query = "under-five mortality by wealth quintile"
    return FakeGraph(
        nodes={
            "country/KEN": {"label": "Kenya", "type": "Country"},
            _SERIES: {"label": "Under-five mortality rate"},
            _SEX: {"label": "Sex"},
            **{
                sv: _member_node(name, {prop: f"undata/{prop}-{value}"})
                for sv, (name, prop, value) in members.items()
            },
        },
        obs={
            f"{sv}|country/KEN": [{"earliestDate": "2010", "latestDate": "2014", "obsCount": 5}]
            for sv in with_data
        },
        detect={
            query: {
                "svs": list(members),
                "cosine_scores": [0.9 - 0.01 * i for i in range(len(members))],
            }
        },
        resolve={"Kenya": "country/KEN"},
    )


def _resolve_two_breakdowns(
    graph: FakeGraph,
    *,
    query: str = "under-five mortality by wealth quintile",
    date_request: DateRequest | None = None,
):
    """Resolve `query` in Kenya over `graph` (`_two_breakdowns_graph`) for the years of
    `date_request`, naming no breakdown the binder opens."""
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=date_request,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_BindNothingLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )


def test_a_series_with_no_total_and_two_breakdowns_offers_each_as_a_candidate(
    bind_catalogue,
) -> None:
    """The WHO series has no total, and the question names no breakdown. Each breakdown is
    a candidate whose identity carries its own slot alone, so a mint of its definition
    serves exactly the members it states: the two by-sex series and their 10 rows in Kenya
    first, as the breakdown with more members holding data there, then the quintile and
    its 5 rows, offered on the `breakdown` basis."""
    result = _resolve_two_breakdowns(
        _two_breakdowns_graph(bind_catalogue, with_data=frozenset(_TWO_BREAKDOWNS))
    )

    assert result.status == "candidates"
    assert [
        (
            [slot.property for slot in spec.definition.identity.slots],
            [sv.ref.ref for sv in spec.resolved.members],
            spec.resolved.stat_vars.served_count,
            spec.resolved.basis,
        )
        for spec in result.specs
    ] == [([_SEX], [_FEMALE, _MALE], 2, None), ([_WEALTH], [_Q5], 1, "breakdown")]
    assert [spec.resolved.coverage.observation_count for spec in result.specs] == [10, 5]
    assert "family_total_absent" not in {warning.code for warning in result.warnings}


_URBAN_MEMBER = "undata/who/MORT_U5_RATE.RESIDENCE_AREA--U"


@pytest.mark.parametrize(
    ("max_candidates", "truncated"),
    [(3, False), (2, True)],
    ids=["every_breakdown_offered", "the_cap_cuts_a_breakdown"],
)
def test_a_cap_below_the_breakdown_count_truncates_the_candidates(
    bind_catalogue, monkeypatch: pytest.MonkeyPatch, max_candidates: int, truncated: bool
) -> None:
    """The series breaks down three ways, by sex, by wealth quintile and by residence area.
    With `max_candidates` below that, the part offers the two breakdowns with most members
    holding data and says it cut one."""
    monkeypatch.setattr(settings(), "max_candidates", max_candidates)
    members = {
        **_TWO_BREAKDOWNS,
        _URBAN_MEMBER: ("Under-five mortality rate [Residence area=Urban]", _RESIDENCE, "U"),
    }

    result = _resolve_two_breakdowns(
        _two_breakdowns_graph(bind_catalogue, with_data=frozenset(members), members=members)
    )

    assert result.status == "candidates"
    assert len(result.specs) == min(max_candidates, 3)
    assert result.candidate_budget_truncated is truncated


@pytest.mark.parametrize("max_candidates", [2, 3, 6, 7])
def test_the_cap_keeps_the_breakdown_holding_data_in_the_years_asked_for(
    bind_catalogue, monkeypatch: pytest.MonkeyPatch, max_candidates: int
) -> None:
    """The series breaks down three ways, or seven with four more one-member breakdowns,
    and only the urban member holds data in Kenya in 2020. With `max_candidates` below the
    breakdown count, the cut keeps the residence-area breakdown, the one holding data in the
    year the question asks for, whatever the cap."""
    monkeypatch.setattr(settings(), "max_candidates", max_candidates)
    extra = {
        f"undata/who/MORT_U5_RATE.EXTRA--{i}": (
            f"Under-five mortality rate [Extra{i}=Group]",
            f"undataExtra{i}",
            "G",
        )
        for i in range(4)
    }
    members = {
        **_TWO_BREAKDOWNS,
        **(extra if max_candidates >= 6 else {}),
        _URBAN_MEMBER: ("Under-five mortality rate [Residence area=Urban]", _RESIDENCE, "U"),
    }
    query = "under-five mortality"
    graph = _two_breakdowns_graph(bind_catalogue, with_data=frozenset(members), members=members)
    graph._obs[f"{_URBAN_MEMBER}|country/KEN"] = [
        {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
    ]
    graph._detect[query] = next(iter(graph._detect.values()))

    result = _resolve_two_breakdowns(
        graph,
        query=query,
        date_request=DateRequest(window=TimeWindow(start_year=2020, end_year=2020), latest=False),
    )

    assert _URBAN_MEMBER in {sv.ref.ref for spec in result.specs for sv in spec.resolved.members}


def test_one_breakdown_left_with_data_answers_by_it_with_a_departure(bind_catalogue) -> None:
    """Of the series' two breakdowns only the by-sex members hold data in Kenya, so the
    quintile candidate drops and the by-sex one answers alone, definite, warning that the
    family has no total with the `closest_match` departure."""
    result = _resolve_two_breakdowns(
        _two_breakdowns_graph(bind_catalogue, with_data=frozenset({_FEMALE, _MALE}))
    )

    assert result.status == "definite"
    (answer,) = result.specs
    assert [slot.property for slot in answer.definition.identity.slots] == [_SEX]
    assert [w.spec_id for w in result.warnings if w.code == "family_total_absent"] == [
        answer.spec_id
    ]
    assert [departure.code for departure in result.departure_facts.values()] == ["closest_match"]


_BY_SEX = {
    _FEMALE: ("Under-five mortality rate [Sex=Female]", "undata/SEX-F"),
    _MALE: ("Under-five mortality rate [Sex=Male]", "undata/SEX-M"),
}


def _resolve_one_breakdown(bind_catalogue):
    """Bind tables holding a WHO series of two by-sex members and no total, each holding
    five points at Kenya, and resolve "under-five mortality" there, naming no breakdown.
    Return the result and the graph."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("undata/p/WHO", UN_SYSTEM, segment="who")],
            un_mirror=un_mirror(
                {
                    sv: un_variable(
                        _SERIES, name=name, constraints={_SEX: ("Sex", value, value[-1])}
                    )
                    for sv, (name, value) in _BY_SEX.items()
                }
            ),
        )
    )
    query = "under-five mortality"
    graph = FakeGraph(
        nodes={
            "country/KEN": {"label": "Kenya", "type": "Country"},
            _SERIES: {"label": "Under-five mortality rate"},
            _SEX: {"label": "Sex"},
            **{sv: _member_node(name, {_SEX: value}) for sv, (name, value) in _BY_SEX.items()},
        },
        obs={
            f"{sv}|country/KEN": [{"earliestDate": "2010", "latestDate": "2014", "obsCount": 5}]
            for sv in _BY_SEX
        },
        detect={query: {"svs": list(_BY_SEX), "cosine_scores": [0.9, 0.89]}},
        resolve={"Kenya": "country/KEN"},
    )

    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=f"{query} in Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_BindNothingLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )
    return result, graph


def test_a_series_with_no_total_and_one_breakdown_answers_by_it_with_a_departure(
    bind_catalogue,
) -> None:
    """The WHO series holds two by-sex members and no total, and the question names no
    breakdown. Its one breakdown answers definite, the identity carrying the sex slot alone,
    and the part discloses that the family has no total and the answer shows it by sex,
    with the `closest_match` departure."""
    result, _graph = _resolve_one_breakdown(bind_catalogue)

    assert result.status == "definite"
    (answer,) = result.specs
    assert [slot.property for slot in answer.definition.identity.slots] == [_SEX]
    assert {sv.ref.ref for sv in answer.resolved.members} == set(_BY_SEX)
    assert [w.spec_id for w in result.warnings if w.code == "family_total_absent"] == [
        answer.spec_id
    ]
    assert [departure.code for departure in result.departure_facts.values()] == ["closest_match"]


@pytest.mark.parametrize(
    ("labels", "bound", "message"),
    [
        (("Sex",), (), "This indicator has no overall figure, so the answer shows it by sex."),
        (
            ("Sex", "Age"),
            (),
            "This indicator has no overall figure, so the answer shows it by sex and age.",
        ),
        (
            ("Sex", "Age"),
            ("p0",),
            "This indicator has no overall figure, so the answer shows it by age.",
        ),
        (
            ("HIV status",),
            (),
            "This indicator has no overall figure, so the answer shows it by HIV status.",
        ),
    ],
    ids=["one_slot", "two_slots", "bound_slot_left_out", "acronym_kept"],
)
def test_a_lone_breakdown_warning_names_its_unbound_slots(
    labels: tuple[str, ...], bound: tuple[str, ...], message: str
) -> None:
    from search.engine.region.materialise_step import (
        lone_breakdown_absent,
        resubmit_family_total_absent_departure,
    )
    from search.engine.retrieval.standard import Narrow
    from search.engine.shape import ShapeDraft, SlotKeyDraft

    slots = tuple(
        SlotKeyDraft(axis="how", property_dcid=f"p{i}", label=label)
        for i, label in enumerate(labels)
    )
    shape = ShapeDraft(
        shape_id="s",
        label="s",
        pop_type_dcid=_SERIES,
        meas_prop_dcid="value",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=slots,
    )
    bindings = [
        SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=["v"])
        for prop in bound
    ]
    decision = Narrow(frozenset(slot.property_dcid for slot in slots), lone_signature=True)

    fact = lone_breakdown_absent(decision, shape=shape, bindings=bindings, served=[_FEMALE])

    assert fact is not None
    warning, _departure = resubmit_family_total_absent_departure(
        fact, spec=None, label_of=lambda dcids: list(dcids)
    )
    assert (warning.code, warning.message) == ("family_total_absent", message)


_QUINTILES = {
    f"undata/who/MORT_U5_RATE.WEALTH_QUANTILE--Q{q}": (f"undata/WEALTH-Q{q}", label)
    for q, label in enumerate(
        ("Lowest quintile", "Second quintile", "Middle quintile", "Fourth quintile"),
        start=1,
    )
} | {_Q5: ("undata/WEALTH-Q5", "Highest quintile")}


_RESIDENCE = "undataResidenceArea"
_URBAN = ("undata/RES-U", "Urban")
_AGE = "undataAge"
_UNDER_FIVE = ("undata/AGE-Y0T4", "Under 5 years old")


def _hold_wealth_series(
    bind_catalogue,
    *,
    un_pass: bool,
    urban: bool = False,
    urban_sex: bool = False,
    aged_sex: bool = False,
    query: str = "under-five mortality by wealth quintile",
) -> FakeGraph:
    """Bind tables holding a WHO series whole: five wealth members, each asserting one
    quintile and, with `urban`, the urban residence area too, and two by-sex members, each
    asserting one sex and, with `urban_sex`, the urban residence area too and, with
    `aged_sex`, the age under five, each member holding five points at Kenya. Return the
    graph whose recall for `query` scores the highest quintile alone, our resolve's or,
    with `un_pass`, the UN resolve's, ours recalling nothing."""
    residence = {_RESIDENCE: _URBAN} if urban else {}
    sex_residence = ({_RESIDENCE: _URBAN} if urban_sex else {}) | (
        {_AGE: _UNDER_FIVE} if aged_sex else {}
    )
    members = {
        **{
            sv: (
                f"Under-five mortality rate [Wealth quantile={label}]",
                {_WEALTH: (value, label), **residence},
            )
            for sv, (value, label) in _QUINTILES.items()
        },
        _FEMALE: (
            "Under-five mortality rate [Sex=Female]",
            {_SEX: ("undata/SEX-F", "Female"), **sex_residence},
        ),
        _MALE: (
            "Under-five mortality rate [Sex=Male]",
            {_SEX: ("undata/SEX-M", "Male"), **sex_residence},
        ),
    }
    slot_names = {
        _WEALTH: "Wealth quantile",
        _SEX: "Sex",
        _RESIDENCE: "Residence area",
        _AGE: "Age",
    }
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("undata/p/WHO", UN_SYSTEM, segment="who")],
            un_mirror=un_mirror(
                {
                    sv: un_variable(
                        _SERIES,
                        name=name,
                        constraints={
                            prop: (slot_names[prop], value, label)
                            for prop, (value, label) in constraints.items()
                        },
                    )
                    for sv, (name, constraints) in members.items()
                }
            ),
        )
    )
    recalled = {query: {"svs": [_Q5], "cosine_scores": [0.9]}}
    return FakeGraph(
        nodes={
            "country/KEN": {"label": "Kenya", "type": "Country"},
            _SERIES: {"label": "Under-five mortality rate"},
            **{prop: {"label": name} for prop, name in slot_names.items()},
            **{
                value: {"label": label}
                for _name, constraints in members.values()
                for value, label in constraints.values()
            },
            **{
                sv: _member_node(
                    name, {prop: value for prop, (value, _label) in constraints.items()}
                )
                for sv, (name, constraints) in members.items()
            },
        },
        obs={
            f"{sv}|country/KEN": [{"earliestDate": "2010", "latestDate": "2014", "obsCount": 5}]
            for sv in members
        },
        detect={query: {"svs": [], "cosine_scores": []}} if un_pass else recalled,
        un_detect=recalled if un_pass else {},
        resolve={"Kenya": "country/KEN"},
    )


_WEALTH_QUERY = "under-five mortality by wealth quintile"


def _resolve_by_wealth_quintile(graph: FakeGraph, query: str = _WEALTH_QUERY):
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=f"{query} in Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_BindNothingLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )


@pytest.mark.parametrize("un_pass", [False, True], ids=["our_pass", "un_pass"])
def test_a_breakdown_on_a_series_with_no_total_serves_its_members_alone(
    bind_catalogue, un_pass: bool
) -> None:
    """ "By wealth quintile" opens the wealth slot of a series with no total, so its identity
    carries that slot alone and the by-sex members fall outside it. The answer serves the
    five quintiles, and the mint's walk of the identity, settled from the tables, states
    the same five: 25 rows in Kenya. The UN pass settles its bindings as ours does."""
    result = _resolve_by_wealth_quintile(_hold_wealth_series(bind_catalogue, un_pass=un_pass))

    assert result.status == "definite"
    (answer,) = result.specs
    assert [slot.property for slot in answer.definition.identity.slots] == [_WEALTH]
    resolved = answer.resolved
    assert {sv.ref.ref for sv in resolved.members} == set(_QUINTILES)
    assert resolved.stat_vars.served_count == len(_QUINTILES)
    assert resolved.download is not None
    assert (resolved.download.rows, resolved.download.series) == (25, len(_QUINTILES))
    assert "membership_unverified" not in {warning.code for warning in result.warnings}


def _breakdown_notice(result) -> tuple[list[str], list[str]]:
    """The `family_total_absent` messages and the departure codes `result` states."""
    return (
        [w.message for w in result.warnings if w.code == "family_total_absent"],
        [departure.code for departure in result.departure_facts.values()],
    )


@pytest.mark.parametrize("un_pass", [False, True], ids=["our_pass", "un_pass"])
def test_a_resubmit_of_a_breakdown_identity_keeps_its_spec_id(
    bind_catalogue, un_pass: bool
) -> None:
    """The identity the breakdown answer mints carries the wealth slot alone. A resubmit of
    it reads its slots as posted, so it serves the same five quintiles under the same
    spec_id. The slot it leaves open counts as named, as the first answer's phrase named
    it, so neither states that the family has no total."""
    from search.engine.region import resolve_spec_resubmit
    from search.models import ResubmitIdentityPayload

    graph = _hold_wealth_series(bind_catalogue, un_pass=un_pass)
    result = _resolve_by_wealth_quintile(graph)
    (answer,) = result.specs

    resubmitted = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=answer.definition.identity),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph, EngineOptions(deadline=None)),
    )

    (spec,) = resubmitted.specs
    assert spec.spec_id == answer.spec_id
    assert {sv.ref.ref for sv in spec.resolved.members} == set(_QUINTILES)
    assert spec.resolved.stat_vars.served_count == len(_QUINTILES)
    assert _breakdown_notice(resubmitted) == _breakdown_notice(result) == ([], [])


@pytest.mark.parametrize("quintiles_hold_data", [True, False], ids=["with_data", "no_data"])
def test_a_breakdown_no_member_holds_alone_answers_by_the_breakdown_holding_it(
    bind_catalogue, quintiles_hold_data: bool
) -> None:
    """Every quintile member of the series also asserts the urban residence area, so "by
    wealth quintile" opens a slot no member asserts alone and no identity serves the
    quintiles by themselves. The one breakdown holding the wealth slot, the quintiles by
    residence area, answers definite, narrower than asked, with the `closest_match`
    departure. The by-sex series leave out the breakdown the question named, so none is
    offered, and with no quintile holding data in Kenya the part answers no data."""
    graph = _hold_wealth_series(bind_catalogue, un_pass=False, urban=True)
    if not quintiles_hold_data:
        for sv in _QUINTILES:
            del graph._obs[f"{sv}|country/KEN"]

    result = _resolve_by_wealth_quintile(graph)

    if not quintiles_hold_data:
        assert (result.status, result.no_data_reason) == ("no_data", "no_observations")
        assert not result.specs
        return
    assert result.status == "definite"
    (answer,) = result.specs
    assert [slot.property for slot in answer.definition.identity.slots] == [_RESIDENCE, _WEALTH]
    assert {sv.ref.ref for sv in answer.resolved.members} == set(_QUINTILES)
    assert [w.message for w in result.warnings if w.code == "family_total_absent"] == [
        "This indicator has no overall figure, so the answer shows it by wealth quantile and "
        "residence area."
    ]
    assert [departure.code for departure in result.departure_facts.values()] == ["closest_match"]


@pytest.mark.parametrize("answer", ["lone_breakdown", "by_sex_candidate"])
def test_a_resubmit_of_a_breakdown_answer_keeps_its_spec_id_and_names_its_breakdown(
    bind_catalogue, answer: str
) -> None:
    """A resubmit of a definite lone-breakdown answer's identity, or of the by-sex candidate
    a series with two breakdowns offers, reads its slots as posted, so it answers definite
    over the same series under the same spec_id. The sex slot the identity leaves open
    counts as named, so the resubmit states the warnings and departures "under-five
    mortality by sex" states over the same series: no `family_total_absent` warning and no
    `closest_match` departure, though the lone-breakdown first answer, which named none,
    states both."""
    from search.engine.region import resolve_spec_resubmit
    from search.models import ResubmitIdentityPayload

    if answer == "lone_breakdown":
        result, graph = _resolve_one_breakdown(bind_catalogue)
        (first,) = result.specs
        assert _breakdown_notice(result) == (
            ["This indicator has no overall figure, so the answer shows it by sex."],
            ["closest_match"],
        )
    else:
        graph = _two_breakdowns_graph(bind_catalogue, with_data=frozenset(_TWO_BREAKDOWNS))
        result = _resolve_two_breakdowns(graph)
        assert result.status == "candidates"
        first = result.specs[0]

    resubmitted = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=first.definition.identity),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph, EngineOptions(deadline=None)),
    )

    assert resubmitted.status == "definite"
    (spec,) = resubmitted.specs
    assert spec.spec_id == first.spec_id
    assert {sv.ref.ref for sv in spec.resolved.members} == {_FEMALE, _MALE}
    assert spec.resolved.stat_vars.served_count == 2
    assert _breakdown_notice(resubmitted) == ([], [])
    named_query = "under-five mortality by sex"
    graph._detect[named_query] = next(iter(graph._detect.values()))
    named = _resolve_two_breakdowns(graph, query=named_query)
    assert named.status == "definite"
    assert [w.code for w in resubmitted.warnings] == [w.code for w in named.warnings]
    assert [d.code for d in resubmitted.departure_facts.values()] == [
        d.code for d in named.departure_facts.values()
    ]


def test_a_narrowed_part_whose_walk_settles_another_query_raises(bind_catalogue) -> None:
    """A narrowed part's walk must settle its identity's query. A walk keyed by the query
    that still carries the sex slot settled another family than a mint of the identity
    serves, so assembling the answer raises; a walk of the identity's own query passes."""
    from types import SimpleNamespace

    from search.engine.region.grounding import check_breakdown_settled
    from search.engine.retrieval.bound_values import (
        StatedMembers,
        WalkedMembers,
        identity_query,
        walk_key,
    )

    (answer,) = _resolve_by_wealth_quintile(
        _hold_wealth_series(bind_catalogue, un_pass=False)
    ).specs
    query = identity_query(answer.definition.identity)
    wider = dataclasses.replace(query, slots={**query.slots, _SEX: ("unbound",)})
    shape = SimpleNamespace(shape_id="un_wealth", omitted_slots=frozenset({_SEX}))
    walked = WalkedMembers(key=walk_key(wider), stated=StatedMembers(served_count=7, digest=None))

    with pytest.raises(AssertionError, match="settles no members"):
        check_breakdown_settled(shape, walked, answer)

    check_breakdown_settled(shape, dataclasses.replace(walked, key=walk_key(query)), answer)


_EMP_SERIES = "undata/ilo/SERIES-EMP_TEMP_NB"
_EMP_TOTAL = "undata/ilo/EMP_TEMP_NB"
_EMP_OCCUPATION = "undataOccupation"
_EMP_AGE = "undataAge"
_EMP_TEACHING = ("undata/OCCUPATION-ISCO08_23", "undata/OCCUPATION-ISCO88_23")
_EMP_TEACHING_ASSOCIATE = "undata/OCCUPATION-ISCO88_33"
_EMP_BY_OCCUPATION = {
    f"{_EMP_TOTAL}.OCCUPATION--{value.rsplit('-', 1)[1]}": {_EMP_OCCUPATION: value}
    for value in (*_EMP_TEACHING, _EMP_TEACHING_ASSOCIATE)
}
_EMP_MEMBERS: dict[str, dict[str, str]] = {
    _EMP_TOTAL: {},
    **_EMP_BY_OCCUPATION,
    f"{_EMP_TOTAL}.OCCUPATION--ISCO08_23__SEX--F": {
        _EMP_OCCUPATION: _EMP_TEACHING[0],
        _SEX: "undata/SEX-F",
    },
    f"{_EMP_TOTAL}.SEX--F": {_SEX: "undata/SEX-F"},
    f"{_EMP_TOTAL}.AGE--Y15T19": {_EMP_AGE: "undata/AGE-Y15T19"},
}
_EMP_QUERY = "teacher count"


class _BindTeachingLLM:
    """Elects the first shape and binds its occupation slot to the three teaching codes, in
    an order other than the sorted one an identity states."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        teaching = SlotBindingDraft(
            axis="how",
            property_dcid=_EMP_OCCUPATION,
            kind="set",
            value_dcids=[_EMP_TEACHING_ASSOCIATE, *_EMP_TEACHING],
        )
        return elected_reply(0, bindings=[teaching]), None


def _resolve_teachers_in_nauru(bind_catalogue):
    """Bind tables holding an ILO employment series whole, with a total, one member per
    teaching occupation, each holding three points at Nauru, and members by sex and by age,
    and resolve "teacher count" in Nauru, the UN pass recalling every member but the total
    and binding the occupation slot to the three teaching codes. No member is the total of
    a set binding, and each teaching member asserts the occupation alone, so the identity
    carries that slot alone."""
    names = {_EMP_OCCUPATION: "Occupation", _SEX: "Sex", _EMP_AGE: "Age"}
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("undata/p/ILO", UN_SYSTEM, segment="ilo")],
            un_mirror=un_mirror(
                {
                    sv: un_variable(
                        _EMP_SERIES,
                        name=sv,
                        constraints={
                            prop: (names[prop], value, value) for prop, value in c.items()
                        },
                    )
                    for sv, c in _EMP_MEMBERS.items()
                }
            ),
        )
    )
    recalled = [sv for sv in _EMP_MEMBERS if sv != _EMP_TOTAL]

    def node(sv: str, constraints: dict[str, str]) -> dict:
        member = _member_node(sv, constraints)
        member["arcs"]["populationType"] = {"nodes": [{"dcid": _EMP_SERIES}]}
        return member

    graph = FakeGraph(
        nodes={
            "country/NRU": {"label": "Nauru", "type": "Country"},
            _EMP_SERIES: {"label": "Employment"},
            **{prop: {"label": name} for prop, name in names.items()},
            **{value: {"label": value} for c in _EMP_MEMBERS.values() for value in c.values()},
            **{sv: node(sv, c) for sv, c in _EMP_MEMBERS.items()},
        },
        obs={
            f"{sv}|country/NRU": [{"earliestDate": "2013", "latestDate": "2024", "obsCount": 3}]
            for sv in _EMP_BY_OCCUPATION
        },
        detect={_EMP_QUERY: {"svs": [], "cosine_scores": []}},
        un_detect={
            _EMP_QUERY: {
                "svs": recalled,
                "cosine_scores": [0.9 - 0.01 * i for i in range(len(recalled))],
            }
        },
        resolve={"Nauru": "country/NRU"},
    )
    return asyncio.run(
        resolve_variable(
            _EMP_QUERY,
            entities=["Nauru"],
            date_request=None,
            detect_query=_EMP_QUERY,
            role_query=f"{_EMP_QUERY} in Nauru",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_BindTeachingLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )


def _settle_queued_presence_reads() -> None:
    """Wait for every availability read the presence read queued to settle."""
    while flights := [flight for flight in place_scope._flights.values() if not flight.done()]:
        wait(flights, timeout=10)


def test_a_held_family_past_the_cold_read_cap_answers_unsettled_then_settles(
    bind_catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A series with no total serves both by-sex members to a question naming no place, and
    recall scores both, the female one first, so neither sex binds. With room for one cold
    presence read, the first request reads the female member's places and gives the male
    member its countries for this request. The answer shows both members; its scope is
    unsettled and its leg degraded, so no cache stores it. The male member's read settles
    behind it, and a second request reads both scopes from the memo, settled, starting no
    availability read."""
    monkeypatch.setattr(place_scope, "MAX_COLD_READS_PER_REQUEST", 1)
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("undata/p/WHO", UN_SYSTEM, segment="who")],
            un_mirror=un_mirror(
                {
                    sv: un_variable(
                        _SERIES,
                        name=f"Under-five mortality rate [Sex={sex}]",
                        constraints={_SEX: ("Sex", f"undata/SEX-{sex[0]}", sex)},
                    )
                    for sv, sex in ((_FEMALE, "Female"), (_MALE, "Male"))
                }
            ),
        )
    )
    query = "under-five mortality"
    facet = {"earliestDate": "2015", "latestDate": "2020", "obsCount": 5}
    graph = FakeGraph(
        nodes={
            "country/KEN": {"label": "Kenya", "type": "Country"},
            _SERIES: {"label": "Under-five mortality rate"},
            _SEX: {"label": "Sex"},
            "undata/SEX-F": {"label": "Female"},
            "undata/SEX-M": {"label": "Male"},
            _FEMALE: _member_node("Under-five mortality rate [Sex=Female]", {_SEX: "undata/SEX-F"}),
            _MALE: _member_node("Under-five mortality rate [Sex=Male]", {_SEX: "undata/SEX-M"}),
        },
        obs={f"{sv}|country/KEN": [facet] for sv in (_FEMALE, _MALE)},
        detect={query: {"svs": [_FEMALE, _MALE], "cosine_scores": [0.9, 0.89]}},
        resolve={},
    )
    submitted: list[object] = []
    submit = place_scope._submit

    def spy(read) -> None:
        submitted.append(read.key)
        submit(read)

    monkeypatch.setattr(place_scope, "_submit", spy)

    def ask():
        return asyncio.run(
            resolve_variable(
                query,
                entities=[],
                date_request=None,
                detect_query=query,
                role_query=query,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=_BindNothingLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(deadline=None),
            )
        )

    first = ask()
    _settle_queued_presence_reads()
    reads_first = len(submitted)
    second = ask()

    for result in (first, second):
        (answer,) = result.specs
        assert result.status == "definite"
        assert [sv.ref.ref for sv in answer.resolved.members] == [_FEMALE, _MALE]
    for result in (first, second):
        assert PLACE_SCOPE_DEFAULTED in {warning.code for warning in result.warnings}
    assert (first.scope_unsettled, first.leg_degraded) == (True, True)
    assert reads_first == 2
    assert (second.scope_unsettled, second.leg_degraded) == (False, False)
    assert len(submitted) == reads_first


@pytest.mark.parametrize(
    ("urban_sex", "aged_sex", "slots", "message"),
    [
        (
            True,
            False,
            [_RESIDENCE, _SEX, _WEALTH],
            "This indicator has no overall figure, so the answer shows it by residence area.",
        ),
        # Only the by-sex members recall did not find assert the age.
        (
            False,
            True,
            [_AGE, _RESIDENCE, _SEX, _WEALTH],
            "This indicator has no overall figure, so the answer shows it by residence area "
            "and age.",
        ),
    ],
    ids=["shared_residence", "age_outside_the_sample"],
)
def test_breakdowns_no_member_holds_together_answer_by_every_member_with_a_departure(
    bind_catalogue, urban_sex: bool, aged_sex: bool, slots: list[str], message: str
) -> None:
    """Every quintile member of the series also asserts the urban residence area, and none
    asserts both a sex and a wealth quintile, so "by sex and by wealth quintile" opens two
    slots no breakdown holds together. The family keeps every slot and serves all seven
    members, and the answer discloses each breakdown the question did not name that a
    served member asserts, read over the whole family though recall found one quintile
    alone, with the `closest_match` departure."""
    query = "under-five mortality by sex and by wealth quintile"
    graph = _hold_wealth_series(
        bind_catalogue,
        un_pass=False,
        urban=True,
        urban_sex=urban_sex,
        aged_sex=aged_sex,
        query=query,
    )

    result = _resolve_by_wealth_quintile(graph, query)

    assert result.status == "definite"
    (answer,) = result.specs
    assert [slot.property for slot in answer.definition.identity.slots] == slots
    assert {sv.ref.ref for sv in answer.resolved.members} == {*_QUINTILES, _FEMALE, _MALE}
    assert _breakdown_notice(result) == ([message], ["closest_match"])
