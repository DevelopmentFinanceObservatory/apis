"""Search reads a UN candidate's facts from the catalogue's UN tables: derivation sends the UN
deployment nothing, a shape holds the variables recall found, and admission follows the
tables."""

from __future__ import annotations

import asyncio
from concurrent.futures import wait

import httpx
import pytest

from materialise import place_scope
from origin_catalogue import BASE, DFO, UN_SYSTEM
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import DetectedSV, DetectLegs, admit_legs
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import PLACE_SCOPE_DEFAULTED
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import read_constraints, read_five_tuple
from search.engine.retrieval.standard import _admit_standard_members, _AdmittedStandardMembers
from statvar_members import member_digest
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
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
        return elected_reply(0), None


_Q5 = "undata/who/MORT_U5_RATE.WEALTH_QUANTILE--Q5"
_WEALTH = "undataWealthQuantile"


def test_a_series_with_no_total_states_the_members_a_mint_of_its_definition_serves(
    bind_catalogue,
) -> None:
    """The WHO series holds a highest-quintile member and two by-sex members, each asserting
    one slot, so none is its total and a mint of the definition, which leaves both slots
    open, serves all three. The answer serves the same three: its members, its coverage and
    its download state the mint's three series and their 15 rows in Kenya, and no member
    stands in for the family."""
    members = {
        _Q5: ("Under-five mortality rate [Wealth quantile=Highest quintile]", _WEALTH, "Q5"),
        _FEMALE: ("Under-five mortality rate [Sex=Female]", _SEX, "F"),
        _MALE: ("Under-five mortality rate [Sex=Male]", _SEX, "M"),
    }
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
    graph = FakeGraph(
        nodes={
            "country/KEN": {"label": "Kenya", "type": "Country"},
            _SERIES: {"label": "Under-five mortality rate"},
            **{
                sv: _member_node(name, {prop: f"undata/{prop}-{value}"})
                for sv, (name, prop, value) in members.items()
            },
        },
        obs={
            f"{sv}|country/KEN": [{"earliestDate": "2010", "latestDate": "2014", "obsCount": 5}]
            for sv in members
        },
        detect={query: {"svs": list(members), "cosine_scores": [0.9, 0.89, 0.88]}},
        resolve={"Kenya": "country/KEN"},
    )

    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
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

    assert result.status == "definite"
    (answer,) = result.specs
    resolved = answer.resolved
    assert [sv.ref.ref for sv in resolved.members] == list(members)
    assert resolved.stat_vars.served_count == len(members)
    assert resolved.coverage.observation_count == 15
    assert resolved.download is not None
    assert (resolved.download.rows, resolved.download.series) == (15, len(members))
    assert "family_total_absent" not in {warning.code for warning in result.warnings}


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
