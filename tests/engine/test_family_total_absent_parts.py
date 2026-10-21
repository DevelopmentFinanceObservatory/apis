"""A part raises `family_total_absent` for the Specs it holds and for no other.

The fact travels with the Spec it describes, so a Spec that leaves the answer takes
its warning with it. It is the election's warning, so the part carries it and the
sentence does not.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.binders import Binder
from search.engine.conjoin import assemble_region
from search.engine.envelope import now_ms
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import FAMILY_TOTAL_ABSENT, ONLY_ALTERNATE_HAS_DATA
from search.engine.retrieval import standard
from search.models import TimeWindow
from tests.engine import test_equivalent_readings_divert as twins
from tests.engine import test_zero_observation_election as reelection
from tests.engine._binders import ReplyBinder, elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_KEN = "country/KEN"
_KEN_NODE = {_KEN: {"label": "Kenya", "type": "Country"}}
_OBS = [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}]


def _node(label: str, *, meas_prop: str, meas_qual: str | None = None, **constraints: str) -> dict:
    """A deaths StatVar asserting `constraints`."""
    arcs = reelection._arcs(
        label, pop_type="MortalityEvent", meas_prop=meas_prop, meas_qual=meas_qual
    )
    if constraints:
        arcs["constraintProperties"] = {"nodes": [{"dcid": prop} for prop in constraints]}
        arcs |= {prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()}
    return {"label": label, "type": "StatisticalVariable", "arcs": arcs}


class _BindLLM:
    def __init__(self, also_plausible: list[int] | None):
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[], also_plausible=self._also_plausible), None


def _resolve(
    query: str,
    nodes: dict[str, dict],
    *,
    binder: Binder | None = None,
    scores: list[float],
    with_data: dict[str, dict],
    also_plausible: list[int] | None = None,
):
    """Resolve `query` in Kenya over `nodes`, in recall order, electing the first shape."""
    graph = FakeGraph(
        nodes={**_KEN_NODE, **nodes},
        detect={query: {"svs": list(nodes), "cosine_scores": scores}},
        resolve={"Kenya": _KEN},
        obs={f"{sv}|{_KEN}": [facet] for sv, facet in with_data.items()},
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            binder=binder,
            llm=_BindLLM(also_plausible),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def _family_notes(result) -> list[str]:
    return [w.message for w in result.warnings if w.code == FAMILY_TOTAL_ABSENT]


def _sentence(result, query: str) -> str:
    response = assemble_region(
        result,
        query=query,
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=5,
    )
    return response.parts[0].spec.resolved.rendered_sentence or ""


def _stat_vars(spec) -> list[str]:
    return [sv.ref.ref for sv in spec.resolved.members]


@pytest.fixture
def status_reads_not_declared(monkeypatch):
    """An absent fixtureStatus keeps a member asserting it in the family beside its total."""
    monkeypatch.setattr(
        standard, "absent_readings", lambda _index: {"fixtureStatus": "not_declared"}
    )


# The count shape is elected. The rate shape holds a total without data and a narrower
# member with data, so a rate Spec names the narrower member.
_COUNT = "TESTFIXTURE/Deaths/Count"
_RATE = "TESTFIXTURE/Deaths/Rate"
_RATE_DISABLED = "TESTFIXTURE/Deaths/Rate_Disabled"
_RATE_NODES = {
    _RATE: _node("Death rate", meas_prop="fixtureRate"),
    _RATE_DISABLED: _node(
        "Death rate, disabled", meas_prop="fixtureRate", fixtureStatus="Disabled"
    ),
}
_RATE_NOTE = "No data for Death rate was found for Kenya, so the answer shows Death rate, disabled."


def test_a_definite_part_warns_and_its_sentence_does_not(status_reads_not_declared):
    result = _resolve(
        "deaths", _RATE_NODES, scores=[0.9, 0.85], with_data={_RATE_DISABLED: _OBS[0]}
    )

    assert result.status == "definite", result
    assert _stat_vars(result.specs[0]) == [_RATE_DISABLED]
    assert _family_notes(result) == [_RATE_NOTE]
    assert _RATE_NOTE not in _sentence(result, "deaths")


def _resolve_rate_in(date_request: DateRequest | None):
    """The rate shape alone: its total holds Kenya 2000 to 2010 and the narrower member
    2015 to 2022."""
    obs = {
        f"{_RATE}|{_KEN}": [{"earliestDate": "2000", "latestDate": "2010", "obsCount": 11}],
        f"{_RATE_DISABLED}|{_KEN}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 8}],
    }
    graph = FakeGraph(
        nodes={**_KEN_NODE, **_RATE_NODES},
        detect={"deaths": {"svs": list(_RATE_NODES), "cosine_scores": [0.9, 0.85]}},
        resolve={"Kenya": _KEN},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            "deaths",
            entities=["Kenya"],
            date_request=date_request,
            detect_query="deaths",
            role_query="deaths",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(None),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def _years(start: int, end: int) -> DateRequest:
    return DateRequest(window=TimeWindow(start_year=start, end_year=end), latest=False)


def test_a_total_with_data_only_outside_the_years_asked_is_said_to_lack_data_in_them(
    status_reads_not_declared,
):
    """Dateless, the total answers and nothing warns. Asked for 2015 to 2020 the narrower
    member answers, and the warning says the total has no data in the years asked: it holds
    Kenya data in other years. Asked for 2011 to 2014, where neither holds a point, the
    total answers for the window and nothing warns."""
    dateless = _resolve_rate_in(None)
    assert dateless.status == "definite", dateless
    assert _stat_vars(dateless.specs[0]) == [_RATE]
    assert _family_notes(dateless) == []

    inside = _resolve_rate_in(_years(2015, 2020))
    assert inside.status == "definite", inside
    assert _stat_vars(inside.specs[0]) == [_RATE_DISABLED]
    assert _family_notes(inside) == [
        "No data for Death rate was found for Kenya in the years asked, so the answer shows "
        "Death rate, disabled."
    ]

    between = _resolve_rate_in(_years(2011, 2014))
    assert (between.status, between.no_data_reason) == ("no_data", "no_observations")
    assert between.window_gap_fact is not None
    assert _family_notes(between) == []


def test_an_alternate_candidate_whose_total_has_no_data_warns(status_reads_not_declared):
    result = _resolve(
        "deaths",
        {_COUNT: _node("Count", meas_prop="fixtureCount"), **_RATE_NODES},
        scores=[0.9, 0.85, 0.8],
        with_data={_COUNT: _OBS[0], _RATE_DISABLED: _OBS[0]},
        also_plausible=[1],
    )

    assert result.status == "candidates", result
    assert _family_notes(result) == [_RATE_NOTE]


def test_a_no_data_part_warns_for_the_alternate_it_holds_and_not_for_the_elected_member(
    status_reads_not_declared,
):
    """The elected shape's male and female members have no data, so the part is no_data
    and holds the rate alternate. `elected_stat_vars` names the male member without a
    Spec, so the only warning is the alternate's."""
    result = _resolve(
        "deaths",
        {
            "EL_M": _node("Deaths, male", meas_prop="fixtureCount", gender="Male"),
            "EL_F": _node("Deaths, female", meas_prop="fixtureCount", gender="Female"),
            **_RATE_NODES,
        },
        scores=[0.9, 0.89, 0.85, 0.8],
        with_data={_RATE_DISABLED: _OBS[0]},
        also_plausible=[1],
        binder=ReplyBinder(),
    )

    assert result.status == "no_data", result
    assert result.gate == ONLY_ALTERNATE_HAS_DATA
    assert result.alternate_match is not None
    assert _stat_vars(result.alternate_match) == [_RATE_DISABLED]
    assert [ref.ref for ref in result.elected_stat_vars or []] == ["EL_M"]
    assert _family_notes(result) == [_RATE_NOTE]


def test_a_reelected_sibling_drops_the_first_shapes_warning_and_note():
    """The elected shape's male and female members have no data in Kenya, so the
    qualifier-variant sibling answers. Its one member is its family's total, and the part
    and its sentence say nothing about the male member."""
    query = reelection._QUERY
    result = _resolve(
        query,
        {
            "EL_M": _node("Deaths, male", meas_prop="count", gender="Male"),
            "EL_F": _node("Deaths, female", meas_prop="count", gender="Female"),
            "SIB": _node("Estimated deaths", meas_prop="count", meas_qual="Estimated"),
        },
        scores=[0.90, 0.89, 0.85],
        with_data={"SIB": _OBS[0]},
    )

    assert result.status == "definite", result
    assert _stat_vars(result.specs[0]) == ["SIB"]
    assert _family_notes(result) == []
    assert "Deaths, male" not in _sentence(result, query)


# The alternate shape holds a root without data and the registry twin of the elected
# World Bank series, which the mint keeps beside the root.
_SDG_ROOT = "sdg/SH_DYN_MORT_ROOT"
_TWIN_NODES = {
    twins._MORT_WB: twins._node("Person", "mortalityWorldBank"),
    _SDG_ROOT: {
        "label": "SDG mortality",
        "type": "StatisticalVariable",
        "arcs": twins._arcs(pop_type="Person", meas_prop="mortalitySdg"),
    },
    twins._MORT_SDG: {
        "label": "SDG under-five mortality",
        "type": "StatisticalVariable",
        "arcs": {
            **twins._arcs(pop_type="Person", meas_prop="mortalitySdg"),
            "constraintProperties": {"nodes": [{"dcid": "fixtureStatus"}]},
            "fixtureStatus": {"nodes": [{"dcid": "Y0T4"}]},
        },
    },
}
_TWIN_DATA = {
    twins._MORT_WB: twins._facet(provenance_id="source/WorldBank"),
    twins._MORT_SDG: twins._facet(provenance_id="source/SDG"),
}
_TWIN_QUERY = "indicators in Kenya"


def test_a_twin_moved_to_equivalent_readings_takes_its_warning_and_note(
    status_reads_not_declared,
):
    result = _resolve(
        _TWIN_QUERY,
        _TWIN_NODES,
        scores=[0.9, 0.88, 0.8],
        with_data=_TWIN_DATA,
        also_plausible=[1],
    )

    assert result.status == "definite", result
    (spec,) = result.specs
    assert _stat_vars(spec) == [twins._MORT_WB]
    assert [_stat_vars(twin) for twin in spec.resolved.equivalent_readings or []] == [
        [twins._MORT_SDG]
    ]
    assert _family_notes(result) == []
    assert "SDG" not in _sentence(result, _TWIN_QUERY)


def test_a_candidates_part_warns_for_its_candidates_and_not_for_a_diverted_twin(
    status_reads_not_declared,
):
    """Three shapes survive: the elected series, its twin, and the rate shape. The twin
    moves under the elected Spec, and the two candidates left raise only the rate
    shape's warning."""
    result = _resolve(
        _TWIN_QUERY,
        {**_TWIN_NODES, **_RATE_NODES},
        scores=[0.9, 0.88, 0.8, 0.79, 0.78],
        with_data={**_TWIN_DATA, _RATE_DISABLED: _OBS[0]},
        also_plausible=[1, 2],
    )

    assert result.status == "candidates", result
    assert [_stat_vars(spec) for spec in result.specs] == [[twins._MORT_WB], [_RATE_DISABLED]]
    assert _family_notes(result) == [_RATE_NOTE]
