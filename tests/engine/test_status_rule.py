"""The status rule: which readings a part offers beside the elected one, and which answers
when the elected one holds no data (`region.status_rule`).

The table tests run `plan_status` on the design's worked examples: "Ghana's GDP" and "add gdp
for nigeria" stay definite when the model nominates the registry twin, the per-capita sibling
and real GDP; "Vietnam GDP per capita over time" stays definite beside the level it names
none of; "Education spending", held only as a share of GDP and a share of government
spending, offers both shares. The pipeline tests run `resolve_variable` on synthetic graphs:
a retrieval margin sets no status, a nomination with data is offered, and a place named
without a direction is read on the one direction that holds data, or offered both ways when
both do.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from code_index.shared import get_index
from search.engine.bind import ShapeRating, _UniversalBindOutput
from search.engine.binders import GeminiBinder
from search.engine.config import settings
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import RegionResult, resolve_variable
from search.engine.region.status_rule import (
    Alternative,
    MenuReading,
    OpenAxes,
    StatusPlan,
    holds_default_price_basis,
    open_axes,
    plan_status,
)
from search.engine.shape import ShapeDraft
from search.engine.stated_words_read import StatedWord, StatedWordsOutput
from search.models import TimeWindow
from series_coverage.schema import RoleNames
from tests.engine import test_first_answer_place_scenarios as scenarios
from tests.engine._binders import RatingBinder, nominee_ratings
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.fixtures import FakeGraph, make_resolver

_GDP = "GrossDomesticProduction"
_GOV_SPENDING = "Amount_EconomicActivity_ExpenditureActivity_Government"


def _shape(
    label: str,
    *,
    pop_type: str = "EconomicActivity",
    meas_prop: str = "amount",
    stat_type: str = "measuredValue",
    meas_qual: str | None = "Nominal",
    meas_denom: str | None = None,
    members: tuple[str, ...] = ("sv",),
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=label,
        label=label,
        pop_type_dcid=pop_type,
        meas_prop_dcid=meas_prop,
        stat_type_dcid=stat_type,
        meas_qual_dcid=meas_qual,
        meas_denom_dcid=meas_denom,
        slot_keys=(),
        sv_arc_facts={member: {} for member in members},
    )


_NOMINAL_GDP = _shape("Nominal gross domestic product")
_IMF_GDP = _shape("Gross domestic product, current prices")
_REAL_GDP = _shape("Real gross domestic product", meas_qual="RealValue")
_RECIPIENT_NOMINAL_GDP = dataclasses.replace(
    _NOMINAL_GDP, declared_roles=("observationAbout", "financingRecipient")
)
_RECIPIENT_REAL_GDP = dataclasses.replace(
    _REAL_GDP, declared_roles=("observationAbout", "financingRecipient")
)
_GDP_PER_CAPITA = _shape("Nominal gross domestic product per capita", meas_denom="Count_Person")
# The same per-person base stated by the ProvisionalNode `PerCapita`, whose arc resolves to
# no node.
_GDP_PER_CAPITA_PROVISIONAL = dataclasses.replace(
    _GDP_PER_CAPITA, meas_denom_dcid="PerCapita", meas_denom_unresolved=True
)
_IMF_GDP_PER_CAPITA = _shape(
    "Gross domestic product per capita, current prices", meas_denom="Count_Person"
)
_EDUCATION_SHARE_OF_GDP = _shape(
    "Percent of nominal GDP spent on education", meas_denom=f"Amount_EconomicActivity_{_GDP}"
)
_EDUCATION_SHARE_OF_SPENDING = _shape(
    "Percent of total government expenditure on education", meas_denom=_GOV_SPENDING
)
_GDP_GROWTH = _shape("Gross domestic product growth rate", stat_type="growthRate")
_MORTALITY = _shape("Mortality event count", pop_type="MortalityEvent", meas_prop="count")
_UNRELATED = _shape("Unrelated topic count", pop_type="UnrelatedThing", meas_prop="count")
# A UN System Data Commons series; our pass can hold one, as a superseded variable our
# recall stands as its UN successor.
_UN_UNRELATED = _shape(
    "Unrelated topic count",
    pop_type="undata/unicef/SERIES-UNRELATED",
    meas_prop="value",
    members=("undata/unicef/SERIES-UNRELATED",),
)


def _menu(*readings: tuple[ShapeDraft, bool | None, bool, bool]) -> list[MenuReading]:
    """Menu readings from (shape, D, nominated, registry twin), indexed from 1."""
    return [
        MenuReading(index=i, shape=shape, has_data=d, nominated=nominated, registry_twin=twin)
        for i, (shape, d, nominated, twin) in enumerate(readings, start=1)
    ]


def _plan(
    elected: ShapeDraft,
    menu: list[MenuReading],
    *,
    stat_type_named: bool = False,
    form_named: bool = False,
    price_basis_named: bool = False,
) -> StatusPlan:
    axes = open_axes(
        elected,
        stat_type_named=stat_type_named,
        form_named=form_named,
        price_basis_named=price_basis_named,
    )
    return plan_status(elected, menu, axes=axes)


@pytest.mark.parametrize(
    ("elected", "menu", "named", "rivals"),
    [
        pytest.param(
            _NOMINAL_GDP,
            _menu(
                (_IMF_GDP, True, True, True),
                (_GDP_PER_CAPITA, True, True, False),
                (_REAL_GDP, True, True, False),
            ),
            {},
            (),
            id="ghana-gdp-stays-definite-beside-its-twin-per-capita-and-real-readings",
        ),
        pytest.param(
            _NOMINAL_GDP,
            _menu((_GDP_PER_CAPITA, True, True, False), (_IMF_GDP, True, False, True)),
            {},
            (),
            id="nigeria-gdp-stays-definite",
        ),
        pytest.param(
            _GDP_PER_CAPITA,
            _menu((_NOMINAL_GDP, True, True, False), (_IMF_GDP_PER_CAPITA, True, True, True)),
            {"form_named": True},
            (),
            id="vietnam-gdp-per-capita-stays-definite-beside-the-level",
        ),
        pytest.param(
            _EDUCATION_SHARE_OF_GDP,
            _menu((_EDUCATION_SHARE_OF_SPENDING, True, False, False)),
            {},
            (Alternative(index=1, basis="structural"),),
            id="education-spending-offers-both-shares",
        ),
        pytest.param(
            _GDP_PER_CAPITA_PROVISIONAL,
            _menu((_GDP_PER_CAPITA, True, True, False)),
            {},
            (),
            id="one-per-person-base-on-two-arcs-is-no-rival-on-an-open-denominator",
        ),
        pytest.param(
            _EDUCATION_SHARE_OF_GDP,
            _menu((_EDUCATION_SHARE_OF_SPENDING, False, True, False)),
            {},
            (),
            id="a-share-with-no-data-where-asked-is-no-rival",
        ),
        pytest.param(
            _EDUCATION_SHARE_OF_GDP,
            _menu((_EDUCATION_SHARE_OF_SPENDING, True, True, False)),
            {"form_named": True},
            (),
            id="a-named-form-closes-the-denominator-even-for-a-nomination",
        ),
        pytest.param(
            _EDUCATION_SHARE_OF_GDP,
            _menu((_EDUCATION_SHARE_OF_SPENDING, None, False, False)),
            {},
            (Alternative(index=1, basis="structural"),),
            id="an-unread-share-is-planned-for-its-own-read-to-decide",
        ),
        pytest.param(
            _GDP_GROWTH,
            _menu((_NOMINAL_GDP, True, False, False)),
            {},
            (Alternative(index=1, basis="structural"),),
            id="a-growth-rate-the-question-names-no-stat-type-for-offers-the-level",
        ),
        pytest.param(
            _GDP_GROWTH,
            _menu((_NOMINAL_GDP, True, True, False)),
            {"stat_type_named": True},
            (),
            id="a-named-growth-rate-offers-no-level",
        ),
        pytest.param(
            _MORTALITY,
            _menu((_UNRELATED, True, True, False)),
            {},
            (Alternative(index=1, basis="nominated"),),
            id="a-nomination-of-another-measure-is-a-rival",
        ),
        pytest.param(
            _MORTALITY,
            _menu((_UN_UNRELATED, True, True, False)),
            {},
            (Alternative(index=1, basis="nominated"),),
            id="a-un-nomination-beside-ours-is-a-rival",
        ),
        pytest.param(
            _UN_UNRELATED,
            _menu((_MORTALITY, True, True, False)),
            {},
            (Alternative(index=1, basis="nominated"),),
            id="our-nomination-beside-a-un-reading-is-a-rival",
        ),
        pytest.param(
            _MORTALITY,
            _menu((_UNRELATED, True, False, False)),
            {},
            (),
            id="another-measure-nobody-nominated-is-no-rival",
        ),
    ],
)
def test_rule_two_offers_only_open_axes_and_nominations(elected, menu, named, rivals):
    assert _plan(elected, menu, **named).rivals == rivals


@pytest.mark.parametrize(
    ("elected", "reading", "basis"),
    [
        pytest.param(_MORTALITY, _UNRELATED, "nominated", id="another-measure"),
        pytest.param(_NOMINAL_GDP, _REAL_GDP, None, id="same-measure-on-a-closed-axis"),
        pytest.param(
            _EDUCATION_SHARE_OF_GDP,
            _EDUCATION_SHARE_OF_SPENDING,
            "structural",
            id="same-measure-on-an-open-axis",
        ),
    ],
)
def test_a_nomination_enters_r_on_its_axes(elected, reading, basis):
    """Rule 2(b): a nominated reading of another measure is a rival, and one of the same
    measure enters R only on an open axis."""
    menu = [MenuReading(index=1, shape=reading, has_data=True, nominated=True)]
    expected = (Alternative(index=1, basis=basis),) if basis else ()
    assert _plan(elected, menu).rivals == expected


def test_a_nominated_registry_twin_is_no_rival():
    menu = [MenuReading(index=1, shape=_IMF_GDP, has_data=True, nominated=True, registry_twin=True)]
    assert _plan(_NOMINAL_GDP, menu).rivals == ()


def test_rule_one_tries_equivalents_then_qualifier_siblings_then_riders():
    """Beside an elected reading with no data, its five-tuple's other reading is an
    equivalent, a price-basis sibling a stand-in, and a nomination of another measure, a
    UN series among them, a rider; a reading with a confirmed absence takes no part."""
    by_creditor = dataclasses.replace(_NOMINAL_GDP, shape_id="by-creditor")
    plan = _plan(
        _NOMINAL_GDP,
        _menu(
            (by_creditor, True, False, False),
            (_REAL_GDP, True, False, False),
            (_UNRELATED, True, True, False),
            (_IMF_GDP, False, False, True),
            (_UN_UNRELATED, True, True, False),
        ),
    )
    assert (plan.twins, plan.equivalents, plan.stand_ins, plan.riders) == (
        (),
        (1,),
        (2,),
        (3, 5),
    )
    assert not plan.sole


@pytest.mark.parametrize(
    ("elected", "sibling", "planned"),
    [
        pytest.param(_NOMINAL_GDP, _RECIPIENT_REAL_GDP, True, id="role-bearing-beside-subject"),
        pytest.param(_RECIPIENT_NOMINAL_GDP, _REAL_GDP, False, id="subject-beside-role-bearing"),
        pytest.param(_RECIPIENT_NOMINAL_GDP, _RECIPIENT_REAL_GDP, True, id="one-role-class"),
        pytest.param(
            dataclasses.replace(
                _NOMINAL_GDP, declared_roles=("financingProvider", "financingRecipient")
            ),
            dataclasses.replace(
                _REAL_GDP, declared_roles=("financingRecipient", "financingProvider")
            ),
            True,
            id="one-role-class-in-another-arc-order",
        ),
    ],
)
def test_a_qualifier_sibling_is_planned_where_its_walk_may_share_the_role_class(
    elected, sibling, planned
):
    """A role-bearing sibling of a subject-only reading is planned, since its walk may elect
    the subject-only family of its five-tuple. A subject-only sibling of a role-bearing
    reading differs from it by a role and is not."""
    plan = _plan(elected, _menu((sibling, True, False, False)))
    assert plan.stand_ins == ((1,) if planned else ())


def test_the_only_reading_with_data_is_sole():
    plan = _plan(_NOMINAL_GDP, _menu((_REAL_GDP, False, True, False)))
    assert plan.sole
    assert not _plan(_NOMINAL_GDP, _menu((_REAL_GDP, None, False, False))).sole


@pytest.mark.parametrize(
    ("shape", "default"),
    [
        (_NOMINAL_GDP, True),
        (dataclasses.replace(_NOMINAL_GDP, meas_qual_dcid=None), True),
        (_REAL_GDP, False),
        (dataclasses.replace(_NOMINAL_GDP, meas_qual_dcid="PurchasingPowerParity"), False),
    ],
)
def test_the_default_price_basis_is_the_declared_first_reading(shape, default):
    assert holds_default_price_basis(shape) is default


def test_an_axis_the_elected_reading_holds_at_its_default_is_closed():
    assert open_axes(
        _NOMINAL_GDP, stat_type_named=False, form_named=False, price_basis_named=False
    ) == OpenAxes(stat_type=False, denominator=False, price_basis=False)
    assert open_axes(
        _REAL_GDP, stat_type_named=False, form_named=False, price_basis_named=True
    ) == OpenAxes(stat_type=False, denominator=False, price_basis=False)
    assert open_axes(
        _REAL_GDP, stat_type_named=False, form_named=False, price_basis_named=False
    ).price_basis


# The pipeline: retrieval scores set no status, and a nomination with data is offered.

_SV_FOREST = "Area_Forest"
_SV_POPULATION = "Count_Person"
_NODES: dict = {
    "country/BRA": {"label": "Brazil", "type": "Country"},
    _SV_FOREST: {
        "label": "Forest Area",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Forest Area"}]},
            "populationType": {"nodes": [{"dcid": "Forest"}]},
            "measuredProperty": {"nodes": [{"dcid": "area"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _SV_POPULATION: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count Person"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}


class _FixedBindLLM:
    """Elects the shape at `chosen`, the first by default, with a caller-supplied
    `also_plausible`, rating its pick at 1."""

    def __init__(self, also_plausible: list[int] | None = None, *, chosen: int = 0) -> None:
        self._also_plausible = also_plausible
        self._chosen = chosen

    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        return _UniversalBindOutput(
            chosen_shape_index=self._chosen,
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
            measuring=[
                ShapeRating(index=self._chosen, rating=1.0),
                *nominee_ratings(self._also_plausible, rated=(self._chosen,)),
            ],
        ), None


# "land use statistic" names what forest area is a kind of, so no word of it names a part
# of what the series counts: the binder rates each word stated.
_LAND_USE_WORDS = ("land", "use", "statistic")


def _land_use_binder() -> RatingBinder:
    return RatingBinder(GeminiBinder(), dict.fromkeys(_LAND_USE_WORDS, 1.0))


def _run(
    variable: str,
    *,
    scores: list[float],
    also_plausible: list[int] | None = None,
    binder: RatingBinder | None = None,
) -> RegionResult:
    svs = [_SV_FOREST, _SV_POPULATION]
    graph = FakeGraph(
        nodes=_NODES,
        detect={variable: {"svs": svs, "cosine_scores": scores}},
        resolve={"Brazil": "country/BRA"},
        obs={
            f"{sv}|country/BRA": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 10}]
            for sv in svs
        },
    )
    return asyncio.run(
        resolve_variable(
            variable,
            entities=["Brazil"],
            date_request=None,
            detect_query=variable,
            role_query=f"{variable} in Brazil",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(also_plausible),
            binder=binder or GeminiBinder(),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_a_thin_retrieval_margin_sets_no_status():
    """Forest area and population scored 0.0046 apart, and nobody nominated population:
    another measure enters the menu only through a nomination."""
    binder = _land_use_binder()
    result = _run("land use statistic", scores=[0.8230, 0.8184], binder=binder)
    assert result.status == "definite"
    assert [spec.resolved.members[0].ref.ref for spec in result.specs] == [_SV_FOREST]
    assert [request.words for request in binder.requests] == [_LAND_USE_WORDS]


def test_a_nomination_with_data_is_offered_as_nominated():
    result = _run("land use statistic", scores=[0.90, 0.70], also_plausible=[1])
    assert result.status == "candidates"
    assert [(spec.resolved.members[0].ref.ref, spec.resolved.basis) for spec in result.specs] == [
        (_SV_FOREST, None),
        (_SV_POPULATION, "nominated"),
    ]


def test_the_only_reading_with_data_is_high_confidence_despite_a_weak_score():
    """Population has no data in Brazil, so forest area is the only admissible reading with
    data where the question asks."""
    graph_obs = {
        f"{_SV_FOREST}|country/BRA": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 10}
        ]
    }
    graph = FakeGraph(
        nodes=_NODES,
        detect={
            "land use statistic": {
                "svs": [_SV_FOREST, _SV_POPULATION],
                "cosine_scores": [0.70, 0.69],
            }
        },
        resolve={"Brazil": "country/BRA"},
        obs=graph_obs,
    )
    binder = _land_use_binder()
    result = asyncio.run(
        resolve_variable(
            "land use statistic",
            entities=["Brazil"],
            date_request=None,
            detect_query="land use statistic",
            role_query="land use statistic in Brazil",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            binder=binder,
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    assert result.specs[0].resolved.confidence == "high"
    assert [request.words for request in binder.requests] == [_LAND_USE_WORDS]


# The admission guards honour the status rule's open axes and run before it, and the
# candidate budget applies once to what the rule offers.

_GDP_GROWTH_SV = "GrowthRate_Amount_EconomicActivity_GrossDomesticProduction"
_GDP_LEVEL_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_PER_CAPITA_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita"
_RURAL_GDP_SV = "Amount_EconomicActivity_GrossDomesticProduction_Rural"
_URBAN_GDP_PER_CAPITA_SV = "Amount_EconomicActivity_GrossDomesticProduction_Urban_PerCapita"
_POPULATION_GROWTH_SV = "GrowthRate_Count_Person"


def _gdp_sv(label: str, stat_type: str, *, per_capita: bool = False) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": stat_type}]},
        "economicActivityType": {"nodes": [{"dcid": _GDP}]},
    }
    if per_capita:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": "Count_Person"}]}
    return {"label": label, "type": "StatisticalVariable", "arcs": arcs}


_GDP_NODES: dict = {
    "country/BRA": {"label": "Brazil", "type": "Country"},
    _GDP_GROWTH_SV: _gdp_sv("Gross domestic product growth rate", "growthRate"),
    _GDP_LEVEL_SV: _gdp_sv("Gross domestic product", "measuredValue"),
    _GDP_PER_CAPITA_SV: _gdp_sv(
        "Gross domestic product per capita", "measuredValue", per_capita=True
    ),
    _RURAL_GDP_SV: _gdp_sv("Rural gross domestic product", "measuredValue"),
    _URBAN_GDP_PER_CAPITA_SV: _gdp_sv(
        "Urban gross domestic product per capita", "measuredValue", per_capita=True
    ),
    _POPULATION_GROWTH_SV: {
        "label": "Population growth rate",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Population growth rate"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "growthRate"}]},
        },
    },
}


@pytest.mark.parametrize(
    ("svs", "max_candidates", "offered", "truncated"),
    [
        pytest.param(
            (_GDP_GROWTH_SV, _GDP_LEVEL_SV),
            6,
            [(_GDP_GROWTH_SV, None), (_GDP_LEVEL_SV, "structural")],
            False,
            id="open-stat-type-offers-the-level",
        ),
        pytest.param(
            (_GDP_GROWTH_SV, _RURAL_GDP_SV, _URBAN_GDP_PER_CAPITA_SV, _POPULATION_GROWTH_SV),
            3,
            [(_GDP_GROWTH_SV, None), (_POPULATION_GROWTH_SV, "nominated")],
            False,
            id="guarded-rivals-take-no-budget",
        ),
        pytest.param(
            (_GDP_GROWTH_SV, _GDP_LEVEL_SV, _GDP_PER_CAPITA_SV, _POPULATION_GROWTH_SV),
            3,
            [
                (_GDP_GROWTH_SV, None),
                (_GDP_LEVEL_SV, "structural"),
                (_GDP_PER_CAPITA_SV, "structural"),
            ],
            True,
            id="the-budget-cut-is-truncation",
        ),
    ],
)
def test_the_guards_read_the_rules_open_axes_before_the_budget(
    monkeypatch, svs, max_candidates, offered, truncated
):
    """ "GDP" in Brazil elects the growth rate and names no stat type, so the stat-type axis
    is open and the level is a structural rival the guard admits. Readings the guard
    refuses (rural and urban GDP, which narrow the residence neither the question nor the
    election names) take no place in the budget, so the nominated population growth rate
    beside them is offered. When the budget drops an admissible rival, the part says it
    truncated its candidates."""
    monkeypatch.setattr(settings(), "max_candidates", max_candidates)
    nominated = [i for i, sv in enumerate(svs) if sv == _POPULATION_GROWTH_SV] or None
    graph = FakeGraph(
        nodes={key: _GDP_NODES[key] for key in ("country/BRA", *svs)},
        detect={
            "GDP": {"svs": list(svs), "cosine_scores": [0.90 - 0.01 * i for i in range(len(svs))]}
        },
        resolve={"Brazil": "country/BRA"},
        obs={
            f"{sv}|country/BRA": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 10}]
            for sv in svs
        },
    )
    result = asyncio.run(
        resolve_variable(
            "GDP",
            entities=["Brazil"],
            date_request=None,
            detect_query="GDP",
            role_query="GDP in Brazil",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(nominated),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "candidates"
    assert [
        (spec.resolved.members[0].ref.ref, spec.resolved.basis) for spec in result.specs
    ] == offered
    assert result.candidate_budget_truncated is truncated


def test_a_re_elected_stand_in_is_not_its_own_alternative():
    """The mortality count in Kenya has no data and its estimated count, the only
    stand-in, answers in its place. The winner is the only admissible reading with data, so it grades
    high at a weak retrieval score."""
    scenario = dataclasses.replace(scenarios._SCENARIOS["named-reelect-adopt"], scores=(0.70, 0.69))
    result = scenarios._resolve(scenario)
    assert result.status == "definite"
    assert [spec.resolved.confidence for spec in result.specs] == ["high"]


# Two shapes the bind menu states as one reading: the rank that orders the menu elects.

_GDP_PER_HEAD_SV = "Amount_EconomicActivity_GrossDomesticProduction_AsAFractionOf_Count_Person"
_INCOME_PER_CAPITA_SV = "Amount_Income_PerCapita"


def _per_person_sv(label: str, pop_type: str, denominator: str) -> dict:
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": pop_type}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementDenominator": {"nodes": [{"dcid": denominator}]},
        },
    }


@pytest.mark.parametrize(
    ("variable", "also_plausible"),
    [("GDP per capita", [2]), ("GDP per capita", [0, 2]), ("living standard", [2])],
    ids=["unnamed", "nominated", "open-denominator"],
)
def test_the_rank_elects_between_two_shapes_the_menu_states_alike(variable, also_plausible):
    """ "GDP per capita" in Brazil holds one label twice: a series dividing by the
    ProvisionalNode `PerCapita`, ranked first, and one dividing by `Count_Person`. The menu
    prints both as per person, and the model elects the second. The first is elected in
    its place, whether or not the model also nominated it, and the second is no rival. It
    is the same reading, on a denominator "GDP per capita" closes and "living standard",
    which names no form, leaves open. The nominated income series is offered beside it."""
    svs = [_GDP_PER_CAPITA_SV, _GDP_PER_HEAD_SV, _INCOME_PER_CAPITA_SV]
    graph = FakeGraph(
        nodes={
            "country/BRA": {"label": "Brazil", "type": "Country"},
            _GDP_PER_CAPITA_SV: _per_person_sv("GDP per capita", "EconomicActivity", "PerCapita"),
            _GDP_PER_HEAD_SV: _per_person_sv("GDP per capita", "EconomicActivity", "Count_Person"),
            _INCOME_PER_CAPITA_SV: _per_person_sv("Income per capita", "Income", "PerCapita"),
        },
        detect={variable: {"svs": svs, "cosine_scores": [0.90, 0.89, 0.80]}},
        resolve={"Brazil": "country/BRA"},
        obs={
            f"{sv}|country/BRA": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 10}]
            for sv in svs
        },
    )
    result = asyncio.run(
        resolve_variable(
            variable,
            entities=["Brazil"],
            date_request=None,
            detect_query=variable,
            role_query=f"{variable} in Brazil",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(also_plausible, chosen=1),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "candidates"
    assert [
        (spec.resolved.members[0].ref.ref, spec.resolved.shape.shape_id, spec.resolved.basis)
        for spec in result.specs
    ] == [
        (_GDP_PER_CAPITA_SV, "economicactivity_amount_measuredvalue_per_percapita", None),
        (_INCOME_PER_CAPITA_SV, "income_amount_measuredvalue_per_percapita", "nominated"),
    ]


# The direction a question leaves unstated: one availability read per role assignment.

_NOR = scenarios._NOR
# Two readings of one five-tuple: income a place pays, and income it receives. Together they
# give one shape declaring both roles and requiring neither, so a place named without a
# direction can take either.
_PAID = "Amount_EconomicActivity_GrossNationalIncome_Nominal_PaidScenario"
_RECEIVED = "Amount_EconomicActivity_GrossNationalIncome_Nominal_ReceivedScenario"
_INCOME = {"economicActivityType": "GrossNationalIncome"}
_PAID_ROW = (_PAID, "f_paid", _NOR, "", [2020, 2021, 2022])
_RECEIVED_ROW = (_RECEIVED, "f_received", "", _NOR, [2020, 2021, 2022])


@pytest.fixture
def _paid_and_received(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scenarios,
        "_NODES",
        {
            **scenarios._NODES,
            _PAID: scenarios._sv(
                "Gross national income paid",
                scenarios._income_arcs("Nominal", _INCOME, ("financingProvider",)),
            ),
            _RECEIVED: scenarios._sv(
                "Gross national income received",
                scenarios._income_arcs("Nominal", _INCOME, ("financingRecipient",)),
            ),
        },
    )
    monkeypatch.setattr(
        scenarios,
        "_ROLE_NAMES",
        {
            **scenarios._ROLE_NAMES,
            _PAID: RoleNames(donor="financingProvider", recipient=None),
            _RECEIVED: RoleNames(donor=None, recipient="financingRecipient"),
        },
    )


def _income_of_norway(
    *rows: tuple[str, str, str, str, list[int]],
    scores: tuple[float, float] = (0.85, 0.84),
    date_request: DateRequest | None = None,
) -> RegionResult:
    """Gross national income with Norway named without a direction, recalled at `scores`,
    the coverage `rows`, and the years `date_request` asks for."""
    return scenarios._resolve(
        scenarios._Scenario(
            query="gross national income Norway",
            entities=("Norway",),
            svs=(_PAID, _RECEIVED),
            scores=scores,
            coverage_rows=rows,
            date_request=date_request,
        )
    )


def _norway_role(result: RegionResult) -> list[str]:
    return [
        next(role for role, dcids in spec.definition.identity.entities.items() if _NOR in dcids)
        for spec in result.specs
    ]


_IN_2022 = DateRequest(window=TimeWindow(start_year=2022, end_year=2022), latest=False)


@pytest.mark.usefixtures("_paid_and_received")
@pytest.mark.parametrize(
    ("rows", "date_request", "axis"),
    [
        ((_PAID_ROW,), None, "origin"),
        ((_RECEIVED_ROW,), None, "destination"),
        (
            (
                (_PAID, "f_paid", _NOR, "", [2010]),
                (_RECEIVED, "f_received", "", _NOR, [2022]),
            ),
            _IN_2022,
            "destination",
        ),
    ],
    ids=["paid", "received", "received-in-the-asked-year"],
)
def test_one_direction_with_data_answers_and_states_the_direction_it_inferred(
    rows, date_request, axis
):
    """A direction holds data when its coverage holds a year the question asks for. Income
    paid in 2010 and received in 2022 answers a 2022 question on the destination axis."""
    result = _income_of_norway(*rows, date_request=date_request)
    assert result.status == "definite"
    assert _norway_role(result) == [axis]
    assert [(fact.code, fact.asked, fact.over) for fact in result.inferred_facts] == [
        ("direction_inferred", "Norway", axis)
    ]


@pytest.mark.usefixtures("_paid_and_received")
@pytest.mark.parametrize(
    ("scores", "confidence"), [((0.85, 0.84), "high"), ((0.7, 0.69), "medium")]
)
def test_both_directions_with_data_offer_a_direction_candidate(scores, confidence):
    """The elected reading and its other direction grade alike. A direction rival is no
    structural or nominated rival, so a weak score grades both medium, neither low
    (`_grade_confidence`)."""
    result = _income_of_norway(_PAID_ROW, _RECEIVED_ROW, scores=scores)
    assert result.status == "candidates"
    assert _norway_role(result) == ["origin", "destination"]
    assert [spec.resolved.basis for spec in result.specs] == [None, "direction"]
    assert [spec.resolved.confidence for spec in result.specs] == [confidence, confidence]
    assert result.inferred_facts == ()


@pytest.mark.usefixtures("_paid_and_received")
def test_no_direction_with_data_refuses_the_unstated_direction():
    result = _income_of_norway()
    assert (result.status, result.no_data_reason) == ("no_data", "role_not_resolved")


_KEN = scenarios._KEN
_ASSISTANCE_GIVEN = "Amount_EconomicActivity_Assistance_GivenScenario"
_REFUGEE_ASSISTANCE = "Amount_EconomicActivity_RefugeeAssistance_ReceivedScenario"


class _RejectWordsBind(scenarios._Bind):
    """`scenarios._Bind` whose stated-words read rates every word not stated, counting its
    calls."""

    word_calls = 0

    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            type(self).word_calls += 1
            return StatedWordsOutput(
                words=[StatedWord(index=i, stated=False) for i in range(20)]
            ), None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def test_a_direction_settled_after_the_bind_judges_the_words_on_the_member_it_serves(
    monkeypatch: pytest.MonkeyPatch,
):
    """ "refugee assistance Kenya" over one shape whose members serve each direction:
    "Assistance given" a place provides and "Refugee assistance received" a place
    receives. Kenya is named without a direction, and only the assistance it receives has
    data, so the direction settles on the recipient after the bind. The words are judged on
    the member the read served, whose name states "refugee", so the binder is never asked
    and the answer stands, as it does when the question names the direction."""
    monkeypatch.setattr(
        scenarios,
        "_NODES",
        {
            **scenarios._NODES,
            **{
                dcid: scenarios._sv(
                    name,
                    {
                        **scenarios._income_arcs("Nominal", _INCOME, (role,)),
                        "name": {"nodes": [{"value": name}]},
                    },
                )
                for dcid, name, role in (
                    (_ASSISTANCE_GIVEN, "Assistance given", "financingProvider"),
                    (_REFUGEE_ASSISTANCE, "Refugee assistance received", "financingRecipient"),
                )
            },
        },
    )
    monkeypatch.setattr(
        scenarios,
        "_ROLE_NAMES",
        {
            **scenarios._ROLE_NAMES,
            _ASSISTANCE_GIVEN: RoleNames(donor="financingProvider", recipient=None),
            _REFUGEE_ASSISTANCE: RoleNames(donor=None, recipient="financingRecipient"),
        },
    )
    monkeypatch.setattr(scenarios, "_Bind", _RejectWordsBind)
    monkeypatch.setattr(_RejectWordsBind, "word_calls", 0)

    result = scenarios._resolve(
        scenarios._Scenario(
            query="refugee assistance Kenya",
            entities=("Kenya",),
            svs=(_ASSISTANCE_GIVEN, _REFUGEE_ASSISTANCE),
            scores=(0.85, 0.84),
            coverage_rows=((_REFUGEE_ASSISTANCE, "f_received", "", _KEN, [2020, 2021, 2022]),),
        )
    )

    assert result.status == "definite"
    assert [sv.ref.ref for spec in result.specs for sv in spec.resolved.members] == [
        _REFUGEE_ASSISTANCE
    ]
    assert _RejectWordsBind.word_calls == 0


_ADJUSTED_PAID = "AdjustedAmount_EconomicActivity_GrossNationalIncome_PaidScenario"
_ADJUSTED_RECEIVED = "AdjustedAmount_EconomicActivity_GrossNationalIncome_ReceivedScenario"


@pytest.mark.usefixtures("_paid_and_received")
def test_an_alternate_binds_on_the_direction_the_elected_reading_was_served_on(monkeypatch):
    """Norway is named without a direction, and only the income it pays has data, so the
    elected reading is served on the origin axis. The adjusted income the model nominates
    measures another property and declares the same two optional roles. It binds Norway on
    the served axis, has data there, and is offered beside the elected reading."""
    adjusted = {
        _ADJUSTED_PAID: ("Gross national income paid, adjusted", ("financingProvider",)),
        _ADJUSTED_RECEIVED: ("Gross national income received, adjusted", ("financingRecipient",)),
    }
    monkeypatch.setattr(
        scenarios,
        "_NODES",
        {
            **scenarios._NODES,
            **{
                dcid: scenarios._sv(
                    label,
                    {
                        **scenarios._income_arcs("Nominal", _INCOME, roles),
                        "measuredProperty": {"nodes": [{"dcid": "adjustedAmount"}]},
                    },
                )
                for dcid, (label, roles) in adjusted.items()
            },
        },
    )
    monkeypatch.setattr(
        scenarios,
        "_ROLE_NAMES",
        {
            **scenarios._ROLE_NAMES,
            _ADJUSTED_PAID: RoleNames(donor="financingProvider", recipient=None),
            _ADJUSTED_RECEIVED: RoleNames(donor=None, recipient="financingRecipient"),
        },
    )
    result = scenarios._resolve(
        scenarios._Scenario(
            query="gross national income Norway",
            entities=("Norway",),
            svs=(_PAID, _RECEIVED, _ADJUSTED_PAID, _ADJUSTED_RECEIVED),
            scores=(0.85, 0.84, 0.80, 0.79),
            also_plausible=(1,),
            coverage_rows=(_PAID_ROW, (_ADJUSTED_PAID, "f_adjusted", _NOR, "", [2020, 2021, 2022])),
        )
    )

    assert result.status == "candidates"
    assert _norway_role(result) == ["origin", "origin"]
    assert [spec.resolved.basis for spec in result.specs] == [None, "nominated"]
    assert [[sv.ref.ref for sv in spec.resolved.members] for spec in result.specs] == [
        [_PAID],
        [_ADJUSTED_PAID],
    ]


# DAC1 states ODA as a share of the donor's GNI and DAC2a as a share of the recipient's
# (`DatasetDeclaration.ratio_base_axis`). Both are code-index rows of the fixture index.
_DAC1_GNI_SHARE = (
    "Amount_FinancialFlow_Net_Disbursement_DAC1AidType1010_Nominal_AsAFractionOf_"
    "PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal"
)
_DAC2A_GNI_SHARE = (
    "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal_AsAFractionOf_"
    "PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal"
)
_KEN = scenarios._KEN
# Norway gives to Kenya, and receives nothing.
_GNI_SHARE_ROWS = (
    (_DAC1_GNI_SHARE, "f_dac1", _NOR, "", [2020, 2021, 2022]),
    (_DAC2A_GNI_SHARE, "f_dac2a", _NOR, _KEN, [2020, 2021, 2022]),
)


def _index_row_node(dcid: str) -> dict:
    """The graph node of the fixture index row `dcid`, with the arcs the row declares."""
    row = next(r for r in get_index().statvars if r.dcid == dcid)
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "measurementDenominator": {"nodes": [{"dcid": row.measurement_denominator}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
        "observationProperties": {"nodes": [{"dcid": r} for r in row.observation_properties]},
    }
    return scenarios._sv(dcid, arcs)


@pytest.fixture
def _gni_shares(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scenarios,
        "_NODES",
        {
            **scenarios._NODES,
            _DAC1_GNI_SHARE: _index_row_node(_DAC1_GNI_SHARE),
            _DAC2A_GNI_SHARE: _index_row_node(_DAC2A_GNI_SHARE),
        },
    )
    monkeypatch.setattr(
        scenarios,
        "_ROLE_NAMES",
        {
            **scenarios._ROLE_NAMES,
            _DAC1_GNI_SHARE: RoleNames(donor="financingProvider", recipient=None),
            _DAC2A_GNI_SHARE: RoleNames(donor="financingProvider", recipient="financingRecipient"),
        },
    )


@pytest.mark.usefixtures("_gni_shares")
@pytest.mark.parametrize(
    ("place", "rows", "answer"),
    [
        pytest.param(
            "Norway",
            _GNI_SHARE_ROWS,
            [(_DAC1_GNI_SHARE, "origin")],
            id="norway-reads-the-donor-share",
        ),
        pytest.param(
            "Kenya",
            _GNI_SHARE_ROWS,
            [(_DAC2A_GNI_SHARE, "destination")],
            id="kenya-reads-the-recipient-share",
        ),
        pytest.param("Norway", _GNI_SHARE_ROWS[1:], [], id="norway-holds-no-recipient-share"),
    ],
)
def test_a_place_named_without_a_direction_owns_the_base_of_a_ratio(place, rows, answer):
    """ "ODA as a share of GNI for Norway" names Norway without a direction. A ratio whose
    dataset declares whose total it divides by reads the place as that party: Norway is the
    donor on DAC1's share of the donor's GNI, and the recipient on DAC2a's share of the
    recipient's, where it holds no data, so DAC2a leaves the menu and DAC1 answers. Where
    DAC1 holds no data for Norway either, the part answers no data, and does not read
    Norway's aid to others as a share of their GNI. Kenya is DAC2a's recipient, and DAC1
    holds no recipient."""
    result = scenarios._resolve(
        scenarios._Scenario(
            query="official development assistance",
            entities=(place,),
            # Recall ranks DAC2a first, and the binder elects the first menu shape.
            svs=(_DAC2A_GNI_SHARE, _DAC1_GNI_SHARE),
            scores=(0.85, 0.84),
            coverage_rows=rows,
        )
    )

    dcid = scenarios._RESOLVE[place]
    assert [
        (
            spec.resolved.members[0].ref.ref,
            next(
                role for role, dcids in spec.definition.identity.entities.items() if dcid in dcids
            ),
        )
        for spec in result.specs
    ] == answer
    assert result.status == ("definite" if answer else "no_data")


@pytest.fixture
def _ids_at_nominal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scenario world's IDS income series at `Nominal`, the declared default price basis
    their live counterparts carry."""
    ids = {
        scenarios._IDS: scenarios._sv(
            "Gross national income (current US$)",
            scenarios._income_arcs("Nominal", _INCOME, ("financingRecipient",)),
        ),
        scenarios._IDS_BC: scenarios._sv(
            "Gross national income (current US$), by creditor",
            scenarios._income_arcs("Nominal", _INCOME, ("financingProvider", "financingRecipient")),
        ),
    }
    monkeypatch.setattr(scenarios, "_NODES", {**scenarios._NODES, **ids})


@pytest.mark.usefixtures("_ids_at_nominal")
@pytest.mark.parametrize(
    ("scenario_id", "status", "members", "bases"),
    [
        (
            "income-role-bearing-elected-subject-alternate",
            "definite",
            [[scenarios._IDS]],
            [None],
        ),
        (
            "income-subject-elected-role-bearing-alternate",
            "candidates",
            [[scenarios._PPP], [scenarios._IDS]],
            [None, "structural"],
        ),
    ],
    ids=["nominal-elected", "ppp-elected"],
)
def test_ids_income_at_the_default_price_basis_closes_the_axis(scenario_id, status, members, bases):
    """Gross national income of Ghana, with the IDS series at `Nominal`. Elected, it holds
    the declared default price basis, so the axis is closed and the PPP reading the model
    nominates differs from it there and stays off the answer. With the PPP reading elected
    the axis is open, and the IDS reading is offered beside it as a structural rival."""
    result = scenarios._resolve(scenarios._SCENARIOS[scenario_id])

    assert result.status == status
    assert [[sv.ref.ref for sv in spec.resolved.members] for spec in result.specs] == members
    assert [spec.resolved.basis for spec in result.specs] == bases
