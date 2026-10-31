"""Checks the unit a heading states: only a unit that changes the magnitude."""

from __future__ import annotations

import pytest

from search.models import UNIT_REF, ResolvedFacet, Shape, StatVar
from search.render.heading import heading_from_parts
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import plan_tokens, sentence_from_plan
from tests.render._helpers import _ref, _rendered, _spec, assert_round_trip

_LIVE_BIRTHS = _ref("TotalReportedLiveBirths", "Total Number Of Reported Live Births")
_GNI = _ref("PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal", "GNI")


def _unit(ref: str, label: str | None, currency: str | None = None) -> dict[str, ResolvedFacet]:
    return {"unit": ResolvedFacet(basis="fallback", ref=ref, label=label, currency=currency)}


def _provenance(ref: str) -> dict[str, ResolvedFacet]:
    return {"provenance": ResolvedFacet(basis="fallback", ref=ref)}


@pytest.mark.parametrize(
    "facets",
    [
        pytest.param(_unit("USDollar", "US dollar", "USD"), id="us-dollar"),
        pytest.param(_unit("CurrencyUSD", "US Dollar", "USD"), id="us-dollar-currency-unit"),
        pytest.param(_unit("MetricTonCO2e", None), id="other-unit"),
        pytest.param(_unit("Year", None), id="single-word-unit"),
        pytest.param(
            {"unit": ResolvedFacet(basis="fallback", selection="per_series")}, id="per-series"
        ),
        pytest.param({}, id="no-unit"),
    ],
)
def test_a_unit_that_keeps_the_magnitude_stays_out_of_the_heading(facets):
    """US dollars, units outside the magnitude classes, and a unit whose series disagree
    leave the heading; the byline carries a plain unit."""
    spec = _spec(measured_property=_ref("gdp", "Gross domestic product"), facets=facets)
    plan = render_sentence_plan(spec)
    heading = [t for t in plan if t.segment == "heading"]
    assert sentence_from_plan(heading) == "Gross domestic product."
    assert not any(t.axis == "unit" for t in heading)


@pytest.mark.parametrize("denominator", ["Count_Person", "PerCapita"])
def test_a_per_head_denominator_reads_per_capita(denominator: str):
    """Both per-head denominators read "per capita" under a money unit."""
    spec = _spec(
        measured_property=_ref("valueAdded", "Value added"),
        measurement_denominator=_ref(denominator, "Total population"),
        facets=_unit("USDollar", "US dollar"),
    )
    assert _rendered(spec) == "Value added per capita."


@pytest.mark.parametrize(
    ("denominator", "unit", "expected"),
    [
        pytest.param(
            _ref("Count_Person", "Total population"),
            _unit("Per1000Persons", "Per 1000 people"),
            "Crude birth rate per 1,000 people.",
            id="per-1000-people-over-count-person",
        ),
        pytest.param(
            _ref("Count_BirthEvent_LiveBirth", "Live births"),
            _unit("Per1000LiveBirths", "Per 1000 live births"),
            "Crude birth rate per 1,000 live births.",
            id="per-1000-live-births-over-live-births",
        ),
        pytest.param(
            None,
            _unit(
                "undata/UNIT_MEASURE-RATIO_COUNT_PER_10000_COUNT_POP", "Count per 10,000 persons"
            ),
            "Crude birth rate per 10,000 people.",
            id="un-unit-with-no-denominator",
        ),
        pytest.param(
            None,
            _unit("undata/UNIT_MEASURE-RATE_DEATH_PER_1000_COUNT", "Death rate per 1,000"),
            "Crude birth rate per 1,000.",
            id="base-of-nothing-the-ref-states",
        ),
    ],
)
def test_a_rate_states_its_base(denominator, unit, expected):
    """A rate reads its base in place of "per capita" or "relative to" the denominator, and
    its token states the unit."""
    spec = _spec(
        measured_property=_ref("crudeBirthRate", "Crude birth rate"),
        measurement_denominator=denominator,
        facets=unit,
    )
    plan = render_sentence_plan(spec)
    assert sentence_from_plan([t for t in plan if t.segment == "heading"]) == expected
    (token,) = [t for t in plan if t.axis == "unit"]
    assert token.states == [UNIT_REF]


@pytest.mark.parametrize(
    ("label", "unit", "expected"),
    [
        pytest.param(
            "Homicides per 100,000 population",
            "Per100000Persons",
            "Homicides per 100,000 population.",
            id="digits",
        ),
        pytest.param(
            "Researchers per million inhabitants",
            "SDG_PER_1000000_POP",
            "Researchers per million inhabitants.",
            id="a-word-for-the-base",
        ),
    ],
)
def test_a_rate_the_graph_name_states_reads_once(label, unit, expected):
    spec = _spec(
        measured_property=_ref("value", "value"),
        facets=_unit(unit, None),
        stat_vars=[
            StatVar(
                ref=_ref("Value_Rate", label),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    assert _rendered(spec) == expected


def test_count_person_under_an_unknown_unit_states_no_scale():
    spec = _spec(
        measured_property=_ref("populationDensity", "Population density"),
        measurement_denominator=_ref("Count_Person", "Total population"),
        facets=_unit("PersonPerSquareKilometer", None),
    )
    assert _rendered(spec) == "Population density."


def test_purchasing_power_per_capita_is_one_unit_token():
    """A per-head rate in PPP international dollars states both in one heading token on
    the unit axis, which states the unit."""
    spec = _spec(
        measured_property=_ref("valueAdded", "Value added"),
        measurement_denominator=_ref("Count_Person", "Total population"),
        facets=_unit("PPPInternationalDollar", "PPP international dollar"),
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Value added per capita in PPP international dollars."
    unit_tokens = [t for t in plan if t.axis == "unit"]
    assert [t.text for t in unit_tokens] == ["per capita in PPP international dollars"]
    assert unit_tokens[0].segment == "heading"
    assert (unit_tokens[0].binding_ref, unit_tokens[0].states) == (UNIT_REF, [UNIT_REF])


@pytest.mark.parametrize(
    ("unit", "states"),
    [
        pytest.param(_unit("Percent", "Percent"), [UNIT_REF], id="share-names-the-percent"),
        pytest.param({}, [], id="ratio-names-only-the-denominator"),
    ],
)
def test_a_ratio_token_states_the_unit_only_when_it_names_it(unit, states):
    """ "As a share of" says the number is a percent. "Relative to" names only the
    denominator, so its token states nothing."""
    spec = _spec(
        measured_property=_ref("debtService", "Debt service"),
        measurement_denominator=_LIVE_BIRTHS,
        facets=unit,
    )
    (token,) = [t for t in render_sentence_plan(spec) if t.axis == "unit"]
    assert token.states == states


@pytest.mark.parametrize(
    ("denominator", "unit", "expected"),
    [
        pytest.param(
            _ref(
                "Amount_EconomicActivity_ExportsOfGoodsServicesAndPrimaryIncome_Nominal",
                "Exports",
            ),
            _unit("Percent", "Percent"),
            "Debt service as a share of exports of goods, services and primary income.",
            id="authored-denominator-noun",
        ),
        pytest.param(
            _LIVE_BIRTHS,
            _unit("Percent", "Percent"),
            "Debt service as a share of total number of reported live births.",
            id="percent-of-unauthored-denominator",
        ),
        pytest.param(
            _LIVE_BIRTHS,
            {},
            "Debt service relative to total number of reported live births.",
            id="ratio-without-percent",
        ),
        pytest.param(None, _unit("Percent", "Percent"), "Debt service in percent.", id="percent"),
        pytest.param(
            _GNI,
            {**_unit("Percent", "Percent"), **_provenance("provenance/OECDDAC1")},
            "Debt service as a share of the provider's GNI.",
            id="share-of-the-providers-own-total",
        ),
        pytest.param(
            _GNI,
            {**_unit("Percent", "Percent"), **_provenance("provenance/OECDDAC2A")},
            "Debt service as a share of the recipient's GNI.",
            id="share-of-the-recipients-own-total",
        ),
        pytest.param(
            _GNI,
            {
                **_unit("Percent", "Percent"),
                **_provenance("provenance/OECDCreditorReportingSystem"),
            },
            "Debt service as a share of GNI.",
            id="undeclared-base",
        ),
    ],
)
def test_a_ratio_or_percent_stays_in_the_heading(denominator, unit, expected: str):
    """A share of another series, a ratio, or a percent changes what the number means. A
    dataset that declares whose total its ratio divides by (`ratio_base_axis`) names that
    party, so DAC2a's share of the recipient's GNI does not read as the provider's."""
    spec = _spec(
        measured_property=_ref("debtService", "Debt service"),
        measurement_denominator=denominator,
        facets=unit,
    )
    assert _rendered(spec) == expected


def test_a_national_currency_stays_out_of_the_heading():
    """A currency other than US dollars reads in the byline, so the heading's noun leaves
    it out."""
    gdp = _ref("gdp", "Gross domestic product")
    spec = _spec(measured_property=gdp, facets=_unit("CurrencyKES", "Kenyan shilling", "KES"))
    plan = render_sentence_plan(spec)
    assert sentence_from_plan([t for t in plan if t.segment == "heading"]) == (
        "Gross domestic product."
    )
    assert [t.text for t in plan if t.axis == "unit"] == ["Kenyan shillings"]


@pytest.mark.parametrize(
    ("measured_property", "denominator", "unit", "expected"),
    [
        pytest.param(
            _ref("overweightPrevalence", "Overweight prevalence"),
            _LIVE_BIRTHS,
            _unit("Percent", "Percent"),
            "Overweight prevalence.",
            id="prevalence-states-the-share",
        ),
        pytest.param(
            _ref("alcohol", "Percentage of population who consume alcohol"),
            None,
            _unit("Percent", "Percent"),
            "Percentage of population who consume alcohol.",
            id="percentage-states-the-percent",
        ),
    ],
)
def test_a_heading_never_states_a_share_twice(measured_property, denominator, unit, expected: str):
    """Ratio words in the noun already state the share."""
    spec = _spec(
        measured_property=measured_property, measurement_denominator=denominator, facets=unit
    )
    assert _rendered(spec) == expected


@pytest.mark.parametrize(
    ("measure_label", "expected"),
    [
        pytest.param(None, "Share of grants.", id="graph-name"),
        pytest.param("share of grants", "Share of grants in percent.", id="authored-noun"),
    ],
)
def test_only_a_graph_name_is_read_for_its_unit(measure_label, expected: str):
    """A graph name's ratio words state the share; an authored noun or measure label is
    never read as text, so the unit states the percent."""
    spec = _spec(
        measured_property=_ref("shareOfGrants", "Share of grants"),
        measure_label=measure_label,
        facets=_unit("Percent", "Percent"),
    )
    assert _rendered(spec) == expected


def test_a_heading_never_states_per_capita_twice():
    """A StatVar label that names "per capita" leaves the per-head unit unstated."""
    spec = _spec(
        measured_property=_ref("emissions", "emissions"),
        measurement_denominator=_ref("PerCapita", "Per capita"),
        facets=_unit("USDollar", "US dollar"),
        stat_vars=[
            StatVar(
                ref=_ref("Emissions_PerCapita_CO2", "CO2 Emissions Per Capita"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    assert _rendered(spec) == "CO2 emissions per capita."


def _shape(denominator) -> Shape:
    return Shape(
        shape_id="test-shape",
        label="Test Shape",
        population_type=None,
        measured_property=_ref("adolescentBirthRate", "Adolescent birth rate"),
        stat_type=_ref("measuredValue", "Measured Value"),
        measurement_qualifier=None,
        measurement_denominator=denominator,
        slot_keys=[],
        member_count=1,
    )


def test_a_pre_materialise_plan_states_only_the_per_head_unit():
    """Before facets resolve (`facets=None`, the bind-time sentence event) only a
    `PerCapita` denominator fixes a per-head rate, so "per capita" reads. A `Count_Person`
    denominator waits for the unit, which may state a base ("per 1,000 people"), and a
    ratio waits for the unit, which decides "as a share of" against "relative to". An empty
    mapping is a resolved answer and states the ratio."""
    per_head = plan_tokens(
        heading_from_parts(
            [], [], _shape(_ref("PerCapita", "")), window=None, pinned_properties=frozenset()
        )
    )
    assert sentence_from_plan(per_head) == "Adolescent birth rate per capita."
    per_person = plan_tokens(
        heading_from_parts(
            [], [], _shape(_ref("Count_Person", "")), window=None, pinned_properties=frozenset()
        )
    )
    assert sentence_from_plan(per_person) == "Adolescent birth rate."
    pre = plan_tokens(
        heading_from_parts([], [], _shape(_LIVE_BIRTHS), window=None, pinned_properties=frozenset())
    )
    assert sentence_from_plan(pre) == "Adolescent birth rate."
    resolved = plan_tokens(
        heading_from_parts(
            [], [], _shape(_LIVE_BIRTHS), window=None, facets={}, pinned_properties=frozenset()
        )
    )
    assert_round_trip(resolved, sentence_from_plan(resolved))
    assert (
        sentence_from_plan(resolved)
        == "Adolescent birth rate relative to total number of reported live births."
    )
