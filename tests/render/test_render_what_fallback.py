"""Pins the generic-shape `what` fallback and when a resolved StatVar label replaces it."""

from __future__ import annotations

import pytest

from search.models import StatVar
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import (
    _gender_slot,
    _how_slot,
    _ref,
    _rendered,
    _spec,
    _what_slot,
    _where_slot,
    assert_round_trip,
)


def test_stopword_measured_property_names_differ_by_sv_not_generic_count():
    """Shapes sharing a stopword measured_property like "count" are disambiguated
    by naming each with its resolved StatVar, keeping structurally distinct
    candidates distinguishable."""
    spec_population = _spec(
        measured_property=_ref("count", "count"),
        stat_vars=[
            StatVar(
                ref=_ref("Count_Person", "Total population"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
        ],
    )
    spec_growth = _spec(
        measured_property=_ref("count", "count"),
        stat_vars=[
            StatVar(
                ref=_ref("GrowthRate_Person", "Rate of Population Growth"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
        ],
    )
    result_population = _rendered(spec_population)
    result_growth = _rendered(spec_growth)
    assert result_population == "Total population."
    assert result_growth == "Rate of population growth."
    assert result_population != result_growth


def test_stopword_measured_property_with_two_svs_falls_to_population_type():
    """A stopword measured_property with more than one candidate StatVar can't
    pick either by name -- falls to the shape's population_type instead."""
    spec = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Household", "Household"),
        stat_vars=[
            StatVar(
                ref=_ref("sv1", "SV One"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
            StatVar(
                ref=_ref("sv2", "SV Two"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
        ],
    )
    assert _rendered(spec) == "Household."


def test_nonstopword_measured_property_humanized_no_how_slot():
    """A non-stopword measured_property with label==dcid (the raw-dcid fallback
    signal) is humanized rather than shown as a bare camelCase dcid."""
    spec = _spec(measured_property=_ref("fertilityRate", "fertilityRate"))
    assert _rendered(spec) == "Fertility rate."


def test_nonstopword_measured_property_humanized_with_how_slot():
    spec = _spec(
        measured_property=_ref("fertilityRate", "fertilityRate"),
        slots=[_how_slot("Female")],
    )
    assert _rendered(spec) == "Fertility rate for Female."


@pytest.mark.parametrize(
    (
        "property_dcid",
        "property_label",
        "stat_type",
        "place",
        "stat_var_dcid",
        "stat_var_label",
        "expected",
    ),
    [
        pytest.param(
            "age",
            "age",
            _ref("medianValue", "medianValue"),
            "Togo",
            "Median_Age_Person",
            "Median age of population",
            "Togo's median age of population.",
            id="richer-label-suppresses-stat-type-extra",
        ),
        pytest.param(
            "area",
            "area",
            None,
            "Kenya",
            "Area_Farm",
            "Total farm area",
            "Kenya's total farm area.",
            id="richer-label-over-generic-area",
        ),
    ],
)
def test_promoted_sv_label_replaces_less_specific_property(
    property_dcid: str,
    property_label: str,
    stat_type,
    place: str,
    stat_var_dcid: str,
    stat_var_label: str,
    expected: str,
):
    """A richer single-StatVar label replaces a non-stopword measured property."""
    spec = _spec(
        measured_property=_ref(property_dcid, property_label),
        stat_type=stat_type,
        slots=[_where_slot(place)],
        stat_vars=[
            StatVar(
                ref=_ref(stat_var_dcid, stat_var_label),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    assert _rendered(spec) == expected


def test_promotion_gate_a_fails_when_sv_label_adds_no_content_word():
    """The label and property have equal content-word sets, so strict-superset promotion fails.
    The unpromoted result retains the `Expectancy` qualifier."""
    spec = _spec(
        measured_property=_ref("lifeExpectancy", "life expectancy"),
        measurement_qualifier=_ref("expectancyQualifier", "Expectancy"),
        stat_vars=[
            StatVar(
                ref=_ref("LifeExpectancy", "Life Expectancy"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    assert _rendered(spec) == "Life expectancy Expectancy."


def test_promotion_gate_a_fails_when_sv_label_drops_a_property_word():
    """The label adds `adjusted` and `gdp` but omits `gross`, `domestic`, and `product`,
    so it fails the superset gate."""
    spec = _spec(
        measured_property=_ref(
            "grossDomesticProductGrowthRate", "gross domestic product growth rate"
        ),
        stat_vars=[
            StatVar(
                ref=_ref("AdjustedGDPGrowthRate", "Adjusted GDP growth rate"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    assert _rendered(spec) == "Gross domestic product growth rate."


def test_promotion_gate_b_fails_when_bound_adjectival_value_unnamed():
    """The label passes the superset gate but omits bound value `Female`.
    The gender fold remains."""
    spec = _spec(
        measured_property=_ref("fertilityRate", "fertilityRate"),
        slots=[_gender_slot("schemaFemale", "Female"), _where_slot("Kenya")],
        stat_vars=[
            StatVar(
                ref=_ref("FertilityRate_Person_Female", "Fertility rate of women"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    assert _rendered(spec) == "Kenya's female fertility rate."


def test_promotion_skipped_with_two_resolved_stat_vars():
    """Two resolved StatVars cannot promote through a single label."""
    spec = _spec(
        measured_property=_ref("age", "age"),
        slots=[_where_slot("Togo")],
        stat_vars=[
            StatVar(
                ref=_ref("Median_Age_Person", "Median age of population"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
            StatVar(
                ref=_ref("Mean_Age_Person", "Mean age of population"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
        ],
    )
    assert _rendered(spec) == "Togo's age."


def test_promoted_sv_label_names_bound_adjectival_value_empties_its_fold():
    """The label includes `Female`, so it promotes.
    The gender fold stays empty to avoid repetition."""
    spec = _spec(
        measured_property=_ref("unemploymentRate", "unemploymentRate"),
        slots=[_gender_slot("schemaFemale", "Female")],
        stat_vars=[
            StatVar(
                ref=_ref(
                    "UnemploymentRate_Person_Female", "Unemployment rate of the female population"
                ),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    gender_tok = next(t for t in plan if t.property_dcid == "gender")
    assert gender_tok.text == ""
    assert sentence_from_plan(plan) == "Unemployment rate of the female population."


def test_promotion_gate_c_fails_when_label_omits_measure_extra_word():
    """The label adds `population` but omits `median`, so the measure-extra gate keeps
    `median value` separate."""
    spec = _spec(
        measured_property=_ref("age", "age"),
        stat_type=_ref("medianValue", "Median Value"),
        slots=[_where_slot("Togo")],
        stat_vars=[
            StatVar(
                ref=_ref("Age_Person", "Age of population"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    assert _rendered(spec) == "Togo's age median value."


def test_promotion_gate_c_passes_when_label_names_per_capita_extra():
    """`CO2 Emissions Per Capita` passes both gates, so the denominator is not duplicated."""
    spec = _spec(
        measured_property=_ref("emissions", "emissions"),
        measurement_denominator=_ref("PerCapita", "Per Capita"),
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


def test_promotion_ignores_per_capita_because_the_unit_states_it():
    """`CO2 Emissions` omits `capita` and is still promoted: per capita belongs to the
    unit token, which states it after the label."""
    spec = _spec(
        measured_property=_ref("emissions", "emissions"),
        measurement_denominator=_ref("PerCapita", "Per Capita"),
        stat_vars=[
            StatVar(
                ref=_ref("Emissions_CO2", "CO2 Emissions"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert sentence_from_plan(plan) == "CO2 Emissions per capita."
    assert [t.text for t in plan if t.axis == "unit"] == ["per capita"]


def test_sv_name_path_suppresses_measure_extras():
    """A single StatVar name supplies the measure without duplicate extras.
    The empty how token retains its property and binding ref for a client chip."""
    spec = _spec(
        measured_property=_ref("amount", "amount"),
        measurement_denominator=_ref("PerCapita", "PerCapita"),
        slots=[_how_slot("CarbonDioxide")],
        stat_vars=[
            StatVar(
                ref=_ref("sv1", "CO2 Emissions Per Capita"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    result = sentence_from_plan(plan)
    assert result == "CO2 emissions per capita."
    assert result.lower().count("per capita") == 1

    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.kind == "slot"
    assert how_tok.text == ""
    assert how_tok.property_dcid == "DevelopmentFinancePurpose"
    assert how_tok.binding_ref == "how:DevelopmentFinancePurpose"


def test_composition_path_renders_per_capita():
    """The composition path states the per-capita denominator as the unit."""
    spec = _spec(
        measured_property=_ref("amount", "amount"),
        population_type=_ref("Household", "Household"),
        measurement_denominator=_ref("PerCapita", "PerCapita"),
        stat_vars=[
            StatVar(
                ref=_ref("sv1", "SV One"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
            StatVar(
                ref=_ref("sv2", "SV Two"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
        ],
    )
    assert _rendered(spec) == "Household per capita."


def test_nondefault_stat_type_token_present():
    """A statType other than "measuredValue" appends a humanized extra token."""
    spec = _spec(
        measured_property=_ref("fertilityRate", "fertilityRate"),
        stat_type=_ref("medianValue", "medianValue"),
    )
    assert _rendered(spec) == "Fertility rate median value."


def test_measure_extra_tokens_are_bare_nonclickable_slot_tokens():
    """Measure-extra tokens have axis `what` and no matching spec slot; the per-capita
    denominator is the unit's token, equally bare."""
    spec = _spec(
        measured_property=_ref("amount", "amount"),
        population_type=_ref("Household", "Household"),
        stat_type=_ref("medianValue", "medianValue"),
        measurement_denominator=_ref("PerCapita", "PerCapita"),
        stat_vars=[
            StatVar(
                ref=_ref("sv1", "SV One"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
            StatVar(
                ref=_ref("sv2", "SV Two"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            ),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    extras = [t for t in plan if t.text in ("median value", "per capita")]
    assert [(t.text, t.axis) for t in extras] == [("median value", "what"), ("per capita", "unit")]
    for tok in extras:
        assert tok.kind == "slot"
        assert tok.property_dcid is None
        assert tok.binding_ref is None


def test_bound_what_slot_never_emits_extras_even_with_nondefault_stat_type():
    """A bound `what` label is the full phrase, even with non-default measure fields;
    a per-head denominator is the unit's and still reads."""
    spec = _spec(
        slots=[_what_slot("ODA Grants")],
        stat_type=_ref("medianValue", "medianValue"),
        measurement_denominator=_ref("PerCapita", "PerCapita"),
    )
    assert _rendered(spec) == "ODA Grants per capita."
