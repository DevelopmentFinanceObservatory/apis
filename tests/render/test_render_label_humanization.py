"""Checks label humanization and how-axis connective wording."""

from __future__ import annotations

import pytest

from search.models import BindingValue, SentenceFacts, Shape, Slot, SlotValue, StatVar
from search.render.heading import heading_from_parts
from search.render.labels import _humanize_ref
from search.render.tokens import plan_tokens, sentence_from_plan
from search.render.what import _normalize_what_label
from tests.render._helpers import (
    _entity_from,
    _entity_to,
    _ref,
    _rendered,
    _slot_key,
    _spec,
    _where_slot,
    assert_round_trip,
)

# Identifier and indicator label normalization.


def test_measured_property_dcid_fallback_splits_underscore_before_camel():
    """Split underscore-joined identifiers before humanizing camelCase segments."""
    spec = _spec(
        measured_property=_ref("Percent_Person_BLS_InLaborForce", "Percent_Person_BLS_InLaborForce")
    )
    assert _rendered(spec) == "Percent person bls in labor force."


@pytest.mark.parametrize(
    ("stat_var_labels", "population_type_label", "expected"),
    [
        pytest.param(
            ["Overweight prevalence among children under 5 years of age (%)"],
            None,
            "Overweight prevalence among children under 5 years of age.",
            id="single-statvar-name",
        ),
        pytest.param(
            ["SV One", "SV Two"],
            "Child population",
            "Child population.",
            id="population-type-with-multiple-statvars",
        ),
    ],
)
def test_allcaps_unsplittable_measured_property_fallback_ladder(
    stat_var_labels: list[str], population_type_label: str | None, expected: str
):
    """The all-caps fallback retains the StatVar and population-type branches."""
    spec = _spec(
        measured_property=_ref("who/NUTOVERWEIGHTPREV", "who/NUTOVERWEIGHTPREV"),
        population_type=(
            _ref("ChildPopulation", population_type_label) if population_type_label else None
        ),
        stat_vars=[
            StatVar(
                ref=_ref(f"who/sv{index}", label),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
            for index, label in enumerate(stat_var_labels, start=1)
        ],
    )
    assert _rendered(spec) == expected


def test_allcaps_unsplittable_measured_property_gated_at_stopword_fallback_too():
    """The final measured-property fallback also rejects unsplittable all-caps identifiers."""
    shape = Shape(
        shape_id="test-shape",
        label="Prevalence Shape",
        population_type=None,
        measured_property=_ref("who/NUTOVERWEIGHTPREV", "who/NUTOVERWEIGHTPREV"),
        stat_type=_ref("measuredValue", "Measured Value"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[],
        member_count=1,
    )
    plan = plan_tokens(
        heading_from_parts(
            [],
            [],
            shape,
            window=None,
            facts=SentenceFacts.of(
                [
                    StatVar(
                        ref=_ref(dcid, label),
                        shape_id="test-shape",
                        slot_values=[],
                        constraint_properties=[],
                    )
                    for dcid, label in (("sv1", "SV One"), ("sv2", "SV Two"))
                ]
            ),
            pinned_properties=frozenset(),
        )
    )
    assert_round_trip(plan, sentence_from_plan(plan))
    assert sentence_from_plan(plan) == "Prevalence Shape."


def test_measurement_qualifier_bracket_becomes_sentence_cased_parenthetical():
    """Bracketed qualifier labels render as sentence-cased parentheticals."""
    spec = _spec(
        measured_property=_ref("fertilityRate", "fertilityRate"),
        measurement_qualifier=_ref("AgeGenderQualifier", "[15 to 19 years old, Female]"),
    )
    assert _rendered(spec) == "Fertility rate (15 to 19 years old, female)."


@pytest.mark.parametrize(
    ("dcid", "label", "expected"),
    [
        pytest.param(
            "RegionalGdpGrowthByMcKinseyCo2Index",
            "Regional GDP Growth by McKinsey CO2 Index",
            "Regional GDP growth by McKinsey CO2 index.",
            id="acronyms-digits-and-mixed-case",
        ),
        pytest.param(
            "growthRateOfPopulation",
            "Rate of Population Growth",
            "Rate of population growth.",
            id="function-word-in-head",
        ),
        pytest.param(
            "povertyBelowLineLast12MonthsPerCapita",
            "Population: Below Poverty Level In The Past 12 Months (Per capita)",
            "Population: below poverty level in the past 12 months (per capita).",
            id="word-bearing-parenthetical",
        ),
        pytest.param(
            "emissionsTotales",
            "Émissions Totales",
            "Émissions totales.",
            id="unicode-title-case",
        ),
    ],
)
def test_sentence_casing_normalizes_title_case_labels(dcid: str, label: str, expected: str):
    """Recase title-case labels while preserving acronyms and punctuation."""
    spec = _spec(
        measured_property=_ref(dcid, label),
    )
    assert _rendered(spec) == expected


@pytest.mark.parametrize(
    ("dcid", "label", "expected"),
    [
        pytest.param(
            "sendaiFrameworkAdoptionScore",
            "Score of adoption and implementation of national Disaster Risk "
            "Reduction strategies in line with the Sendai Framework",
            "Score of adoption and implementation of national Disaster Risk "
            "Reduction strategies in line with the Sendai Framework.",
            id="sentence-case-proper-noun",
        ),
        pytest.param(
            "povertyHeadcountRatioNationalLines",
            "Poverty headcount ratio at national poverty lines",
            "Poverty headcount ratio at national poverty lines.",
            id="sentence-case-label",
        ),
    ],
)
def test_sentence_casing_rejects_non_title_case_labels(dcid: str, label: str, expected: str):
    """Leave labels already in sentence case unchanged."""
    spec = _spec(
        measured_property=_ref(dcid, label),
    )
    assert _rendered(spec) == expected


def test_promotion_gate_a_content_words_ignore_punctuation():
    """Ignore punctuation when comparing the measured property with StatVar labels."""
    spec = _spec(
        measured_property=_ref("fertilityRate", "fertility rate"),
        stat_vars=[
            StatVar(
                ref=_ref("SP_DYN_TFRT_IN", "Fertility rate, total (births per woman)"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    assert _rendered(spec) == "Fertility rate, total (births per woman)."


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        pytest.param(
            "Infant mortality rate [Age=Under 1 year old]",
            "Infant mortality rate (under 1 year old)",
            id="one-slice",
        ),
        pytest.param(
            "Infant mortality rate [Age=Under 1 year old, Sex=Male]",
            "Infant mortality rate (under 1 year old, male)",
            id="two-slices",
        ),
        pytest.param(
            "Current health expenditure per capita "
            "[Health expenditure category=Current health expenditure]",
            "Current health expenditure per capita",
            id="slice-named-by-the-head",
        ),
    ],
)
def test_slice_list_bracket_renders_its_values(label: str, expected: str):
    """A graph label's ``Key=Value`` slice list renders its values without keys, and only
    the values the label's head does not already name."""
    assert _normalize_what_label(label) == expected


def test_single_token_title_case_label_untouched():
    """Keep a single-token label unchanged."""
    assert _normalize_what_label("Livestock") == "Livestock"


def test_catalog_prefix_retained_and_recased():
    """Retain the catalog prefix and recase following words."""
    spec = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Population6564", "Population: Years 65 Onwards"),
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
    assert _rendered(spec) == "Population: years 65 onwards."


@pytest.mark.parametrize(
    ("dcid", "label", "expected"),
    [
        pytest.param(
            "developmentFinanceFlowPct",
            "Development Finance Flow (%).",
            "Development finance flow.",
            id="adjacent-period",
        ),
        pytest.param("countPct", "Rate (#) .", "Rate.", id="spaced-period"),
    ],
)
def test_trailing_period_after_symbol_parenthetical_strips_both(
    dcid: str, label: str, expected: str
):
    """Strip a symbol-only parenthetical and its trailing period."""
    spec = _spec(
        measured_property=_ref(dcid, label),
    )
    assert _rendered(spec) == expected


def test_measured_property_label_ending_in_period_renders_one_terminal_period():
    """Strip the label's terminal period before appending the where clause."""
    spec = _spec(
        measured_property=_ref(
            "GrowthRatePersonInGivenYear", "Population growth rate in a given year."
        ),
        slots=[_where_slot("Kenya")],
    )
    result = _rendered(spec)
    assert result == "Kenya's population growth rate in a given year."
    assert not result.endswith("..")


# Bound-value humanization and how-axis connectives.


def test_how_slot_raw_dcid_label_is_humanized():
    """Humanize a raw DCID label while retaining the default `for` connective."""
    slot = Slot(
        key=_slot_key("how", "economicActivity", "Economic Activity", "activity"),
        binding=BindingValue(
            value=SlotValue(
                ref=_ref("GrossDomesticProduction", "GrossDomesticProduction"),
                value_kind="entity",
            )
        ),
    )
    spec = _spec(slots=[slot])
    assert _rendered(spec) == "Development finance flow for gross domestic production."


def test_where_slot_raw_dcid_label_is_humanized():
    """An entity's dcid-equal label splits at camel case and keeps its case, as "Gavi"
    keeps its capital (mls-029)."""
    slot = Slot(
        key=_slot_key("where", "DevelopmentFinanceRecipient", "Recipient", "recipient"),
        binding=BindingValue(
            value=SlotValue(
                ref=_ref("SomeUnresolvedRegion", "SomeUnresolvedRegion"), value_kind="entity"
            )
        ),
    )
    spec = _spec(slots=[slot])
    assert _rendered(spec) == "Some Unresolved Region's development finance flow."


def test_directional_to_entity_raw_dcid_label_is_humanized():
    spec = _spec(entities=[_entity_to("SomeUnresolvedRegion", "SomeUnresolvedRegion")])
    assert _rendered(spec) == "Development finance flow to Some Unresolved Region."


def test_directional_from_entity_raw_dcid_label_is_humanized():
    spec = _spec(entities=[_entity_from("SomeUnresolvedDonor", "SomeUnresolvedDonor")])
    assert _rendered(spec) == "Some Unresolved Donor's development finance flow."


def test_directional_entity_real_label_never_mangled():
    """Keep a distinct authored directional label unchanged."""
    spec = _spec(entities=[_entity_from("country/USA", "the United States")])
    assert _rendered(spec) == "The United States' development finance flow."


def test_how_connective_adjectival_property_has_no_connective_word():
    """Demographic values fold before the what noun without a connective."""
    slot = Slot(
        key=_slot_key("how", "maritalStatus", "Marital Status", "marital status"),
        binding=BindingValue(
            value=SlotValue(
                ref=_ref("MarriedAndNotSeparated", "MarriedAndNotSeparated"),
                value_kind="entity",
            )
        ),
    )
    spec = _spec(slots=[slot])
    result = _rendered(spec)
    assert result == "Married and not separated development finance flow."
    assert "for " not in result


def test_how_connective_of_property():
    """Categorical place properties use the `of` connective."""
    slot = Slot(
        key=_slot_key("how", "placeOfBirth", "Place of Birth", "place of birth"),
        binding=BindingValue(
            value=SlotValue(ref=_ref("BornOutsideUS", "BornOutsideUS"), value_kind="entity")
        ),
    )
    spec = _spec(slots=[slot])
    assert _rendered(spec) == "Development finance flow of born outside us."


def test_humanize_real_label_passes_through_verbatim():
    """Keep authored acronym casing."""
    label = "WHO eHealth OECD Profile"
    assert _humanize_ref("SomeDcidXyz", label) == label
