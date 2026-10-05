"""Checks the candidates sentence and the axes that distinguish
candidate specs."""

from __future__ import annotations

import pytest

from search.models import (
    BindingUnbound,
    BindingValue,
    Departure,
    ResolvedFacet,
    SentenceToken,
    Slot,
    SlotValue,
    Spec,
    StatVar,
    TimeWindow,
)
from search.render.candidates import (
    candidate_differences,
    candidates_sentence_plan,
    readings_text,
)
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import (
    _entity_from,
    _entity_to,
    _gender_slot,
    _how_slot,
    _ref,
    _slot_key,
    _source_slot,
    _spec,
    _what_slot,
    _where_slot,
    assert_round_trip,
)

# ---------------------------------------------------------------------------
# candidate_differences
# ---------------------------------------------------------------------------


def _rendered_spec(spec: Spec) -> Spec:
    """Attach a rendered sentence and plan to a spec."""
    plan = render_sentence_plan(spec)
    return spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={"sentence_plan": plan, "rendered_sentence": sentence_from_plan(plan)}
            )
        }
    )


def _how_value_slot(dcid: str, label: str) -> Slot:
    """Build a purpose slot with a selectable DCID for comparison tests."""
    return Slot(
        key=_slot_key("how", "DevelopmentFinancePurpose", "Purpose", "purpose"),
        binding=BindingValue(value=SlotValue(ref=_ref(dcid, label), value_kind="entity")),
    )


def _candidate_difference_specs(bucket: str) -> tuple[Spec, Spec]:
    if bucket == "what":
        return (
            _spec(slots=[_what_slot("ODA Grants")]),
            _spec(slots=[_what_slot("ODA Loans")]),
        )
    if bucket == "how":
        return (
            _spec(slots=[_what_slot("ODA Grants"), _how_value_slot("DAC/Health", "health")]),
            _spec(
                slots=[
                    _what_slot("ODA Grants"),
                    _how_value_slot("DAC/Malariacontrol", "malaria control"),
                ]
            ),
        )
    if bucket == "where":
        return (
            _spec(entities=[_entity_to("country/ETH", "Ethiopia")]),
            _spec(entities=[_entity_to("country/KEN", "Kenya")]),
        )
    if bucket == "funder":
        return (
            _spec(entities=[_entity_from("country/USA", "the United States")]),
            _spec(entities=[_entity_from("country/DEU", "Germany")]),
        )
    return (
        _spec(slots=[_source_slot("OECDsrc", "OECD")]),
        _spec(slots=[_source_slot("WBsrc", "World Bank")]),
    )


@pytest.mark.parametrize(
    ("bucket", "sentences", "token_selector", "summary_noun"),
    [
        pytest.param(
            "what", ("ODA Grants.", "ODA Loans."), ("axis", "what"), "indicator", id="what"
        ),
        pytest.param(
            "how",
            ("ODA Grants for health.", "ODA Grants for malaria control."),
            ("axis", "how"),
            "purpose",
            id="how",
        ),
        pytest.param(
            "where",
            ("Development finance flow to Ethiopia.", "Development finance flow to Kenya."),
            ("axis", "where"),
            "place",
            id="where",
        ),
        pytest.param(
            "funder",
            (
                "The United States' development finance flow.",
                "Germany's development finance flow.",
            ),
            ("role", "origin"),
            "funder",
            id="funder",
        ),
        pytest.param(
            "source",
            (
                "Development finance flow. OECD.",
                "Development finance flow. World Bank.",
            ),
            ("axis", "source"),
            "data source",
            id="source",
        ),
    ],
)
def test_candidate_differences_single_bucket(
    bucket: str, sentences: tuple[str, str], token_selector: tuple[str, str], summary_noun: str
):
    spec_a, spec_b = _candidate_difference_specs(bucket)
    assert render_sentence(spec_a) == sentences[0]
    assert render_sentence(spec_b) == sentences[1]
    spec_a, spec_b = _rendered_spec(spec_a), _rendered_spec(spec_b)

    differing = candidate_differences([spec_a, spec_b])
    assert differing == (summary_noun,)
    field, value = token_selector
    token = next(token for token in spec_a.resolved.sentence_plan if getattr(token, field) == value)
    assert token.distinguishes is True


def test_candidate_differences_when_bucket():
    spec_2015 = _rendered_spec(_spec(window=TimeWindow(start_year=2015, end_year=2015)))
    spec_2016 = _rendered_spec(_spec(window=TimeWindow(start_year=2016, end_year=2016)))
    assert spec_2015.resolved.rendered_sentence == "Development finance flow in 2015."
    assert spec_2016.resolved.rendered_sentence == "Development finance flow in 2016."

    differing = candidate_differences([spec_2015, spec_2016])
    assert differing == ("time period",)
    when_tok = next(t for t in spec_2015.resolved.sentence_plan if t.axis == "when")
    assert when_tok.distinguishes is True


def test_candidate_differences_multi_bucket_pair():
    grants = _rendered_spec(
        _spec(
            slots=[_what_slot("ODA Grants"), _source_slot("OECDsrc", "OECD")],
            window=TimeWindow(start_year=2015, end_year=2015),
        )
    )
    loans = _rendered_spec(
        _spec(
            slots=[_what_slot("ODA Loans"), _source_slot("WBsrc", "World Bank")],
            window=TimeWindow(start_year=2016, end_year=2016),
        )
    )
    assert grants.resolved.rendered_sentence == "ODA Grants in 2015. OECD."
    assert loans.resolved.rendered_sentence == "ODA Loans in 2016. World Bank."

    differing = candidate_differences([grants, loans])
    assert differing == ("indicator", "time period", "data source")


def test_candidate_differences_identical_specs_no_diff():
    spec = _spec(slots=[_what_slot("ODA Grants"), _where_slot("Kenya")])
    spec_a = _rendered_spec(spec)
    spec_b = _rendered_spec(spec)

    differing = candidate_differences([spec_a, spec_b])
    assert differing == ()
    assert all(tok.distinguishes is None for tok in spec_a.resolved.sentence_plan)
    assert all(tok.distinguishes is None for tok in spec_b.resolved.sentence_plan)


def test_candidate_differences_how_axis_compares_values_not_capitalization():
    """A pre-what adjectival fold present on only one candidate shifts which
    token opens the sentence, and so which token gets capitalised, but that
    is not why these two differ: the "gender" how-axis property itself is
    bound on one candidate and left open on the other. Resolved values
    decide a how-axis difference, not rendered text."""
    with_gender = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Population", "population"),
        slots=[_gender_slot("schemaFemale", "Female")],
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
    unbound_gender = Slot(
        key=_slot_key("how", "gender", "Gender", "gender"), binding=BindingUnbound()
    )
    without_gender = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Population", "population"),
        slots=[unbound_gender],
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
    spec_a = _rendered_spec(with_gender)
    spec_b = _rendered_spec(without_gender)
    assert spec_a.resolved.rendered_sentence == "Female population."
    assert spec_b.resolved.rendered_sentence == "Population."

    differing = candidate_differences([spec_a, spec_b])
    assert differing == ("gender",)


def test_candidate_differences_ignores_how_property_silent_on_both():
    """A declared-but-unbound how slot present on only one candidate, with
    nothing (no binding, no served-member fact) asserted about it on EITHER
    candidate, resolves to the same "nothing to say" signature on both: two
    candidates whose only bound how phrase is identical must not be read as
    differing just because one of them also declares that extra silent
    slot."""
    bound_only = _rendered_spec(_spec(slots=[_how_slot("Health")]))
    with_extra_unbound = _rendered_spec(
        _spec(
            slots=[
                _how_slot("Health"),
                Slot(key=_slot_key("how", "gender", "Gender", "gender"), binding=BindingUnbound()),
            ]
        )
    )
    assert bound_only.resolved.rendered_sentence == with_extra_unbound.resolved.rendered_sentence

    differing = candidate_differences([bound_only, with_extra_unbound])
    assert differing == ()
    for spec in (bound_only, with_extra_unbound):
        assert all(tok.distinguishes is None for tok in spec.resolved.sentence_plan)


def test_candidate_differences_how_bucket_property_granular():
    """distinguishes is property-granular for the how axis: a how-property
    shared by every candidate never gets marked, even though another
    how-property differs. Only the property that actually differs does."""
    modality_key = _slot_key("how", "DevelopmentFinanceModality", "Modality", "modality")
    grant_slot = Slot(
        key=modality_key,
        binding=BindingValue(
            value=SlotValue(ref=_ref("DevFinModality/Grant", "Grant"), value_kind="entity")
        ),
    )
    loan_slot = Slot(
        key=modality_key,
        binding=BindingValue(
            value=SlotValue(ref=_ref("DevFinModality/Loan", "Loan"), value_kind="entity")
        ),
    )
    grant_spec = _rendered_spec(_spec(slots=[_how_slot("Health"), grant_slot]))
    loan_spec = _rendered_spec(_spec(slots=[_how_slot("Health"), loan_slot]))
    assert_round_trip(grant_spec.resolved.sentence_plan, grant_spec.resolved.rendered_sentence)
    assert_round_trip(loan_spec.resolved.sentence_plan, loan_spec.resolved.rendered_sentence)

    differing = candidate_differences([grant_spec, loan_spec])
    assert differing == ("modality",)

    def _how_token(spec: Spec, text: str) -> SentenceToken:
        return next(t for t in spec.resolved.sentence_plan if t.axis == "how" and t.text == text)

    assert _how_token(grant_spec, "Health").distinguishes is None
    assert _how_token(loan_spec, "Health").distinguishes is None
    assert _how_token(grant_spec, "Grant").distinguishes is True
    assert _how_token(loan_spec, "Loan").distinguishes is True


def test_candidate_differences_how_axis_agrees_when_one_leaves_it_to_a_fact():
    """One candidate binds a how-property directly, the other leaves it
    unbound but the served members agree on the same value via
    open_axis_facts. An axis one sibling renders because it is bound and
    another renders because a constant fact speaks for it (or doesn't render
    at all) is not a difference when the resolved value is the same either
    way."""
    from search.models import OpenConstraintAxisFact

    bound_grant = Slot(
        key=_slot_key("how", "CRSTypeOfFinance", "Finance", "finance type"),
        binding=BindingValue(
            value=SlotValue(
                ref=_ref("CRSTypeOfFinance/StandardGrant", "Standard Grant"), value_kind="entity"
            )
        ),
    )
    unbound_finance = Slot(
        key=_slot_key("how", "CRSTypeOfFinance", "Finance", "finance type"),
        binding=BindingUnbound(),
    )
    agreeing_fact = OpenConstraintAxisFact(
        property="CRSTypeOfFinance",
        varies=False,
        value=_ref("CRSTypeOfFinance/StandardGrant", "Standard Grant"),
        distinct_count=1,
        undeclared_count=0,
    )

    spec_bound = _rendered_spec(_spec(slots=[bound_grant]))
    spec_fact_only = _rendered_spec(_spec(slots=[unbound_finance], open_axis_facts=[agreeing_fact]))

    differing = candidate_differences([spec_bound, spec_fact_only])
    assert differing == ()


def test_candidate_differences_omits_property_this_run_has_no_noun_for():
    """A how-axis property that differs across candidates, but that no
    candidate declares a Slot for and axis_declarations carries no entry
    for either, names no noun in the header: naming its dcid there would
    leak the same internal identifier the per-clause rendering already
    stays silent about for an undeclared noun. The property's resolved
    values still differ, and its own tokens are still marked, even though
    the header omits it."""
    from search.models import OpenConstraintAxisFact

    grant_fact = OpenConstraintAxisFact(
        property="CRSTypeOfFinance",
        varies=False,
        value=_ref("CRSTypeOfFinance/StandardGrant", "Standard Grant"),
        distinct_count=1,
        undeclared_count=0,
    )
    loan_fact = OpenConstraintAxisFact(
        property="CRSTypeOfFinance",
        varies=False,
        value=_ref("CRSTypeOfFinance/Loan", "Loan"),
        distinct_count=1,
        undeclared_count=0,
    )
    spec_grant = _rendered_spec(_spec(open_axis_facts=[grant_fact]))
    spec_loan = _rendered_spec(_spec(open_axis_facts=[loan_fact]))

    differing = candidate_differences([spec_grant, spec_loan])
    assert differing == ()
    for spec in (spec_grant, spec_loan):
        how_tok = next(t for t in spec.resolved.sentence_plan if t.axis == "how" and t.text)
        assert how_tok.property_dcid == "CRSTypeOfFinance"
        assert how_tok.distinguishes is True


# ---------------------------------------------------------------------------
# candidates_sentence_plan
# ---------------------------------------------------------------------------


def _gdp_reading(source: str, start: int, end: int) -> Spec:
    return _rendered_spec(
        _spec(
            slots=[_what_slot("Nominal GDP")],
            window=TimeWindow(start_year=start, end_year=end),
            measurement_qualifier=_ref("Nominal", "Nominal"),
            facets={
                "unit": ResolvedFacet(
                    basis="fallback", ref="USDollar", label="US Dollar", currency="USD"
                )
            },
            resolved_sources=[_ref(source.lower(), source)],
        )
    )


def test_a_candidates_sentence_counts_the_readings_then_states_each_led_by_source_and_span():
    """weo-019: "{N} readings." then each reading's heading, its span moved into a
    byline led by the source; the plan's own departures close it. Every token is text."""
    readings = [_gdp_reading("WDI", 1960, 2024), _gdp_reading("IMF WEO", 1980, 2025)]
    departure = Departure(
        code="modelled_values", text="Some of these values are modelled estimates."
    )
    plan = candidates_sentence_plan(readings, departures=[departure])
    assert sentence_from_plan(plan) == (
        "2 readings. "
        "Nominal GDP. WDI · 1960 to 2024 · Current US dollars. "
        "Nominal GDP. IMF WEO · 1980 to 2025 · Current US dollars. "
        "Some of these values are modelled estimates."
    )
    assert all(t.kind == "text" for t in plan)
    assert [t.text for t in plan if t.segment == "departure" and t.text.strip()] == [departure.text]
    assert {t.segment for t in plan if t.text in {"WDI", "1960 to 2024"}} == {"byline"}


@pytest.mark.parametrize(
    ("shown", "truncated", "total", "expected"),
    [
        pytest.param(3, False, None, "3 readings.", id="all-shown"),
        pytest.param(5, True, 12, "12 readings; the first 5 are shown.", id="known-total"),
        pytest.param(5, True, None, "More than 5 readings; the first 5 are shown.", id="unknown"),
        pytest.param(1, True, 4, "4 readings; the first is shown.", id="one-shown"),
    ],
)
def test_the_count_of_readings(shown: int, truncated: bool, total: int | None, expected: str):
    assert readings_text(shown, truncated, total) == expected


def test_a_reading_without_a_span_or_byline_keeps_its_heading():
    plan = candidates_sentence_plan([_rendered_spec(_spec(slots=[_what_slot("ODA Grants")]))])
    assert sentence_from_plan(plan) == "1 reading. ODA Grants."


def _sdg_poverty_reading(population_type: str, label: str) -> Spec:
    """A crs-022 reading, whose heading and byline name no value its population type
    holds."""
    return _spec(
        slots=[
            _what_slot("Official development assistance grants for poverty reduction"),
            _source_slot("sdgSrc", "SDG Indicators Database"),
        ],
        window=TimeWindow(start_year=2000, end_year=2022),
        population_type=_ref(population_type, label),
    )


def test_crs_022_readings_that_read_alike_state_what_sets_them_apart():
    """crs-022: two readings whose heading and byline read the same each add the value
    that sets them apart, here their population type, to the byline."""
    readings = [
        _sdg_poverty_reading(
            "SDG_DC_ODA_POVLG",
            "Official development assistance grants for poverty reduction, by recipient countries",
        ),
        _sdg_poverty_reading(
            "SDG_DC_ODA_POVDLG",
            "Official development assistance grants for poverty reduction, by donor countries",
        ),
    ]
    plan = candidates_sentence_plan(readings)
    heading = "Official development assistance grants for poverty reduction."
    assert sentence_from_plan(plan) == (
        f"2 readings. {heading} SDG Indicators Database · 2000 to 2022 · Official development "
        f"assistance grants for poverty reduction, by recipient countries. {heading} SDG "
        "Indicators Database · 2000 to 2022 · Official development assistance grants for "
        "poverty reduction, by donor countries."
    )
