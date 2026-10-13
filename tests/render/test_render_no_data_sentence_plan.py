"""Checks no-data envelopes, near-match continuations, and requested-window notes."""

from __future__ import annotations

from typing import cast

import pytest

from search.engine.assemble import with_sentence
from search.models import (
    Coverage,
    DateRange,
    Departure,
    GraphRef,
    NearestRealSpec,
    NoDataDenominator,
    NoDataGate,
    NoDataReason,
    Relaxation,
    ResolvedQualifier,
    SentenceToken,
    Spec,
    StatVar,
    TimeWindow,
)
from search.render.departures import no_data_limit_text
from search.render.heading import LATEST_YEAR_LABEL
from search.render.no_data import (
    composed_variable_outcome,
    no_data_phrase,
    no_data_sentence_plan,
)
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import (
    _entity_from,
    _entity_subject,
    _entity_to,
    _ref,
    _rendered,
    _spec,
    _what_slot,
    _where_place_slot,
    assert_round_trip,
)

# ---------------------------------------------------------------------------
# no_data_sentence_plan: a no-data envelope composed around a near-match's
# own already-rendered sentence
# ---------------------------------------------------------------------------


def _matched_spec(spec: Spec) -> Spec:
    return with_sentence(spec)


def _nearest_real(spec: Spec, *, dropped_property: str) -> NearestRealSpec:
    rendered = with_sentence(spec)
    return NearestRealSpec(
        spec_id=rendered.spec_id,
        definition=rendered.definition,
        resolved=rendered.resolved,
        relaxation=Relaxation(
            axis="how", property_dcid=dropped_property, dropped_value_dcids=["Health"]
        ),
    )


@pytest.mark.parametrize(
    ("slots", "entities", "expected"),
    [
        pytest.param(
            [_where_place_slot("country/KEN", "Kenya")],
            [_entity_subject("country/KEN", "Kenya")],
            "No data for Kenya's development finance flow.",
            id="subject-place",
        ),
        pytest.param(
            [],
            [_entity_from("country/USA", "United States"), _entity_to("country/ETH", "Ethiopia")],
            "No data for United States' development finance flow to Ethiopia.",
            id="donor-and-recipient",
        ),
        pytest.param(
            [],
            [_entity_from("country/USA", "United States")],
            "No data for United States' development finance flow.",
            id="donor-only",
        ),
        pytest.param([], None, "No data for development finance flow.", id="no-entity"),
    ],
)
def test_no_data_sentence_plan_no_member_has_data_at_place(slots, entities, expected: str):
    """A place gap states the checked series' heading in claim form, naming the entity,
    measure, and counterpart that were checked, as text tokens with the entity's case
    kept; nothing follows without departures."""
    flagged = _matched_spec(_spec(slots=slots, entities=entities))
    plan = no_data_sentence_plan(
        "no_observations", gate="no_member_has_data_at_place", flagged_match=flagged
    )
    assert sentence_from_plan(plan) == expected
    assert all(t.kind == "text" and t.segment == "heading" for t in plan)


def test_a_no_data_plan_ends_with_its_departures():
    """crs-010: the claim heading, then the part's departures in the departure segment,
    the `scope_limited` one naming what the check covered."""
    flagged = _matched_spec(
        _spec(
            entities=[
                _entity_from("GatesFoundation", "Gates Foundation"),
                _entity_to("country/NGA", "Nigeria"),
            ]
        )
    )
    plan = no_data_sentence_plan(
        "no_observations",
        gate="no_member_has_data_at_place",
        flagged_match=flagged,
        departures=[
            Departure(
                code="scope_limited",
                binding_ref="how:crsTypeOfFlow",
                text=no_data_limit_text(["the education sector", "ODA"]),
            ),
            Departure(code="modelled_values", text="These values are modelled estimates."),
        ],
    )
    assert sentence_from_plan(plan) == (
        "No data for Gates Foundation's development finance flow to Nigeria. "
        "This check covered only the education sector and ODA. "
        "These values are modelled estimates."
    )
    departure = [t.text for t in plan if t.segment == "departure" and t.text.strip()]
    assert departure == [
        "This check covered only the education sector and ODA.",
        "These values are modelled estimates.",
    ]


@pytest.mark.parametrize(
    ("reason", "variable_term", "expected", "unexpected_caution"),
    [
        pytest.param(
            "no_matching_slice",
            "net ODA received",
            'No published series for "net ODA received" matches the combination this query '
            "asks for. Data exists for development finance flow. Check that it measures "
            "what you asked for.",
            None,
            id="ordinary-reason",
        ),
        pytest.param(
            "variable_not_resolved",
            "aid flows",
            'Could not match "aid flows" to a supported indicator. Try a different wording '
            "or a related indicator. Data exists for development finance flow.",
            "Check that it measures what you asked for.",
            id="trailing-imperative",
        ),
    ],
)
def test_no_data_sentence_plan_only_alternate_has_data_composes_nearest_series(
    reason: str, variable_term: str, expected: str, unexpected_caution: str | None
):
    """A reason-specific imperative replaces the generic alternate caution."""
    alt = _matched_spec(_spec())
    plan = no_data_sentence_plan(
        reason,
        gate="only_alternate_has_data",
        alternate_match=alt,
        variable_term=variable_term,
    )
    assert sentence_from_plan(plan) == expected
    if unexpected_caution is not None:
        assert unexpected_caution not in sentence_from_plan(plan)


# Role-bearing alternates require confirmation against the bound axis (S2).


def test_no_data_sentence_plan_withholds_a_directional_alternate_with_no_role_signal():
    """With no role evidence, withhold an alternate that recasts donor Canada as recipient."""
    alt = _matched_spec(_spec(entities=[_entity_to("country/CAN", "Canada")]))
    plan = no_data_sentence_plan(
        "no_observations",
        gate="only_alternate_has_data",
        alternate_match=alt,
        elected_stat_vars=[GraphRef(ref="Amount_Flow_Variant", label="ODA disbursed")],
    )
    rendered = sentence_from_plan(plan)
    assert rendered == (
        "No data for ODA disbursed. Try a different place, time period, or indicator."
    )
    assert "Canada" not in rendered


def test_no_data_sentence_plan_withholds_a_directional_alternate_on_the_open_axis():
    """With only origin bound, a destination alternate remains unconfirmed."""
    alt = _matched_spec(_spec(entities=[_entity_to("country/CAN", "Canada")]))
    plan = no_data_sentence_plan(
        "no_observations",
        gate="only_alternate_has_data",
        alternate_match=alt,
        bound_role_dcids={"origin": ("country/USA",)},
    )
    rendered = sentence_from_plan(plan)
    assert rendered == (
        "No data found for this query. Try a different place, time period, or indicator."
    )
    assert "Canada" not in rendered


def test_no_data_sentence_plan_shows_a_directional_alternate_confirmed_on_the_bound_axis():
    """An alternate matches when its origin and DCID match the query's."""
    alt = _matched_spec(_spec(entities=[_entity_from("country/CAN", "Canada")]))
    plan = no_data_sentence_plan(
        "no_observations",
        gate="only_alternate_has_data",
        alternate_match=alt,
        bound_role_dcids={"origin": ("country/CAN",)},
    )
    rendered = sentence_from_plan(plan)
    assert "Data exists for" in rendered
    assert "Canada" in rendered


def test_no_data_sentence_plan_shows_a_directional_alternate_with_both_axes_bound():
    """A both-bound query can show an alternate with the same donor and recipient."""
    alt = _matched_spec(
        _spec(
            entities=[
                _entity_from("country/CAN", "Canada"),
                _entity_to("country/KEN", "Kenya"),
            ]
        )
    )
    plan = no_data_sentence_plan(
        "no_observations",
        gate="only_alternate_has_data",
        alternate_match=alt,
        bound_role_dcids={"origin": ("country/CAN",), "destination": ("country/KEN",)},
    )
    rendered = sentence_from_plan(plan)
    assert "Data exists for" in rendered
    assert "Canada" in rendered and "Kenya" in rendered


def test_no_data_sentence_plan_withholds_a_swapped_alternate_with_both_axes_bound():
    """With both axes bound, reject an alternate that swaps Canada's role."""
    alt = _matched_spec(_spec(entities=[_entity_to("country/CAN", "Canada")]))
    plan = no_data_sentence_plan(
        "no_observations",
        gate="only_alternate_has_data",
        alternate_match=alt,
        bound_role_dcids={"origin": ("country/CAN",), "destination": ("country/KEN",)},
    )
    rendered = sentence_from_plan(plan)
    assert rendered == (
        "No data found for this query. Try a different place, time period, or indicator."
    )
    assert "Canada" not in rendered


def test_no_data_sentence_plan_subject_only_alternate_is_never_withheld():
    """Subject-only alternates have no flow role to invert."""
    alt = _matched_spec(_spec())
    plan = no_data_sentence_plan(
        "no_observations",
        gate="only_alternate_has_data",
        alternate_match=alt,
    )
    assert "Data exists for" in sentence_from_plan(plan)


@pytest.mark.parametrize(
    ("gate", "variable_term", "expected"),
    [
        pytest.param(
            "no_member_asserts_bound_properties",
            "maternal mortality ratio",
            'No published series for "maternal mortality ratio" matches the combination this '
            "query asks for. Data exists for development finance flow. Check that it "
            "measures what you asked for.",
            id="production-gate",
        ),
        pytest.param(
            "some_future_gate",
            "net ODA received",
            'No published series for "net ODA received" matches the combination this query '
            "asks for. Data exists for development finance flow. Check that it measures "
            "what you asked for.",
            id="unknown-gate",
        ),
    ],
)
def test_no_data_sentence_plan_flagged_match_fallback(gate: str, variable_term: str, expected: str):
    """Known and unknown gates retain the flagged-match envelope."""
    flagged = _matched_spec(_spec())
    plan = no_data_sentence_plan(
        "no_matching_slice",
        # The future-gate row deliberately exercises the runtime fallback for
        # a code outside the frozen public Literal.
        gate=cast(NoDataGate, gate),
        flagged_match=flagged,
        variable_term=variable_term,
    )
    assert sentence_from_plan(plan) == expected


def test_no_data_sentence_plan_only_alternate_has_data_names_elected_concept_first():
    """Name the graph-confirmed empty concept before its substitute."""
    alt = _matched_spec(_spec())
    plan = no_data_sentence_plan(
        "no_observations",
        gate="only_alternate_has_data",
        alternate_match=alt,
        variable_term="climate finance",
        elected_stat_vars=[
            GraphRef(
                ref="sdg/DC_FIN_CLIMT",
                label="Total climate-specific financial support provided",
            )
        ],
    )
    assert sentence_from_plan(plan) == (
        "No data for total climate-specific financial support provided. Data exists for "
        "development finance flow. Check that it measures what you asked for."
    )
    assert plan[0] == SentenceToken(
        kind="text", text="No data for total climate-specific financial support provided."
    )


def test_no_data_sentence_plan_bare_phrase_names_elected_concept_when_confirmed():
    """The bare no-data phrase names a confirmed concept before generic guidance."""
    plan = no_data_sentence_plan(
        "no_observations",
        elected_stat_vars=[GraphRef(ref="Count_Person", label="Total population")],
    )
    assert plan == [
        SentenceToken(
            kind="text",
            text=(
                "No data for total population. Try a different place, time period, or indicator."
            ),
        )
    ]


def test_no_data_sentence_plan_nearest_real_names_nearest_series_when_no_flagged_match():
    nearest = [_nearest_real(_spec(), dropped_property="TestPurposeProp")]
    plan = no_data_sentence_plan(
        "role_incompatible", nearest_real=nearest, variable_term="bilateral aid disbursements"
    )
    assert sentence_from_plan(plan) == (
        'No data for "bilateral aid disbursements" as a flow to or from any place '
        "named in this query. Data exists for development finance flow. Check that it "
        "measures what you asked for."
    )


def test_no_data_sentence_plan_role_incompatible_drops_directional_flagged_match():
    """A retained flagged identity can confirm the resolved entities without
    contradicting a refusal to publish that directional flow."""
    flagged = _matched_spec(
        _spec(
            entities=[
                _entity_from("country/FRA", "France"),
                _entity_to("country/KEN", "Kenya"),
            ]
        )
    )
    assert flagged.resolved.rendered_sentence == "France's development finance flow to Kenya."

    plan = no_data_sentence_plan("role_incompatible", flagged_match=flagged)

    assert plan == [
        SentenceToken(
            kind="text",
            text=(
                "The matched series is not published as a flow to or from any place named in "
                "this query."
            ),
        )
    ]
    assert "France's development finance flow to Kenya." not in sentence_from_plan(plan)


def test_no_data_sentence_plan_qualifier_not_bound_drops_widened_flagged_match():
    """A widened flagged match stays out of the `qualifier_not_bound` response."""
    flagged = _matched_spec(_spec())
    qualifier = ResolvedQualifier(text="budget support", named=[], slot=None, disposition="unbound")

    plan = no_data_sentence_plan(
        "qualifier_not_bound", flagged_match=flagged, qualifier_terms=[qualifier]
    )

    assert plan == [
        SentenceToken(
            kind="text",
            text='"budget support" names a category the search could not apply to this query.',
        )
    ]
    assert "matching series" not in sentence_from_plan(plan)


_GDP_REF = GraphRef(ref="Amount_EconomicActivity_GDP", label="Gross domestic product")
_GNI_REF = GraphRef(ref="Amount_EconomicActivity_GNI", label="Gross national income")


@pytest.mark.parametrize(
    ("denominator", "said"),
    [
        (
            NoDataDenominator(asked=_GDP_REF, found=_GNI_REF),
            " The question names gross domestic product as the base, and the closest series "
            "divides by gross national income.",
        ),
        (
            NoDataDenominator(asked=_GDP_REF, found=None),
            " The question names gross domestic product as the base, and the closest series "
            "states none.",
        ),
        (
            NoDataDenominator(asked=None, found=_GNI_REF),
            " The closest series divides by gross national income.",
        ),
        (NoDataDenominator(), ""),
    ],
    ids=["asked-and-found", "asked-only", "found-only", "neither"],
)
def test_a_denominator_refusal_names_the_denominators_and_offers_the_flagged_match(
    denominator: NoDataDenominator, said: str
):
    """The phrase stays first, for the envelope to reframe; the denominators compared, the
    flagged match as the closest series and the level nearest_real follow."""
    plan = no_data_sentence_plan(
        "denominator_not_available",
        flagged_match=_matched_spec(_spec()),
        nearest_real=[_nearest_real(_spec(), dropped_property="measurementDenominator")],
        variable_term="debt to GDP ratio",
        denominator=denominator,
    )
    assert plan[0].text == 'No series for "debt to GDP ratio" is in the form it asks for.'
    assert sentence_from_plan(plan) == (
        'No series for "debt to GDP ratio" is in the form it asks for.'
        f"{said} The closest series is development finance flow. "
        "Data exists for development finance flow."
    )


def test_no_data_sentence_plan_bare_phrase_when_no_near_match():
    """A bare no-observations response includes its generic guidance."""
    plan = no_data_sentence_plan("no_observations")
    assert plan == [
        SentenceToken(
            kind="text",
            text="No data found for this query. Try a different place, time period, or indicator.",
        )
    ]


def test_no_data_sentence_plan_flattens_near_match_token_metadata():
    """Near-match tokens lose editable slot and entity metadata."""
    near_match = _spec(
        slots=[_what_slot("Total population"), _where_place_slot("country/KEN", "Kenya")],
        entities=[_entity_subject("country/KEN", "Kenya")],
    )

    plan = no_data_sentence_plan(
        "no_matching_slice",
        gate="only_alternate_has_data",
        alternate_match=near_match,
        variable_term="net ODA received",
    )

    assert sentence_from_plan(plan) == (
        'No published series for "net ODA received" matches the combination this query '
        "asks for. Data exists for Kenya's total population. Check that it measures what "
        "you asked for."
    )
    for token in plan[1:]:
        assert token.kind != "slot"
        assert token.axis is None
        assert token.property_dcid is None
        assert token.binding_ref is None
        assert token.role is None
        assert token.entity_refs == []
        assert token.distinguishes is None
        assert token.part_index is None


def _stating_nothing() -> Spec:
    """A Spec whose heading and byline state nothing: no measure, StatVar or shape
    name, no place, period or source."""
    spec = _spec(stat_vars=[])
    shape = spec.resolved.shape.model_copy(
        update={"label": "", "population_type": None, "measured_property": None, "stat_type": None}
    )
    return spec.model_copy(update={"resolved": spec.resolved.model_copy(update={"shape": shape})})


@pytest.mark.parametrize(
    ("reason", "gate", "alternate", "expected"),
    [
        pytest.param(
            "no_observations",
            "no_member_has_data_at_place",
            False,
            'No data for "net ODA received". Try a different place, time period, or indicator.',
            id="place-gap-claim",
        ),
        pytest.param(
            "no_matching_slice",
            "only_alternate_has_data",
            True,
            'No published series for "net ODA received" matches the combination this query'
            " asks for.",
            id="alternate",
        ),
        pytest.param(
            "no_observations",
            None,
            False,
            'No data for "net ODA received". Try a different place, time period, or indicator.',
            id="nearest-series",
        ),
    ],
)
def test_a_near_match_whose_heading_states_nothing_leaves_the_bare_phrase(
    reason: NoDataReason, gate: NoDataGate | None, alternate: bool, expected: str
):
    """A near-match whose heading has no text names nothing, so the verdict stands
    alone."""
    match = _stating_nothing()
    assert sentence_from_plan(render_sentence_plan(match)) == ""
    plan = no_data_sentence_plan(
        reason,
        gate=gate,
        flagged_match=None if alternate else match,
        alternate_match=match if alternate else None,
        variable_term="net ODA received",
    )
    assert sentence_from_plan(plan) == expected


@pytest.mark.parametrize(
    "reason",
    [
        "no_observations",
        "variable_not_resolved",
        "no_matching_slice",
        "role_incompatible",
        "entity_not_resolved",
    ],
)
@pytest.mark.parametrize(
    "gate",
    [
        "no_member_has_data_at_place",
        "only_alternate_has_data",
        "no_member_asserts_bound_properties",
    ],
)
def test_no_data_sentence_plan_head_is_always_no_data_phrase(reason: str, gate: str):
    """Every gate starts with the full no-data phrase for its reason and variable, which
    the envelope swaps for its own framing of a reason that quotes the variable. A place
    gap for any other reason opens with its claim instead."""
    variable_term = "unemployment rate"
    near_match = _matched_spec(
        _spec(entities=[_entity_subject("country/KEN", "Kenya")])
        if gate == "no_member_has_data_at_place"
        else _spec()
    )
    kwargs = (
        {"alternate_match": near_match}
        if gate == "only_alternate_has_data"
        else {"flagged_match": near_match}
    )
    plan = no_data_sentence_plan(
        reason, gate=cast(NoDataGate, gate), variable_term=variable_term, **kwargs
    )
    expected_head = no_data_phrase(reason, variable_term=variable_term)
    if gate == "no_member_has_data_at_place" and composed_variable_outcome(reason) is None:
        expected_head = "No data for "
    assert plan[0] == SentenceToken(kind="text", text=expected_head)


# Requested windows outside the confirmed date range.


def _population_stat_var(start: str, end: str) -> StatVar:
    return StatVar(
        ref=_ref("Count_Person", "Total population"),
        shape_id="test-shape",
        slot_values=[],
        constraint_properties=[],
        data_date_range=DateRange(start=start, end=end),
    )


def _window_note_spec(
    *,
    ask_start: int,
    ask_end: int,
    series_start: str = "1960",
    series_end: str = "2024",
    stat_vars: list[StatVar] | None = None,
    coverage: Coverage | None = None,
    available_at_place: bool | None = None,
) -> Spec:
    return _spec(
        slots=[_where_place_slot("country/KEN", "Kenya")],
        entities=[_entity_subject("country/KEN", "Kenya")],
        window=TimeWindow(start_year=ask_start, end_year=ask_end),
        requested_window=TimeWindow(start_year=ask_start, end_year=ask_end),
        measured_property_label="Total population",
        stat_vars=(
            stat_vars if stat_vars is not None else [_population_stat_var(series_start, series_end)]
        ),
        coverage=coverage,
        available_at_place=available_at_place,
    )


@pytest.mark.parametrize(
    ("ask_start", "ask_end", "available_at_place"),
    [
        pytest.param(1875, 1875, None, id="before-series"),
        pytest.param(2030, 2035, None, id="after-series"),
        pytest.param(1950, 2030, None, id="clipped-both-sides"),
        pytest.param(1875, 1875, False, id="flagged-match"),
    ],
)
def test_a_named_window_states_the_served_span_and_no_note(
    ask_start: int, ask_end: int, available_at_place: bool | None
):
    """The spec sentence holds its heading and byline only: the period is the span the
    answer serves, and a gap against the years asked is a `year_unavailable` departure on
    the part, never a note here."""
    spec = _window_note_spec(
        ask_start=ask_start, ask_end=ask_end, available_at_place=available_at_place
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert all(token.kind != "note" for token in plan)
    assert _rendered(spec) == "Kenya's total population, 1960 to 2024."


def test_a_latest_request_states_the_served_year_and_opens_the_byline_with_its_label():
    spec = _spec(
        slots=[_where_place_slot("country/KEN", "Kenya")],
        entities=[_entity_subject("country/KEN", "Kenya")],
        measured_property_label="Total population",
        stat_vars=[_population_stat_var("2024", "2024")],
        requested_window="latest",
    )
    plan = render_sentence_plan(spec)
    assert _rendered(spec) == "Kenya's total population in 2024. Latest year with data."
    label = next(token for token in plan if token.text == LATEST_YEAR_LABEL)
    assert (label.kind, label.segment) == ("text", "byline")


# Recorded queries: each no-data heading in claim form.

_MLS_020_STAT_VAR = (
    "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
    "CRSPurposeCode11110_Nominal"
)


def test_mls_020_quotes_the_users_words_and_never_a_stat_var_dcid():
    """mls-020: the elected StatVar has no graph name (its label is its dcid), so the
    heading names the variable in the user's own words."""
    year = "You asked for 2010. The data covers only 2011 to 2024."
    plan = no_data_sentence_plan(
        "no_observations",
        elected_stat_vars=[GraphRef(ref=_MLS_020_STAT_VAR, label=_MLS_020_STAT_VAR)],
        variable_term="imputed multilateral ODA",
        departures=[Departure(code="year_unavailable", binding_ref="when:", text=year)],
    )
    assert sentence_from_plan(plan) == (
        'No data for "imputed multilateral ODA". Try a different place, time period, or '
        f"indicator. {year}"
    )


def test_crs_020_names_the_series_with_data_in_claim_form():
    """crs-020: the elected series has no data at the place and an alternate does. The
    heading names the checked measure by its graph name, then the series that holds
    data."""
    alternate = _matched_spec(
        _spec(
            slots=[_where_place_slot("africa/SSA", "Sub-Saharan Africa")],
            entities=[_entity_subject("africa/SSA", "Sub-Saharan Africa")],
            window=TimeWindow(start_year=2000, end_year=2022),
        )
    )
    plan = no_data_sentence_plan(
        "no_observations",
        gate="only_alternate_has_data",
        alternate_match=alternate,
        elected_stat_vars=[
            GraphRef(
                ref="sdg/DC_ODA_POVDLG",
                label="Official development assistance grants for poverty reduction",
            )
        ],
        variable_term="aid",
    )
    assert sentence_from_plan(plan) == (
        "No data for official development assistance grants for poverty reduction. Data "
        "exists for Sub-Saharan Africa's development finance flow, 2000 to 2022. Check "
        "that it measures what you asked for."
    )


def test_ids_028_a_role_refusal_is_a_claim():
    plan = no_data_sentence_plan("role_incompatible", variable_term="debt share")
    assert sentence_from_plan(plan) == (
        'No data for "debt share" as a flow to or from any place named in this query.'
    )
