"""Test no-data match composition and NoData gate validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.engine.assemble import build_no_data_part
from search.engine.departures import window_gap_fact
from search.engine.retrieval.read_plan import WindowGap
from search.models import (
    EntityRoleKey,
    GraphRef,
    NearestRealSpec,
    NoData,
    NoDataPart,
    Relaxation,
    ResolvedFilter,
    SlotKey,
    Spec,
    TimeWindow,
    Warning,
)
from search.render.plan import render_sentence
from tests.engine._assemble_fixtures import (
    _directional_entity,
    _make_promoting_spec,
    _make_spec,
)

# nearest_real suggestions carry their own sentences.


def _to_nearest_real(spec, *, dropped_property: str) -> NearestRealSpec:
    return NearestRealSpec(
        spec_id=spec.spec_id,
        definition=spec.definition,
        resolved=spec.resolved,
        relaxation=Relaxation(
            axis="how", property_dcid=dropped_property, dropped_value_dcids=["Health"]
        ),
    )


def test_build_no_data_part_keeps_how_property_that_distinguishes_nearest_real_suggestions():
    """nearest_real siblings can collapse the same way build_candidates_part's
    clamped list can. _suggest_nearest_real relaxes one binding at a time
    to unbound, so a bound-vs-unbound pair between two suggestions is exactly
    the shape the guard exists to keep distinct."""
    spec_bound = _make_promoting_spec(purpose_dcid="Health", place_dcid="country/ETH")
    spec_unbound = _make_promoting_spec(purpose_dcid=None, place_dcid="country/ETH")
    nearest_real = [
        _to_nearest_real(spec_bound, dropped_property="TestPlaceProp"),
        _to_nearest_real(spec_unbound, dropped_property="TestPurposeProp"),
    ]

    part = build_no_data_part(
        "no_matching_slice",
        variable_text="test variable",
        warnings=[],
        nearest_real=nearest_real,
    )

    assert part.no_data.nearest_real is not None
    assert len(part.no_data.nearest_real) == 2
    first, second = part.no_data.nearest_real
    assert first.resolved.rendered_sentence != second.resolved.rendered_sentence


# Compose flagged and alternate matches into the no-data sentence.


def _with_slot_filter(spec: Spec, slot_filter: ResolvedFilter) -> Spec:
    trace = spec.resolved.trace.model_copy(
        update={"slot_filters": [*spec.resolved.trace.slot_filters, slot_filter]}
    )
    return spec.model_copy(update={"resolved": spec.resolved.model_copy(update={"trace": trace})})


def _heading(part: NoDataPart) -> str:
    """The part's sentence before its limits and departures."""
    return "".join(t.text for t in part.sentence_plan or [] if t.segment != "departure")


def test_build_no_data_part_composes_no_member_has_data_at_place_gate():
    """The claim states the checked series, then the sentence naming the flow type the
    engine filled by its declared default, which the request never named."""
    spec = _with_slot_filter(
        _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH"),
        ResolvedFilter(
            key=SlotKey(
                axis="how",
                property=GraphRef(ref="crsTypeOfFlow", label="CRS type of flow"),
                label="flow",
            ),
            binding_kind="value",
            refs=[GraphRef(ref="CRSTypeOfFlow10", label="official development assistance")],
            provenance="declared_default",
        ),
    )
    near_sentence = render_sentence(spec)

    part = build_no_data_part(
        "no_observations",
        variable_text="test variable",
        warnings=[],
        flagged_match=spec,
        gate="no_member_has_data_at_place",
        elected_stat_vars=[GraphRef(ref="Count_Person", label="Count person")],
    )

    assert part.no_data.flagged_match is not None
    assert part.no_data.flagged_match.resolved.rendered_sentence == near_sentence
    assert _heading(part) == "No data for ETH's ODA Grants for health."
    limits = [
        t.text for t in part.sentence_plan or [] if t.segment == "departure" and t.text != " "
    ]
    assert limits == ["This check covered only ODA."]


def test_build_no_data_part_no_member_has_data_at_place_names_recipient_not_donor():
    """The claim names the donor and the recipient the availability gate checked."""
    spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        entities=[
            _directional_entity("country/USA", "origin", "United States"),
            _directional_entity("country/ETH", "destination", "Ethiopia"),
        ],
    )
    part = build_no_data_part(
        "no_observations",
        variable_text="test variable",
        warnings=[],
        flagged_match=spec,
        gate="no_member_has_data_at_place",
    )

    assert _heading(part) == "No data for United States' ODA Grants to Ethiopia for health."


def test_build_no_data_part_no_member_has_data_at_place_donor_only_falls_back():
    """Without a recipient, the claim names the donor and the checked place."""
    spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        entities=[_directional_entity("country/USA", "origin", "United States")],
    )
    part = build_no_data_part(
        "no_observations",
        variable_text="test variable",
        warnings=[],
        flagged_match=spec,
        gate="no_member_has_data_at_place",
    )

    assert _heading(part) == "No data for United States' ODA Grants for health in ETH."


_WEAK = Warning(code="retrieval_score_weak", severity="info", message="weak")


@pytest.mark.parametrize(
    ("variable_text", "warnings", "departs"),
    [
        pytest.param("cost of living", [_WEAK], True, id="weak-and-unnamed"),
        pytest.param("cost of living", [], False, id="strong-score"),
        pytest.param("net ODA received", [_WEAK], False, id="names-the-measure"),
    ],
)
def test_build_no_data_part_composes_only_alternate_has_data_gate(
    variable_text: str, warnings: list[Warning], departs: bool
):
    """The alternate's sentence follows the verdict. It departs as `closest_match` only
    when the closest-match rule fires on its reading; no note token is written."""
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")
    near_sentence = render_sentence(spec)

    part = build_no_data_part(
        "no_matching_slice",
        variable_text=variable_text,
        warnings=warnings,
        alternate_match=spec,
        gate="only_alternate_has_data",
        elected_stat_vars=[GraphRef(ref="Count_Person_Alternate", label="Alternate count")],
    )

    assert part.no_data.flagged_match is None
    assert part.no_data.alternate_match is not None
    alternate_plan = part.no_data.alternate_match.resolved.sentence_plan or []
    assert part.no_data.alternate_match.resolved.rendered_sentence == near_sentence
    assert {t.segment for t in alternate_plan} <= {"heading", "byline"}
    assert not [t for t in part.sentence_plan or [] if t.kind == "note"]
    lead = (
        f'No published series for "{variable_text}" matches the combination this query '
        "asks for. Data exists for ETH's ODA Grants for health. Check that it measures "
        "what you asked for."
    )
    if not departs:
        assert part.departures == []
        assert part.rendered_sentence == lead
        return
    [departure] = part.departures
    assert (departure.code, departure.asked, departure.binding_ref) == (
        "closest_match",
        variable_text,
        None,
    )
    assert departure.shown
    assert part.rendered_sentence == f"{lead} {departure.text}"


def test_build_no_data_part_calls_elected_kenya_debt_match_matching_series():
    """A bound-property refusal keeps the elected debt series as the match."""
    spec = _make_promoting_spec(purpose_dcid="Debt", place_dcid="country/KEN")
    spec.resolved.members[0].ref.label = "External debt stocks"

    part = build_no_data_part(
        "no_matching_slice",
        variable_text="external debt",
        warnings=[],
        flagged_match=spec,
        gate="no_member_asserts_bound_properties",
        elected_stat_vars=[
            GraphRef(ref="Test_Measured_Property_GDP", label="External debt stocks")
        ],
    )

    assert "The matching series is " in part.rendered_sentence
    assert "nearest series" not in part.rendered_sentence


def test_build_no_data_part_counts_the_elected_stat_vars_on_the_wire():
    """Count the elected StatVars beside the alternate match."""
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")
    elected = [GraphRef(ref="Count_MedicalConditionIncident_ConditionHIVAIDS", label="HIV cases")]

    part = build_no_data_part(
        "no_observations",
        variable_text="test variable",
        warnings=[],
        alternate_match=spec,
        gate="only_alternate_has_data",
        elected_stat_vars=elected,
    )

    assert part.no_data.elected_count == 1


def test_build_no_data_part_carries_gate_and_alternate_match_onto_the_wire():
    """Carry the gate and render the alternate match's own sentence."""
    alternate = _make_spec(
        sv_dcid="sdg/SH_HIV_INCD", purpose_dcid="DAC/Health", place_dcid="country/ETH"
    )
    near_sentence = render_sentence(alternate)

    part = build_no_data_part(
        "no_matching_slice",
        variable_text="test variable",
        warnings=[],
        gate="only_alternate_has_data",
        alternate_match=alternate,
    )

    assert part.no_data.flagged_match is None
    assert part.no_data.gate == "only_alternate_has_data"
    assert part.no_data.alternate_match is not None
    assert part.no_data.alternate_match.resolved.rendered_sentence == near_sentence


def test_build_no_data_part_composes_no_member_asserts_bound_properties_gate():
    """This gate uses the no_matching_slice reason."""
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")
    near_sentence = render_sentence(spec)

    part = build_no_data_part(
        "no_matching_slice",
        variable_text="maternal mortality ratio",
        warnings=[],
        flagged_match=spec,
        gate="no_member_asserts_bound_properties",
        elected_stat_vars=[GraphRef(ref="Count_Person", label="Count person")],
    )

    assert part.no_data.flagged_match is not None
    assert part.no_data.flagged_match.resolved.rendered_sentence == near_sentence
    assert _heading(part) == (
        'No published series for "maternal mortality ratio" matches the combination this '
        "query asks for. The matching series is ETH's ODA Grants for health. Check that it "
        "measures what you asked for."
    )


def test_build_no_data_part_composes_nearest_real_lead_in_when_no_flagged_match():
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")
    nearest_real = [_to_nearest_real(spec, dropped_property="TestPurposeProp")]

    part = build_no_data_part(
        "role_incompatible",
        variable_text="bilateral aid disbursements",
        warnings=[],
        nearest_real=nearest_real,
    )

    assert part.no_data.nearest_real is not None
    assert part.rendered_sentence == (
        'No data for "bilateral aid disbursements" as a flow to or from any place '
        "named in this query. Data exists for ETH's ODA Grants for health. Check that it "
        "measures what you asked for."
    )


def test_build_no_data_part_bare_phrase_when_neither_flagged_match_nor_nearest_real():
    """A bare no_observations envelope carries generic guidance."""
    part = build_no_data_part(
        "no_observations",
        variable_text=None,
        warnings=[],
    )

    expected = "No data found for this query. Try a different place, time period, or indicator."
    assert part.no_data.flagged_match is None
    assert part.no_data.nearest_real is None
    assert part.rendered_sentence == expected
    assert part.sentence_plan is not None
    assert len(part.sentence_plan) == 1
    assert part.sentence_plan[0].text == expected


@pytest.mark.parametrize(
    ("variable_text", "expected_sentence"),
    [
        pytest.param(
            "net ODA disbursements",
            'Could not match "net ODA disbursements" to a supported indicator. '
            "Try a different wording or a related indicator.",
            id="present-term",
        ),
        pytest.param(
            None,
            "Could not tell which indicator this query asks for. "
            "Try adding an indicator such as population or GDP.",
            id="none-term",
        ),
    ],
)
def test_build_no_data_part_variable_text(
    variable_text: str | None, expected_sentence: str
) -> None:
    """Use the supplied variable text as-is."""
    part = build_no_data_part(
        "variable_not_resolved",
        variable_text=variable_text,
        warnings=[],
    )

    assert part.rendered_sentence == expected_sentence
    assert part.sentence_plan is not None
    assert len(part.sentence_plan) == 1
    assert part.sentence_plan[0].text == expected_sentence


# NoData gate and match validation.


def test_no_data_alternate_match_requires_explicit_gate():
    """An alternate match requires an explicit gate."""
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")

    outcome = NoData(
        reason="no_matching_slice",
        gate="only_alternate_has_data",
        alternate_match=spec,
    )

    assert outcome.flagged_match is None
    assert outcome.alternate_match is spec
    assert outcome.gate == "only_alternate_has_data"


def test_no_data_flagged_match_requires_explicit_gate():
    """A flagged match requires an explicit gate."""
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")

    outcome = NoData(
        reason="no_observations",
        gate="no_member_has_data_at_place",
        flagged_match=spec,
    )

    assert outcome.flagged_match is spec
    assert outcome.alternate_match is None
    assert outcome.gate == "no_member_has_data_at_place"


def test_no_data_flagged_match_and_alternate_match_never_both_present():
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")

    with pytest.raises(ValidationError):
        NoData(
            reason="no_observations",
            flagged_match=spec,
            alternate_match=spec,
            gate="no_member_has_data_at_place",
        )


def test_no_data_gate_present_without_either_spec_is_rejected():
    with pytest.raises(ValidationError):
        NoData(reason="no_observations", gate="no_member_has_data_at_place")


def test_no_data_spec_present_without_gate_is_rejected():
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")
    with pytest.raises(ValidationError):
        NoData(reason="no_matching_slice", alternate_match=spec)


def test_no_data_role_incompatible_allows_its_pre_materialise_flagged_match():
    """A role-incompatible refusal can carry a match without a gate."""
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")

    outcome = NoData(reason="role_incompatible", flagged_match=spec)

    assert outcome.gate is None
    assert outcome.flagged_match is spec


def test_no_data_other_flagged_match_without_gate_is_rejected():
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")

    with pytest.raises(ValidationError):
        NoData(reason="no_observations", flagged_match=spec)


@pytest.mark.parametrize(
    ("gate", "carrier"),
    [
        ("only_alternate_has_data", "flagged_match"),
        ("no_member_has_data_at_place", "alternate_match"),
        ("no_member_asserts_bound_properties", "alternate_match"),
    ],
)
def test_no_data_gate_requires_its_matching_carrier(gate: str, carrier: str):
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")

    with pytest.raises(ValidationError):
        NoData(reason="no_observations", gate=gate, **{carrier: spec})


def test_no_data_rejects_unknown_gate():
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")

    with pytest.raises(ValidationError):
        NoData(reason="no_observations", gate="unknown_gate", flagged_match=spec)


# role_axis supplies the open flow axis to no_observations phrasing.


@pytest.mark.parametrize(
    ("role_axis", "expected_sentence"),
    [
        pytest.param(
            "destination",
            "No recipient has data for this query. "
            "Try a different place, time period, or indicator.",
            id="destination",
        ),
        pytest.param(
            None,
            "No data found for this query. Try a different place, time period, or indicator.",
            id="none",
        ),
    ],
)
def test_build_no_data_part_role_axis(
    role_axis: EntityRoleKey | None, expected_sentence: str
) -> None:
    part = build_no_data_part(
        "no_observations",
        variable_text=None,
        warnings=[],
        role_axis=role_axis,
    )
    assert part.rendered_sentence == expected_sentence


@pytest.mark.parametrize(
    ("label", "year", "span", "departure"),
    [
        pytest.param(
            "Gross domestic product (GDP), Per capita, purchasing power parity (PPP) "
            "international dollar, ICP benchmarks 2017-2021",
            1975,
            ("1980", "2025"),
            "You asked for 1975. The data covers only 1980 to 2025.",
            id="weo-032",
        ),
        pytest.param(
            "Nominal gross domestic product per capita",
            2025,
            ("1960", "2024"),
            "You asked for 2025. The data covers only 1960 to 2024.",
            id="weo-045",
        ),
    ],
)
def test_a_window_answer_departs_as_year_unavailable_with_no_note(
    label: str, year: int, span: tuple[str, str], departure: str
):
    """A recorded no-data answer for a year its series holds no point in: the verdict, then
    the `year_unavailable` departure, and no note token."""
    gap = WindowGap(years=TimeWindow(start_year=year, end_year=year), first=span[0], last=span[1])
    part = build_no_data_part(
        "no_observations",
        variable_text="GDP per capita",
        warnings=[],
        elected_stat_vars=[GraphRef(ref="sv/recorded", label=label)],
        window_gap_fact=window_gap_fact(gap),
    )

    assert [(d.code, d.binding_ref, d.asked, d.text) for d in part.departures] == [
        ("year_unavailable", "when:", str(year), departure)
    ]
    assert part.rendered_sentence == (
        f"No data for {label[0].lower()}{label[1:]}. Try a different place, time period, or "
        "indicator. "
        f"{departure}"
    )
    assert all(token.kind != "note" for token in part.sentence_plan or [])


@pytest.mark.parametrize(
    ("reason", "terms"),
    [
        ("entity_not_resolved", {"entity_terms": ["Atlantis"]}),
        ("variable_not_resolved", {}),
        ("no_observations", {}),
    ],
)
def test_build_no_data_part_message_is_its_bare_sentence(reason, terms):
    """`no_data.message` equals the part's sentence when no near match is named."""
    part = build_no_data_part(reason, variable_text="atlantis gdp", warnings=[], **terms)

    assert part.no_data.message == part.rendered_sentence
    for term in terms.get("entity_terms", ()):
        assert term in part.no_data.message
