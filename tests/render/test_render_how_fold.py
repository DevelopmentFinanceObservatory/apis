"""Checks how-slot folding, duplicate suppression, and empty placeholders."""

from __future__ import annotations

from search.models import (
    BindingSet,
    BindingUnbound,
    BindingValue,
    Shape,
    Slot,
    SlotValue,
    StatVar,
    TimeWindow,
)
from search.render.heading import heading_from_parts
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import plan_tokens, sentence_from_plan
from tests.render._helpers import (
    _gender_slot,
    _how_slot,
    _ref,
    _rendered,
    _residence_type_slot,
    _slot_key,
    _spec,
    _what_slot,
    _where_slot,
    assert_round_trip,
)

# Adjectival how-slot word order.


def test_single_adjectival_how_slot_folds_before_what():
    """A single bound adjectival how-slot folds ahead of the "what" noun:
    "Female population in Kenya", not "Population, Female" / "... for Female"."""
    spec = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Population", "population"),
        slots=[_gender_slot("schemaFemale", "Female"), _where_slot("Kenya")],
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
    assert sentence_from_plan(plan) == "Kenya's female population."

    # The folded token is a real "how" slot token -- axis/property_dcid/
    # binding_ref intact so frontend chips stay clickable and anchored --
    # positioned ahead of "what" in the token list rather than after it.
    gender_tok = next(t for t in plan if t.property_dcid == "gender")
    assert gender_tok.kind == "slot"
    assert gender_tok.axis == "how"
    assert gender_tok.binding_ref == "how:gender"
    assert gender_tok.text == "female"

    what_idx = next(i for i, t in enumerate(plan) if t.axis == "what")
    assert plan.index(gender_tok) < what_idx


def test_multiple_adjectival_how_slots_fold_in_slot_order():
    """Two bound adjectival how-slots fold together, in their existing slot
    order among themselves: "Female Urban population", not "Urban Female ..."."""
    spec = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Population", "population"),
        slots=[
            _gender_slot("schemaFemale", "Female"),
            _residence_type_slot("schemaUrban", "Urban"),
            _where_slot("Kenya"),
        ],
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
    assert _rendered(spec) == "Kenya's female Urban population."


def test_adjectival_and_nonadjectival_how_slots_mix():
    """An adjectival how-slot folds before "what"; a non-adjectival one stays in
    the post-noun how-position with its own connective, unaffected by folding."""
    slot = Slot(
        key=_slot_key("how", "gender", "Gender", "gender"),
        binding=BindingValue(
            value=SlotValue(ref=_ref("schemaFemale", "Female"), value_kind="entity")
        ),
    )
    spec = _spec(slots=[slot, _how_slot("health")])
    assert _rendered(spec) == "Female development finance flow for health."


def test_unbound_adjectival_how_slot_stays_in_how_position():
    """An UNBOUND adjectival slot stays unfolded: its empty placeholder token
    stays in the post-noun how-position (unmoved), so the frontend's
    narrow-further hook still finds it exactly where it always was."""
    unbound_gender = Slot(
        key=_slot_key("how", "gender", "Gender", "gender"),
        binding=BindingUnbound(),
    )
    spec = _spec(slots=[_what_slot("ODA Grants"), unbound_gender])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "ODA Grants."

    what_idx = next(i for i, t in enumerate(plan) if t.axis == "what")
    how_idx = next(i for i, t in enumerate(plan) if t.axis == "how")
    assert how_idx > what_idx  # stays after "what"
    how_tok = plan[how_idx]
    assert how_tok.kind == "slot"
    assert how_tok.text == ""
    assert how_tok.binding_ref == "how:gender"


def test_set_bound_adjectival_how_slot_folds():
    """A set-bound (2+ values) adjectival slot folds the same way a
    single-value bound slot does -- one folded token, joined labels."""
    slot = Slot(
        key=_slot_key("how", "gender", "Gender", "gender"),
        binding=BindingSet(
            values=[
                SlotValue(ref=_ref("schemaFemale", "Female"), value_kind="entity"),
                SlotValue(ref=_ref("schemaMale", "Male"), value_kind="entity"),
            ]
        ),
    )
    spec = _spec(slots=[slot])
    assert _rendered(spec) == "Female and Male development finance flow."


# Avoid repeating adjectival values already present in a StatVar label.


def test_adjectival_fold_skips_word_already_named_by_sv_name_fallback():
    """Count_Person_Male's own graph name is "Male population" (a stopword
    measured_property routes _fallback_what to this single-SV-name branch --
    see _fallback_what); a resubmit-derived gender=Male how-slot binding
    suppresses its own fold of "Male" ahead of it, since the SV name already
    carries the word. Live bug: rendered
    "Male Male population in Kenya."."""
    spec = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Person", "Person"),
        slots=[_gender_slot("schemaMale", "Male"), _where_slot("Kenya")],
        stat_vars=[
            StatVar(
                ref=_ref("Count_Person_Male", "Male population"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Kenya's male population."

    # The gender slot token survives with its full metadata (still a clickable
    # frontend chip) -- only its text is suppressed, not the token itself.
    gender_tok = next(t for t in plan if t.property_dcid == "gender")
    assert gender_tok.kind == "slot"
    assert gender_tok.axis == "how"
    assert gender_tok.text == ""
    assert gender_tok.binding_ref == "how:gender"
    # The StatVar name says the value, so the measure token states the slot.
    assert gender_tok.states == []
    assert next(t for t in plan if t.axis == "what").states == ["how:gender"]


def test_adjectival_fold_skips_multiple_words_already_named_by_sv_name():
    """ "Total urban female population" buries BOTH the gender and
    residenceType constraint words mid-string (not as a shared prefix) --
    both bound adjectival how-slots must be suppressed, not just the first
    one folded. A bound `what` slot (rather than the single-SV-name path)
    keeps this test on the per-member redundancy gate (_adjectival_fold_text)
    instead of the StatVar-name fold, which would drop both how-slots
    regardless of what either one names."""
    spec = _spec(
        slots=[
            _what_slot("Total urban female population"),
            _gender_slot("schemaFemale", "Female"),
            _residence_type_slot("Urban", "Urban"),
            _where_slot("Kenya"),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Kenya's total urban female population."
    what_tok = next(t for t in plan if t.axis == "what")
    assert what_tok.states == [what_tok.binding_ref, "how:gender", "how:residenceType"]


def test_adjectival_fold_still_applies_when_word_not_named_by_what_text():
    """Guard: the redundancy check is specific, not a blanket suppression of
    every adjectival fold -- a value NOT already named in what_text folds
    normally. A bound `what` slot (rather than the single-SV-name path) keeps
    this test on the per-member redundancy gate instead of the StatVar-name
    fold, which would drop the whole how-slot regardless of what it names."""
    spec = _spec(
        slots=[
            _what_slot("Male population"),
            _residence_type_slot("Urban", "Urban"),
            _where_slot("Kenya"),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Kenya's urban Male population."


# Value-based duplicate and connective handling.


def test_boolean_dcid_how_value_drops_to_empty_placeholder():
    """A boolean how-slot member -- a ref dcid whose tail case-folds to
    True/False -- drops to an empty placeholder with no connective token, the
    same as an unbound slot."""
    slot = Slot(
        key=_slot_key(
            "how",
            "fixedBroadbandSubscription",
            "Fixed Broadband Subscription",
            "fixed broadband subscription",
        ),
        binding=BindingValue(value=SlotValue(ref=_ref("True", "True"), value_kind="enum_value")),
    )
    spec = _spec(
        measured_property=_ref("Count_Person_UsingInternet", "Percent of internet users"),
        slots=[slot, _where_slot("Ethiopia")],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Ethiopia's percent of internet users."
    assert not any(t.kind == "text" and t.text == " for " for t in plan)
    # Nothing in the sentence says the dropped value, so no token states its slot.
    assert not any("how:fixedBroadbandSubscription" in t.states for t in plan)
    how_tok = next(t for t in plan if t.property_dcid == "fixedBroadbandSubscription")
    assert how_tok.kind == "slot"
    assert how_tok.axis == "how"
    assert how_tok.text == ""
    assert how_tok.binding_ref == "how:fixedBroadbandSubscription"


def test_boolean_literal_how_value_drops_to_empty_placeholder():
    """A literal "true"/"false" how-slot member (no ref at all) is a boolean
    flag too -- the same drop as a boolean dcid, case-folded.

    A literal-valued what/how SlotValue never survives spec identity minting
    (see spec_id.py's `_slot_binding_member`), so this calls
    heading_from_parts directly -- the same bypass
    test_two_adjectival_folds_one_suppressed_one_not_exact_tokens uses --
    rather than building a full Spec through the `_spec()` fixture helper."""
    shape = Shape(
        shape_id="test-shape",
        label="Test Shape",
        population_type=_ref("DevelopmentFinance", "Development Finance"),
        measured_property=_ref("DevelopmentFinanceFlow", "Development Finance Flow"),
        stat_type=_ref("measuredValue", "Measured Value"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[],
        member_count=1,
    )
    slot = Slot(
        key=_slot_key("how", "isProvisional", "Provisional Flag", "provisional flag"),
        binding=BindingValue(value=SlotValue(value_kind="literal", literal="True")),
    )
    slots = [_what_slot("ODA Grants"), slot]
    plan = plan_tokens(
        heading_from_parts(slots, [], shape, window=None, pinned_properties=frozenset())
    )
    assert_round_trip(plan, sentence_from_plan(plan))
    assert sentence_from_plan(plan) == "ODA Grants."
    how_tok = next(t for t in plan if t.property_dcid == "isProvisional")
    assert how_tok.kind == "slot"
    assert how_tok.text == ""
    assert how_tok.binding_ref == "how:isProvisional"


def test_how_phrase_leading_preposition_gets_no_connective():
    """A post-what how-slot phrase that already opens with a preposition
    ("with covered employment") gets no connective word of its own -- "for
    with covered employment" would double the preposition -- but the plan
    still separates it from the "what" phrase with a plain space."""
    slot = Slot(
        key=_slot_key("how", "measurementMethod", "Measurement Method", "measurement method"),
        binding=BindingValue(
            value=SlotValue(
                ref=_ref("WithCoveredEmployment", "with covered employment"),
                value_kind="enum_value",
            )
        ),
    )
    spec = _spec(
        measured_property=_ref("MovingAverage1Week", "moving average 1 week window"),
        slots=[slot],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Moving average 1 week window with covered employment."
    assert not any(t.kind == "text" and t.text == " for " for t in plan)
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.text == "with covered employment"


def test_post_what_slot_folds_when_value_named_in_parenthetical():
    """A non-adjectival how-slot's value folds to an empty placeholder when
    its label already appears inside the what-text's own trailing
    parenthetical, case-insensitively -- "for Female" would repeat the word
    "(female)" already carries. A bound `what` slot (rather than the
    single-SV-name path) keeps this test on the per-member redundancy gate
    (_how_member_text's _already_named_in check) instead of the StatVar-name
    fold, which would drop the how-slot regardless of what it names."""
    slot = Slot(
        key=_slot_key("how", "measurementMethod", "Measurement Method", "measurement method"),
        binding=BindingValue(
            value=SlotValue(ref=_ref("schemaFemale", "Female"), value_kind="enum_value")
        ),
    )
    spec = _spec(
        slots=[_what_slot("Maternal mortality ratio (female)"), slot],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Maternal mortality ratio (female)."
    assert not any(t.kind == "text" and t.text == " for " for t in plan)
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.kind == "slot"
    assert how_tok.text == ""
    assert how_tok.binding_ref == "how:measurementMethod"


def test_set_bound_adjectival_slot_folds_whole_when_any_member_named():
    """An adjectival set slot empties when any member already appears in the what text.

    A bound `what` slot keeps the test on the per-member redundancy gate; a single StatVar name
    would drop the whole how-slot regardless.
    """
    slot = Slot(
        key=_slot_key("how", "gender", "Gender", "gender"),
        binding=BindingSet(
            values=[
                SlotValue(ref=_ref("schemaFemale", "Female"), value_kind="entity"),
                SlotValue(ref=_ref("schemaMale", "Male"), value_kind="entity"),
            ]
        ),
    )
    spec = _spec(
        slots=[_what_slot("Female population"), slot, _where_slot("Togo")],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Togo's female population."
    # The noun says one member of two, so no token states the slot.
    assert not any("how:gender" in t.states for t in plan)
    gender_tok = next(t for t in plan if t.property_dcid == "gender")
    assert gender_tok.kind == "slot"
    assert gender_tok.axis == "how"
    assert gender_tok.text == ""
    assert gender_tok.binding_ref == "how:gender"


def test_post_what_set_bound_slot_drops_only_the_named_member():
    """A post-what set-bound how-slot drops only the member(s) whose label
    already appears in what_text, joining whatever survives with its own
    connective -- unlike the pre-what whole-fold above, this is a per-member
    survival, not an all-or-nothing one. A bound `what` slot (rather than the
    single-SV-name path) keeps this test on that per-member gate instead of
    the StatVar-name fold, which would drop the whole how-slot regardless."""
    slot = Slot(
        key=_slot_key("how", "populationSubgroup", "Population Subgroup", "population subgroup"),
        binding=BindingSet(
            values=[
                SlotValue(ref=_ref("schemaFemale", "Female"), value_kind="enum_value"),
                SlotValue(ref=_ref("schemaMale", "Male"), value_kind="enum_value"),
            ]
        ),
    )
    spec = _spec(slots=[_what_slot("Female population"), slot])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Female population for Male."
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.kind == "slot"
    assert how_tok.text == "Male"
    assert how_tok.binding_ref == "how:populationSubgroup"
    assert how_tok.states == ["how:populationSubgroup"]
    assert "how:populationSubgroup" not in next(t for t in plan if t.axis == "what").states


# A single StatVar label can account for its bound how-slots.


def test_statvar_name_folds_how_slot_for_connective_property():
    """The single-SV-name rung names "Nominal gross domestic product" directly
    (measured_property "amount" is a stopword) -- the bound
    activitySource=GrossDomesticProduction how-slot, which would otherwise
    render "for gross domestic production", folds to its empty placeholder.
    The empty token carries its slot's property_dcid/binding_ref, so a client
    chip can show the constraint even though the sentence itself does not
    restate it."""
    how_slot = Slot(
        key=_slot_key("how", "activitySource", "Activity Source", "activity source"),
        binding=BindingValue(
            value=SlotValue(
                ref=_ref("GrossDomesticProduction", "GrossDomesticProduction"),
                value_kind="entity",
            )
        ),
    )
    spec = _spec(
        measured_property=_ref("amount", "amount"),
        slots=[how_slot, _where_slot("Guatemala")],
        window=TimeWindow(start_year=1960, end_year=2024),
        resolved_sources=[_ref("WorldDevelopmentIndicators", "World Development Indicators")],
        stat_vars=[
            StatVar(
                ref=_ref(
                    "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
                    "Nominal gross domestic product",
                ),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == (
        "Guatemala's nominal gross domestic product, 1960 to 2024. World Development Indicators."
    )
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.kind == "slot"
    assert how_tok.text == ""
    assert how_tok.property_dcid == "activitySource"
    assert how_tok.binding_ref == "how:activitySource"
    assert next(t for t in plan if t.axis == "what").states == ["how:activitySource"]


def test_statvar_name_folds_how_slot_of_connective_property():
    """Same fold for an "of"-connective how property (placeOfResidenceClassification):
    "Total urban population" already names the single resolved StatVar, so the
    bound placeOfResidenceClassification=Urban how-slot -- which would
    otherwise render "of Urban" -- folds to its empty placeholder too."""
    how_slot = Slot(
        key=_slot_key(
            "how",
            "placeOfResidenceClassification",
            "Place Of Residence Classification",
            "residence classification",
        ),
        binding=BindingValue(value=SlotValue(ref=_ref("Urban", "Urban"), value_kind="entity")),
    )
    spec = _spec(
        measured_property=_ref("count", "count"),
        slots=[how_slot, _where_slot("Kenya")],
        stat_vars=[
            StatVar(
                ref=_ref("Count_Person_Urban", "Total urban population"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Kenya's total urban population."
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.text == ""
    assert how_tok.property_dcid == "placeOfResidenceClassification"


def test_statvar_name_folds_how_slot_catalog_title_case_label():
    """The fold applies just as readily to a catalog Title-Case StatVar label
    that itself needs sentence-casing: "Percent of Working Age Population in
    the Labor Force" normalizes to sentence case, and the bound
    employmentStatus=BLS_InLaborForce how-slot folds away underneath it just
    the same."""
    how_slot = Slot(
        key=_slot_key("how", "employmentStatus", "Employment Status", "employment status"),
        binding=BindingValue(
            value=SlotValue(ref=_ref("BLS_InLaborForce", "In the Labor Force"), value_kind="entity")
        ),
    )
    spec = _spec(
        measured_property=_ref("count", "count"),
        slots=[how_slot],
        stat_vars=[
            StatVar(
                ref=_ref(
                    "Percent_Person_BLS_InLaborForce",
                    "Percent of Working Age Population in the Labor Force",
                ),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == "Percent of working age population in the labor force."
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.text == ""


def test_how_slot_renders_normally_with_two_resolved_stat_vars():
    """Guard: two resolved StatVars can't fold by a single name (used_sv_name
    stays False, the same gate test_promotion_skipped_with_two_resolved_stat_vars
    exercises for the "what" token itself) -- a bound how-slot's own clause
    renders exactly as it always has."""
    spec = _spec(
        measured_property=_ref("fertilityRate", "fertilityRate"),
        slots=[_how_slot("Female")],
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
    assert _rendered(spec) == "Fertility rate for Female."


def test_keep_how_properties_prevents_the_fold():
    """keep_how_properties names a how-slot property dcid that must render even
    when used_sv_name would otherwise fold it away -- the guard
    assemble.assemble_candidates uses to keep two candidates distinct (see
    tests/engine/test_assemble_candidates_sentences.py)."""
    how_slot = Slot(
        key=_slot_key("how", "activitySource", "Activity Source", "activity source"),
        binding=BindingValue(
            value=SlotValue(
                ref=_ref("GrossDomesticProduction", "GrossDomesticProduction"),
                value_kind="entity",
            )
        ),
    )
    spec = _spec(
        measured_property=_ref("amount", "amount"),
        slots=[how_slot],
        stat_vars=[
            StatVar(
                ref=_ref(
                    "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
                    "Nominal gross domestic product",
                ),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ],
    )
    plan = render_sentence_plan(spec, keep_how_properties=frozenset({"activitySource"}))
    rendered = sentence_from_plan(plan)
    assert_round_trip(plan, rendered)
    assert rendered == "Nominal gross domestic product for gross domestic production."
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.text == "gross domestic production"
