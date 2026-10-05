"""Checks sentence plans, token metadata, and graph-free sentence rendering."""

from __future__ import annotations

from search.models import (
    BindingUnbound,
    OpenConstraintAxisFact,
    Slot,
    SlotKey,
    StatVar,
    TimeWindow,
)
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import (
    _entity_from,
    _entity_to,
    _how_slot,
    _ref,
    _slot_key,
    _spec,
    _unbound_how_slot,
    _what_slot,
    _where_slot,
    assert_round_trip,
)

# ---------------------------------------------------------------------------
# render_sentence_plan: the token structure render_sentence is derived from
# ---------------------------------------------------------------------------


def test_sentence_from_plan_matches_render_sentence():
    """The rendered sentence matches the text joined from its token plan."""
    spec = _spec(
        slots=[_what_slot("ODA grants"), _how_slot("health")],
        entities=[
            _entity_to("country/ETH", "Ethiopia"),
            _entity_from("country/USA", "the United States"),
        ],
        window=TimeWindow(start_year=2015, end_year=2023),
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))


def test_sentence_plan_what_token_carries_binding_ref():
    """A bound what token carries its slot reference and property."""
    spec = _spec(slots=[_what_slot("ODA Grants")])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    what_tok = plan[0]
    assert what_tok.kind == "slot"
    assert what_tok.text == "ODA Grants"
    assert what_tok.axis == "what"
    assert what_tok.property_dcid == "DevelopmentFinanceScheme"
    assert what_tok.binding_ref == "what:DevelopmentFinanceScheme"
    assert what_tok.role is None
    assert what_tok.entity_refs == []


def test_sentence_plan_fallback_what_has_no_binding_ref():
    """Fallback wording has no editable slot reference."""
    spec = _spec(measured_property_label="Development Finance Flow")
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    what_tok = plan[0]
    assert what_tok.kind == "slot"
    assert what_tok.text == "Development finance flow"
    assert what_tok.axis == "what"
    assert what_tok.property_dcid is None
    assert what_tok.binding_ref is None


def test_sentence_plan_unbound_how_slot_is_empty_placeholder():
    """An unbound how slot emits an empty token without a connective."""
    unbound_purpose = Slot(
        key=_slot_key("how", "DevelopmentFinancePurpose", "Purpose", "purpose"),
        binding=BindingUnbound(),
    )
    spec = _spec(slots=[_what_slot("ODA Grants"), unbound_purpose])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    how_toks = [t for t in plan if t.axis == "how"]
    assert len(how_toks) == 1
    assert how_toks[0].kind == "slot"
    assert how_toks[0].text == ""
    assert how_toks[0].binding_ref == "how:DevelopmentFinancePurpose"
    assert not any(t.kind == "text" and t.text == "for " for t in plan)
    assert sentence_from_plan(plan) == "ODA Grants."


def test_sentence_plan_no_how_slot_declared_emits_no_token():
    """A shape without a how slot emits no how token."""
    spec = _spec(slots=[_what_slot("ODA Grants")])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert not any(t.axis == "how" for t in plan)


# Unbound how slots and open-axis facts.


def test_sentence_plan_open_axis_fact_varying_renders_empty_placeholder():
    """A varying open axis stays empty while retaining its binding reference (C89)."""
    spec = _spec(
        slots=[_what_slot("ODA Grants"), _unbound_how_slot()],
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="DevelopmentFinancePurpose",
                varies=True,
                distinct_count=2,
                undeclared_count=0,
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.kind == "slot"
    assert how_tok.text == ""
    assert how_tok.binding_ref == "how:DevelopmentFinancePurpose"
    assert sentence_from_plan(plan) == "ODA Grants."


def test_sentence_plan_open_axis_fact_agreeing_on_nothing_renders_empty_placeholder():
    """A constant axis without an agreed value stays empty."""
    spec = _spec(
        slots=[_what_slot("ODA Grants"), _unbound_how_slot()],
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="DevelopmentFinancePurpose",
                varies=False,
                distinct_count=0,
                undeclared_count=1,
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.kind == "slot"
    assert how_tok.text == ""
    assert sentence_from_plan(plan) == "ODA Grants."


def test_sentence_plan_open_axis_fact_agreeing_on_a_value_states_it():
    """A constant open axis renders its agreed value beside a bound what label."""
    spec = _spec(
        slots=[_what_slot("ODA Grants"), _unbound_how_slot()],
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="DevelopmentFinancePurpose",
                varies=False,
                value=_ref("DAC/Health", "health"),
                distinct_count=1,
                undeclared_count=0,
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.kind == "slot"
    assert how_tok.text == "health"
    assert how_tok.binding_ref == "how:DevelopmentFinancePurpose"
    assert sentence_from_plan(plan) == "ODA Grants for health."


def test_sentence_plan_open_axis_fact_index_miss_value_renders_humanized():
    """An agreed value without an index label is humanized from its DCID."""
    spec = _spec(
        slots=[_what_slot("ODA Grants"), _unbound_how_slot()],
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="DevelopmentFinancePurpose",
                varies=False,
                value=_ref("Years15Onwards", "Years15Onwards"),
                distinct_count=1,
                undeclared_count=0,
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.kind == "slot"
    assert how_tok.text == "years 15 onwards"
    assert sentence_from_plan(plan) == "ODA Grants for years 15 onwards."


def test_sentence_plan_open_axis_fact_resolved_value_states_it_with_undeclared_noun():
    """A resolved fact-only value renders without a declared axis noun."""
    spec = _spec(
        slots=[_what_slot("ODA Grants")],
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="DevelopmentFinanceType",
                varies=False,
                value=_ref("Grant", "grants"),
                distinct_count=1,
                undeclared_count=0,
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.text == "grants"
    assert sentence_from_plan(plan) == "ODA Grants for grants."


def test_sentence_plan_open_axis_fact_with_no_noun_at_all_stays_silent():
    """An undeclared fact-only axis with no noun stays silent."""
    spec = _spec(
        slots=[_what_slot("ODA Grants")],
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="SomeUndeclaredProperty",
                varies=True,
                distinct_count=2,
                undeclared_count=0,
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert not any(t.axis == "how" for t in plan)
    assert sentence_from_plan(plan) == "ODA Grants."


def test_sentence_plan_how_clause_order_follows_axis_declarations():
    """Bound clauses precede facts; fact clauses follow declaration order."""
    modality_slot = Slot(
        key=_slot_key("how", "DevelopmentFinanceModality", "Modality", "modality"),
        binding=BindingUnbound(),
    )
    spec = _spec(
        slots=[modality_slot, _how_slot("health")],
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="DevelopmentFinanceModality",
                varies=True,
                distinct_count=3,
                undeclared_count=0,
            ),
            OpenConstraintAxisFact(
                property="DevelopmentFinanceType",
                varies=False,
                value=_ref("Grant", "grants"),
                distinct_count=1,
                undeclared_count=0,
            ),
        ],
    )
    plan = render_sentence_plan(
        spec,
        axis_declarations={
            "DevelopmentFinanceModality": "modality",
            "DevelopmentFinanceType": "finance type",
        },
    )
    assert_round_trip(plan, sentence_from_plan(plan))
    how_toks = [t for t in plan if t.axis == "how"]
    assert [t.text for t in how_toks] == ["health", "", "grants"]
    assert how_toks[1].binding_ref == "how:DevelopmentFinanceModality"
    assert how_toks[2].binding_ref is None


def test_sentence_plan_no_open_axis_facts_is_byte_for_byte_unchanged():
    """Without open-axis facts, an unbound how slot remains empty."""
    spec = _spec(slots=[_what_slot("ODA Grants"), _unbound_how_slot()])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    how_toks = [t for t in plan if t.axis == "how"]
    assert len(how_toks) == 1
    assert how_toks[0].text == ""
    assert sentence_from_plan(plan) == "ODA Grants."


def test_sentence_plan_directional_to_entity_token():
    """A directional recipient token carries its entity reference and role."""
    spec = _spec(entities=[_entity_to("country/ETH", "Ethiopia")])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    idx = next(i for i, t in enumerate(plan) if t.role == "destination")
    text_tok, slot_tok = plan[idx - 1], plan[idx]
    assert text_tok.kind == "text" and text_tok.text == " to "
    assert slot_tok.kind == "slot"
    assert slot_tok.text == "Ethiopia"
    assert slot_tok.axis == "where"
    assert slot_tok.role == "destination"
    assert slot_tok.property_dcid is None
    assert slot_tok.binding_ref == "where:"
    assert slot_tok.entity_refs == ["country/ETH"]


def test_sentence_plan_directional_to_multiple_entities_single_token():
    """Multiple recipients share one slot token with ordered entity references."""
    spec = _spec(
        entities=[
            _entity_to("country/ETH", "Ethiopia"),
            _entity_to("country/KEN", "Kenya"),
        ]
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_toks = [t for t in plan if t.axis == "where"]
    assert len(where_toks) == 1
    idx = next(i for i, t in enumerate(plan) if t.role == "destination")
    text_tok, slot_tok = plan[idx - 1], plan[idx]
    assert text_tok.kind == "text" and text_tok.text == " to "
    assert slot_tok.kind == "slot"
    assert slot_tok.text == "Ethiopia and Kenya"
    assert slot_tok.role == "destination"
    assert slot_tok.entity_refs == ["country/ETH", "country/KEN"]


def test_sentence_plan_directional_to_entity_token_with_backing_where_slot():
    """The recipient token carries both slot metadata and the entity reference."""
    spec = _spec(
        slots=[_where_slot("Ethiopia")],
        entities=[_entity_to("country/ETH", "Ethiopia")],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    idx = next(i for i, t in enumerate(plan) if t.role == "destination")
    text_tok, slot_tok = plan[idx - 1], plan[idx]
    assert text_tok.kind == "text" and text_tok.text == " to "
    assert slot_tok.kind == "slot"
    assert slot_tok.text == "Ethiopia"
    assert slot_tok.axis == "where"
    assert slot_tok.role == "destination"
    assert slot_tok.property_dcid == "DevelopmentFinanceRecipient"
    assert slot_tok.binding_ref == "where:DevelopmentFinanceRecipient"
    assert slot_tok.entity_refs == ["country/ETH"]
    assert sentence_from_plan(plan) == "Development finance flow to Ethiopia."


def test_sentence_plan_directional_from_entity_token():
    """A lone donor leads the heading in possessive form, carrying the origin role, the
    origin ref, and its entity reference."""
    spec = _spec(entities=[_entity_from("country/USA", "the United States")])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    slot_tok, suffix_tok = plan[0], plan[1]
    assert (suffix_tok.kind, suffix_tok.text) == ("text", "'")
    assert slot_tok.text == "The United States"
    assert slot_tok.role == "origin"
    assert slot_tok.axis is None
    assert slot_tok.binding_ref == "origin:"
    assert slot_tok.entity_refs == ["country/USA"]
    assert sentence_from_plan(plan) == "The United States' development finance flow."


def test_sentence_plan_when_slot_empty_placeholder_and_bound():
    """The when token keeps its reference and contains its connective when bound."""
    when_slot = SlotKey(axis="when", property=None, label="time window")
    unbound = Slot(key=when_slot, binding=BindingUnbound())

    spec = _spec(slots=[unbound])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_toks = [t for t in plan if t.axis == "when"]
    assert len(when_toks) == 1
    assert when_toks[0].text == ""
    assert when_toks[0].binding_ref == "when:"

    spec_bound = _spec(slots=[unbound], window=TimeWindow(start_year=2020, end_year=2020))
    plan_bound = render_sentence_plan(spec_bound)
    assert_round_trip(plan_bound, render_sentence(spec_bound))
    when_toks_bound = [t for t in plan_bound if t.axis == "when"]
    assert len(when_toks_bound) == 1
    assert when_toks_bound[0].text == "in 2020"
    assert when_toks_bound[0].binding_ref == "when:"


def _debt_service_spec(open_axis_facts: list[OpenConstraintAxisFact]):
    """One served StatVar under a generic measured property, so its name is the what phrase."""
    return _spec(
        slots=[
            Slot(
                key=_slot_key("how", "sourceIndicatorCode", "Source indicator code", "source"),
                binding=BindingUnbound(),
            )
        ],
        measured_property=_ref("amount", "amount"),
        population_type=_ref("FinancialFlow", "financial flow"),
        stat_vars=[
            StatVar(
                ref=_ref("DebtService", "Debt service on external debt (current US$)"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=["sourceIndicatorCode"],
            )
        ],
        open_axis_facts=open_axis_facts,
    )


def test_sentence_plan_folds_an_agreed_value_the_one_stat_var_name_states():
    """The answer is one StatVar and its name is the what phrase. The value it asserts on
    an open axis is part of that name, so the clause is folded to an empty token, as a
    bound how slot is."""
    spec = _debt_service_spec(
        [
            OpenConstraintAxisFact(
                property="sourceIndicatorCode",
                varies=False,
                value=_ref("DT.TDS.DECT.CD", "DT.TDS.DECT.CD"),
                distinct_count=1,
                undeclared_count=0,
            )
        ]
    )

    plan = render_sentence_plan(spec)

    assert_round_trip(plan, render_sentence(spec))
    how_tok = next(t for t in plan if t.axis == "how")
    assert how_tok.text == ""
    assert how_tok.binding_ref == "how:sourceIndicatorCode"
    assert sentence_from_plan(plan) == "Debt service on external debt (current US$)."


def test_entities_the_answer_serves_no_data_for_add_nothing_to_the_sentence():
    """The spec sentence holds its heading and byline only: entities its read plan serves
    nothing at stay on `ResolvedSpec.unserved_entities`, and the sentence reads as
    though every entity were served."""
    spec = _spec(
        slots=[_what_slot("ODA grants")],
        entities=[
            _entity_to("country/KEN", "Kenya"),
            _entity_to("country/GHA", "Ghana"),
            _entity_to("country/TGO", "Togo"),
        ],
    )
    unserved = spec.model_copy(
        update={
            "resolved": spec.resolved.stating_unserved(
                (("destination", "country/GHA"), ("destination", "country/TGO"))
            )
        }
    )

    plan = render_sentence_plan(unserved)

    assert (
        render_sentence(unserved)
        == render_sentence(spec)
        == "ODA grants to Kenya, Ghana, and Togo."
    )
    assert all(token.kind != "note" for token in plan)
    assert unserved.model_dump() == spec.model_dump()
    assert unserved.resolved.model_copy(update={"headline": None}).unserved_entities == (
        ("destination", "country/GHA"),
        ("destination", "country/TGO"),
    )
