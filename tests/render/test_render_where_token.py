"""Pins the resolved `where` fallback, its coverage counts, and its editable slot backing."""

from __future__ import annotations

from search.models import (
    BindingUnbound,
    BreadthDim,
    CoverageBreadth,
    CoverageExact,
    ResolvedScope,
    Slot,
    TimeWindow,
)
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import (
    _entity_to,
    _rendered,
    _slot_key,
    _spec,
    _where_slot,
    assert_round_trip,
)


def _unbound_where_slot() -> Slot:
    return Slot(
        key=_slot_key("where", "DevelopmentFinanceRecipient", "Recipient", "recipient"),
        binding=BindingUnbound(),
    )


def _countries_coverage(count: int) -> CoverageExact:
    return CoverageExact(
        has_data=True,
        observation_count=count,
        dimensions=[BreadthDim(label="countries", count=count)],
    )


def _recipients_coverage(count: int) -> CoverageBreadth:
    """A role-bearing answer resolved over a scope the engine enumerated: breadth, one
    "recipients" dimension, no observation_count."""
    return CoverageBreadth(has_data=True, dimensions=[BreadthDim(label="recipients", count=count)])


def test_where_token_resolves_to_default_scope_country_count():
    """A `CoverageExact` countries dimension renders the default scope as `across N countries`."""
    spec = _spec(slots=[_unbound_where_slot()], coverage=_countries_coverage(37))
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.text == "across 37 countries"
    assert where_tok.binding_ref == "where:DevelopmentFinanceRecipient"
    assert sentence_from_plan(plan) == "Development finance flow across 37 countries."


def test_where_token_resolves_to_default_scope_country_count_of_one_drops_the_phrase():
    """A bare count of 1 has no country name to render, so the fallback stays empty."""
    spec = _spec(slots=[_unbound_where_slot()], coverage=_countries_coverage(1))
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.text == ""
    assert sentence_from_plan(plan) == "Development finance flow."


def test_where_token_resolves_to_enumerated_recipient_count():
    """A `recipients` breadth dimension reads the open destination role as a count, a
    role-only token behind the role's preposition."""
    spec = _spec(slots=[_unbound_where_slot()], coverage=_recipients_coverage(140))
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.text == "140 recipients"
    assert where_tok.role == "destination"
    assert where_tok.entity_refs == []
    assert sentence_from_plan(plan) == "Development finance flow to 140 recipients."


def test_where_token_resolves_to_enumerated_recipient_count_of_one_drops_the_phrase():
    """A count of 1 has no recipient name to render, so the where token stays empty."""
    spec = _spec(slots=[_unbound_where_slot()], coverage=_recipients_coverage(1))
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.text == ""
    assert sentence_from_plan(plan) == "Development finance flow."


def test_where_token_stays_empty_when_breadth_coverage_has_no_recipients_dimension():
    """A `sources` dimension does not populate the `where` token."""
    coverage = CoverageBreadth(has_data=True, dimensions=[BreadthDim(label="sources", count=2)])
    spec = _spec(slots=[_unbound_where_slot()], coverage=coverage)
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.text == ""


def test_where_token_keeps_slot_backing_on_resolved_fallback():
    """The resolved fallback remains clickable because `where` has editable options."""
    where_slot = _unbound_where_slot()
    spec = _spec(slots=[where_slot], coverage=_countries_coverage(12))
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.property_dcid == "DevelopmentFinanceRecipient"
    assert where_tok.binding_ref == where_slot.key.as_ref()


def test_bound_where_slot_wins_over_resolved_fallback():
    """A bound where slot takes precedence over the resolved-coverage fallback."""
    spec = _spec(slots=[_where_slot("Africa")], coverage=_countries_coverage(54))
    assert _rendered(spec) == "Africa's development finance flow."


def test_where_token_stays_empty_with_bare_coverage():
    """CoverageBare (candidates specs, no-entity variable disambiguation) has
    no dimensions at all. The where token stays the empty placeholder."""
    spec = _spec(slots=[_unbound_where_slot()])  # default coverage is CoverageBare
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.text == ""


def test_where_token_stays_empty_when_coverage_has_no_countries_dimension():
    """A CoverageExact with dimensions, but none labeled "countries" (e.g. a
    single-entity "sources"/"observations" breadth pair). The where token
    stays the empty placeholder, avoiding a guess at an unrelated
    dimension."""
    coverage = CoverageExact(
        has_data=True,
        observation_count=5,
        dimensions=[
            BreadthDim(label="sources", count=2),
            BreadthDim(label="observations", count=5),
        ],
    )
    spec = _spec(slots=[_unbound_where_slot()], coverage=coverage)
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.text == ""


def test_directional_to_wins_over_where_resolved_fallback():
    """A directional `to` entity takes precedence over resolved coverage."""
    spec = _spec(
        slots=[_unbound_where_slot()],
        entities=[_entity_to("country/ETH", "Ethiopia")],
        coverage=_countries_coverage(20),
    )
    result = _rendered(spec)
    assert result == "Development finance flow to Ethiopia."
    assert "countries" not in result


def test_countries_breadth_with_a_window_names_the_scope_and_the_window_apart():
    """Incomplete facet dates make windowed coverage broad and withhold observation counts.
    The country count comes from entity-level data checks; the window renders separately."""
    coverage = CoverageBreadth(
        has_data=True,
        dimensions=[
            BreadthDim(label="countries", count=2),
            BreadthDim(label="observations", count=9),
        ],
        window=TimeWindow(start_year=2012, end_year=2018),
    )
    spec = _spec(slots=[_unbound_where_slot()], coverage=coverage)
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.text == "across 2 countries"


# Resolved-where tokens carry the open role axis for client joins.


def test_where_token_role_joins_open_destination_scope():
    """An open destination scope reads as the destination role on the where slot."""
    scope = ResolvedScope(axis="destination", members=[])
    spec = _spec(slots=[_unbound_where_slot()], coverage=_recipients_coverage(140), scope=scope)
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.role == "destination"
    assert where_tok.text == "140 recipients"
    assert where_tok.binding_ref == "where:DevelopmentFinanceRecipient"
    assert sentence_from_plan(plan) == "Development finance flow to 140 recipients."


def test_where_token_role_joins_open_origin_scope():
    """An open origin role reads as its own role-only token with the origin ref; the
    where slot keeps its empty token."""
    scope = ResolvedScope(axis="origin", members=[])
    coverage = CoverageBreadth(has_data=True, dimensions=[BreadthDim(label="providers", count=54)])
    spec = _spec(slots=[_unbound_where_slot()], coverage=coverage, scope=scope)
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    origin_tok = next(t for t in plan if t.role == "origin")
    assert (origin_tok.text, origin_tok.axis, origin_tok.binding_ref) == (
        "54 providers",
        None,
        "origin:",
    )
    assert next(t for t in plan if t.axis == "where").text == ""
    assert sentence_from_plan(plan) == "Development finance flow from 54 providers."


def test_where_token_role_stays_none_for_observationabout_scope():
    """`observationAbout` is not a role axis, so the scope adds no role to the token."""
    scope = ResolvedScope(axis="observationAbout", members=[])
    spec = _spec(slots=[_unbound_where_slot()], coverage=_countries_coverage(37), scope=scope)
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.role is None
    assert where_tok.text == "across 37 countries"
    assert sentence_from_plan(plan) == "Development finance flow across 37 countries."


def test_where_token_role_stays_none_when_scope_is_none():
    """resolved.scope is None on every captured response where a place was
    named (or where the engine never defaulted an entity axis at all): the
    stamp is then a no-op, leaving the token's role unstamped."""
    spec = _spec(slots=[_unbound_where_slot()], coverage=_countries_coverage(12))
    plan = render_sentence_plan(spec)
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.role is None
    assert where_tok.text == "across 12 countries"
    assert sentence_from_plan(plan) == "Development finance flow across 12 countries."


def test_bound_where_slot_byte_for_byte_unchanged_with_scope_present():
    """A bound where token keeps its text and role when a resolved scope is present."""
    scope = ResolvedScope(axis="destination", members=[])
    spec = _spec(slots=[_where_slot("Africa")], coverage=_countries_coverage(54), scope=scope)
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    where_tok = next(t for t in plan if t.axis == "where")
    assert where_tok.role is None
    assert where_tok.text == "Africa"
    assert sentence_from_plan(plan) == "Africa's development finance flow."
