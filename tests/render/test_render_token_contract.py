"""Pins token order, separators, capitalization, and the single terminal period.
Empty fallback text exercises lazy separators on otherwise empty plans."""

from __future__ import annotations

from search.models import (
    BindingUnbound,
    BindingValue,
    SentenceToken,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    StatVar,
    TimeWindow,
)
from search.render.heading import heading_from_parts
from search.render.plan import plain_sentence_plan, render_sentence, render_sentence_plan
from search.render.tokens import _capitalize_first_token, plan_tokens, sentence_from_plan
from tests.render._helpers import (
    _entity_to,
    _gender_slot,
    _how_slot,
    _ref,
    _residence_type_slot,
    _slot_key,
    _spec,
    _what_slot,
    assert_round_trip,
)


def test_how_connective_token_carries_own_spaces():
    """A non-adjectival how-slot's connective is one token, ' for ', spaced on
    both sides -- no separate separator token is needed around it."""
    spec = _spec(slots=[_what_slot("ODA Grants"), _how_slot("health")])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    idx = next(i for i, t in enumerate(plan) if t.axis == "how" and t.text == "health")
    assert plan[idx - 1].kind == "text"
    assert plan[idx - 1].text == " for "


def test_adjectival_fold_gets_lazy_separator_before_what():
    """A bound adjectival how-slot folds before `what`, separated by one lazy space."""
    spec = _spec(
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
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    fold_idx = next(i for i, t in enumerate(plan) if t.property_dcid == "gender")
    what_idx = next(i for i, t in enumerate(plan) if t.axis == "what")
    assert plan[fold_idx].text == "Female"
    assert plan[what_idx].text == "population"
    assert what_idx == fold_idx + 2
    assert plan[fold_idx + 1].kind == "text"
    assert plan[fold_idx + 1].text == " "


def test_empty_adjectival_fold_has_no_separator_either_side():
    """A redundant adjectival fold is empty, so `what` receives no separator."""
    spec = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Person", "Person"),
        slots=[_gender_slot("schemaMale", "Male")],
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
    fold_idx = next(i for i, t in enumerate(plan) if t.property_dcid == "gender")
    what_idx = next(i for i, t in enumerate(plan) if t.axis == "what")
    assert fold_idx == 0
    assert plan[fold_idx].text == ""
    assert what_idx == fold_idx + 1
    assert render_sentence(spec) == "Male population."


def test_two_empty_adjectival_folds_in_a_row_produce_no_separators():
    """Both bound adjectives already appear in `what`.
    Their folds are empty and unseparated. A bound `what` slot keeps the test on the per-member
    redundancy gate rather than the StatVar-name fold."""
    spec = _spec(
        slots=[
            _what_slot("Total urban female population"),
            _gender_slot("schemaFemale", "Female"),
            _residence_type_slot("Urban", "Urban"),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    what_idx = next(i for i, t in enumerate(plan) if t.axis == "what")
    assert what_idx == 2
    assert plan[0].text == ""
    assert plan[1].text == ""
    assert render_sentence(spec) == "Total urban female population."


def test_two_adjectival_folds_one_suppressed_one_not_exact_tokens():
    """`what` already names `Female`, so that fold is empty. `Urban` remains before `what`.
    The separator precedes the first non-empty token, avoiding stray spaces. A bound `what` slot
    keeps the test on the per-member redundancy gate; a StatVar name would drop both how-slots."""
    shape = Shape(
        shape_id="test-shape",
        label="Test Shape",
        population_type=_ref("Person", "Person"),
        measured_property=_ref("count", "count"),
        stat_type=_ref("measuredValue", "Measured Value"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[],
        member_count=1,
    )
    slots = [
        _what_slot("Female population"),
        _gender_slot("schemaFemale", "Female"),
        _residence_type_slot("Urban", "Urban"),
    ]
    plan = plan_tokens(
        heading_from_parts(slots, [], shape, window=None, pinned_properties=frozenset())
    )
    assert_round_trip(plan, sentence_from_plan(plan))
    assert sentence_from_plan(plan) == "Urban Female population."
    assert [t.text for t in plan] == ["", "Urban", " ", "Female population", "."]


def test_window_separator_token_is_space_for_single_year():
    spec = _spec(window=TimeWindow(start_year=2020, end_year=2020))
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    idx = next(i for i, t in enumerate(plan) if t.axis == "when")
    assert plan[idx - 1].kind == "text"
    assert plan[idx - 1].text == " "


def test_window_separator_token_is_comma_for_range():
    spec = _spec(window=TimeWindow(start_year=2015, end_year=2023))
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    idx = next(i for i, t in enumerate(plan) if t.axis == "when")
    assert plan[idx - 1].kind == "text"
    assert plan[idx - 1].text == ", "


def test_source_opens_the_byline_after_the_heading_period():
    """The source token follows the heading's terminal period and a space, with or
    without a window, and every byline token carries the byline segment."""
    source_slot = Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingValue(value=SlotValue(ref=_ref("OECDsrc", "OECD"), value_kind="source")),
    )
    for window in (None, TimeWindow(start_year=2015, end_year=2023)):
        spec = _spec(slots=[source_slot], window=window)
        plan = render_sentence_plan(spec)
        assert_round_trip(plan, render_sentence(spec))
        idx = next(i for i, t in enumerate(plan) if t.axis == "source" and t.text)
        assert [t.text for t in plan[idx - 2 : idx]] == [".", " "]
        assert plan[idx - 2].segment == "heading"
        assert {t.segment for t in plan[idx - 1 :]} == {"byline"}


def test_terminal_period_appears_exactly_once():
    """A plan whose last-rendered value already ends in a period -- a real
    provenance label that names itself with a trailing period ("U.S. Census
    Bureau.") -- closes with exactly one period, not two."""
    source_slot = Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingValue(
            value=SlotValue(ref=_ref("uscb/ACS", "U.S. Census Bureau."), value_kind="source")
        ),
    )
    spec = _spec(
        slots=[_what_slot("ODA grants"), _how_slot("health"), source_slot],
        entities=[_entity_to("country/ETH", "Ethiopia")],
        window=TimeWindow(start_year=2015, end_year=2023),
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    rendered = sentence_from_plan(plan)
    assert rendered == ("ODA grants to Ethiopia for health, 2015 to 2023. U.S. Census Bureau.")
    assert not rendered.endswith("..")
    assert plan[-1].text == "U.S. Census Bureau."


def test_first_token_capitalized_when_it_is_a_fold():
    """The first token in the plan is upper-cased even when it is a folded
    how-slot phrase, not the "what" token."""
    spec = _spec(
        measured_property=_ref("count", "count"),
        population_type=_ref("Population", "population"),
        slots=[_gender_slot("schemaFemale", "female")],
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
    assert plan[0].property_dcid == "gender"
    assert plan[0].text == "Female"


def test_first_token_capitalized_when_it_is_the_what_token():
    """With no pre-what fold, the "what" token itself is the plan's first
    token and is the one upper-cased."""
    spec = _spec(measured_property=_ref("someProperty", "some property"))
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert plan[0].axis == "what"
    assert plan[0].text == "Some property"


def test_first_token_capitalized_skips_leading_punctuation():
    """Capitalisation skips leading punctuation and empty tokens to the first letter."""
    tokens = [
        SentenceToken(kind="slot", text="", axis="what"),
        SentenceToken(kind="slot", text="(relative to total live births)", axis="unit"),
    ]
    _capitalize_first_token(tokens)
    assert sentence_from_plan(tokens) == "(Relative to total live births)"


def test_plain_sentence_plan_round_trips():
    text = "2 possible interpretations."
    plan = plain_sentence_plan(text)
    assert_round_trip(plan, text)
    assert len(plan) == 1
    assert plan[0].kind == "text"
    assert plan[0].text == text


def test_shape_skeleton_all_slots_unbound_renders_what_plus_period():
    """The skeleton has unbound slots and no entities.
    It renders fallback `what` and one period."""
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
    slots = [
        Slot(
            key=_slot_key("how", "DevelopmentFinancePurpose", "Purpose", "purpose"),
            binding=BindingUnbound(),
        ),
        Slot(
            key=_slot_key("where", "DevelopmentFinanceRecipient", "Recipient", "recipient"),
            binding=BindingUnbound(),
        ),
        Slot(
            key=SlotKey(axis="when", property=None, label="time window"), binding=BindingUnbound()
        ),
        Slot(key=SlotKey(axis="source", property=None, label="source"), binding=BindingUnbound()),
    ]
    plan = plan_tokens(
        heading_from_parts(slots, [], shape, window=None, pinned_properties=frozenset())
    )
    assert_round_trip(plan, sentence_from_plan(plan))
    assert sentence_from_plan(plan) == "Development finance flow."
    assert sum(1 for t in plan if t.text == ".") == 1


# ---------------------------------------------------------------------------
# Case 28: connective lazy-separator. A connective token drops its own leading
# whitespace/comma when nothing precedes it in the plan -- reachable when the
# "what" fallback text is itself empty (region/sentence_events.py's pre-materialise SSE
# paths can pass a shape with no label/measured_property/population_type).
# ---------------------------------------------------------------------------


def _empty_what_shape() -> Shape:
    return Shape(
        shape_id="test-shape",
        label="",
        population_type=None,
        measured_property=None,
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[],
        member_count=1,
    )


def test_connective_drops_leading_separator_to_entity_when_what_text_empty():
    """A "to" entity with nothing preceding it in the plan loses the
    connective's leading space, and the capitalized first letter lands on
    "To", not a stray leading space."""
    plan = plan_tokens(
        heading_from_parts(
            [],
            [_entity_to("country/ETH", "Ethiopia")],
            _empty_what_shape(),
            window=None,
            pinned_properties=frozenset(),
        )
    )
    assert sentence_from_plan(plan) == "To Ethiopia."
    assert [t.text for t in plan] == ["", "To ", "Ethiopia", "."]


def test_connective_drops_leading_separator_window_only_when_what_text_empty():
    """A window-only plan (empty what-text, no entities, no declared slots)
    loses the window separator's leading comma, and the capitalized first
    letter lands on the window phrase itself."""
    plan = plan_tokens(
        heading_from_parts(
            [],
            [],
            _empty_what_shape(),
            window=TimeWindow(start_year=2015, end_year=2023),
            pinned_properties=frozenset(),
        )
    )
    assert sentence_from_plan(plan) == "2015 to 2023."
    assert [t.text for t in plan] == ["", "", "2015 to 2023", "."]


def test_a_byline_with_no_heading_text_opens_the_sentence():
    """A source-only plan (empty what-text, no declared slots, one resolved
    provenance) opens on the source, with no leading space."""
    plan = plan_tokens(
        heading_from_parts(
            [],
            [],
            _empty_what_shape(),
            window=None,
            resolved_sources=[_ref("dc/base/WDI", "WDI")],
            pinned_properties=frozenset(),
        )
    )
    assert sentence_from_plan(plan) == "WDI."
    assert [t.text for t in plan] == ["", "WDI", "."]
