"""Checks sentence token order for slots, directional entities, and time windows."""

from __future__ import annotations

import pytest

from search.models import (
    BindingSet,
    BindingUnbound,
    BindingValue,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from tests.render._helpers import (
    _entity_from,
    _entity_to,
    _how_slot,
    _ref,
    _rendered,
    _slot_key,
    _spec,
    _what_slot,
    _where_slot,
)

# ---------------------------------------------------------------------------
# Case 1: measured thing from value-bound `what` slot
# ---------------------------------------------------------------------------


def test_what_slot_bound_uses_value_label():
    spec = _spec(slots=[_what_slot("ODA Grants")])
    assert _rendered(spec) == "ODA Grants."


# ---------------------------------------------------------------------------
# Case 2: no `what` slot -> falls back to shape.measured_property.label
# ---------------------------------------------------------------------------


def test_no_what_slot_falls_back_to_measured_property():
    spec = _spec(measured_property_label="Development Finance Flow")
    assert _rendered(spec) == "Development finance flow."


# ---------------------------------------------------------------------------
# Case 3: `what` slot present but unbound -> falls back to measured_property
# ---------------------------------------------------------------------------


def test_unbound_what_slot_falls_back_to_measured_property():
    slot = Slot(
        key=_slot_key("what", "DevelopmentFinanceScheme", "Scheme", "scheme"),
        binding=BindingUnbound(),
    )
    spec = _spec(slots=[slot], measured_property_label="ODA Flow")
    assert _rendered(spec) == "ODA Flow."


# ---------------------------------------------------------------------------
# Case 4: value-bound `how` slot -> "... for <label>"
# ---------------------------------------------------------------------------


def test_how_slot_adds_for_phrase():
    spec = _spec(slots=[_what_slot("ODA Grants"), _how_slot("health")])
    assert _rendered(spec) == "ODA Grants for health."


# ---------------------------------------------------------------------------
# Case 5: set-bound slot (>=2 members) -> "A and B" join
# ---------------------------------------------------------------------------


def test_set_bound_how_slot_joins_labels():
    slot = Slot(
        key=_slot_key("how", "DevelopmentFinancePurpose", "Purpose", "purpose"),
        binding=BindingSet(
            values=[
                SlotValue(ref=_ref("DAC/Health", "health"), value_kind="entity"),
                SlotValue(ref=_ref("DAC/Education", "education"), value_kind="entity"),
            ]
        ),
    )
    spec = _spec(slots=[slot])
    assert _rendered(spec) == "Development finance flow for health and education."


# ---------------------------------------------------------------------------
# Case 6: directional "to" entity -> "to <label>"
# ---------------------------------------------------------------------------


def test_directional_to_entity_renders_to_phrase():
    spec = _spec(entities=[_entity_to("country/ETH", "Ethiopia")])
    assert _rendered(spec) == "Development finance flow to Ethiopia."


# ---------------------------------------------------------------------------
# Case 7: value-bound `where` slot, no directional entity -> "in <label>"
# ---------------------------------------------------------------------------


def test_where_slot_renders_in_phrase():
    spec = _spec(slots=[_where_slot("Kenya")])
    assert _rendered(spec) == "Kenya's development finance flow."


# ---------------------------------------------------------------------------
# Case 8: directional "to" entity AND bound `where` slot -> "to <entity>" only
# ---------------------------------------------------------------------------


def test_directional_to_takes_precedence_over_where_slot():
    spec = _spec(
        slots=[_where_slot("Kenya")],
        entities=[_entity_to("country/ETH", "Ethiopia")],
    )
    result = _rendered(spec)
    assert result == "Development finance flow to Ethiopia."
    assert " in " not in result


# ---------------------------------------------------------------------------
# Case 9: `where` unbound + directional "to" entity -> "to <label>"
# ---------------------------------------------------------------------------


def test_where_unbound_with_directional_to_entity():
    where_unbound = Slot(
        key=_slot_key("where", "DevelopmentFinanceRecipient", "Recipient", "recipient"),
        binding=BindingUnbound(),
    )
    spec = _spec(
        slots=[where_unbound],
        entities=[_entity_to("country/ETH", "Ethiopia")],
    )
    assert _rendered(spec) == "Development finance flow to Ethiopia."


# ---------------------------------------------------------------------------
# Case 10: directional "from" entity -> "from <label>"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Case 10b: multiple entities on the same direction -> joined labels, one token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recipients", "expected"),
    [
        pytest.param(
            ("country/ETH", "Ethiopia", "country/KEN", "Kenya"),
            "Development finance flow to Ethiopia and Kenya.",
            id="two_recipients",
        ),
        pytest.param(
            (
                "country/ETH",
                "Ethiopia",
                "country/KEN",
                "Kenya",
                "country/UGA",
                "Uganda",
            ),
            "Development finance flow to Ethiopia, Kenya, and Uganda.",
            id="three_recipients",
        ),
    ],
)
def test_directional_to_entities_join_labels(recipients: tuple[str, ...], expected: str):
    spec = _spec(
        entities=[
            _entity_to(ref, label)
            for ref, label in zip(recipients[::2], recipients[1::2], strict=True)
        ]
    )
    assert _rendered(spec) == expected


def test_directional_mixed_from_and_multiple_to():
    """Join donor and recipient groups independently."""
    spec = _spec(
        entities=[
            _entity_to("country/ETH", "Ethiopia"),
            _entity_to("country/KEN", "Kenya"),
            _entity_from("country/USA", "the United States"),
        ]
    )
    result = _rendered(spec)
    assert result == "The United States' development finance flow to Ethiopia and Kenya."


# ---------------------------------------------------------------------------
# Case 11: full-sentence join order
# ---------------------------------------------------------------------------


def test_full_join_order():
    """The heading orders the possessive donor, what, recipient, how, and window."""
    spec = _spec(
        slots=[
            _what_slot("ODA grants"),
            _how_slot("health"),
        ],
        entities=[
            _entity_to("country/ETH", "Ethiopia"),
            _entity_from("country/USA", "the United States"),
        ],
        window=TimeWindow(start_year=2015, end_year=2023),
    )
    expected = "The United States' ODA grants to Ethiopia for health, 2015 to 2023."
    assert _rendered(spec) == expected


# ---------------------------------------------------------------------------
# Case 12: time window rendering variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start_year", "end_year", "expected"),
    [
        pytest.param(
            2015,
            2023,
            "Development finance flow, 2015 to 2023.",
            id="closed-range",
        ),
        pytest.param(2020, 2020, "Development finance flow in 2020.", id="same-year"),
        pytest.param(2015, None, "Development finance flow, since 2015.", id="start-only"),
        pytest.param(None, 2020, "Development finance flow, until 2020.", id="end-only"),
    ],
)
def test_window_rendering_variants(start_year: int | None, end_year: int | None, expected: str):
    """Each time-window shape has distinct wording."""
    spec = _spec(window=TimeWindow(start_year=start_year, end_year=end_year))
    assert _rendered(spec) == expected


# ---------------------------------------------------------------------------
# Case 13: value-bound `source` slot -> the byline names it
# ---------------------------------------------------------------------------


def test_source_slot_names_the_source_in_the_byline():
    source_slot = Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingValue(value=SlotValue(ref=_ref("OECDsrc", "OECD"), value_kind="source")),
    )
    spec = _spec(slots=[source_slot])
    assert _rendered(spec) == "Development finance flow. OECD."
