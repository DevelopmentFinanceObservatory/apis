"""Pins validation of `requested_all` claims before default or warning suppression, and
the claims an operation over a category makes without bind."""

from __future__ import annotations

import pytest

from search.engine.bind import SlotBindingDraft, validate_requested_all_bindings
from search.engine.region.phase_rank_bind import _open_operation_slots
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import Operation

_PHRASE = "across all flows"
# The elected shape's constraint slots: both CRS axes the tests bind.
_SLOTS = (
    SlotKeyDraft(axis="how", property_dcid="crsTypeOfFlow", label="flow type"),
    SlotKeyDraft(axis="how", property_dcid="crsTypeOfFinance", label="finance type"),
)


def _unbound(property_dcid: str = "crsTypeOfFlow", **kwargs) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid=property_dcid, kind="unbound", value_dcids=[], **kwargs
    )


def test_evidence_not_requested_all_passes_through_untouched_even_with_a_phrase_set():
    """Bindings without `evidence="requested_all"` pass through unchanged."""
    mentioned_unbound = _unbound(evidence="mentioned", requested_all_phrase=_PHRASE)
    value_binding = SlotBindingDraft(
        axis="how",
        property_dcid="crsTypeOfFlow",
        kind="value",
        value_dcids=["CRSTypeOfFlow10"],
        evidence="mentioned",
        requested_all_phrase=_PHRASE,
    )

    validated, applied = validate_requested_all_bindings(
        [mentioned_unbound, value_binding], [_PHRASE], slot_keys=_SLOTS
    )

    assert validated == [mentioned_unbound, value_binding]
    assert applied == frozenset()


def test_requested_all_with_no_phrase_is_demoted():
    """A missing phrase demotes a `requested_all` claim to `mentioned`."""
    binding = _unbound(evidence="requested_all", requested_all_phrase=None)

    validated, applied = validate_requested_all_bindings([binding], [_PHRASE], slot_keys=_SLOTS)

    assert len(validated) == 1
    assert validated[0].evidence == "mentioned"
    assert validated[0].requested_all_phrase is None
    assert validated[0].kind == "unbound"
    assert applied == frozenset()


def test_requested_all_with_a_non_verbatim_phrase_is_demoted():
    """A phrase absent from the query's dropped qualifiers is demoted."""
    binding = _unbound(evidence="requested_all", requested_all_phrase="a phrase never offered")

    validated, applied = validate_requested_all_bindings([binding], [_PHRASE], slot_keys=_SLOTS)

    assert len(validated) == 1
    assert validated[0].evidence == "mentioned"
    assert validated[0].requested_all_phrase is None
    assert applied == frozenset()


def test_requested_all_on_a_value_or_set_binding_is_demoted():
    """Only an unbound slot can carry a valid `requested_all` claim."""
    binding = SlotBindingDraft(
        axis="how",
        property_dcid="crsTypeOfFlow",
        kind="value",
        value_dcids=["CRSTypeOfFlow10"],
        evidence="requested_all",
        requested_all_phrase=_PHRASE,
    )

    validated, applied = validate_requested_all_bindings([binding], [_PHRASE], slot_keys=_SLOTS)

    assert validated[0].evidence == "mentioned"
    assert validated[0].requested_all_phrase is None
    assert applied == frozenset()


def test_requested_all_with_a_verbatim_phrase_validates():
    """A verbatim phrase on an unbound slot validates and is reported applied."""
    binding = _unbound(evidence="requested_all", requested_all_phrase=_PHRASE)

    validated, applied = validate_requested_all_bindings([binding], [_PHRASE], slot_keys=_SLOTS)

    assert validated == [binding]
    assert applied == frozenset({_PHRASE})


def test_a_mixed_batch_only_demotes_the_invalid_claim():
    """Each binding is validated against its own phrase."""
    valid = _unbound(
        property_dcid="crsTypeOfFlow", evidence="requested_all", requested_all_phrase=_PHRASE
    )
    invalid = _unbound(
        property_dcid="crsTypeOfFinance",
        evidence="requested_all",
        requested_all_phrase="not offered",
    )

    validated, applied = validate_requested_all_bindings(
        [valid, invalid], [_PHRASE], slot_keys=_SLOTS
    )

    assert validated[0] == valid
    assert validated[1].evidence == "mentioned"
    assert validated[1].requested_all_phrase is None
    assert applied == frozenset({_PHRASE})


@pytest.mark.parametrize(
    ("property_dcid", "phrase"),
    [
        # "total" asks for a sum and names no slot.
        ("crsTypeOfFlow", "total"),
        # The phrase names the flow slot, and the claim is on the instrument slot.
        ("crsTypeOfFinance", _PHRASE),
    ],
)
def test_requested_all_whose_phrase_names_another_slot_or_none_is_demoted(
    property_dcid: str, phrase: str
):
    """A verbatim phrase that does not name the claimed slot by its declared plain name is
    demoted, so a declared default still binds the slot."""
    binding = _unbound(property_dcid, evidence="requested_all", requested_all_phrase=phrase)

    validated, applied = validate_requested_all_bindings([binding], [phrase], slot_keys=_SLOTS)

    assert validated[0].evidence == "mentioned"
    assert validated[0].requested_all_phrase is None
    assert applied == frozenset()


# Slots on properties no code system declares, labelled as the graph labels them; the index
# declares each a slot plain name.
_UNDECLARED_SLOTS = (
    SlotKeyDraft(axis="how", property_dcid="dacTypeOfFinance", label="DAC type of finance"),
    SlotKeyDraft(
        axis="how", property_dcid="currencyOfDenomination", label="currency of denomination"
    ),
)


@pytest.mark.parametrize(
    ("property_dcid", "phrase", "stands"),
    [
        # "net bilateral ODA to Kenya": "bilateral" does not name the slot, so the phrase
        # stays a dropped qualifier no answer binds.
        ("dacTypeOfFinance", "bilateral", False),
        ("dacTypeOfFinance", "of", False),
        # A word of the graph label names no slot: "DAC" names donors, "type" any type.
        ("dacTypeOfFinance", "all DAC donors", False),
        ("dacTypeOfFinance", "across all aid types", False),
        ("dacTypeOfFinance", "across all types of finance", True),
        ("currencyOfDenomination", "held in different currencies", True),
    ],
)
def test_requested_all_on_a_slot_outside_the_code_systems_must_name_its_plain_name(
    property_dcid: str, phrase: str, stands: bool
):
    """A claim on a property no code system declares stands only when its phrase names the
    slot's declared plain name, singular or plural."""
    binding = _unbound(property_dcid, evidence="requested_all", requested_all_phrase=phrase)

    validated, applied = validate_requested_all_bindings(
        [binding], [phrase], slot_keys=_UNDECLARED_SLOTS
    )

    assert (validated[0].evidence == "requested_all") is stands
    assert applied == (frozenset({phrase}) if stands else frozenset())


def test_requested_all_on_a_property_the_elected_shape_lacks_is_demoted():
    """A claim for a property absent from the elected shape is demoted."""
    binding = _unbound(
        property_dcid="notARealSlotOnThisShape",
        evidence="requested_all",
        requested_all_phrase=_PHRASE,
    )

    validated, applied = validate_requested_all_bindings([binding], [_PHRASE], slot_keys=_SLOTS)

    assert validated[0].evidence == "mentioned"
    assert validated[0].requested_all_phrase is None
    assert applied == frozenset()


# ---------------------------------------------------------------------------
# An operation over a category opens the slot its phrase names
# ---------------------------------------------------------------------------

_CRS_SHAPE = ShapeDraft(
    shape_id="crs",
    label="CRS",
    pop_type_dcid="",
    meas_prop_dcid="",
    stat_type_dcid="",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(
        SlotKeyDraft(axis="how", property_dcid="crsTypeOfFlow", label="type of flow"),
        SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),
        SlotKeyDraft(axis="what", property_dcid="crsCooperationModality", label="modality"),
        SlotKeyDraft(axis="when", property_dcid=None, label="year"),
    ),
)


def _rank(phrase: str, over: str | None = "category") -> Operation:
    return Operation(verb="rank", phrase=phrase, over=over)


def _opened(phrase: str) -> SlotBindingDraft:
    return _unbound(evidence="requested_all", requested_all_phrase=phrase)


def test_a_category_operation_opens_the_open_slot_its_phrase_names():
    """An unbound or absent slot the phrase names becomes a requested_all claim."""
    flow = _unbound()
    absent_flow = SlotBindingDraft(axis="how", property_dcid="crsTypeOfFlow", kind="absent")

    for binding in (flow, absent_flow):
        assert _open_operation_slots([binding], [_rank("top 5 flow types")], _CRS_SHAPE) == [
            _opened("top 5 flow types")
        ]


def test_a_category_operation_adds_a_claim_for_a_named_slot_bind_omitted():
    """A named slot bind left out of its bindings is added, open."""
    purpose = _unbound("crsPurpose")

    opened = _open_operation_slots([purpose], [_rank("the three largest modalities")], _CRS_SHAPE)

    assert opened == [
        purpose,
        SlotBindingDraft(
            axis="what",
            property_dcid="crsCooperationModality",
            kind="unbound",
            evidence="requested_all",
            requested_all_phrase="the three largest modalities",
        ),
    ]


def test_a_slot_bind_gave_a_value_stays_bound():
    """A value bind chose for the named slot is kept."""
    oda = SlotBindingDraft(
        axis="how", property_dcid="crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"]
    )

    assert _open_operation_slots([oda], [_rank("top 5 flows")], _CRS_SHAPE) == [oda]


def test_an_operation_naming_no_slot_or_over_another_axis_opens_nothing():
    """A category phrase naming no declared slot, or a rank over origin, changes nothing."""
    bindings = [_unbound(), _unbound("crsPurpose")]

    for operation in (
        _rank("Which got the most"),
        _rank("top 5 flow providers", over="origin"),
        _rank("flowers"),
    ):
        assert _open_operation_slots(bindings, [operation], _CRS_SHAPE) == bindings


def test_a_phrase_naming_two_slots_opens_neither():
    """ "top 5 sectors by aid flows" names the sector and the flow slot, so which one the
    rank runs over is unsaid and both stay as bind left them."""
    bindings = [_unbound(), _unbound("crsPurpose")]

    assert _open_operation_slots(bindings, [_rank("top 5 sectors by aid flows")], _CRS_SHAPE) == (
        bindings
    )
