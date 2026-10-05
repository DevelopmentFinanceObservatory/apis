"""Pins validation of `sole_option` claims against the option sets the index supplies."""

from __future__ import annotations

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.region.phase_rank_bind import _validate_sole_option_bindings
from search.engine.shape import ShapeDraft, SlotKeyDraft

_FLOW = "crsTypeOfFlow"
_FOUR_FLOWS = ["CRSTypeOfFlow10", "CRSTypeOfFlow21", "CRSTypeOfFlow30", "CRSTypeOfFlow60"]


def _shape(options: list[str] | None, *, index_sourced: bool) -> ShapeDraft:
    return ShapeDraft(
        shape_id="financialflow_amount_measuredvalue_nominal",
        label="net disbursements",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid=_FLOW, label="CRS type of flow"),),
        slot_taxonomy={} if options is None else {f"how:{_FLOW}": options},
        slot_taxonomy_index_sourced=index_sourced,
    )


def _flow(value: str, evidence: str) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid=_FLOW, kind="value", value_dcids=[value], evidence=evidence
    )


def test_a_sole_option_claim_on_an_index_slot_offering_several_values_leaves_it_unbound():
    """ "Net disbursements" names no flow. The recalled members all hold one flow and the
    index's family holds four, so the slot offers four."""
    (validated,) = _validate_sole_option_bindings(
        [_flow("CRSTypeOfFlow21", "sole_option")], _shape(_FOUR_FLOWS, index_sourced=True)
    )

    assert (validated.kind, validated.value_dcids) == ("unbound", [])
    assert validated.property_dcid == _FLOW


@pytest.mark.parametrize(
    ("evidence", "options", "index_sourced"),
    [
        # The slot offers one value.
        ("sole_option", ["CRSTypeOfFlow21"], True),
        # No option set to check the claim against.
        ("sole_option", None, True),
        # The phrase named the value.
        ("mentioned", _FOUR_FLOWS, True),
        # The option set is a recalled sample, which the claim is not checked against.
        ("sole_option", _FOUR_FLOWS, False),
    ],
)
def test_a_binding_its_slot_or_the_phrase_supports_stays(evidence, options, index_sourced):
    binding = _flow("CRSTypeOfFlow21", evidence)

    validated = _validate_sole_option_bindings(
        [binding], _shape(options, index_sourced=index_sourced)
    )

    assert validated == [binding]
