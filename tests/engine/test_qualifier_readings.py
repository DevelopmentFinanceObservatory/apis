"""Tests choosing a two-reading span's code or measure reading by the shape it is read against."""

from __future__ import annotations

import pytest

from search.engine.qualifier_match import QualifierMatch, QualifierReadings
from search.engine.region.qualifier_readings import match_query_readings, reading_for_shape
from search.engine.shape import ShapeDraft, SlotKeyDraft

_PURPOSE_SLOT = SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose")

_PPP = QualifierReadings(
    matched_text="PPP",
    code=QualifierMatch(
        code_system="crsPurpose",
        code_dcid="CRSPurposeCode25010",
        label="business policy and administration",
        source="synonym",
        matched_text="PPP",
    ),
    measure=QualifierMatch(
        code_system="measurementQualifier",
        code_dcid="PurchasingPowerParity",
        label="purchasing power parity",
        source="synonym",
        matched_text="PPP",
    ),
)


def _shape(
    *, pop_type: str, meas_qual: str, slots: tuple[SlotKeyDraft, ...] = (), roles=None
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=f"{pop_type}_{meas_qual}",
        label=pop_type,
        pop_type_dcid=pop_type,
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=meas_qual,
        meas_denom_dcid=None,
        slot_keys=slots,
        declared_roles=roles,
    )


_CRS_FLOW = _shape(
    pop_type="FinancialFlow",
    meas_qual="Nominal",
    slots=(_PURPOSE_SLOT,),
    roles=("financingProvider", "financingRecipient"),
)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param(
            _shape(pop_type="EconomicActivity", meas_qual="PurchasingPowerParity"),
            _PPP.measure,
            id="gdp-ppp-reads-the-price-basis",
        ),
        pytest.param(_CRS_FLOW, _PPP.code, id="crs-flow-reads-purpose-25010"),
        pytest.param(_shape(pop_type="EconomicActivity", meas_qual="Nominal"), None, id="neither"),
        pytest.param(
            _shape(
                pop_type="EconomicActivity",
                meas_qual="PurchasingPowerParity",
                slots=(_PURPOSE_SLOT,),
            ),
            None,
            id="both",
        ),
    ],
)
def test_a_shape_keeps_the_one_reading_it_carries(shape: ShapeDraft, expected):
    assert reading_for_shape(_PPP, shape) == expected


def test_query_readings_are_owned_only_from_the_variable_and_query_text():
    readings, owned = match_query_readings("aid to Kenya", ["PPP"], query_text="aid to Kenya PPP")
    assert [r.measure.code_dcid for r in readings] == ["PurchasingPowerParity"]
    assert owned == {"ppp"}

    readings, owned = match_query_readings("aid to Kenya", ["PPP"])
    assert [r.code.code_dcid for r in readings] == ["CRSPurposeCode25010"]
    assert owned == frozenset()
