"""Tests the flow family read from a shape's members, on constraint values from the dev graph."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from search.heading.flow_family import flow_family

_CRS_GROSS_ODA = {
    "crsTypeOfFlow": "CRSTypeOfFlow10",
    "financialFlowType": "Disbursement",
    "nettingBasis": "Gross",
}
_CRS_COMMITMENT = {"crsTypeOfFlow": "CRSTypeOfFlow10", "financialObligationType": "Commitment"}
_MLS_IMPUTED = {**_CRS_GROSS_ODA, "dac2aAidType": "DAC2aAidType106"}
# Amount_FinancialFlow_ServicingPayment_Nominal and its ByCreditor twin.
_IDS_DEBT_SERVICE = {"financialFlowType": "ServicingPayment"}
_IDS_DEBT_SERVICE_BLAT = {**_IDS_DEBT_SERVICE, "idsCreditorType": "IDSCreditorTypeBLAT"}
_IDS_INTEREST = {"financialFlowType": "InterestPayment"}
_IDS_DISBURSEMENT = {
    "financialFlowType": "Disbursement",
    "includesIMFCredit": "WithIMFCredit",
    "maturityBand": "LongTermMaturity",
}
_IDS_OUTSTANDING = {"financialPositionType": "AmountOutstanding"}
_IDS_ARREARS = {"financialPositionType": "PrincipalArrears"}
_IDS_UNDISBURSED = {"financialPositionType": "AmountUndisbursed"}
_WEO_GDP_PPP = {
    "currencyConversionBasis": "PurchasingPowerParityRate",
    "fixedCapitalConsumptionBasis": "GrossOfFixedCapitalConsumption",
    "productTaxBasis": "IncludingProductTaxes",
}


@pytest.mark.parametrize(
    ("population_type", "members", "expected"),
    [
        ("FinancialFlow", [_CRS_GROSS_ODA], "aid"),
        ("FinancialObligation", [_CRS_COMMITMENT], "aid"),
        ("FinancialFlow", [_MLS_IMPUTED], "aid"),
        ("FinancialFlow", [_IDS_DEBT_SERVICE, _IDS_DEBT_SERVICE], "debt_service"),
        (
            "FinancialFlow",
            [_IDS_DEBT_SERVICE, _IDS_DEBT_SERVICE_BLAT, _IDS_INTEREST],
            "debt_service",
        ),
        ("FinancialFlow", [_IDS_DISBURSEMENT], "debt_lending"),
        ("FinancialPosition", [_IDS_OUTSTANDING], "debt_stock"),
        ("FinancialPosition", [_IDS_OUTSTANDING, _IDS_ARREARS], "debt_stock"),
        ("FinancialPosition", [_IDS_UNDISBURSED], "debt_lending"),
        ("EconomicProduction", [_WEO_GDP_PPP], None),
    ],
)
def test_flow_family_reads_constraint_values(
    population_type: str, members: list[Mapping[str, str]], expected: str | None
) -> None:
    assert flow_family(population_type=population_type, members=members) == expected


def test_members_in_different_families_have_none() -> None:
    members = [_IDS_DEBT_SERVICE, _IDS_DISBURSEMENT]
    assert flow_family(population_type="FinancialFlow", members=members) is None


def test_a_member_outside_every_family_leaves_the_shape_without_one() -> None:
    members = [_IDS_DEBT_SERVICE, {}]
    assert flow_family(population_type="FinancialFlow", members=members) is None


def test_no_members_have_no_family() -> None:
    assert flow_family(population_type="FinancialFlow", members=[]) is None
