"""Tests the flow family read from a shape's members: the account a code-index row's
datasets declare, and the constraint values of a series outside the index."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from code_index.flow_family import flow_family
from code_index.shared import get_index

# Code-index rows of each dataset declaring the provider's account.
_CRS_GROSS_ODA_ROW = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_CRS_COMMITMENT_ROW = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_IMPUTED_ROW = (
    "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
    "CRSPurposeCode100_Nominal"
)
_DAC1_NET_DISBURSEMENT_ROW = "Amount_FinancialFlow_Net_Disbursement_DAC1AidType8863_Nominal"
_DAC1_COMMITMENT_ROW = "AmountMobilised_FinancialObligation_Commitment_DAC1AidType2233_Nominal"
_DAC2A_NET_DISBURSEMENT_ROW = "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal"

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
    "dcid",
    [
        _CRS_GROSS_ODA_ROW,
        _CRS_COMMITMENT_ROW,
        _IMPUTED_ROW,
        _DAC1_NET_DISBURSEMENT_ROW,
        _DAC1_COMMITMENT_ROW,
        _DAC2A_NET_DISBURSEMENT_ROW,
    ],
)
def test_an_index_row_reads_in_its_datasets_account(dcid: str) -> None:
    """A row of a provider's-account table reads as aid, whatever its constraint values:
    DAC1 and DAC2a disbursements carry `financialFlowType = Disbursement` and no
    `crsTypeOfFlow`, as an IDS disbursement does."""
    row = next(row for row in get_index().statvars if row.dcid == dcid)
    family = flow_family(
        index=get_index(),
        population_type=row.population_type,
        stat_vars=[row.dcid],
        members=[row.constraint_properties],
    )
    assert family == "aid"


@pytest.mark.parametrize(
    ("population_type", "members", "expected"),
    [
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
def test_a_series_outside_the_index_reads_its_constraint_values(
    population_type: str, members: list[Mapping[str, str]], expected: str | None
) -> None:
    stat_vars = ["Amount_FinancialFlow_ServicingPayment_Nominal"] * len(members)
    family = flow_family(
        index=get_index(), population_type=population_type, stat_vars=stat_vars, members=members
    )
    assert family == expected


def test_an_index_row_beside_a_series_outside_the_index_has_none() -> None:
    family = flow_family(
        index=get_index(),
        population_type="FinancialFlow",
        stat_vars=[_DAC1_NET_DISBURSEMENT_ROW, "Amount_FinancialFlow_ServicingPayment_Nominal"],
        members=[_IDS_DEBT_SERVICE],
    )
    assert family is None


def test_members_in_different_families_have_none() -> None:
    members = [_IDS_DEBT_SERVICE, _IDS_DISBURSEMENT]
    assert (
        flow_family(
            index=get_index(), population_type="FinancialFlow", stat_vars=[], members=members
        )
        is None
    )


def test_a_member_outside_every_family_leaves_the_shape_without_one() -> None:
    members = [_IDS_DEBT_SERVICE, {}]
    assert (
        flow_family(
            index=get_index(), population_type="FinancialFlow", stat_vars=[], members=members
        )
        is None
    )


def test_no_members_have_no_family() -> None:
    assert (
        flow_family(index=get_index(), population_type="FinancialFlow", stat_vars=[], members=[])
        is None
    )
