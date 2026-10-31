"""Tests the code index's IDS and WEO measure nouns, on five-tuples and constraints from the dev graph."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from search.heading.measure_noun import MeasureNoun, denominator_noun, measure_noun

_GDP_BASIS = {
    "fixedCapitalConsumptionBasis": "GrossOfFixedCapitalConsumption",
    "productTaxBasis": "IncludingProductTaxes",
}


@pytest.mark.parametrize(
    ("five_tuple", "constraints", "expected"),
    [
        pytest.param(
            ("FinancialFlow", "amount", "measuredValue"),
            {"financialFlowType": "ServicingPayment"},
            MeasureNoun(
                noun="external debt service",
                axes=("populationType", "measuredProperty", "statType", "financialFlowType"),
            ),
            id="ids-002-debt-service",
        ),
        pytest.param(
            ("FinancialFlow", "amount", "measuredValue"),
            {"financialFlowType": "ServicingPayment", "idsCreditorType": "IDSCreditorTypeBLAT"},
            MeasureNoun(
                noun="external debt service",
                axes=("populationType", "measuredProperty", "statType", "financialFlowType"),
            ),
            id="creditor-type-left-to-its-how-slot",
        ),
        pytest.param(
            ("FinancialPosition", "amount", "measuredValue"),
            {"financialPositionType": "AmountOutstanding"},
            MeasureNoun(
                noun="outstanding external debt",
                axes=("populationType", "measuredProperty", "statType", "financialPositionType"),
            ),
            id="ids-006-outstanding",
        ),
        pytest.param(
            ("EconomicProduction", "valueAdded", "measuredValue"),
            {"currencyConversionBasis": "PurchasingPowerParityRate", **_GDP_BASIS},
            MeasureNoun(
                noun="GDP",
                axes=(
                    "populationType",
                    "measuredProperty",
                    "statType",
                    "currencyConversionBasis",
                    "fixedCapitalConsumptionBasis",
                    "productTaxBasis",
                ),
            ),
            id="weo-005-gdp-ppp",
        ),
        pytest.param(
            ("EconomicActivity", "amount", "growthRate"),
            {"activitySource": "GrossDomesticProduction"},
            MeasureNoun(
                noun="GDP growth",
                axes=("populationType", "measuredProperty", "statType", "activitySource"),
            ),
            id="gdp-growth",
        ),
        pytest.param(
            ("EconomicUnit", "primaryIncome", "measuredValue"),
            {"fixedCapitalConsumptionBasis": "GrossOfFixedCapitalConsumption"},
            MeasureNoun(
                noun="gross national income",
                axes=(
                    "populationType",
                    "measuredProperty",
                    "statType",
                    "fixedCapitalConsumptionBasis",
                ),
            ),
            id="weo-gni",
        ),
    ],
)
def test_measure_noun_for_gallery_measures(
    five_tuple: tuple[str, str, str], constraints: Mapping[str, str], expected: MeasureNoun
) -> None:
    population_type, measured_property, stat_type = five_tuple
    assert (
        measure_noun(
            population_type=population_type,
            measured_property=measured_property,
            stat_type=stat_type,
            constraints=constraints,
        )
        == expected
    )


def test_ids_037_share_of_exports_names_its_base_through_the_denominator() -> None:
    # Amount_FinancialFlow_ServicingPayment_AsAFractionOf_Amount_EconomicActivity_
    # ExportsOfGoodsServicesAndPrimaryIncome_Nominal.
    noun = measure_noun(
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        constraints={"financialFlowType": "ServicingPayment"},
    )
    assert noun is not None
    assert noun.noun == "external debt service"
    assert (
        denominator_noun("Amount_EconomicActivity_ExportsOfGoodsServicesAndPrimaryIncome_Nominal")
        == "exports of goods, services and primary income"
    )


def test_gni_as_primary_income_names_gni() -> None:
    assert (
        denominator_noun("PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal")
        == "GNI"
    )


def test_a_measure_outside_the_table_has_no_noun() -> None:
    assert (
        measure_noun(
            population_type="FinancialFlow",
            measured_property="amount",
            stat_type="measuredValue",
            constraints={"financialFlowType": "Disbursement", "maturityBand": "LongTermMaturity"},
        )
        is None
    )


def test_per_head_denominators_have_no_noun() -> None:
    assert denominator_noun("Count_Person") is None
    assert denominator_noun(None) is None
