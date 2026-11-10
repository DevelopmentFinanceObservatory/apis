"""The verdict on each declared measure a request names, on what a reading states of itself.

Each row names the variable text, the reading's evidence and the verdict on each name
`qualifier_match.declared_measures_named` reads in the text, in text order. The pairs copy
the fixture code index's declared measures (`MeasureIndex.nouns`, `partition_labels`) and the
StatVars that answered each case in the recorded panel or a dev probe.
"""

from __future__ import annotations

import pytest

from search.engine.measure_verdict import ReadingEvidence, Verdict, measure_verdicts
from search.engine.qualifier_match import declared_measures_named

_ACTIVITY = (
    ("populationType", "EconomicActivity"),
    ("measuredProperty", "amount"),
    ("statType", "measuredValue"),
)
_WDI_GDP = frozenset({*_ACTIVITY, ("activitySource", "GrossDomesticProduction")})
_GDP_GROWTH = (_WDI_GDP - {("statType", "measuredValue")}) | {("statType", "growthRate")}
_GNI_AT_PPP = frozenset({*_ACTIVITY, ("activitySource", "GrossNationalIncome")})
_HEALTH = frozenset(
    {
        *_ACTIVITY,
        ("activitySource", "ExpenditureActivity"),
        ("expenditureType", "HealthcareExpenditure"),
    }
)
_EDUCATION = (_HEALTH - {("expenditureType", "HealthcareExpenditure")}) | {
    ("expenditureType", "EducationExpenditure")
}
_WEO = (
    ("populationType", "EconomicProduction"),
    ("measuredProperty", "valueAdded"),
    ("statType", "measuredValue"),
    ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
)
_WEO_GDP_PPP = frozenset(
    {
        *_WEO,
        ("productTaxBasis", "IncludingProductTaxes"),
        ("currencyConversionBasis", "PurchasingPowerParityRate"),
    }
)
_WEO_GDP_AT_MARKET_RATES = (
    _WEO_GDP_PPP - {("currencyConversionBasis", "PurchasingPowerParityRate")}
) | {("currencyConversionBasis", "MarketExchangeRate")}
_WEO_GVA = frozenset({*_WEO, ("productTaxBasis", "ExcludingProductTaxes")})
_GROSS_CAPITAL_FORMATION = frozenset(
    {
        ("populationType", "FinalExpenditure"),
        ("measuredProperty", "amount"),
        ("statType", "measuredValue"),
        ("finalExpenditureType", "CapitalFormation"),
        ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
    }
)
_DEBT_SERVICE = frozenset(
    {
        ("populationType", "FinancialFlow"),
        ("measuredProperty", "amount"),
        ("statType", "measuredValue"),
        ("financialFlowType", "ServicingPayment"),
    }
)
_DISBURSEMENTS = frozenset(
    {
        ("populationType", "FinancialFlow"),
        ("measuredProperty", "amount"),
        ("statType", "measuredValue"),
        ("financialFlowType", "Disbursement"),
    }
)
_GROSS = _DISBURSEMENTS | {("nettingBasis", "Gross")}
_NET = _DISBURSEMENTS | {("nettingBasis", "Net")}
_GDP_NAME = "Gross domestic product (GDP)"


@pytest.mark.parametrize(
    ("variable_text", "evidence", "verdicts"),
    [
        pytest.param(
            "GDP",
            ReadingEvidence(
                pairs=_GDP_GROWTH, noun="GDP growth", member_names=("GDP growth (annual %)",)
            ),
            ["unknown"],
            id="gdp-answered-with-gdp-growth-whose-pairs-hold-gdp-growth",
        ),
        pytest.param(
            "GDP",
            ReadingEvidence(noun="GDP growth", member_names=("GDP growth (annual %)",)),
            ["unknown"],
            id="gdp-answered-with-gdp-growth-by-name-alone",
        ),
        pytest.param(
            "GDP",
            ReadingEvidence(
                pairs=_GROSS_CAPITAL_FORMATION,
                noun="gross capital formation",
                member_names=("Gross capital formation, Percent of GDP",),
                denominator_name=_GDP_NAME,
            ),
            ["base"],
            id="gdp-a-share-of-gdp-divides-by-named-as-its-base",
        ),
        pytest.param(
            "GDP",
            ReadingEvidence(pairs=_HEALTH, denominator_pairs=_WDI_GDP),
            ["base"],
            id="gdp-the-reading-divides-by-held-by-its-denominator-pairs",
        ),
        pytest.param(
            "external health expenditure as a share of GDP",
            ReadingEvidence(
                pairs=_EDUCATION,
                member_properties=(frozenset({"activitySource", "expenditureType"}),),
                denominator_pairs=_WDI_GDP,
            ),
            ["contradicts", "base"],
            id="issue-47a-education-share-drafted-for-a-health-share",
        ),
        pytest.param(
            "external health expenditure as a share of GDP",
            ReadingEvidence(
                pairs=_EDUCATION,
                noun="government expenditure",
                member_names=("Government expenditure on education, total (% of GDP)",),
                denominator_name=_GDP_NAME,
            ),
            ["unknown", "holds"],
            id="issue-47a-education-share-answered-for-a-health-share",
        ),
        pytest.param(
            "health expenditure as a share of GDP",
            ReadingEvidence(
                pairs=_HEALTH,
                noun="health expenditure",
                member_names=("Current health expenditure (% of GDP)",),
                denominator_name=_GDP_NAME,
            ),
            ["holds", "base"],
            id="numerator-held-by-pairs-and-base-by-the-denominator-name",
        ),
        pytest.param(
            "debt service to GNI",
            ReadingEvidence(
                pairs=_DEBT_SERVICE,
                noun="external debt service",
                member_names=("External debt service, percent of GDP",),
                denominator_name=_GDP_NAME,
            ),
            ["unknown"],
            id="debt-service-to-gni-answered-with-a-share-of-gdp",
        ),
        pytest.param(
            "GNI per capita",
            ReadingEvidence(
                pairs=_WDI_GDP,
                noun="GDP",
                member_names=("Nominal gross domestic product per capita",),
            ),
            ["unknown"],
            id="gni-per-capita-answered-with-gdp-per-capita",
        ),
        pytest.param(
            "GNI per capita",
            ReadingEvidence(
                pairs=_GNI_AT_PPP,
                noun="gross national income based on purchasing power parity per capita",
            ),
            ["holds"],
            id="gni-per-capita-answered-with-an-undeclared-encoding-named-gni",
        ),
        pytest.param(
            "GNI",
            ReadingEvidence(
                pairs=frozenset(
                    {
                        ("populationType", "EconomicActivity"),
                        ("measuredProperty", "amount"),
                        ("statType", "growthRate"),
                        ("activitySource", "GrossNationalIncome"),
                    }
                ),
                noun="GNI growth",
                member_names=("GNI growth (annual %)",),
            ),
            ["unknown"],
            id="gni-answered-with-gni-growth",
        ),
        pytest.param(
            "GDP",
            ReadingEvidence(
                pairs=frozenset(
                    {
                        ("populationType", "EconomicProduction"),
                        ("measuredProperty", "outputGap"),
                        ("statType", "measuredValue"),
                        ("productTaxBasis", "IncludingProductTaxes"),
                        ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
                    }
                ),
                noun="output gap",
                member_names=("Output gap, Percent of potential GDP",),
                denominator_name="Potential gross domestic product (GDP)",
            ),
            ["unknown"],
            id="gdp-answered-with-the-output-gap-divided-by-potential-gdp",
        ),
        pytest.param(
            "GDP",
            ReadingEvidence(
                pairs=frozenset(
                    {
                        *_WEO,
                        ("productTaxBasis", "IncludingProductTaxes"),
                        ("cyclicalAdjustment", "CyclicallyAdjusted"),
                    }
                )
            ),
            ["unknown"],
            id="gdp-answered-with-potential-gdp-a-declared-noun-refining-it",
        ),
        pytest.param(
            "disbursements",
            ReadingEvidence(pairs=_GROSS),
            ["holds"],
            id="disbursements-answered-with-gross-disbursements-a-label-refining-it",
        ),
        pytest.param(
            "national income per person",
            ReadingEvidence(pairs=_WDI_GDP, noun="GDP"),
            [],
            id="national-income-per-person-names-no-declared-measure",
        ),
        pytest.param(
            "net disbursements",
            ReadingEvidence(pairs=_GROSS, noun="gross disbursements"),
            ["unknown"],
            id="net-disbursements-answered-with-gross",
        ),
        pytest.param(
            "net disbursements",
            ReadingEvidence(pairs=_NET, noun="ODA net disbursements"),
            ["holds"],
            id="net-disbursements-held",
        ),
        pytest.param(
            "GNI",
            ReadingEvidence(noun="gross national income"),
            ["holds"],
            id="an-acronym-names-the-measure-its-expansion-names",
        ),
        pytest.param(
            "GDP",
            ReadingEvidence(pairs=_WEO_GDP_PPP),
            ["holds"],
            id="a-name-three-declarations-share-held-by-one",
        ),
        pytest.param(
            "GDP",
            ReadingEvidence(
                pairs=_WEO_GVA,
                member_properties=(frozenset({"productTaxBasis", "fixedCapitalConsumptionBasis"}),),
            ),
            ["contradicts"],
            id="a-name-three-declarations-share-contradicted-by-each-of-its-population-type",
        ),
        pytest.param(
            "GDP",
            ReadingEvidence(
                pairs=_WEO_GDP_AT_MARKET_RATES,
                member_properties=(
                    frozenset(
                        {
                            "productTaxBasis",
                            "fixedCapitalConsumptionBasis",
                            "currencyConversionBasis",
                        }
                    ),
                ),
            ),
            ["holds"],
            id="a-name-three-declarations-share-contradicted-by-one-held-by-another",
        ),
        pytest.param(
            "health expenditure",
            ReadingEvidence(
                pairs=frozenset(
                    {
                        ("populationType", "WHO_GHED_CHE_pc_US_SHA2011"),
                        ("measuredProperty", "value"),
                        ("statType", "measuredValue"),
                    }
                ),
                noun="Current health expenditure per capita",
            ),
            ["holds"],
            id="a-series-the-graph-identifies-by-name-alone",
        ),
        pytest.param(
            "gross disbursements",
            ReadingEvidence(
                pairs=_GROSS, noun="ODA", member_names=("Official development assistance",)
            ),
            ["holds"],
            id="a-reading-whose-codes-identify-the-declared-measure",
        ),
    ],
)
def test_measure_verdicts(
    variable_text: str, evidence: ReadingEvidence, verdicts: list[Verdict]
) -> None:
    named = declared_measures_named(variable_text)

    assert measure_verdicts(named, evidence) == verdicts
