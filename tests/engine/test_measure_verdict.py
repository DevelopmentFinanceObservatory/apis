"""The verdict on each declared measure a request names, on what a reading states of itself.

Each row names the variable text, the reading's evidence and the verdict on each name
`qualifier_match.declared_measures_named` reads in the text, in text order. The pairs copy
the fixture code index's declared measures (`MeasureIndex.nouns`, `partition_labels`) and the
StatVars that answered each case in the recorded panel or a dev probe.
"""

from __future__ import annotations

import pytest

from search.engine import qualifier_match
from search.engine.measure_verdict import (
    PropertyEvidence,
    ReadingEvidence,
    Verdict,
    measure_verdicts,
    property_evidence,
    reading_evidence,
)
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


_FIVE_TUPLE_KEYS = frozenset({"populationType", "measuredProperty", "statType"})


def _reading(
    *,
    pairs: frozenset[tuple[str, str]] = frozenset(),
    members: tuple[frozenset[str], ...] | None = None,
    noun: str | None = None,
    member_names: tuple[str, ...] = (),
    denominator_pairs: frozenset[tuple[str, str]] | None = None,
    denominator_name: str | None = None,
) -> ReadingEvidence:
    """The evidence of a reading whose five-tuple and constraint values are `pairs`, its
    members declaring the constraint properties `members`: by default one member declaring
    every constraint value, and none known when `members` is empty."""
    constraints = [(prop, value) for prop, value in pairs if prop not in _FIVE_TUPLE_KEYS]
    if members is None:
        members = (frozenset(prop for prop, _value in constraints),)
    properties = {
        prop: PropertyEvidence(
            values=frozenset(value for on, value in constraints if on == prop),
            every_member=bool(members) and all(prop in member for member in members),
        )
        for prop in {prop for prop, _value in constraints}
    }
    return ReadingEvidence(
        five_tuple=pairs - frozenset(constraints),
        properties=properties,
        noun=noun,
        member_names=member_names,
        denominator_pairs=denominator_pairs,
        denominator_name=denominator_name,
    )


@pytest.mark.parametrize(
    ("variable_text", "evidence", "verdicts"),
    [
        pytest.param(
            "GDP",
            _reading(pairs=_GDP_GROWTH, noun="GDP growth", member_names=("GDP growth (annual %)",)),
            ["unknown"],
            id="gdp-answered-with-gdp-growth-whose-pairs-hold-gdp-growth",
        ),
        pytest.param(
            "GDP",
            _reading(noun="GDP growth", member_names=("GDP growth (annual %)",)),
            ["unknown"],
            id="gdp-answered-with-gdp-growth-by-name-alone",
        ),
        pytest.param(
            "GDP",
            _reading(
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
            _reading(pairs=_HEALTH, denominator_pairs=_WDI_GDP),
            ["base"],
            id="gdp-the-reading-divides-by-held-by-its-denominator-pairs",
        ),
        pytest.param(
            "external health expenditure as a share of GDP",
            _reading(
                pairs=_EDUCATION,
                members=(frozenset({"activitySource", "expenditureType"}),),
                denominator_pairs=_WDI_GDP,
            ),
            ["contradicts", "base"],
            id="issue-47a-education-share-drafted-for-a-health-share",
        ),
        pytest.param(
            "external health expenditure as a share of GDP",
            _reading(
                pairs=_EDUCATION,
                noun="government expenditure",
                member_names=("Government expenditure on education, total (% of GDP)",),
                denominator_name=_GDP_NAME,
            ),
            ["contradicts", "base"],
            id="issue-47a-education-share-answered-for-a-health-share",
        ),
        pytest.param(
            "health expenditure as a share of GDP",
            _reading(
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
            _reading(
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
            _reading(
                pairs=_WDI_GDP,
                noun="GDP",
                member_names=("Nominal gross domestic product per capita",),
            ),
            ["unknown"],
            id="gni-per-capita-answered-with-gdp-per-capita",
        ),
        pytest.param(
            "GNI per capita",
            _reading(
                pairs=_GNI_AT_PPP,
                noun="gross national income based on purchasing power parity per capita",
            ),
            ["holds"],
            id="gni-per-capita-answered-with-an-undeclared-encoding-named-gni",
        ),
        pytest.param(
            "GNI",
            _reading(
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
            _reading(
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
            _reading(
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
            _reading(pairs=_GROSS),
            ["holds"],
            id="disbursements-answered-with-gross-disbursements-a-label-refining-it",
        ),
        pytest.param(
            "national income per person",
            _reading(pairs=_WDI_GDP, noun="GDP"),
            [],
            id="national-income-per-person-names-no-declared-measure",
        ),
        pytest.param(
            "net disbursements",
            _reading(pairs=_GROSS, noun="gross disbursements"),
            ["contradicts"],
            id="net-disbursements-answered-with-gross",
        ),
        pytest.param(
            "net disbursements",
            _reading(pairs=_NET, noun="ODA net disbursements"),
            ["holds"],
            id="net-disbursements-held",
        ),
        pytest.param(
            "GNI",
            _reading(noun="gross national income"),
            ["holds"],
            id="an-acronym-names-the-measure-its-expansion-names",
        ),
        pytest.param(
            "GDP",
            _reading(pairs=_WEO_GDP_PPP),
            ["holds"],
            id="a-name-three-declarations-share-held-by-one",
        ),
        pytest.param(
            "GDP",
            _reading(
                pairs=_WEO_GVA,
                members=(frozenset({"productTaxBasis", "fixedCapitalConsumptionBasis"}),),
            ),
            ["contradicts"],
            id="a-name-three-declarations-share-contradicted-by-each-of-its-population-type",
        ),
        pytest.param(
            "GDP",
            _reading(
                pairs=_WEO_GDP_AT_MARKET_RATES,
                members=(
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
            _reading(
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
            _reading(pairs=_GROSS, noun="ODA", member_names=("Official development assistance",)),
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


@pytest.mark.parametrize(
    ("member_value", "summary_value", "verdict"),
    [
        pytest.param(
            "HealthcareExpenditure", "EducationExpenditure", "holds", id="members-hold-it"
        ),
        pytest.param(
            "EducationExpenditure", "HealthcareExpenditure", "some", id="only-the-summary-holds-it"
        ),
    ],
)
def test_a_summary_value_reaches_some_and_never_decides_holds(
    member_value: str, summary_value: str, verdict: Verdict
) -> None:
    """A held family's summary holds values without saying which members declare them, so
    one of them neither stops the members' own values holding "health expenditure" nor
    makes the measure held where no member declares it."""
    evidence = reading_evidence(
        population_type="EconomicActivity",
        measured_property="amount",
        stat_type="measuredValue",
        properties=property_evidence(
            [{"activitySource": "ExpenditureActivity", "expenditureType": member_value}],
            summary=[("expenditureType", summary_value)],
        ),
        noun=None,
        denominator_pairs=(),
    )

    assert measure_verdicts(declared_measures_named("health expenditure"), evidence) == [verdict]


def test_a_text_is_read_for_its_named_measures_once_per_pair_of_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`measure_verdicts` reads the noun and every member name of each reading it judges, so
    a text read again under the same indexes takes the answer already read."""
    text = "GDP growth of the memo test"
    tokenized: list[str] = []
    tokenize = qualifier_match.tokenize
    monkeypatch.setattr(
        qualifier_match, "tokenize", lambda words: tokenized.append(words) or tokenize(words)
    )

    first = declared_measures_named(text)
    first.clear()
    second = declared_measures_named(text)

    assert [name.occurrences for name in second] == [((0, 10),)]
    assert tokenized == [text]
