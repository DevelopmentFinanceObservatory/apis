"""The closest-match rule, on inputs copied from recorded panel responses.

Each row's variable text, StatVar names and bound codes come from the response the panel
recorded for that query id, or from the dev answer its id names; the measure noun is the one
the heading shows for its shape (`search.heading.measure_noun`, or the CRS flow and partition
names). The rule reads no retrieval score.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from scripts.sentence_golden import golden_parts
from search.engine.closest_match import closest_match_fact, reading_inputs
from search.engine.measure_verdict import ReadingEvidence, measure_verdicts
from search.engine.qualifier_match import declared_measures_named
from search.heading.departure_fact import DepartureFact
from search.models import CandidatesPart, DefinitePart, GraphRef, Spec
from search.models.spec import OpenAxisValueCount, OpenConstraintAxisFact

_PANEL_DIR = Path(__file__).parents[1] / "fixtures" / "sentence_panel"
_PRODUCT_TAX = "productTaxBasis"
_INCLUDING_TAXES = GraphRef(ref="IncludingProductTaxes", label="Including product taxes")
_GDP_PPP_SHAPE = "Gross domestic product (GDP), per capita, purchasing power parity (PPP)"


@pytest.mark.parametrize(
    ("variable_text", "measure_noun", "stat_var_names", "bound_codes", "fires"),
    [
        pytest.param(
            "cost of living",
            "health expenditure",
            ["Healthcare expenditure per person"],
            ["ExpenditureActivity", "HealthcareExpenditure"],
            True,
            id="weo-038-cost-of-living-fires",
        ),
        pytest.param(
            "GGXWDG_NGDP",
            "GDP",
            ["Gross domestic product (GDP), Current prices"],
            ["GrossOfFixedCapitalConsumption", "IncludingProductTaxes"],
            True,
            id="weo-039-imf-code-answered-with-gdp-fires",
        ),
        pytest.param(
            "debt service",
            "external debt service",
            ["Debt service on external debt, by creditor (current US$)"],
            ["ServicingPayment"],
            False,
            id="ids-002-debt-service",
        ),
        pytest.param(
            "aid",
            "ODA gross disbursements",
            [
                "Official development assistance gross disbursements for education "
                "facility and training (standard grant)"
            ],
            ["CRSTypeOfFlow10", "Disbursement", "Gross"],
            False,
            id="crs-001-aid-names-the-oda-default",
        ),
        pytest.param(
            "aid",
            "gross disbursements",
            ["Official development assistance (gross disbursements)"],
            ["DAC2aAidType206", "Disbursement", "Gross"],
            False,
            id="crs-001-dac2a-aid-names-the-oda-code-also-called",
        ),
        pytest.param(
            "loans",
            "gross ODA disbursements",
            [
                "Official development assistance gross disbursements for education policy "
                "and administrative management (standard grant)",
                "Official development assistance gross disbursements for teacher training "
                "(standard grant)",
            ],
            ["Disbursement", "Gross"],
            True,
            id="crs-033-loans-answered-with-gross-oda-fires",
        ),
        pytest.param(
            "GDP",
            "GDP",
            ["Nominal gross domestic product"],
            ["GrossDomesticProduction"],
            False,
            id="weo-004-gdp",
        ),
        pytest.param(
            "GDP",
            "nominal gross domestic product",
            ["Nominal gross domestic product"],
            [],
            False,
            id="std-02-gdp-names-gross-domestic-product",
        ),
        pytest.param(
            "gross domestic product",
            "GDP",
            ["GDP (current US$)"],
            [],
            False,
            id="gross-domestic-product-names-gdp",
        ),
        pytest.param(
            "gross",
            "GDP",
            ["GDP (current US$)"],
            [],
            True,
            id="a-word-of-a-declared-name-names-no-measure",
        ),
        pytest.param(
            "national income per person",
            "GDP",
            ["Nominal gross domestic product per capita"],
            ["GrossDomesticProduction"],
            True,
            id="ghana-national-income-answered-with-gdp-per-capita-fires",
        ),
        pytest.param(
            "GDP per capita PPP",
            "gross national income based on purchasing power parity per capita",
            ["Gross National Income Based on Purchasing Power Parity Per Capita"],
            ["GrossNationalIncome"],
            True,
            id="weo-020-a-declared-measure-not-shown-fires-beside-shared-words",
        ),
        pytest.param(
            "GNI",
            "gross national income",
            ["Gross national income"],
            ["GrossOfFixedCapitalConsumption"],
            False,
            id="a-declared-acronym-names-its-expansion",
        ),
        pytest.param(
            "investment as share of GDP",
            "gross capital formation",
            ["Gross capital formation, Percent of GDP"],
            ["CapitalFormation", "GrossOfFixedCapitalConsumption"],
            False,
            id="weo-008-gdp-in-statvar-name",
        ),
        pytest.param(
            "humanitarian aid",
            "ODA gross disbursements",
            [
                "Official development assistance gross disbursements for material relief "
                "assistance and service (standard grant, core support to non-governmental "
                "organisations, other private bodies, public-private partnerships, and "
                "research institutes)"
            ],
            ["CRSPurposeCode700", "CRSTypeOfFlow10", "Disbursement", "Gross"],
            False,
            id="crs-030-bound-code-label",
        ),
        pytest.param(
            "climate finance",
            "gross disbursements",
            [
                "Official development assistance gross disbursements for education "
                "facility and training (standard grant)",
                "Official development assistance gross disbursements for public finance "
                "management (standard grant, project-type intervention)",
            ],
            ["Disbursement", "Gross"],
            True,
            id="crs-031-one-member-name-names-that-member",
        ),
        pytest.param(
            "of the",
            "GDP",
            ["Nominal gross domestic product"],
            [],
            False,
            id="no-content-word-asked",
        ),
        pytest.param(
            "GDP growth",
            "GDP",
            ["Nominal gross domestic product"],
            ["GrossDomesticProduction"],
            True,
            id="a-shown-name-inside-a-longer-asked-noun-fires",
        ),
        pytest.param(
            "net disbursements",
            "gross disbursements",
            ["Official development assistance (gross disbursements)"],
            ["Disbursement", "Gross"],
            True,
            id="a-shown-name-inside-a-longer-asked-label-fires",
        ),
        pytest.param(
            "GDP",
            "GDP growth",
            ["GDP growth (annual %)"],
            ["GrossDomesticProduction"],
            True,
            id="a-longer-shown-name-holding-the-asked-one-fires",
        ),
        pytest.param(
            "GDP growth",
            "GDP growth",
            ["Gross domestic product growth rate"],
            ["GrossDomesticProduction"],
            False,
            id="the-longer-asked-name-shown",
        ),
        pytest.param(
            "gross national income per capita",
            "GDP",
            ["Nominal gross domestic product per capita"],
            ["GrossDomesticProduction"],
            True,
            id="a-declared-measure-asked-by-name-against-gdp-per-capita-fires",
        ),
        pytest.param(
            "GNI per capita",
            "GDP",
            ["Nominal gross domestic product per capita"],
            ["GrossDomesticProduction"],
            True,
            id="a-declared-acronym-names-its-measure-against-gdp-per-capita-fires",
        ),
        pytest.param(
            "GNI per capita",
            "gross national income",
            ["Gross national income per capita"],
            ["GrossNationalIncome"],
            False,
            id="a-declared-acronym-whose-measure-is-shown",
        ),
    ],
)
def test_closest_match_fact(
    variable_text: str,
    measure_noun: str,
    stat_var_names: list[str],
    bound_codes: list[str],
    fires: bool,
) -> None:
    fact = closest_match_fact(
        variable_text=variable_text,
        evidence=ReadingEvidence(noun=measure_noun, member_names=tuple(stat_var_names)),
        bound_codes=bound_codes,
    )
    expected = (
        DepartureFact(
            code="closest_match", binding_ref=None, asked=variable_text, shown=measure_noun
        )
        if fires
        else None
    )
    assert fact == expected


@pytest.mark.parametrize(
    ("variable_text", "series_codes", "fires"),
    [
        pytest.param("DT.TDS.DEGG.CD", ["DT.TDS.DEGG.CD"], False, id="the-code-names-its-series"),
        pytest.param("DT.TDS.DEGG.CD", [], True, id="a-code-standing-for-another-series"),
    ],
)
def test_a_series_code_standing_for_the_answer_names_its_measure(
    variable_text: str, series_codes: list[str], fires: bool
) -> None:
    fact = closest_match_fact(
        variable_text=variable_text,
        evidence=ReadingEvidence(
            noun="external debt service",
            member_names=("Debt service on external debt, public sector (PPG) (TDS, current US$)",),
        ),
        bound_codes=["ServicingPayment"],
        series_codes=series_codes,
    )
    assert (fact is not None) is fires


_HEALTH = (
    ("populationType", "EconomicActivity"),
    ("measuredProperty", "amount"),
    ("statType", "measuredValue"),
    ("activitySource", "ExpenditureActivity"),
    ("expenditureType", "HealthcareExpenditure"),
)
_EDUCATION = (*_HEALTH[:4], ("expenditureType", "EducationExpenditure"))
_DEBT_SERVICE = (
    ("populationType", "FinancialFlow"),
    ("measuredProperty", "amount"),
    ("statType", "measuredValue"),
    ("financialFlowType", "ServicingPayment"),
)
_GROSS_CAPITAL_FORMATION = (
    ("populationType", "FinalExpenditure"),
    ("measuredProperty", "amount"),
    ("statType", "measuredValue"),
    ("finalExpenditureType", "CapitalFormation"),
    ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
)
_GDP = "Gross domestic product (GDP)"
_SHAPE_KEYS = frozenset({"populationType", "measuredProperty", "statType"})


@pytest.mark.parametrize(
    ("variable_text", "measure_noun", "stat_var_name", "pairs", "denominator", "fires"),
    [
        pytest.param(
            "external health expenditure as a share of GDP",
            "government expenditure",
            "Government expenditure on education, total (% of GDP)",
            _EDUCATION,
            _GDP,
            True,
            id="issue-47-education-shown-for-health-share-of-gdp-fires",
        ),
        pytest.param(
            "debt service to GNI",
            "external debt service",
            "External debt service, percent of GDP",
            _DEBT_SERVICE,
            _GDP,
            True,
            id="a-share-of-gdp-shown-for-a-share-of-gni-fires",
        ),
        pytest.param(
            "health expenditure as a share of GDP",
            "health expenditure",
            "Current health expenditure (% of GDP)",
            _HEALTH,
            _GDP,
            False,
            id="numerator-and-base-shown",
        ),
        pytest.param(
            "health expenditure as a share of GDP",
            "health expenditure",
            "Current health expenditure",
            _HEALTH,
            _GDP,
            False,
            id="a-base-only-the-denominator-names-is-shown",
        ),
        pytest.param(
            "GDP",
            "gross capital formation",
            "Gross capital formation, Percent of GDP",
            _GROSS_CAPITAL_FORMATION,
            _GDP,
            False,
            id="weo-009-gdp-answered-with-a-share-of-gdp-is-silent",
        ),
    ],
)
def test_closest_match_fact_on_a_ratio(
    variable_text: str,
    measure_noun: str,
    stat_var_name: str,
    pairs: tuple[tuple[str, str], ...],
    denominator: str,
    fires: bool,
) -> None:
    """Every declared measure the words name must be held or be the base the reading
    divides by (its denominator's name). The base counts whether or not the words asked for
    it, so "GDP" answered with a share of GDP stays silent."""
    fact = closest_match_fact(
        variable_text=variable_text,
        evidence=ReadingEvidence(
            pairs=frozenset(pairs),
            noun=measure_noun,
            member_names=(stat_var_name,),
            denominator_name=denominator,
        ),
        bound_codes=[value for prop, value in pairs if prop not in _SHAPE_KEYS],
    )
    assert (fact is not None) == fires


def _panel_reading(query_id: str, shape_label: str) -> Spec:
    """The reading of `query_id` in the recorded sentence panel whose shape `shape_label`
    names."""
    for path in sorted(_PANEL_DIR.glob("*.jsonl.gz")):
        for text in gzip.decompress(path.read_bytes()).splitlines():
            line = json.loads(text)
            if line["id"] != query_id:
                continue
            for part in golden_parts(line["parts"]):
                specs = (
                    [part.spec]
                    if isinstance(part, DefinitePart)
                    else part.candidates.specs
                    if isinstance(part, CandidatesPart)
                    else []
                )
                for spec in specs:
                    if spec.resolved.shape.label == shape_label:
                        return spec
    raise AssertionError(f"no {shape_label!r} reading of {query_id}")


_SPLIT_AXIS = OpenConstraintAxisFact(
    property=_PRODUCT_TAX,
    varies=True,
    distinct_count=2,
    undeclared_count=0,
    values=[
        OpenAxisValueCount(value=_INCLUDING_TAXES, count=1),
        OpenAxisValueCount(
            value=GraphRef(ref="ExcludingProductTaxes", label="Excluding product taxes"), count=1
        ),
    ],
)


@pytest.mark.parametrize(
    ("open_axis", "verdicts"),
    [
        pytest.param("agreed", ["holds"], id="every-member-holds"),
        pytest.param("split", ["unknown"], id="members-split"),
        pytest.param("some-undeclared", ["unknown"], id="some-undeclared"),
        pytest.param(None, ["unknown"], id="no-open-axis"),
    ],
)
def test_a_value_every_member_holds_on_an_open_axis_reaches_the_pairs(
    open_axis: str | None, verdicts: list[str]
) -> None:
    """The weo-020 reading of GDP per capita at purchasing power parity binds its product
    tax basis on a slot. Left on no slot, the value every served member holds on the open
    axis (`ResolvedSpec.open_axis_facts`) is in the reading's pairs, so its pairs alone hold
    "GDP". Members split between including and excluding the taxes hold neither value, and
    neither do members of which one declares no tax basis."""
    spec = _panel_reading("weo-020", _GDP_PPP_SHAPE)
    resolved = spec.resolved
    slots = [
        s for s in resolved.slots if s.key.property is None or s.key.property.ref != _PRODUCT_TAX
    ]
    facts = {
        "agreed": [
            OpenConstraintAxisFact(
                property=_PRODUCT_TAX,
                varies=False,
                value=_INCLUDING_TAXES,
                distinct_count=1,
                undeclared_count=0,
                values=[OpenAxisValueCount(value=_INCLUDING_TAXES, count=1)],
            )
        ],
        "split": [_SPLIT_AXIS],
        "some-undeclared": [
            OpenConstraintAxisFact(
                property=_PRODUCT_TAX,
                varies=False,
                value=_INCLUDING_TAXES,
                distinct_count=1,
                undeclared_count=1,
                values=[OpenAxisValueCount(value=_INCLUDING_TAXES, count=1)],
            )
        ],
    }
    reading = spec.model_copy(
        update={
            "resolved": resolved.model_copy(
                update={
                    "slots": slots,
                    "open_axis_facts": facts[open_axis] if open_axis else None,
                }
            )
        }
    )

    evidence, _bound, _series = reading_inputs(reading, measure_noun="")
    pairs_alone = dataclasses.replace(evidence, noun=None, member_names=())

    assert measure_verdicts(declared_measures_named("GDP"), pairs_alone) == verdicts
