"""The closest-match rule, on inputs copied from recorded panel responses.

Each row's variable text, weak flag, StatVar names and bound codes come from the
response the panel recorded for that query id; the measure noun is the one the heading
shows for its shape (`search.heading.measure_noun`, or the CRS flow and partition names).
"""

from __future__ import annotations

import pytest

from search.engine.closest_match import closest_match_fact
from search.heading.departure_fact import DepartureFact


@pytest.mark.parametrize(
    ("variable_text", "weak", "measure_noun", "stat_var_names", "bound_codes", "fires"),
    [
        pytest.param(
            "cost of living",
            True,
            "health expenditure",
            ["Healthcare expenditure per person"],
            ["ExpenditureActivity", "HealthcareExpenditure"],
            True,
            id="weo-038-cost-of-living-fires",
        ),
        pytest.param(
            "GGXWDG_NGDP",
            True,
            "GDP",
            ["Gross domestic product (GDP), Current prices"],
            ["GrossOfFixedCapitalConsumption", "IncludingProductTaxes"],
            True,
            id="weo-039-imf-code-answered-with-gdp-fires",
        ),
        pytest.param(
            "debt service",
            True,
            "external debt service",
            ["Debt service on external debt, by creditor (current US$)"],
            ["ServicingPayment"],
            False,
            id="ids-002-debt-service",
        ),
        pytest.param(
            "aid",
            False,
            "ODA gross disbursements",
            [
                "Official development assistance gross disbursements for education "
                "facility and training (standard grant)"
            ],
            ["CRSTypeOfFlow10", "Disbursement", "Gross"],
            False,
            id="crs-001-aid-strong-score",
        ),
        pytest.param(
            "aid",
            True,
            "ODA gross disbursements",
            [
                "Official development assistance gross disbursements for education "
                "facility and training (standard grant)"
            ],
            ["CRSTypeOfFlow10", "Disbursement", "Gross"],
            False,
            id="crs-001-aid-weak-score-names-the-oda-default",
        ),
        pytest.param(
            "GDP",
            False,
            "GDP",
            ["Nominal gross domestic product"],
            ["GrossDomesticProduction"],
            False,
            id="weo-004-gdp-strong-score",
        ),
        pytest.param(
            "investment as share of GDP",
            True,
            "gross capital formation",
            ["Gross capital formation, Percent of GDP"],
            ["CapitalFormation", "GrossOfFixedCapitalConsumption"],
            False,
            id="weo-008-gdp-in-statvar-name",
        ),
        pytest.param(
            "humanitarian aid",
            True,
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
            True,
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
            True,
            "GDP",
            ["Nominal gross domestic product"],
            [],
            False,
            id="no-content-word-asked",
        ),
    ],
)
def test_closest_match_fact(
    variable_text: str,
    weak: bool,
    measure_noun: str,
    stat_var_names: list[str],
    bound_codes: list[str],
    fires: bool,
) -> None:
    fact = closest_match_fact(
        variable_text=variable_text,
        retrieval_score_weak=weak,
        measure_noun=measure_noun,
        stat_var_names=stat_var_names,
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
