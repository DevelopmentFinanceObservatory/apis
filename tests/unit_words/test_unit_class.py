"""Tests the unit class and whether it changes a series' magnitude, on units seen on the dev graph."""

from __future__ import annotations

import pytest
from dfo_spec_kit.units import Rate

from unit_words.unit_class import UnitClass, classify_unit

# The shape `graph.currency_units()` returns: CurrencyUnit members, the `Currency` members
# named by their code (INR), and USDollar.
_CURRENCY_UNITS = {"CurrencyUSD": "USD", "CurrencyKES": "KES", "INR": "INR", "USDollar": "USD"}
_EXPORTS = "Amount_EconomicActivity_ExportsOfGoodsServicesAndPrimaryIncome_Nominal"


@pytest.mark.parametrize(
    ("unit", "denominator", "expected", "changes_magnitude"),
    [
        pytest.param(
            "CurrencyUSD",
            None,
            UnitClass("currency", "CurrencyUSD", "USD", per_head=False, ratio_denominator=None),
            False,
            id="ids-002-usd",
        ),
        pytest.param(
            "USDollar",
            None,
            UnitClass("currency", "USDollar", "USD", per_head=False, ratio_denominator=None),
            False,
            id="usd-under-another-dcid",
        ),
        pytest.param(
            "CurrencyKES",
            None,
            UnitClass("currency", "CurrencyKES", "KES", per_head=False, ratio_denominator=None),
            True,
            id="national-currency",
        ),
        pytest.param(
            "INR",
            None,
            UnitClass("currency", "INR", "INR", per_head=False, ratio_denominator=None),
            True,
            id="national-currency-graph-currency-type",
        ),
        pytest.param(
            "PPPInternationalDollar",
            "Count_Person",
            UnitClass(
                "purchasing_power",
                "PPPInternationalDollar",
                None,
                per_head=True,
                ratio_denominator=None,
            ),
            True,
            id="weo-005-ppp-per-capita",
        ),
        pytest.param(
            "USDollar",
            "PerCapita",
            UnitClass("currency", "USDollar", "USD", per_head=True, ratio_denominator=None),
            True,
            id="per-capita-spelling",
        ),
        pytest.param(
            "Percent",
            _EXPORTS,
            UnitClass("percent", "Percent", None, per_head=False, ratio_denominator=_EXPORTS),
            True,
            id="ids-037-percent-of-exports",
        ),
        pytest.param(
            "ChainedUSDollarBaseYr2017",
            None,
            UnitClass(
                "currency",
                "ChainedUSDollarBaseYr2017",
                "USD",
                per_head=False,
                ratio_denominator=None,
                base_year=2017,
                prices="constant",
                chained=True,
            ),
            False,
            id="chained-usd-constant-2017",
        ),
        pytest.param(
            "SDG_CU_USD",
            None,
            UnitClass(
                "currency",
                "SDG_CU_USD",
                "USD",
                per_head=False,
                ratio_denominator=None,
                prices="current",
            ),
            False,
            id="sdg-current-usd",
        ),
        pytest.param(
            "undata/UNIT_MEASURE-CR_USD",
            None,
            UnitClass(
                "currency",
                "undata/UNIT_MEASURE-CR_USD",
                "USD",
                per_head=False,
                ratio_denominator=None,
            ),
            False,
            id="undata-usd",
        ),
        pytest.param(
            "undata/UNIT_MEASURE-CR_USD_K",
            None,
            UnitClass(
                "currency",
                "undata/UNIT_MEASURE-CR_USD_K",
                "USD",
                per_head=False,
                ratio_denominator=None,
                prices="constant",
            ),
            False,
            id="undata-thousand-usd",
        ),
        pytest.param(
            "Year",
            None,
            UnitClass("other", "Year", None, per_head=False, ratio_denominator=None),
            False,
            id="unclassified-unit",
        ),
        pytest.param(
            "Per1000Persons",
            "Count_Person",
            UnitClass(
                "other",
                "Per1000Persons",
                None,
                per_head=False,
                ratio_denominator=None,
                rate=Rate(1000, "people"),
            ),
            True,
            id="crude-birth-rate-count-base-over-count-person",
        ),
        pytest.param(
            "Per1000Persons",
            "PerCapita",
            UnitClass(
                "other",
                "Per1000Persons",
                None,
                per_head=False,
                ratio_denominator=None,
                rate=Rate(1000, "people"),
            ),
            True,
            id="count-base-clears-a-per-capita-denominator",
        ),
        pytest.param(
            "Per1000LiveBirths",
            "Count_BirthEvent_LiveBirth",
            UnitClass(
                "other",
                "Per1000LiveBirths",
                None,
                per_head=False,
                ratio_denominator=None,
                rate=Rate(1000, "live births"),
            ),
            True,
            id="count-base-takes-the-place-of-a-ratio-denominator",
        ),
        pytest.param(
            "PersonPerSquareKilometer",
            "Count_Person",
            UnitClass(
                "other",
                "PersonPerSquareKilometer",
                None,
                per_head=True,
                ratio_denominator=None,
            ),
            True,
            id="count-person-under-a-physical-unit-reads-per-head",
        ),
        pytest.param(
            "Percent",
            "Count_Person",
            UnitClass("percent", "Percent", None, per_head=False, ratio_denominator=None),
            True,
            id="census-percent-of-people-is-no-per-head-amount",
        ),
        pytest.param(
            None,
            "Count_Person",
            UnitClass("none", None, None, per_head=True, ratio_denominator=None),
            True,
            id="count-person-over-no-unit-reads-per-head",
        ),
        pytest.param(
            None,
            None,
            UnitClass("none", None, None, per_head=False, ratio_denominator=None),
            False,
            id="no-unit",
        ),
    ],
)
def test_classify_unit(
    unit: str | None, denominator: str | None, expected: UnitClass, changes_magnitude: bool
) -> None:
    unit_class = classify_unit(unit=unit, denominator=denominator, currency_units=_CURRENCY_UNITS)
    assert unit_class == expected
    assert unit_class.changes_magnitude is changes_magnitude
