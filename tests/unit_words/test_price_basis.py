"""Tests the price basis the unit token states."""

from __future__ import annotations

import pytest

from unit_words.price_basis import (
    US_DOLLARS,
    PriceBasis,
    price_basis,
    price_basis_phrase,
)
from unit_words.unit_class import classify_unit

_USD = classify_unit(unit="CurrencyUSD", denominator=None, currency_units={"CurrencyUSD": "USD"})
_CHAINED = classify_unit(unit="ChainedUSDollarBaseYr2017", denominator=None, currency_units={})


@pytest.mark.parametrize(
    ("requested", "unit_class", "qualifier", "expected"),
    [
        pytest.param(None, _USD, "Nominal", PriceBasis("current", None), id="nominal-shape"),
        pytest.param(None, _USD, "RealValue", PriceBasis("constant", None), id="real-shape"),
        pytest.param(
            None, _CHAINED, None, PriceBasis("constant", 2017, chained=True), id="chained-unit"
        ),
        pytest.param(
            PriceBasis("constant", 2022),
            _USD,
            "Nominal",
            PriceBasis("constant", 2022),
            id="request-converts-nominal",
        ),
        pytest.param(None, _USD, "Adjusted", None, id="qualifier-not-a-price-basis"),
        pytest.param(None, None, None, None, id="nothing-states-one"),
    ],
)
def test_price_basis(requested, unit_class, qualifier, expected) -> None:
    assert (
        price_basis(requested=requested, unit_class=unit_class, measurement_qualifier=qualifier)
        == expected
    )


@pytest.mark.parametrize(
    ("basis", "expected"),
    [
        (PriceBasis("current", None), "current US dollars"),
        (PriceBasis("constant", 2022), "constant 2022 US dollars"),
        (PriceBasis("constant", None), "constant US dollars"),
        (None, "US dollars"),
    ],
)
def test_price_basis_phrase(basis: PriceBasis | None, expected: str) -> None:
    assert price_basis_phrase(basis, currency_noun=US_DOLLARS) == expected
