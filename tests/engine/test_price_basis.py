"""Tests the price basis the unit token states, and which dropped phrases name one."""

from __future__ import annotations

import pytest

from search.engine.unit.bind import names_price_basis
from search.heading.price_basis import (
    US_DOLLARS,
    PriceBasis,
    price_basis,
    price_basis_phrase,
)
from search.heading.unit_class import classify_unit
from search.models import IdentityUnitRequest

_USD = classify_unit(unit="CurrencyUSD", denominator=None, currency_units={"CurrencyUSD": "USD"})
_CHAINED = classify_unit(unit="ChainedUSDollarBaseYr2017", denominator=None, currency_units={})


@pytest.mark.parametrize(
    ("unit_request", "unit_class", "qualifier", "expected"),
    [
        pytest.param(None, _USD, "Nominal", PriceBasis("current", None), id="nominal-shape"),
        pytest.param(None, _USD, "RealValue", PriceBasis("constant", None), id="real-shape"),
        pytest.param(None, _CHAINED, None, PriceBasis("constant", 2017), id="chained-unit"),
        pytest.param(
            IdentityUnitRequest(currency="USD", prices="constant", base_year=2022),
            _USD,
            "Nominal",
            PriceBasis("constant", 2022),
            id="request-converts-nominal",
        ),
        pytest.param(None, _USD, "Adjusted", None, id="qualifier-not-a-price-basis"),
        pytest.param(None, None, None, None, id="nothing-states-one"),
    ],
)
def test_price_basis(unit_request, unit_class, qualifier, expected) -> None:
    assert (
        price_basis(
            unit_request=unit_request, unit_class=unit_class, measurement_qualifier=qualifier
        )
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("constant 2022 dollars", True),
        ("in real terms", True),
        ("at current prices", True),
        ("in euros", False),
        ("by sector", False),
    ],
)
def test_names_price_basis(text: str, expected: bool) -> None:
    assert names_price_basis(text) is expected
