"""The price basis of a series, current or constant prices, which belongs to its unit.

A measure noun states no price basis, so it reads "gross disbursements" whatever the prices. The
byline's unit token (`models.departure.UNIT_REF`) states it with the currency: "current US
dollars", "constant 2022 US dollars". Two measure partitions that differ only in price basis
are that token's options, so `measure_label` and the measure menu name neither.

Three facts set the price basis, and the first that holds wins:

- a unit request on the identity (`IdentityUnitRequest`), which converts the answer to its
  prices and base year;
- a unit in constant prices of a year (`UnitClass.base_year`, `ChainedUSDollarBaseYr2017`);
- the shape's `measurementQualifier`, listed in `PRICES_BY_QUALIFIER`: `Nominal` is current
  prices and `RealValue` constant prices. Other qualifiers (`Adjusted`) are no price basis
  and stay in the measure noun.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from code_index.labels import PRICES_BY_QUALIFIER, PricesKind
from search.heading.unit_class import UnitClass
from search.models import IdentityUnitRequest

US_DOLLARS: Final = "US dollars"
"""The currency noun of a unit `UnitClass.currency` reads as USD."""


@dataclass(frozen=True, slots=True)
class PriceBasis:
    """Current or constant prices, with the base year of constant prices when known."""

    prices: PricesKind
    base_year: int | None


def price_basis(
    *,
    unit_request: IdentityUnitRequest | None,
    unit_class: UnitClass | None,
    measurement_qualifier: str | None,
) -> PriceBasis | None:
    """The answer's price basis, or `None` when no fact states one.

    Args:
        unit_request: the identity's unit request, when the request converts the answer.
        unit_class: the class of the served unit (`classify_unit`).
        measurement_qualifier: the shape's `measurementQualifier` dcid.
    """
    if unit_request is not None:
        return PriceBasis(prices=unit_request.prices, base_year=unit_request.base_year)
    if unit_class is not None and unit_class.base_year is not None:
        return PriceBasis(prices="constant", base_year=unit_class.base_year)
    prices = PRICES_BY_QUALIFIER.get(measurement_qualifier) if measurement_qualifier else None
    return PriceBasis(prices=prices, base_year=None) if prices is not None else None


def price_basis_phrase(basis: PriceBasis | None, *, currency_noun: str) -> str:
    """The unit token's text: "current US dollars", "constant 2022 US dollars".

    Constant prices with no known base year read "constant US dollars". No price basis
    reads the currency noun alone.
    """
    if basis is None:
        return currency_noun
    if basis.prices == "constant" and basis.base_year is not None:
        return f"constant {basis.base_year} {currency_noun}"
    return f"{basis.prices} {currency_noun}"


def price_basis_option_label(prices: PricesKind) -> str:
    """The label of a unit option that switches to `prices`: "current prices" or
    "constant prices"."""
    return f"{prices} prices"
