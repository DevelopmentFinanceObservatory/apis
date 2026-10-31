"""The price basis of a series, current or constant prices, which belongs to its unit.

A measure noun states no price basis, so it reads "gross disbursements" whatever the prices. The
byline's unit token (`models.departure.UNIT_REF`) states it with the currency: "current US
dollars", "constant 2022 US dollars". Two measure partitions that differ only in price basis
are that token's options, so `measure_label` and the measure menu name neither.

Three facts set the price basis, and the first that holds wins:

- the prices and base year of a unit request on the identity, which converts the answer to
  them;
- the price basis the unit registry gives a priced money unit (`UnitClass.prices`):
  `SDG_CU_USD` current, `CR_USD_K` constant, `ChainedUSDollarBaseYr2017` chained 2017;
- the shape's `measurementQualifier`, listed in `PRICES_BY_QUALIFIER`: `Nominal` is current
  prices and `RealValue` constant prices. Other qualifiers (`Adjusted`) are no price basis
  and stay in the measure noun.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from code_index.labels import PRICES_BY_QUALIFIER, PricesKind
from unit_words.unit_class import UnitClass

US_DOLLARS: Final = "US dollars"
"""The currency noun of a unit `UnitClass.currency` reads as USD."""


@dataclass(frozen=True, slots=True)
class PriceBasis:
    """Current or constant prices, with the base year of constant prices when known, and
    whether constant prices are chain-linked."""

    prices: PricesKind
    base_year: int | None
    chained: bool = False


def price_basis(
    *,
    requested: PriceBasis | None,
    unit_class: UnitClass | None,
    measurement_qualifier: str | None,
) -> PriceBasis | None:
    """The answer's price basis, or `None` when no fact states one.

    Args:
        requested: the prices and base year of the identity's unit request, when the request
            converts the answer.
        unit_class: the class of the served unit (`classify_unit`).
        measurement_qualifier: the shape's `measurementQualifier` dcid.
    """
    if requested is not None:
        return requested
    if unit_class is not None and unit_class.prices is not None:
        return PriceBasis(
            prices=unit_class.prices, base_year=unit_class.base_year, chained=unit_class.chained
        )
    prices = PRICES_BY_QUALIFIER.get(measurement_qualifier) if measurement_qualifier else None
    return PriceBasis(prices=prices, base_year=None) if prices is not None else None


def price_basis_phrase(basis: PriceBasis | None, *, currency_noun: str) -> str:
    """The unit token's text: "current US dollars", "constant 2022 US dollars".

    Constant prices with no known base year read "constant US dollars", and chain-linked
    ones "chained 2017 US dollars". No price basis reads the currency noun alone.
    """
    if basis is None:
        return currency_noun
    prices = "chained" if basis.chained else basis.prices
    if basis.prices == "constant" and basis.base_year is not None:
        return f"{prices} {basis.base_year} {currency_noun}"
    return f"{prices} {currency_noun}"


def price_basis_option_label(prices: PricesKind) -> str:
    """The label of a unit option that switches to `prices`: "current prices" or
    "constant prices"."""
    return f"{prices} prices"
