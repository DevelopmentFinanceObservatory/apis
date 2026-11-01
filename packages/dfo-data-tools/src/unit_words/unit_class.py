"""Which class a series' unit falls in, and whether the class changes the magnitude.

A heading states the unit when it changes what the number means: "Japan's GDP per capita
in PPP international dollars", "Ghana's external debt service as a share of exports". A
plain US-dollar amount leaves the unit to the byline.

The class comes from graph types:

- A currency is a unit the unit registry reads as money (`dfo_spec_kit.units.parse_unit`:
  `CurrencyUSD`, a bare `INR`, `USDollar`, the UN `CR_USD`, `ChainedUSDollarBaseYr2017`),
  with the price basis, base year and chaining a priced ref states (`SDG_CU_USD` current,
  `CR_USD_K` constant), else any other dcid `graph.currency_units()` maps to an ISO 4217
  code. USD reads as USD whatever its dcid.
- PPP and international dollars are the `PurchasingPowerUnit` node
  `PPPInternationalDollar` and the `UnitOfMeasure` nodes `InternationalDollar` and
  `PurchasingPowerStandard`, listed in `_PURCHASING_POWER_UNITS`.
- Percent and a rate over a stated count come from the unit registry too. A rate's base
  is the unit's own: the World Development Indicators crude birth rate divides by
  `Count_Person` and is served in `Per1000Persons`, so it is a rate per 1,000 people.
- Per head is a `measurementDenominator` of `PerCapita` (base Data Commons) or
  `Count_Person` (WEO, IDS) unless the unit is a percent or a registry rate. Tonnes and kWh
  per person are per head, and so is a series whose unit is not yet known. A share of the
  population is no per-person quantity, and a rate states its own base.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from dfo_spec_kit.units import Money, Other, Percent, Rate, UnpricedMoney, parse_unit

PER_CAPITA_DENOMINATOR: Final = "PerCapita"
"""The base Data Commons `measurementDenominator` that frames a measure as per capita. A
`Count_Person` denominator is a per-person amount too, except under a rate, which states its
own base (a crude birth rate in `Per1000Persons`), or a percent, a share (a census share in
`Percent`)."""

PER_HEAD_DENOMINATORS: Final = frozenset({"Count_Person", PER_CAPITA_DENOMINATOR})
"""`measurementDenominator` dcids of a per-person rate."""

_PURCHASING_POWER_UNITS: Final = frozenset(
    {"PPPInternationalDollar", "InternationalDollar", "PurchasingPowerStandard"}
)
_PLAIN_CURRENCY: Final = "USD"

type UnitKind = Literal["currency", "purchasing_power", "percent", "other", "none"]
"""`currency`: an ISO 4217 currency. `purchasing_power`: PPP or international dollars.
`percent`: a percentage. `other`: a unit outside these classes, a rate over a stated count
among them. `none`: the series names no unit."""

type Prices = Literal["current", "constant"]


def is_known_non_currency(unit: str) -> bool:
    """Whether `unit` is one the unit registry types as no money (a percent, a count, a rate)
    or a PPP or international-dollar unit, which no graph currency unit can be, so
    classifying it needs no `graph.currency_units()` read."""
    value = parse_unit(unit)
    return not isinstance(value, Money | UnpricedMoney | Other) or unit in _PURCHASING_POWER_UNITS


def is_per_head_denominator(denominator: str | None) -> bool:
    """Whether a `measurementDenominator` dcid makes the measure a per-person rate."""
    return denominator in PER_HEAD_DENOMINATORS


@dataclass(frozen=True, slots=True)
class UnitClass:
    """The class of a series' unit and its denominator.

    `unit` is the facet unit dcid. `currency` is its ISO 4217 code when `kind` is
    `currency`. `per_head` is true for a per-person rate. `ratio_denominator` is the
    `measurementDenominator` dcid of any other ratio ("% of exports"), which
    `measure_noun.denominator_noun` names. `prices` is the price basis the unit registry
    gives a priced money unit (`SDG_CU_USD` current, `CR_USD_K` constant), else `None`;
    `base_year` is the year of a unit in constant prices of that year, and `chained` is
    true for a chain-linked one (`ChainedUSDollarBaseYr2017`). `rate` is
    the typed value of a unit over a stated count, else `None`; it sets neither `per_head`
    nor `ratio_denominator`, since its base is the denominator.
    """

    kind: UnitKind
    unit: str | None
    currency: str | None
    per_head: bool
    ratio_denominator: str | None
    base_year: int | None = None
    rate: Rate | None = None
    prices: Prices | None = None
    chained: bool = False

    @property
    def changes_magnitude(self) -> bool:
        """Whether the unit changes what the number means, so the heading states it.

        True for a per-head rate, a rate over a stated count, a percent or other ratio, PPP
        or international dollars, and any currency other than US dollars. False for US
        dollars and for units outside these classes, which the byline states.
        """
        if self.per_head or self.rate is not None or self.ratio_denominator is not None:
            return True
        if self.kind in ("percent", "purchasing_power"):
            return True
        return self.kind == "currency" and self.currency != _PLAIN_CURRENCY


def rate_phrase(rate: Rate) -> str:
    """A rate's base as a heading reads it: "per 1,000 people", or "per 1,000" when the unit
    states nothing counted."""
    base = f"per {rate.count_base:,}"
    return base if rate.counted is None else f"{base} {rate.counted}"


def classify_unit(
    *,
    unit: str | None,
    denominator: str | None,
    currency_units: Mapping[str, str],
) -> UnitClass:
    """Classify a series' unit and denominator.

    Args:
        unit: the facet unit dcid (`ResolvedSpec.facets["unit"].ref`), or `None` when the
            series names none.
        denominator: the shape's `measurementDenominator` dcid, or `None`.
        currency_units: `graph.currency_units()`, unit dcid to ISO 4217 code.

    Returns:
        The unit's class. A rate over a stated count takes the place of the denominator.
        Otherwise the denominator sets `per_head` or `ratio_denominator`, so a per-capita
        amount in Kenyan shillings is a `currency` with `per_head` true. A per-head
        denominator under a percent sets neither, so a census share over `Count_Person` in
        `Percent` is a plain `percent`.
    """
    value = parse_unit(unit) if unit is not None else None
    currency: str | None = None
    if isinstance(value, Money | UnpricedMoney):
        currency = value.currency
    elif unit is not None:
        currency = currency_units.get(unit)
    kind: UnitKind
    if unit is None:
        kind = "none"
    elif currency is not None:
        kind = "currency"
    elif unit in _PURCHASING_POWER_UNITS:
        kind = "purchasing_power"
    elif isinstance(value, Percent):
        kind = "percent"
    else:
        kind = "other"
    rate = value if isinstance(value, Rate) else None
    per_head = rate is None and kind != "percent" and is_per_head_denominator(denominator)
    money = value if isinstance(value, Money) else None
    return UnitClass(
        kind=kind,
        unit=unit,
        currency=currency,
        per_head=per_head,
        ratio_denominator=None
        if rate is not None or is_per_head_denominator(denominator)
        else denominator,
        base_year=money.base_year if money is not None else None,
        rate=rate,
        prices=money.prices if money is not None else None,
        chained=money is not None and money.chained,
    )
