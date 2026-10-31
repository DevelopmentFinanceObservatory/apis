"""Which class a series' unit falls in, and whether the class changes the magnitude.

A heading states the unit when it changes what the number means: "Japan's GDP per capita
in PPP international dollars", "Ghana's external debt service as a share of exports". A
plain US-dollar amount leaves the unit to the byline.

The class comes from graph types:

- A currency is any dcid `graph.currency_units()` maps to an ISO 4217 code: the
  `CurrencyUnit` members (`CurrencyUSD`, `CurrencyKES`), the base Data Commons `Currency`
  members named by their code (`INR`), and the `CurrencyUnitOfMeasure` nodes (`USDollar`).
  USD reads as USD whatever its dcid.
  `_DECLARED_CURRENCY_UNITS` adds the US-dollar units no graph node types: the UN System
  Data Commons `undata/UNIT_MEASURE-CR_USD` and `undata/UNIT_MEASURE-CR_USD_K`, and
  `ChainedUSDollarBaseYr2017`, constant 2017 US dollars, typed only `UnitOfMeasure`.
- PPP and international dollars are the `PurchasingPowerUnit` node
  `PPPInternationalDollar` and the `UnitOfMeasure` nodes `InternationalDollar` and
  `PurchasingPowerStandard`, listed in `_PURCHASING_POWER_UNITS`.
- Percent and a rate over a stated count come from the unit registry
  (`dfo_spec_kit.units.parse_unit`). A rate's base is the unit's own: the World Development
  Indicators crude birth rate divides by `Count_Person` and is served in `Per1000Persons`,
  so it is a rate per 1,000 people.
- Per head is a `measurementDenominator` of `PerCapita` (base Data Commons), or of
  `Count_Person` (WEO, IDS) under a currency, purchasing-power or percent unit. A
  `Count_Person` rate under any other unit, or before its unit is known, states no scale,
  so nothing reads it as per head.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from dfo_spec_kit.units import Money, Percent, Rate, UnpricedMoney, parse_unit

PER_CAPITA_DENOMINATOR: Final = "PerCapita"
"""The base Data Commons `measurementDenominator` that frames a measure as per capita. A
`Count_Person` denominator is a per-person rate too, but also the denominator of rates
whose unit states another base (a crude birth rate in `Per1000Persons`)."""

PER_HEAD_DENOMINATORS: Final = frozenset({"Count_Person", PER_CAPITA_DENOMINATOR})
"""`measurementDenominator` dcids of a per-person rate."""

_PURCHASING_POWER_UNITS: Final = frozenset(
    {"PPPInternationalDollar", "InternationalDollar", "PurchasingPowerStandard"}
)
_PLAIN_CURRENCY: Final = "USD"


@dataclass(frozen=True, slots=True)
class _DeclaredCurrencyUnit:
    currency: str
    base_year: int | None


_DECLARED_CURRENCY_UNITS: Final[Mapping[str, _DeclaredCurrencyUnit]] = MappingProxyType(
    {
        "undata/UNIT_MEASURE-CR_USD": _DeclaredCurrencyUnit("USD", None),
        "undata/UNIT_MEASURE-CR_USD_K": _DeclaredCurrencyUnit("USD", None),
        "ChainedUSDollarBaseYr2017": _DeclaredCurrencyUnit("USD", 2017),
    }
)

type UnitKind = Literal["currency", "purchasing_power", "percent", "other", "none"]
"""`currency`: an ISO 4217 currency. `purchasing_power`: PPP or international dollars.
`percent`: a percentage. `other`: a unit outside these classes, a rate over a stated count
among them. `none`: the series names no unit."""


def is_known_non_currency(unit: str) -> bool:
    """Whether `unit` is a percent or a PPP or international-dollar unit, which no graph
    currency unit can be, so classifying it needs no `graph.currency_units()` read."""
    return isinstance(parse_unit(unit), Percent) or unit in _PURCHASING_POWER_UNITS


def is_per_head_denominator(denominator: str | None) -> bool:
    """Whether a `measurementDenominator` dcid makes the measure a per-person rate."""
    return denominator in PER_HEAD_DENOMINATORS


@dataclass(frozen=True, slots=True)
class UnitClass:
    """The class of a series' unit and its denominator.

    `unit` is the facet unit dcid. `currency` is its ISO 4217 code when `kind` is
    `currency`. `per_head` is true for a per-person rate. `ratio_denominator` is the
    `measurementDenominator` dcid of any other ratio ("% of exports"), which
    `measure_noun.denominator_noun` names. `base_year` is the year of a unit
    in constant prices of that year (`ChainedUSDollarBaseYr2017`), else `None`. `rate` is
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

    @property
    def rate_phrase(self) -> str | None:
        """The rate's base as a heading reads it, "per 1,000 people", else `None`."""
        if self.rate is None:
            return None
        base = f"per {self.rate.count_base:,}"
        return base if self.rate.counted is None else f"{base} {self.rate.counted}"

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


def classify_unit(
    *,
    unit: str | None,
    denominator: str | None,
    currency_units: Mapping[str, str],
) -> UnitClass:
    """Classify a series' unit and denominator.

    Args:
        unit: the facet unit dcid (`ResolvedSpec.facets["unit"].ref`), or `None` when the
            series names none or its series disagree.
        denominator: the shape's `measurementDenominator` dcid, or `None`.
        currency_units: `graph.currency_units()`, unit dcid to ISO 4217 code.

    Returns:
        The unit's class. A rate over a stated count takes the place of the denominator.
        Otherwise the denominator sets `per_head` or `ratio_denominator`, so a per-capita
        amount in Kenyan shillings is a `currency` with `per_head` true. A `Count_Person`
        denominator under a unit that fixes no per-person reading sets neither.
    """
    value = parse_unit(unit) if unit is not None else None
    declared = _DECLARED_CURRENCY_UNITS.get(unit) if unit is not None else None
    currency: str | None = None
    if declared is not None:
        currency = declared.currency
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
    per_head = rate is None and (
        denominator == PER_CAPITA_DENOMINATOR
        or (
            is_per_head_denominator(denominator)
            and (kind not in ("other", "none") or isinstance(value, Money | UnpricedMoney))
        )
    )
    stated = rate is not None or is_per_head_denominator(denominator)
    return UnitClass(
        kind=kind,
        unit=unit,
        currency=currency,
        per_head=per_head,
        ratio_denominator=None if stated else denominator,
        base_year=declared.base_year if declared is not None else None,
        rate=rate,
    )
