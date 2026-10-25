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
- Percent is `Percent` and `SDG_PERCENT`, listed in `_PERCENT_UNITS`.
- A count base is a unit that names the count it divides by, "per 1,000 people" or "per
  100,000 live births", listed with its base in `_COUNT_BASE_UNITS`. A count base is the
  unit's own: the World Development Indicators crude birth rate,
  `Count_BirthEvent_LiveBirth_AsFractionOf_Count_Person`, divides by `Count_Person` and is
  served in `Per1000Persons`, so it is a rate per 1,000 people.
- Per head is a `measurementDenominator` of `Count_Person` (WEO, IDS) or `PerCapita` (base
  Data Commons), listed in `PER_HEAD_DENOMINATORS`, under a unit with no count base.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

PER_CAPITA_DENOMINATOR: Final = "PerCapita"
"""The base Data Commons `measurementDenominator` that frames a measure as per capita. A
`Count_Person` denominator is a per-person rate too, but also the denominator of rates
whose unit names a count base (a crude birth rate in `Per1000Persons`)."""

PER_HEAD_DENOMINATORS: Final = frozenset({"Count_Person", PER_CAPITA_DENOMINATOR})
"""`measurementDenominator` dcids of a per-person rate."""

_PURCHASING_POWER_UNITS: Final = frozenset(
    {"PPPInternationalDollar", "InternationalDollar", "PurchasingPowerStandard"}
)
_PERCENT_UNITS: Final = frozenset({"Percent", "SDG_PERCENT"})
_PLAIN_CURRENCY: Final = "USD"


@dataclass(frozen=True, slots=True)
class CountBase:
    """The count a unit divides by: `size` of `noun`, a plural ("people")."""

    size: int
    noun: str

    @property
    def phrase(self) -> str:
        """The base as a heading reads it: "per 1,000 people"."""
        return f"per {self.size:,} {self.noun}"


_PEOPLE: Final = "people"
_LIVE_BIRTHS: Final = "live births"

_COUNT_BASE_UNITS: Final[Mapping[str, CountBase]] = MappingProxyType(
    {
        "Per1000Persons": CountBase(1000, _PEOPLE),
        "Per100000Persons": CountBase(100_000, _PEOPLE),
        "Per100000Females": CountBase(100_000, "females"),
        "Per100000Males": CountBase(100_000, "males"),
        "Per1000LiveBirths": CountBase(1000, _LIVE_BIRTHS),
        "Per1000FemaleLiveBirths": CountBase(1000, "female live births"),
        "Per1000MaleLiveBirths": CountBase(1000, "male live births"),
        "SDG_PER_1000_POP": CountBase(1000, _PEOPLE),
        "SDG_PER_100000_POP": CountBase(100_000, _PEOPLE),
        "SDG_PER_1000_UNINFECTED_POP": CountBase(1000, "uninfected people"),
        "SDG_PER_1000_LIVE_BIRTHS": CountBase(1000, _LIVE_BIRTHS),
        "SDG_PER_100000_LIVE_BIRTHS": CountBase(100_000, _LIVE_BIRTHS),
        "undata/UNIT_MEASURE-RATIO_COUNT_CASES_PER_COUNT_1000_COUNT_POP": CountBase(1000, _PEOPLE),
        "undata/UNIT_MEASURE-RATIO_COUNT_DEATHS_PER_1000_COUNT_LIVEBIRTHS": CountBase(
            1000, _LIVE_BIRTHS
        ),
        "undata/UNIT_MEASURE-RATE_MORTALITY_PER_1000_COUNT_LIVEBIRTHS": CountBase(
            1000, _LIVE_BIRTHS
        ),
    }
)
"""Units that name the count they divide by, as the World Development Indicators, the
UN SDG indicators and the UN System Data Commons serve them."""


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
`percent`: a percentage. `other`: a unit outside these classes. `none`: the series names
no unit."""


def is_known_non_currency(unit: str) -> bool:
    """Whether `unit` is a percent or a PPP or international-dollar unit, which no graph
    currency unit can be, so classifying it needs no `graph.currency_units()` read."""
    return unit in _PERCENT_UNITS or unit in _PURCHASING_POWER_UNITS


def is_per_head_denominator(denominator: str | None) -> bool:
    """Whether a `measurementDenominator` dcid makes the measure a per-person rate."""
    return denominator in PER_HEAD_DENOMINATORS


def fixes_per_head(denominator: str | None) -> bool:
    """Whether a `measurementDenominator` dcid makes the measure per capita before its unit
    is known. `PerCapita` does. A `Count_Person` rate waits for its unit, which may name a
    count base (`_COUNT_BASE_UNITS`)."""
    return denominator == PER_CAPITA_DENOMINATOR


@dataclass(frozen=True, slots=True)
class UnitClass:
    """The class of a series' unit and its denominator.

    `unit` is the facet unit dcid. `currency` is its ISO 4217 code when `kind` is
    `currency`. `per_head` is true for a per-person rate. `ratio_denominator` is the
    `measurementDenominator` dcid of any other ratio ("% of exports"), which
    `measure_noun.denominator_noun` names. `base_year` is the year of a unit
    in constant prices of that year (`ChainedUSDollarBaseYr2017`), else `None`. `base` is
    the count a unit names as its base ("per 1,000 people"), else `None`; a unit with a
    base sets neither `per_head` nor `ratio_denominator`, since the base is the
    denominator.
    """

    kind: UnitKind
    unit: str | None
    currency: str | None
    per_head: bool
    ratio_denominator: str | None
    base_year: int | None = None
    base: CountBase | None = None

    @property
    def changes_magnitude(self) -> bool:
        """Whether the unit changes what the number means, so the heading states it.

        True for a per-head rate, a percent or other ratio, a unit with a count base, PPP
        or international dollars, and any currency other than US dollars. False for US
        dollars and for units outside these classes, which the byline states.
        """
        if self.per_head or self.ratio_denominator is not None or self.base is not None:
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
        The unit's class. A unit with a count base (`_COUNT_BASE_UNITS`) sets `base`, and
        the denominator is ignored. Otherwise the denominator sets `per_head` or
        `ratio_denominator` whatever the unit's kind, so a per-capita amount in Kenyan
        shillings is a `currency` with `per_head` true.
    """
    base = _COUNT_BASE_UNITS.get(unit) if unit is not None else None
    per_head = base is None and is_per_head_denominator(denominator)
    ratio_denominator = None if per_head or base is not None else denominator
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
    elif unit in _PERCENT_UNITS:
        kind = "percent"
    else:
        kind = "other"
    return UnitClass(
        kind=kind,
        unit=unit,
        currency=currency,
        per_head=per_head,
        ratio_denominator=ratio_denominator,
        base_year=declared.base_year if declared is not None else None,
        base=base,
    )
