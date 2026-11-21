"""The price bases a part can serve (`ResolvedSpec.price_bases`).

A part serves current or constant prices in three ways: the basis of its own unit (the
series' own, and a unit request's), a unit option that publishes the same measure at the other basis (an `alternative_measures` entry
on the unit axis, `spec_build._menu_siblings`), and a unit request its conversion recipe
converts (`ResolvedSpec.unit_conversion`). The list is the union of the three, so it names
no basis the data cannot serve: WDI's remittances, published in US dollars with no constant
sibling and no recipe, list current prices alone, and a DAC table whose recipe deflates
lists both.
"""

from __future__ import annotations

from typing import Literal

from code_index.labels import PRICE_BASIS_AXIS, PRICES_BY_QUALIFIER
from search.models import IdentityFacetValueRequest, ResolvedSpec, Spec
from unit_words.price_basis import price_basis
from unit_words.unit_class import classify_unit

type PriceBasis = Literal["current", "constant"]

_ORDER: tuple[PriceBasis, ...] = ("current", "constant")


def servable_price_bases(spec: Spec) -> list[PriceBasis]:
    """The price bases `spec`'s part can serve, current before constant: a unit request's,
    the served series' own (`_native_prices`), each unit option's, and each a currency of
    its conversion converts at least one served year at. Empty when none states a basis, as
    for a count or a share."""
    resolved = spec.resolved
    bases: set[PriceBasis] = set()
    unit_binding = (spec.definition.identity.facets or {}).get("unit")
    if isinstance(unit_binding, IdentityFacetValueRequest):
        bases.add(unit_binding.request.prices)
    if (native := _native_prices(resolved)) is not None:
        bases.add(native)
    for alternative in resolved.alternative_measures or ():
        if alternative.axis != "unit":
            continue
        bases.update(
            PRICES_BY_QUALIFIER[value.value.ref]
            for value in alternative.measure_axis_values
            if value.axis == PRICE_BASIS_AXIS and value.value.ref in PRICES_BY_QUALIFIER
        )
    conversion = resolved.unit_conversion
    for currency in conversion.currencies if conversion is not None else ():
        if currency.current_years is not None:
            bases.add("current")
        if currency.constant_years is not None:
            bases.add("constant")
    return [basis for basis in _ORDER if basis in bases]


def with_price_bases(spec: Spec) -> Spec:
    """A copy of `spec` stating the price bases its part can serve (`servable_price_bases`)."""
    resolved = spec.resolved.model_copy(update={"price_bases": servable_price_bases(spec)})
    return spec.model_copy(update={"resolved": resolved})


def _native_prices(resolved: ResolvedSpec) -> PriceBasis | None:
    """The basis the served series are published in, whatever unit request converts them:
    the basis the byline's unit token reads from the series (`unit_words.price_basis.
    price_basis`: the unit's own, then the `measurementQualifier` of the shape or the served
    partition), else current prices for a currency unit that states none (`USDollar`), which
    a conversion reads as nominal (`materialise.facets.nominal_currency`). None for a unit
    that is no money."""
    unit = resolved.facets.get("unit")
    if unit is None or unit.ref is None:
        return None
    denominator = resolved.shape.measurement_denominator
    unit_class = classify_unit(
        unit=unit.ref,
        denominator=denominator.ref if denominator is not None else None,
        currency_units={unit.ref: unit.currency} if unit.currency is not None else {},
    )
    if unit_class.currency is None and unit_class.prices is None:
        return None
    basis = price_basis(
        requested=None, unit_class=unit_class, measurement_qualifier=_qualifier(resolved)
    )
    return "current" if basis is None else basis.prices


def _qualifier(resolved: ResolvedSpec) -> str | None:
    """The `measurementQualifier` the shape binds, else the one the served partition holds."""
    qualifier = resolved.shape.measurement_qualifier
    if qualifier is not None:
        return qualifier.ref
    return next(
        (
            value.value.ref
            for value in resolved.measure_axis_values
            if value.axis == PRICE_BASIS_AXIS
        ),
        None,
    )
