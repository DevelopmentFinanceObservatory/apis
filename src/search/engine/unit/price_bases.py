"""The price bases a part can serve and the route to each (`ResolvedSpec.price_bases`).

A part reaches current or constant prices by three routes, each read from a member the Spec
already states:

- `served`: the bases the Spec serves as built, the unit facet's own
  (`ResolvedFacet.prices`), which under a bound unit request are the request's prices;
- `alternative`: a unit option publishing the same measure at another basis, an
  `alternative_measures` entry on the unit axis (`spec_build._menu_siblings`);
- `unit_request`: a basis a currency of the conversion offer converts at least one served
  year to (`ResolvedSpec.unit_conversion`).

So the list names no basis the data cannot serve: WDI's remittances, published in US
dollars with no constant sibling and no recipe, list current prices served alone, and a
DAC table whose recipe deflates lists current served and constant by unit request.
"""

from __future__ import annotations

from typing import Literal

from code_index.labels import PRICE_BASIS_AXIS, PRICES_BY_QUALIFIER, PricesKind
from search.models import IdentityFacetValueRequest, PriceBasisRoute, Spec

type Route = Literal["served", "alternative", "unit_request"]

_ORDER: tuple[PricesKind, ...] = ("current", "constant")


def servable_price_bases(spec: Spec) -> list[PriceBasisRoute] | None:
    """The routes to each price basis `spec`'s part can serve: its served bases
    (`served_prices`), then each unit option's basis, then each basis its conversion offer
    converts to, current before constant within each route. A route to a basis the Spec
    already serves is left out, since taking it changes nothing. Empty when the served
    units are no money. None where the served bases are unknown (`served_prices`), since the
    list would then understate."""
    served = served_prices(spec)
    if served is None:
        return None
    resolved = spec.resolved
    alternatives = {
        PRICES_BY_QUALIFIER[value.value.ref]
        for alternative in resolved.alternative_measures or ()
        if alternative.axis == "unit"
        for value in alternative.measure_axis_values
        if value.axis == PRICE_BASIS_AXIS and value.value.ref in PRICES_BY_QUALIFIER
    }
    conversion = resolved.unit_conversion
    converted: set[PricesKind] = set()
    for currency in conversion.currencies if conversion is not None else ():
        if currency.current_years is not None:
            converted.add("current")
        if currency.constant_years is not None:
            converted.add("constant")
    routes: list[tuple[Route, set[PricesKind]]] = [
        ("served", set(served)),
        ("alternative", alternatives - set(served)),
        ("unit_request", converted - set(served)),
    ]
    return [
        PriceBasisRoute(prices=prices, via=via)
        for via, bases in routes
        for prices in _ORDER
        if prices in bases
    ]


def served_prices(spec: Spec) -> list[PricesKind] | None:
    """The price bases `spec` serves as built: a bound unit request's prices, else the unit
    facet's (`ResolvedFacet.prices`). None where the Spec carries no unit facet and no
    request, or the facet states no prices."""
    unit_binding = (spec.definition.identity.facets or {}).get("unit")
    if isinstance(unit_binding, IdentityFacetValueRequest):
        return [unit_binding.request.prices]
    unit = spec.resolved.facets.get("unit")
    return None if unit is None else unit.prices


def with_price_bases(spec: Spec) -> Spec:
    """A copy of `spec` stating the routes to the price bases its part can serve
    (`servable_price_bases`)."""
    resolved = spec.resolved.model_copy(update={"price_bases": servable_price_bases(spec)})
    return spec.model_copy(update={"resolved": resolved})
