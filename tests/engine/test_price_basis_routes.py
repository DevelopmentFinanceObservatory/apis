"""Each route `ResolvedSpec.price_bases` lists reaches its basis, and a unit request at a basis
the list leaves out refuses. The fixture is a convertible pair of series under the DAC
recipe (`test_resubmit_promote_elected_unit_narrowing`), the control a known-convertible DAC
answer: with Kenya's deflator loaded it reaches constant prices by unit request, and with a
deflator for Uganda alone it lists current prices alone and a constant-price request
refuses."""

from __future__ import annotations

from typing import Literal

import pytest

from search.engine.region.result import RegionResult
from search.engine.unit.conversion_catalogue import bound_conversion_catalogue
from search.engine.unit.price_bases import served_prices
from search.models import (
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    PriceBasisRoute,
    ResubmitIdentityPayload,
)
from tests.engine.test_resubmit_promote_elected_unit_narrowing import (
    _ENTITY,
    _UGANDA,
    _YEAR,
    _catalogue,
    _request,
)

_BASE_YEAR = int(_YEAR)
_DEFLATORS = {_ENTITY: {_BASE_YEAR: 100.0}}
_ELSEWHERE = {_UGANDA: {_BASE_YEAR: 100.0}}
"""A deflator table holding Uganda alone, so Kenya's rows convert to no constant price."""


def _resolve(prices: Literal["current", "constant"] | None, *, deflated: bool) -> RegionResult:
    """The fixture resubmitted with no unit request (`prices` None) or with a US dollar
    request at `prices`, under a catalogue holding deflators when `deflated`."""
    fixture = _request(currency="USD", deflators=_DEFLATORS if deflated else _ELSEWHERE)
    facets = (
        None
        if prices is None
        else {
            "unit": IdentityFacetValueRequest(
                request=IdentityUnitRequest(
                    currency="USD",
                    prices=prices,
                    base_year=_BASE_YEAR if prices == "constant" else None,
                )
            )
        }
    )
    resubmit = fixture.__class__(
        inp=ResubmitIdentityPayload(
            identity=fixture.inp.identity.model_copy(update={"facets": facets})
        ),
        graph=fixture.graph,
        sdmx=fixture.sdmx,
        coverage=fixture.coverage,
        options=fixture.options,
    )
    catalogue = _catalogue(
        {"USD": frozenset({_BASE_YEAR}), "EUR": frozenset({_BASE_YEAR})},
        deflator_years={(_ENTITY if deflated else _UGANDA): frozenset({_BASE_YEAR})},
    )
    with bound_conversion_catalogue(catalogue):
        return resubmit.resolve()


def _routes(region: RegionResult) -> list[PriceBasisRoute]:
    assert region.status == "definite", region.no_data_reason
    routes = region.specs[0].resolved.price_bases
    assert routes is not None
    return routes


@pytest.mark.parametrize("deflated", [True, False], ids=["deflated", "no-deflators"])
def test_each_listed_route_reaches_its_basis_and_an_omitted_basis_refuses(
    deflated: bool,
) -> None:
    routes = _routes(_resolve(None, deflated=deflated))
    listed = {route.prices for route in routes}

    # The control: deflators make constant prices reachable by unit request.
    assert (PriceBasisRoute(prices="constant", via="unit_request") in routes) == deflated
    for route in routes:
        if route.via == "served":
            continue
        assert route.via == "unit_request", route
        reached = _resolve(route.prices, deflated=deflated)
        assert reached.status == "definite", (route, reached.no_data_reason)
        assert served_prices(reached.specs[0]) == [route.prices]
        assert PriceBasisRoute(prices=route.prices, via="served") in _routes(reached)
    bases: tuple[Literal["current", "constant"], ...] = ("current", "constant")
    omitted = [prices for prices in bases if prices not in listed]
    assert omitted == ([] if deflated else ["constant"])
    for prices in omitted:
        assert _resolve(prices, deflated=deflated).status != "definite", prices
