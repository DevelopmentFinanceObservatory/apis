"""The price bases a part can serve and the route to each (`ResolvedSpec.price_bases`): the
bases its unit facet serves, its unit options' and those its conversion recipe converts
to, and no other. The unit facet states its own bases (`ResolvedFacet.prices`)."""

from __future__ import annotations

from typing import Literal

import pytest

from search.engine.region.facets import _resolved_facets
from search.engine.same_shape import _merged_prices
from search.engine.unit.conversion import with_unit_conversion
from search.engine.unit.conversion_catalogue import bound_conversion_catalogue
from search.engine.unit.price_bases import servable_price_bases
from search.engine.unit.refs import UnitPrices
from search.models import (
    GraphRef,
    IdentityUnitRequest,
    MeasureAxisValue,
    PriceBasisRoute,
    ResolvedFacet,
    SeriesFacetRef,
    Spec,
    SpecAlternativeMeasure,
)
from tests.engine.test_unit_conversion_offer import (
    _CATALOGUE,
    _CRS,
    _NO_RECIPE,
    _members,
    _plan,
    _requested,
)
from tests.render._helpers import _spec

_WDI = "provenance/WorldBankWDI"
type _Prices = list[Literal["current", "constant"]] | None


def _in(
    unit: str = "CurrencyUSD",
    *,
    prices: _Prices,
    stated: _Prices | Literal["all"] = "all",
    provenance: str = _CRS,
) -> Spec:
    """A Spec served in `unit`, whose facet states `prices`, of them `stated` as the data
    states them (every one by default), from `provenance`."""
    return _spec(
        facets={
            "unit": ResolvedFacet(
                basis="fallback",
                ref=unit,
                currency=None if unit == "Percent" else "USD",
                prices=prices,
                stated_prices=prices if stated == "all" else stated,
            ),
            "provenance": ResolvedFacet(basis="fallback", ref=provenance),
        }
    )


def _converted(spec: Spec, *, provenance: str = _CRS) -> Spec:
    """`spec` stating its conversions under the test catalogue, over a plan serving Germany
    from 1990 to 2024 under `provenance`."""
    plan = _plan((1990, 2024), (("observationAbout", "country/DEU"),), provenance=provenance)
    with bound_conversion_catalogue(_CATALOGUE):
        return with_unit_conversion(spec, members=_members(), plans=(plan,))


def _with_constant_option(spec: Spec) -> Spec:
    """`spec` with a unit option at constant prices, as a table publishing both bases has."""
    option = SpecAlternativeMeasure(
        definition=spec.definition,
        label="constant prices",
        axis="unit",
        measure_axis_values=[
            MeasureAxisValue(
                axis="measurementQualifier", value=GraphRef(ref="RealValue", label="Real Value")
            )
        ],
    )
    return spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"alternative_measures": [option]})}
    )


def _routes(*pairs: tuple[str, str]) -> list[PriceBasisRoute]:
    return [PriceBasisRoute.model_validate({"prices": p, "via": v}) for p, v in pairs]


def test_a_part_with_no_unit_option_and_no_recipe_serves_current_prices_alone() -> None:
    """WDI's remittances, in US dollars with no constant sibling and no conversion recipe.
    The current prices the engine infers for an unpriced currency are served all the same."""
    spec = _converted(
        _in("USDollar", prices=["current"], stated=[], provenance=_WDI), provenance=_WDI
    )

    assert spec.resolved.unit_conversion is None
    assert spec.resolved.price_bases == _routes(("current", "served"))


def test_a_part_whose_recipe_deflates_reaches_constant_prices_by_unit_request() -> None:
    """The control: a DAC table whose recipe converts. Current prices by unit request are
    left out, since the Spec already serves them."""
    spec = _converted(_in(prices=["current"]))

    assert spec.resolved.unit_conversion is not None
    assert spec.resolved.price_bases == _routes(("current", "served"), ("constant", "unit_request"))


def test_a_unit_option_reaches_its_basis_without_a_recipe() -> None:
    spec = _converted(
        _with_constant_option(_in(prices=["current"], provenance=_NO_RECIPE)),
        provenance=_NO_RECIPE,
    )

    assert spec.resolved.price_bases == _routes(("current", "served"), ("constant", "alternative"))


def test_an_answer_a_unit_request_converts_serves_the_requested_basis() -> None:
    """A Spec bound to constant prices by unit request serves them as built, and reaches
    current prices by a unit request at current prices."""
    request = IdentityUnitRequest(currency="USD", prices="constant", base_year=2020)
    spec = _converted(_requested(_in(prices=["constant"]), request))

    assert spec.resolved.price_bases == _routes(("constant", "served"), ("current", "unit_request"))


@pytest.mark.parametrize(
    ("prices", "routes"),
    [
        pytest.param([], [], id="no-money"),
        pytest.param(None, None, id="unknown"),
    ],
)
def test_a_facet_stating_no_basis_lists_none_and_one_stating_nothing_is_unknown(
    prices: _Prices, routes: list[PriceBasisRoute] | None
) -> None:
    assert servable_price_bases(_in("Percent", prices=prices, provenance=_WDI)) == routes


def test_an_answer_with_no_catalogue_leaves_its_price_bases_unstated() -> None:
    """With no conversion catalogue loaded the conversion is unknown, so the list would
    understate a DAC part's constant prices."""
    plan = _plan((1990, 2024), (("observationAbout", "country/DEU"),))
    spec = with_unit_conversion(_in(prices=["current"]), members=_members(), plans=(plan,))

    assert spec.resolved.price_bases is None


# An unpriced currency's current prices are inferred; a registry-priced unit states its own.
_PRICES: dict[str, UnitPrices | None] = {
    "CurrencyUSD": UnitPrices("current", stated=False),
    "SDG_CU_USD": UnitPrices("current", stated=True),
    "ChainedUSDollarBaseYr2017": UnitPrices("constant", stated=True),
    "Percent": None,
}


@pytest.mark.parametrize(
    ("units", "requested", "prices", "stated"),
    [
        pytest.param(["CurrencyUSD"], None, ["current"], [], id="one-unit"),
        pytest.param(["SDG_CU_USD"], None, ["current"], ["current"], id="one-priced-unit"),
        pytest.param(
            ["CurrencyUSD", "ChainedUSDollarBaseYr2017"],
            None,
            ["current", "constant"],
            ["constant"],
            id="per-series-units",
        ),
        # A basis one unit states and another infers is inferred.
        pytest.param(
            ["SDG_CU_USD", "CurrencyUSD"], None, ["current"], [], id="stated-beside-inferred"
        ),
        pytest.param(["Percent", "CurrencyUSD"], None, ["current"], [], id="beside-no-money"),
        pytest.param(["Percent"], None, [], [], id="no-money"),
        # A series naming no unit leaves the bases unknown.
        pytest.param(["CurrencyUSD", None], None, None, None, id="a-series-with-no-unit"),
        # A unit request states its own prices whatever the units.
        pytest.param(["CurrencyUSD"], "constant", ["constant"], ["constant"], id="unit-request"),
    ],
)
def test_the_unit_facet_states_the_bases_of_every_served_unit(
    units: list[str | None],
    requested: Literal["current", "constant"] | None,
    prices: _Prices,
    stated: _Prices,
) -> None:
    maps = [{"unit": SeriesFacetRef(ref=unit)} if unit is not None else {} for unit in units]

    facets = _resolved_facets(maps, {}, prices_of=_PRICES.get, requested_prices=requested)

    assert (facets["unit"].prices, facets["unit"].stated_prices) == (prices, stated)


def _unit_facet(prices: _Prices, stated: _Prices) -> ResolvedFacet:
    return ResolvedFacet(basis="fallback", ref="CurrencyUSD", prices=prices, stated_prices=stated)


@pytest.mark.parametrize(
    ("members", "every_member", "merged"),
    [
        pytest.param(
            [(["current"], ["current"]), (["constant"], ["constant"])],
            True,
            (["current", "constant"], ["current", "constant"]),
            id="each-stated",
        ),
        # A basis one member infers is inferred in the merge, though another states it.
        pytest.param(
            [(["current"], ["current"]), (["current"], [])],
            True,
            (["current"], []),
            id="stated-beside-inferred",
        ),
        pytest.param([(["current"], ["current"]), (None, None)], True, (None, None), id="unknown"),
        pytest.param([(["current"], ["current"])], False, (None, None), id="a-member-unitless"),
    ],
)
def test_a_merged_unit_facet_states_a_basis_no_member_infers(members, every_member, merged):
    facets = [_unit_facet(prices, stated) for prices, stated in members]

    assert _merged_prices(facets, every_member=every_member) == merged
