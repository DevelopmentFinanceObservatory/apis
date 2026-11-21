"""The price bases a part can serve (`ResolvedSpec.price_bases`): its own unit's, its unit
options', and those its conversion recipe converts to, and no other."""

from __future__ import annotations

from search.engine.unit.conversion import with_unit_conversion
from search.engine.unit.conversion_catalogue import bound_conversion_catalogue
from search.engine.unit.price_bases import servable_price_bases
from search.models import (
    GraphRef,
    IdentityUnitRequest,
    MeasureAxisValue,
    ResolvedFacet,
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
    _usd,
)
from tests.render._helpers import _ref, _spec

_WDI = "provenance/WorldBankWDI"


def _converted(spec: Spec, *, provenance: str = _CRS) -> Spec:
    """`spec` stating its conversions under the test catalogue, over a plan serving Germany
    from 1990 to 2024 under `provenance`."""
    plan = _plan((1990, 2024), (("observationAbout", "country/DEU"),), provenance=provenance)
    with bound_conversion_catalogue(_CATALOGUE):
        return with_unit_conversion(spec, members=_members(), plans=(plan,))


def _in(unit: str, *, currency: str | None, provenance: str) -> Spec:
    return _spec(
        facets={
            "unit": ResolvedFacet(basis="fallback", ref=unit, currency=currency),
            "provenance": ResolvedFacet(basis="fallback", ref=provenance),
        }
    )


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


def test_a_part_with_no_unit_option_and_no_recipe_serves_current_prices_alone() -> None:
    """WDI's remittances, in US dollars with no constant sibling and no conversion recipe."""
    spec = _converted(_in("USDollar", currency="USD", provenance=_WDI), provenance=_WDI)

    assert spec.resolved.unit_conversion is None
    assert spec.resolved.price_bases == ["current"]


def test_a_part_whose_recipe_deflates_serves_both_bases() -> None:
    spec = _converted(_usd(_CRS, served=(1990, 2024)))

    assert spec.resolved.unit_conversion is not None
    assert spec.resolved.price_bases == ["current", "constant"]


def test_a_unit_option_serves_its_basis_without_a_recipe() -> None:
    spec = _converted(
        _with_constant_option(_in("USDollar", currency="USD", provenance=_NO_RECIPE)),
        provenance=_NO_RECIPE,
    )

    assert spec.resolved.unit_conversion is None
    assert spec.resolved.price_bases == ["current", "constant"]


def test_a_constant_price_series_serves_its_own_basis() -> None:
    """A series in constant prices states its basis by its `measurementQualifier`, and no
    recipe converts it."""
    spec = _spec(
        measurement_qualifier=_ref("RealValue", "Real Value"),
        facets={"unit": ResolvedFacet(basis="fallback", ref="USDollar", currency="USD")},
    )

    assert _converted(spec, provenance=_NO_RECIPE).resolved.price_bases == ["constant"]


def test_an_answer_a_unit_request_converts_serves_the_requested_basis_and_the_native_one() -> None:
    request = IdentityUnitRequest(currency="USD", prices="constant", base_year=2020)
    spec = _converted(_requested(_usd(_CRS, served=(1990, 2024)), request))

    assert spec.resolved.price_bases == ["current", "constant"]


def test_a_unit_that_is_no_money_serves_no_price_basis() -> None:
    spec = _converted(_in("Percent", currency=None, provenance=_WDI), provenance=_WDI)

    assert spec.resolved.price_bases == []


def test_a_unit_request_keeps_the_series_own_basis() -> None:
    """A constant-price series served under a current-price request, which no recipe
    converts, still serves the constant prices it is published in."""
    spec = _spec(
        measurement_qualifier=_ref("RealValue", "Real Value"),
        facets={"unit": ResolvedFacet(basis="fallback", ref="USDollar", currency="USD")},
    )
    request = IdentityUnitRequest(currency="USD", prices="current")

    assert servable_price_bases(_requested(spec, request)) == ["current", "constant"]


def test_an_answer_with_no_catalogue_leaves_its_price_bases_unstated() -> None:
    """With no conversion catalogue loaded the conversion is unknown, so the list would
    understate a DAC part's constant prices."""
    plan = _plan((1990, 2024), (("observationAbout", "country/DEU"),))
    spec = with_unit_conversion(_usd(_CRS, served=(1990, 2024)), members=_members(), plans=(plan,))

    assert spec.resolved.price_bases is None
