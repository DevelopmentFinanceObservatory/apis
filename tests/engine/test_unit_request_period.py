"""The period a unit request shows when its conversion tables cover only part of the years
an answer serves (`convertibility.unit_request_period`), read from the answer's plan and the
loaded conversion catalogue, and the plan read over it (`ReadPlan.windowed`)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import pytest

from materialise.facets import FacetSelection, UnitRequest
from search.engine.clients.graph import Facet
from search.engine.region.convertibility import unit_request_period
from search.engine.retrieval.read_plan import PairPlan, PlanSeries, served_dates
from search.engine.unit.conversion import ConversionPeriod
from search.engine.unit.conversion_catalogue import ConversionCatalogue, RecipeTables
from search.models import IdentityCurrencyRole, IdentityUnitRequest, TimeWindow

_SV = "TEST/Period_Flow"
_CRS = "provenance/OECD_CRS"
_IDS = "provenance/WB_IDS"
_DONOR = "country/DEU"
_RECIPIENT = "country/GTM"
_ROLES = ("financingProvider", "financingRecipient")


def _years(first: int, last: int) -> frozenset[int]:
    return frozenset(range(first, last + 1))


def _tables(*, usd_from: int = 1960, donor_deflator_from: int = 1960) -> RecipeTables:
    return RecipeTables(
        rate_years={
            "EUR": _years(1960, 2031),
            "GTQ": _years(1980, 2031),
            "USD": _years(usd_from, 2031),
        },
        deflator_years={_DONOR: _years(donor_deflator_from, 2026), _RECIPIENT: _years(1980, 2026)},
        base_year_min=1960,
        base_year_max=2025,
    )


def _catalogue(dac: RecipeTables, ids: RecipeTables | None = None) -> ConversionCatalogue:
    return ConversionCatalogue(
        recipes=MappingProxyType({_CRS: "ConversionRecipeDAC", _IDS: "ConversionRecipeIDS"}),
        tables=MappingProxyType(
            {"ConversionRecipeDAC": dac, "ConversionRecipeIDS": ids or _tables(usd_from=1980)}
        ),
        national_currencies=MappingProxyType({_DONOR: "EUR", _RECIPIENT: "GTQ"}),
        default_base_year=2024,
    )


def _plan(first: int, last: int, *, provenance: str = _CRS) -> PairPlan:
    """German dollar flows to Guatemala, one point a year from `first` to `last`."""
    years = [str(year) for year in range(first, last + 1)]
    facet = Facet(
        obs_count=len(years),
        earliest_date=years[0],
        latest_date=years[-1],
        facet_id="f1",
        provenance_id=provenance,
        unit="CurrencyUSD",
        dates=tuple(years),
    )
    series = PlanSeries(
        stat_var=_SV,
        role_tuple=(("financingProvider", _DONOR), ("financingRecipient", _RECIPIENT)),
        candidates=(facet,),
        group=(facet,),
    )
    return PairPlan(members=(_SV,), years=None, series=(series,))


def _elected(
    request: IdentityUnitRequest | None, provenance: str = _CRS
) -> dict[str, tuple[Mapping[str, FacetSelection], ...]]:
    """`_SV`'s dollar facet, converted under `request`, or read as it is with None."""

    def selection(axis: str, ref: str, unit: UnitRequest | None = None) -> FacetSelection:
        return FacetSelection(
            axis=axis,
            ref=ref,
            basis="bound",
            default_policy=None,
            default_policy_version=None,
            selection_reason=None,
            dispatch_facet_id="f1",
            request=unit,
        )

    unit = (
        UnitRequest(currency="GTQ", prices=request.prices, base_year=request.base_year)
        if request is not None
        else None
    )
    return {
        _SV: (
            {
                "unit": selection("unit", "CurrencyUSD", unit),
                "provenance": selection("provenance", provenance),
            },
        )
    }


def _period(
    plan: PairPlan,
    request: IdentityUnitRequest,
    catalogue: ConversionCatalogue,
    *,
    provenance: str = _CRS,
    converts: bool = True,
) -> ConversionPeriod | None:
    return unit_request_period(
        plan,
        request,
        _elected(request if converts else None, provenance),
        declared_roles=_ROLES,
        catalogue=catalogue,
    )


_GTQ = IdentityUnitRequest(currency="GTQ", prices="current")
_EUR_2024 = IdentityUnitRequest(currency="EUR", prices="constant", base_year=2024)


@pytest.mark.parametrize(
    ("plan", "request_", "catalogue", "provenance", "expected"),
    [
        pytest.param(
            _plan(1976, 2024),
            _GTQ,
            _catalogue(_tables()),
            _CRS,
            ConversionPeriod(served=(1976, 2024), years=(1980, 2024), limited_by="exchange_rates"),
            id="target-rates-start-later",
        ),
        pytest.param(
            _plan(1976, 2024),
            IdentityUnitRequest(
                currency=IdentityCurrencyRole(role="destination"), prices="current"
            ),
            _catalogue(_tables()),
            _CRS,
            ConversionPeriod(served=(1976, 2024), years=(1980, 2024), limited_by="exchange_rates"),
            id="role-currency",
        ),
        pytest.param(
            _plan(1970, 2024, provenance=_IDS),
            _EUR_2024,
            _catalogue(_tables()),
            _IDS,
            ConversionPeriod(served=(1970, 2024), years=(1980, 2024), limited_by="exchange_rates"),
            id="native-rates-start-later",
        ),
        pytest.param(
            _plan(1970, 2024),
            _EUR_2024,
            _catalogue(_tables(donor_deflator_from=1990)),
            _CRS,
            ConversionPeriod(served=(1970, 2024), years=(1990, 2024), limited_by="deflators"),
            id="donor-deflator-starts-later",
        ),
        pytest.param(_plan(1990, 2024), _GTQ, _catalogue(_tables()), _CRS, None, id="covered"),
        pytest.param(
            _plan(1960, 1975), _GTQ, _catalogue(_tables()), _CRS, None, id="nothing-covered"
        ),
        pytest.param(_plan(1976, 2024), _GTQ, None, _CRS, None, id="no-catalogue"),
        pytest.param(
            _plan(1976, 2024),
            IdentityUnitRequest(currency="GTQ", prices="constant", base_year=1975),
            _catalogue(_tables()),
            _CRS,
            None,
            id="no-base-year-rate",
        ),
    ],
)
def test_the_period_is_the_longest_run_every_needed_table_covers(
    plan, request_, catalogue, provenance, expected
):
    assert _period(plan, request_, catalogue, provenance=provenance) == expected


def test_a_natively_satisfied_facet_reads_no_table():
    assert _period(_plan(1976, 2024), _GTQ, _catalogue(_tables()), converts=False) is None


def test_the_plan_read_over_the_period_serves_its_years():
    period = _period(_plan(1976, 2024), _GTQ, _catalogue(_tables()))
    assert period is not None
    assert period.window == TimeWindow(start_year=1980, end_year=2024)
    assert served_dates(_plan(1976, 2024).windowed(period.window)) == {_SV: ("1980", "2024")}
