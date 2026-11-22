"""Each route `ResolvedSpec.price_bases` lists reaches its basis, and a unit request at a basis
the list leaves out refuses. The fixture is a convertible pair of series under the DAC
recipe (`test_resubmit_promote_elected_unit_narrowing`), the control a known-convertible DAC
answer: with Kenya's deflator loaded it reaches constant prices by unit request, and with a
deflator for Uganda alone it lists current prices alone and a constant-price request
refuses. The fixture's members, republished in real terms, give the unit option a series to
serve."""

from __future__ import annotations

from typing import Literal

import pytest

import search.engine.spec_build as spec_build_module
import tests.engine.test_resubmit_promote_elected_unit_narrowing as narrowing
from code_index.types import MeasureAxis, MeasureIndex, MeasurePartitionLabel, MeasureValueRow
from search.engine.region.result import RegionResult
from search.engine.retrieval.bound_values import MeasurePartition
from search.engine.spec_build import _measure_alternatives
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
    _SV_A,
    _SV_B,
    _UGANDA,
    _YEAR,
    _catalogue,
    _request,
)
from tests.materialise._rate_table_support import CONVERTIBLE_STAT_VAR_ARCS

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


_MEASURED = "testConvertMultiExpenditure"
_PRICED_INDEX = MeasureIndex(
    axes=(
        MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=(_MEASURED,)),
        MeasureAxis(
            axis="measurementQualifier", kind="shape", role="free", prefer=("Nominal", "RealValue")
        ),
    ),
    values=tuple(
        MeasureValueRow(
            dcid=dcid, axis=axis, label=label, synonyms=(), short_name=None, matchable=True
        )
        for dcid, axis, label in (
            (_MEASURED, "measuredProperty", "expenditure"),
            ("Nominal", "measurementQualifier", "nominal"),
            ("RealValue", "measurementQualifier", "real terms"),
        )
    ),
    partition_labels=(
        MeasurePartitionLabel(axis_values=(("measuredProperty", _MEASURED),), label="expenditure"),
    ),
)
"""A measure index whose one measure the fixture publishes at current and constant prices."""


def _partition(qualifier: str) -> MeasurePartition:
    return MeasurePartition(
        values=(("measuredProperty", _MEASURED), ("measurementQualifier", qualifier)),
        members=(_SV_A, _SV_B),
    )


def test_the_unit_option_beside_a_unit_request_serves_the_basis_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Spec serving current prices by unit request lists constant prices by its unit
    option. Resubmitting the option's definition, as Search emits it, serves constant
    prices from the fixture's members published in real terms."""
    monkeypatch.setattr(spec_build_module, "get_measure_index", lambda: _PRICED_INDEX)
    elected = _request(currency="USD").inp.identity
    alternatives, _ = _measure_alternatives(
        elected, _partition("Nominal"), (_partition("RealValue"),)
    )
    assert alternatives is not None
    (option,) = alternatives
    assert option.axis == "unit"
    assert [(value.axis, value.value.ref) for value in option.measure_axis_values][-1] == (
        "measurementQualifier",
        "RealValue",
    )

    real = {"dcid": "RealValue"}
    monkeypatch.setattr(
        narrowing,
        "_NODES",
        {
            dcid: (
                {**node, "arcs": {**node["arcs"], "measurementQualifier": {"nodes": [real]}}}
                if "arcs" in node
                else node
            )
            for dcid, node in narrowing._NODES.items()
        },
    )
    monkeypatch.setattr(
        narrowing,
        "CONVERTIBLE_STAT_VAR_ARCS",
        {**CONVERTIBLE_STAT_VAR_ARCS, "measurementQualifier": ("RealValue",)},
    )
    fixture = _request(currency="USD")
    resubmit = fixture.__class__(
        inp=ResubmitIdentityPayload(identity=option.definition.identity),
        graph=fixture.graph,
        sdmx=fixture.sdmx,
        coverage=fixture.coverage,
        options=fixture.options,
    )
    region = resubmit.resolve()

    assert region.status == "definite", region.no_data_reason
    (spec,) = region.specs
    assert spec.spec_id == option.definition.spec_id
    assert served_prices(spec) == ["constant"]
