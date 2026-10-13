"""IMF World Economic Outlook nominal GDP as the dev graph holds it, served by fakes.

`fixtures/weo_gdp_2016_2030.json` records the variable's facets and observations at France,
Kenya, Zimbabwe and Ecuador, and each place's `nationalCurrencyOf` arcs, read from the dev
graph on 2026-10-11. Each place holds four facets under one provenance: reported values with
no stated method and `Estimated` values, each in US dollars and in the place's own currency.
Ecuador's own currency is the US dollar, so it holds two.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import httpx

from dc_sdmx import SdmxClient
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_group,
    materialise_preview,
    materialise_spec,
)
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.facets import FacetBinding
from materialise.graph import Facet
from materialise.pipeline import MaterialisationResult, PreviewResult
from materialise.rate_tables import RateTableCache
from materialise.window import RequestedWindow

from ._rate_table_support import RateTableServer, conversion_arcs, recipe_table_bodies
from .conftest import FIXTURES_DIR, FakeGraphReader, only_chunk

GDP = "ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_Nominal"
WEO = "provenance/IMFWorldEconomicOutlook"
FRANCE = "country/FRA"
KENYA = "country/KEN"
ZIMBABWE = "country/ZWE"
ECUADOR = "country/ECU"
PLACES = (FRANCE, KENYA, ZIMBABWE, ECUADOR)
ESTIMATED = "Estimated"

type Values = dict[tuple[str, str], dict[str, float]]
"""Observations by `(place, facet id)`, then year."""

_IMF_RECIPE = "ConversionRecipeIMF"
"""The conversion recipe WEO's provenance names, which converts its nominal GDP."""

type Rates = Mapping[str, Mapping[int, float]]
"""Exchange rates by currency dcid, then year, in units of the currency per US dollar."""

_NONE = "NotApplicable"

_BASE_URL = "https://ours.example.org"
"""The SDMX data route's cell for a dimension a facet declares no value on."""

_SDMX_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


@cache
def _recorded() -> Any:
    """The fixture as JSON. Its shape is fixed, so readers index it directly."""
    return json.loads((FIXTURES_DIR / "weo_gdp_2016_2030.json").read_text())


def recorded_facets() -> dict[str, list[Facet]]:
    """Each place's facets, with the whole series' span and count."""
    recorded = _recorded()
    by_place: dict[str, list[Facet]] = {}
    for series in recorded["series"]:
        meta = recorded["facets"][series["facet"]]
        by_place.setdefault(series["place"], []).append(
            Facet(
                facet_id=series["facet"],
                provenance_id=meta["provenanceId"],
                measurement_method=meta["measurementMethod"],
                unit=meta["unit"],
                obs_count=series["count"],
                earliest_date=series["earliest"],
                latest_date=series["latest"],
            )
        )
    return by_place


def recorded_values() -> Values:
    """Each recorded series' observations from 2016 to 2030."""
    return {
        (series["place"], series["facet"]): {
            year: float(value) for year, value in series["observations"].items()
        }
        for series in _recorded()["series"]
    }


def recorded_currencies() -> dict[str, tuple[str, ...]]:
    """The currencies whose `nationalCurrencyOf` names each place."""
    return {place: tuple(units) for place, units in _recorded()["national_currency_of"].items()}


def facet(place: str, *, unit: str, method: str | None) -> Facet:
    """The recorded facet at `place` in `unit` with `method`."""
    (found,) = [
        candidate
        for candidate in recorded_facets()[place]
        if candidate.unit == unit and candidate.measurement_method == method
    ]
    return found


def facet_id(place: str, *, unit: str, method: str | None) -> str:
    """The id of the recorded facet at `place` in `unit` with `method`."""
    found = facet(place, unit=unit, method=method).facet_id
    assert found is not None
    return found


def graph(
    facets_by_place: Mapping[str, list[Facet]],
    *,
    currencies: Mapping[str, Sequence[str]] | None = None,
    variables: Sequence[str] = (GDP,),
    rates: Rates | None = None,
    nominal: Sequence[str] | None = None,
) -> FakeGraphReader:
    """A graph holding the facets under each of `variables`, each currency typed
    `CurrencyUnit`, and each currency's `nationalCurrencyOf` arcs from `currencies` (the
    recorded ones by default). Given `rates`, WEO names the IMF recipe and each of `nominal`
    (`variables` by default) is a nominal GDP it converts, so a requested currency among
    `rates` converts."""
    named: dict[str, list[str]] = {}
    for place, units in (recorded_currencies() if currencies is None else currencies).items():
        for unit in units:
            named.setdefault(unit, []).append(place)
    units = {
        candidate.unit
        for facets in facets_by_place.values()
        for candidate in facets
        if candidate.unit is not None
    } | set(named)
    arcs = {
        unit: {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": tuple(named.get(unit, ()))}
        for unit in sorted(units)
    }
    if rates is not None:
        recipe = conversion_arcs(recipe_id=_IMF_RECIPE, provenance_id=WEO, currencies=rates)
        arcs = recipe | arcs
        for variable in variables if nominal is None else nominal:
            arcs[variable] = {
                "measurementQualifier": ("Nominal",),
                "populationType": ("EconomicProduction",),
            }
    return FakeGraphReader(
        facets={
            (variable, place): list(facets)
            for variable in variables
            for place, facets in facets_by_place.items()
        },
        arcs=arcs,
    )


def sdmx(
    facets_by_place: Mapping[str, list[Facet]],
    values: Values,
    *,
    variables: Sequence[str] = (GDP,),
    rates: Rates | None = None,
) -> SdmxClient:
    """Serves every recorded observation of the facets, variables and places a read names,
    each of `variables` holding the same observations, and the default place scope's
    availability from the same pairs. Given `rates`, it serves the IMF recipe's exchange
    rate table from them."""
    availability = graph(facets_by_place, currencies={}, variables=variables).scope_availability()
    by_id = {
        candidate.facet_id: candidate for facets in facets_by_place.values() for candidate in facets
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if (response := availability.respond(request)) is not None:
            return response
        wanted = set(str(request.url.params.get("c[facetId]", "")).split(","))
        places = str(request.url.params.get("c[observationAbout]", "")).split(",")
        named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
        rows = [
            f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{place},{by_id[facet_id].unit or _NONE},"
            f"{by_id[facet_id].measurement_method or _NONE},P1Y,{by_id[facet_id].provenance_id},{year},{value!r},,"
            f"{facet_id}\r\n"
            for variable in variables
            if variable in named
            for (place, facet_id), series in values.items()
            if place in places and facet_id in wanted and facet_id in by_id
            for year, value in series.items()
        ]
        return httpx.Response(
            200,
            text=_SDMX_HEADER + "".join(rows),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    if rates is not None:
        tables = recipe_table_bodies(_IMF_RECIPE, rates=rates)
        return RateTableServer(tables, fallback=handler).client(base_url=_BASE_URL)
    return SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))


def _request(
    places: Sequence[str],
    *,
    window: RequestedWindow,
    facet_bindings: Mapping[str, FacetBinding] | None,
    variable: str = GDP,
) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{variable}",
        role_bindings={"observationAbout": list(places)},
        declared_roles=("observationAbout",),
        facet_bindings=dict(facet_bindings or {}),
        window=window,
    )


def preview(
    places: Sequence[str],
    *,
    window: RequestedWindow,
    facet_bindings: Mapping[str, FacetBinding] | None = None,
    facets_by_place: Mapping[str, list[Facet]] | None = None,
    values: Values | None = None,
    currencies: Mapping[str, Sequence[str]] | None = None,
) -> PreviewResult:
    """A Datasets preview of WEO GDP at `places`, over the recorded data by default. It
    reports each axis's selection."""
    facets_by_place = recorded_facets() if facets_by_place is None else facets_by_place
    return materialise_preview(
        _request(places, window=window, facet_bindings=facet_bindings),
        sdmx=sdmx(facets_by_place, recorded_values() if values is None else values),
        graph=graph(facets_by_place, currencies=currencies),
        limit=1000,
    )


def mint(
    places: Sequence[str],
    *,
    window: RequestedWindow,
    facet_bindings: Mapping[str, FacetBinding] | None = None,
    facets_by_place: Mapping[str, list[Facet]] | None = None,
    values: Values | None = None,
    currencies: Mapping[str, Sequence[str]] | None = None,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    rates: Rates | None = None,
) -> MaterialisationResult:
    """A Datasets mint of WEO GDP at `places`, over the recorded data by default. It
    carries the trace and the series figures a description is built from. Given `rates`,
    a requested currency converts under the IMF recipe at those rates."""
    facets_by_place = recorded_facets() if facets_by_place is None else facets_by_place
    reader = graph(facets_by_place, currencies=currencies, rates=rates)
    return materialise_spec(
        _request(places, window=window, facet_bindings=facet_bindings),
        sdmx=sdmx(facets_by_place, recorded_values() if values is None else values, rates=rates),
        graph=reader,
        enrichment=EnrichmentLookup(reader, parties={}),
        materialiser_build="build-test",
        capability_version="cap-test",
        ceiling=ceiling,
        rate_cache=None if rates is None else RateTableCache(),
    )


def mint_group(
    places: Sequence[str],
    *,
    variables: Sequence[str],
    holding: Sequence[str],
    window: RequestedWindow,
    facet_bindings: Mapping[str, FacetBinding] | None = None,
    facets_by_place: Mapping[str, list[Facet]] | None = None,
    values: Values | None = None,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    rates: Rates | None = None,
) -> MaterialisationResult:
    """A Datasets group mint of `variables` at `places`, fetched in one read. Each of
    `holding` holds GDP's recorded facets and observations, over the recorded data by
    default; the other variables hold nothing. Given `rates`, a requested currency converts
    under the IMF recipe at those rates."""
    facets_by_place = recorded_facets() if facets_by_place is None else facets_by_place
    reader = graph(facets_by_place, variables=holding, rates=rates, nominal=variables)
    return materialise_group(
        [
            _request(places, window=window, facet_bindings=facet_bindings, variable=variable)
            for variable in variables
        ],
        sdmx=sdmx(
            facets_by_place,
            recorded_values() if values is None else values,
            variables=holding,
            rates=rates,
        ),
        graph=reader,
        enrichment=EnrichmentLookup(reader, parties={}),
        materialiser_build="build-test",
        capability_version="cap-test",
        ceiling=ceiling,
        member_concurrency=1,
        rate_cache=None if rates is None else RateTableCache(),
    )


def rows(result: PreviewResult | MaterialisationResult) -> list[dict[str, Any]]:
    """The read's rows as mappings from column to cell."""
    frame = result.frame if isinstance(result, PreviewResult) else only_chunk(result)
    return [dict(zip(frame.columns, row, strict=True)) for row in frame.rows]
