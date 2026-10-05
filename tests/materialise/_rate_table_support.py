"""SDMX and graph fixtures for currency conversion, shared by the materialise,
engine and datasets suites.

`RateTableServer` serves whole rate and deflator tables the way
`materialise.rate_tables.load_table` reads them: `/availability` lists a
StatVar's `observationAbout` entities as SDMX-JSON, `/data` returns its rows
as SDMX-CSV. A `fallback` handler answers every other StatVar, so one
transport serves a test's observations and its conversion tables.

`recipe_table_bodies` builds a recipe's two table bodies from plain values,
and `conversion_arcs` the graph arcs a conversion reads: the provenance's
`conversionRecipe`, the table provenance's `lastDataRefreshDate`, and each
currency node's `typeOf` and `nationalCurrencyOf`.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Collection, Mapping, Sequence

import httpx

from dc_sdmx import SdmxClient
from materialise.recipe_bodies import RECIPE_BODIES

SDMX_BASE_URL = "https://datacommons.one.org/core/api"

DEFAULT_REFRESH_DATE = "2026-01-01T00:00:00Z"

CONVERTIBLE_STAT_VAR_ARCS: Mapping[str, tuple[str, ...]] = {
    "measurementQualifier": ("Nominal",),
    "populationType": ("FinancialFlow",),
}
"""The arcs a StatVar needs to convert under the DAC or IDS recipe: a
positive `Nominal` qualifier and a population type the recipe lists."""

RATE_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

_PROJECTION_METHOD = "IMFProjection"
_DATA_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_AVAILABILITY_JSON_CONTENT_TYPE = "application/vnd.sdmx.structure+json;version=2.0.0"

type Handler = Callable[[httpx.Request], httpx.Response]


def rate_row(
    *,
    stat_var: str,
    entity: str,
    year: str,
    value: str,
    provenance: str,
    method: str = "NotApplicable",
    observation_period: str = "P1Y",
    facet_id: str = "f1",
) -> str:
    """One SDMX-CSV data row about `entity` under the subject-only shape."""
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{entity},NotApplicable,{method},"
        f"{observation_period},{provenance},{year},{value},,{facet_id}\r\n"
    )


def rate_csv(*rows: str) -> str:
    return RATE_CSV_HEADER + "".join(rows)


def recipe_table_bodies(
    recipe_id: str,
    *,
    rates: Mapping[str, Mapping[int, float]],
    deflators: Mapping[str, Mapping[int, float]] | None = None,
    projected: Collection[tuple[str, int]] = (),
) -> dict[str, str]:
    """`RateTableServer` bodies for the tables `RECIPE_BODIES[recipe_id]` reads.

    `rates` maps a currency dcid (`CurrencyEUR`) to its value per year, in
    units of the currency per US dollar. The loader requires a `CurrencyUSD`
    row of 1.0 for every year it serves. `deflators` maps a country or
    aggregate dcid to its deflator per year, and is omitted for current-price
    tests. Each `(dcid, year)` in `projected` is served as an
    `IMFProjection` row.
    """
    recipe = RECIPE_BODIES[recipe_id]

    def body(stat_var: str, series: Mapping[str, Mapping[int, float]]) -> str:
        return rate_csv(
            *(
                rate_row(
                    stat_var=stat_var,
                    entity=dcid,
                    year=str(year),
                    value=repr(value),
                    provenance=recipe.table_provenance,
                    method=_PROJECTION_METHOD if (dcid, year) in projected else "NotApplicable",
                    facet_id="projected" if (dcid, year) in projected else "actual",
                )
                for dcid, by_year in series.items()
                for year, value in by_year.items()
            )
        )

    bodies = {recipe.exchange_rate_variable: body(recipe.exchange_rate_variable, rates)}
    if deflators is not None:
        bodies[recipe.deflator_variable] = body(recipe.deflator_variable, deflators)
    return bodies


def conversion_arcs(
    *,
    recipe_id: str,
    provenance_id: str,
    currencies: Mapping[str, Sequence[str]] | Collection[str],
    refresh_date: str = DEFAULT_REFRESH_DATE,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Graph arcs, in `FakeGraphReader` form, for converting data published
    under `provenance_id` with `recipe_id`.

    `currencies` names the `CurrencyUnit` dcids to type. A mapping also gives
    each currency the places it is the national currency of, through
    `nationalCurrencyOf`, which a role-shaped target reads in reverse. Merge
    the StatVar's own arcs
    (`CONVERTIBLE_STAT_VAR_ARCS`) into the result.
    """
    places_by_currency = (
        currencies if isinstance(currencies, Mapping) else dict.fromkeys(currencies, ())
    )
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        provenance_id: {"conversionRecipe": (recipe_id,)},
        RECIPE_BODIES[recipe_id].table_provenance: {"lastDataRefreshDate": (refresh_date,)},
    }
    for currency, places in places_by_currency.items():
        arcs[currency] = {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": tuple(places)}
    return arcs


def _entities_in(body: str) -> list[str]:
    """`observationAbout` values in `body`, first-seen order, deduplicated."""
    reader = csv.DictReader(io.StringIO(body))
    return list(dict.fromkeys(row["observationAbout"] for row in reader))


def _availability_body(entities: Sequence[str]) -> str:
    key_values = [{"id": "observationAbout", "values": list(entities)}] if entities else []
    return json.dumps({"data": {"dataConstraints": [{"cubeRegions": [{"keyValues": key_values}]}]}})


class RateTableServer:
    """Serves each StatVar's CSV bodies in turn, one per `/data` request,
    the last repeating. `/availability` lists the entities of the body the
    StatVar's next `/data` request will serve, unless `listed` overrides the
    listing. A request for a StatVar with no body goes to `fallback`, and
    without one fails the test. Every request lands in `requests`.
    """

    def __init__(
        self,
        bodies: Mapping[str, str | Sequence[str]],
        *,
        listed: Mapping[str, Sequence[str]] | None = None,
        fallback: Handler | None = None,
    ) -> None:
        self._bodies = {
            stat_var: [body] if isinstance(body, str) else list(body)
            for stat_var, body in bodies.items()
        }
        self._served = dict.fromkeys(self._bodies, 0)
        self._listed = dict(listed or {})
        self._fallback = fallback
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(
        self, *, base_url: str = SDMX_BASE_URL, url_budget_bytes: int | None = None
    ) -> SdmxClient:
        if url_budget_bytes is None:
            return SdmxClient(base_url=base_url, transport=self.transport())
        return SdmxClient(
            base_url=base_url, transport=self.transport(), url_budget_bytes=url_budget_bytes
        )

    def data_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if "/data/" in r.url.path]

    def availability_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if "/availability/" in r.url.path]

    def _next_body(self, stat_var: str) -> str:
        bodies = self._bodies[stat_var]
        return bodies[min(self._served[stat_var], len(bodies) - 1)]

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one request; usable as an `httpx.MockTransport` handler."""
        self.requests.append(request)
        stat_var = query_params(request).get("c[variableMeasured]", "")
        if stat_var not in self._bodies:
            if self._fallback is None:
                raise AssertionError(f"RateTableServer holds no table for {stat_var!r}")
            return self._fallback(request)
        if "/availability/" in request.url.path:
            entities = self._listed.get(stat_var)
            if entities is None:
                entities = _entities_in(self._next_body(stat_var))
            return httpx.Response(
                200,
                text=_availability_body(entities),
                headers={"content-type": _AVAILABILITY_JSON_CONTENT_TYPE},
            )
        body = self._next_body(stat_var)
        self._served[stat_var] += 1
        return httpx.Response(200, text=body, headers={"content-type": _DATA_CSV_CONTENT_TYPE})


def query_params(request: httpx.Request) -> dict[str, str]:
    return dict(httpx.QueryParams(request.url.query.decode()))
