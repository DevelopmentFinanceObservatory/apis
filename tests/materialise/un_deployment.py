"""A stub of our mixer and the UN System Data Commons over `httpx.MockTransport`, serving one
WHO series family, for tests that read UN rows through a real origin graph client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

OURS_API = "https://ours.example.org/core/api"
UN_API = "https://un.example.org/core/api"

UN_FAMILY = "undata/who/MORT_U5_RATE.WEALTH_QUANTILE--"
UN_VARIABLE = f"{UN_FAMILY}Q1"
UN_SECOND_VARIABLE = f"{UN_FAMILY}Q2"
UN_FACET_ID = "18296491596171160445"
UN_PROVENANCE = "undata/p/WHO"
UN_UNIT = "undata/UNIT_MEASURE-RATIO_COUNT_DEATHS_PER_1000_COUNT_LIVEBIRTHS"

# The WHO series as the UN deployment serves it at KEN, oldest first, and a GHA one.
UN_SERIES = {
    "country/KEN": [
        ("1993", 129.32726),
        ("1998", 136.17859),
        ("2003", 148.88945),
        ("2008", 97.53082),
        ("2014", 56.70805),
    ],
    "country/GHA": [("2008", 110.5), ("2014", 88.0)],
}


_EARTH_COUNTRIES_ARC = "<-containedInPlace+{typeOf:Country}"


@dataclass
class UnDeployments:
    """Our mixer, which holds no UN data, and a stub UN deployment answering observation GETs.

    A GET naming a place in `refused` gets the UN front door's 403 refusal page, as the real one
    does for some places;
    `status` other than 200 fails every UN observation read, or only the dated ones that read
    rows when `fail_rows_only`; `unreachable` drops the connection. Its SDMX `/availability`
    lists `available` as the places holding the WHO family's data, fails with
    `availability_status` other than 200, and records each read in `availability_requests`.
    `short_by` raises each series' `obsCount` past the dated observations served, as a truncated
    read would. `arcs` maps a node the UN deployment holds to its outgoing arcs; any other node
    has none. Our mixer lists `countries` as Earth's countries, for the default place scope's
    countries fallback, recording each such read in `countries_requests`. It holds each place
    in `available` as a node typed `Place`, as it holds the places the real UN deployment
    lists, and finds every other node with no arcs.
    """

    refused: set[str] = field(default_factory=set)
    status: int = 200
    fail_rows_only: bool = False
    unreachable: bool = False
    short_by: int = 0
    arcs: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    available: list[str] = field(default_factory=lambda: list(UN_SERIES))
    availability_status: int = 200
    countries: list[str] = field(default_factory=lambda: list(UN_SERIES))
    un_requests: list[httpx.Request] = field(default_factory=list)
    availability_requests: list[httpx.Request] = field(default_factory=list)
    countries_requests: list[httpx.Request] = field(default_factory=list)

    def _earth_countries(self, request: httpx.Request) -> httpx.Response:
        self.countries_requests.append(request)
        nodes = [{"dcid": country} for country in self.countries]
        return httpx.Response(
            200, json={"data": {"Earth": {"arcs": {"containedInPlace+": {"nodes": nodes}}}}}
        )

    def _un_availability(self, request: httpx.Request) -> httpx.Response:
        self.availability_requests.append(request)
        if self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if self.availability_status != 200:
            return httpx.Response(
                self.availability_status, json={"code": 14, "message": "unavailable"}
            )
        variables = str(request.url.params["c[variableMeasured]"]).split(",")
        values = self.available if any(v.startswith(UN_FAMILY) for v in variables) else []
        region = {"keyValues": [{"id": "observationAbout", "values": values}]}
        return httpx.Response(200, json={"data": {"dataConstraints": [{"cubeRegions": [region]}]}})

    def _our_node(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        places = set(self.available) if payload["property"] == "->typeOf" else set()
        return httpx.Response(
            200,
            json={
                "data": {
                    node: {
                        "arcs": {"typeOf": {"nodes": [{"dcid": "Place"}]}} if node in places else {}
                    }
                    for node in payload["nodes"]
                }
            },
        )

    def _un_node(self, request: httpx.Request) -> httpx.Response:
        nodes = json.loads(request.content)["nodes"]
        return httpx.Response(
            200,
            json={
                "data": {
                    node: {
                        "arcs": {
                            arc: {"nodes": [{"dcid": value} for value in values]}
                            for arc, values in self.arcs.get(node, {}).items()
                        }
                    }
                    for node in nodes
                }
            },
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != httpx.URL(UN_API).host:
            if request.url.path.endswith("/v2/observation"):
                return httpx.Response(200, json={"byVariable": {}, "facets": {}})
            if json.loads(request.content).get("property") == _EARTH_COUNTRIES_ARC:
                return self._earth_countries(request)
            return self._our_node(request)
        if request.url.path.endswith("/v2/node"):
            return self._un_node(request)
        if "/sdmx/v3/availability/" in request.url.path:
            return self._un_availability(request)
        self.un_requests.append(request)
        if self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        dated = "value" in request.url.params.get_list("select")
        if self.status != 200 and (dated or not self.fail_rows_only):
            return httpx.Response(self.status, json={"message": "unavailable"})
        entities = request.url.params.get_list("entity.dcids")
        if self.refused & set(entities):
            return httpx.Response(
                403,
                text="<!doctype html><title>403</title>403 Forbidden",
                headers={"content-type": "text/html; charset=UTF-8"},
            )
        by_entity = {}
        for place in entities:
            observations = UN_SERIES.get(place)
            if not observations:
                continue
            block: dict = {
                "facetId": UN_FACET_ID,
                "obsCount": len(observations) + self.short_by,
                "earliestDate": observations[0][0],
                "latestDate": observations[-1][0],
            }
            if dated:
                block["observations"] = [{"date": d, "value": v} for d, v in observations]
            by_entity[place] = {"orderedFacets": [block]}
        variables = request.url.params.get_list("variable.dcids")
        return httpx.Response(
            200,
            json={
                "byVariable": {
                    variable: {"byEntity": by_entity}
                    for variable in variables
                    if variable.startswith(UN_FAMILY)
                },
                "facets": {
                    UN_FACET_ID: {
                        "provenanceId": UN_PROVENANCE,
                        "provenanceUrl": "https://www.who.int/data",
                        "unit": UN_UNIT,
                        "observationPeriod": "P1Y",
                    }
                },
            },
        )
