"""A materialisation reads a pinned identity under its pinned provenances
(`adapter.pinned_provenance_scope`), and `serve.prewrite_definition` runs its facet probe in that
scope. Admission keeps every admitted provenance at every place (#1271), so a base pin serves at
a place where a DFO table also holds data, and inside the scope DFO's facet leads when it is at
least as broad as the best base facet."""

from __future__ import annotations

import json

import httpx
import pytest

from datasets_api.adapter import pinned_provenance_scope
from dc_sdmx import SdmxClient
from materialise import FacetBinding, MaterialisationRequest, probe_collection_facets
from materialise.facets import joint_dispatch_facet_ids
from materialise.generation import _select_member_facets
from materialise.window import RequestedWindow
from origin_catalogue import BASE, DFO
from origin_catalogue.graph import build_graph_client
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

_CRS = "provenance/OECDCreditorReportingSystem"
_WDI = "dc/base/WorldDevelopmentIndicators"
_IMF = "dc/base/IMF_WorldEconomicOutlook"
_KEN = "country/KEN"
_VARIABLE = "Amount_Debt_Service"


def _three_facets(request: httpx.Request) -> httpx.Response:
    """Every observation read finds one facet under each of CRS (DFO), WDI and IMF (base) at
    every named entity. A node read finds no arcs."""
    payload = json.loads(request.content)
    if request.url.path.endswith("/v2/node"):
        return httpx.Response(200, json={"data": {node: {"arcs": {}} for node in payload["nodes"]}})
    variables, entities = payload["variable"]["dcids"], payload["entity"]["dcids"]
    ordered = [{"facetId": provenance, "obsCount": 1} for provenance in (_CRS, _WDI, _IMF)]
    block = {"orderedFacets": ordered}
    return httpx.Response(
        200,
        json={
            "byVariable": {v: {"byEntity": dict.fromkeys(entities, block)} for v in variables},
            "facets": {p: {"provenanceId": p} for p in (_CRS, _WDI, _IMF)},
        },
    )


def _sdmx_never_read() -> SdmxClient:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the probe read SDMX: {request.url}")

    return SdmxClient(base_url="https://sdmx.example.org", transport=httpx.MockTransport(refuse))


@pytest.mark.parametrize(
    ("pin", "candidates", "elected"),
    [
        pytest.param(None, [_CRS, _WDI, _IMF], _CRS, id="unpinned"),
        pytest.param({"state": "value", "ref": _WDI}, [_WDI], _WDI, id="base-value"),
        # Equal breadth, count and dates: the provenance id breaks the tie.
        pytest.param({"state": "set", "refs": [_WDI, _IMF]}, [_WDI, _IMF], _IMF, id="base-set"),
        # A pin holding the DFO table keeps every pinned provenance, and DFO's facet, as broad
        # as WDI's at the one place, leads inside the pin.
        pytest.param({"state": "set", "refs": [_CRS, _WDI]}, [_CRS, _WDI], _CRS, id="dfo-in-set"),
    ],
)
def test_the_materialisation_probe_reads_under_the_pinned_provenances(
    bind_catalogue, pin: dict | None, candidates: list[str], elected: str
) -> None:
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row(_CRS, DFO),
                provenance_row(_WDI, BASE),
                provenance_row(_IMF, BASE),
            ],
            dfo_variables=[_VARIABLE],
            base_variables={_VARIABLE: ([_WDI, _IMF], [])},
        )
    )
    identity = {} if pin is None else {"facets": {"provenance": pin}}
    request = MaterialisationRequest(
        stat_var=f"dcid:{_VARIABLE}",
        role_bindings={"observationAbout": [_KEN]},
        declared_roles=("observationAbout",),
        facet_bindings={}
        if pin is None or pin["state"] == "set"
        else {"provenance": FacetBinding(state="bound", ref=pin["ref"])},
        window=RequestedWindow(kind="all"),
    )
    client = build_graph_client(
        base_url="https://ours.example.org/core/api",
        token=None,
        transport=httpx.MockTransport(_three_facets),
    )

    with client, pinned_provenance_scope(identity):
        probe = probe_collection_facets(
            (request,), graph=client, availability=_sdmx_never_read(), settle=True
        )
        (member,) = probe.members
        election = _select_member_facets(
            request, member.candidates, graph=client, breadth=member.breadth_in(request.window)
        )

    assert [facet.provenance_id for facet in member.candidates] == candidates
    assert joint_dispatch_facet_ids(member.candidates, election, member=0)[0] == elected
