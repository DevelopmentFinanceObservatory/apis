"""A request that binds no entity is fetched at the places its probe resolved, and a fetched
row serves only where that probe admitted its facet for the row's own StatVar. Pins the two
together on every fetch path: one member, a preview, a collection chunk and a group.
"""

from __future__ import annotations

import httpx
import pyarrow as pa
import pytest

from dc_sdmx import SdmxClient
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_group,
    materialise_preview,
    materialise_spec,
)
from materialise.facets import probe_collection_facets
from materialise.graph import Facet
from materialise.window import RequestedWindow

from ._group_support import _BARE_B, _BARE_C, _BOUND_FACETS, _SHARED_FACET, _SNAPSHOT
from .conftest import FakeGraphReader, FakeRevisionReader

_ABOUT = "observationAbout"
_KEN = "country/KEN"
_TGO = "country/TGO"
_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_SERVICE = {"materialiser_build": "build-123", "capability_version": "cap-v1"}

# The facet admission passes at Togo for `_BARE_B`, where it withholds `_SHARED_FACET`.
_OTHER_FACET = Facet(
    facet_id="facetOther",
    provenance_id="PROV2",
    measurement_method="METH1",
    unit="UNIT1",
    obs_count=1,
    earliest_date="2015",
    latest_date="2020",
)

# What the probe admits: `_SHARED_FACET` for `_BARE_B` at Kenya and for `_BARE_C` at Togo.
_ADMITTED = {
    (_BARE_B, _KEN): [_SHARED_FACET],
    (_BARE_B, _TGO): [_OTHER_FACET],
    (_BARE_C, _TGO): [_SHARED_FACET],
}

# What the row read returns under `c[facetId]=facet1`: the admitted rows in 2015, and a 2020
# row of `_SHARED_FACET` for `_BARE_B` at Togo, where its probe did not admit that facet.
_SERVED = {(_BARE_B, _KEN): "2015", (_BARE_B, _TGO): "2020", (_BARE_C, _TGO): "2015"}


def _graph() -> FakeGraphReader:
    return FakeGraphReader(facets=_ADMITTED, child_places={("Earth", "Country"): [_KEN, _TGO]})


def _unbound(bare_stat_var: str, *, window: str = "all") -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{bare_stat_var}",
        role_bindings={},
        declared_roles=(_ABOUT,),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind=window),
    )


def _sdmx() -> tuple[SdmxClient, list[httpx.Request]]:
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if (response := _graph().scope_availability().respond(request)) is not None:
            return response
        recorded.append(request)
        named = str(request.url.params["c[variableMeasured]"]).split(",")
        places = str(request.url.params[f"c[{_ABOUT}]"]).split(",")
        assert request.url.params["c[facetId]"] == "facet1"
        text = _HEADER + "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{bare},{place},UNIT1,METH1,NotApplicable,PROV1,"
            f"{_SERVED[(bare, place)]},10,,facet1\r\n"
            for bare in named
            for place in places
            if (bare, place) in _SERVED
        )
        return httpx.Response(
            200, text=text, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    ), recorded


def _fetched(recorded: list[httpx.Request]) -> dict[str, str]:
    """The `c[observationAbout]` filter each StatVar was read under."""
    return {
        bare: str(r.url.params[f"c[{_ABOUT}]"])
        for r in recorded
        for bare in str(r.url.params["c[variableMeasured]"]).split(",")
    }


def _rows(frames) -> list[tuple[str, str, str]]:
    """Each served row's StatVar, place and period."""
    table = pa.concat_tables([frame.table for frame in frames])
    return sorted(
        zip(
            [ref.removeprefix("dcid:") for ref in table["stat_var"].to_pylist()],
            table[_ABOUT].to_pylist(),
            table["time_period"].to_pylist(),
            strict=True,
        )
    )


@pytest.mark.parametrize("window", ["all", "latest"])
def test_a_member_binding_no_entity_serves_a_facet_only_at_the_default_places_admitting_it(window):
    sdmx, recorded = _sdmx()

    result = materialise_spec(
        _unbound(_BARE_B, window=window),
        sdmx=sdmx,
        graph=_graph(),
        enrichment=EnrichmentLookup(_graph()),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        **_SERVICE,
    )

    # Both countries hold data, so the fetch names both. Togo's row is withheld, and under a
    # latest window its later period does not displace Kenya's.
    assert _fetched(recorded) == {_BARE_B: f"{_KEN},{_TGO}"}
    assert _rows(result.frame_chunks()) == [(_BARE_B, _KEN, "2015")]


def test_a_preview_binding_no_entity_counts_a_facet_only_at_the_default_places_admitting_it():
    sdmx, recorded = _sdmx()

    preview = materialise_preview(_unbound(_BARE_B), sdmx=sdmx, graph=_graph(), limit=10)

    assert _fetched(recorded) == {_BARE_B: f"{_KEN},{_TGO}"}
    assert preview.total_rows == 1
    assert _rows([preview.frame]) == [(_BARE_B, _KEN, "2015")]


@pytest.mark.parametrize("supply_probe", [False, True])
@pytest.mark.parametrize("window", ["all", "latest"])
def test_group_members_each_serve_at_their_own_places_under_their_own_admission(
    window, supply_probe
):
    graph = _graph()
    requests = [_unbound(_BARE_B, window=window), _unbound(_BARE_C, window=window)]
    probe = (
        probe_collection_facets(
            requests, settle=True, graph=graph, availability=graph.scope_availability()
        )
        if supply_probe
        else None
    )
    sdmx, recorded = _sdmx()

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        precomputed_probe=probe,
        **_SERVICE,
    )

    assert _fetched(recorded) == {_BARE_B: f"{_KEN},{_TGO}", _BARE_C: _TGO}
    assert _rows(result.frame_chunks()) == [(_BARE_B, _KEN, "2015"), (_BARE_C, _TGO, "2015")]
