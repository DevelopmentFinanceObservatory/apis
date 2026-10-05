"""A role-bearing StatVar with one named recipient returns observations through the preview path.

The test uses a hand-built SDMX client to verify the real fetch filters and request count.
"""

from __future__ import annotations

import asyncio

import httpx

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.region.codes import OBSERVATIONS_PROBE_FAILED
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

_SV_DCID = "TESTFIXTURE/SearchPreviewRoleProbeE2E/Flow"
_KEN = "country/KEN"

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_MATERIALISE_BASE_URL = "https://example.org/core/api"

# The StatVar family declares both provider/recipient (see its
# observationProperties arc below), so the real SDMX wire carries a column
# for both regardless of which one this query bound -- only "recipient" is
# named here. The preview reads "provider" open and takes each planned
# series' rows by its provider dcid, so every row carries the provider the
# coverage index holds for this pair.
_NOR = "country/NOR"
_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,provider,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_POINTS = (("2019", 1.0), ("2020", 2.0))


def _csv_body() -> str:
    rows = [_CSV_HEADER]
    for date_str, value in _POINTS:
        rows.append(
            ",".join(
                [
                    "dataflow",
                    "DC:DF_OBS(1.0.0)",
                    "I",
                    f"dcid:{_SV_DCID}",
                    _NOR,
                    _KEN,
                    "NotApplicable",
                    "NotApplicable",
                    "NotApplicable",
                    "dc/base/SearchPreviewRoleProbeE2E",
                    date_str,
                    str(value),
                    "",
                    "f1",
                ]
            )
            + "\r\n"
        )
    return "".join(rows)


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def test_single_named_recipient_role_bearing_shape_gets_its_series(monkeypatch):
    """The preview fetch returns the recipient's series without a probe-failure warning."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _NOR, _KEN, [2019, 2020])],
        role_names={_SV_DCID: RoleNames(donor="provider", recipient="recipient")},
        # Must match the CSV body's own `provenance` cell for facet_id
        # "f1" exactly: the delivered rows survive only under the facet
        # select_facet actually elects, so a mismatched provenance here
        # elects a facet the real fetch never reports, filtering every row
        # away and delivering zero points.
        facet_metadata={"f1": FacetMetadata(provenance_id="dc/base/SearchPreviewRoleProbeE2E")},
    )
    body = _csv_body()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params.get("c[variableMeasured]") == _SV_DCID
        assert params.get("c[recipient]") == _KEN
        requests.append(request)
        # c[TIME_PERIOD] (present on the LATEST-scoped facet probe, absent
        # on the real fetch) is the only param that varies between the two
        # requests probe_facets/materialise_preview issue for this pair --
        # both are answered identically here, from the same body.
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    materialise_sdmx = MaterialiseSdmxClient(
        base_url=_MATERIALISE_BASE_URL, transport=httpx.MockTransport(handler)
    )
    materialise_graph = FakeGraphReader()

    graph = FakeGraph(
        nodes={
            "Kenya": {"label": "Kenya", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _SV_DCID: {
                "label": "Test flow",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                    "observationProperties": {
                        "nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]
                    },
                },
            },
        },
        detect={"aid to Kenya": {"svs": [_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _KEN},
        obs={},
    )
    sdmx = FakeSdmxClient(
        observations={
            _SV_DCID: {
                (("recipient", _KEN),): [
                    Facet(
                        earliest_date="2019",
                        latest_date="2020",
                        obs_count=2,
                        dates=["2019", "2020"],
                        values=[1.0, 2.0],
                        facet_id="f1",
                    )
                ],
            },
        },
    )

    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Kenya"],
            date_request=None,
            detect_query="aid to Kenya",
            role_query="aid to Kenya",
            entity_roles=door_roles_for_test_query("aid to Kenya", ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_steps=[],
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        )
    )

    assert result.status == "definite", result
    assert not any(w.code == OBSERVATIONS_PROBE_FAILED for w in result.warnings)
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert len(observations.series) >= 1
    assert observations.series[0].total_obs == 2

    # One real SDMX call for this one pair: candidates come from `coverage`,
    # an in-memory lookup, so no facet probe call precedes
    # materialise_preview's own real, unbounded series read.
    time_periods = [r.url.params.get("c[TIME_PERIOD]") for r in requests]
    assert time_periods == [None]
