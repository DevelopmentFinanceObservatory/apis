"""A mint request's scope pins a method or unit axis to the facets that declare no value.

`{"ref": null}` on `measurementMethod` or `unit` reaches the identity as the null facet ref and
materialises the facet without that value. On `provenance` it refuses `request_invalid`, the
code the search identity path returns for the same binding.
"""

from __future__ import annotations

import httpx

from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxClient
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_mint_request_unit import _first_value
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR = "MalariaGrantDisbursements"
_RECIPIENT = "country/KEN"
_PROVENANCE = "OECD_DAC2A"
_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
# The grant-equivalent facet has more points, so an unbound method elects it. SDMX writes
# `NotApplicable` for a facet with no method.
_ROWS = {
    "f-grant": [("GrantEquivalent", "2015", "700"), ("GrantEquivalent", "2016", "710")],
    "f-none": [("NotApplicable", "2015", "1000")],
}


def _sdmx() -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(httpx.QueryParams(request.url.query.decode()))
        wanted = params.get("c[facetId]", ",".join(_ROWS)).split(",")
        body = _HEADER + "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},CurrencyUSD,{method},"
            f"NotApplicable,{_PROVENANCE},{year},{value},,{facet_id}\r\n"
            for facet_id, rows in _ROWS.items()
            if facet_id in wanted
            for method, year, value in rows
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )


def _client(artifact_store: FakeArtifactStore):
    store = FakeDatasetStore()
    coverage = build_coverage_index(
        [
            (_STAT_VAR, "f-grant", "country/USA", _RECIPIENT, [2015, 2016]),
            (_STAT_VAR, "f-none", "country/USA", _RECIPIENT, [2015]),
        ],
        role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "f-grant": FacetMetadata(
                provenance_id=_PROVENANCE, measurement_method="GrantEquivalent", unit="CurrencyUSD"
            ),
            "f-none": FacetMetadata(provenance_id=_PROVENANCE, unit="CurrencyUSD"),
        },
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            _PROVENANCE: {"isPartOf": ()},
        },
        labels={_PROVENANCE: "OECD DAC2a", _RECIPIENT: "Kenya"},
    )
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=_sdmx(),
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=FakeCoverageService(coverage),
    )


def _body(facets: dict) -> dict:
    return {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [f"dcid:{_STAT_VAR}"],
            "scope": {"entities": {"recipient": [_RECIPIENT]}, "facets": facets},
        },
        "options": {"bundle": False},
    }


def test_a_null_method_ref_mints_the_facet_that_declares_no_method():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store)

    pinned = client.post("/api/datasets", json=_body({"measurementMethod": {"ref": None}}))
    unpinned = client.post("/api/datasets", json=_body({}))

    assert pinned.status_code == 200, pinned.text
    assert unpinned.status_code == 200, unpinned.text
    manifest = pinned.json()
    assert manifest["definition"]["identity"]["facets"] == {
        "measurementMethod": {"state": "value", "ref": None}
    }
    assert _first_value(manifest, artifact_store) == 1000.0
    assert _first_value(unpinned.json(), artifact_store) == 700.0


def test_a_null_provenance_ref_refuses_request_invalid():
    client = _client(FakeArtifactStore())

    resp = client.post("/api/datasets", json=_body({"provenance": {"ref": None}}))

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/scope"
    assert "a null ref is only valid on the measurementMethod and unit axes" in problem["detail"]
