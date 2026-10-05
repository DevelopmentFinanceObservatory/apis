"""A StatVar without `observationProperties` defaults to `observationAbout` and mints
when scoped to that role. Mint and materialisation are real; graph, SDMX, and storage are faked.
"""

from __future__ import annotations

import httpx

from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

# Mirrors LifeExpectancy_Person, which has neither role nor constraint arcs.
_STAT_VAR = "LifeExpectancy_Person"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _csv_row(*, entity: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR_REF},{entity},Years,dc-aggregate,"
        f"NotApplicable,WorldBank,{year},{value},,facet1\r\n"
    )


_CSV_BODY = (
    _CSV_HEADER
    + _csv_row(entity="country/KEN", year="2015", value="61.4")
    + _csv_row(entity="country/KEN", year="2016", value="61.8")
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="WorldBank",
    measurement_method="dc-aggregate",
    unit="Years",
    obs_count=2,
    earliest_date="2015",
    latest_date="2016",
)


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "measuredProperty": ("lifeExpectancy",),
                "populationType": ("Person",),
            }
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"WorldBank": "World Bank", "country/KEN": "Kenya"},
    )


def _sdmx_transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


_BODY = {
    "schema_version": 1,
    "input": {
        "kind": "statvar_set",
        "statvars": [_STAT_VAR_REF],
        "scope": {
            "entities": {"observationAbout": ["country/KEN"]},
            "facets": {
                "provenance": {"ref": "WorldBank"},
                "measurementMethod": {"ref": "dc-aggregate"},
                "unit": {"ref": "Years"},
            },
        },
    },
    "options": {"bundle": False},
}


def test_a_single_default_role_statvar_mints_a_200_manifest():
    """A StatVar without role arcs mints when scoped to the default `observationAbout` role."""
    store = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport([])),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
    )

    resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"].startswith("spec_")
    assert manifest["content_version"].startswith("cv_")
    # The response includes the rendered artifact and series statistics, not internal frame1.
    assert len(manifest["artifacts"]) == 2
    artifact = next(a for a in manifest["artifacts"] if a["rendering"] != {"statistics": "series"})
    assert artifact["artifact_digest"].startswith("sha256:")
