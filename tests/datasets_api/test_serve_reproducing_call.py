"""Tests that bundle reproducing calls follow `DEFINITION_ROUTE_LIVE` and target the definition
route with the options used for the mint.
"""

from __future__ import annotations

import io
import json
import zipfile
from urllib.parse import urlsplit

import httpx
import pytest

import datasets_api.serve as serve_module
from datasets_api.mint_models import GroupbyOptions, MintOptions, PivotOptions
from datasets_api.query_options import encode_query_options
from materialise.graph import Facet
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"
_STAT_VAR = "ReproducingCallProbe"
_REF = f"dcid:{_STAT_VAR}"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

_BODY = {
    "schema_version": 1,
    "input": {
        "kind": "statvar_set",
        "statvars": [_REF],
        "scope": {
            "entities": {"recipient": [_RECIPIENT]},
            "facets": {
                "provenance": {"ref": "OECD_DAC2A"},
                "measurementMethod": {"ref": "GrantEquivalent"},
                "unit": {"ref": "USD"},
            },
        },
    },
    "options": {"bundle": True},
}


def _coverage_service() -> FakeCoverageService:
    """A one-facet `CoverageIndex` for `_STAT_VAR`, shaped to reproduce
    `_FACET`: same facet_id/provenance/measurement_method/unit, and
    `n_obs`/`last_year` from the one 2020 row `_CSV_BODY` carries.
    """
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", ABSENT_ENTITY, _RECIPIENT, [2020])],
        role_names={_STAT_VAR: RoleNames(donor=None, recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(*, artifact_store: FakeArtifactStore):
    from dc_sdmx import SdmxClient

    store = FakeDatasetStore()
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, _RECIPIENT): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=_CSV_BODY,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
    )
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _descriptor(artifact_store: FakeArtifactStore, manifest: dict) -> dict:
    zip_bytes = artifact_store.fetch(manifest["artifacts"][0]["url"])
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return json.loads(zf.read("datapackage.json"))


def test_reproducing_call_absent_when_definition_route_is_not_live(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(serve_module, "DEFINITION_ROUTE_LIVE", False)
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    descriptor = _descriptor(artifact_store, resp.json())
    assert "reproducing_call" not in descriptor["dfo"]


def test_reproducing_call_targets_the_definition_route_by_default():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    spec_id = manifest["definition"]["spec_id"]
    descriptor = _descriptor(artifact_store, manifest)
    call = descriptor["dfo"]["reproducing_call"]
    assert call["method"] == "GET"
    assert urlsplit(call["url"]).path == f"/api/datasets/{spec_id}"
    # The reproducing call includes `bundle` and the default `format`.
    assert call["options"] == {"format": "csv", "bundle": "true"}

    reproduced = client.get(urlsplit(call["url"]).path, params=call["options"])

    assert reproduced.status_code == 200, reproduced.text
    reproduced_manifest = reproduced.json()
    assert reproduced_manifest["definition"]["spec_id"] == spec_id
    assert reproduced_manifest["content_version"] == manifest["content_version"]


class TestReproducingCallOptions:
    """`query_options.encode_query_options`: the flat, dotted, string-valued
    query map a `MintOptions` becomes, matching the definition-retrieval
    route's own query grammar key for key -- proved here by table, byte for
    byte, independent of the end-to-end tests above.
    """

    def test_defaults_emit_only_format_and_bundle(self):
        assert encode_query_options(MintOptions()) == {
            "format": "csv",
            "bundle": "true",
        }

    def test_a_non_default_format_and_bundle_false(self):
        options = MintOptions(format="parquet", bundle=False)
        assert encode_query_options(options) == {
            "format": "parquet",
            "bundle": "false",
        }

    def test_a_pivot_emits_dotted_index_and_columns_keys(self):
        options = MintOptions(pivot=PivotOptions(index="recipient", columns="time_period"))
        assert encode_query_options(options) == {
            "format": "csv",
            "bundle": "true",
            "pivot.index": "recipient",
            "pivot.columns": "time_period",
        }

    def test_columns_emits_a_comma_joined_value(self):
        options = MintOptions(columns=["stat_var", "recipient", "value"])
        assert encode_query_options(options) == {
            "format": "csv",
            "bundle": "true",
            "columns": "stat_var,recipient,value",
        }

    def test_a_groupby_emits_a_comma_joined_by_and_the_op(self):
        """Regression for #848: the reproducing URL includes the manifest's groupby options."""
        options = MintOptions(groupby=GroupbyOptions(by=["recipient", "year"], op="sum"))
        assert encode_query_options(options) == {
            "format": "csv",
            "bundle": "true",
            "groupby.by": "recipient,year",
            "groupby.op": "sum",
        }

    def test_ungrouped_options_emit_no_groupby_keys(self):
        assert encode_query_options(MintOptions()) == {
            "format": "csv",
            "bundle": "true",
        }
        ungrouped = encode_query_options(MintOptions(columns=["stat_var"]))
        assert "groupby.by" not in ungrouped
        assert "groupby.op" not in ungrouped


def test_the_reproducing_call_round_trips_a_groupby_through_the_get_door():
    """Regression for #848: following the reproducing URL applies the same groupby options.

    This fixture decodes no CRS axis, so it does not mint a separate collapsed version.
    """
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = dict(_BODY)
    body["options"] = {"bundle": True, "groupby": {"by": ["recipient"], "op": "sum"}}

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    call = _descriptor(artifact_store, manifest)["dfo"]["reproducing_call"]
    assert call["options"]["groupby.by"] == "recipient"
    assert call["options"]["groupby.op"] == "sum"

    reproduced = client.get(urlsplit(call["url"]).path, params=call["options"])

    assert reproduced.status_code == 200, reproduced.text
    assert reproduced.json()["content_version"] == manifest["content_version"]
