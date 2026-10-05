"""Mints and previews of a role-bearing StatVar that declares one role, through the spec
door. The coverage index holds a row per recipient for such a StatVar, with an empty donor
(`series_coverage.schema.ABSENT_ENTITY`), so its facets come from coverage as a two-role
StatVar's do.
"""

from __future__ import annotations

import httpx

from datasets_api.adapter import build_identity_document
from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from datasets_api.shape import resolve_shape
from dc_sdmx import SdmxClient
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_GHA = "country/GHA"
_DEBT_SERVICE = "DebtService"
_AID = "AidDisbursements"

_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _row(place: str, year: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_DEBT_SERVICE},{place},CurrencyUSD,NotApplicable,P1Y,WorldBankIDS,"
        f"{year},10,,history\r\n"
    )


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _DEBT_SERVICE: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        },
        labels={"WorldBankIDS": "World Bank International Debt Statistics"},
    )


def _coverage() -> FakeCoverageService:
    """Names a two-role aid StatVar and the debt series, whose rows have no donor."""
    return FakeCoverageService(
        build_coverage_index(
            [
                (_AID, "facet1", "country/USA", _GHA, [2015]),
                (_DEBT_SERVICE, "history", ABSENT_ENTITY, _GHA, [2015, 2016]),
                (_DEBT_SERVICE, "projection", ABSENT_ENTITY, _GHA, [2025]),
            ],
            role_names={
                _AID: RoleNames(donor="provider", recipient="recipient"),
                _DEBT_SERVICE: RoleNames(donor=None, recipient="recipient"),
            },
            facet_metadata={
                "facet1": FacetMetadata(
                    provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
                ),
                "history": FacetMetadata(provenance_id="WorldBankIDS", unit="CurrencyUSD"),
                "projection": FacetMetadata(
                    provenance_id="WorldBankIDS", measurement_method="Estimated", unit="CurrencyUSD"
                ),
            },
        )
    )


def _client(recorded: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        text = _HEADER + _row(_GHA, "2015") + _row(_GHA, "2016")
        return httpx.Response(
            200, text=text, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage(),
    )


def _body(**door_options) -> dict:
    shape = resolve_shape(f"dcid:{_DEBT_SERVICE}", graph=_graph())
    document = build_identity_document(shape, scope=Scope(entities={"destination": [_GHA]}))
    assert document["entities"] == {"destination": [_GHA]}
    sidecar = {
        "schema_version": 1,
        "spec_id": recompute_identity(document).spec_id,
        "identity": document,
    }
    return {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": sidecar["spec_id"], "definition": sidecar},
        **door_options,
    }


def test_a_recipient_only_series_mints_the_rows_at_its_bound_recipient():
    recorded: list[httpx.Request] = []

    resp = _client(recorded).post("/api/datasets", json=_body(options={"bundle": False}))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["coverage"]["rows"] == 2
    assert [request.url.params["c[recipient]"] for request in recorded] == [_GHA]
    facets = {facet["axis"]: facet["ref"] for facet in manifest["trace"]["facets"]}
    assert facets["provenance"] == "WorldBankIDS"
    assert facets["unit"] == "CurrencyUSD"
    assert facets["measurementMethod"] is None


def test_a_recipient_only_series_previews_the_rows_the_mint_serves():
    resp = _client([]).post("/api/datasets", json=_body(preview={"limit": 5}))

    assert resp.status_code == 200, resp.text
    assert resp.json()["members"][0]["total_rows"] == 2
