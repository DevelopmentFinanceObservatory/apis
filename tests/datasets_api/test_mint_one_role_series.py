"""Mints of a role-bearing StatVar that declares one role, through the spec
door. The coverage index holds a row per entity on that role's axis for such a StatVar, with
the other column empty (`series_coverage.schema.ABSENT_ENTITY`), so its facets come from
coverage as a two-role StatVar's do. A recipient-only series and a provider-only series run
each case.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

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
_FRA = "country/FRA"
_DEU = "country/DEU"
_AID = "AidDisbursements"


@dataclass(frozen=True)
class _OneRole:
    """A one-role StatVar, the axis its one role sits on, and the entity a scope binds."""

    stat_var: str
    role: str
    axis: str
    entity: str
    provenance: str

    def coverage_row(self, facet_id: str, entity: str, years: list[int]):
        donor, recipient = (
            (entity, ABSENT_ENTITY) if self.axis == "origin" else (ABSENT_ENTITY, entity)
        )
        return (self.stat_var, facet_id, donor, recipient, years)

    @property
    def role_names(self) -> RoleNames:
        if self.axis == "origin":
            return RoleNames(donor=self.role, recipient=None)
        return RoleNames(donor=None, recipient=self.role)


_RECIPIENT_ONLY = _OneRole(
    stat_var="DebtService",
    role="recipient",
    axis="destination",
    entity=_GHA,
    provenance="WorldBankIDS",
)
_PROVIDER_ONLY = _OneRole(
    stat_var="AidMobilisedByProvider",
    role="provider",
    axis="origin",
    entity=_FRA,
    provenance="OECDDAC1",
)
_CASES = pytest.mark.parametrize(
    "case", [_RECIPIENT_ONLY, _PROVIDER_ONLY], ids=["recipient-only", "provider-only"]
)


def _header(case: _OneRole) -> str:
    return (
        f"STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,{case.role},unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )


def _row(case: _OneRole, place: str, year: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{case.stat_var},{place},CurrencyUSD,NotApplicable,P1Y,"
        f"{case.provenance},{year},10,,history\r\n"
    )


def _graph(case: _OneRole) -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            case.stat_var: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": (case.role,),
                "measuredProperty": ("amount",),
            }
        },
        labels={case.provenance: case.provenance},
    )


def _coverage(case: _OneRole) -> FakeCoverageService:
    """Names a two-role aid StatVar and the one-role series, whose rows leave the role it
    lacks empty. The one-role series holds a second entity outside the bound scope."""
    other = _DEU if case.axis == "origin" else "country/KEN"
    return FakeCoverageService(
        build_coverage_index(
            [
                (_AID, "facet1", "country/USA", _GHA, [2015]),
                case.coverage_row("history", case.entity, [2015, 2016]),
                case.coverage_row("history", other, [2015]),
                case.coverage_row("projection", case.entity, [2025]),
            ],
            role_names={
                _AID: RoleNames(donor="provider", recipient="recipient"),
                case.stat_var: case.role_names,
            },
            facet_metadata={
                "facet1": FacetMetadata(
                    provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
                ),
                "history": FacetMetadata(provenance_id=case.provenance, unit="CurrencyUSD"),
                "projection": FacetMetadata(
                    provenance_id=case.provenance,
                    measurement_method="Estimated",
                    unit="CurrencyUSD",
                ),
            },
        )
    )


def _client(case: _OneRole, recorded: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        text = _header(case) + _row(case, case.entity, "2015") + _row(case, case.entity, "2016")
        return httpx.Response(
            200, text=text, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(case),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage(case),
    )


def _body(case: _OneRole, *, entities: dict[str, list[str]], **door_options) -> dict:
    shape = resolve_shape(f"dcid:{case.stat_var}", graph=_graph(case))
    document = build_identity_document(shape, scope=Scope(entities=entities))
    assert document["entities"] == entities
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


@_CASES
def test_a_one_role_series_mints_the_rows_at_its_bound_entity(case: _OneRole):
    recorded: list[httpx.Request] = []
    body = _body(case, entities={case.axis: [case.entity]}, options={"bundle": False})

    resp = _client(case, recorded).post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["coverage"]["rows"] == 2
    assert [request.url.params[f"c[{case.role}]"] for request in recorded] == [case.entity]
    facets = {facet["axis"]: facet["ref"] for facet in manifest["trace"]["facets"]}
    assert facets["provenance"] == case.provenance
    assert facets["unit"] == "CurrencyUSD"
    assert facets["measurementMethod"] is None


@_CASES
def test_an_unscoped_one_role_series_fetches_on_its_own_axis(case: _OneRole):
    # With no entity bound, the fetch lists every dcid coverage holds on the one axis the
    # series declares (`CoverageIndex.exhaustive_single_axis_filter`).
    recorded: list[httpx.Request] = []
    body = _body(case, entities={}, options={"bundle": False})

    resp = _client(case, recorded).post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    other = _DEU if case.axis == "origin" else "country/KEN"
    assert [request.url.params[f"c[{case.role}]"] for request in recorded] == [
        ",".join(sorted([case.entity, other]))
    ]
