"""Tests that minting uses the packaged code index to decode declared CRS axes by default.
The fixture uses a real indexed StatVar. Undeclared constraint properties stay undecoded.
"""

from __future__ import annotations

import csv
import io

import httpx

from code_index.shared import get_index
from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

# This indexed StatVar has four declared CRS axes and two undeclared constraints.
# The undeclared constraints stay undecoded.
_STAT_VAR = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_PROVIDER = "country/USA"
_RECIPIENT = "country/KEN"
_BASE_URL = "https://datacommons.one.org/core/api"

# Expected labels from the packaged index, in decode-plan column order.
_EXPECTED_DECODED_LABELS = {
    "stat_var_crsCooperationModality": "general budget support",
    "stat_var_crsPurpose": "vocational training",
    "stat_var_crsTypeOfFinance": "standard grant",
    "stat_var_crsTypeOfFlow": "official development assistance",
}

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
    "facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER
    + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_PROVIDER},{_RECIPIENT},USD,GrantEquivalent,"
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

_SCOPE = {
    "entities": {"financingProvider": [_PROVIDER], "financingRecipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("financingProvider", "financingRecipient"),
                "measuredProperty": ("amount",),
            },
            # Named so the download's default name columns fill.
            _PROVIDER: {"name": ("United States",)},
            _RECIPIENT: {"name": ("Kenya",)},
            "GrantEquivalent": {"name": ("Grant equivalent",)},
        },
        # Facet probing merges bound entities, so the recipient entry covers both roles here.
        facets={(_STAT_VAR, _RECIPIENT): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )


def _coverage_service() -> FakeCoverageService:
    """A one-facet `CoverageIndex` for `_STAT_VAR`, shaped to reproduce
    `_FACET`: same facet_id/provenance/measurement_method/unit, and
    `n_obs`/`last_year` from the one 2020 row `_CSV_BODY` carries.
    `financingProvider` binds the coverage `donor` column, matching this
    StatVar's own declared roles.
    """
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", _PROVIDER, _RECIPIENT, [2020])],
        role_names={
            _STAT_VAR: RoleNames(donor="financingProvider", recipient="financingRecipient")
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(*, artifact_store: FakeArtifactStore | None = None):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(),
        sdmx=SdmxClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text=_CSV_BODY,
                    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                )
            ),
        ),
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def test_the_fixture_statvar_declares_the_expected_codes():
    # Keep the fixture aligned with the packaged index's StatVar declaration.
    index = get_index()
    statvar = next(row for row in index.statvars if row.dcid == _STAT_VAR)
    assert statvar.observation_properties == ("financingProvider", "financingRecipient")
    decoded = {
        f"stat_var_{code_system}": index.code_by_dcid(value).label
        for code_system, value in statvar.constraint_properties.items()
        if code_system in index.declarations
    }
    assert decoded == _EXPECTED_DECODED_LABELS


def test_a_mint_response_carries_every_declared_code_system_as_its_own_column():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert not any(w["code"] == "enrichment_unavailable" for w in manifest["warnings"])
    artifact = manifest["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    header, data = rows[0], rows[1]
    statvar = next(row for row in get_index().statvars if row.dcid == _STAT_VAR)
    code_systems = [decoded.removeprefix("stat_var_") for decoded in _EXPECTED_DECODED_LABELS]
    # Each declared axis is a code column and its label column, in property order after the
    # entities and their names. The packaged index declares neither undeclared constraint, so
    # both stay out.
    assert header[5:13] == [name for cs in code_systems for name in (cs, f"{cs}_label")]
    assert not {"financialFlowType", "nettingBasis"} & set(header)
    for code_system, expected_label in zip(
        code_systems, _EXPECTED_DECODED_LABELS.values(), strict=True
    ):
        assert data[header.index(code_system)] == statvar.constraint_properties[code_system]
        assert data[header.index(f"{code_system}_label")] == expected_label


def test_a_decoded_column_is_selectable_by_name_through_the_mint_door():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    keep = [
        "stat_var",
        "financingProvider",
        "financingRecipient",
        "time_period",
        "value",
        "unit",
        "provenance",
        "stat_var_crsPurpose",
    ]
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False, "columns": keep},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    # Naming one decoded column keeps that axis alone.
    assert rows[0] == [
        "stat_var",
        "financingProvider",
        "financingProvider_label",
        "financingRecipient",
        "financingRecipient_label",
        "crsPurpose",
        "crsPurpose_label",
        "time_period",
        "value",
        "unit",
        "unit_label",
        "currency",
        "price_basis",
        "base_year",
        "provenance",
        "provenance_label",
    ]
    assert rows[1][5:7] == ["CRSPurposeCode11330", "vocational training"]


def test_a_mistyped_decoded_column_name_is_offered_as_an_alternative():
    # The misspelt decoded column should suggest `stat_var_crsPurpose` as an alternative.
    client = _client()
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False, "columns": ["stat_var", "value", "stat_var_crsPurpse"]},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "column_unknown"
    assert problem["pointer"] == "/options/columns"
    assert "stat_var_crsPurpose" in {a["ref"] for a in problem["alternatives"]}
