"""Tests xlsx and parquet minting, bundled xlsx output, and format size refusals.
The composed app and materialiser are real. Graph, SDMX, and artifact storage are fakes.
"""

from __future__ import annotations

import functools
import hashlib
import io
import zipfile

import httpx
import openpyxl
import pyarrow.parquet as pq

import datasets_api.render as render_module
from dc_sdmx import SdmxClient
from materialise.formats import EXCEL_MAX_CELL_CHARS, write_parquet
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

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _csv_row(*, recipient: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{recipient},USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )


_CSV_BODY = _CSV_HEADER + _csv_row(recipient="country/KEN")

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)

_SCOPE = {
    "entities": {"recipient": ["country/KEN"]},
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
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
    )


def _sdmx_transport(*, body: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return httpx.MockTransport(handler)


def _coverage_service() -> FakeCoverageService:
    """A one-facet `CoverageIndex` for `_STAT_VAR`, shaped to reproduce
    `_FACET`: same facet_id/provenance/measurement_method/unit, and
    `n_obs`/`last_year` from the one 2015 row `_CSV_BODY` carries.
    """
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", ABSENT_ENTITY, "country/KEN", [2015])],
        role_names={_STAT_VAR: RoleNames(donor=None, recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(*, artifact_store, body: str = _CSV_BODY, graph: FakeMintGraphReader | None = None):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(body=body)),
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _body(*, format: str, bundle: bool = False) -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"format": format, "bundle": bundle},
    }


class TestXlsxMintServesACorrectWorkbookFromASignedUrl:
    def test_milestone(self):
        artifact_store = FakeArtifactStore()
        client = _client(artifact_store=artifact_store)

        resp = client.post("/api/datasets", json=_body(format="xlsx"))

        assert resp.status_code == 200, resp.text
        artifact = resp.json()["artifacts"][0]
        assert artifact["media_type"] == (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        fetched = artifact_store.fetch(artifact["url"])
        assert hashlib.sha256(fetched).hexdigest() == artifact["artifact_digest"].removeprefix(
            "sha256:"
        )
        workbook = openpyxl.load_workbook(io.BytesIO(fetched))
        header = [cell.value for cell in next(workbook.active.iter_rows(max_row=1))]
        assert header[:3] == ["stat_var", "stat_var_label", "recipient"]


class TestParquetMintServesACorrectFileFromASignedUrl:
    def test_milestone(self):
        artifact_store = FakeArtifactStore()
        client = _client(artifact_store=artifact_store)

        resp = client.post("/api/datasets", json=_body(format="parquet"))

        assert resp.status_code == 200, resp.text
        artifact = resp.json()["artifacts"][0]
        assert artifact["media_type"] == "application/vnd.apache.parquet"

        fetched = artifact_store.fetch(artifact["url"])
        assert hashlib.sha256(fetched).hexdigest() == artifact["artifact_digest"].removeprefix(
            "sha256:"
        )
        table = pq.read_table(io.BytesIO(fetched))
        assert table.column_names[:3] == ["stat_var", "stat_var_label", "recipient"]


def test_a_bundled_xlsx_mint_wraps_the_written_workbook_in_a_data_package():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_body(format="xlsx", bundle=True))

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    assert artifact["media_type"] == "application/zip"

    fetched = artifact_store.fetch(artifact["url"])
    with zipfile.ZipFile(io.BytesIO(fetched)) as zf:
        names = set(zf.namelist())
    assert "datapackage.json" in names
    assert "data/oecd-dac2a.xlsx" in names


def test_excel_limit_refuses_a_mint_whose_frame_breaches_the_cell_character_cap():
    graph = _graph()
    graph._arcs["country/KEN"] = {"name": ("x" * (EXCEL_MAX_CELL_CHARS + 1),)}
    client = _client(artifact_store=FakeArtifactStore(), graph=graph)

    resp = client.post("/api/datasets", json=_body(format="xlsx"))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "excel_limit"
    assert problem["retryable"] is False


def test_parquet_size_bound_refuses_a_mint_whose_written_output_exceeds_its_bound(monkeypatch):
    monkeypatch.setitem(
        render_module._FORMAT_WRITERS, "parquet", functools.partial(write_parquet, size_bound=1)
    )
    client = _client(artifact_store=FakeArtifactStore())

    resp = client.post("/api/datasets", json=_body(format="parquet"))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "parquet_size_bound"
    assert problem["retryable"] is False
