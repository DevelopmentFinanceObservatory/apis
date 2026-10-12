"""Pins mint-door selection and pivots across CSV, XLSX, and Parquet, including refusals,
cap boundaries, and stable `content_version`. Materialisation and registry are real; graph,
SDMX, and artifact storage are faked.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence

import httpx
import openpyxl
import pyarrow.parquet as pq
import pytest

import datasets_api.render as render_module
from datasets_api.config import DatasetsBuildSettings
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from materialise.reshape import PivotPlan, ReshapePlan, Table, apply_reshapes
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


def _csv_row(*, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


_CSV_BODY = (
    _CSV_HEADER
    + _csv_row(year="2015", value="1000")
    + _csv_row(year="2016", value="2000")
    + _csv_row(year="2017", value="3000")
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=3,
    earliest_date="2015",
    latest_date="2017",
)

_SCOPE = {
    "entities": {"recipient": ["country/KEN"]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


_NAMED_NODES: dict[str, dict[str, tuple[str, ...]]] = {
    "GrantEquivalent": {"name": ("Grant equivalent",)},
}
"""Names the facet's measurement method, which every long download's default
`measurementMethod_label` reads."""

_STAT_VAR_NAME = ("Malaria grant disbursements",)


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
                "name": _STAT_VAR_NAME,
            },
            "country/KEN": {
                "countryAlpha3Code": ("KEN",),
                "countryAlpha2Code": ("KE",),
                "name": ("Kenya",),
                "typeOf": ("Country",),
            },
            **_NAMED_NODES,
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
    )


def _coverage_service(
    *,
    donor_role: str | None = None,
    donor: str | None = None,
    recipient: str = "country/KEN",
    donors: Sequence[str] | None = None,
    facet_id: str = "facet1",
    provenance: str = "OECD_DAC2A",
    measurement_method: str = "GrantEquivalent",
    unit: str = "USD",
    years: Sequence[int] = (2015, 2016, 2017),
) -> FakeCoverageService:
    """Build a coverage facet for this StatVar, role, and year set."""
    coverage_donor = "country/USA" if donor is None and donor_role is not None else donor
    coverage_donor = ABSENT_ENTITY if coverage_donor is None else coverage_donor
    coverage_donors = (coverage_donor,) if donors is None else donors
    index = build_coverage_index(
        [
            (_STAT_VAR, facet_id, covered_donor, recipient, list(years))
            for covered_donor in coverage_donors
        ],
        role_names={_STAT_VAR: RoleNames(donor=donor_role, recipient="recipient")},
        facet_metadata={
            facet_id: FacetMetadata(
                provenance_id=provenance, measurement_method=measurement_method, unit=unit
            )
        },
    )
    return FakeCoverageService(index)


def _client(
    *,
    store: FakeDatasetStore | None = None,
    artifact_store=None,
    pivot_max_output_cells: int | None = None,
):
    store = store if store is not None else FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        settings=(
            DatasetsBuildSettings(pivot_max_output_cells=pivot_max_output_cells)
            if pivot_max_output_cells is not None
            else None
        ),
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
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _body(
    *,
    pivot: dict | None = None,
    columns: list | None = None,
    groupby: dict | None = None,
    format: str | None = None,
) -> dict:
    options: dict = {"bundle": False}
    if format is not None:
        options["format"] = format
    if pivot is not None:
        options["pivot"] = pivot
    if columns is not None:
        options["columns"] = columns
    if groupby is not None:
        options["groupby"] = groupby
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": options,
    }


def test_a_pivot_mint_serves_a_wide_csv_from_a_signed_url():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _body(pivot={"index": "recipient", "columns": "time_period"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]

    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    assert rows[0] == ["recipient", "2015", "2016", "2017"]
    assert rows[1][0] == "country/KEN"
    assert [float(v) for v in rows[1][1:]] == [1000.0, 2000.0, 3000.0]
    assert artifact["rendering"] == {
        "format": "csv",
        "bundle": False,
        "pivot": {"index": "recipient", "columns": "time_period"},
        render_module.LAYOUT_KEY: render_module.DATA_LAYOUT_VERSION,
    }


def test_a_pivot_mint_serves_a_wide_xlsx_from_a_signed_url():
    # Exercise the `xlsx` to `excel` cap mapping through the XLSX writer.
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _body(pivot={"index": "recipient", "columns": "time_period"}, format="xlsx")

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    assert (
        artifact["media_type"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    fetched = artifact_store.fetch(artifact["url"])
    workbook = openpyxl.load_workbook(io.BytesIO(fetched))
    rows = [[cell.value for cell in row] for row in workbook.active.iter_rows()]
    assert rows[0] == ["recipient", "2015", "2016", "2017"]
    assert rows[1][0] == "country/KEN"
    assert [float(v) for v in rows[1][1:]] == [1000.0, 2000.0, 3000.0]


def test_a_pivot_mint_serves_a_wide_parquet_from_a_signed_url():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _body(pivot={"index": "recipient", "columns": "time_period"}, format="parquet")

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    assert artifact["media_type"] == "application/vnd.apache.parquet"
    fetched = artifact_store.fetch(artifact["url"])
    parquet_table = pq.read_table(io.BytesIO(fetched))
    assert parquet_table.column_names == ["recipient", "2015", "2016", "2017"]
    rows = parquet_table.to_pylist()
    assert len(rows) == 1
    assert rows[0]["recipient"] == "country/KEN"
    assert [rows[0]["2015"], rows[0]["2016"], rows[0]["2017"]] == [1000.0, 2000.0, 3000.0]


def test_pivoting_the_render_never_moves_the_content_version():
    store = FakeDatasetStore()
    plain = _client(store=store)
    plain_resp = plain.post("/api/datasets", json=_body())
    assert plain_resp.status_code == 200, plain_resp.text

    pivoted = _client(store=store)
    pivoted_resp = pivoted.post(
        "/api/datasets", json=_body(pivot={"index": "recipient", "columns": "time_period"})
    )
    assert pivoted_resp.status_code == 200, pivoted_resp.text

    assert plain_resp.json()["content_version"] == pivoted_resp.json()["content_version"]


def test_pivot_not_unique_refuses_when_the_projection_collides():
    # All years share (recipient, unit), so this pivot collapses three rows into one cell.
    client = _client()
    body = _body(pivot={"index": "recipient", "columns": "unit"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "pivot_not_unique"
    assert problem["retryable"] is False
    assert problem["issues"]


def test_pivot_not_homogeneous_refuses_when_a_dropped_grain_column_varies_within_an_index_group():
    # Facet axes are fixed for the request. Donor varies in 2016 but is dropped by this pivot,
    # so combining rows by recipient and year would mix donors.
    csv_header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,donor,unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
        "scalingFactor,facetId\r\n"
    )

    def csv_row(*, year: str, value: str, donor: str) -> str:
        return (
            f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,{donor},USD,"
            f"GrantEquivalent,NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
        )

    csv_body = (
        csv_header
        + csv_row(year="2015", value="1000", donor="country/USA")
        + csv_row(year="2016", value="2000", donor="country/TZA")
        + csv_row(year="2017", value="3000", donor="country/USA")
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient", "donor"),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=csv_body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
    )
    store = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(donor_role="donor", donors=("country/USA", "country/TZA")),
    )
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [_STAT_VAR_REF],
            "scope": {
                "entities": {"recipient": ["country/KEN"]},
                "facets": _SCOPE["facets"],
            },
        },
        "options": {"bundle": False, "pivot": {"index": "recipient", "columns": "time_period"}},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "pivot_not_homogeneous"
    assert problem["retryable"] is False
    assert {issue["pointer"] for issue in problem["issues"]} == {"/options/pivot"}
    assert {issue["member"] for issue in problem["issues"]} == {"donor='country/KEN'"}


def test_column_unknown_refuses_when_pivot_names_a_column_the_frame_lacks():
    client = _client()
    body = _body(pivot={"index": "recipient", "columns": "not_a_real_column"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "column_unknown"
    assert problem["retryable"] is False
    assert problem["pointer"] == "/options/pivot"


def test_pivot_over_column_cap_refuses_when_the_output_width_exceeds_the_cap(monkeypatch):
    monkeypatch.setattr(render_module, "_column_cap", lambda format: 2)
    client = _client()
    body = _body(pivot={"index": "recipient", "columns": "time_period"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "pivot_over_column_cap"
    assert problem["retryable"] is False
    assert problem["pointer"] == "/options/pivot"


def test_pivot_over_cell_cap_refuses_when_the_total_cell_count_exceeds_the_cap():
    # Four cells exceed the cap of 2 while width remains within its cap.
    client = _client(pivot_max_output_cells=2)
    body = _body(pivot={"index": "recipient", "columns": "time_period"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "pivot_over_cell_cap"
    assert problem["retryable"] is False
    assert problem["pointer"] == "/options/pivot"


def test_pivot_over_column_cap_passes_at_exactly_the_cap(monkeypatch):
    # Four output columns meet the cap.
    monkeypatch.setattr(render_module, "FORMAT_COLUMN_CAPS", {"csv": 4, "excel": 4, "parquet": 4})
    client = _client()
    body = _body(pivot={"index": "recipient", "columns": "time_period"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text


def test_pivot_over_column_cap_refuses_when_the_output_width_exceeds_the_cap_by_one(monkeypatch):
    # Width 4 exceeds the cap of 3 by one.
    monkeypatch.setattr(render_module, "FORMAT_COLUMN_CAPS", {"csv": 3, "excel": 3, "parquet": 3})
    client = _client()
    body = _body(pivot={"index": "recipient", "columns": "time_period"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "pivot_over_column_cap"
    assert problem["pointer"] == "/options/pivot"


def test_pivot_over_cell_cap_passes_at_exactly_the_cap():
    # One recipient and four output columns produce four cells.
    client = _client(pivot_max_output_cells=4)
    body = _body(pivot={"index": "recipient", "columns": "time_period"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text


def test_pivot_over_cell_cap_refuses_when_the_cell_count_exceeds_the_cap_by_one():
    # Four cells exceed the cap of 3 by one.
    client = _client(pivot_max_output_cells=3)
    body = _body(pivot={"index": "recipient", "columns": "time_period"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "pivot_over_cell_cap"
    assert problem["pointer"] == "/options/pivot"


_CANONICAL_SELECTION = [
    "stat_var",
    "stat_var_label",
    "recipient",
    "recipient_label",
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
"""`_SELECTION_KEEP` in the long layout's order (`materialise.projection`), each kept code
column followed by its name and `unit` by its split."""
_SELECTION_KEEP = ["value", "time_period", "recipient", "stat_var", "unit", "provenance"]
"""A selection naming its columns out of the frame's order."""


def test_a_columns_selection_mint_writes_the_canonical_order():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    keep = _SELECTION_KEEP
    body = _body(columns=keep)

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    assert rows[0] == _CANONICAL_SELECTION
    # Only requested reshape options appear in the rendering.
    assert artifact["rendering"] == {
        "format": "csv",
        "bundle": False,
        "columns": keep,
        render_module.LAYOUT_KEY: render_module.DATA_LAYOUT_VERSION,
    }


def test_a_columns_selection_mint_writes_the_canonical_order_in_xlsx():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _body(columns=_SELECTION_KEEP, format="xlsx")

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    workbook = openpyxl.load_workbook(io.BytesIO(fetched))
    header = [cell.value for cell in next(workbook.active.iter_rows(max_row=1))]
    assert header == _CANONICAL_SELECTION


def test_a_columns_selection_mint_writes_the_canonical_order_in_parquet():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _body(columns=_SELECTION_KEEP, format="parquet")

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    parquet_table = pq.read_table(io.BytesIO(fetched))
    assert parquet_table.column_names == _CANONICAL_SELECTION


def test_a_duplicate_columns_entry_refuses_request_invalid_before_any_sdmx_call():
    client = _client(artifact_store=FakeArtifactStore())
    body = _body(columns=["stat_var", "recipient", "time_period", "value", "unit", "recipient"])

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/options/columns"


@pytest.mark.parametrize(
    ("groupby", "pointer"),
    [
        pytest.param(
            {"by": ["time_period"], "op": "sum", "levels": {"crsPurpose": 0}},
            "/options/groupby/levels",
            id="level-axis-not-kept",
        ),
        pytest.param(
            {"by": ["time_period", "crsTypeOfFlow"], "op": "sum", "levels": {"crsTypeOfFlow": 0}},
            "/options/groupby/levels/crsTypeOfFlow",
            id="flat-axis",
        ),
        pytest.param(
            {"by": ["time_period", "recipient"], "op": "sum", "levels": {"recipient": 0}},
            "/options/groupby/levels/recipient",
            id="role-column",
        ),
        pytest.param(
            {"by": ["crsPurpose"], "op": "sum", "levels": {"crsPurpose": -1}},
            "/options/groupby/levels/crsPurpose",
            id="negative-depth",
        ),
    ],
)
def test_a_groupby_level_the_deployment_cannot_roll_up_refuses_request_invalid(
    groupby: dict, pointer: str
):
    client = _client()

    resp = client.post("/api/datasets", json=_body(groupby=groupby))

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == pointer


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"format": "parquet", "bundle": True, "layout": "frame"}, id="bundle"),
        pytest.param({"format": "parquet", "layout": "frame"}, id="bundle-by-default"),
        pytest.param({"format": "csv", "bundle": False, "layout": "frame"}, id="csv"),
        pytest.param(
            {
                "format": "parquet",
                "bundle": False,
                "layout": "frame",
                "pivot": {"index": "recipient", "columns": "time_period"},
            },
            id="pivot",
        ),
        pytest.param({"format": "parquet", "bundle": False, "layout": 2}, id="layout-version"),
    ],
)
def test_a_frame_layout_outside_a_bare_unpivoted_parquet_file_refuses_request_invalid(
    options: dict,
):
    client = _client()
    body = _body()
    body["options"] = options

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/options/layout"


def test_a_groupby_depth_past_the_integer_conversion_limit_refuses_on_both_doors():
    client = _client()
    depth = "1" + "0" * 4300
    groupby = {"by": ["crsPurpose"], "op": "sum", "levels": {"crsPurpose": 0}}
    body = json.dumps(_body(groupby=groupby)).replace('"crsPurpose": 0', f'"crsPurpose": {depth}')

    responses = [
        client.post("/api/datasets", content=body, headers={"content-type": "application/json"}),
        client.get(
            "/api/datasets/spec_000000"
            f"?groupby.by=crsPurpose&groupby.op=sum&groupby.levels=crsPurpose:{depth}"
        ),
    ]

    for resp in responses:
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"
    assert responses[1].json()["pointer"] == "/options/groupby/levels/crsPurpose"


def test_a_groupby_level_on_a_flat_axis_refuses_on_the_definition_door_too():
    client = _client()

    resp = client.get(
        "/api/datasets/spec_000000"
        "?groupby.by=time_period,crsTypeOfFlow&groupby.op=sum&groupby.levels=crsTypeOfFlow:0"
    )

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/options/groupby/levels/crsTypeOfFlow"


def test_column_not_droppable_refuses_when_selection_drops_grain():
    # Grain, unit, and provenance are required; selection can drop display columns only.
    # The refusal names all four omitted columns.
    client = _client()
    body = _body(columns=["recipient", "value"])

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "column_not_droppable"
    assert problem["retryable"] is False
    issues = problem["issues"]
    assert [issue["pointer"] for issue in issues] == ["/options/columns"] * len(issues)
    assert {issue["member"] for issue in issues} == {
        "stat_var",
        "time_period",
        "unit",
        "provenance",
    }
    assert "stat_var" in problem["detail"]
    assert "time_period" in problem["detail"]
    assert "unit" in problem["detail"]
    assert "provenance" in problem["detail"]


def test_column_unknown_refuses_when_selection_names_a_column_the_frame_lacks():
    client = _client()
    body = _body(columns=["recipient", "not_a_real_column"])

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "column_unknown"
    assert problem["retryable"] is False
    assert problem["pointer"] == "/options/columns"


def test_a_bare_enrichment_field_name_refuses_column_unknown_with_qualified_alternatives():
    # Enrichment fields require an entity-column prefix, such as `recipient_iso3`.
    # Alternatives include valid tokens and table columns; this fixture has no indexed codes.
    client = _client()
    body = _body(columns=["recipient", "value", "iso3"])

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "column_unknown"
    assert problem["retryable"] is False
    assert problem["pointer"] == "/options/columns"
    real_columns = {
        "stat_var",
        "recipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    }
    qualified_tokens = {
        "recipient_iso2",
        "recipient_iso3",
        "recipient_kind",
        "recipient_label",
        "recipient_type",
    }
    assert {a["ref"] for a in problem["alternatives"]} == real_columns | qualified_tokens


def test_a_second_bare_enrichment_field_name_also_refuses_column_unknown():
    # `label` also requires an entity-column prefix.
    client = _client()
    body = _body(columns=["recipient", "value", "label"])

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "column_unknown"
    assert problem["retryable"] is False
    assert problem["pointer"] == "/options/columns"


def test_a_recognised_but_disabled_enrichment_field_refuses_capability_unavailable(monkeypatch):
    # Disabled fields return `capability_unavailable`. Invalid tokens return `column_unknown`.
    monkeypatch.setattr(render_module, "ENABLED_ENRICHMENTS", frozenset())
    client = _client()
    body = _body(columns=["recipient", "value", "recipient_iso3"])

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 501, resp.text
    problem = resp.json()
    assert problem["code"] == "capability_unavailable"
    assert problem["retryable"] is False
    assert problem["pointer"] == "/options/columns"


def test_recipient_iso3_enriches_the_render_with_the_graph_declared_alpha3_code():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _body(
        columns=[
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "provenance",
            "recipient_iso3",
        ]
    )

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert not any(w["code"] == "enrichment_unavailable" for w in manifest["warnings"])
    artifact = manifest["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    position = rows[0].index("recipient_iso3")
    assert rows[0][position - 2 : position] == ["recipient", "recipient_label"]
    assert all(row[position] == "KEN" for row in rows[1:])


def test_recipient_label_enriches_the_render_with_the_graph_declared_name():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _body(
        columns=[
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "provenance",
            "recipient_label",
        ]
    )

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    position = rows[0].index("recipient_label")
    assert rows[0][position - 1] == "recipient"
    assert all(row[position] == "Kenya" for row in rows[1:])


def test_recipient_type_enriches_the_render_with_the_place_type():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _body(
        columns=[
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "provenance",
            "recipient_type",
        ]
    )

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    rows = list(csv.reader(io.StringIO(artifact_store.fetch(artifact["url"]).decode("utf-8"))))
    position = rows[0].index("recipient_type")
    assert rows[0][position - 2 : position] == ["recipient", "recipient_label"]
    assert all(row[position] == "Country" for row in rows[1:])


def test_an_entity_without_a_graph_declared_code_yields_an_empty_cell_and_an_info_warning():
    # Uganda has no code or name arc, so the empty enrichment cell raises an info warning.
    csv_header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )
    csv_body = (
        csv_header + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/UGA,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
                "name": _STAT_VAR_NAME,
            },
            "country/UGA": {"name": ("Uganda",)},
            **_NAMED_NODES,
        },
        facets={(_STAT_VAR, "country/UGA"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=csv_body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
    )
    store = FakeDatasetStore()
    artifact_store = FakeArtifactStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(recipient="country/UGA", years=[2015]),
    )
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [_STAT_VAR_REF],
            "scope": {
                "entities": {"recipient": ["country/UGA"]},
                "facets": _SCOPE["facets"],
            },
        },
        "options": {
            "bundle": False,
            "columns": [
                "stat_var",
                "recipient",
                "time_period",
                "value",
                "unit",
                "provenance",
                "recipient_iso3",
            ],
        },
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    enrichment_warnings = [w for w in manifest["warnings"] if w["code"] == "enrichment_unavailable"]
    assert enrichment_warnings == [
        {
            "code": "enrichment_unavailable",
            "severity": "info",
            "message": (
                "A name, code, or requested enrichment column has no value for some rows: "
                "the graph names nothing for the value, the name read failed, or two codes "
                "of the column's code system share its label. Those cells are left empty "
                "rather than fabricated. measurementMethod_label holds only a name that "
                "differs from the method's dcid, and its empty cells never raise this "
                "warning. Column 'recipient_iso3' left 1 cell empty."
            ),
        }
    ]
    artifact = manifest["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    assert rows[1][rows[0].index("recipient_iso3")] == ""


def test_two_requests_where_only_one_has_gaps_names_only_that_column():
    # Only `provider_label` lacks a graph value; `recipient_iso3` resolves.
    csv_header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,provider,unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
        "scalingFactor,facetId\r\n"
    )
    csv_body = (
        csv_header + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,country/USA,USD,"
        f"GrantEquivalent,NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient", "provider"),
                "measuredProperty": ("amount",),
                "name": _STAT_VAR_NAME,
            },
            "country/KEN": {"countryAlpha3Code": ("KEN",), "name": ("Kenya",)},
            **_NAMED_NODES,
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=csv_body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
    )
    store = FakeDatasetStore()
    artifact_store = FakeArtifactStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(years=[2015]),
    )
    keep = [
        "stat_var",
        "recipient",
        "provider",
        "time_period",
        "value",
        "unit",
        "provenance",
        "provider_label",
        "recipient_iso3",
    ]
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [_STAT_VAR_REF],
            "scope": {
                "entities": {"recipient": ["country/KEN"], "provider": ["country/USA"]},
                "facets": _SCOPE["facets"],
            },
        },
        "options": {"bundle": False, "columns": keep},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    enrichment_warnings = [w for w in manifest["warnings"] if w["code"] == "enrichment_unavailable"]
    assert len(enrichment_warnings) == 1
    assert "provider_label" in enrichment_warnings[0]["message"]
    assert "recipient_iso3" not in enrichment_warnings[0]["message"]


def test_two_entity_columns_each_enrich_with_their_own_field_in_one_request():
    # Both entity columns request their own enrichment field.
    csv_header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,provider,unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
        "scalingFactor,facetId\r\n"
    )
    csv_body = (
        csv_header + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,country/USA,USD,"
        f"GrantEquivalent,NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient", "provider"),
                "measuredProperty": ("amount",),
                "name": _STAT_VAR_NAME,
            },
            "country/KEN": {"countryAlpha3Code": ("KEN",), "name": ("Kenya",)},
            "country/USA": {"name": ("United States",)},
            **_NAMED_NODES,
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=csv_body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
    )
    store = FakeDatasetStore()
    artifact_store = FakeArtifactStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(years=[2015]),
    )
    keep = [
        "stat_var",
        "recipient",
        "provider",
        "time_period",
        "value",
        "unit",
        "provenance",
        "provider_label",
        "recipient_iso3",
    ]
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [_STAT_VAR_REF],
            "scope": {
                "entities": {"recipient": ["country/KEN"], "provider": ["country/USA"]},
                "facets": _SCOPE["facets"],
            },
        },
        "options": {"bundle": False, "columns": keep},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert not any(w["code"] == "enrichment_unavailable" for w in manifest["warnings"])
    artifact = manifest["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    # Each enrichment sits beside its entity column and that entity's name.
    assert rows[0] == [
        "stat_var",
        "stat_var_label",
        "recipient",
        "recipient_label",
        "recipient_iso3",
        "provider",
        "provider_label",
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
    assert rows[1][rows[0].index("provider_label")] == "United States"
    assert rows[1][rows[0].index("recipient_iso3")] == "KEN"


def test_a_genuine_label_role_column_selects_rather_than_colliding_with_the_enrichment_refusal():
    # The bare token selects this frame's role column before enrichment parsing.
    csv_header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,label,unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
        "scalingFactor,facetId\r\n"
    )

    def csv_row(*, year: str, value: str) -> str:
        return (
            f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,country/USA,USD,"
            f"GrantEquivalent,NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
        )

    csv_body = (
        csv_header
        + csv_row(year="2015", value="1000")
        + csv_row(year="2016", value="2000")
        + csv_row(year="2017", value="3000")
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient", "label"),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=csv_body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
    )
    store = FakeDatasetStore()
    artifact_store = FakeArtifactStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(donor_role="label"),
    )
    keep = ["stat_var", "recipient", "label", "time_period", "value", "unit", "provenance"]
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [_STAT_VAR_REF],
            "scope": {
                "entities": {"recipient": ["country/KEN"], "label": ["country/USA"]},
                "facets": _SCOPE["facets"],
            },
        },
        "options": {"bundle": False, "columns": keep},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    fetched = artifact_store.fetch(artifact["url"])
    rows = list(csv.reader(io.StringIO(fetched.decode("utf-8"))))
    assert rows[0] == [
        "stat_var",
        "stat_var_label",
        "recipient",
        "recipient_label",
        "label",
        "label_label",
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
    assert rows[1][rows[0].index("label")] == "country/USA"


def test_selection_runs_before_pivot_so_a_pivot_widened_column_is_not_selectable():
    # Selection runs before pivoting, so the long frame has no `2015` column.
    client = _client()
    body = _body(
        columns=["recipient", "2015"],
        pivot={"index": "recipient", "columns": "time_period"},
    )

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "column_unknown"
    assert problem["pointer"] == "/options/columns"


def test_a_groupby_request_naming_a_role_column_is_a_no_op():
    # Role columns are known but are not summable axes, so this request changes no rows.
    client = _client()
    body = _body(groupby={"by": ["recipient"], "op": "sum"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text


def test_a_groupby_request_naming_a_real_but_undecoded_axis_is_a_no_op():
    # This synthetic StatVar has no indexed code system to collapse; the real CRS path is tested separately.
    client = _client()
    body = _body(groupby={"by": ["crsCooperationModality"], "op": "sum"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text


def test_selection_and_pivot_plans_never_produce_a_reshape_warning():
    # Selection and pivot have no enrichment plan, so neither emits warnings.
    table = Table(
        columns=("stat_var", "recipient", "time_period", "value"),
        rows=(("dcid:MalariaGrantDisbursements", "country/KEN", "2015", 1000),),
    )

    _selected, select_warnings = apply_reshapes(
        table, plan=ReshapePlan(select=("stat_var", "recipient", "time_period", "value"))
    )
    assert select_warnings == ()

    _pivoted, pivot_warnings = apply_reshapes(
        table,
        plan=ReshapePlan(
            pivot=PivotPlan(
                index="recipient", columns="time_period", column_cap=100, cell_cap=1_000
            )
        ),
    )
    assert pivot_warnings == ()


_ROLE_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,donor,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    "scalingFactor,facetId\r\n"
)


def _two_role_client(
    rows: Sequence[tuple[str, str, str]], *, pivot_max_output_cells: int | None = None
):
    """A client over `(recipient, donor, year)` rows of one StatVar and facet."""
    csv_body = _ROLE_CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{recipient},{donor},USD,"
        f"GrantEquivalent,NotApplicable,OECD_DAC2A,{year},{1000 + i},,facet1\r\n"
        for i, (recipient, donor, year) in enumerate(rows)
    )
    recipients = sorted({recipient for recipient, _, _ in rows})
    years_by_pair: dict[tuple[str, str], set[int]] = {}
    for recipient, donor, year in rows:
        years_by_pair.setdefault((donor, recipient), set()).add(int(year))
    coverage = FakeCoverageService(
        build_coverage_index(
            [
                (_STAT_VAR, "facet1", donor, recipient, sorted(years))
                for (donor, recipient), years in years_by_pair.items()
            ],
            role_names={_STAT_VAR: RoleNames(donor="donor", recipient="recipient")},
            facet_metadata={
                "facet1": FacetMetadata(
                    provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
                )
            },
        )
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient", "donor"),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, recipient): [_FACET] for recipient in recipients},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    store = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        settings=(
            DatasetsBuildSettings(pivot_max_output_cells=pivot_max_output_cells)
            if pivot_max_output_cells is not None
            else None
        ),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text=csv_body,
                    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                )
            ),
        ),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage,
    )
    scope = {
        "entities": {
            "recipient": recipients,
            "donor": sorted({donor for _, donor, _ in rows}),
        },
        "facets": _SCOPE["facets"],
    }
    return client, scope


def _two_role_body(scope: dict, *, index: str, columns: str) -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": scope},
        "options": {"bundle": False, "pivot": {"index": index, "columns": columns}},
    }


class _RowReadSpy:
    """Wraps `render.fold_pivot_chunks`, recording whether the fold read any chunk's rows."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rows_read = False
        self.calls = 0
        fold = render_module.fold_pivot_chunks

        def spied(chunks, **kwargs):
            self.calls += 1
            return fold((_RowReadProxy(chunk, self) for chunk in chunks), **kwargs)

        monkeypatch.setattr(render_module, "fold_pivot_chunks", spied)


class _RowReadProxy:
    def __init__(self, chunk: Table, spy: _RowReadSpy) -> None:
        self.columns = chunk.columns
        self._chunk = chunk
        self._spy = spy

    @property
    def rows(self) -> tuple[tuple[object, ...], ...]:
        self._spy.rows_read = True
        return self._chunk.rows


def test_a_pivot_mint_over_the_cell_cap_refuses_from_the_stored_shape_without_reading_rows(
    monkeypatch,
):
    # Two recipients by three output columns is six cells, over a cap of five.
    rows = [
        ("country/KEN", "country/USA", "2015"),
        ("country/KEN", "country/USA", "2016"),
        ("country/TGO", "country/USA", "2015"),
        ("country/TGO", "country/USA", "2016"),
    ]
    client, scope = _two_role_client(rows, pivot_max_output_cells=5)
    spy = _RowReadSpy(monkeypatch)

    resp = client.post(
        "/api/datasets", json=_two_role_body(scope, index="recipient", columns="time_period")
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "pivot_over_cell_cap"
    assert spy.calls == 1
    assert spy.rows_read is False


def test_a_duplicate_key_pivot_refusal_names_the_varying_column_as_an_axis():
    # Two recipients share donor USA in 2015, so (donor, time_period) repeats and recipient varies.
    rows = [
        ("country/KEN", "country/USA", "2015"),
        ("country/KEN", "country/TZA", "2016"),
        ("country/TGO", "country/USA", "2015"),
    ]
    client, scope = _two_role_client(rows)

    resp = client.post(
        "/api/datasets", json=_two_role_body(scope, index="donor", columns="time_period")
    )

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "pivot_not_unique"
    assert [(issue.get("member"), issue.get("axis")) for issue in problem["issues"]] == [
        ("('country/USA', '2015')", None),
        (None, "recipient"),
    ]
