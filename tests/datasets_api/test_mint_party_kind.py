"""A DAC2A mint of France's aid serves each party's kind from the fixture index's party
declaration: a long-layout file writes `financingProvider_kind` and `financingRecipient_kind`
in words by default, and a frame-layout file writes `financingProvider_party_kind` and
`financingRecipient_party_kind` on a bare `party_kind` token."""

from __future__ import annotations

import io

import httpx
import pyarrow.parquet as pq
import pytest

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

_STAT_VAR = "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"
_PROVENANCE = "OECD_DAC2A"
_DONOR = "country/FRA"
# A country, DAC's Africa total, and the African aid no country holds.
_RECIPIENTS = {
    "country/KEN": ("Country",),
    "DACAfrica": ("DACGroup",),
    "DACAfricaUnallocated": ("DACRecipientUnallocatedByCountry",),
}
_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingRecipient,financingProvider,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
    "facetId\r\n"
)
_FACET = Facet(
    facet_id="facet1",
    provenance_id=_PROVENANCE,
    measurement_method="OECD_DAC2A",
    unit="USD",
    obs_count=3,
    earliest_date="2022",
    latest_date="2022",
)


def _client(store: FakeDatasetStore, artifact_store: FakeArtifactStore):
    csv_body = _HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{recipient},{_DONOR},USD,OECD_DAC2A,"
        f"NotApplicable,{_PROVENANCE},2022,{1000 + i},,facet1\r\n"
        for i, recipient in enumerate(_RECIPIENTS)
    )
    coverage = FakeCoverageService(
        build_coverage_index(
            [(_STAT_VAR, "facet1", _DONOR, recipient, [2022]) for recipient in _RECIPIENTS],
            role_names={
                _STAT_VAR: RoleNames(donor="financingProvider", recipient="financingRecipient")
            },
            facet_metadata={
                "facet1": FacetMetadata(
                    provenance_id=_PROVENANCE, measurement_method="OECD_DAC2A", unit="USD"
                )
            },
        )
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("financingProvider", "financingRecipient"),
                "measuredProperty": ("amount",),
            },
            _DONOR: {"typeOf": ("Country",), "name": ("France",)},
            **{dcid: {"typeOf": types} for dcid, types in _RECIPIENTS.items()},
        },
        facets={(_STAT_VAR, recipient): [_FACET] for recipient in _RECIPIENTS},
        labels={_PROVENANCE: "OECD DAC2a"},
    )
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(
            base_url="https://datacommons.one.org/core/api",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text=csv_body,
                    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                )
            ),
        ),
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage,
    )


def _body(options: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [f"dcid:{_STAT_VAR}"],
            "scope": {
                "entities": {
                    "financingProvider": [_DONOR],
                    "financingRecipient": sorted(_RECIPIENTS),
                },
                "facets": {
                    "provenance": {"ref": _PROVENANCE},
                    "measurementMethod": {"ref": "OECD_DAC2A"},
                    "unit": {"ref": "USD"},
                },
            },
        },
        "options": options,
    }


@pytest.mark.parametrize(
    ("options", "suffix", "expected"),
    [
        pytest.param(
            {"format": "parquet", "bundle": False},
            "_kind",
            [
                ("DACAfrica", "total", "official"),
                ("DACAfricaUnallocated", "unallocated", "official"),
                ("country/KEN", "country or organisation", "official"),
            ],
            id="long-layout-default",
        ),
        pytest.param(
            {"format": "parquet", "bundle": False, "layout": "frame", "columns": ["party_kind"]},
            "_party_kind",
            [
                ("DACAfrica", "total", "party"),
                ("DACAfricaUnallocated", "residual", "party"),
                ("country/KEN", "party", "party"),
            ],
            id="frame-layout-bare-token",
        ),
    ],
)
def test_a_dac2a_mint_serves_each_partys_kind_for_every_role(options, suffix, expected):
    artifact_store = FakeArtifactStore()
    client = _client(FakeDatasetStore(), artifact_store)
    body = _body(options)

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    # The recipients have no graph name, so only their label columns can warn.
    assert not any(
        w["code"] == "enrichment_unavailable" and w.get("column", "").endswith(suffix)
        for w in manifest["warnings"]
    )
    table = pq.read_table(io.BytesIO(artifact_store.fetch(manifest["artifacts"][0]["url"])))
    rows = sorted(
        zip(
            table.column("financingRecipient").to_pylist(),
            table.column(f"financingRecipient{suffix}").to_pylist(),
            table.column(f"financingProvider{suffix}").to_pylist(),
            strict=True,
        )
    )
    assert rows == expected


_FRAME = {"format": "parquet", "bundle": False, "layout": "frame"}


@pytest.mark.parametrize(
    ("columns", "written"),
    [
        pytest.param(
            ["party_kind"],
            {"financingProvider_party_kind", "financingRecipient_party_kind"},
            id="kind",
        ),
        pytest.param(
            ["party_class"],
            {"financingProvider_party_class", "financingRecipient_party_class"},
            id="class",
        ),
        pytest.param(
            ["financingProvider_party_whole"], {"financingProvider_party_whole"}, id="one-whole"
        ),
        pytest.param(["financingRecipient_label"], set(), id="no-party-token"),
    ],
)
def test_a_frame_layout_file_writes_only_the_party_cells_its_tokens_ask_for(columns, written):
    """A reader that treats an unknown `{role}_` suffix as a role key must never meet a party
    cell it did not ask for."""
    artifact_store = FakeArtifactStore()
    client = _client(FakeDatasetStore(), artifact_store)

    resp = client.post("/api/datasets", json=_body({**_FRAME, "columns": columns}))

    assert resp.status_code == 200, resp.text
    table = pq.read_table(io.BytesIO(artifact_store.fetch(resp.json()["artifacts"][0]["url"])))
    assert {column for column in table.column_names if "_party_" in column} == written


def test_a_frame_layout_file_reads_each_partys_class_and_whole_cells():
    artifact_store = FakeArtifactStore()
    client = _client(FakeDatasetStore(), artifact_store)

    resp = client.post(
        "/api/datasets", json=_body({**_FRAME, "columns": ["party_class", "party_whole"]})
    )

    assert resp.status_code == 200, resp.text
    table = pq.read_table(io.BytesIO(artifact_store.fetch(resp.json()["artifacts"][0]["url"])))
    rows = sorted(
        zip(
            table.column("financingRecipient").to_pylist(),
            table.column("financingRecipient_party_class").to_pylist(),
            table.column("financingRecipient_party_whole").to_pylist(),
            table.column("financingProvider_party_class").to_pylist(),
            strict=True,
        )
    )
    assert rows == [
        ("DACAfrica", "", "false", "official"),
        ("DACAfricaUnallocated", "residual", "false", "official"),
        ("country/KEN", "official", "false", "official"),
    ]


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(_FRAME, id="frame-layout"),
        pytest.param(
            {
                "format": "csv",
                "bundle": False,
                "pivot": {"index": "financingRecipient", "columns": "time_period"},
            },
            id="bare-pivot",
        ),
    ],
)
def test_a_totals_empty_class_counts_as_no_missing_value(options):
    """The empty `party_class` of a total with no single covered class is its value, so no
    layout warns `enrichment_unavailable` for it."""
    client = _client(FakeDatasetStore(), FakeArtifactStore())

    resp = client.post(
        "/api/datasets", json=_body({**options, "columns": ["party_class", "party_whole"]})
    )

    assert resp.status_code == 200, resp.text
    assert [w for w in resp.json()["warnings"] if w["code"] == "enrichment_unavailable"] == []


def test_a_long_layout_file_answers_party_field_tokens_with_its_kind_column_once():
    artifact_store = FakeArtifactStore()
    client = _client(FakeDatasetStore(), artifact_store)
    columns = ["party_kind", "financingRecipient_party_class", "financingProvider_kind"]

    resp = client.post(
        "/api/datasets", json=_body({"format": "parquet", "bundle": False, "columns": columns})
    )

    assert resp.status_code == 200, resp.text
    table = pq.read_table(io.BytesIO(artifact_store.fetch(resp.json()["artifacts"][0]["url"])))
    kind_columns = [column for column in table.column_names if column.endswith("_kind")]
    assert kind_columns == ["financingProvider_kind", "financingRecipient_kind"]
    assert not {
        f"{role}_{field}"
        for role in ("financingProvider", "financingRecipient")
        for field in ("party_kind", "party_class", "party_whole")
    } & set(table.column_names)
