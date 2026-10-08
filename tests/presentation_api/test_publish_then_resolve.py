"""Publishes a frame through the Datasets publish door and resolves cards over it through the
Presentation write door, both doors mounted on one app over one SQLite-backed store.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from assembler.catalogue import CatalogueRelease
from assembler.fakes import FakeReleaseRegistry, FakeRevisionStore
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.doors.publish import build_publish_door
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.version_store import LiveVersionRecordWriter
from dfo_platform.app import Surface, create_app
from presentation_api.app import presentation_surface
from presentation_store.frame_store import LiveFrameStore
from tests.artifact_store.conftest import FakeGCSClient
from tests.datasets_api._version_record_store import FakeVersionRecordDatasetStore
from tests.presentation_store.test_frame_store import _FakeGraphReader, _FakeReferenceWriter

_PRINCIPAL = "publisher-principal"
_FRAME_REF = {"spec_id": "ops_" + "a" * 32, "content_version": "ocv_" + "b" * 32}
_FRAME_UNIT = {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}}
_PAYLOAD_UNIT = {
    "numerator": "US Dollar",
    "denominator": {"stat_var": "dcid:Count_Person", "label": "Total population"},
}
_PUBLISH_BODY = {
    "schema_version": 1,
    **_FRAME_REF,
    "role_keys": ["recipient"],
    "rows": [
        {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
            "time_period": period,
            "value": value,
            "unit": _FRAME_UNIT,
        }
        for period, value in (("2023", 3.1), ("2024", 3.34))
    ],
    "input_content_versions": ["cv_" + "c" * 32],
}


@dataclass(frozen=True)
class _Deployment:
    client: TestClient
    artifact_store: LiveArtifactStore


@pytest.fixture
def deployment(release: CatalogueRelease) -> Iterator[_Deployment]:
    store = FakeVersionRecordDatasetStore()
    artifact_store = LiveArtifactStore(
        buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
        inventory=LiveArtifactInventory(store=store),
        client=FakeGCSClient(),
    )
    snapshot_reader = LiveSnapshotReader(store=store)
    publish_door = build_publish_door(
        artifact_store=artifact_store,
        snapshot_reader=snapshot_reader,
        version_writer=LiveVersionRecordWriter(store=store),
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    datasets = Surface(
        name="datasets",
        prefix="/api/datasets",
        router=publish_door.router,
        capability_version=DATASETS_CAPABILITY_VERSION,
        codes=DATASETS_CODE_CATALOGUE,
        classify=publish_door.classify,
        exception_types=publish_door.exception_types,
        exception_exemplars=publish_door.exception_exemplars,
        middleware=publish_door.middleware,
    )
    presentation = presentation_surface(
        frame_store=LiveFrameStore(
            snapshots=snapshot_reader,
            artifacts=artifact_store,
            graph=_FakeGraphReader(
                labels={
                    "USDollar": "US Dollar",
                    "Count_Person": "Total population",
                    "Amount_GDP": "Nominal GDP",
                    "Amount_ODA": "Aid",
                    "country/ETH": "Ethiopia",
                }
            ),
            reference_writer=_FakeReferenceWriter(),
        ),
        revision_store=FakeRevisionStore(),
        release_registry=FakeReleaseRegistry(release=release),
    )
    try:
        with TestClient(
            create_app([datasets, presentation]), raise_server_exceptions=False
        ) as client:
            yield _Deployment(client=client, artifact_store=artifact_store)
    finally:
        store.close()


def test_a_published_composed_unit_frame_resolves_to_cards_naming_both_sides(
    deployment: _Deployment,
):
    published = deployment.client.post(
        "/api/datasets/frames/publish", json=_PUBLISH_BODY, headers={"x-dfo-principal": _PRINCIPAL}
    )
    assert published.status_code == 200, published.text
    document = json.loads(
        deployment.artifact_store.get(
            artifact_digest=published.json()["published"]["artifact_digest"], tier=Tier.working
        )
    )
    unit_index = document["columns"].index("unit")
    assert [row[unit_index] for row in document["rows"]] == [_FRAME_UNIT, _FRAME_UNIT]

    select = {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"}
    components = {}
    for member in ("stat_card", "table", "line_chart", "bar_chart"):
        created = deployment.client.post(
            "/api/presentation/components",
            json={"intent": {"type": member, "frame_ref": _FRAME_REF, "select": select}},
        )
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["fact_summary"]["unit"] == _PAYLOAD_UNIT
        components[member] = body["resolved_payload"]["component"]

    assert components["stat_card"]["unit"] == _PAYLOAD_UNIT
    assert [row["unit"] for row in components["table"]["rows"]] == [_PAYLOAD_UNIT] * 2
    for chart in ("line_chart", "bar_chart"):
        assert [series["unit"] for series in components[chart]["series"]] == [_PAYLOAD_UNIT]


def test_a_published_composed_stat_var_names_its_series_on_every_card(deployment: _Deployment):
    """The frame's rows sit under a StatVar the graph does not hold. The entry published with
    it gives the series a label on a chart, a stat card and a table export."""
    stat_var = "dcid:ops_ae6d604db16b8e2378c4a2deb8acf943"
    label = "Nominal GDP per Total population"
    published = deployment.client.post(
        "/api/datasets/frames/publish",
        json={
            **_PUBLISH_BODY,
            "role_keys": [],
            "rows": [
                {"stat_var": stat_var, "time_period": period, "value": value}
                for period, value in (("2023", 3.1), ("2024", 3.34))
            ],
            "composed_stat_vars": [
                {
                    "stat_var": stat_var,
                    "relation": "quotient",
                    "first": ["dcid:Amount_GDP"],
                    "second": ["dcid:Count_Person"],
                }
            ],
        },
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert published.status_code == 200, published.text

    components = {}
    for member in ("line_chart", "stat_card", "table"):
        created = deployment.client.post(
            "/api/presentation/components",
            json={
                "intent": {
                    "type": member,
                    "frame_ref": _FRAME_REF,
                    "select": {"stat_var": stat_var},
                }
            },
        )
        assert created.status_code == 201, created.text
        components[member] = created.json()

    line_chart = components["line_chart"]["resolved_payload"]["component"]
    assert [series["label"] for series in line_chart["series"]] == [label]
    assert line_chart["y_axis"]["label"] == label
    assert components["stat_card"]["resolved_payload"]["component"]["label"] == label
    assert [one["label"] for one in components["line_chart"]["fact_summary"]["series"]] == [label]
    table = components["table"]
    exported = deployment.client.get(
        f"/api/presentation/components/{table['component_id']}/revisions/{table['revision']}/data",
        params={"format": "csv", "bundle": "false"},
    )
    assert exported.status_code == 200, exported.text
    rows = list(csv.DictReader(io.StringIO(exported.text)))
    assert [row["series"] for row in rows] == [label] * 2
    assert [row["stat_var_label"] for row in rows] == [label] * 2


@pytest.mark.parametrize(
    ("second", "label"),
    [
        (["dcid:Amount_GDP"], "Aid as a share of Nominal GDP"),
        (["dcid:Amount_ODA"], "Share of Aid"),
    ],
    ids=["two-sides", "equal-sides"],
)
def test_a_published_share_names_its_series_and_carries_the_percent_unit(
    deployment: _Deployment, second: list[str], label: str
):
    """A share entry passes the publish door and words the series on a chart, a stat card and
    the fact summary. A bare `Percent` unit cell reads `Percent` on a line chart's series and in
    the fact summary."""
    stat_var = "dcid:ops_ae6d604db16b8e2378c4a2deb8acf943"
    published = deployment.client.post(
        "/api/datasets/frames/publish",
        json={
            **_PUBLISH_BODY,
            "role_keys": [],
            "rows": [
                {"stat_var": stat_var, "time_period": period, "value": value, "unit": "Percent"}
                for period, value in (("2023", 4.1), ("2024", 5.33))
            ],
            "composed_stat_vars": [
                {
                    "stat_var": stat_var,
                    "relation": "share",
                    "first": ["dcid:Amount_ODA"],
                    "second": second,
                }
            ],
        },
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert published.status_code == 200, published.text

    components = {}
    for member in ("line_chart", "stat_card"):
        created = deployment.client.post(
            "/api/presentation/components",
            json={
                "intent": {
                    "type": member,
                    "frame_ref": _FRAME_REF,
                    "select": {"stat_var": stat_var},
                }
            },
        )
        assert created.status_code == 201, created.text
        components[member] = created.json()

    line_chart = components["line_chart"]["resolved_payload"]["component"]
    assert [series["label"] for series in line_chart["series"]] == [label]
    assert [series["unit"] for series in line_chart["series"]] == ["Percent"]
    assert components["stat_card"]["resolved_payload"]["component"]["label"] == label
    fact_summary = components["line_chart"]["fact_summary"]
    assert fact_summary["unit"] == "Percent"
    assert [one["label"] for one in fact_summary["series"]] == [label]


def test_a_composed_stat_var_beside_a_graph_one_leads_its_series_label(deployment: _Deployment):
    """A chart over two StatVars keyed by `recipient` labels each line `<stat var> / <entity>`.
    The composed StatVar's lines lead with its worded label, and no label in a chart or a table
    holds its ref."""
    composed = "dcid:ops_ae6d604db16b8e2378c4a2deb8acf943"
    published = deployment.client.post(
        "/api/datasets/frames/publish",
        json={
            **_PUBLISH_BODY,
            "rows": [
                {
                    "stat_var": stat_var,
                    "recipient": "country/ETH",
                    "time_period": period,
                    "value": value,
                }
                for stat_var in ("dcid:Amount_ODA", composed)
                for period, value in (("2023", 3.1), ("2024", 3.34))
            ],
            "composed_stat_vars": [
                {
                    "stat_var": composed,
                    "relation": "quotient",
                    "first": ["dcid:Amount_ODA"],
                    "second": ["dcid:Count_Person"],
                }
            ],
        },
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert published.status_code == 200, published.text

    components = {}
    for member in ("line_chart", "table"):
        created = deployment.client.post(
            "/api/presentation/components",
            json={
                "intent": {
                    "type": member,
                    "frame_ref": _FRAME_REF,
                    "select": {"stat_var": ["dcid:Amount_ODA", composed]},
                }
            },
        )
        assert created.status_code == 201, created.text
        components[member] = created.json()["resolved_payload"]["component"]

    line_chart, table = components["line_chart"], components["table"]
    assert [series["label"] for series in line_chart["series"]] == [
        "Aid / Ethiopia",
        "Aid per Total population / Ethiopia",
    ]
    shown = [
        *(series["label"] for series in line_chart["series"]),
        line_chart["y_axis"]["label"],
        line_chart["title"]["text"],
        *(cell for row in table["rows"] for cell in row["dimensions"].values()),
    ]
    assert {row["dimensions"]["stat_var"] for row in table["rows"]} == {
        "Aid",
        "Aid per Total population",
    }
    assert not [text for text in shown if "dcid:ops_" in text]


@pytest.mark.parametrize(
    ("structured", "key_text"),
    [
        (_FRAME_UNIT, '{"denominator":{"stat_var":"dcid:Count_Person"},"numerator":"USDollar"}'),
        ({"currency": "USD", "prices": "current"}, "USD/current"),
    ],
    ids=["composed", "converted"],
)
def test_a_frame_whose_unit_ref_spells_a_structured_unit_s_key_is_not_stored(
    deployment: _Deployment, structured: dict[str, object], key_text: str
):
    """Grouping would put the structured rows under the string rows' unit, so no reader opens
    such a frame and the publish door refuses to store it."""
    rows = [
        {**row, "unit": unit, "value": value}
        for unit, value in ((structured, 1.0), (key_text, 99.0))
        for row in _PUBLISH_BODY["rows"]
    ]
    published = deployment.client.post(
        "/api/datasets/frames/publish",
        json={**_PUBLISH_BODY, "rows": rows},
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert published.status_code == 400, published.text
    assert published.json()["code"] == "request_invalid"
    assert published.json()["pointer"] == "/rows"

    view = deployment.client.get(
        f"/api/presentation/frames/{_FRAME_REF['spec_id']}/{_FRAME_REF['content_version']}"
    )
    assert view.status_code == 404, view.text
    assert not deployment.artifact_store._client.buckets
