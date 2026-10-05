"""Checks the component data route's bytes, headers, and refusals through the components door."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import time
import zipfile
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import presentation_store.frame_store as frame_store_module
from assembler.fakes import FakeFrameStore, FakeRevisionStore
from assembler.ports import FrameRef, FrameUnavailable
from presentation_api.config import PRESENTATION_COMPONENTS_TIMEOUT_SECONDS

_STAT_CARD_INTENT = {
    "type": "stat_card",
    "frame_ref": {
        "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
        "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
    },
    "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
}


@pytest.fixture
def revision_store() -> FakeRevisionStore:
    return FakeRevisionStore(clock=lambda: datetime(2026, 10, 4, 9, 0, tzinfo=UTC))


def _create(client: TestClient, intent: dict[str, object]) -> str:
    response = client.post("/api/presentation/components", json={"intent": intent})
    assert response.status_code == 201
    return response.json()["component_id"]


def _data_url(component_id: str, revision: int = 1) -> str:
    return f"/api/presentation/components/{component_id}/revisions/{revision}/data"


def test_the_default_download_is_a_bundle_named_for_the_revision(client: TestClient):
    """The bundle links under `DFO_PUBLIC_SITE_URL`, which the test session sets."""
    component_id = _create(client, _STAT_CARD_INTENT)

    response = client.get(_data_url(component_id))
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    descriptor = json.loads(archive.read("datapackage.json"))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == (
        'attachment; filename="dfo_dcid-malariagrantdisbursements-ethiopia_2026-10-04_csv_r1.zip"'
    )
    assert response.headers["cache-control"] == "private, max-age=3600"
    assert "CITATION.cff" in archive.namelist()
    assert "data/dcid-malariagrantdisbursements-ethiopia.csv" in archive.namelist()
    assert descriptor["id"] == f"https://site.example{_data_url(component_id)}"


def test_the_export_loads_its_frame_under_the_door_s_read_deadline(
    client: TestClient, frame_store: FakeFrameStore, monkeypatch: pytest.MonkeyPatch
):
    """The job's `load()` sees the deadline the components door set before it submitted it."""
    component_id = _create(client, _STAT_CARD_INTENT)
    seen: list[float | None] = []
    load = frame_store.load

    def recording_load(*, frame_ref: FrameRef):
        seen.append(frame_store_module._door_deadline.get())
        return load(frame_ref=frame_ref)

    monkeypatch.setattr(frame_store, "load", recording_load)
    timeout = PRESENTATION_COMPONENTS_TIMEOUT_SECONDS
    budget = timeout - min(1.0, timeout * 0.1)

    before = time.monotonic()
    response = client.get(_data_url(component_id))
    after = time.monotonic()

    assert response.status_code == 200
    [deadline] = seen
    assert deadline is not None
    assert before + budget <= deadline <= after + budget


def test_bundle_false_serves_the_bare_file(client: TestClient):
    component_id = _create(client, _STAT_CARD_INTENT)

    response = client.get(_data_url(component_id), params={"format": "csv", "bundle": "false"})
    rows = list(csv.DictReader(io.StringIO(response.text)))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == (
        'attachment; filename="dfo_dcid-malariagrantdisbursements-ethiopia_2026-10-04_r1.csv"'
    )
    assert [(row["time_period"], row["value"]) for row in rows] == [("2024", "441.9")]


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"format": "json"}, id="unknown-format"),
        pytest.param({"bundle": "maybe"}, id="unparseable-bundle"),
    ],
)
def test_a_bad_query_refuses_request_invalid(client: TestClient, params):
    component_id = _create(client, _STAT_CARD_INTENT)

    response = client.get(_data_url(component_id), params=params)

    assert response.status_code == 400
    assert response.json()["code"] == "request_invalid"


def test_a_note_refuses_not_found(client: TestClient):
    component_id = _create(client, {"type": "note_narration", "text": "Disbursements rose."})

    response = client.get(_data_url(component_id))

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert response.json()["retryable"] is False


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(_data_url("cmp_01ARYZ6S41YYYYYYYYYYYYYYYY"), id="unknown-component"),
        pytest.param(None, id="unknown-revision"),
    ],
)
def test_an_unknown_component_or_revision_refuses_not_found(client: TestClient, url):
    if url is None:
        url = _data_url(_create(client, _STAT_CARD_INTENT), revision=2)

    response = client.get(url)

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_a_table_stored_before_row_series_ids_refuses_not_found(
    client: TestClient, revision_store: FakeRevisionStore
):
    component_id = _create(client, {**_STAT_CARD_INTENT, "type": "table"})
    records = revision_store._revisions[component_id]
    payload = records[-1].resolved_payload
    rows = [
        {key: value for key, value in row.items() if key != "series_id"}
        for row in payload["component"]["rows"]
    ]
    stored = {**payload, "component": {**payload["component"], "rows": rows}}
    records[-1] = dataclasses.replace(records[-1], resolved_payload=stored)

    response = client.get(_data_url(component_id))

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert response.json()["retryable"] is False


def test_a_revoked_frame_refuses_frame_unavailable(
    client: TestClient, frame_store: FakeFrameStore, monkeypatch: pytest.MonkeyPatch
):
    component_id = _create(client, _STAT_CARD_INTENT)

    def revoked(*, frame_ref: FrameRef):
        raise FrameUnavailable(reason="revoked", detail=f"{frame_ref.spec_id} was revoked")

    monkeypatch.setattr(frame_store, "load", revoked)
    response = client.get(_data_url(component_id))

    assert response.status_code == 404
    assert response.json()["code"] == "frame_unavailable"
    assert response.json()["retryable"] is False


def test_rows_that_disagree_with_the_card_refuse_internal_error(
    client: TestClient, frame_store: FakeFrameStore, frame_ref: FrameRef
):
    component_id = _create(client, _STAT_CARD_INTENT)
    resolved = frame_store.load(frame_ref=frame_ref)
    columns = resolved.frame["columns"]
    value_index = columns.index("value")
    rows = [[*row[:value_index], 0.0, *row[value_index + 1 :]] for row in resolved.frame["rows"]]
    frame_store.put(
        resolved_frame=dataclasses.replace(resolved, frame={**resolved.frame, "rows": rows})
    )

    response = client.get(_data_url(component_id))

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"


def test_an_xlsx_export_of_a_label_illegal_in_excel_refuses_request_invalid(
    client: TestClient, frame_store: FakeFrameStore, frame_ref: FrameRef
):
    component_id = _create(client, _STAT_CARD_INTENT)
    resolved = frame_store.load(frame_ref=frame_ref)
    labels = {ref: f"{label}\x01" for ref, label in resolved.labels.items()}
    frame_store.put(resolved_frame=dataclasses.replace(resolved, labels=labels))

    response = client.get(_data_url(component_id), params={"format": "xlsx"})

    assert response.status_code == 400
    assert response.json()["code"] == "request_invalid"
    assert "csv" in response.json()["detail"]
