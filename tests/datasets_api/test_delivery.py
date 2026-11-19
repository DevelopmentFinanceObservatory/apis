"""Each door that serves data logs one `datasets.delivery` line naming who it served and what
(`datasets_api.delivery`), and a vouched Studio caller reaches the allowance doors."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from dataset_store.snapshot import ManifestFacts
from datasets_api.delivery import (
    MEASURES_CAP,
    Delivery,
    delivering,
    log_delivery,
    record_analysis,
    record_file,
    record_manifest,
    record_rows_served,
)
from datasets_api.manifest_models import Manifest
from dfo_platform.caller import (
    UNRESOLVED_IDENTITY,
    reset_delivery_identity,
    set_delivery_identity,
)
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR

from .test_exports import (
    _CALLER_HEADERS,
    _KENYA,
    _PLACE_STAT_VAR,
    _Rig,
    caller_key,  # noqa: F401 -- autouse: configures the caller digest key
)

_LOGGER = "datasets_api.delivery"
_PRINCIPAL = "website-secret"
_CONVERSATION = "conversation:" + "cd" * 16
_STUDIO_HEADERS = {
    **_CALLER_HEADERS,
    "x-dfo-caller-principal": _PRINCIPAL,
    "x-dfo-caller": _CONVERSATION,
}
_FIELDS = {
    "door",
    "trace_id",
    "admission",
    "caller",
    "caller_source",
    "upstream",
    "surface",
    "client",
    "user_agent",
    "spec_id",
    "content_version",
    "export_id",
    "measures",
    "measures_truncated",
    "measure_count",
    "series_count",
    "rows_available",
    "rows_served",
}


def _lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    """The structured fields of every delivery line, each holding exactly `_FIELDS`."""
    lines = [
        fields
        for record in caplog.records
        if record.name == _LOGGER and (fields := getattr(record, JSON_FIELDS_ATTR, None))
    ]
    for fields in lines:
        assert set(fields) == _FIELDS
    return lines


def _place_mint_body() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [f"dcid:{_PLACE_STAT_VAR}"],
            "scope": {
                "entities": {"observationAbout": [_KENYA]},
                "facets": {
                    "provenance": {"ref": "WorldBank"},
                    "measurementMethod": {"ref": "dc-aggregate"},
                    "unit": {"ref": "Years"},
                },
            },
        },
        "options": {"bundle": False},
    }


@pytest.fixture
def vouching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the principal that vouches for `x-dfo-caller`."""
    monkeypatch.setenv("DFO_CALLER_PRINCIPAL", _PRINCIPAL)


def test_a_mint_then_an_observations_read_serve_the_rows_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        rig = _Rig("fetched", monkeypatch, dataset="place")
        frame = rig.frame
        observations = rig.client.get(
            f"/api/datasets/{frame['spec_id']}/{frame['content_version']}/observations",
            headers=_CALLER_HEADERS,
        )

    assert observations.status_code == 200, observations.text
    points = sum(len(series["points"]) for series in observations.json()["series"])
    assert points == 2
    mint, read = _lines(caplog)
    assert (mint["door"], read["door"]) == ("mint", "observations")
    assert mint["measures"] == [_PLACE_STAT_VAR]
    assert read["measures"] is None
    for line in (mint, read):
        assert line["spec_id"] == frame["spec_id"]
        assert line["content_version"] == frame["content_version"]
        assert line["measure_count"] == 1
        assert line["rows_available"] == rig.coverage_rows
    assert mint["rows_served"] + read["rows_served"] == points


def test_a_background_mint_logs_admission_background(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    rig = _Rig("fetched", monkeypatch, dataset="place")

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        resp = rig.client.post(
            "/api/datasets",
            json=_place_mint_body(),
            headers={**_CALLER_HEADERS, "X-DFO-Mint-Admission": "background"},
        )

    assert resp.status_code == 200, resp.text
    (line,) = _lines(caplog)
    assert line["door"] == "mint"
    assert line["admission"] == "background"


@pytest.mark.parametrize(
    ("headers", "surface", "caller", "source"),
    [
        pytest.param(
            {**_CALLER_HEADERS, "x-dfo-surface": "studio"},
            "direct",
            "address",
            "address",
            id="unvouched-studio-is-direct",
        ),
        pytest.param(
            {**_STUDIO_HEADERS, "x-dfo-surface": "api"},
            "studio",
            _CONVERSATION,
            "vouched",
            id="conversation-overrules-vouched-api",
        ),
    ],
)
@pytest.mark.usefixtures("vouching")
def test_the_surface_is_believed_only_beside_the_principal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    caller_key: str,  # noqa: F811
    headers: dict[str, str],
    surface: str,
    caller: str,
    source: str,
):
    rig = _Rig("fetched", monkeypatch, dataset="place")
    frame = rig.frame

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        resp = rig.client.get(
            f"/api/datasets/{frame['spec_id']}/{frame['content_version']}", headers=headers
        )

    assert resp.status_code == 200, resp.text
    (line,) = _lines(caplog)
    assert line["door"] == "snapshot"
    assert line["surface"] == surface
    assert line["caller"] == (caller_key if caller == "address" else caller)
    assert line["caller_source"] == source
    assert line["measures"] == [_PLACE_STAT_VAR]
    assert line["rows_served"] == 0


@pytest.mark.usefixtures("vouching")
def test_a_studio_caller_prepares_links_and_pins(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A vouched `conversation:` caller names no allowance of its own, so the allowance doors
    answer it as they answer the upstream's own address."""
    rig = _Rig("fetched", monkeypatch, dataset="place")
    frame = rig.frame
    path = f"/api/datasets/{frame['spec_id']}/{frame['content_version']}"
    [artifact] = rig.client.get(path).json()["artifacts"][:1]

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        prepared = rig.client.post(
            "/api/datasets/exports",
            json={"frame": frame, "format": "csv", "bundle": False},
            headers=_STUDIO_HEADERS,
        )
        link = rig.client.get(
            f"/api/datasets/exports/{prepared.json()['export_id']}",
            headers=_STUDIO_HEADERS,
            follow_redirects=False,
        )
        pinned = rig.client.post(
            f"{path}/pin",
            json={"schema_version": 1, "artifact_digest": artifact["artifact_digest"]},
            headers=_STUDIO_HEADERS,
        )

    assert prepared.status_code == 200, prepared.text
    assert link.status_code == 302, link.text
    assert pinned.status_code == 200, pinned.text
    export, signed = _lines(caplog)
    assert (export["door"], signed["door"]) == ("export", "export_link")
    for line in (export, signed):
        assert line["export_id"] == prepared.json()["export_id"]
        assert line["caller"] == _CONVERSATION
        assert line["surface"] == "studio"
        assert line["rows_available"] == prepared.json()["rows"]
    assert (export["rows_served"], signed["rows_served"]) == (0, prepared.json()["rows"])


@pytest.mark.usefixtures("vouching")
def test_a_studio_caller_runs_an_analysis(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    rig = _Rig("derived", monkeypatch)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        resp = rig.client.post(
            "/api/datasets/analysis",
            json={"tree": rig.frame, "latest": True},
            headers=_STUDIO_HEADERS,
        )

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    (line,) = _lines(caplog)
    member = answer["ref"]["members"][0]
    assert line["door"] == "analysis"
    assert (line["spec_id"], line["content_version"]) == (
        member["spec"]["spec_id"],
        member["content_version"],
    )
    assert line["spec_id"].startswith("ops_")
    assert (line["caller"], line["surface"], line["user_agent"]) == (_CONVERSATION, "studio", None)
    assert line["measures"] is None
    assert line["measure_count"] == answer["stat_var_count"]
    assert (line["rows_available"], line["rows_served"]) == (answer["rows"], 0)


def test_a_derived_file_serves_the_frames_rows(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    rig = _Rig("derived", monkeypatch)
    frame = rig.frame

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        resp = rig.client.get(
            f"/api/datasets/derived/{frame['spec_id']}/{frame['content_version']}/file"
            "?format=csv&bundle=false",
            headers=_CALLER_HEADERS,
        )

    assert resp.status_code == 200, resp.text
    rows = len(resp.text.strip().splitlines()) - 1
    (line,) = _lines(caplog)
    assert line["door"] == "derived_file"
    assert (line["spec_id"], line["content_version"]) == (
        frame["spec_id"],
        frame["content_version"],
    )
    assert (line["rows_available"], line["rows_served"]) == (rows, rows)
    assert line["measures"] is None


def test_a_file_selector_read_serves_the_versions_rows(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    rig = _Rig("fetched", monkeypatch, dataset="place")
    frame = rig.frame

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        resp = rig.client.get(
            f"/api/datasets/{frame['spec_id']}/{frame['content_version']}/file"
            "?format=parquet&bundle=false",
            headers={
                **_CALLER_HEADERS,
                "X-DFO-Client": "dfo-python/0.1.0 (CPython 3.14)",
                "User-Agent": "dfo-python/0.1.0 (CPython 3.14)",
            },
            follow_redirects=False,
        )

    assert resp.status_code == 302, resp.text
    (line,) = _lines(caplog)
    assert line["door"] == "file_selector"
    assert line["client"] == "dfo-python/0.1.0"
    assert line["user_agent"] == "dfo-python/0.1.0 (CPython 3.14)"
    assert line["surface"] == "direct"
    assert line["rows_served"] == rig.coverage_rows
    assert line["measures"] is None


_FILE = {"format": "csv", "bundle": True, "layout": "2"}


@pytest.mark.parametrize(
    ("rendering", "rows_served"),
    [
        pytest.param(_FILE, 7, id="plain-bundle"),
        pytest.param({"format": "parquet", "bundle": False, "layout": "frame"}, 7, id="frame"),
        pytest.param({"statistics": "series"}, 0, id="statistics"),
        pytest.param({**_FILE, "groupby": {"by": ["year"]}}, None, id="groupby"),
        pytest.param({**_FILE, "pivot": {"columns": "year"}}, None, id="pivot"),
        pytest.param({**_FILE, "columns": ["year"]}, 7, id="columns-keeps-rows"),
        pytest.param({**_FILE, "without_party_totals": True}, None, id="party-cut"),
        pytest.param({**_FILE, "bundle": False}, None, id="bare-long-csv-drops-totals"),
    ],
)
def test_a_file_serves_the_versions_rows_only_when_its_rendering_keeps_them(
    rendering: dict[str, object], rows_served: int | None
):
    stored = SimpleNamespace(
        subject_id="spec_x",
        content_version="cv",
        facts=ManifestFacts(
            prepared_at=datetime(2026, 1, 1, tzinfo=UTC),
            definition={"spec_id": "spec_x"},
            coverage={"rows": 7},
            citation={},
            trace={},
            warnings=(),
        ),
    )

    with delivering() as delivery:
        record_file(stored, SimpleNamespace(rendering=rendering))  # ty: ignore[invalid-argument-type]

    assert (delivery.rows_available, delivery.rows_served) == (7, rows_served)


def test_a_recording_fault_warns_and_leaves_the_call_running(caplog: pytest.LogCaptureFixture):
    # The answer lacks `content_version`. The spec id is readable, and the line must stay empty.
    answer = {
        "ref": {"members": [{"spec": {"spec_id": "ops_x"}}]},
        "rows": 4,
        "stat_var_count": 1,
    }
    with caplog.at_level(logging.INFO, logger=_LOGGER), delivering() as delivery:
        record_analysis(answer)
        record_rows_served(3)
        log_delivery("analysis", delivery)

    assert delivery.rows_served == 3
    assert delivery.spec_id is None
    assert _lines(caplog) == []
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.WARNING]


def test_a_long_statvar_list_is_cut_at_the_cap(caplog: pytest.LogCaptureFixture):
    manifest = _manifest_with_stat_vars(MEASURES_CAP + 1)
    token = set_delivery_identity(UNRESOLVED_IDENTITY)
    try:
        with caplog.at_level(logging.INFO, logger=_LOGGER), delivering() as delivery:
            record_manifest(manifest)
            log_delivery("mint", delivery)
    finally:
        reset_delivery_identity(token)

    (line,) = _lines(caplog)
    assert len(line["measures"]) == MEASURES_CAP
    assert line["measures_truncated"] is True
    assert line["measure_count"] == MEASURES_CAP + 1


def test_a_call_that_recorded_no_spec_logs_nothing(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_delivery("mint", Delivery())

    assert _lines(caplog) == []


def _manifest_with_stat_vars(n: int) -> Manifest:
    return Manifest.model_construct(
        definition=SimpleNamespace(spec_id="spec_x"),
        content_version="cv",
        coverage=SimpleNamespace(rows=3),
        series_count=1,
        trace={
            "resolved_stat_vars": [
                {"ref": f"dcid:Var{i:04d}", "roles": [], "observation_count": 1} for i in range(n)
            ]
        },
    )
