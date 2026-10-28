"""Checks component writes, reads, and refusal responses through the API door."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import presentation_api.doors.components as components_door
from assembler.catalogue import CatalogueRelease
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import FrameRef, FrameUnavailable, ResolvedFrame
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_resolved_frame,
    malaria_ethiopia_rows,
)
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform.app import create_app
from presentation_api.app import presentation_surface
from presentation_store import GeneratedIdCollision, PresentationStoreUnavailable
from spec_registry.errors import SpecRegistryStoreError, SpecRegistryUnavailable
from tests.presentation_store.test_frame_store import _FakeSpecReader, _store


def _release_excluding(release: CatalogueRelease, member: str) -> CatalogueRelease:
    """Return a release that omits `member` from every served-member map."""
    return dataclasses.replace(
        release,
        component_schemas={k: v for k, v in release.component_schemas.items() if k != member},
        intent_schemas={k: v for k, v in release.intent_schemas.items() if k != member},
        intent_schemas_tolerant={
            k: v for k, v in release.intent_schemas_tolerant.items() if k != member
        },
        minimal_intents={k: v for k, v in release.minimal_intents.items() if k != member},
        encoding_classes={k: v for k, v in release.encoding_classes.items() if k != member},
    )


_LEGACY_SESSION_HEADER = "x-dfo-session"
"""The header name older clients send. The API ignores it."""

_LINE_CHART_INTENT = {
    "type": "line_chart",
    "frame_ref": {
        "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
        "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
    },
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
    "window": {"start": "2015", "end": "2024"},
    "title": "Malaria grant disbursements to Ethiopia",
}

_STAT_CARD_INTENT = {
    "type": "stat_card",
    "frame_ref": {
        "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
        "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
    },
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
}

_NOTE_NARRATION_INTENT = {
    "type": "note_narration",
    "text": "Disbursements rose each year after 2020.",
}

_SOURCE_CARD_INTENT = {
    "type": "source_card",
    "frame_ref": {
        "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
        "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
    },
}


def _create(client: TestClient, *, intent: dict | None = None):
    return client.post(
        "/api/presentation/components",
        json={"intent": intent or _LINE_CHART_INTENT},
    )


def test_create_returns_201_with_a_minted_component_and_the_rest_doors_own_affordance(
    client: TestClient,
):
    resp = _create(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["component_id"].startswith("cmp_")
    assert body["revision"] == 1
    assert body["composition_id"] is None
    assert body["composition_revision"] is None
    assert body["changed"] == []
    component_id = body["component_id"]
    assert body["affordance"] == (
        f"Revise this component with PATCH /api/presentation/components/{component_id}, "
        "sending only the fields to change and this reply's write_token as x-dfo-write-token, "
        f"or read it back with GET /api/presentation/components/{component_id}."
    )
    assert body["write_token"].startswith("cmw_")
    assert "composition_write_token" not in body
    assert body["fact_summary"]["series"]


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        ({"start": "2018"}, {"start": "2018", "end": "2024"}),
        ({"end": "2018"}, {"start": "2015", "end": "2018"}),
    ],
)
def test_a_one_sided_window_runs_to_the_edge_of_the_frames_coverage(
    client: TestClient, window: dict[str, str], expected: dict[str, str]
):
    """The intent grammar lets a window omit a side: it runs to the edge of
    the frame's coverage (2015..2024 here)."""
    resp = _create(client, intent={**_LINE_CHART_INTENT, "window": window})
    assert resp.status_code == 201
    assert resp.json()["fact_summary"]["window"] == expected


def test_create_in_a_new_composition(client: TestClient):
    intent = dict(_LINE_CHART_INTENT)
    resp = client.post(
        "/api/presentation/components",
        json={"intent": intent, "composition_id": "new"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["composition_id"].startswith("cps_")
    assert body["composition_revision"] == 1


def test_a_no_op_revise_and_a_get_on_a_composition_member_report_the_real_id_and_revision(
    client: TestClient,
):
    created = client.post(
        "/api/presentation/components",
        json={"intent": dict(_LINE_CHART_INTENT), "composition_id": "new"},
    ).json()
    component_id = created["component_id"]
    composition_id = created["composition_id"]

    no_op = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": _LINE_CHART_INTENT["title"]}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert no_op.status_code == 200
    no_op_body = no_op.json()
    assert no_op_body["changed"] == []
    assert no_op_body["composition_id"] == composition_id
    assert no_op_body["composition_revision"] == created["composition_revision"]

    got = client.get(f"/api/presentation/components/{component_id}")
    assert got.status_code == 200
    got_body = got.json()
    assert got_body["composition_id"] == composition_id
    assert got_body["composition_revision"] == created["composition_revision"]


def test_revise_reports_changed_fields_and_bumps_the_revision(client: TestClient):
    created = _create(client).json()
    component_id = created["component_id"]

    resp = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "A new title"}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 2
    assert body["changed"] == [
        {"field": "title", "prior": _LINE_CHART_INTENT["title"], "new": "A new title"}
    ]
    assert "affordance" not in body


def test_a_no_op_revise_reports_the_current_revision_and_no_changes(client: TestClient):
    created = _create(client).json()
    component_id = created["component_id"]

    resp = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": _LINE_CHART_INTENT["title"]}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == created["revision"]
    assert body["changed"] == []


def test_get_component_reads_back_a_pinned_revision(client: TestClient):
    created = _create(client).json()
    component_id = created["component_id"]
    client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "A new title"}},
        headers={"x-dfo-write-token": created["write_token"]},
    )

    resp = client.get(
        f"/api/presentation/components/{component_id}",
        params={"revision": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 1
    assert body["stored_intent"]["title"] == _LINE_CHART_INTENT["title"]
    assert body["composition_id"] is None
    assert body["composition_revision"] is None

    current = client.get(f"/api/presentation/components/{component_id}")
    assert current.json()["revision"] == 2


def test_create_revise_and_both_gets_serve_the_stored_envelope(
    client: TestClient, revision_store: FakeRevisionStore
):
    """POST, PATCH, and both GET forms return the stored payload verbatim."""
    created = _create(client).json()
    component_id = created["component_id"]
    revised = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "A new title"}},
        headers={"x-dfo-write-token": created["write_token"]},
    ).json()

    def _get(**params):
        resp = client.get(
            f"/api/presentation/components/{component_id}",
            params=params,
        )
        assert resp.status_code == 200
        return resp.json()["resolved_payload"]

    def _stored(revision):
        record = revision_store.read_revision(component_id=component_id, revision=revision)
        assert record is not None
        return record.resolved_payload

    assert created["resolved_payload"] == _stored(1)
    assert revised["resolved_payload"] == _stored(2)
    assert _get(revision=1) == _stored(1)
    assert _get(revision=2) == _stored(2)
    assert _get() == _stored(2)
    assert _stored(1)["revision"] == 1
    assert _stored(2)["revision"] == 2
    assert _stored(2)["component"]["title"]["text"] == "A new title"


def _create_and_revise_once(client: TestClient) -> dict:
    """A two-revision component: a create, then one revise. Returns the create body."""
    created = _create(client).json()
    component_id = created["component_id"]
    client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "A new title"}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    return created


def test_get_component_at_the_head_revision_number_succeeds(client: TestClient):
    created = _create_and_revise_once(client)
    component_id = created["component_id"]
    resp = client.get(
        f"/api/presentation/components/{component_id}",
        params={"revision": 2},
    )
    assert resp.status_code == 200
    assert resp.json()["revision"] == 2


def test_get_component_one_past_the_head_is_not_found(client: TestClient):
    created = _create_and_revise_once(client)
    component_id = created["component_id"]
    resp = client.get(
        f"/api/presentation/components/{component_id}",
        params={"revision": 3},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert body["retryable"] is False


def test_get_component_at_revision_999_is_not_found(client: TestClient):
    created = _create_and_revise_once(client)
    component_id = created["component_id"]
    resp = client.get(
        f"/api/presentation/components/{component_id}",
        params={"revision": 999},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert body["retryable"] is False


@pytest.mark.parametrize("revision", [0, -1])
def test_get_component_at_a_non_positive_revision_is_request_invalid(
    client: TestClient, revision: int
):
    """`revision` below 1 fails query validation and has no body pointer."""
    created = _create_and_revise_once(client)
    component_id = created["component_id"]
    resp = client.get(
        f"/api/presentation/components/{component_id}",
        params={"revision": revision},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False


def test_a_revise_of_a_component_that_vanished_mid_flight_gets_edit_conflict(
    client: TestClient, revision_store: FakeRevisionStore
):
    """A vanished component yields `edit_conflict`. A moved revision is rebased."""
    created = _create_and_revise_once(client)
    component_id = created["component_id"]
    revision_store._revisions[component_id] = []

    resp = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "one more"}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "edit_conflict"


def test_stat_card_create(client: TestClient):
    resp = _create(client, intent=_STAT_CARD_INTENT)
    assert resp.status_code == 201
    assert resp.json()["fact_summary"]["series"]


def test_note_narration_create(client: TestClient, revision_store: FakeRevisionStore):
    """The stored note narration has model authorship and no input sources."""
    resp = _create(client, intent=_NOTE_NARRATION_INTENT)
    assert resp.status_code == 201
    body = resp.json()

    record = revision_store.read_revision(component_id=body["component_id"], revision=1)
    envelope = record.resolved_payload
    assert envelope["component"]["type"] == "note_narration"
    assert envelope["component"]["authorship"] == "model"
    assert envelope["provenance"]["inputs"] == []
    assert envelope["provenance"].get("sources", []) == []


def test_source_card_create(client: TestClient, revision_store: FakeRevisionStore):
    resp = _create(client, intent=_SOURCE_CARD_INTENT)
    assert resp.status_code == 201
    body = resp.json()

    record = revision_store.read_revision(component_id=body["component_id"], revision=1)
    envelope = record.resolved_payload
    provenance_ids = {source["id"] for source in envelope["provenance"]["sources"]}
    assert envelope["component"]["source_refs"]
    assert set(envelope["component"]["source_refs"]) <= provenance_ids


# -- Refusal wire shapes ----------------------------------------------------


def test_a_create_without_a_session_header_succeeds(client: TestClient):
    resp = client.post("/api/presentation/components", json={"intent": _LINE_CHART_INTENT})
    assert resp.status_code == 201


@pytest.mark.parametrize(
    "headers",
    [
        [(_LEGACY_SESSION_HEADER, "session-a")],
        [(_LEGACY_SESSION_HEADER, "")],
        [(_LEGACY_SESSION_HEADER, "session-a"), (_LEGACY_SESSION_HEADER, "session-b")],
    ],
    ids=["single", "empty", "repeated"],
)
def test_a_session_header_is_ignored_on_create_revise_and_read(
    client: TestClient, headers: list[tuple[str, str]]
):
    created = client.post(
        "/api/presentation/components", json={"intent": _LINE_CHART_INTENT}, headers=headers
    )
    assert created.status_code == 201
    component_id = created.json()["component_id"]

    revised = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "A new title"}},
        headers=[*headers, ("x-dfo-write-token", created.json()["write_token"])],
    )
    assert revised.status_code == 200
    read = client.get(f"/api/presentation/components/{component_id}", headers=headers)
    assert read.status_code == 200
    assert read.json()["revision"] == 2


def test_a_body_with_an_unknown_top_level_field_is_request_invalid(client: TestClient):
    resp = client.post(
        "/api/presentation/components",
        json={"intent": _LINE_CHART_INTENT, "extra_field": "nope"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False


def test_a_body_carrying_session_is_request_invalid(client: TestClient):
    resp = client.post(
        "/api/presentation/components",
        json={"intent": _LINE_CHART_INTENT, "session": "sneaky"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False


def test_a_body_missing_intent_is_request_invalid(client: TestClient):
    resp = client.post(
        "/api/presentation/components",
        json={},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False


def test_a_body_carrying_schema_version_is_unsupported_schema_version(client: TestClient):
    """An extra `schema_version` field maps to non-retryable 422 `unsupported_schema_version`."""
    resp = client.post(
        "/api/presentation/components",
        json={"intent": _LINE_CHART_INTENT, "schema_version": 3},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "unsupported_schema_version"
    assert body["retryable"] is False


def test_a_different_session_header_reads_and_revises_the_component(client: TestClient):
    """The legacy session header plays no part: a caller naming another session value reads by
    id, and revises with the write token."""
    created = client.post(
        "/api/presentation/components",
        json={"intent": _LINE_CHART_INTENT},
        headers={_LEGACY_SESSION_HEADER: "session-a"},
    )
    component_id = created.json()["component_id"]

    revised = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "Revised by another caller"}},
        headers={
            _LEGACY_SESSION_HEADER: "session-b",
            "x-dfo-write-token": created.json()["write_token"],
        },
    )
    assert revised.status_code == 200
    assert revised.json()["revision"] == 2

    read = client.get(
        f"/api/presentation/components/{component_id}",
        params={"revision": 1},
        headers={_LEGACY_SESSION_HEADER: "session-b"},
    )
    assert read.status_code == 200
    assert read.json()["stored_intent"]["title"] == _LINE_CHART_INTENT["title"]


def test_a_revise_of_an_unknown_component_id_is_not_found(client: TestClient):
    resp = client.patch(
        "/api/presentation/components/cmp_does_not_exist",
        json={"intent": {"title": "x"}},
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert resp.json()["retryable"] is False


def test_a_revision_scoped_get_of_an_unknown_component_id_is_not_found(client: TestClient):
    resp = client.get(
        "/api/presentation/components/cmp_does_not_exist",
        params={"revision": 1},
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert resp.json()["retryable"] is False


def test_a_revise_carrying_composition_id_is_binding_invalid(client: TestClient):
    created = _create(client).json()
    component_id = created["component_id"]

    resp = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "x"}, "composition_id": "cps_something"},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "binding_invalid"
    assert body["type"] == "/codes/binding_invalid"


def test_a_missing_type_on_create_is_schema_invalid_and_names_the_legal_types(client: TestClient):
    resp = client.post(
        "/api/presentation/components",
        json={"intent": {"select": []}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "schema_invalid"
    # The missing type returns the legal union members as `minimal_intent`.
    assert body["retryable"] is True
    assert "minimal legal intent" in body["detail"]


def test_a_create_missing_select_is_schema_invalid_with_the_minimal_intent(client: TestClient):
    resp = client.post(
        "/api/presentation/components",
        json={"intent": {"type": "line_chart", "frame_ref": _LINE_CHART_INTENT["frame_ref"]}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "schema_invalid"
    assert body["retryable"] is True
    assert "minimal legal intent" in body["detail"]


def test_a_constraint_violation_carries_constraint_id_and_a_codes_uri(release: CatalogueRelease):
    """Two incompatible units trigger the write-reachable `comparable_units_required` constraint."""
    from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
    from assembler.ports import FrameRef
    from assembler.testing import build_resolved_frame
    from dfo_platform.app import create_app
    from presentation_api.app import presentation_surface

    ref = FrameRef(spec_id="spec_" + "e" * 32, content_version="cv_" + "f" * 32)
    resolved = build_resolved_frame(
        frame_ref=ref,
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": "unit/USD",
                "measurementMethod": None,
                "provenance": "src_a",
            },
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 2.0,
                "unit": "unit/EUR",
                "measurementMethod": None,
                "provenance": "src_a",
            },
        ],
    )
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=resolved)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=FakeReleaseRegistry(release=release),
    )
    intent = {
        "type": "line_chart",
        "frame_ref": {"spec_id": ref.spec_id, "content_version": ref.content_version},
        "select": {"stat_var": "dcid:X", "unit": ["unit/USD", "unit/EUR"]},
    }
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        resp = _create(client, intent=intent)
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "constraint_violation"
    assert body["constraint_id"] == "comparable_units_required"
    assert body["type"] == "/codes/constraint_violation"
    assert body["pointer"] == "/select"


def test_an_unbound_facet_splitting_a_series_carries_the_structural_view_as_issues(
    release: CatalogueRelease,
):
    from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
    from dfo_platform.app import create_app
    from presentation_api.app import presentation_surface
    from tests.presentation_api.conftest import (
        build_multi_provenance_resolved_frame,
        multi_provenance_frame_ref,
    )

    resolved = build_multi_provenance_resolved_frame()
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=resolved)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=FakeReleaseRegistry(release=release),
    )
    ref = multi_provenance_frame_ref()
    intent = {
        "type": "line_chart",
        "frame_ref": {"spec_id": ref.spec_id, "content_version": ref.content_version},
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
        },
    }
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/presentation/components",
            json={"intent": intent},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "binding_invalid"
    assert body["issues"], "the facet split refusal must carry the structural view as issues[]"
    provenance_issue = next(issue for issue in body["issues"] if issue["axis"] == "provenance")
    assert {a["ref"] for a in provenance_issue["alternatives"]} == {"src_oecd_dac", "src_other"}
    # The frame has no labels for these provenance values.
    assert {a["label"] for a in provenance_issue["alternatives"]} == {"src_oecd_dac", "src_other"}
    # The frame resolves the recipient label to Ethiopia.
    recipient_issue = next(issue for issue in body["issues"] if issue["axis"] == "recipient")
    assert {"ref": "country/ETH", "label": "Ethiopia"} in recipient_issue["alternatives"]


def test_a_series_under_a_set_and_a_null_unit_refuses_at_the_unit_pointer(
    release: CatalogueRelease,
):
    """One StatVar and place under `USDollar` and under a null unit is one series split in
    two. The refusal points at `unit` and offers the one bindable value."""
    ref = FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "d" * 32)
    resolved = build_resolved_frame(
        frame_ref=ref,
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:GDP",
                "recipient": "country/GTM",
                "time_period": "2020",
                "value": value,
                "unit": unit,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for unit, value in (("USDollar", 77.6), (None, 16.9))
        ],
    )
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=resolved)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=FakeReleaseRegistry(release=release),
    )
    intent = {
        "type": "table",
        "frame_ref": {"spec_id": ref.spec_id, "content_version": ref.content_version},
        "select": {"stat_var": "dcid:GDP", "recipient": "country/GTM"},
    }
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        resp = _create(client, intent=intent)
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "binding_invalid"
    assert body["pointer"] == "/select/unit"
    assert body["alternatives"] == [{"ref": "USDollar", "label": "USDollar"}]


def test_an_unknown_value_refusal_carries_the_frames_resolved_label(release: CatalogueRelease):
    from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
    from dfo_platform.app import create_app
    from presentation_api.app import presentation_surface
    from tests.presentation_api.conftest import (
        build_multi_provenance_resolved_frame,
        multi_provenance_frame_ref,
    )

    resolved = build_multi_provenance_resolved_frame()
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=resolved)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=FakeReleaseRegistry(release=release),
    )
    ref = multi_provenance_frame_ref()
    intent = {
        "type": "line_chart",
        "frame_ref": {"spec_id": ref.spec_id, "content_version": ref.content_version},
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/GHA",
        },
    }
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/presentation/components",
            json={"intent": intent},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "binding_invalid"
    assert body["alternatives"] == [{"ref": "country/ETH", "label": "Ethiopia"}]


# -- component_id path param format validation ------------------------------


def test_a_malformed_component_id_on_revise_is_request_invalid(client: TestClient):
    resp = client.patch(
        "/api/presentation/components/cmp_bad!chars",
        json={"intent": {"title": "x"}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False


def test_a_malformed_component_id_on_read_is_request_invalid(client: TestClient):
    resp = client.get("/api/presentation/components/not-even-prefixed")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False


def test_a_well_formed_but_unknown_component_id_still_reaches_not_found(client: TestClient):
    # The ID passes format validation; the store returns not found.
    resp = client.get("/api/presentation/components/cmp_does_not_exist")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_an_unpublished_ops_frame_ref_is_frame_unavailable_and_retryable(
    revision_store: FakeRevisionStore, release: CatalogueRelease
):
    """An unpublished `ops_`/`ocv_` frame returns retryable `frame_unavailable`."""

    class _UnpublishedOpsFrameStore:
        def load(self, *, frame_ref: FrameRef):
            raise FrameUnavailable(
                reason="ops_unsupported",
                detail=f"{frame_ref.spec_id!r} was not published to the platform",
            )

        def pin(self, *, frame_ref: FrameRef, component_id: str):
            raise AssertionError("pin should not run after an ops_unsupported refusal")

    intent = dict(_LINE_CHART_INTENT)
    intent["frame_ref"] = {
        "spec_id": "ops_00000000000000000000000000000000",
        "content_version": "ocv_00000000000000000000000000000000",
    }
    app = create_app(
        [
            presentation_surface(
                frame_store=_UnpublishedOpsFrameStore(),
                revision_store=revision_store,
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    with TestClient(app, raise_server_exceptions=False) as stub_client:
        resp = _create(stub_client, intent=intent)
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "frame_unavailable"
    assert body["retryable"] is True


@pytest.mark.parametrize(
    ("fault", "status", "code", "retryable"),
    [
        ("unavailable", 503, "service_unavailable", True),
        ("store-error", 503, "service_unavailable", True),
        ("quarantined", 404, "frame_unavailable", False),
    ],
)
def test_a_create_whose_frame_registry_read_fails_answers_a_declared_refusal(
    fault: str,
    status: int,
    code: str,
    retryable: bool,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
):
    """A real frame store reads the registry to scope a frame ref's revocation check when it
    loads. Each fault that read can raise answers a declared refusal through the create route,
    never 500, and writes no component. A create pins nothing, so the load is the only read."""
    spec_reader = _FakeSpecReader()
    if fault == "quarantined":
        spec_reader.quarantined = frozenset({malaria_ethiopia_frame_ref().spec_id})
    else:
        spec_reader.definition_error = (
            SpecRegistryUnavailable("spec registry pool borrow timed out")
            if fault == "unavailable"
            else SpecRegistryStoreError("spec registry read failed")
        )
    live_store = _store(spec_reader=spec_reader)
    surface = presentation_surface(
        frame_store=live_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as stub_client:
        resp = _create(stub_client)

    assert (resp.status_code, resp.json()["code"], resp.json()["retryable"]) == (
        status,
        code,
        retryable,
    ), resp.text
    assert revision_store._revisions == {}


def test_a_published_ops_frame_ref_reaches_component_created(
    revision_store: FakeRevisionStore, release: CatalogueRelease
):
    """A published `ops_` frame creates a component.

    The content version uses `cv_` because the envelope schema pattern for `content_version` has no
    `ocv_` alternative yet, unlike `spec_id`.
    """
    frame_ref = FrameRef(spec_id="ops_" + "3" * 32, content_version="cv_" + "4" * 32)
    frame_store = FakeFrameStore()
    frame_store.put(
        resolved_frame=build_resolved_frame(
            frame_ref=frame_ref,
            role_keys=("recipient",),
            rows=malaria_ethiopia_rows(),
            labels={"country/ETH": "Ethiopia"},
        )
    )
    intent = dict(_LINE_CHART_INTENT)
    intent["frame_ref"] = {
        "spec_id": frame_ref.spec_id,
        "content_version": frame_ref.content_version,
    }
    app = create_app(
        [
            presentation_surface(
                frame_store=frame_store,
                revision_store=revision_store,
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    with TestClient(app, raise_server_exceptions=False) as published_client:
        resp = _create(published_client, intent=intent)
    assert resp.status_code == 201
    assert resp.json()["component_id"].startswith("cmp_")


# -- door coverage for the three codes test_codes.py only drove synthetically


def test_a_release_mismatch_is_retryable_after_a_delay(
    frame_store: FakeFrameStore, revision_store: FakeRevisionStore, release: CatalogueRelease
):
    """A release missing `source_card` maps the valid intent to `release_mismatch`."""
    reduced = _release_excluding(release, "source_card")
    release_registry = FakeReleaseRegistry(release=reduced)
    surface = presentation_surface(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    app: FastAPI = create_app([surface])
    with TestClient(app, raise_server_exceptions=False) as reduced_client:
        intent = {**_LINE_CHART_INTENT, "type": "source_card"}
        resp = _create(reduced_client, intent=intent)
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "release_mismatch"
    assert body["type"] == "/codes/release_mismatch"
    assert body["retryable"] is True


def test_a_stored_bar_chart_payload_missing_the_renamed_axis_keys_is_release_mismatch(
    client: TestClient, revision_store: FakeRevisionStore
):
    """Legacy bar-chart axis names fail the current release schema."""
    intent = {**_LINE_CHART_INTENT, "type": "bar_chart"}
    created = _create(client, intent=intent).json()
    component_id = created["component_id"]

    revisions = revision_store._revisions[component_id]
    current = revisions[-1]
    component = dict(current.resolved_payload["component"])
    component["x_axis"] = component.pop("category_axis")
    component["y_axis"] = component.pop("value_axis")
    payload = dict(current.resolved_payload)
    payload["component"] = component
    revisions[-1] = dataclasses.replace(current, resolved_payload=payload)

    resp = client.get(f"/api/presentation/components/{component_id}")
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "release_mismatch"
    assert body["retryable"] is True


def test_a_create_into_an_unknown_composition_id_is_not_found(client: TestClient):
    resp = client.post(
        "/api/presentation/components",
        json={"intent": _LINE_CHART_INTENT, "composition_id": "cps_does_not_exist"},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert body["type"] == "/codes/not_found"
    assert body["retryable"] is False


def test_a_window_exceeding_the_points_cap_is_resource_exhausted(release: CatalogueRelease):
    """A 500-point cap rejects a wide daily window before materialization."""
    from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
    from assembler.ports import FrameRef
    from assembler.testing import build_resolved_frame
    from dfo_platform.app import create_app
    from presentation_api.app import presentation_surface

    ref = FrameRef(spec_id="spec_" + "5" * 32, content_version="cv_" + "5" * 32)
    resolved = build_resolved_frame(
        frame_ref=ref,
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020-01-01",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
        ],
    )
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=resolved)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=FakeReleaseRegistry(release=release),
    )
    intent = {
        "type": "line_chart",
        "frame_ref": {"spec_id": ref.spec_id, "content_version": ref.content_version},
        "select": {"recipient": "country/ETH"},
        "window": {"start": "2020-01-01", "end": "2030-01-01"},
    }
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        resp = _create(client, intent=intent)
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "resource_exhausted"
    assert body["type"] == "/codes/resource_exhausted"
    assert body["retryable"] is False


class _BlockingFrameStore(FakeFrameStore):
    """Block `load()` on an event until the test releases the executor call."""

    def __init__(self, *, frames: Mapping[FrameRef, ResolvedFrame]) -> None:
        super().__init__(frames=frames)
        self.release = threading.Event()

    def load(self, *, frame_ref: FrameRef) -> ResolvedFrame:
        self.release.wait(timeout=5.0)
        return super().load(frame_ref=frame_ref)


def test_an_executor_timeout_renders_code_timeout_retryable_at_504(
    monkeypatch: pytest.MonkeyPatch,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
):
    """An executor timeout returns retryable 504 without sleeping through the real deadline."""
    monkeypatch.setattr(components_door, "PRESENTATION_COMPONENTS_TIMEOUT_SECONDS", 0.05)
    blocking_frame_store = _BlockingFrameStore(
        frames={malaria_ethiopia_frame_ref(): malaria_ethiopia_resolved_frame()}
    )
    surface = presentation_surface(
        frame_store=blocking_frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    app: FastAPI = create_app([surface])
    with TestClient(app, raise_server_exceptions=False) as client:
        try:
            resp = _create(client)
        finally:
            # Releases the executor worker parked in `_BlockingFrameStore
            # .load()` before this `with` block exits, so `TestClient
            # .__exit__`'s lifespan shutdown (`DoorExecutor.drain`) finds the
            # permit free.
            blocking_frame_store.release.set()
    assert resp.status_code == 504
    body = resp.json()
    assert body["code"] == "timeout"
    assert body["retryable"] is True


def test_the_backpressure_permit_covers_encoding_the_reply(
    monkeypatch: pytest.MonkeyPatch,
    frame_store: FakeFrameStore,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
):
    """Reply serialization holds the sole permit and runs in the executor."""
    monkeypatch.setattr(components_door, "PRESENTATION_COMPONENTS_MAX_CONCURRENCY", 1)
    rendering = threading.Event()
    finish_render = threading.Event()
    render_threads: list[str] = []
    serialize = components_door.serialize_create_reply

    def blocking_serialize(reply):
        render_threads.append(threading.current_thread().name)
        rendering.set()
        finish_render.wait(timeout=5.0)
        return serialize(reply)

    monkeypatch.setattr(components_door, "serialize_create_reply", blocking_serialize)
    surface = presentation_surface(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    with (
        TestClient(create_app([surface]), raise_server_exceptions=False) as client,
        ThreadPoolExecutor(max_workers=1) as caller,
    ):
        first = caller.submit(_create, client)
        try:
            assert rendering.wait(timeout=5.0)
            concurrent = _create(client)
        finally:
            finish_render.set()
        first_resp = first.result(timeout=5.0)

    assert concurrent.status_code == 503
    assert concurrent.json()["code"] == "service_unavailable"
    assert first_resp.status_code == 201
    assert len(render_threads) == 1
    assert render_threads[0].startswith("presentation-components")


def test_a_transient_datasets_store_failure_is_service_unavailable_and_retryable():
    """A `DatasetsStoreUnavailable` with no SQLSTATE (a bare driver/network
    failure) classifies transient.
    """
    spec = components_door.classify_components_exception(DatasetsStoreUnavailable("exemplar"))
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


def test_a_permanent_datasets_store_failure_is_internal_error_and_not_retryable():
    """A permanent SQLSTATE maps to non-retryable 500."""
    spec = components_door.classify_components_exception(
        DatasetsStoreUnavailable("exemplar", sqlstate="23503")
    )
    assert spec is not None
    assert spec.status == 500
    assert spec.code == "internal_error"
    assert spec.retryable is False


@pytest.mark.parametrize(
    "sqlstate",
    [
        pytest.param("42601", id="sqlstate-class-42"),
        pytest.param("23503", id="sqlstate-class-23"),
    ],
)
def test_permanent_presentation_store_failures_are_internal_errors_and_not_retryable(sqlstate):
    spec = components_door.classify_components_exception(
        PresentationStoreUnavailable("exemplar", sqlstate=sqlstate)
    )
    assert spec is not None
    assert spec.status == 500
    assert spec.code == "internal_error"
    assert spec.retryable is False


def test_a_transient_presentation_store_failure_with_a_sqlstate_is_service_unavailable():
    """A transient SQLSTATE maps to retryable 503."""
    spec = components_door.classify_components_exception(
        PresentationStoreUnavailable("exemplar", sqlstate="08006")
    )
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


@pytest.mark.parametrize("subject_kind", ("component", "composition"))
def test_a_fresh_generated_id_collision_is_service_unavailable_and_retryable(subject_kind: str):
    """A scoped fresh-id collision is retryable because the next create allocates a new ULID."""
    spec = components_door.classify_components_exception(
        GeneratedIdCollision(subject_kind=subject_kind)
    )
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


def test_a_presentation_store_failure_with_no_sqlstate_is_service_unavailable_and_retryable():
    """A failure without SQLSTATE maps to retryable 503."""
    spec = components_door.classify_components_exception(PresentationStoreUnavailable("exemplar"))
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


def test_the_components_door_refuses_to_start_without_the_public_site_url(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("DFO_PUBLIC_SITE_URL", raising=False)

    with pytest.raises(ValueError, match="DFO_PUBLIC_SITE_URL"), TestClient(app):
        pass
