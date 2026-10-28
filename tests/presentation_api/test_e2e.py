"""Offline end-to-end tests for the Presentation API, using fake stores and frames."""

from __future__ import annotations

import jsonschema
import pytest
from fastapi.testclient import TestClient

from assembler.catalogue import CatalogueRelease
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import FrameRef, FrameUnavailable
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_resolved_frame,
    malaria_ethiopia_rows,
)
from dfo_platform.app import create_app
from presentation_api.app import presentation_surface
from presentation_api.refusals import EXPIRED_FRAME_FIX


def _frame_ref_dict(ref: FrameRef) -> dict[str, str]:
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


_LINE_CHART_INTENT = {
    "type": "line_chart",
    "frame_ref": _frame_ref_dict(malaria_ethiopia_frame_ref()),
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
    "window": {"start": "2015", "end": "2024"},
    "title": "Malaria grant disbursements to Ethiopia",
}

_STAT_CARD_INTENT = {
    "type": "stat_card",
    "frame_ref": _frame_ref_dict(malaria_ethiopia_frame_ref()),
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
}


def _schema_registry(release: CatalogueRelease):
    """Build a registry that resolves envelope references by schema `$id`."""
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    resources = [
        Resource.from_contents(schema, default_specification=DRAFT202012)
        for schema in release.component_schemas.values()
    ]
    return Registry().with_resources(
        (resource.id(), resource) for resource in resources if resource.id() is not None
    )


def _create(client: TestClient, *, intent: dict, composition_id=None, write_token=None):
    body: dict[str, object] = {"intent": intent}
    if composition_id is not None:
        body["composition_id"] = composition_id
    headers = {"x-dfo-write-token": write_token} if write_token is not None else {}
    return client.post("/api/presentation/components", json=body, headers=headers)


# -- create, revise, no-op retry, illegal patch -----------------------------


def test_create_line_chart_from_a_worked_style_intent_stores_a_schema_valid_envelope(
    client: TestClient, revision_store: FakeRevisionStore, release_registry: FakeReleaseRegistry
):
    resp = _create(client, intent=_LINE_CHART_INTENT)
    assert resp.status_code == 201
    body = resp.json()
    assert body["component_id"].startswith("cmp_")
    assert body["revision"] == 1
    assert body["catalog_version"] == release_registry.pinned().catalog_version
    assert body["affordance"] is not None
    assert body["component_id"] in body["affordance"]

    record = revision_store.read_current(component_id=body["component_id"])
    assert record is not None
    assert record.revision == 1
    release = release_registry.pinned()
    registry = _schema_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )


def test_revise_window_advances_the_revision_and_updates_x_axis(
    client: TestClient, revision_store: FakeRevisionStore
):
    """Changing `window` updates `x_axis` without changing `y_axis`."""
    created = _create(client, intent=_LINE_CHART_INTENT).json()
    component_id = created["component_id"]

    resp = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"window": {"start": "2017", "end": "2020"}}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 2
    # An object-valued field's changed entry carries only `field` -- no
    # `prior`/`new` -- (`presentation_api.reply.serialize_write_reply`).
    assert body["changed"] == [{"field": "window"}]

    before = revision_store.read_revision(component_id=component_id, revision=1)
    after = revision_store.read_revision(component_id=component_id, revision=2)
    before_component = before.resolved_payload["component"]
    after_component = after.resolved_payload["component"]
    assert before_component["x_axis"]["window"] == {"start": "2015", "end": "2024"}
    assert after_component["x_axis"]["window"] == {"start": "2017", "end": "2020"}
    # `window` is the only changed field in the component subtree.
    assert before_component["title"] == after_component["title"]
    assert before_component["y_axis"] == after_component["y_axis"]

    current = client.get(f"/api/presentation/components/{component_id}").json()
    assert current["revision"] == 2
    revision_1 = client.get(
        f"/api/presentation/components/{component_id}",
        params={"revision": 1},
    ).json()
    assert revision_1["revision"] == 1
    assert revision_1["stored_intent"]["window"] == {"start": "2015", "end": "2024"}


def test_a_no_op_revise_leaves_the_revision_and_the_store_unchanged(
    client: TestClient, revision_store: FakeRevisionStore
):
    created = _create(client, intent=_LINE_CHART_INTENT).json()
    component_id = created["component_id"]

    resp = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"window": _LINE_CHART_INTENT["window"]}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 1
    assert body["changed"] == []
    # A re-sent patch resolving to the same payload mints nothing -- no
    # revision 2 exists to read back, the observable proof a transport
    # retry landed twice without compounding.
    assert revision_store.read_revision(component_id=component_id, revision=2) is None


def test_an_illegal_patch_field_among_legal_ones_refuses_and_applies_nothing(
    client: TestClient, revision_store: FakeRevisionStore
):
    created = _create(client, intent=_LINE_CHART_INTENT).json()
    component_id = created["component_id"]

    resp = client.patch(
        f"/api/presentation/components/{component_id}",
        json={
            "intent": {
                "title": "Renamed",
                "window": {"start": "2016", "end": "2020"},
                "bogus_field": "nope",
            }
        },
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "schema_invalid"
    assert body["pointer"] == "/bogus_field"
    assert body["retryable"] is False
    # An illegal field for the type carries its legal fields as
    # `alternatives`, not folded into `detail` prose.
    assert body["alternatives"]
    assert all({"ref", "label"} <= entry.keys() for entry in body["alternatives"])
    assert revision_store.read_revision(component_id=component_id, revision=2) is None


# -- scale and range on a line_chart intent ----------------------------------


def test_a_line_chart_intent_carrying_scale_mints_it_onto_the_y_axis(
    client: TestClient, revision_store: FakeRevisionStore
):
    """Positive fixture values let a log scale pass `log_scale_requires_positive_values`."""
    intent = {**_LINE_CHART_INTENT, "scale": "log"}
    resp = _create(client, intent=intent)
    assert resp.status_code == 201
    component_id = resp.json()["component_id"]

    record = revision_store.read_revision(component_id=component_id, revision=1)
    assert record.resolved_payload["component"]["y_axis"]["scale"] == "log"


def test_a_bar_chart_intent_carrying_log_scale_refuses_constraint_violation_at_scale(
    client: TestClient,
):
    """A log-scaled bar chart fails `bar_zero_in_domain` at `/scale` before storage."""
    intent = {**_LINE_CHART_INTENT, "type": "bar_chart", "scale": "log"}
    resp = _create(client, intent=intent)
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "constraint_violation"
    assert body["constraint_id"] == "bar_zero_in_domain"
    assert body["pointer"] == "/scale"
    assert body["retryable"] is False
    assert "component_id" not in body


# -- facet split: one-corrected-write recovery -------------------------------


def test_a_facet_split_refuses_then_a_single_corrected_write_succeeds(
    release: CatalogueRelease,
):
    from tests.presentation_api.conftest import (
        build_multi_provenance_resolved_frame,
        multi_provenance_frame_ref,
    )

    resolved = build_multi_provenance_resolved_frame()
    ref = multi_provenance_frame_ref()
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=resolved)
    app = create_app(
        [
            presentation_surface(
                frame_store=frame_store,
                revision_store=FakeRevisionStore(),
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    base_select = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    }
    unbound_intent = {
        "type": "line_chart",
        "frame_ref": _frame_ref_dict(ref),
        "select": base_select,
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        refused = _create(client, intent=unbound_intent)
        assert refused.status_code == 400
        refused_body = refused.json()
        assert refused_body["code"] == "binding_invalid"
        assert refused_body["issues"], "the facet split refusal must carry issues[]"
        provenance_issue = next(
            issue for issue in refused_body["issues"] if issue["axis"] == "provenance"
        )
        assert {a["ref"] for a in provenance_issue["alternatives"]} == {"src_oecd_dac", "src_other"}

        # One corrected write, no describe_frame round trip in between.
        corrected_intent = {
            **unbound_intent,
            "select": {**base_select, "provenance": "src_oecd_dac"},
        }
        corrected = _create(client, intent=corrected_intent)
    assert corrected.status_code == 201


# -- composition: create-in-composition and a second append -----------------


def test_a_second_create_into_the_same_composition_appends_and_advances_composition_revision(
    client: TestClient,
):
    first = _create(client, intent=_LINE_CHART_INTENT, composition_id="new").json()
    composition_id = first["composition_id"]
    assert composition_id.startswith("cps_")
    assert first["composition_revision"] == 1
    assert first["composition_write_token"].startswith("cpw_")

    second = _create(
        client,
        intent=_STAT_CARD_INTENT,
        composition_id=composition_id,
        write_token=first["composition_write_token"],
    )
    assert second.status_code == 201
    second_body = second.json()
    assert second_body["composition_id"] == composition_id
    assert second_body["composition_revision"] == 2


def _live_store_client(release: CatalogueRelease) -> TestClient:
    """A client over `LivePresentationStore` and the SQL-shaped fake database."""
    from presentation_store.store import LivePresentationStore
    from tests.presentation_store.conftest import FakePresentationStore

    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=malaria_ethiopia_resolved_frame())
    app = create_app(
        [
            presentation_surface(
                frame_store=frame_store,
                revision_store=LivePresentationStore(store=FakePresentationStore()),
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    return TestClient(app, raise_server_exceptions=False)


def test_an_append_into_an_existing_composition_needs_its_write_token_on_the_live_store(
    release: CatalogueRelease,
):
    """The `cps_` id alone does not grant an append. The composition's `cpw_` token does, and
    the component's `cmw_` token does not."""
    with _live_store_client(release) as client:
        first = _create(client, intent=_LINE_CHART_INTENT, composition_id="new").json()
        composition_id = first["composition_id"]

        without_token = _create(client, intent=_STAT_CARD_INTENT, composition_id=composition_id)
        with_component_token = _create(
            client,
            intent=_STAT_CARD_INTENT,
            composition_id=composition_id,
            write_token=first["write_token"],
        )
        second = _create(
            client,
            intent=_STAT_CARD_INTENT,
            composition_id=composition_id,
            write_token=first["composition_write_token"],
        )

    for refused in (without_token, with_component_token):
        assert refused.status_code == 403
        assert refused.json()["code"] == "write_refused"
    assert second.status_code == 201
    assert second.json()["composition_id"] == first["composition_id"]
    assert second.json()["composition_revision"] == 2


def test_a_create_into_an_unknown_composition_id_is_not_found_on_the_live_store(
    release: CatalogueRelease,
):
    with _live_store_client(release) as client:
        resp = _create(client, intent=_LINE_CHART_INTENT, composition_id="cps_never_minted")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_a_revise_carrying_composition_id_refuses_binding_invalid(client: TestClient):
    created = _create(client, intent=_LINE_CHART_INTENT).json()
    resp = client.patch(
        f"/api/presentation/components/{created['component_id']}",
        json={"intent": {"title": "x"}, "composition_id": "cps_something"},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "binding_invalid"


# -- the component id grants read, the write token grants revise -------------


def test_the_id_reads_and_the_write_token_revises_on_the_live_store(
    release: CatalogueRelease,
):
    """Create under one session header, revise under another with the write token, read with
    no header at all."""
    with _live_store_client(release) as client:
        created = client.post(
            "/api/presentation/components",
            json={"intent": _LINE_CHART_INTENT},
            headers={"x-dfo-session": "session-a"},
        ).json()
        component_id = created["component_id"]

        refused = client.patch(
            f"/api/presentation/components/{component_id}",
            json={"intent": {"title": "Revised without the token"}},
            headers={"x-dfo-session": "session-a"},
        )
        revised = client.patch(
            f"/api/presentation/components/{component_id}",
            json={"intent": {"title": "Revised by another caller"}},
            headers={"x-dfo-session": "session-b", "x-dfo-write-token": created["write_token"]},
        )
        current = client.get(f"/api/presentation/components/{component_id}")
        unknown = client.get("/api/presentation/components/cmp_never_created")

    assert refused.status_code == 403
    assert refused.json()["code"] == "write_refused"
    assert revised.status_code == 200
    assert revised.json()["revision"] == 2
    assert current.status_code == 200
    assert current.json()["revision"] == 2
    assert current.json()["stored_intent"]["title"] == "Revised by another caller"
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "not_found"


# -- reads: get_component revisions, get_frame_view's three modes -----------


def test_get_frame_view_three_modes_agree_with_the_frame_fixture(
    client: TestClient, frame_ref: FrameRef
):
    url = f"/api/presentation/frames/{frame_ref.spec_id}/{frame_ref.content_version}"
    structural = client.get(url).json()
    recipient_dim = next(d for d in structural["dimensions"] if d["name"] == "recipient")

    dimension_mode = client.get(url, params={"dimension": "recipient"}).json()
    assert dimension_mode["total_count"] == recipient_dim["total_count"]
    assert set(dimension_mode["values"]) == set(recipient_dim["values"])

    query_mode = client.get(url, params={"query": "eth"}).json()
    query_recipient = next(d for d in query_mode["dimensions"] if d["name"] == "recipient")
    assert query_recipient["values"] == dimension_mode["values"]


# -- frame refusals: ops_ spec, revoked, expired -----------------------------


def _create_with_store(frame_store, *, intent: dict, release: CatalogueRelease) -> TestClient:
    app = create_app(
        [
            presentation_surface(
                frame_store=frame_store,
                revision_store=FakeRevisionStore(),
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    client = TestClient(app, raise_server_exceptions=False)
    return _create(client, intent=intent)


def test_an_unpublished_ops_prefixed_frame_ref_refuses_frame_unavailable_and_retryable(
    release: CatalogueRelease,
):
    """An unpublished `ops_`/`ocv_` frame returns retryable `frame_unavailable`."""
    intent = {
        **_LINE_CHART_INTENT,
        "frame_ref": {"spec_id": "ops_" + "0" * 32, "content_version": "ocv_" + "0" * 32},
    }
    resp = _create_with_store(
        _TypedRefusalFrameStore(reason="ops_unsupported"), intent=intent, release=release
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "frame_unavailable"
    assert body["retryable"] is True


def test_a_published_ops_prefixed_frame_ref_reaches_component_created(release: CatalogueRelease):
    """A published `ops_` frame creates a component.

    The content version uses `cv_` because the envelope schema pattern for `content_version` has no
    `ocv_` alternative yet, unlike `spec_id`.
    """
    frame_ref = FrameRef(spec_id="ops_" + "1" * 32, content_version="cv_" + "2" * 32)
    frame_store = FakeFrameStore()
    frame_store.put(
        resolved_frame=build_resolved_frame(
            frame_ref=frame_ref,
            role_keys=("recipient",),
            rows=malaria_ethiopia_rows(),
            labels={"country/ETH": "Ethiopia"},
        )
    )
    intent = {
        **_LINE_CHART_INTENT,
        "frame_ref": {"spec_id": frame_ref.spec_id, "content_version": frame_ref.content_version},
    }
    resp = _create_with_store(frame_store, intent=intent, release=release)
    assert resp.status_code == 201
    assert resp.json()["component_id"].startswith("cmp_")


class _TypedRefusalFrameStore:
    """A minimal `FrameStore`-shaped stub (structural, like every fake in
    `assembler.fakes`) whose `load` always refuses with one fixed reason, for
    exercising a `FrameUnavailable` reason beyond `"io"`, the one
    `assembler.fakes.FakeFrameStore` raises for an unregistered frame.
    """

    def __init__(self, *, reason: str) -> None:
        self._reason = reason

    def load(self, *, frame_ref: FrameRef):
        raise FrameUnavailable(reason=self._reason, detail=f"frame is {self._reason}")

    def pin(self, *, frame_ref: FrameRef, component_id: str) -> bool:
        raise AssertionError("pin should never be reached once load refuses")


def _frames_get(frame_store, *, ref: FrameRef, release: CatalogueRelease) -> TestClient:
    app = create_app(
        [
            presentation_surface(
                frame_store=frame_store,
                revision_store=FakeRevisionStore(),
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        return client.get(
            f"/api/presentation/frames/{ref.spec_id}/{ref.content_version}",
        )


def test_a_revoked_frame_is_frame_unavailable_and_not_retryable(release: CatalogueRelease):
    resp = _frames_get(
        _TypedRefusalFrameStore(reason="revoked"), ref=malaria_ethiopia_frame_ref(), release=release
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "frame_unavailable"
    assert body["retryable"] is False


def test_an_expired_frame_is_frame_unavailable_and_retryable(release: CatalogueRelease):
    resp = _frames_get(
        _TypedRefusalFrameStore(reason="expired"), ref=malaria_ethiopia_frame_ref(), release=release
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "frame_unavailable"
    assert body["retryable"] is True
    assert body["detail"].endswith(EXPIRED_FRAME_FIX)


def test_a_never_stored_frame_is_frame_unavailable_and_not_retryable(release: CatalogueRelease):
    resp = _frames_get(
        _TypedRefusalFrameStore(reason="unknown"), ref=malaria_ethiopia_frame_ref(), release=release
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "frame_unavailable"
    assert body["retryable"] is False


# -- stat_card change ---------------------------------------------------------


def test_stat_card_create_mints_without_a_change_field(
    client: TestClient, revision_store: FakeRevisionStore
):
    """A stat card without `change` omits the payload key and reports `None`."""
    resp = _create(client, intent=_STAT_CARD_INTENT)
    assert resp.status_code == 201
    body = resp.json()
    assert body["fact_summary"]["change"] is None

    record = revision_store.read_current(component_id=body["component_id"])
    assert "change" not in record.resolved_payload["component"]


def test_a_stat_card_intent_carrying_change_mints_and_reads_back_the_change(
    client: TestClient, revision_store: FakeRevisionStore
):
    """The change summary and stored payload include the 2023-to-2024 derivation."""
    intent = {
        **_STAT_CARD_INTENT,
        "change": {"derivation": "difference", "basis": ["2023", "2024"]},
    }
    resp = _create(client, intent=intent)
    assert resp.status_code == 201
    body = resp.json()
    expected_change = {
        "derivation": "difference",
        "basis": ["2023", "2024"],
        "value": pytest.approx(8.2),
    }
    assert body["fact_summary"]["change"] == expected_change

    record = revision_store.read_current(component_id=body["component_id"])
    assert record.resolved_payload["component"]["change"] == expected_change


# -- release stamp consistency -----------------------------------------------


def test_release_catalog_version_matches_every_stored_envelopes_stamp(
    client: TestClient, revision_store: FakeRevisionStore
):
    line = _create(client, intent=_LINE_CHART_INTENT).json()
    stat = _create(client, intent=_STAT_CARD_INTENT).json()

    release = client.get("/api/presentation/release").json()

    for component_id in (line["component_id"], stat["component_id"]):
        record = revision_store.read_current(component_id=component_id)
        assert record.catalog_version == release["catalog_version"]
