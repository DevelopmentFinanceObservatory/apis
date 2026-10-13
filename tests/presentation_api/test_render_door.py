"""Checks the two render routes: who may call them, which target frames they accept, what they
answer, and the status each refusal carries."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import zipfile
from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import presentation_api.doors.render as render_door
from assembler.catalogue import load_release
from assembler.errors import (
    BindingInvalid,
    ConstraintViolation,
    ReleaseMismatch,
    ResourceExhausted,
    SchemaInvalid,
)
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import EditConflict, FrameRef, FrameUnavailable, ResolvedFrame
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_rows,
)
from datasets_api.live_refresh import LiveRefreshPrincipal
from datasets_api.recipe_store import RecipeRecord
from dfo_platform.app import create_app
from presentation_api.app import presentation_surface
from tests.assembler.test_accept_card_descriptions import _record as _description_record
from tests.datasets_api._snapshot_doubles import FakeSnapshotReader, _location, _version

_PRINCIPAL = "live-refresh-principal"
_AUTH = {"x-dfo-principal": _PRINCIPAL}
_MOVED_BY = 1000.0

_STORED = malaria_ethiopia_frame_ref()
_TARGET = FrameRef(spec_id=_STORED.spec_id, content_version="cv_" + "b" * 32)
_OTHER_DATASET = FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "b" * 32)
_UNRECORDED = FrameRef(spec_id=_STORED.spec_id, content_version="cv_" + "e" * 32)
_OPS_STORED = FrameRef(spec_id="ops_" + "1" * 32, content_version="ocv_" + "1" * 32)
_OPS_TARGET = FrameRef(spec_id="ops_" + "2" * 32, content_version="ocv_" + "2" * 32)

_BY_RECIPIENT = {"by": ["recipient"], "op": "sum"}
_BY_STAT_VAR = {"by": ["stat_var"], "op": "sum"}


def _body(frame: FrameRef) -> dict[str, str]:
    return {"spec_id": frame.spec_id, "content_version": frame.content_version}


def _frame(
    frame_ref: FrameRef,
    *,
    moved_by: float = 0.0,
    recipients: tuple[str, ...] = ("country/ETH",),
) -> ResolvedFrame:
    """The malaria frame under `frame_ref` for each of `recipients`, every value `moved_by`
    higher."""
    return build_resolved_frame(
        frame_ref=frame_ref,
        role_keys=("recipient",),
        rows=[
            {**row, "recipient": recipient, "value": row["value"] + moved_by}
            for recipient in recipients
            for row in malaria_ethiopia_rows()
        ],
        labels={"country/ETH": "Ethiopia"},
        retrieved_on=date(2026, 10, 6),
    )


def _intent(member: str, frame: FrameRef = _STORED) -> dict[str, object]:
    if member == "note_narration":
        return {"type": member, "text": "Disbursements rose."}
    intent: dict[str, object] = {"type": member, "frame_ref": _body(frame)}
    if member != "source_card":
        intent["select"] = {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
        }
    return intent


class _Recipes:
    """Recipes by frame pair. Only `normal_form_hash` matters to the door."""

    def __init__(self) -> None:
        self.hashes: dict[FrameRef, str] = {}

    def read_recipe(self, *, spec_id: str, content_version: str) -> RecipeRecord | None:
        normal_form_hash = self.hashes.get(
            FrameRef(spec_id=spec_id, content_version=content_version)
        )
        if normal_form_hash is None:
            return None
        return RecipeRecord(
            spec_id=spec_id,
            content_version=content_version,
            normal_form_hash=normal_form_hash,
            replay_key="rpk_" + "0" * 32,
            body={"anchor": None},
        )


def _record(
    snapshots: FakeSnapshotReader, frame: FrameRef, *, groupby: dict[str, object] | None = None
) -> None:
    """Record `frame` as a plain version, or as a collapse with `groupby`."""
    key = (frame.spec_id, frame.content_version)
    snapshots.versions[key] = dataclasses.replace(
        _version(content_version=frame.content_version),
        subject_id=frame.spec_id,
        base_content_version=None if groupby is None else "cv_" + "9" * 32,
    )
    rendering: dict[str, object] = {"format": "parquet", "bundle": False}
    if groupby is not None:
        rendering["groupby"] = groupby
    snapshots.locations[key] = [_location(rendering=rendering)]


@pytest.fixture
def frame_store() -> FakeFrameStore:
    store = FakeFrameStore()
    for frame in (_STORED, _OPS_STORED):
        store.put(resolved_frame=_frame(frame))
    for frame in (_TARGET, _OTHER_DATASET, _UNRECORDED, _OPS_TARGET):
        store.put(resolved_frame=_frame(frame, moved_by=_MOVED_BY))
    return store


@pytest.fixture
def revision_store() -> FakeRevisionStore:
    return FakeRevisionStore(clock=lambda: datetime(2026, 10, 4, 9, 0, tzinfo=UTC))


@pytest.fixture
def snapshots() -> FakeSnapshotReader:
    reader = FakeSnapshotReader()
    for frame in (_STORED, _TARGET, _OTHER_DATASET):
        _record(reader, frame)
    return reader


@pytest.fixture
def recipes() -> _Recipes:
    return _Recipes()


def _app(
    *,
    frame_store: FakeFrameStore,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
    **kwargs: object,
) -> FastAPI:
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        **kwargs,
    )
    return create_app([surface])


@pytest.fixture
def app(frame_store, revision_store, release_registry, snapshots, recipes) -> FastAPI:
    return _app(
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        snapshot_reader=snapshots,
        recipe_reader=recipes,
        live_refresh=LiveRefreshPrincipal(header="x-dfo-principal", principal=_PRINCIPAL),
    )


def _create(client: TestClient, member: str = "line_chart", frame: FrameRef = _STORED) -> str:
    response = client.post("/api/presentation/components", json={"intent": _intent(member, frame)})
    assert response.status_code == 201, response.text
    return response.json()["component_id"]


def _render_url(component_id: str, revision: int = 1) -> str:
    return f"/api/presentation/components/{component_id}/revisions/{revision}/render"


def _render(client: TestClient, component_id: str, frame: FrameRef, *, revision: int = 1):
    return client.post(
        _render_url(component_id, revision), json={"frame": _body(frame)}, headers=_AUTH
    )


def _render_data(client: TestClient, component_id: str, frame: FrameRef, **params: str):
    return client.get(
        f"{_render_url(component_id)}/data", params={**_body(frame), **params}, headers=_AUTH
    )


def _values(payload: dict) -> list[float]:
    (series,) = payload["component"]["series"]
    return [point["value"] for point in series["points"] if "value" in point]


# -- admission --------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param([], id="no-header"),
        pytest.param([("x-dfo-principal", "someone-else")], id="another-principal"),
        pytest.param(
            [("x-dfo-principal", _PRINCIPAL), ("x-dfo-principal", _PRINCIPAL)], id="repeated"
        ),
    ],
)
@pytest.mark.parametrize("route", ["render", "render/data"])
def test_a_caller_other_than_the_live_refresh_principal_is_refused_before_the_request_is_read(
    client: TestClient, headers, route
):
    """The path, query, and body are all invalid, and the caller is still the refusal."""
    url = f"/api/presentation/components/not-a-component/revisions/0/{route}"

    if route == "render":
        response = client.post(url, content=b"{not json", headers=headers)
    else:
        response = client.get(url, params={"format": "nope"}, headers=headers)

    assert response.status_code == 403
    assert response.json()["code"] == "refresh_not_permitted"
    assert response.json()["retryable"] is False


@pytest.mark.parametrize("route", ["render", "render/data"])
def test_a_surface_given_no_principal_refuses_every_caller(
    frame_store, revision_store, release_registry, snapshots, recipes, route
):
    app = _app(
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        snapshot_reader=snapshots,
        recipe_reader=recipes,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        component_id = _create(client)
        url = f"{_render_url(component_id)}"
        answers = [
            client.post(url, json={"frame": _body(_STORED)}, headers=headers)
            if route == "render"
            else client.get(f"{url}/data", params=_body(_STORED), headers=headers)
            for headers in ({}, _AUTH, {"x-dfo-principal": ""})
        ]

    assert [(answer.status_code, answer.json()["code"]) for answer in answers] == [
        (403, "refresh_not_permitted")
    ] * 3


def test_the_other_component_routes_still_take_no_caller(client: TestClient):
    component_id = _create(client)

    read = client.get(f"/api/presentation/components/{component_id}")
    data = client.get(f"/api/presentation/components/{component_id}/revisions/1/data")

    assert (read.status_code, data.status_code) == (200, 200)


def test_the_composed_app_passes_the_configured_principal_to_the_render_routes(
    monkeypatch: pytest.MonkeyPatch,
):
    """A malformed component id is `request_invalid` only once the caller is admitted."""
    import dfo_apis.app as dfo_apis_app
    from materialise.rate_tables import RateTableCache

    url = "/api/presentation/components/not-a-component/revisions/1/render"
    answers = {}
    for configured in (_PRINCIPAL, ""):
        monkeypatch.setenv("LIVE_REFRESH_PRINCIPAL", configured)
        surface = dfo_apis_app._presentation_factory(RateTableCache())
        # No lifespan runs, so no live store is built. Both answers come before any store read.
        client = TestClient(create_app([surface]), raise_server_exceptions=False)
        answers[configured] = client.post(url, json={"frame": _body(_STORED)}, headers=_AUTH)

    assert (answers[_PRINCIPAL].status_code, answers[_PRINCIPAL].json()["code"]) == (
        400,
        "request_invalid",
    )
    assert (answers[""].status_code, answers[""].json()["code"]) == (403, "refresh_not_permitted")


# -- the answer -------------------------------------------------------------


def test_render_answers_a_component_read_drawn_on_the_target_and_appends_nothing(
    client: TestClient,
    frame_store: FakeFrameStore,
    revision_store: FakeRevisionStore,
    monkeypatch: pytest.MonkeyPatch,
):
    component_id = _create(client)
    read = client.get(f"/api/presentation/components/{component_id}").json()

    def no_write(**_kwargs: object):
        raise AssertionError("a render opened a write")

    monkeypatch.setattr(revision_store, "begin_write", no_write)
    monkeypatch.setattr(revision_store, "allocate_component_id", no_write)

    first = _render(client, component_id, _TARGET)
    second = _render(client, component_id, _TARGET)

    assert (first.status_code, second.status_code) == (200, 200)
    assert first.headers["cache-control"] == "no-store"
    rendered = first.json()
    assert set(rendered) == set(read)
    assert (rendered["component_id"], rendered["revision"]) == (component_id, 1)
    assert rendered["stored_intent"] == read["stored_intent"]
    provenance = rendered["resolved_payload"]["provenance"]
    assert provenance["inputs"] == [_body(_TARGET)]
    assert provenance["refreshed_from"] == 1
    assert _values(rendered["resolved_payload"]) == [
        value + _MOVED_BY for value in _values(read["resolved_payload"])
    ]
    assert client.get(f"/api/presentation/components/{component_id}").json() == read
    assert len(revision_store._revisions[component_id]) == 1
    assert frame_store.pins.count((_TARGET, component_id)) == 1


def test_create_stores_a_description_and_render_composes_one_live(
    frame_store: FakeFrameStore,
    revision_store: FakeRevisionStore,
    snapshots: FakeSnapshotReader,
    recipes: _Recipes,
):
    """The stored and target frames' versions hold description records. Under the installed
    catalogue, as a deployment runs, the create route stores a description worded from the
    surface's own vocabulary source, and the render route answers one marked live, on the
    target frame, under the same vocabulary revision."""
    for ref in (_STORED, _TARGET):
        frame = frame_store.load(frame_ref=ref)
        frame_store.put(
            resolved_frame=dataclasses.replace(frame, description_record=_description_record(frame))
        )
    app = _app(
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=FakeReleaseRegistry(release=load_release()),
        snapshot_reader=snapshots,
        recipe_reader=recipes,
        live_refresh=LiveRefreshPrincipal(header="x-dfo-principal", principal=_PRINCIPAL),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        component_id = _create(client)
        rendered = _render(client, component_id, _TARGET).json()["resolved_payload"]

    stored = revision_store.read_revision(component_id=component_id, revision=1)
    assert stored is not None
    issued = stored.resolved_payload["description"]
    live = rendered["description"]
    assert issued["describes"] == {
        "kind": "card_revision",
        "component_id": component_id,
        "revision": 1,
    }
    assert "rendering" not in issued
    assert (live["rendering"], live["frame"]) == (
        "live",
        {"id": _TARGET.spec_id, "content_version": _TARGET.content_version},
    )
    assert issued["built_from"]["vocabulary"].startswith("voc_")
    assert live["built_from"]["vocabulary"] == issued["built_from"]["vocabulary"]


def test_a_source_card_renders_with_the_target_frame_s_sources(
    client: TestClient, frame_store: FakeFrameStore
):
    """The stored frame's rows cite `src_oecd_dac` and the target's `src_refreshed`."""
    frame_store.put(
        resolved_frame=build_resolved_frame(
            frame_ref=_TARGET,
            role_keys=("recipient",),
            rows=[{**row, "provenance": "src_refreshed"} for row in malaria_ethiopia_rows()],
        )
    )
    component_id = _create(client, "source_card")

    response = _render(client, component_id, _TARGET)

    assert response.status_code == 200, response.text
    payload = response.json()["resolved_payload"]
    sources = payload["provenance"]["sources"]
    assert payload["provenance"]["inputs"] == [_body(_TARGET)]
    assert payload["provenance"]["refreshed_from"] == 1
    assert [source["name"] for source in sources] == ["src_refreshed"]
    assert payload["component"] == {
        "type": "source_card",
        "source_refs": [source["id"] for source in sources],
    }


# -- acceptance -------------------------------------------------------------


def test_the_stored_frame_is_accepted_with_no_record_of_it(
    frame_store, revision_store, release_registry
):
    """A surface with neither reader accepts the frame the revision is stored on, and only it."""
    app = _app(
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        live_refresh=LiveRefreshPrincipal(header="x-dfo-principal", principal=_PRINCIPAL),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        component_id = _create(client)
        stored = _render(client, component_id, _STORED)
        other = _render(client, component_id, _TARGET)

    assert stored.status_code == 200
    assert stored.json()["resolved_payload"]["provenance"]["refreshed_from"] == 1
    assert (other.status_code, other.json()["code"]) == (409, "render_target_mismatch")


def test_a_collapsed_frame_is_accepted_for_a_collapse_with_the_same_groupby(
    client: TestClient, snapshots: FakeSnapshotReader
):
    _record(snapshots, _STORED, groupby=_BY_RECIPIENT)
    _record(snapshots, _TARGET, groupby=_BY_RECIPIENT)
    component_id = _create(client)

    response = _render(client, component_id, _TARGET)

    assert response.status_code == 200, response.text


def test_a_frame_that_was_both_renderings_accepts_one_like_it_and_refuses_a_collapse(
    client: TestClient, snapshots: FakeSnapshotReader
):
    """The stored version links a groupby file and a plain file. `_TARGET` links both too, and
    `_UNRECORDED` is recorded here as a collapse with that groupby from another version."""
    both = [
        _location(digest="sha256:sum", rendering={"format": "parquet", "groupby": _BY_RECIPIENT}),
        _location(digest="sha256:plain", rendering={"format": "parquet"}),
    ]
    for frame in (_STORED, _TARGET):
        snapshots.locations[(frame.spec_id, frame.content_version)] = both
    _record(snapshots, _UNRECORDED, groupby=_BY_RECIPIENT)
    component_id = _create(client)

    alike = _render(client, component_id, _TARGET)
    collapse = _render(client, component_id, _UNRECORDED)

    assert alike.status_code == 200, alike.text
    assert (collapse.status_code, collapse.json()["code"]) == (409, "render_target_mismatch")


@pytest.mark.parametrize(
    ("stored_groupby", "target", "target_groupby"),
    [
        pytest.param(None, _OTHER_DATASET, None, id="another-dataset"),
        pytest.param(None, _TARGET, _BY_RECIPIENT, id="plain-card-collapsed-target"),
        pytest.param(_BY_RECIPIENT, _TARGET, None, id="collapsed-card-plain-target"),
        pytest.param(_BY_RECIPIENT, _TARGET, _BY_STAT_VAR, id="another-groupby"),
        pytest.param(None, _UNRECORDED, None, id="target-version-not-recorded"),
        pytest.param(None, _OPS_TARGET, None, id="derived-target-for-a-dataset-card"),
    ],
)
def test_a_dataset_card_refuses_a_target_that_is_not_its_dataset_rendering(
    client: TestClient,
    frame_store: FakeFrameStore,
    snapshots: FakeSnapshotReader,
    stored_groupby,
    target,
    target_groupby,
):
    _record(snapshots, _STORED, groupby=stored_groupby)
    if target.spec_id.startswith("spec_") and target != _UNRECORDED:
        _record(snapshots, target, groupby=target_groupby)
    component_id = _create(client)

    render = _render(client, component_id, target)
    data = _render_data(client, component_id, target)

    for response in (render, data):
        assert response.status_code == 409
        assert response.json()["code"] == "render_target_mismatch"
        assert response.json()["retryable"] is False
    assert (target, component_id) not in frame_store.pins


def test_a_derived_card_renders_on_a_frame_of_the_same_recipe(
    client: TestClient, recipes: _Recipes
):
    recipes.hashes = {_OPS_STORED: "rnf_" + "a" * 32, _OPS_TARGET: "rnf_" + "a" * 32}
    component_id = _create(client, frame=_OPS_STORED)

    response = _render(client, component_id, _OPS_TARGET)

    assert response.status_code == 200, response.text
    assert response.json()["resolved_payload"]["provenance"]["inputs"] == [_body(_OPS_TARGET)]


@pytest.mark.parametrize(
    ("hashes", "target"),
    [
        pytest.param(
            {_OPS_STORED: "rnf_" + "a" * 32, _OPS_TARGET: "rnf_" + "b" * 32},
            _OPS_TARGET,
            id="another-recipe",
        ),
        pytest.param({_OPS_TARGET: "rnf_" + "a" * 32}, _OPS_TARGET, id="stored-frame-no-recipe"),
        pytest.param({_OPS_STORED: "rnf_" + "a" * 32}, _OPS_TARGET, id="target-no-recipe"),
        pytest.param({}, _OPS_TARGET, id="no-recipe-at-all"),
        pytest.param(
            {_OPS_STORED: "rnf_" + "a" * 32}, _TARGET, id="dataset-target-for-a-derived-card"
        ),
    ],
)
def test_a_derived_card_refuses_a_target_that_is_not_a_frame_of_its_recipe(
    client: TestClient, recipes: _Recipes, hashes, target
):
    recipes.hashes = hashes
    component_id = _create(client, frame=_OPS_STORED)

    render = _render(client, component_id, target)
    data = _render_data(client, component_id, target)

    for response in (render, data):
        assert (response.status_code, response.json()["code"]) == (409, "render_target_mismatch")


def test_a_note_refuses_render_target_mismatch(client: TestClient):
    component_id = _create(client, "note_narration")

    render = _render(client, component_id, _TARGET)
    data = _render_data(client, component_id, _TARGET)

    for response in (render, data):
        assert (response.status_code, response.json()["code"]) == (409, "render_target_mismatch")


# -- refusals ---------------------------------------------------------------


@pytest.mark.parametrize("revision", [None, 2], ids=["unknown-component", "unknown-revision"])
@pytest.mark.parametrize("route", ["render", "render/data"])
def test_an_unknown_component_or_revision_refuses_not_found(client: TestClient, revision, route):
    component_id = "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY" if revision is None else _create(client)
    url = _render_url(component_id, revision or 1)

    if route == "render":
        response = client.post(url, json={"frame": _body(_TARGET)}, headers=_AUTH)
    else:
        response = client.get(f"{url}/data", params=_body(_TARGET), headers=_AUTH)

    assert (response.status_code, response.json()["code"]) == (404, "not_found")


@pytest.mark.parametrize(
    "request_json",
    [
        pytest.param({}, id="no-frame"),
        pytest.param({"frame": _body(_TARGET), "caller": "x"}, id="unknown-field"),
        pytest.param({"frame": {"spec_id": "bad id", "content_version": "cv_1"}}, id="bad-id"),
    ],
)
def test_a_malformed_render_body_refuses_request_invalid(client: TestClient, request_json):
    component_id = _create(client)

    response = client.post(_render_url(component_id), json=request_json, headers=_AUTH)

    assert (response.status_code, response.json()["code"]) == (400, "request_invalid")


def test_a_target_the_stored_intent_does_not_bind_refuses_binding_invalid_at_422(
    client: TestClient, frame_store: FakeFrameStore
):
    """The target holds Kenya and Ghana, and the card selects Ethiopia. A target holding one
    recipient would draw it in Ethiopia's place."""
    frame_store.put(resolved_frame=_frame(_TARGET, recipients=("country/KEN", "country/GHA")))
    component_id = _create(client)

    render = _render(client, component_id, _TARGET)
    data = _render_data(client, component_id, _TARGET)

    for response in (render, data):
        assert (response.status_code, response.json()["code"]) == (422, "binding_invalid")
        assert response.json()["pointer"] == "/select/recipient"
    assert (_TARGET, component_id) not in frame_store.pins


def test_a_target_frame_that_does_not_load_refuses_frame_unavailable_at_409(
    client: TestClient, frame_store: FakeFrameStore, monkeypatch: pytest.MonkeyPatch
):
    component_id = _create(client)

    def expired(*, frame_ref: FrameRef):
        raise FrameUnavailable(reason="expired", detail="aged out")

    monkeypatch.setattr(frame_store, "load", expired)

    response = _render(client, component_id, _TARGET)

    assert (response.status_code, response.json()["code"]) == (409, "frame_unavailable")
    assert response.json()["retryable"] is True


@pytest.mark.parametrize(
    ("raised", "status", "code", "retryable"),
    [
        pytest.param(
            BindingInvalid(
                pointer="/select/recipient",
                dimension="recipient",
                legal_values=("country/KEN",),
                truncated=False,
                detail="absent",
            ),
            422,
            "binding_invalid",
            False,
            id="binding-invalid",
        ),
        pytest.param(
            ConstraintViolation(
                constraint_id="log_scale_requires_positive_values",
                pointer="/scale",
                detail="non-positive",
                fix="use linear",
            ),
            422,
            "constraint_violation",
            False,
            id="constraint-violation",
        ),
        pytest.param(
            SchemaInvalid(pointer="/select", detail="missing", minimal_intent={"type": "x"}),
            400,
            "schema_invalid",
            False,
            id="schema-invalid-missing-field",
        ),
        pytest.param(
            SchemaInvalid(pointer="/type", detail="unknown", legal_fields=("type",)),
            400,
            "schema_invalid",
            False,
            id="schema-invalid-illegal-field",
        ),
        pytest.param(
            FrameUnavailable(reason="expired", detail="aged out"),
            409,
            "frame_unavailable",
            True,
            id="frame-unavailable-expired",
        ),
        pytest.param(
            FrameUnavailable(reason="revoked", detail="revoked"),
            409,
            "frame_unavailable",
            False,
            id="frame-unavailable-revoked",
        ),
        pytest.param(
            ReleaseMismatch(catalog_version="2099.1.0", detail="newer"),
            409,
            "release_mismatch",
            True,
            id="release-mismatch",
        ),
        pytest.param(
            ResourceExhausted(pointer="/select", limit_name="MAX_SERIES", detail="too many"),
            409,
            "resource_exhausted",
            False,
            id="resource-exhausted",
        ),
        pytest.param(EditConflict(), 409, "edit_conflict", True, id="edit-conflict"),
    ],
)
def test_a_refusal_carries_its_render_status_and_retries_only_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, raised, status, code, retryable
):
    """The caller replays a stored intent, so a code that asks for a corrected request answers
    `retryable` false."""
    component_id = _create(client)

    def refuse(**_kwargs: object):
        raise raised

    monkeypatch.setattr(render_door, "render_component", refuse)
    monkeypatch.setattr(render_door, "export_component_data", refuse)

    render = _render(client, component_id, _TARGET)
    data = _render_data(client, component_id, _TARGET)

    for response in (render, data):
        body = response.json()
        assert (response.status_code, body["code"], body["retryable"]) == (
            status,
            code,
            retryable,
        )


# -- the data export --------------------------------------------------------


def test_render_data_serves_the_target_frame_s_rows_and_pins_nothing(
    client: TestClient, frame_store: FakeFrameStore
):
    component_id = _create(client, "stat_card")
    stored = client.get(
        f"/api/presentation/components/{component_id}/revisions/1/data",
        params={"bundle": "false"},
    )

    response = _render_data(client, component_id, _TARGET, bundle="false")

    def values(text: str) -> list[float]:
        return [float(row["value"]) for row in csv.DictReader(io.StringIO(text))]

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"] == (
        'attachment; filename="dfo_dcid-malariagrantdisbursements-ethiopia_2026-10-06_r1.csv"'
    )
    assert values(response.text) == [value + _MOVED_BY for value in values(stored.text)]
    assert (_TARGET, component_id) not in frame_store.pins


def test_the_default_render_download_is_a_bundle_dated_by_the_target_frame(client: TestClient):
    """The revision was created on 2026-10-04 and the target frame retrieved on 2026-10-06."""
    component_id = _create(client, "stat_card")

    response = _render_data(client, component_id, _TARGET)
    descriptor = json.loads(zipfile.ZipFile(io.BytesIO(response.content)).read("datapackage.json"))

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == (
        'attachment; filename="dfo_dcid-malariagrantdisbursements-ethiopia_2026-10-06_csv_r1.zip"'
    )
    assert descriptor["dfo"]["component"]["frame_ref"] == _body(_TARGET)
    assert descriptor["dfo"]["prepared_at"] == "2026-10-06T00:00:00Z"
    assert "reproducing_call" not in descriptor["dfo"]


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"format": "json"}, id="unknown-format"),
        pytest.param({"bundle": "maybe"}, id="unparseable-bundle"),
        pytest.param({"spec_id": "bad id"}, id="bad-spec-id"),
    ],
)
def test_a_bad_render_data_query_refuses_request_invalid(client: TestClient, params):
    component_id = _create(client, "stat_card")

    response = _render_data(client, component_id, _TARGET, **params)

    assert (response.status_code, response.json()["code"]) == (400, "request_invalid")


def test_render_data_of_a_source_card_refuses_not_found(client: TestClient):
    component_id = _create(client, "source_card")

    response = _render_data(client, component_id, _TARGET)

    assert (response.status_code, response.json()["code"]) == (404, "not_found")


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
