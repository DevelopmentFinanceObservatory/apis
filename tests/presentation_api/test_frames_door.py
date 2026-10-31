"""Checks frame views, typed refusals, and executor backpressure through the API."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Never

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import presentation_api.app as presentation_app
import presentation_api.doors.frames as frames_door
import presentation_store.frame_store as frame_store_module
from assembler.catalogue import CatalogueRelease
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import FrameRef, FrameUnavailable
from assembler.testing import build_resolved_frame, malaria_ethiopia_rows
from dataset_store.snapshot import SnapshotRevoked
from datasets_api.doors.definition import SpecNotFound
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.doors.refresh import RefreshRebuildFailed
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform.app import create_app
from materialise.reshape import ColumnUnknownError
from presentation_api.app import presentation_surface
from presentation_api.config import PRESENTATION_FRAMES_TIMEOUT_SECONDS
from presentation_api.doors.frames import classify_frames_exception
from presentation_api.newest_frames import NewestPending
from spec_registry.errors import (
    IdentityQuarantined,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from tests.presentation_api.conftest import BARE_SPEC_ID, FakeNewestSpecs
from tests.presentation_store.test_frame_store import _posted_frame_fixture, _store


def _frame_url(frame_ref: FrameRef) -> str:
    return f"/api/presentation/frames/{frame_ref.spec_id}/{frame_ref.content_version}"


def _get_frame(client: TestClient, frame_ref: FrameRef):
    return client.get(_frame_url(frame_ref))


def test_structural_view_mode(client: TestClient, frame_ref: FrameRef):
    resp = client.get(_frame_url(frame_ref))
    assert resp.status_code == 200
    body = resp.json()
    names = {dimension["name"] for dimension in body["dimensions"]}
    assert "recipient" in names
    assert body["series_count"] >= 1


def test_the_frame_load_runs_under_the_door_s_read_deadline(
    client: TestClient, frame_store: FakeFrameStore, frame_ref: FrameRef, monkeypatch
):
    """The job's `load()` sees the deadline the door set before it submitted the job."""
    seen: list[float | None] = []
    load = frame_store.load

    def recording_load(*, frame_ref: FrameRef):
        seen.append(frame_store_module._door_deadline.get())
        return load(frame_ref=frame_ref)

    monkeypatch.setattr(frame_store, "load", recording_load)
    timeout = PRESENTATION_FRAMES_TIMEOUT_SECONDS
    budget = timeout - min(1.0, timeout * 0.1)

    before = time.monotonic()
    response = _get_frame(client, frame_ref)
    after = time.monotonic()

    assert response.status_code == 200
    [deadline] = seen
    assert deadline is not None
    assert before + budget <= deadline <= after + budget


def test_dimension_mode_lists_every_value_uncapped(client: TestClient, frame_ref: FrameRef):
    resp = client.get(
        _frame_url(frame_ref),
        params={"dimension": "recipient"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dimension"] == "recipient"
    assert "country/ETH" in body["values"]


def test_query_mode_filters_every_dimension(client: TestClient, frame_ref: FrameRef):
    resp = client.get(_frame_url(frame_ref), params={"query": "eth"})
    assert resp.status_code == 200
    body = resp.json()
    recipient = next(d for d in body["dimensions"] if d["name"] == "recipient")
    assert recipient["values"] == ["country/ETH"]


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [("x-dfo-session", "session-a")],
        [("x-dfo-session", "")],
        [("x-dfo-session", "session-a"), ("x-dfo-session", "session-b")],
    ],
    ids=["none", "set", "empty", "repeated"],
)
def test_a_frame_read_ignores_the_session_header(
    client: TestClient, frame_ref: FrameRef, headers: list[tuple[str, str]]
):
    resp = client.get(_frame_url(frame_ref), headers=headers)
    assert resp.status_code == 200


def test_an_unknown_dimension_is_binding_invalid(client: TestClient, frame_ref: FrameRef):
    resp = client.get(
        _frame_url(frame_ref),
        params={"dimension": "not_a_real_dimension"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "binding_invalid"


def test_an_unregistered_frame_is_frame_unavailable(client: TestClient):
    # The fake maps an unregistered frame to retryable 503 `io`.
    resp = client.get(
        "/api/presentation/frames/spec_00000000000000000000000000000000/"
        "cv_00000000000000000000000000000000",
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "frame_unavailable"
    assert body["retryable"] is True


# -- spec_id/content_version path param format validation --------------------


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(
            "/api/presentation/frames/spec_bad!chars/cv_00000000000000000000000000000000",
            id="malformed_spec_id",
        ),
        pytest.param(
            "/api/presentation/frames/spec_00000000000000000000000000000000/not-a-content-version",
            id="malformed_content_version",
        ),
    ],
)
def test_malformed_frame_path_identifiers_are_request_invalid(client: TestClient, url: str):
    resp = client.get(
        url,
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False


class _OpsUnsupportedFrameStore:
    """Stub the retryable refusal for an unpublished `ops_`/`ocv_` frame."""

    def load(self, *, frame_ref: FrameRef) -> Never:
        raise FrameUnavailable(
            reason="ops_unsupported",
            detail=f"ops spec {frame_ref.spec_id} was not published to the platform",
        )

    def pin(self, *, frame_ref: FrameRef, component_id: str) -> Never:
        raise AssertionError("pin should not run after an ops_unsupported refusal")


def test_an_unpublished_ops_prefixed_spec_id_maps_the_typed_refusal_to_not_found(
    release: CatalogueRelease,
):
    app = create_app(
        [
            presentation_surface(
                frame_store=_OpsUnsupportedFrameStore(),
                revision_store=FakeRevisionStore(),
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get(
            "/api/presentation/frames/ops_00000000000000000000000000000000/"
            "cv_00000000000000000000000000000000",
        )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "frame_unavailable"
    # Publishing the frame pair makes this retryable 404 succeed.
    assert body["retryable"] is True


class _RefusingFrameStore(FakeFrameStore):
    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason

    def load(self, *, frame_ref: FrameRef) -> Never:
        raise FrameUnavailable(reason=self.reason, detail="exemplar")


@pytest.mark.parametrize(
    ("reason", "status", "retryable"),
    [
        pytest.param("unknown", 404, False, id="lost-never-stored"),
        pytest.param("expired", 404, True, id="lost-aged-out"),
        pytest.param("revoked", 404, False, id="permanent-taken-down"),
        pytest.param("quarantined", 404, False, id="permanent-quarantined"),
    ],
)
def test_a_frame_that_does_not_load_names_its_reason_as_the_cause(
    release: CatalogueRelease, reason: str, status: int, retryable: bool
):
    app = create_app(
        [
            presentation_surface(
                frame_store=_RefusingFrameStore(reason),
                revision_store=FakeRevisionStore(),
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get(
            "/api/presentation/frames/spec_00000000000000000000000000000000/"
            "cv_00000000000000000000000000000000",
        )

    assert resp.status_code == status, resp.text
    body = resp.json()
    assert (body["code"], body["retryable"]) == ("frame_unavailable", retryable)
    assert body["cause"]["code"] == reason


def test_an_ocv_prefixed_content_version_reaches_the_frame_store(
    release: CatalogueRelease,
):
    """The door accepts `ocv_` content versions and leaves frame lookup to the store."""
    app = create_app(
        [
            presentation_surface(
                frame_store=_OpsUnsupportedFrameStore(),
                revision_store=FakeRevisionStore(),
                release_registry=FakeReleaseRegistry(release=release),
            )
        ]
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get(
            "/api/presentation/frames/ops_00000000000000000000000000000000/"
            "ocv_00000000000000000000000000000000",
        )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "frame_unavailable"
    assert body["retryable"] is True


def test_a_published_ops_prefixed_frame_ref_serves_the_structural_view(
    release: CatalogueRelease,
):
    """A published `ops_`/`ocv_` frame returns its structural view."""
    frame_ref = FrameRef(spec_id="ops_" + "e" * 32, content_version="ocv_" + "f" * 32)
    frame_store = FakeFrameStore()
    frame_store.put(
        resolved_frame=build_resolved_frame(
            frame_ref=frame_ref,
            role_keys=("recipient",),
            rows=malaria_ethiopia_rows(),
            labels={"country/ETH": "Ethiopia"},
        )
    )
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
        resp = client.get(
            f"/api/presentation/frames/{frame_ref.spec_id}/{frame_ref.content_version}",
        )
    assert resp.status_code == 200
    body = resp.json()
    names = {dimension["name"] for dimension in body["dimensions"]}
    assert "recipient" in names


def test_the_backpressure_permit_covers_encoding_the_reply(
    monkeypatch: pytest.MonkeyPatch,
    frame_store: FakeFrameStore,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
    frame_ref: FrameRef,
):
    """Encoding holds the sole permit, so a concurrent read returns 503."""
    monkeypatch.setattr(frames_door, "PRESENTATION_FRAMES_MAX_CONCURRENCY", 1)
    rendering = threading.Event()
    finish_render = threading.Event()
    render_threads: list[str] = []

    class BlockingJSONResponse(JSONResponse):
        def render(self, content: object) -> bytes:
            render_threads.append(threading.current_thread().name)
            rendering.set()
            finish_render.wait(timeout=5.0)
            return super().render(content)

    monkeypatch.setattr(frames_door, "JSONResponse", BlockingJSONResponse)
    surface = presentation_surface(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    with (
        TestClient(create_app([surface]), raise_server_exceptions=False) as client,
        ThreadPoolExecutor(max_workers=1) as caller,
    ):
        first = caller.submit(_get_frame, client, frame_ref)
        try:
            assert rendering.wait(timeout=5.0)
            concurrent = _get_frame(client, frame_ref)
        finally:
            finish_render.set()
        first_resp = first.result(timeout=5.0)

    assert concurrent.status_code == 503
    assert concurrent.json()["code"] == "service_unavailable"
    assert first_resp.status_code == 200
    assert len(render_threads) == 1
    assert render_threads[0].startswith("presentation-frames")


# -- DatasetsStoreUnavailable transient/permanent split -----------------


def test_a_transient_datasets_store_failure_is_service_unavailable_and_retryable():
    """A failure without SQLSTATE maps to retryable 503."""
    spec = classify_frames_exception(DatasetsStoreUnavailable("exemplar"))
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


def test_a_permanent_datasets_store_failure_is_internal_error_and_not_retryable():
    """A permanent-class SQLSTATE maps to non-retryable 500."""
    spec = classify_frames_exception(DatasetsStoreUnavailable("exemplar", sqlstate="23503"))
    assert spec is not None
    assert spec.status == 500
    assert spec.code == "internal_error"
    assert spec.retryable is False


class _RegistryDownFrameStore(FakeFrameStore):
    """A frame store whose revocation-scope read finds the spec registry unavailable."""

    def load(self, *, frame_ref: FrameRef) -> Never:
        raise SpecRegistryUnavailable("spec registry pool borrow timed out")


def test_a_frame_load_the_registry_cannot_scope_answers_retryable_service_unavailable(
    frame_ref: FrameRef, revision_store: FakeRevisionStore, release_registry: FakeReleaseRegistry
):
    """A frame load fails closed when the registry cannot name the ids its revocation check
    covers, and the route answers retryable 503 as a Datasets door does for an unavailable
    store, never 500."""
    surface = presentation_surface(
        frame_store=_RegistryDownFrameStore(),
        revision_store=revision_store,
        release_registry=release_registry,
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        resp = _get_frame(client, frame_ref)

    assert resp.status_code == 503, resp.text
    assert (resp.json()["code"], resp.json()["retryable"]) == ("service_unavailable", True)


@pytest.mark.parametrize(
    ("fault", "status", "code", "retryable"),
    [
        ("unavailable", 503, "service_unavailable", True),
        ("store-error", 503, "service_unavailable", True),
        ("quarantined", 404, "frame_unavailable", False),
    ],
)
def test_a_live_frame_load_whose_registry_read_fails_answers_a_declared_refusal(
    fault: str,
    status: int,
    code: str,
    retryable: bool,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
):
    """A real frame store reads the registry to scope a frame ref's revocation check. Each
    fault that read can raise answers a declared refusal through the frames route, never 500."""
    snapshots, artifacts, frame_ref, _served_id, spec_reader = _posted_frame_fixture()
    if fault == "quarantined":
        spec_reader.quarantined = frozenset({frame_ref.spec_id})
    else:
        spec_reader.definition_error = (
            SpecRegistryUnavailable("spec registry pool borrow timed out")
            if fault == "unavailable"
            else SpecRegistryStoreError("spec registry read failed")
        )
    surface = presentation_surface(
        frame_store=_store(snapshots=snapshots, artifacts=artifacts, spec_reader=spec_reader),
        revision_store=revision_store,
        release_registry=release_registry,
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        resp = _get_frame(client, frame_ref)

    assert (resp.status_code, resp.json()["code"], resp.json()["retryable"]) == (
        status,
        code,
        retryable,
    ), resp.text


# -- a frame named by its dataset id alone -----------------------------------


def test_a_bare_spec_id_reads_the_frame_it_resolves_and_names_its_pair(
    newest_client: TestClient, newest_specs: FakeNewestSpecs, frame_ref: FrameRef
):
    """The pair route's body, with the served id and version the dataset id resolved to."""
    pair = _get_frame(newest_client, frame_ref).json()

    resp = newest_client.get(f"/api/presentation/frames/{BARE_SPEC_ID}")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        **pair,
        "spec_id": frame_ref.spec_id,
        "content_version": frame_ref.content_version,
    }
    assert newest_specs.asked == [BARE_SPEC_ID]


@pytest.mark.parametrize(
    ("params", "key"), [({"dimension": "recipient"}, "values"), ({"query": "eth"}, "dimensions")]
)
def test_a_bare_spec_id_takes_the_pair_route_s_dimension_and_query(
    newest_client: TestClient, frame_ref: FrameRef, params: dict[str, str], key: str
):
    pair = newest_client.get(_frame_url(frame_ref), params=params).json()

    resp = newest_client.get(f"/api/presentation/frames/{BARE_SPEC_ID}", params=params)

    assert resp.status_code == 200, resp.text
    assert resp.json()[key] == pair[key]
    assert resp.json()["content_version"] == frame_ref.content_version


@pytest.mark.parametrize(
    "spec_id",
    [
        pytest.param("ops_" + "c" * 32, id="derived"),
        pytest.param("spec_" + "C" * 32, id="upper-hex"),
        pytest.param("spec_" + "c" * 31, id="short"),
    ],
)
def test_a_bare_id_the_runner_takes_no_spec_root_of_is_request_invalid(
    newest_client: TestClient, newest_specs: FakeNewestSpecs, spec_id: str
):
    resp = newest_client.get(f"/api/presentation/frames/{spec_id}")

    assert resp.status_code == 400
    assert resp.json()["code"] == "request_invalid"
    assert newest_specs.asked == []


@pytest.mark.parametrize(
    ("answer", "status", "retryable", "cause"),
    [
        pytest.param(NewestPending(retry_after_s=2.0), 503, True, "io", id="building"),
        pytest.param(SpecNotFound(id=BARE_SPEC_ID), 404, False, "unknown", id="not-registered"),
        pytest.param(MintBackpressure(), 503, True, "io", id="mint-backpressure"),
        pytest.param(
            IdentityQuarantined(subject_id=BARE_SPEC_ID, id_kind="spec"),
            404,
            False,
            "quarantined",
            id="quarantined",
        ),
        pytest.param(
            SnapshotRevoked(id=BARE_SPEC_ID, content_version="cv", reason=None),
            404,
            False,
            "revoked",
            id="revoked",
        ),
        pytest.param(
            ColumnUnknownError("no such column", column="x"),
            404,
            False,
            "unknown",
            id="code-analysis-does-not-map",
        ),
    ],
)
def test_a_bare_spec_id_that_cannot_be_served_is_frame_unavailable(
    newest_client: TestClient,
    newest_specs: FakeNewestSpecs,
    answer: object,
    status: int,
    retryable: bool,
    cause: str,
):
    newest_specs.answer = answer  # ty: ignore[invalid-assignment]

    resp = newest_client.get(f"/api/presentation/frames/{BARE_SPEC_ID}")

    assert resp.status_code == status, resp.text
    assert resp.json()["code"] == "frame_unavailable"
    assert resp.json()["retryable"] is retryable
    assert resp.json()["cause"]["code"] == cause


def test_bare_reads_at_their_limit_refuse_and_leave_pair_reads_their_permits(
    monkeypatch: pytest.MonkeyPatch,
    frame_store: FakeFrameStore,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
    frame_ref: FrameRef,
):
    """A rebuild holds a bare read's permit for seconds, so pair reads run beside it."""
    monkeypatch.setattr(frames_door, "PRESENTATION_FRAMES_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(presentation_app, "PRESENTATION_FRAMES_BARE_MAX_CONCURRENCY", 1)
    rebuilding = threading.Event()
    finish_rebuild = threading.Event()

    class Rebuilding(FakeNewestSpecs):
        def newest_of_spec(self, spec_id: str) -> FrameRef:
            rebuilding.set()
            finish_rebuild.wait(timeout=5.0)
            return frame_ref

    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        newest_frames=Rebuilding(),
    )
    url = f"/api/presentation/frames/{BARE_SPEC_ID}"
    with (
        TestClient(create_app([surface]), raise_server_exceptions=False) as client,
        ThreadPoolExecutor(max_workers=1) as caller,
    ):
        first = caller.submit(client.get, url)
        try:
            assert rebuilding.wait(timeout=5.0)
            concurrent = client.get(url)
            pair = _get_frame(client, frame_ref)
        finally:
            finish_rebuild.set()
        first_resp = first.result(timeout=5.0)

    assert concurrent.status_code == 503, concurrent.text
    assert concurrent.json()["code"] == "service_unavailable"
    assert pair.status_code == 200, pair.text
    assert first_resp.status_code == 200, first_resp.text


def test_a_bare_spec_id_without_a_datasets_surface_is_frame_unavailable(client: TestClient):
    resp = client.get(f"/api/presentation/frames/{BARE_SPEC_ID}")

    assert resp.status_code == 404
    assert resp.json()["code"] == "frame_unavailable"


def test_a_bare_spec_id_whose_rebuild_failed_for_good_is_internal_error(
    newest_client: TestClient, newest_specs: FakeNewestSpecs
):
    """A server fault answers 500, as the analysis route answers it."""
    newest_specs.answer = RefreshRebuildFailed(operation_id="op_1", refusal_code="over_ceiling")

    resp = newest_client.get(f"/api/presentation/frames/{BARE_SPEC_ID}")

    assert resp.status_code == 500, resp.text
    assert resp.json()["code"] == "internal_error"
    assert resp.json()["retryable"] is False
