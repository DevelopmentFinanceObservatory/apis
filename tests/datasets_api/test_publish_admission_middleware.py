"""Tests `datasets_api.doors.publish._admit_publish_body_size` through HTTP. Middleware
admission runs before FastAPI reads and decodes the body.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.doors.publish import (
    MAX_PUBLISH_BODY_BYTES,
    PublishBodyTooLarge,
    PublishContentLengthRequired,
    build_publish_door,
    classify_publish_exception,
)
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import NullVersionRecordWriter
from dfo_platform.codes import PLATFORM_CODE_CATALOGUE, merge_catalogues
from dfo_platform.problem import Envelope
from tests.artifact_store.conftest import FakeGCSClient

from ._artifact_doubles import FakeArtifactStore
from ._snapshot_doubles import FakeSnapshotReader, make_client_for_door
from ._version_record_store import FakeVersionRecordDatasetStore

_PUBLISH_ROUTE = "/api/datasets/frames/publish"
_PRINCIPAL = "publisher-principal"
_AUTH_HEADERS = {"x-dfo-principal": _PRINCIPAL}

# Any Content-Length strictly above MAX_PUBLISH_BODY_BYTES trips the
# check; +1 is the tightest such value and so the strongest test of the
# boundary itself.
_OVER_LIMIT_CONTENT_LENGTH = MAX_PUBLISH_BODY_BYTES + 1
_UNDER_LIMIT_CONTENT_LENGTH = MAX_PUBLISH_BODY_BYTES - 1


def _client() -> TestClient:
    door = build_publish_door(
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        version_writer=NullVersionRecordWriter(),
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    return make_client_for_door(door)


def _real_publish_client() -> TestClient:
    """Wires shared SQLite stores for a successful publish."""
    store = FakeVersionRecordDatasetStore()
    artifact_store = LiveArtifactStore(
        buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
        inventory=LiveArtifactInventory(store=store),
        client=FakeGCSClient(),
    )
    door = build_publish_door(
        artifact_store=artifact_store,
        snapshot_reader=LiveSnapshotReader(store=store),
        version_writer=LiveVersionRecordWriter(store=store),
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    return make_client_for_door(door)


def _valid_publish_body() -> dict[str, object]:
    return {
        "schema_version": 1,
        "spec_id": "ops_abc123",
        "content_version": "ocv_abc123",
        "role_keys": ["recipient"],
        "rows": [
            {
                "stat_var": "dcid:Exemplar",
                "time_period": "2020",
                "value": 1.0,
                "recipient": "country/KEN",
            }
        ],
        "input_content_versions": ["cv_abc123"],
    }


def _chunked_body(payload: bytes) -> Iterator[bytes]:
    """Makes httpx send chunked transfer encoding without a `Content-Length` header (#894)."""
    yield payload


def test_oversized_content_length_with_malformed_json_refuses_before_decode() -> None:
    """An oversized malformed body gets the admission refusal before JSON decoding."""
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=b"{not valid json at all",
        headers={
            "content-length": str(_OVER_LIMIT_CONTENT_LENGTH),
            "content-type": "application/json",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "request_invalid"
    assert "admission" in body["detail"] and "Content-Length" in body["detail"]


def test_under_limit_malformed_json_reaches_the_ordinary_decode_failure() -> None:
    """The same malformed body below the limit reaches FastAPI's JSON decoder."""
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=b"{not valid json at all",
        headers={
            "content-length": str(_UNDER_LIMIT_CONTENT_LENGTH),
            "content-type": "application/json",
            **_AUTH_HEADERS,
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "request_invalid"
    assert "admission" not in body["detail"]


def test_response_matches_a_door_raised_request_invalid_refusal_field_by_field() -> None:
    """The middleware and route exception paths render the same envelope apart from `trace_id`."""
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=b"irrelevant -- never read",
        headers={
            "content-length": str(_OVER_LIMIT_CONTENT_LENGTH),
            "content-type": "application/json",
        },
    )
    assert response.status_code == 400

    spec = classify_publish_exception(
        PublishBodyTooLarge(content_length=_OVER_LIMIT_CONTENT_LENGTH, limit=MAX_PUBLISH_BODY_BYTES)
    )
    assert spec is not None
    envelope = Envelope(
        catalogue=merge_catalogues(PLATFORM_CODE_CATALOGUE, DATASETS_CODE_CATALOGUE),
        capability_version=DATASETS_CAPABILITY_VERSION,
    )
    expected = envelope.details(
        status=spec.status,
        code=spec.code,
        detail=spec.detail,
        retryable=spec.retryable,
    )

    body = response.json()
    for field in ("type", "title", "status", "detail", "code", "capability_version", "retryable"):
        assert body[field] == getattr(expected, field), field

    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-dfo-trace-id"] == body["trace_id"]
    assert "x-dfo-commit" in response.headers
    assert "Strict-Transport-Security" in response.headers


def test_response_matches_a_genuinely_door_raised_refusal_over_the_wire() -> None:
    """Compares middleware output with a duplicate-row `PublishFrameInvalid` response."""
    client = _client()

    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
    }
    door_raised = client.post(
        _PUBLISH_ROUTE,
        json={
            "schema_version": 1,
            "spec_id": "ops_abc123",
            "content_version": "ocv_abc123",
            "role_keys": ["recipient"],
            "rows": [row, row],  # duplicate grain -> Frame.build's own FrameError
            "input_content_versions": ["cv_abc123"],
        },
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert door_raised.status_code == 400
    door_raised_body = door_raised.json()
    assert door_raised_body["code"] == "request_invalid"

    admission_refused = client.post(
        _PUBLISH_ROUTE,
        content=b"irrelevant -- never read",
        headers={
            "content-length": str(_OVER_LIMIT_CONTENT_LENGTH),
            "content-type": "application/json",
        },
    )
    assert admission_refused.status_code == 400
    admission_body = admission_refused.json()
    assert admission_body["code"] == "request_invalid"

    for field in ("type", "title", "status", "code", "capability_version", "retryable"):
        assert admission_body[field] == door_raised_body[field], field
    # Each refusal has its own detail text.

    for header in ("content-type", "cache-control"):
        assert admission_refused.headers[header] == door_raised.headers[header]
    assert "x-dfo-trace-id" in admission_refused.headers
    assert "x-dfo-trace-id" in door_raised.headers


def test_admission_check_ignores_a_request_to_another_route() -> None:
    """The admission middleware applies only to `_PUBLISH_ROUTE`."""
    client = _client()

    response = client.post(
        "/api/datasets/frames/publish/not-a-real-path",
        content=b"irrelevant",
        headers={"content-length": str(_OVER_LIMIT_CONTENT_LENGTH)},
    )

    assert response.status_code == 404


def test_chunked_request_with_no_content_length_refuses_before_reaching_the_route() -> None:
    """A chunked request without `Content-Length` hits the #894 admission guard before auth.

    The request omits `x-dfo-principal`; `publish_not_permitted` would mean it reached auth.
    """
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=_chunked_body(b'{"irrelevant": "never read"}'),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["code"] != "publish_not_permitted"
    assert "Content-Length" in body["detail"]


def test_unparseable_content_length_refuses_the_same_way_as_absent() -> None:
    """An unparseable `Content-Length` receives the same 400 refusal as a missing header."""
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=b"irrelevant -- never read",
        headers={"content-length": "not-a-number", "content-type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "request_invalid"
    assert "Content-Length" in body["detail"]

    spec = classify_publish_exception(
        PublishContentLengthRequired(content_length_header="not-a-number")
    )
    assert spec is not None
    assert body["detail"] == spec.detail


def test_normal_request_with_valid_content_length_under_the_limit_still_publishes() -> None:
    """A valid under-limit request succeeds through the publish path with a live version writer."""
    client = _real_publish_client()

    response = client.post(_PUBLISH_ROUTE, json=_valid_publish_body(), headers=_AUTH_HEADERS)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["published"]["created"] is True
    assert body["published"]["spec_id"] == "ops_abc123"
