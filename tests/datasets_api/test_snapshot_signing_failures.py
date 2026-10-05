"""Tests signing-failure classification at the helper, snapshot-door, and composed-app layers."""

from __future__ import annotations

import logging

import httpx
import pytest

from artifact_store import ArtifactNotFoundError, ArtifactSigningError, Tier
from datasets_api.codes_snapshot import SnapshotSigningFailed
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.snapshot import (
    build_snapshot_door,
    classify_snapshot_exception,
    read_snapshot_manifest,
)
from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxClient
from spec_registry import LiveSpecRegistry

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import (
    _NOW,
    FakeSnapshotReader,
    FakeSpecReader,
    _location,
    _store_and_locate,
    _version,
    make_client_for_door,
)
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter


def test_signing_a_missing_artifact_propagates_unclassified():
    # An orphaned location raises ArtifactNotFoundError and reaches the generic 500 handler.
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [_location(digest="sha256:never-stored", stored_at=_NOW)]},
    )
    with pytest.raises(ArtifactNotFoundError):
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv1",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=FakeArtifactStore(),
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )


@pytest.mark.parametrize(
    ("error_class_name", "status_code"),
    [
        pytest.param(None, None, id="unattributed"),
        pytest.param("ServiceUnavailable", 503, id="service-unavailable-503"),
        pytest.param("TooManyRequests", 429, id="rate-limited-429"),
        pytest.param("RefreshError", None, id="refresh-no-status"),
    ],
)
def test_transient_or_unattributed_signing_failure_maps_to_service_unavailable(
    error_class_name: str | None, status_code: int | None
):
    exc = SnapshotSigningFailed(
        bucket="fake-bucket",
        object_name="sha256:deadbeef",
        digest="sha256:deadbeef",
        error_class_name=error_class_name,
        status_code=status_code,
    )

    spec = classify_snapshot_exception(exc)

    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


@pytest.mark.parametrize(
    ("error_class_name", "status_code"),
    [
        pytest.param("RefreshError", 403, id="refresh-403"),
        pytest.param("MalformedError", None, id="malformed-no-status"),
    ],
)
def test_permanent_signing_failure_maps_to_internal_error_non_retryable(
    error_class_name: str, status_code: int | None
):
    exc = SnapshotSigningFailed(
        bucket="fake-bucket",
        object_name="sha256:deadbeef",
        digest="sha256:deadbeef",
        error_class_name=error_class_name,
        status_code=status_code,
    )

    spec = classify_snapshot_exception(exc)

    assert spec is not None
    assert spec.status == 500
    assert spec.code == "internal_error"
    assert spec.retryable is False


class _SigningFailureArtifactStore(FakeArtifactStore):
    """`put()` succeeds normally; `signed_get_url()` always raises
    `ArtifactSigningError` -- stands in for a transient IAM signBlob outage
    hitting this door while it re-signs a URL for an already-stored
    artifact (mirrors the mint door's own `_SigningFailureArtifactStore`,
    `tests/datasets_api/test_mint_error_classification.py`).
    """

    def signed_get_url(
        self, digest: str, *, tier: Tier, ttl, response_type: str, response_disposition: str
    ):
        raise ArtifactSigningError(bucket="fake-bucket", object_name=digest, digest=digest)


class _CredentialFailureArtifactStore(FakeArtifactStore):
    """`put()` succeeds normally; `signed_get_url()` always raises
    `ArtifactSigningError` shaped like a permission/credential failure
    (`error_class_name="RefreshError"`, `status_code=403`) -- stands in for
    a broken IAM policy hitting this door while it re-signs a URL for an
    already-stored artifact.
    """

    def signed_get_url(
        self, digest: str, *, tier: Tier, ttl, response_type: str, response_disposition: str
    ):
        raise ArtifactSigningError(
            bucket="fake-bucket",
            object_name=digest,
            digest=digest,
            error_class_name="RefreshError",
            status_code=403,
        )


def test_read_snapshot_manifest_wraps_a_signing_failure_without_resurrecting_context():
    """The wrapper leaves `__context__` and `__cause__` unset for Starlette middleware."""
    store = _SigningFailureArtifactStore()
    location = _store_and_locate(store, b"hello")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    with pytest.raises(SnapshotSigningFailed) as excinfo:
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv1",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )

    assert excinfo.value.__context__ is None
    assert excinfo.value.__cause__ is None


def test_signing_failure_maps_to_service_unavailable_through_the_real_door():
    store = _SigningFailureArtifactStore()
    location = _store_and_locate(store, b"hello")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 503, resp.text
    problem = resp.json()
    assert problem["code"] == "service_unavailable"
    assert problem["retryable"] is True


def test_permission_denied_signing_failure_maps_to_internal_error_through_the_real_door():
    store = _CredentialFailureArtifactStore()
    location = _store_and_locate(store, b"hello")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 500, resp.text
    problem = resp.json()
    assert problem["code"] == "internal_error"
    assert problem["retryable"] is False


def test_composed_surface_snapshot_signing_failure_attributes_the_snapshot_door_not_mint(
    caplog: pytest.LogCaptureFixture,
):
    """Mint classification runs first; snapshot failures use `SnapshotSigningFailed`.

    All door dependencies are injected for app lifespan; only the snapshot route is exercised.
    """
    store = FakeDatasetStore()
    # A pinned location avoids expiry against the real clock; the composed surface has no `now` override.
    location = _location(digest="sha256:deadbeef", tier="pinned")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=_SigningFailureArtifactStore(),
        graph=FakeMintGraphReader(),
        sdmx=SdmxClient(
            base_url="https://datacommons.one.org/core/api",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="")),
        ),
        revision_reader=SentinelRevisionReader(),
        # The composed app also requires the pin door's reference writer at construction.
        reference_writer=FakeArtifactReferenceWriter(),
    )

    with caplog.at_level(logging.ERROR):
        resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 503, resp.text
    problem = resp.json()
    assert problem["code"] == "service_unavailable"
    assert problem["retryable"] is True
    assert any("snapshot door" in r.getMessage() for r in caplog.records)
    assert not any("mint door" in r.getMessage() for r in caplog.records)
