"""Tests signing-failure classification at the download door's helper, door, and composed-app
layers."""

from __future__ import annotations

import logging

import httpx
import pytest

from artifact_store import ArtifactNotFoundError, ArtifactSigningError, Tier
from datasets_api.codes_snapshot import SnapshotSigningFailed
from datasets_api.doors.files import build_files_door, classify_files_exception, resolve_link
from dc_sdmx import SdmxClient
from spec_registry import LiveSpecRegistry

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps, fake_deps_cell
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
    location = _location(digest="sha256:never-stored", stored_at=_NOW)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    with pytest.raises(ArtifactNotFoundError):
        resolve_link(
            id="spec_a",
            content_version="cv1",
            digest=location.artifact_digest,
            deps=fake_deps(
                spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
                snapshot_reader=reader,
                artifact_store=FakeArtifactStore(),
                now=lambda: _NOW,
            ),
        )


@pytest.mark.parametrize(
    ("error_class_name", "status_code", "status", "code", "retryable"),
    [
        pytest.param(None, None, 503, "service_unavailable", True, id="unattributed"),
        pytest.param("ServiceUnavailable", 503, 503, "service_unavailable", True, id="503"),
        pytest.param("TooManyRequests", 429, 503, "service_unavailable", True, id="429"),
        pytest.param("RefreshError", None, 503, "service_unavailable", True, id="refresh"),
        pytest.param("RefreshError", 403, 500, "internal_error", False, id="refresh-403"),
        pytest.param("MalformedError", None, 500, "internal_error", False, id="malformed"),
    ],
)
def test_a_signing_failure_maps_by_its_cause(
    error_class_name: str | None, status_code: int | None, status: int, code: str, retryable: bool
):
    exc = SnapshotSigningFailed(
        bucket="fake-bucket",
        object_name="sha256:deadbeef",
        digest="sha256:deadbeef",
        error_class_name=error_class_name,
        status_code=status_code,
    )

    spec = classify_files_exception(exc)

    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (status, code, retryable)


class _SigningFailureArtifactStore(FakeArtifactStore):
    """`put()` succeeds normally; signing always raises `ArtifactSigningError` with the given
    cause, standing in for an IAM signBlob outage or a broken IAM policy."""

    def __init__(self, *, error_class_name: str | None = None, status_code: int | None = None):
        super().__init__()
        self._cause = (error_class_name, status_code)

    def signed_location_url(
        self,
        digest: str,
        *,
        tier: Tier,
        bucket: str,
        object_name: str,
        ttl,
        response_type: str,
        response_disposition: str,
    ):
        raise ArtifactSigningError(
            bucket=bucket,
            object_name=object_name,
            digest=digest,
            error_class_name=self._cause[0],
            status_code=self._cause[1],
        )


def test_resolve_link_wraps_a_signing_failure_without_resurrecting_context():
    """The wrapper leaves `__context__` and `__cause__` unset for Starlette middleware."""
    store = _SigningFailureArtifactStore()
    location = _store_and_locate(store, b"hello")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    with pytest.raises(SnapshotSigningFailed) as excinfo:
        resolve_link(
            id="spec_a",
            content_version="cv1",
            digest=location.artifact_digest,
            deps=fake_deps(
                spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
                snapshot_reader=reader,
                artifact_store=store,
                now=lambda: _NOW,
            ),
        )

    assert excinfo.value.__context__ is None
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize(
    ("store", "status", "code", "retryable"),
    [
        pytest.param(_SigningFailureArtifactStore(), 503, "service_unavailable", True, id="outage"),
        pytest.param(
            _SigningFailureArtifactStore(error_class_name="RefreshError", status_code=403),
            500,
            "internal_error",
            False,
            id="permission-denied",
        ),
    ],
)
def test_a_signing_failure_maps_through_the_real_door(
    store: FakeArtifactStore, status: int, code: str, retryable: bool
):
    location = _store_and_locate(store, b"hello")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_files_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )

    resp = make_client_for_door(door).get(
        f"/api/datasets/spec_a/cv1/files/{location.artifact_digest}", follow_redirects=False
    )

    assert resp.status_code == status, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == (code, retryable)


def test_composed_surface_download_signing_failure_attributes_the_download_door(
    caplog: pytest.LogCaptureFixture,
):
    """The composed surface classifies a download's signing failure through the download door.

    All door dependencies are injected for app lifespan; only the download route is exercised.
    """
    # A pinned location avoids expiry against the real clock; the composed surface has no `now`
    # override.
    location = _location(digest="sha256:deadbeef", tier="pinned")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=_SigningFailureArtifactStore(),
        graph=FakeMintGraphReader(),
        sdmx=SdmxClient(
            base_url="https://datacommons.one.org/core/api",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="")),
        ),
        # The composed app also requires the pin door's reference writer at construction.
        reference_writer=FakeArtifactReferenceWriter(),
    )

    with caplog.at_level(logging.ERROR):
        resp = client.get("/api/datasets/spec_a/cv1/files/sha256:deadbeef", follow_redirects=False)

    assert resp.status_code == 503, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == ("service_unavailable", True)
    assert any("download door" in r.getMessage() for r in caplog.records)
