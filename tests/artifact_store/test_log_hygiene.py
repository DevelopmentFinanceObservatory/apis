"""Ensure artifact-store code emits no logs and signing failures expose no secret material.

The app-level test includes traceback output and a driver exception containing a signed URL.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import google.auth
import google.cloud.storage._signing as _signing
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage

from artifact_store.inventory import ArtifactLocation
from artifact_store.store import LiveArtifactStore, Tier
from dfo_platform.app import Surface, create_app

_LEAKED_SIGNED_URL = (
    "https://storage.googleapis.com/dev-fin-observatory-datasets-working/"
    "sha256:deadbeef?X-Goog-Algorithm=GOOG4-RSA-SHA256"
    "&X-Goog-Credential=runtime-sa%40dev-fin-observatory.iam.gserviceaccount.com%2F20260101"
    "&X-Goog-Signature=deadbeefcafebabe1234deadbeefcafebabe1234"
)

_BUCKETS = {
    Tier.working: "dev-fin-observatory-datasets-working",
    Tier.pinned: "dev-fin-observatory-datasets-pinned",
}


class _NullInventory:
    """Return the configured location so the test reaches signing; `record` must not run."""

    def record(self, **kwargs: object) -> None:
        raise AssertionError("record() must not run for a signing-only failure")

    def get_location(self, digest: str, *, tier: Tier) -> ArtifactLocation:
        return ArtifactLocation(
            bucket=_BUCKETS[tier], object_name=digest, stored_at=datetime.now(UTC), renewal=0
        )


def test_package_source_never_references_logging():
    src_root = Path(__file__).resolve().parents[2] / "src" / "artifact_store"
    py_files = list(src_root.glob("*.py"))
    assert py_files, "expected to find artifact_store's own source files"
    for path in py_files:
        text = path.read_text()
        assert "logging" not in text, (
            f"{path} references logging -- artifact_store is meant to log nothing itself, "
            "leaving that to its callers"
        )


def test_inventory_write_failed_marker_is_stable_and_exported():
    # A caller surfacing ArtifactInventoryError keys its own reconciliation
    # event on this exact string -- pinning it here catches an accidental
    # rename that would silently break an alert policy keyed on it.
    import artifact_store
    from artifact_store.errors import ARTIFACT_INVENTORY_WRITE_FAILED_MARKER

    assert ARTIFACT_INVENTORY_WRITE_FAILED_MARKER == "artifact_inventory_write_failed"
    assert artifact_store.ARTIFACT_INVENTORY_WRITE_FAILED_MARKER is (
        ARTIFACT_INVENTORY_WRITE_FAILED_MARKER
    )


@pytest.fixture
def _fake_adc(monkeypatch):
    class _Credentials:
        service_account_email = "runtime-sa@dev-fin-observatory.iam.gserviceaccount.com"
        token = "fake-access-token"
        valid = True

        def refresh(self, request) -> None:
            pass

    monkeypatch.setattr(google.auth, "default", lambda: (_Credentials(), "dev-fin-observatory"))


def test_signing_failure_with_leaked_signature_material_never_reaches_the_rendered_log(
    _fake_adc, monkeypatch, caplog
):
    def _boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError(f"IAM signBlob call failed fetching {_LEAKED_SIGNED_URL}")

    monkeypatch.setattr(_signing, "_sign_message", _boom)

    gcs_client = storage.Client(credentials=AnonymousCredentials(), project="dev-fin-observatory")
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=_NullInventory(), client=gcs_client)

    router = APIRouter()

    @router.get("/api/artifact-test/boom")
    async def boom():
        # Leave the error unhandled so the platform exception handler logs it.
        store.signed_location_url(
            "sha256:deadbeef",
            tier=Tier.working,
            bucket=_BUCKETS[Tier.working],
            object_name="sha256:deadbeef",
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="data.csv"',
        )

    surface = Surface(
        name="artifact-test",
        prefix="/api/artifact-test",
        router=router,
        capability_version="v-test",
        codes={},
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="dfo_platform.app"):
        response = client.get("/api/artifact-test/boom")

    assert response.status_code == 500
    assert caplog.text  # something was logged -- the assertions below aren't vacuous
    for leaked in (
        "X-Goog-Signature",
        "X-Goog-Credential",
        "Signature=",
        "deadbeefcafebabe1234",
        _LEAKED_SIGNED_URL,
    ):
        assert leaked not in caplog.text
