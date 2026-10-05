"""Tests artifact headers and signing failures, which become `RenderedArtifactSigningFailed`
without exception context.
"""

from __future__ import annotations

import pytest

from artifact_store import ArtifactSigningError, Tier
from datasets_api.codes_materialise import RenderedArtifactSigningFailed
from datasets_api.render import _store_artifact

from ._artifact_doubles import FakeArtifactStore
from ._render_inputs import PREPARED_AT

_SOURCES = ["OECD Creditor Reporting System (CRS)"]


class _SigningFailureArtifactStore(FakeArtifactStore):
    """`put()` stores the object. `signed_get_url()` raises `ArtifactSigningError`."""

    def signed_get_url(
        self, digest: str, *, tier: Tier, ttl, response_type: str, response_disposition: str
    ):
        raise ArtifactSigningError(bucket="fake-bucket", object_name=digest, digest=digest)


def test_store_artifact_signs_the_real_media_type_and_a_filename_naming_the_sources():
    artifact_store = FakeArtifactStore()

    stored = _store_artifact(
        data=b"a,b\n1,2\n",
        media_type="text/csv",
        rendering={"format": "csv"},
        subject_id="spec_exemplar",
        content_version="cv_exemplar",
        source_names=_SOURCES,
        prepared_at=PREPARED_AT,
        artifact_store=artifact_store,
    )

    response_type, response_disposition = artifact_store.signed_headers[
        (Tier.working, stored.artifact_digest)
    ]
    assert response_type == "text/csv"
    assert response_disposition == (
        'attachment; filename="dfo_oecd-creditor-reporting-system_2026-10-04_exempl.csv"'
    )


def test_store_artifact_signs_the_bundle_zip_media_type_and_filename():
    artifact_store = FakeArtifactStore()

    stored = _store_artifact(
        data=b"PK\x03\x04fake zip bytes",
        media_type="application/zip",
        rendering={"format": "csv", "bundle": True},
        subject_id="spec_exemplar",
        content_version="cv_exemplar",
        source_names=_SOURCES,
        prepared_at=PREPARED_AT,
        artifact_store=artifact_store,
    )

    response_type, response_disposition = artifact_store.signed_headers[
        (Tier.working, stored.artifact_digest)
    ]
    assert response_type == "application/zip"
    assert response_disposition == (
        'attachment; filename="dfo_oecd-creditor-reporting-system_2026-10-04_csv_exempl.zip"'
    )


def test_store_artifact_wraps_a_signing_failure_without_resurrecting_context():
    """`_store_artifact` re-raises a caught `ArtifactSigningError` as
    `RenderedArtifactSigningFailed` only after its own except clause has
    exited -- proving `__context__`/`__cause__` stay unset on the wrapper,
    so Starlette's own `BaseHTTPMiddleware` re-raise (`raise app_exc from
    app_exc.__cause__ or app_exc.__context__`) has nothing to resurrect.
    """
    with pytest.raises(RenderedArtifactSigningFailed) as excinfo:
        _store_artifact(
            data=b"hello",
            media_type="text/csv",
            rendering={"format": "csv"},
            subject_id="spec_exemplar",
            content_version="cv_exemplar",
            source_names=_SOURCES,
            prepared_at=PREPARED_AT,
            artifact_store=_SigningFailureArtifactStore(),
        )

    assert excinfo.value.__context__ is None
    assert excinfo.value.__cause__ is None
