"""Create V4 signed GET URLs through IAM `signBlob`.

Cloud Run uses IAM-managed service-account credentials. Refresh ADC and pass
its access token with the account email so `google-cloud-storage` uses IAM
signing. The account needs `roles/iam.serviceAccountTokenCreator` on itself.

This store caps URLs at one hour, below V4's seven-day maximum.
"""

from __future__ import annotations

from datetime import timedelta

import google.auth
from google.api_core.exceptions import GoogleAPICallError
from google.auth.transport.requests import Request as _GoogleAuthRequest
from google.cloud import storage

from artifact_store.errors import ArtifactSigningError

# V4's maximum signed-URL lifetime.
V4_MAX_TTL = timedelta(days=7)

# Store-specific lifetime cap for URLs included in manifests.
SIGNED_URL_TTL_CEILING = timedelta(hours=1)

# GCS truncates expiry to whole seconds. Shorter TTLs produce expired URLs.
SIGNED_URL_TTL_FLOOR = timedelta(seconds=1)

# Signed URL reuse window. Callers subtract this from advertised expiry because
# a cached URL can be this much older than the response carrying it.
SIGNED_URL_REUSE_WINDOW = timedelta(minutes=5)


def sign_get_url(
    *,
    client: storage.Client,
    bucket: str,
    object_name: str,
    ttl: timedelta,
    response_type: str,
    response_disposition: str,
) -> str:
    """Sign one GET URL with the requested response type and disposition in its query.

    Raises `ValueError` outside the store TTL bounds before credential or
    network work. Signing failures raise `ArtifactSigningError` with safe error details.
    """
    if ttl <= timedelta(0):
        raise ValueError(f"ttl must be positive, got {ttl!r}")
    if ttl < SIGNED_URL_TTL_FLOOR:
        raise ValueError(
            f"ttl {ttl!r} is below this store's {SIGNED_URL_TTL_FLOOR!r} floor -- "
            f"V4 signing serializes ttl with whole-second resolution, so anything "
            f"shorter would sign an already-expired URL"
        )
    if ttl > SIGNED_URL_TTL_CEILING:
        raise ValueError(
            f"ttl {ttl!r} exceeds this store's {SIGNED_URL_TTL_CEILING!r} ceiling "
            f"(V4 signed URLs cap at {V4_MAX_TTL!r} regardless)"
        )
    signing_failed = False
    error_class_name: str | None = None
    error_status_code: int | None = None
    try:
        credentials, _project = google.auth.default()
        credentials.refresh(_GoogleAuthRequest())
        blob = client.bucket(bucket).blob(object_name)
        return blob.generate_signed_url(
            version="v4",
            expiration=ttl,
            method="GET",
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
            response_type=response_type,
            response_disposition=response_disposition,
        )
    except Exception as exc:
        # Capture safe fields, then raise after this block to keep driver context out.
        error_class_name = type(exc).__qualname__
        error_status_code = exc.code if isinstance(exc, GoogleAPICallError) else None
        signing_failed = True
    if signing_failed:
        raise ArtifactSigningError(
            bucket=bucket,
            object_name=object_name,
            digest=object_name,
            error_class_name=error_class_name,
            status_code=error_status_code,
        )
    raise AssertionError("unreachable: the try block above either returns or raises")
