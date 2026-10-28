"""Create V4 signed GET URLs through IAM `signBlob`.

Cloud Run uses IAM-managed service-account credentials. ADC is discovered once
per process and refreshed only when its token is missing or expired; its
access token and account email make `google-cloud-storage` use IAM signing.
The account needs `roles/iam.serviceAccountTokenCreator` on itself.

This store caps URLs at one hour, below V4's seven-day maximum.

With `STORAGE_EMULATOR_HOST` set, which `google-cloud-storage` also reads to send every call
to a GCS emulator, and `DFO_LOCAL_STACK=1`, which only the local stack's `local.env` sets,
`sign_get_url` returns the object's unsigned media link, with the response-type and disposition
overrides in its query, and needs no credentials. `STORAGE_EMULATOR_HOST` set without
`DFO_LOCAL_STACK=1` refuses with `StorageEmulatorRefused` when a `LiveArtifactStore` is built,
at start-up, and on every signing (`emulator_host`). The link is
on `STORAGE_EMULATOR_LINK_BASE` when that is set, as the local stack sets it to the API's own
origin, which serves the emulator's media path, and on the emulator otherwise. The emulator
checks no signature and ignores the overrides.
"""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from typing import Protocol, cast
from urllib.parse import quote, urlencode

import google.auth
from google.api_core.exceptions import GoogleAPICallError
from google.auth.transport.requests import Request as _GoogleAuthRequest
from google.cloud import storage

from artifact_store.errors import ArtifactSigningError, StorageEmulatorRefused

# V4's maximum signed-URL lifetime.
V4_MAX_TTL = timedelta(days=7)

# Store-specific lifetime cap for signed URLs.
SIGNED_URL_TTL_CEILING = timedelta(hours=1)

# GCS truncates expiry to whole seconds. Shorter TTLs produce expired URLs.
SIGNED_URL_TTL_FLOOR = timedelta(seconds=1)

# Signed URL reuse window. A cached URL can be this much older than the response carrying
# it, so a caller signs for longer than this to hand out a URL with time left.
SIGNED_URL_REUSE_WINDOW = timedelta(minutes=1)

# The variable `google-cloud-storage` reads to point its client at a GCS emulator.
STORAGE_EMULATOR_ENV = "STORAGE_EMULATOR_HOST"

# The origin an emulator media link is served on, where it is not the emulator itself.
STORAGE_EMULATOR_LINK_BASE_ENV = "STORAGE_EMULATOR_LINK_BASE"

# Set to 1 by the local stack alone; the one setting under which the emulator's unsigned links go out.
LOCAL_STACK_ENV = "DFO_LOCAL_STACK"


def emulator_host() -> str | None:
    """The GCS emulator `STORAGE_EMULATOR_HOST` names, or `None` when it is unset.

    Raises:
        StorageEmulatorRefused: It is set and `DFO_LOCAL_STACK` is not `1`.
    """
    host = os.environ.get(STORAGE_EMULATOR_ENV)
    if not host:
        return None
    if os.environ.get(LOCAL_STACK_ENV) != "1":
        raise StorageEmulatorRefused
    return host


class _ServiceAccountCredentials(Protocol):
    """The ADC surface IAM signing reads. Cloud Run's compute-engine credentials carry the
    service-account email that `google.auth.credentials.Credentials` does not declare."""

    service_account_email: str
    token: str | None

    @property
    def valid(self) -> bool: ...

    def refresh(self, request: _GoogleAuthRequest) -> None: ...


class _ProcessCredentials:
    """This process's ADC, discovered on first use and refreshed only when invalid.

    `google.auth` reports a credential invalid when it has no token or the token is
    within its refresh threshold of expiry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._credentials: _ServiceAccountCredentials | None = None

    def current(self) -> _ServiceAccountCredentials:
        with self._lock:
            if self._credentials is None:
                credentials, _project = google.auth.default()
                self._credentials = cast(_ServiceAccountCredentials, credentials)
            if not self._credentials.valid:
                self._credentials.refresh(_GoogleAuthRequest())
            return self._credentials


_process_credentials = _ProcessCredentials()


def sign_get_url(
    *,
    client: storage.Client,
    bucket: str,
    object_name: str,
    digest: str,
    ttl: timedelta,
    response_type: str,
    response_disposition: str,
) -> str:
    """Sign one GET URL for `object_name` with the requested response type and disposition
    in its query. `digest` names the bytes in a signing error.

    On the local stack (`emulator_host`), returns the object's unsigned media link instead, on
    `STORAGE_EMULATOR_LINK_BASE` when that is set and on the emulator otherwise.

    Raises `ValueError` outside the store TTL bounds before credential or
    network work, and `StorageEmulatorRefused` for an emulator outside the local stack.
    Signing failures raise `ArtifactSigningError` with safe error details.
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
    if emulator := emulator_host():
        return _emulator_media_url(
            os.environ.get(STORAGE_EMULATOR_LINK_BASE_ENV) or emulator,
            bucket=bucket,
            object_name=object_name,
            response_type=response_type,
            response_disposition=response_disposition,
        )
    signing_failed = False
    error_class_name: str | None = None
    error_status_code: int | None = None
    try:
        credentials = _process_credentials.current()
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
            digest=digest,
            error_class_name=error_class_name,
            status_code=error_status_code,
        )
    raise AssertionError("unreachable: the try block above either returns or raises")


def emulator_media_path(*, bucket: str, object_name: str) -> str:
    """The emulator's media download path for `object_name` in `bucket`, query included."""
    return f"/download/storage/v1/b/{bucket}/o/{quote(object_name, safe='')}?alt=media"


def _emulator_media_url(
    base: str, *, bucket: str, object_name: str, response_type: str, response_disposition: str
) -> str:
    """The media download URL for `object_name` in `bucket` under `base`, with the response
    overrides a signed URL carries."""
    overrides = urlencode(
        {
            "response-content-type": response_type,
            "response-content-disposition": response_disposition,
        },
        quote_via=quote,
    )
    path = emulator_media_path(bucket=bucket, object_name=object_name)
    return f"{base.rstrip('/')}{path}&{overrides}"
