"""Pin the real GCS V4 URL construction while stubbing ADC and IAM `signBlob` calls."""

from __future__ import annotations

import base64
from datetime import timedelta
from urllib.parse import parse_qs, unquote, urlsplit

import google.auth
import google.cloud.storage._signing as _signing
import pytest
from google.api_core.exceptions import PermissionDenied
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage

from artifact_store.errors import ArtifactSigningError
from artifact_store.signing import (
    SIGNED_URL_TTL_CEILING,
    SIGNED_URL_TTL_FLOOR,
    STORAGE_EMULATOR_ENV,
    sign_get_url,
)

_SA_EMAIL = "runtime-sa@dev-fin-observatory.iam.gserviceaccount.com"


class _FakeCredentials:
    """Provide the token and service-account email used by IAM signing.

    The token is absent until the first refresh. Setting `expired` makes the credential invalid
    until the next refresh, as `google.auth.credentials.Credentials.valid` reports.
    """

    def __init__(self) -> None:
        self.service_account_email = _SA_EMAIL
        self.token: str | None = None
        self.expired = False
        self.refresh_count = 0

    @property
    def refreshed(self) -> bool:
        return self.refresh_count > 0

    @property
    def valid(self) -> bool:
        return self.token is not None and not self.expired

    def refresh(self, request) -> None:
        self.refresh_count += 1
        self.token = f"fake-access-token-{self.refresh_count}"
        self.expired = False


@pytest.fixture
def gcs_client() -> storage.Client:
    # Avoid ADC discovery at client construction; signing receives fake credentials below.
    return storage.Client(credentials=AnonymousCredentials(), project="dev-fin-observatory")


@pytest.fixture
def stub_signing(monkeypatch):
    """Stub credential discovery and IAM signing, leaving GCS URL construction real."""
    fake_credentials = _FakeCredentials()
    monkeypatch.setattr(google.auth, "default", lambda: (fake_credentials, "dev-fin-observatory"))
    monkeypatch.setattr(
        _signing,
        "_sign_message",
        lambda *args, **kwargs: base64.b64encode(b"fake-signature-bytes").decode("ascii"),
    )
    return fake_credentials


def test_signed_url_is_v4_single_object_and_carries_the_expected_params(gcs_client, stub_signing):
    url = sign_get_url(
        client=gcs_client,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:deadbeef",
        digest="sha256:deadbeef",
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="deadbeef.csv"',
    )

    parsed = urlsplit(url)
    assert unquote(parsed.path) == "/dev-fin-observatory-datasets-working/sha256:deadbeef"
    params = parse_qs(parsed.query)
    assert params["X-Goog-Algorithm"] == ["GOOG4-RSA-SHA256"]
    assert params["X-Goog-Credential"][0].startswith(f"{_SA_EMAIL}/")
    assert params["X-Goog-Expires"] == ["300"]
    assert "X-Goog-Signature" in params
    assert stub_signing.refreshed is True


def test_signed_url_carries_response_type_and_disposition_in_the_signed_query_string(
    gcs_client, stub_signing
):
    url = sign_get_url(
        client=gcs_client,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:deadbeef",
        digest="sha256:deadbeef",
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="deadbeef.csv"',
    )

    params = parse_qs(urlsplit(url).query)
    assert params["response-content-type"] == ["text/csv"]
    assert params["response-content-disposition"] == ['attachment; filename="deadbeef.csv"']


def test_ttl_at_the_ceiling_is_accepted(gcs_client, stub_signing):
    url = sign_get_url(
        client=gcs_client,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:atceiling",
        digest="sha256:atceiling",
        ttl=SIGNED_URL_TTL_CEILING,
        response_type="text/csv",
        response_disposition='attachment; filename="atceiling.csv"',
    )
    params = parse_qs(urlsplit(url).query)
    assert params["X-Goog-Expires"] == [str(int(SIGNED_URL_TTL_CEILING.total_seconds()))]


def test_emulator_host_returns_the_media_link_without_credential_work(gcs_client, monkeypatch):
    def _boom():
        raise AssertionError("google.auth.default() must not run against the emulator")

    monkeypatch.setattr(google.auth, "default", _boom)
    monkeypatch.setenv(STORAGE_EMULATOR_ENV, "http://127.0.0.1:54443/")

    url = sign_get_url(
        client=gcs_client,
        bucket="dfo-local-working",
        object_name="sha256:deadbeef",
        digest="sha256:deadbeef",
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="deadbeef.csv"',
    )

    assert url == (
        "http://127.0.0.1:54443/download/storage/v1/b/dfo-local-working/o/"
        "sha256%3Adeadbeef?alt=media"
    )


def test_ttl_past_the_ceiling_is_refused_before_any_credential_work(gcs_client, monkeypatch):
    def _boom():
        raise AssertionError("google.auth.default() must not run for an out-of-bounds ttl")

    monkeypatch.setattr(google.auth, "default", _boom)

    with pytest.raises(ValueError, match="ceiling"):
        sign_get_url(
            client=gcs_client,
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:toolong",
            digest="sha256:toolong",
            ttl=SIGNED_URL_TTL_CEILING + timedelta(seconds=1),
            response_type="text/csv",
            response_disposition='attachment; filename="toolong.csv"',
        )


@pytest.mark.parametrize("ttl", [timedelta(0), timedelta(seconds=-1)])
def test_non_positive_ttl_is_refused(gcs_client, ttl):
    with pytest.raises(ValueError, match="positive"):
        sign_get_url(
            client=gcs_client,
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:zero",
            digest="sha256:zero",
            ttl=ttl,
            response_type="text/csv",
            response_disposition='attachment; filename="zero.csv"',
        )


def test_ttl_below_one_second_is_refused_before_any_credential_work(gcs_client, monkeypatch):
    # A ttl that would serialize to X-Goog-Expires=0 (google-cloud-storage
    # truncates with int(ttl.total_seconds())) must never reach signing.
    def _boom():
        raise AssertionError("google.auth.default() must not run for a sub-floor ttl")

    monkeypatch.setattr(google.auth, "default", _boom)

    with pytest.raises(ValueError, match="floor"):
        sign_get_url(
            client=gcs_client,
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:subsecond",
            digest="sha256:subsecond",
            ttl=timedelta(milliseconds=999),
            response_type="text/csv",
            response_disposition='attachment; filename="subsecond.csv"',
        )


def test_ttl_at_exactly_the_floor_is_accepted(gcs_client, stub_signing):
    url = sign_get_url(
        client=gcs_client,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:atfloor",
        digest="sha256:atfloor",
        ttl=SIGNED_URL_TTL_FLOOR,
        response_type="text/csv",
        response_disposition='attachment; filename="atfloor.csv"',
    )
    params = parse_qs(urlsplit(url).query)
    assert params["X-Goog-Expires"] == [str(int(SIGNED_URL_TTL_FLOOR.total_seconds()))]


def test_signing_failure_is_wrapped_without_chaining_the_driver_exception(
    gcs_client, stub_signing, monkeypatch
):
    # The driver message contains a signature; the typed error must omit it and leave context unset.
    def _boom(*args, **kwargs):
        raise RuntimeError("IAM signBlob call failed: ...&X-Goog-Signature=deadbeefcafebabe1234")

    monkeypatch.setattr(_signing, "_sign_message", _boom)

    with pytest.raises(ArtifactSigningError) as excinfo:
        sign_get_url(
            client=gcs_client,
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:deadbeef",
            digest="sha256:deadbeef",
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="deadbeef.csv"',
        )

    assert excinfo.value.bucket == "dev-fin-observatory-datasets-working"
    assert excinfo.value.object_name == "sha256:deadbeef"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert "X-Goog-Signature" not in str(excinfo.value)


def test_signing_failure_carries_the_failing_class_name_and_status_code(
    gcs_client, stub_signing, monkeypatch
):
    # Preserve the safe error class and HTTP status, while withholding the driver message.
    def _boom(*args, **kwargs):
        raise PermissionDenied("caller does not have permission to sign this exact url")

    monkeypatch.setattr(_signing, "_sign_message", _boom)

    with pytest.raises(ArtifactSigningError) as excinfo:
        sign_get_url(
            client=gcs_client,
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:deadbeef",
            digest="sha256:deadbeef",
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="deadbeef.csv"',
        )

    assert excinfo.value.error_class_name == "PermissionDenied"
    assert excinfo.value.status_code == 403
    assert "caller does not have permission" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_signing_failure_with_no_status_code_still_carries_the_class_name(
    gcs_client, stub_signing, monkeypatch
):
    def _boom(*args, **kwargs):
        raise RuntimeError("a plain programming defect, no status code at all")

    monkeypatch.setattr(_signing, "_sign_message", _boom)

    with pytest.raises(ArtifactSigningError) as excinfo:
        sign_get_url(
            client=gcs_client,
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:deadbeef",
            digest="sha256:deadbeef",
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="deadbeef.csv"',
        )

    assert excinfo.value.error_class_name == "RuntimeError"
    assert excinfo.value.status_code is None
    assert "programming defect" not in str(excinfo.value)


def test_signings_in_one_process_discover_adc_once_and_refresh_only_on_expiry(
    gcs_client, monkeypatch
):
    monkeypatch.setattr(
        _signing,
        "_sign_message",
        lambda *args, **kwargs: base64.b64encode(b"fake-signature-bytes").decode("ascii"),
    )
    credentials = _FakeCredentials()
    discoveries: list[None] = []

    def _default():
        discoveries.append(None)
        return credentials, "dev-fin-observatory"

    monkeypatch.setattr(google.auth, "default", _default)

    def _sign() -> None:
        sign_get_url(
            client=gcs_client,
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:deadbeef",
            digest="sha256:deadbeef",
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="deadbeef.csv"',
        )

    for _ in range(5):
        _sign()
    assert (len(discoveries), credentials.refresh_count) == (1, 1)

    credentials.expired = True
    for _ in range(3):
        _sign()
    assert (len(discoveries), credentials.refresh_count) == (1, 2)
