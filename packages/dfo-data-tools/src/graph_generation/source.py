"""Generation-token sources for Cloud Run and fixed local values.

Cloud Run reads use ADC bearer tokens. Inject a transport and credentials factory for tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

# The Cloud Run Admin API v2 root. `service` (the full resource name,
# `projects/<p>/locations/<r>/services/<s>`) is appended to it.
_ADMIN_API_BASE_URL = "https://run.googleapis.com/v2"

# Bound a hung Admin API read on the refresh thread.
_DEFAULT_TIMEOUT_SECONDS = 5.0


class GenerationSourceError(Exception):
    """Raised for HTTP, timeout, JSON, response-field, or ADC failures.

    `GraphGeneration` logs read failures and retains the last successful stamp."""


@dataclass(frozen=True, slots=True)
class GenerationReadResult:
    """A generation token and optional serving URLs for logs and host checks.

    Fixed sources return an empty URL tuple. URLs support logs and host checks. Revisions
    determine the token."""

    token: str
    urls: tuple[str, ...] = ()


class GenerationSource(Protocol):
    def read(self) -> GenerationReadResult:
        """Returns the current generation read, or raises
        `GenerationSourceError`.
        """
        ...


class FixedGenerationSource:
    """Read the fixed `DFO_GRAPH_GENERATION` token without an HTTP request."""

    def __init__(self, *, token: str) -> None:
        self._token = token

    def read(self) -> GenerationReadResult:
        return GenerationReadResult(token=self._token)


class Credentials(Protocol):
    """Credential fields and refresh method used by this source."""

    token: str | None
    valid: bool

    def refresh(self, request: object) -> None: ...


def _default_credentials_factory() -> Credentials:
    import google.auth

    credentials, _project = google.auth.default()
    return credentials


class CloudRunGenerationSource:
    """Read a Cloud Run v2 service and derive its serving-revision token.

    Join sorted revisions with `percent > 0` using `+`. If `trafficStatuses` is absent, use the
    final path segment of `latestReadyRevision`. Invalid responses raise `GenerationSourceError`."""

    def __init__(
        self,
        *,
        service: str,
        transport: httpx.BaseTransport | None = None,
        credentials_factory: Callable[[], Credentials] = _default_credentials_factory,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._service = service
        self._credentials_factory = credentials_factory
        # Create credentials on first read so ADC failures use the contained read path.
        # Refresh credentials whenever `valid` is false.
        self._credentials: Credentials | None = None
        self._client = httpx.Client(
            base_url=_ADMIN_API_BASE_URL, transport=transport, timeout=timeout
        )

    def read(self) -> GenerationReadResult:
        token = self._bearer_token()
        try:
            response = self._client.get(
                f"/{self._service}", headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            raise GenerationSourceError(
                f"Cloud Run Admin API request for {self._service!r} failed: {exc}"
            ) from exc
        if not (200 <= response.status_code < 300):
            raise GenerationSourceError(
                f"Cloud Run Admin API returned HTTP {response.status_code} for "
                f"{self._service!r}: {response.text!r}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GenerationSourceError(
                f"Cloud Run Admin API response for {self._service!r} is not valid JSON: {exc}"
            ) from exc
        return GenerationReadResult(
            token=self._token_from_payload(payload), urls=self._urls_from_payload(payload)
        )

    def _bearer_token(self) -> str:
        import google.auth.exceptions
        from google.auth.transport.requests import Request as _GoogleAuthRequest

        try:
            if self._credentials is None:
                self._credentials = self._credentials_factory()
            credentials = self._credentials
            if not credentials.valid:
                credentials.refresh(_GoogleAuthRequest())
        except google.auth.exceptions.GoogleAuthError as exc:
            raise GenerationSourceError(
                f"ADC credentials for {self._service!r} could not be obtained or refreshed: {exc}"
            ) from exc
        if not credentials.token:
            raise GenerationSourceError(
                f"ADC refresh produced no bearer token for {self._service!r}"
            )
        return credentials.token

    def _token_from_payload(self, payload: object) -> str:
        if not isinstance(payload, dict):
            raise GenerationSourceError(
                f"Cloud Run Admin API response for {self._service!r} is not a JSON object"
            )
        traffic_statuses = payload.get("trafficStatuses")
        if traffic_statuses is not None:
            return self._token_from_traffic_statuses(traffic_statuses)
        latest_ready_revision = payload.get("latestReadyRevision")
        if not isinstance(latest_ready_revision, str) or not latest_ready_revision:
            raise GenerationSourceError(
                f"Cloud Run Admin API response for {self._service!r} names neither "
                "trafficStatuses nor a valid latestReadyRevision"
            )
        return latest_ready_revision.rsplit("/", 1)[-1]

    def _token_from_traffic_statuses(self, traffic_statuses: object) -> str:
        """Return sorted positive-traffic revisions joined with `+`.

        Percentages are omitted, so weight changes across the same revision set keep the token
        unchanged. During a split, cached entries may derive from either revision."""
        if not isinstance(traffic_statuses, list):
            raise GenerationSourceError(
                f"Cloud Run Admin API response for {self._service!r} has a non-list trafficStatuses"
            )
        revisions: list[str] = []
        for entry in traffic_statuses:
            if not isinstance(entry, dict):
                raise GenerationSourceError(
                    f"Cloud Run Admin API response for {self._service!r} has a "
                    "non-object trafficStatuses entry"
                )
            percent = entry.get("percent", 0)
            if not isinstance(percent, int | float) or percent <= 0:
                continue
            revision = entry.get("revision")
            if not isinstance(revision, str) or not revision:
                raise GenerationSourceError(
                    f"Cloud Run Admin API response for {self._service!r} has a "
                    "trafficStatuses entry with percent > 0 and no revision"
                )
            revisions.append(revision)
        if not revisions:
            raise GenerationSourceError(
                f"Cloud Run Admin API response for {self._service!r} names no "
                "trafficStatuses entry with percent > 0"
            )
        return "+".join(sorted(revisions))

    def _urls_from_payload(self, payload: dict[str, object]) -> tuple[str, ...]:
        urls = payload.get("urls")
        if not isinstance(urls, list):
            return ()
        return tuple(url for url in urls if isinstance(url, str))
