"""The Data Commons service's serving revision, read through the Cloud Run Admin API v2.

`CloudRunAdmin.serving` reads the service and names the revisions that take traffic;
`CloudRunAdmin.created_at` reads one revision's creation time. The builder compares that time
with the import's completion, because the mixer reads Spanner as of the last ingestion it saw
when it started. `graph_generation.source` derives its generation token from the same read.

Reads use ADC bearer tokens unless a credentials factory is given. Inject a transport and a
credentials factory for tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import httpx

# The Cloud Run Admin API v2 root. `service` (the full resource name,
# `projects/<p>/locations/<r>/services/<s>`) is appended to it.
_ADMIN_API_BASE_URL = "https://run.googleapis.com/v2"

# Bound a hung Admin API read.
_DEFAULT_TIMEOUT_SECONDS = 5.0


class ServingRevisionError(Exception):
    """Raised for HTTP, timeout, JSON, response-field, or ADC failures."""


@dataclass(frozen=True, slots=True)
class ServingRevision:
    """The revisions taking traffic, sorted, and the service's URLs. `token` joins the revisions
    with `+`."""

    revisions: tuple[str, ...]
    urls: tuple[str, ...] = ()

    @property
    def token(self) -> str:
        return "+".join(self.revisions)


class Credentials(Protocol):
    """Credential fields and refresh method used by this reader."""

    token: str | None
    valid: bool

    def refresh(self, request: object) -> None: ...


def _default_credentials_factory() -> Credentials:
    import google.auth

    credentials, _project = google.auth.default()
    return credentials


class CloudRunAdmin:
    """Reads one Cloud Run v2 service and its revisions.

    The serving revisions are those with `percent > 0` in `trafficStatuses`, or the final path
    segment of `latestReadyRevision` when `trafficStatuses` is absent. Invalid responses raise
    `ServingRevisionError`."""

    def __init__(
        self,
        *,
        service: str,
        transport: httpx.BaseTransport | None = None,
        credentials_factory: Callable[[], Credentials] | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._service = service
        self._credentials_factory = credentials_factory or _default_credentials_factory
        # Create credentials on first read so ADC failures use the contained read path.
        # Refresh credentials whenever `valid` is false.
        self._credentials: Credentials | None = None
        self._client = httpx.Client(
            base_url=_ADMIN_API_BASE_URL, transport=transport, timeout=timeout
        )

    def serving(self) -> ServingRevision:
        payload = self._get(f"/{self._service}")
        return ServingRevision(
            revisions=self._revisions_from_payload(payload), urls=self._urls_from_payload(payload)
        )

    def created_at(self, revision: str) -> datetime:
        """The revision's `createTime`, timezone-aware."""
        payload = self._get(f"/{self._service}/revisions/{revision}")
        created = payload.get("createTime")
        try:
            if not isinstance(created, str):
                raise ValueError(f"createTime is {created!r}")
            parsed = datetime.fromisoformat(created)
            if parsed.tzinfo is None:
                raise ValueError(f"createTime {created!r} names no timezone")
        except ValueError as exc:
            raise ServingRevisionError(
                f"Cloud Run Admin API revision {revision!r} has no valid createTime: {exc}"
            ) from exc
        return parsed

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str) -> dict[str, object]:
        token = self._bearer_token()
        try:
            response = self._client.get(path, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            raise ServingRevisionError(
                f"Cloud Run Admin API request for {path!r} failed: {exc}"
            ) from exc
        if not (200 <= response.status_code < 300):
            raise ServingRevisionError(
                f"Cloud Run Admin API returned HTTP {response.status_code} for "
                f"{path!r}: {response.text!r}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServingRevisionError(
                f"Cloud Run Admin API response for {path!r} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ServingRevisionError(
                f"Cloud Run Admin API response for {path!r} is not a JSON object"
            )
        return payload

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
            raise ServingRevisionError(
                f"ADC credentials for {self._service!r} could not be obtained or refreshed: {exc}"
            ) from exc
        if not credentials.token:
            raise ServingRevisionError(
                f"ADC refresh produced no bearer token for {self._service!r}"
            )
        return credentials.token

    def _revisions_from_payload(self, payload: dict[str, object]) -> tuple[str, ...]:
        traffic_statuses = payload.get("trafficStatuses")
        if traffic_statuses is not None:
            return self._revisions_from_traffic_statuses(traffic_statuses)
        latest_ready_revision = payload.get("latestReadyRevision")
        if not isinstance(latest_ready_revision, str) or not latest_ready_revision:
            raise ServingRevisionError(
                f"Cloud Run Admin API response for {self._service!r} names neither "
                "trafficStatuses nor a valid latestReadyRevision"
            )
        return (latest_ready_revision.rsplit("/", 1)[-1],)

    def _revisions_from_traffic_statuses(self, traffic_statuses: object) -> tuple[str, ...]:
        """The positive-traffic revisions, sorted.

        Percentages are left out, so weight changes across the same revision set keep the
        token unchanged."""
        if not isinstance(traffic_statuses, list):
            raise ServingRevisionError(
                f"Cloud Run Admin API response for {self._service!r} has a non-list trafficStatuses"
            )
        revisions: list[str] = []
        for entry in traffic_statuses:
            if not isinstance(entry, dict):
                raise ServingRevisionError(
                    f"Cloud Run Admin API response for {self._service!r} has a "
                    "non-object trafficStatuses entry"
                )
            percent = entry.get("percent", 0)
            if not isinstance(percent, int | float) or percent <= 0:
                continue
            revision = entry.get("revision")
            if not isinstance(revision, str) or not revision:
                raise ServingRevisionError(
                    f"Cloud Run Admin API response for {self._service!r} has a "
                    "trafficStatuses entry with percent > 0 and no revision"
                )
            revisions.append(revision)
        if not revisions:
            raise ServingRevisionError(
                f"Cloud Run Admin API response for {self._service!r} names no "
                "trafficStatuses entry with percent > 0"
            )
        return tuple(sorted(revisions))

    def _urls_from_payload(self, payload: dict[str, object]) -> tuple[str, ...]:
        urls = payload.get("urls")
        if not isinstance(urls, list):
            return ()
        return tuple(url for url in urls if isinstance(url, str))


__all__ = ["CloudRunAdmin", "Credentials", "ServingRevision", "ServingRevisionError"]
