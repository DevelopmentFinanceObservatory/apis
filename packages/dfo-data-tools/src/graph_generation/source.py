"""Generation-token sources for Cloud Run and fixed local values.

Cloud Run reads use ADC bearer tokens. Inject a transport and credentials factory for tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from graph_release.serving_revision import CloudRunAdmin, Credentials, ServingRevisionError

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


class CloudRunGenerationSource:
    """Read a Cloud Run v2 service and derive its serving-revision token through
    `graph_release.serving_revision.CloudRunAdmin`.

    Join sorted revisions with `percent > 0` using `+`. If `trafficStatuses` is absent, use the
    final path segment of `latestReadyRevision`. Invalid responses raise `GenerationSourceError`."""

    def __init__(
        self,
        *,
        service: str,
        transport: httpx.BaseTransport | None = None,
        credentials_factory: Callable[[], Credentials] | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._admin = CloudRunAdmin(
            service=service,
            transport=transport,
            credentials_factory=credentials_factory,
            timeout=timeout,
        )

    def read(self) -> GenerationReadResult:
        try:
            serving = self._admin.serving()
        except ServingRevisionError as exc:
            raise GenerationSourceError(str(exc)) from exc
        return GenerationReadResult(token=serving.token, urls=serving.urls)
