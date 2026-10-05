"""Typed errors for the `dc_sdmx` client.

Envoy/ESPv2 transcodes mixer errors to JSON with a gRPC `code`. The client
uses the `message` and records the actual HTTP status as `upstream_status`.
Known message substrings select typed subclasses. Unmatched messages raise
`SdmxHttpError`. A body-byte cap takes precedence over HTTP error mapping.
"""

from __future__ import annotations

from typing import Literal


class SdmxClientError(Exception):
    """Base class for every error this client raises."""


class SdmxQueryError(SdmxClientError):
    """A request rejected before HTTP for an invalid query shape or filter value."""


class SdmxTransportError(SdmxClientError):
    """Wraps an `httpx.RequestError` (connection failure, timeout, etc.)."""


class SdmxHttpError(SdmxClientError):
    """An untyped non-2xx response from the mixer.

    `upstream_status` is the HTTP status. `message` is the transcoded gRPC
    message or raw body. Client-side `SdmxUnsupportedFilterError` instances
    have `upstream_status=None` because no response exists.
    """

    def __init__(self, message: str, *, upstream_status: int | None) -> None:
        status_text = (
            str(upstream_status)
            if upstream_status is not None
            else "none (rejected client-side before any HTTP call)"
        )
        super().__init__(f"SDMX request failed with status {status_text}: {message}")
        self.upstream_status = upstream_status
        self.message = message


class SdmxResponseError(SdmxClientError):
    """A 2xx response cannot be parsed as the expected SDMX wire format.

    This covers `/data` responses without SDMX-CSV (some deployments return
    HTTP 200 with `{}` until the Envoy/ESPv2 descriptor routes SDMX), malformed
    CSV headers or rows, and invalid `/availability` JSON shapes. Client calls
    set `upstream_status`. Direct parser calls set it to `None`.
    """

    def __init__(self, message: str, *, upstream_status: int | None = None) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status


class SdmxMissingFilterError(SdmxHttpError):
    """The mixer reports a missing required `variableMeasured` filter."""


class SdmxUnsupportedFilterError(SdmxHttpError):
    """The mixer rejects a filter or availability component, such as `/data`'s
    `OBS_VALUE` filter or `/availability`'s `TIME_PERIOD` component.
    """


class SdmxContainmentFilterError(SdmxHttpError):
    """A containment filter lacks either `containedInPlace+` or `typeOf`."""


class SdmxIncompatibleVariablesError(SdmxHttpError):
    """Variables in one request resolve to different entity-role sets."""


class SdmxRoleCeilingError(SdmxHttpError):
    """A variable exceeds the mixer's limit of 3 entity roles per observation."""


class SdmxUnexpectedResponseShapeError(SdmxHttpError):
    """An HTML WAF or proxy challenge detected by content type and failed JSON parsing."""


class SdmxCapExceededError(SdmxClientError):
    """A read exceeded a caller's row, byte, or availability-value cap.

    `cap_kind` is `"rows"`, `"bytes"`, or `"values"`. `cap` is the limit and
    `observed` is the count at abort. Row and byte reads stop at the first
    excess, so `observed` is a lower bound. Value counts are exact because
    the JSON body is parsed before that cap is checked.
    """

    def __init__(
        self,
        message: str,
        *,
        cap_kind: Literal["rows", "bytes", "values"],
        cap: int,
        observed: int,
    ) -> None:
        super().__init__(message)
        self.cap_kind = cap_kind
        self.cap = cap
        self.observed = observed
