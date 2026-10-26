"""Who called: the Cloud Run identity token's claims, logged once per request.

Cloud Run's IAM layer checks a caller's Google ID token and strips its signature
before the container sees the header, so the payload read here is unverified. It
names the caller for logs and says nothing about authorisation.

`X-Serverless-Authorization` carries the token when a caller sends one, as Cloud
Run checks it ahead of `Authorization`. `Authorization` is the fallback. Neither
header's value, signature included, reaches a log line. Only the `email` and `aud`
claims and a presence flag do.

`log_request` writes one `dfo_platform.request` line per request, through the
`json_fields` record attribute that `dfo_platform.cloud_logging` merges into the
entry. While the service is open to every caller, filtering these lines for a
missing token or an unexpected `email` lists the callers an invoker-only service
would turn away.

The same line carries the SDK's `X-DFO-Client` header, which names the client
as `<name>/<version> (<runtime>)`, for example `dfo-python/0.1.0 (CPython 3.14)`.
The caller writes the value, so it is logged with non-printable characters removed
and clipped, as `client_header`, and split into `client_name`, `client_version` and
`client_runtime` when it has that shape. A request without the header logs none of
these fields.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass

from dfo_platform.cloud_logging import JSON_FIELDS_ATTR

SERVERLESS_AUTHORIZATION_HEADER = "x-serverless-authorization"
AUTHORIZATION_HEADER = "authorization"
CLIENT_HEADER = "X-DFO-Client"
"""Request header naming the SDK client, which browsers may send cross-origin."""

_CLIENT_FORMAT = re.compile(
    r"(?P<name>[^/\s()]+)/(?P<version>[^\s()]+)(?: \((?P<runtime>[^()]+)\))?"
)

_MAX_FIELD_LENGTH = 256
"""Longest caller-controlled string logged. A junk token costs one short line."""

_MAX_CREDENTIAL_LENGTH = 4096
"""Longest credential decoded. A Google ID token is about 1 KB, so a longer value is junk,
and decoding it would let a crafted payload exhaust the JSON parser's recursion limit."""

_log = logging.getLogger("dfo_platform.request")


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    """The unverified identity a request's token claims.

    `email` and `aud` are `None` when no token arrived, the token is not a JWT, or the
    claim is absent or not a string.
    """

    token_present: bool
    email: str | None = None
    aud: str | None = None


def _clip(value: str) -> str:
    return value[:_MAX_FIELD_LENGTH]


def _clean(value: str) -> str:
    """Drop non-printable characters (C0 and C1 controls, tabs, the soft hyphen) and
    surrounding space, then clip."""
    return _clip("".join(ch for ch in value if ch.isprintable()).strip())


def _claim(claims: Mapping[str, object], name: str) -> str | None:
    value = claims.get(name)
    return _clip(value) if isinstance(value, str) else None


def _unverified_claims(credential: str) -> Mapping[str, object]:
    """Decode a JWT's payload segment, or return no claims when it cannot be read."""
    if len(credential) > _MAX_CREDENTIAL_LENGTH:
        return {}
    segments = credential.split(".")
    if len(segments) < 2:
        return {}
    payload = segments[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (ValueError, RecursionError):
        # ValueError covers bad base64, bad UTF-8 and bad JSON. RecursionError covers
        # a payload nested deeper than the parser allows.
        return {}
    return claims if isinstance(claims, dict) else {}


def caller_identity_from_headers(headers: Mapping[str, str]) -> CallerIdentity:
    """Read the caller's claimed identity from the request headers. It never raises.

    `headers` must look up names case-insensitively, as Starlette's `Headers` does.
    """
    value = (
        headers.get(SERVERLESS_AUTHORIZATION_HEADER, "").strip()
        or headers.get(AUTHORIZATION_HEADER, "").strip()
    )
    if not value:
        return CallerIdentity(token_present=False)
    scheme, _, credential = value.partition(" ")
    claims = _unverified_claims(credential.strip() if scheme.lower() == "bearer" else "")
    return CallerIdentity(
        token_present=True, email=_claim(claims, "email"), aud=_claim(claims, "aud")
    )


def client_fields_from_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return the `X-DFO-Client` log fields, or none when the header is absent or blank.

    `client_header` is the cleaned value. The parsed fields appear only when the value
    has the `<name>/<version> (<runtime>)` shape, `client_runtime` only when it names one.
    """
    raw = headers.get(CLIENT_HEADER)
    value = _clean(raw) if raw is not None else ""
    if not value:
        return {}
    fields = {"client_header": value}
    if match := _CLIENT_FORMAT.fullmatch(value):
        fields["client_name"] = match["name"]
        fields["client_version"] = match["version"]
        if match["runtime"] is not None:
            fields["client_runtime"] = match["runtime"]
    return fields


def log_request(*, method: str, path: str, status: int | None, headers: Mapping[str, str]) -> None:
    """Log one line for a finished request with the caller's claimed identity and the
    SDK client its `X-DFO-Client` header names.

    `status` is `None` when the request ended without a response.
    """
    caller = caller_identity_from_headers(headers)
    user_agent = headers.get("user-agent")
    _log.info(
        "request",
        extra={
            JSON_FIELDS_ATTR: {
                "http_method": method,
                "http_path": path,
                "http_status": status,
                "user_agent": _clip(user_agent) if user_agent is not None else None,
                "caller_token_present": caller.token_present,
                "caller_email": caller.email,
                "caller_aud": caller.aud,
                **client_fields_from_headers(headers),
            }
        },
    )
