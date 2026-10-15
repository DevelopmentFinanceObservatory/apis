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
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from dfo_platform.cloud_logging import JSON_FIELDS_ATTR

SERVERLESS_AUTHORIZATION_HEADER = "x-serverless-authorization"
AUTHORIZATION_HEADER = "authorization"

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


def log_request(*, method: str, path: str, status: int | None, headers: Mapping[str, str]) -> None:
    """Log one line for a finished request with the caller's claimed identity.

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
            }
        },
    )
