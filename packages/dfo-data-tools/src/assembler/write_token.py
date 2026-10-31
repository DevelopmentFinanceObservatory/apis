"""Write tokens grant revise. A `cmp_` or `cps_` id grants only read.

A create mints a `cmw_` token for the new component, and `composition_id="new"` mints a
`cpw_` token for the new composition. A share's create mints a `shw_` token, which grants
replacing and deleting that share (`presentation_api.doors.shares`). Each token is its prefix plus
`secrets.token_urlsafe(16)`, 128 random bits. The store keeps only the unsalted SHA-256
digest, and the reply returns the token once.

A client may instead send its own `cmw_` token with a component create, or its own `shw_` token
with a share create, in the same format (`require_client_write_token`). The token then keys the
create: `create_request_digest` and `request_digest` hash the request it came with, and a
retried create with the same token and the same request returns what the first create made.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Mapping

import rfc8785

__all__ = [
    "COMPONENT_WRITE_TOKEN_PREFIX",
    "COMPOSITION_WRITE_TOKEN_PREFIX",
    "SHARE_WRITE_TOKEN_PREFIX",
    "CreateRequestInvalid",
    "create_request_digest",
    "mint_write_token",
    "request_digest",
    "require_client_write_token",
    "write_token_digest",
    "write_token_matches",
]

COMPONENT_WRITE_TOKEN_PREFIX = "cmw_"
COMPOSITION_WRITE_TOKEN_PREFIX = "cpw_"
SHARE_WRITE_TOKEN_PREFIX = "shw_"

# `secrets.token_urlsafe(16)` returns 22 characters from the url-safe base64 alphabet.
_TOKEN_BODY = re.compile(r"[A-Za-z0-9_-]{22}")


class CreateRequestInvalid(Exception):
    """A create's client write token, or the request it keys, cannot be accepted. Maps to
    `request_invalid`. `pointer` names the offending body field."""

    def __init__(self, *, pointer: str, detail: str) -> None:
        self.pointer = pointer
        self.detail = detail
        super().__init__(detail)


def mint_write_token(*, prefix: str) -> str:
    return f"{prefix}{secrets.token_urlsafe(16)}"


def write_token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def write_token_matches(*, presented: str | None, stored_digest: bytes | None) -> bool:
    """Compare digests in constant time. A missing token or a missing digest never matches."""
    if presented is None or stored_digest is None:
        return False
    return hmac.compare_digest(write_token_digest(presented), stored_digest)


def require_client_write_token(token: str, *, prefix: str = COMPONENT_WRITE_TOKEN_PREFIX) -> str:
    """Return `token` if it is `prefix` plus 22 url-safe base64 characters, the form
    `mint_write_token` mints, else raise `CreateRequestInvalid` at `/write_token`."""
    if token.startswith(prefix) and _TOKEN_BODY.fullmatch(token, len(prefix)):
        return token
    raise CreateRequestInvalid(
        pointer="/write_token",
        detail=(
            f"write_token must be {prefix} followed by 22 url-safe base64 characters; "
            f'mint it as "{prefix}" + secrets.token_urlsafe(16) or an equivalent CSPRNG'
        ),
    )


def request_digest(request: Mapping[str, object], *, pointer: str) -> bytes:
    """SHA-256 of the RFC 8785 canonical JSON of `request` as sent.

    Canonical JSON makes the digest independent of key order. A request RFC 8785 cannot
    represent, such as one holding an integer beyond 2**53 or a non-finite number, raises
    `CreateRequestInvalid` at `pointer`."""
    try:
        canonical = rfc8785.dumps(dict(request))
    except rfc8785.CanonicalizationError as exc:
        raise CreateRequestInvalid(
            pointer=pointer,
            detail=f"a create that sends write_token needs a request in I-JSON: {exc}",
        ) from exc
    return hashlib.sha256(canonical).digest()


def create_request_digest(*, intent: Mapping[str, object], composition_id: str | None) -> bytes:
    """`request_digest` of a component create's `{"intent", "composition_id"}`, refusing at
    `/intent`."""
    return request_digest(
        {"intent": dict(intent), "composition_id": composition_id}, pointer="/intent"
    )
