"""Write tokens grant revise. A `cmp_` or `cps_` id grants only read.

A create mints a `cmw_` token for the new component, and `composition_id="new"` mints a
`cpw_` token for the new composition. Each token is its prefix plus
`secrets.token_urlsafe(16)`, 128 random bits. The store keeps only the unsalted SHA-256
digest, and the reply returns the token once.

A client may instead send its own `cmw_` token with a create, in the same format. The token
then keys the create: `create_request_digest` hashes the request it came with, and a retried
create with the same token and the same request returns the first create's draft.
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
    "CreateRequestInvalid",
    "create_request_digest",
    "mint_write_token",
    "require_client_write_token",
    "write_token_digest",
    "write_token_matches",
]

COMPONENT_WRITE_TOKEN_PREFIX = "cmw_"
COMPOSITION_WRITE_TOKEN_PREFIX = "cpw_"

# `secrets.token_urlsafe(16)` returns 22 characters from the url-safe base64 alphabet.
_CLIENT_WRITE_TOKEN = re.compile(rf"{COMPONENT_WRITE_TOKEN_PREFIX}[A-Za-z0-9_-]{{22}}")


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


def require_client_write_token(token: str) -> str:
    """Return `token` if it is `cmw_` plus 22 url-safe base64 characters, else raise
    `CreateRequestInvalid` at `/write_token`."""
    if _CLIENT_WRITE_TOKEN.fullmatch(token):
        return token
    raise CreateRequestInvalid(
        pointer="/write_token",
        detail=(
            "write_token must be cmw_ followed by 22 url-safe base64 characters; "
            'mint it as "cmw_" + secrets.token_urlsafe(16) or an equivalent CSPRNG'
        ),
    )


def create_request_digest(*, intent: Mapping[str, object], composition_id: str | None) -> bytes:
    """SHA-256 of the RFC 8785 canonical JSON of `{"intent", "composition_id"}` as sent.

    Canonical JSON makes the digest independent of key order. An intent RFC 8785 cannot
    represent, such as an integer beyond 2**53 or a non-finite number, raises
    `CreateRequestInvalid` at `/intent`."""
    try:
        canonical = rfc8785.dumps({"intent": dict(intent), "composition_id": composition_id})
    except rfc8785.CanonicalizationError as exc:
        raise CreateRequestInvalid(
            pointer="/intent",
            detail=f"a create that sends write_token needs an intent in I-JSON: {exc}",
        ) from exc
    return hashlib.sha256(canonical).digest()
