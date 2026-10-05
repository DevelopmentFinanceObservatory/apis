"""Verifies the worker route's OIDC token after Cloud Run IAM and internal ingress.

Google's verifier checks signature, expiry, issuer, and audience. `LiveOidcVerifier` checks
`email_verified` and matches the token's `email` to the configured service account.
"""

from __future__ import annotations

from typing import Protocol


class OidcRejected(Exception):
    """The bearer token failed verification. The worker route returns `reason` in its 401 body."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"OIDC token rejected: {reason}")


class OidcVerifier(Protocol):
    def verify(self, token: str, *, audience: str, expected_email: str) -> None:
        """Raises `OidcRejected` unless the token verifies for `audience` and `expected_email`."""
        ...


class FakeOidcVerifier:
    """Accepts listed tokens, or all tokens when `valid_tokens` is unset.

    Use `rejecting()` to reject every token.
    """

    def __init__(self, *, valid_tokens: frozenset[str] | None = None) -> None:
        self._valid_tokens = valid_tokens

    def verify(self, token: str, *, audience: str, expected_email: str) -> None:
        if self._valid_tokens is not None and token not in self._valid_tokens:
            raise OidcRejected(reason="token not in this fake's own accepted set")

    @classmethod
    def rejecting(
        cls, *, reason: str = "fake configured to reject every token"
    ) -> FakeOidcVerifier:
        verifier = cls(valid_tokens=frozenset())
        return verifier


class LiveOidcVerifier:
    """Checks Google's token claims and the configured service-account email."""

    def verify(self, token: str, *, audience: str, expected_email: str) -> None:
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token

        try:
            claims = google_id_token.verify_oauth2_token(
                token, google_requests.Request(), audience=audience
            )
        except (GoogleAuthError, ValueError) as exc:
            raise OidcRejected(reason=f"token did not verify: {exc}") from exc
        email = claims.get("email")
        if email != expected_email:
            raise OidcRejected(reason=f"token email {email!r} is not the configured invoker")
        if not claims.get("email_verified", False):
            raise OidcRejected(reason="token's own email claim is not verified")


__all__ = ["FakeOidcVerifier", "LiveOidcVerifier", "OidcRejected", "OidcVerifier"]
