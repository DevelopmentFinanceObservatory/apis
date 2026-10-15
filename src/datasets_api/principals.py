"""Admit callers by the principal they name.

Callers name themselves in one header, `DATASETS_PRINCIPAL_HEADER` (default `x-dfo-principal`),
whose value is a shared secret the calling service sets itself. Nothing in this service or in
front of it sets, replaces, or removes the header, so knowing a secret is what admits a caller.
Each configured secret holds one grant:

- `DATASETS_FIRST_PARTY_PRINCIPAL` holds `publish`, which opens the publish and recipe routes.
  `ai` sends it.
- `DATASETS_PIN_FIRST_PARTY_PRINCIPAL` holds `pin`, which opens the pin route. The website
  sends it.
- `LIVE_REFRESH_PRINCIPAL` holds `refresh`, which opens the frame refresh and the Presentation
  render routes. The website and `ai` both send it.

A secret configured under two names holds both grants, and an empty secret holds none.
`Principals.parse` reads one request's header into a `Principal`, comparing the value with every
configured secret in constant time, so a response time reveals neither which secret matched nor
a prefix of one. A missing or repeated header names nobody. `Principals.require` returns the
check a route runs before FastAPI reads its body, refusing a principal that lacks the grant with
the grant's 403 refusal.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from fastapi import Request, Response
from fastapi.routing import APIRoute

from datasets_api.config import DatasetsBuildSettings
from dfo_platform.problem import Refusal

logger = logging.getLogger(__name__)


class Grant(StrEnum):
    publish = "publish"
    pin = "pin"
    refresh = "refresh"


class PublishNotPermitted(Refusal):
    """The caller lacks the `publish` grant."""

    code = "publish_not_permitted"

    def __init__(self, *, presented: bool = False) -> None:
        self.presented = presented
        detail = "the publish permission is not granted to this caller"
        if not presented:
            detail = f"no principal header was presented; {detail}"
        super().__init__(detail)


class PinNotPermitted(Refusal):
    """The caller lacks the `pin` grant."""

    code = "pin_not_permitted"

    def __init__(self, *, presented: bool = False) -> None:
        self.presented = presented
        detail = "the pin permission is not granted to this caller"
        if not presented:
            detail = f"no principal header was presented; {detail}"
        super().__init__(detail)


class RefreshNotPermitted(Refusal):
    """The caller lacks the `refresh` grant."""

    code = "refresh_not_permitted"

    def __init__(self, *, presented: bool = False) -> None:
        self.presented = presented
        detail = "live refresh is restricted to its first-party principal"
        if not presented:
            detail = f"no principal header was presented; {detail}"
        super().__init__(detail)


@dataclass(frozen=True)
class Principal:
    """The caller that one request names. `presented` is the header's value, the first of several
    when it repeats, and `None` when it is absent; it stays out of the repr, since it may be a
    secret. `grants` is empty unless exactly one value matched a configured secret."""

    presented: str | None = field(repr=False)
    grants: frozenset[Grant]

    @property
    def label(self) -> str | None:
        """A stable name for the presented value that does not disclose it, for logs and stored
        rows. It is the first 16 hex digits of its SHA-256. `None` when no value was presented."""
        if self.presented is None:
            return None
        return f"secret-sha256:{hashlib.sha256(self.presented.encode()).hexdigest()[:16]}"

    def refusal(self, grant: Grant) -> Refusal:
        """The 403 a route needing `grant` answers this caller with."""
        match grant:
            case Grant.publish:
                return PublishNotPermitted(presented=self.presented is not None)
            case Grant.pin:
                return PinNotPermitted(presented=self.presented is not None)
            case Grant.refresh:
                return RefreshNotPermitted(presented=self.presented is not None)


@dataclass(frozen=True)
class Principals:
    """The principal header and the grant each configured secret holds."""

    header: str
    secrets: tuple[tuple[str, Grant], ...]

    @classmethod
    def resolve(
        cls,
        *,
        settings: DatasetsBuildSettings,
        principal_header: str | None = None,
        publish: frozenset[str] | None = None,
        pin: frozenset[str] | None = None,
        refresh: frozenset[str] | None = None,
    ) -> Self:
        """Default the header and each grant's secrets to `settings`."""
        by_grant = {
            Grant.publish: publish if publish is not None else {settings.first_party_principal},
            Grant.pin: pin if pin is not None else {settings.pin_first_party_principal},
            Grant.refresh: refresh if refresh is not None else {settings.live_refresh_principal},
        }
        return cls(
            header=principal_header if principal_header is not None else settings.principal_header,
            secrets=tuple(
                (secret, grant) for grant, secrets in by_grant.items() for secret in sorted(secrets)
            ),
        )

    def parse(self, request: Request) -> Principal:
        """Read the principal `request` names from its headers alone."""
        presented = request.headers.getlist(self.header)
        if len(presented) != 1:
            if presented:
                logger.warning(
                    "principal check saw %d values for the %r principal header on one "
                    "request; refusing rather than trusting either one",
                    len(presented),
                    self.header,
                )
            return Principal(presented=presented[0] if presented else None, grants=frozenset())
        value = presented[0].encode()
        grants: set[Grant] = set()
        # Every secret is compared, with no early exit, so the time does not depend on which
        # secret matched. An empty secret matches nothing.
        for secret, grant in self.secrets:
            if secret and hmac.compare_digest(value, secret.encode()):
                grants.add(grant)
        return Principal(presented=presented[0], grants=frozenset(grants))

    def require(self, grant: Grant) -> Callable[[Request], Principal]:
        """The check a route needing `grant` runs on each request before reading its body. It
        returns the caller's `Principal`.

        Raises:
            PublishNotPermitted, PinNotPermitted, RefreshNotPermitted: The caller lacks `grant`.
        """

        def check(request: Request) -> Principal:
            principal = self.parse(request)
            if grant not in principal.grants:
                raise principal.refusal(grant)
            return principal

        return check


def granted_route_class(check: Callable[[Request], Principal]) -> type[APIRoute]:
    """A route class that runs `check` before FastAPI validates the path or reads a body, and
    leaves the admitted caller in `request.state.principal`. Build a door's `APIRouter` with it
    so every route on the router is gated."""

    class _GrantedRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handler = super().get_route_handler()

            async def check_then_handle(request: Request) -> Response:
                request.state.principal = check(request)
                return await handler(request)

            return check_then_handle

    return _GrantedRoute


__all__ = [
    "Grant",
    "PinNotPermitted",
    "Principal",
    "Principals",
    "PublishNotPermitted",
    "RefreshNotPermitted",
    "granted_route_class",
]
