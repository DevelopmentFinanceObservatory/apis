"""Admit callers of the live-refresh routes.

The frame refresh (`datasets_api.doors.refresh`) and the Presentation render routes take one
first-party principal, `LIVE_REFRESH_PRINCIPAL`, in the header the publish door reads
(`DATASETS_PRINCIPAL_HEADER`, default `x-dfo-principal`). `LiveRefreshPrincipal.admit` refuses
every other caller with `refresh_not_permitted` (403), and an unset or empty principal refuses
all of them.

The value is a shared secret, and the calling service puts it in the header itself. The website
and dfo-ai send `LIVE_REFRESH_PRINCIPAL` on these routes, as dfo-ai sends
`DATASETS_FIRST_PARTY_PRINCIPAL` on a publish and on the recipe routes and the website sends
`DATASETS_PIN_FIRST_PARTY_PRINCIPAL` on its pin call. Nothing in this service or in front of it
sets, replaces, or removes the header, so knowing the value is what admits a caller. Each
service holds it server-side.

The principal is separate from the publish and pin principals. It opens no publish, recipe, or
pin route, and neither of those principals opens a live-refresh route.

This module imports no door, so a door on either surface can use it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Self

from fastapi import Request, Response
from fastapi.routing import APIRoute

from datasets_api.config import DatasetsBuildSettings
from datasets_api.principals import same_principal
from dfo_platform import ProblemSpec

logger = logging.getLogger(__name__)


class RefreshNotPermitted(Exception):
    """The caller is not the live-refresh principal. Maps to `refresh_not_permitted` (403)."""

    def __init__(self, *, presented: bool) -> None:
        self.presented = presented
        detail = "live refresh is restricted to its first-party principal"
        if not presented:
            detail = f"no principal header was presented; {detail}"
        super().__init__(detail)


def classify_live_refresh_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, RefreshNotPermitted):
        return ProblemSpec(
            status=403, code="refresh_not_permitted", detail=str(exc), retryable=False
        )
    return None


@dataclass(frozen=True)
class LiveRefreshPrincipal:
    """The one caller allowed on the live-refresh routes, and the header that names a caller.
    An empty `principal` admits nobody."""

    header: str
    principal: str

    @classmethod
    def resolve(
        cls,
        *,
        settings: DatasetsBuildSettings,
        principal_header: str | None = None,
        principal: str | None = None,
    ) -> Self:
        """Default the header to `settings.principal_header` and the principal to
        `settings.live_refresh_principal`."""
        return cls(
            header=principal_header if principal_header is not None else settings.principal_header,
            principal=principal if principal is not None else settings.live_refresh_principal,
        )

    def admit(self, request: Request) -> None:
        """Check the caller from the headers alone, before a door reads the body.

        A missing or repeated principal header refuses, as does any value other than the
        configured principal. The comparison takes constant time.

        Raises:
            RefreshNotPermitted: The request does not name the live-refresh principal.
        """
        presented = request.headers.getlist(self.header)
        if not presented:
            raise RefreshNotPermitted(presented=False)
        if len(presented) > 1:
            logger.warning(
                "live-refresh principal check saw %d values for the %r principal header on "
                "one request; refusing rather than trusting either one",
                len(presented),
                self.header,
            )
            raise RefreshNotPermitted(presented=True)
        if not same_principal(presented[0], self.principal):
            raise RefreshNotPermitted(presented=True)

    def route_class(self) -> type[APIRoute]:
        """A route class that runs `admit` before FastAPI validates the path or reads a body.
        Build a door's `APIRouter` with it so every route on the router is gated."""
        admit = self.admit

        class _LiveRefreshRoute(APIRoute):
            def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
                handler = super().get_route_handler()

                async def admit_then_handle(request: Request) -> Response:
                    admit(request)
                    return await handler(request)

                return admit_then_handle

        return _LiveRefreshRoute


__all__ = [
    "LiveRefreshPrincipal",
    "RefreshNotPermitted",
    "classify_live_refresh_exception",
]
