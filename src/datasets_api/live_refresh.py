"""Admit callers of the Presentation render routes.

The render routes admit the `refresh` grant alone (`datasets_api.principals`), which only
`LIVE_REFRESH_PRINCIPAL` holds. `LiveRefreshPrincipal` is that one-secret view, built by the
composition and handed to the Presentation surface, and an unset or empty principal refuses
every caller with `refresh_not_permitted` (403). The Datasets frame refresh reads the same
secret through `Principals`.

This module imports no door, so a door on either surface can use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from fastapi import Request
from fastapi.routing import APIRoute

from datasets_api.config import DatasetsBuildSettings
from datasets_api.principals import (
    Grant,
    Principal,
    Principals,
    RefreshNotPermitted,
    granted_route_class,
)
from dfo_platform import ProblemSpec
from dfo_platform.codes import registered_status


def classify_live_refresh_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, RefreshNotPermitted):
        status = registered_status(exc.code)
        return ProblemSpec(status=status, code=exc.code, detail=str(exc), retryable=False)
    return None


@dataclass(frozen=True)
class LiveRefreshPrincipal:
    """The one caller allowed on the render routes, and the header that names a caller. An
    empty `principal` admits nobody."""

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

    def admit(self, request: Request) -> Principal:
        """Check the caller from the headers alone, before a door reads the body.

        Raises:
            RefreshNotPermitted: The request does not name the live-refresh principal.
        """
        secrets = ((self.principal, Grant.refresh),)
        return Principals(header=self.header, secrets=secrets).require(Grant.refresh)(request)

    def route_class(self) -> type[APIRoute]:
        """A route class that runs `admit` before FastAPI validates the path or reads a body."""
        return granted_route_class(self.admit)


__all__ = [
    "LiveRefreshPrincipal",
    "RefreshNotPermitted",
    "classify_live_refresh_exception",
]
