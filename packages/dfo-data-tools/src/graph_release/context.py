"""The graph release bound to the current request or operation.

A request binds one release at entry, beside the origin catalogue, and every
stage reads it back, so one request never reads two releases across a swap.
Context variables do not cross `ThreadPoolExecutor.submit`, so work handed to
a pool runs under a copied context, as it does for the catalogue
(`origin_catalogue.context`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    # Annotation-only, so binding a release imports no part reader (NumPy, pyarrow).
    from graph_release.service import Release, ReleaseService

URI_ENV_VAR = "GRAPH_RELEASE_URI"

_BOUND: ContextVar[Release | None] = ContextVar("graph_release", default=None)


class _ReleaseEnv(BaseSettings):
    """Env reads for the release URI, constructed fresh on each call so a
    `GRAPH_RELEASE_URI` set after import is still honoured."""

    model_config = SettingsConfigDict()

    uri: str | None = Field(default=None, alias=URI_ENV_VAR)

    @field_validator("uri", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


def release_uri() -> str | None:
    """`GRAPH_RELEASE_URI`, or `None` when it is unset or blank."""
    return _ReleaseEnv().uri


class ReleaseNotBound(RuntimeError):
    """A release was read with none bound to the current context."""


def bind_release(release: Release) -> Token[Release | None]:
    """Binds `release` to the current context. Pass the token to
    `unbind_release` to restore the previous binding."""
    return _BOUND.set(release)


def unbind_release(token: Token[Release | None]) -> None:
    _BOUND.reset(token)


@contextmanager
def bound_release(release: Release) -> Iterator[Release]:
    """Binds `release` for the body of the `with` block."""
    token = _BOUND.set(release)
    try:
        yield release
    finally:
        _BOUND.reset(token)


def bound_served_release(
    service: ReleaseService | None,
) -> AbstractContextManager[Release | None]:
    """Binds `service`'s served release for the body of the `with` block, and
    nothing when `service` is `None`, as when `GRAPH_RELEASE_URI` is unset
    (`optional_release_service`)."""
    return nullcontext() if service is None else bound_release(service.snapshot())


def optional_release_service() -> ReleaseService | None:
    """The process-wide release service when `GRAPH_RELEASE_URI` is set, `None`
    when it is unset or blank.

    A surface outside the Datasets one calls this rather than
    `graph_release.service.optional_release_service`, since `service` imports
    NumPy and pyarrow (the `graph-release` extra) and this module does so only
    when a release is configured.
    """
    if release_uri() is None:
        return None
    from graph_release import service

    return service.optional_release_service()


def bound_release_or_none() -> Release | None:
    """The bound release, or `None` when none is bound. For work that carries
    its caller's binding into a context of its own."""
    return _BOUND.get()


def current_release() -> Release:
    """The bound release. Raises `ReleaseNotBound` when none is bound."""
    release = _BOUND.get()
    if release is None:
        raise ReleaseNotBound(
            "no graph release is bound to this context; bind one at request entry, or "
            "run pooled work under contextvars.copy_context()"
        )
    return release


__all__ = [
    "URI_ENV_VAR",
    "ReleaseNotBound",
    "bind_release",
    "bound_release",
    "bound_release_or_none",
    "bound_served_release",
    "current_release",
    "optional_release_service",
    "release_uri",
    "unbind_release",
]
