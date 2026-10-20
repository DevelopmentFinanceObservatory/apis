"""The graph release bound to the current request or operation.

A request binds one release at entry, beside the origin catalogue, and every
stage reads it back, so one request never reads two releases across a swap.
Context variables do not cross `ThreadPoolExecutor.submit`, so work handed to
a pool runs under a copied context, as it does for the catalogue
(`origin_catalogue.context`).

`GRAPH_RELEASE_COVERAGE` turns on the release's serving views
(`graph_release.service.ReleaseViews`). The service builds them as it opens a release.
`code_index.shared.get_index` reads the bound release's views whenever it carries them, and with
the flag on refuses to read anything else (`current_views`). The flag is read here, on every
call, since `get_index` asks for it per read and a test may set it after import.

`GRAPH_RELEASE_ORIGIN` turns on the release's origin view (`graph_release.origin`), from which
the origin catalogue reads its DFO half (`served_origin`). It is read per call too, and needs
`GRAPH_RELEASE_COVERAGE` on: a cache that holds what the origin view decides (a search
response, an options answer, a node envelope, the search catalogue's build) keys on the coverage
build, which names the release only with coverage views on.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar, Token
from functools import lru_cache
from typing import TYPE_CHECKING, Final

from pydantic import Field, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    # Annotation-only, so binding a release imports no part reader (NumPy, pyarrow).
    from graph_release.origin import ReleaseOrigin
    from graph_release.service import Release, ReleaseService, ReleaseViews

_COVERAGE_ENV_VAR: Final[str] = "GRAPH_RELEASE_COVERAGE"
_ORIGIN_ENV_VAR: Final[str] = "GRAPH_RELEASE_ORIGIN"
_FLAG: Final[TypeAdapter[bool]] = TypeAdapter(bool)

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


class ReleaseViewsNotBuilt(RuntimeError):
    """The bound release's views were read, and its service built none."""


class OriginWithoutCoverage(RuntimeError):
    """`GRAPH_RELEASE_ORIGIN` is on and `GRAPH_RELEASE_COVERAGE` is off."""


def coverage_views_enabled() -> bool:
    """Whether `GRAPH_RELEASE_COVERAGE` is on. Unset or blank is off. A value pydantic does
    not read as a boolean raises `pydantic.ValidationError`."""
    return _flag(os.environ.get(_COVERAGE_ENV_VAR))


def origin_release_enabled() -> bool:
    """Whether `GRAPH_RELEASE_ORIGIN` is on, read as `coverage_views_enabled` reads its flag.
    Raises `OriginWithoutCoverage` when it is on and `GRAPH_RELEASE_COVERAGE` is off."""
    if not _flag(os.environ.get(_ORIGIN_ENV_VAR)):
        return False
    if not coverage_views_enabled():
        raise OriginWithoutCoverage(
            f"{_ORIGIN_ENV_VAR} is on and {_COVERAGE_ENV_VAR} is off; turn both on or both off"
        )
    return True


@lru_cache(maxsize=8)
def _flag(raw: str | None) -> bool:
    value = (raw or "").strip()
    return bool(value) and _FLAG.validate_python(value)


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


def current_views() -> ReleaseViews:
    """The bound release's serving views. Raises `ReleaseNotBound` when no release is bound
    and `ReleaseViewsNotBuilt` when its service built no views."""
    release = current_release()
    if release.views is None:
        raise ReleaseViewsNotBuilt(
            f"the bound graph release {release.release_id!r} carries no views; "
            f"its service was built with {_COVERAGE_ENV_VAR} off"
        )
    return release.views


def served_origin_key() -> str | None:
    """The bound release's id with `GRAPH_RELEASE_ORIGIN` on, `None` with it off. A cache whose
    entries derive from the origin catalogue's DFO half keys on it beside the catalogue build,
    so a request bound to one release never reads what another release decided. Raises
    `ReleaseNotBound` when the flag is on and no release is bound."""
    return current_release().release_id if origin_release_enabled() else None


def served_origin() -> ReleaseOrigin | None:
    """The bound release's origin view with `GRAPH_RELEASE_ORIGIN` on, `None` with it off.
    Raises `ReleaseNotBound` when the flag is on and no release is bound, and
    `ReleaseViewsNotBuilt` when its service built no origin view."""
    if not origin_release_enabled():
        return None
    release = current_release()
    if release.origin is None:
        raise ReleaseViewsNotBuilt(
            f"the bound graph release {release.release_id!r} carries no origin view; "
            f"its service was built with {_ORIGIN_ENV_VAR} off"
        )
    return release.origin


__all__ = [
    "URI_ENV_VAR",
    "OriginWithoutCoverage",
    "ReleaseNotBound",
    "ReleaseViewsNotBuilt",
    "bind_release",
    "bound_release",
    "bound_release_or_none",
    "bound_served_release",
    "coverage_views_enabled",
    "current_release",
    "current_views",
    "optional_release_service",
    "origin_release_enabled",
    "release_uri",
    "served_origin",
    "served_origin_key",
    "unbind_release",
]
