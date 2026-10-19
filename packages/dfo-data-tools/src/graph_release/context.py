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

if TYPE_CHECKING:
    # Annotation-only, so binding a release imports no part reader (NumPy, pyarrow).
    from graph_release.service import Release, ReleaseService

_BOUND: ContextVar[Release | None] = ContextVar("graph_release", default=None)


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
    "ReleaseNotBound",
    "bind_release",
    "bound_release",
    "bound_release_or_none",
    "bound_served_release",
    "current_release",
    "unbind_release",
]
