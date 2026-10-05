"""Request-scoped semaphore and ownership for trailing tasks.

``core.resolve_events`` binds one ``RequestScope`` for each request. A
``ContextVar`` makes it available below the entry point without threading it
through every call. Missing scopes raise ``MissingConcurrencyScope``.
That keeps helpers within the request limit and task ownership. Direct pipeline
tests bind ``scoped_concurrency()`` when they reach these helpers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from search.engine.config import settings


class MissingConcurrencyScope(RuntimeError):
    """A concurrency helper ran outside ``core.resolve_events`` or
    ``scoped_concurrency()``. This is a programming error. Callers must bind a scope.
    """


class RequestScope:
    """One request's shared concurrency limit and unclaimed trailing tasks.

    Variable fan-out and dataset resubmission share one semaphore, so their
    combined work stays under the configured limit. ``create_owned_task``
    registers returned tasks. Completion removes them. Scope exit cancels and
    awaits every unclaimed task. ``claim_withheld_read`` grants the request's one
    ``withheld_by_registry`` re-read.
    """

    __slots__ = ("_withheld_read_claimed", "semaphore", "tasks")

    def __init__(self) -> None:
        self.semaphore = asyncio.Semaphore(settings().max_variable_concurrency)
        self.tasks: set[asyncio.Task[Any]] = set()
        self._withheld_read_claimed = False

    def claim_withheld_read(self) -> bool:
        """True for the first caller in this request, False after. Call it on the event loop."""
        if self._withheld_read_claimed:
            return False
        self._withheld_read_claimed = True
        return True


_scope: ContextVar[RequestScope | None] = ContextVar("_search_concurrency_scope", default=None)


@asynccontextmanager
async def scoped_concurrency() -> AsyncIterator[RequestScope]:
    """Bind a fresh request scope and cancel any unclaimed tasks on exit.

    Call ``cancel()`` before the first await, including during cancellation
    unwinding. Then gather unfinished tasks with ``return_exceptions=True``.
    Consumed tasks remove themselves from the registry when they finish.
    """
    scope = RequestScope()
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)
        owned = list(scope.tasks)
        for task in owned:
            if not task.done():
                task.cancel()
        for task in owned:
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)


def current_scope() -> RequestScope:
    """The active `RequestScope`, or raise `MissingConcurrencyScope`."""
    scope = _scope.get()
    if scope is None:
        raise MissingConcurrencyScope(
            "no request-scoped concurrency bound -- call this from inside "
            "core.resolve_events, or open one with scoped_concurrency() for "
            "a direct call below it"
        )
    return scope


def bounded_semaphore() -> asyncio.Semaphore:
    """Return the semaphore shared by the current request's bounded call sites."""
    return current_scope().semaphore


def create_owned_task[T](coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    """Schedule and register a coroutine for a helper to return.

    With no bound scope, close the coroutine and raise
    ``MissingConcurrencyScope`` before scheduling it. This avoids an
    unawaited-coroutine warning.
    """
    scope = _scope.get()
    if scope is None:
        coro.close()
        raise MissingConcurrencyScope(
            "no request-scoped concurrency bound -- call this from inside "
            "core.resolve_events, or open one with scoped_concurrency() for "
            "a direct call below it"
        )
    task = asyncio.create_task(coro)
    scope.tasks.add(task)
    task.add_done_callback(scope.tasks.discard)
    return task
