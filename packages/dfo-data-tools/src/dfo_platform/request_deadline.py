"""A request-scoped absolute deadline that bounds every outbound read made on the request's
behalf, including reads whose call site passes no deadline of its own.

A route binds the request's hard deadline with `request_deadline`. The graph and SDMX clients
narrow each read's explicit deadline with `bounded_deadline`, so an explicit deadline still
applies as the tighter bound. Outside a bound request no deadline is added.

The value is a `ContextVar`. It reaches tasks and `asyncio.to_thread` workers, which copy the
caller's context, and pool threads only through `submit_with_context` or
`run_in_executor_with_context` (`dfo_platform.executor`). A bare `Executor.submit` or
`loop.run_in_executor` runs without it.

A shared flight, a read one request starts and other requests or doors join, runs detached
from any one request's deadline. It runs in `detached_context`, a copy of the caller's context
that keeps the bound origin catalogue and binds the flight's own deadline in place of the
request's. `submit_detached` submits a function in such a context with the deadline cleared,
for a flight that passes its deadline explicitly. `SingleFlightCache` runs every fetch it
shares in one that binds the flight's deadline. Each request, the one that started the flight
included, stops waiting at its own deadline while the flight runs on.

httpx applies a timeout per transport phase, and its read timeout to each socket read, so a
response that trickles bytes can outlive the deadline by up to one read timeout.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from contextlib import contextmanager

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "dfo_request_deadline", default=None
)


@contextmanager
def request_deadline(deadline: float) -> Iterator[None]:
    """Bind `deadline`, an absolute `time.monotonic()` value, for reads made inside the block."""
    token = _deadline.set(deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def bounded_deadline(deadline: float | None) -> float | None:
    """The earlier of `deadline` and the bound request deadline. None when neither is set."""
    ambient = _deadline.get()
    if ambient is None:
        return deadline
    if deadline is None:
        return ambient
    return min(deadline, ambient)


def detached_context(*, deadline: float | None = None) -> contextvars.Context:
    """A copy of the caller's context with the request deadline bound to `deadline`, or cleared
    when it is None. Every other context variable, such as the bound origin catalogue, carries
    over as `submit_with_context` (`dfo_platform.executor`) carries it."""
    context = contextvars.copy_context()
    context.run(_deadline.set, deadline)
    return context


def submit_detached[**P, R](
    executor: Executor, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> Future[R]:
    """Submit `fn` in `detached_context()`, with the request deadline cleared.

    For a shared flight that outlives the request that starts it and passes its own deadline
    to the reads it makes.
    """
    context = detached_context()

    def run() -> R:
        return context.run(fn, *args, **kwargs)

    return executor.submit(run)
