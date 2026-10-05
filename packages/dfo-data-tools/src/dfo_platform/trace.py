"""Request trace IDs shared by diagnostics, problem bodies, and SSE `done` events.

A `ContextVar` carries the ID from app middleware to exception handlers and response helpers.
"""

from __future__ import annotations

import contextvars
import uuid

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar("dfo_trace_id", default=None)

TRACE_ID_HEADER = "x-dfo-trace-id"
"""Response header carrying this request's trace_id, on every response from
every surface except one marked shared-cacheable (`Cache-Control: public`):
such a response is stored and replayed by a shared or private cache for its
whole freshness lifetime, so it carries no per-request identifier rather
than falsifying one on every later request served from that cache entry.
Lowercase to match the surface's own `x-dfo-principal` convention
(`datasets_api.config.DATASETS_PRINCIPAL_HEADER`)."""


def new_trace_id() -> str:
    """Mint a fresh trace ID as 32 lowercase hexadecimal characters."""
    return uuid.uuid4().hex


def set_trace_id(trace_id: str) -> contextvars.Token[str | None]:
    """Bind `trace_id` and return a token for `reset_trace_id`."""
    return _current.set(trace_id)


def reset_trace_id(token: contextvars.Token[str | None]) -> None:
    _current.reset(token)


def get_trace_id() -> str:
    """Return the current trace ID, minting and binding one outside a request context."""
    trace_id = _current.get()
    if trace_id is None:
        trace_id = new_trace_id()
        _current.set(trace_id)
    return trace_id
