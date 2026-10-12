"""Request trace IDs shared by diagnostics, problem bodies, SSE `done` events, and logs.

Two `ContextVar`s carry per-request IDs from app middleware to exception
handlers, response helpers, and log records: the platform's own minted trace
ID, and the Cloud Trace ID the load balancer sent with the request.
"""

from __future__ import annotations

import contextvars
import re
import uuid
from collections.abc import Mapping

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar("dfo_trace_id", default=None)
_cloud_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dfo_cloud_trace_id", default=None
)

# `TRACE_ID/SPAN_ID;o=OPTIONS`, with the span and options parts optional.
_CLOUD_TRACE_CONTEXT_RE = re.compile(r"([0-9a-f]{1,32})(?:/|;|$)", re.IGNORECASE)
# W3C `version-traceid-parentid-flags`.
_TRACEPARENT_RE = re.compile(r"[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}", re.IGNORECASE)

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


def cloud_trace_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Return the Cloud Trace ID from `X-Cloud-Trace-Context`, else from `traceparent`.

    Cloud Run's front end sets both headers on every request. `headers` must
    look up names case-insensitively, as Starlette's `Headers` does. Returns
    `None` when neither header carries a well-formed, non-zero trace ID.
    """
    candidates = (
        (headers.get("x-cloud-trace-context"), _CLOUD_TRACE_CONTEXT_RE),
        (headers.get("traceparent"), _TRACEPARENT_RE),
    )
    for value, pattern in candidates:
        if value is None:
            continue
        match = pattern.match(value.strip())
        if match is not None and match.group(1).strip("0"):
            return match.group(1).lower()
    return None


def set_cloud_trace_id(trace_id: str | None) -> contextvars.Token[str | None]:
    """Bind the request's Cloud Trace ID and return a token for `reset_cloud_trace_id`."""
    return _cloud_trace.set(trace_id)


def reset_cloud_trace_id(token: contextvars.Token[str | None]) -> None:
    _cloud_trace.reset(token)


def get_cloud_trace_id() -> str | None:
    """Return the Cloud Trace ID bound to the current request, or `None` outside one."""
    return _cloud_trace.get()
