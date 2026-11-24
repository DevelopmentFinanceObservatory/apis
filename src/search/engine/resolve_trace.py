"""One Langfuse trace per resolve a route ran, so a grader can read what was asked and what came
back.

`queue_resolve_trace` queues a trace named `search.resolve` whose Langfuse trace id is the
request's own trace id (`dfo_platform.trace.get_trace_id`, 32 lowercase hex digits), the
`trace_id` the `resolve` log line carries, so a grader fetches the trace by that id. The trace's
one observation carries:

- input: the request's whole `input` as posted, in its JSON form without null members, every
  string clipped to `QUERY_CHARS`. A raw_text request carries its `query`; a parsed request,
  as MCP clients send, carries each part's `variable_text`, entities and time text, and
  `raw_query` only when the client sent one.
- output: `kind` (`resolution`, `catalogue` or `declined`, null when the request ended before
  its response), `status` (a resolution's status, a declined answer's reason) and `parts`, each
  part's `status` and up to `MAX_SPECS` `specs` in rank order (`part_specs`), each a `spec_id`
  and its `rendered_sentence` clipped to `DEFINITION_CHARS`. `error_code` is the
  `ResolveRecord.error_code` the log line carries.
- metadata: `surface`, the delivery surface of the request (`resolve_line`).

Only a `website` or `mcp` request (`TRACED_SURFACES`) is traced. A Studio turn is traced by the
harness, and `api`, `direct` and `unknown` requests are mostly crawlers and scripts. A
`dataset_resubmit` request is never traced: its entries carry the original `raw_query` beside
new text, so the trace would not say what was asked. Tracing is off unless `SEARCH_TRACE_ENABLED`
is `1` or `true` and `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are both set. The keys alone
leave it off, since they also configure the eval dataset sync. `LANGFUSE_BASE_URL` names the
host, and the SDK reads its other `LANGFUSE_*` settings itself. The client exports through a
tracer provider of its own, so no other span in the process reaches Langfuse.

Delivery: the route's `resolve_line` calls `queue_resolve_trace` when the request ends, before it
writes its log line. The payload goes to one worker thread, which records the span and flushes the
client at once. At most `PENDING_CAP` traces wait for the worker; a trace past the cap is
dropped. The request then awaits its own export for up to `EXPORT_WAIT_S`
(`await_resolve_trace`) on the event loop, so other requests keep running. The service runs on
Cloud Run with request-based CPU, so a thread left working after its request ends stalls until
the next request; waiting inside the request exports the trace while the instance still has CPU.
A trace still exporting at the cap finishes later, and is lost if the instance dies first. A
request being cancelled does not wait. The lifespan's shutdown waits up to `SHUTDOWN_DRAIN_S` for
the worker to finish the traces already queued (`drain_resolve_traces`).

The log line's `trace` field records what became of the request's trace (`TraceOutcome`): `off`
while tracing is off, `skipped` for a request not traced, `dropped` past the cap, `sent` once
exported, `timed_out` when the export outlasted the wait, `queued` when the request was cancelled
before it could wait, and `failed` when building or exporting the trace raised. A failure is
logged at WARNING by type name only, since a message could carry query text, and never reaches
the request.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from search.models import (
    CandidatesPart,
    DeclinedAnswer,
    DefinitePart,
    Part,
    Resolution,
    ResolveInput,
    ResolveResponse,
    Spec,
)

if TYPE_CHECKING:
    from langfuse import Langfuse


logger = logging.getLogger(__name__)

TRACE_NAME: Final = "search.resolve"
MAX_SPECS: Final = 10
# The site's AI policy says a search record keeps the first 500 characters of the question,
# so every string of the traced input, website and MCP alike, stops there.
QUERY_CHARS: Final = 500
DEFINITION_CHARS: Final = 500
KEY_ENV_VARS: Final = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
ENABLED_ENV_VAR: Final = "SEARCH_TRACE_ENABLED"
TRACED_SURFACES: Final = frozenset({"website", "mcp"})
# Traces waiting for the worker, past which a trace is dropped, so a stalled export cannot hold
# an unbounded number of payloads in memory.
PENDING_CAP: Final = 100
# How long a request waits for its own trace to export while it still has CPU.
EXPORT_WAIT_S: Final = 1.0
SHUTDOWN_DRAIN_S: Final = 2.0

_TRACE_ID: Final = re.compile(r"[0-9a-f]{32}")
_EXECUTOR: Final = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resolve-trace")
_PENDING: Final = threading.BoundedSemaphore(PENDING_CAP)


class TraceOutcome(StrEnum):
    """What became of a request's trace, the `resolve` log line's `trace` field."""

    off = "off"
    skipped = "skipped"
    dropped = "dropped"
    queued = "queued"
    sent = "sent"
    timed_out = "timed_out"
    failed = "failed"


def part_specs(part: Part) -> list[Spec]:
    """The specs a part answers with, in rank order, at most `MAX_SPECS`: a definite part's
    spec, a candidates part's specs, none for a no_data or unsupported part."""
    if isinstance(part, DefinitePart):
        return [part.spec]
    if isinstance(part, CandidatesPart):
        return part.candidates.specs[:MAX_SPECS]
    return []


def tracing_enabled() -> bool:
    """Whether `SEARCH_TRACE_ENABLED` is `1` or `true` and both Langfuse keys are set."""
    enabled = os.environ.get(ENABLED_ENV_VAR, "").strip().lower() in {"1", "true"}
    return enabled and all(os.environ.get(name) for name in KEY_ENV_VARS)


def queue_resolve_trace(
    *,
    trace_id: str | None,
    inp: ResolveInput,
    response: ResolveResponse | None,
    error_code: str | None,
    surface: str,
) -> tuple[TraceOutcome, Future[TraceOutcome] | None]:
    """Queue the `search.resolve` trace of one resolve. It never raises.

    Returns the outcome so far and, for a queued trace, the future its export resolves with.
    Nothing is queued while tracing is off (`off`), for a surface outside `TRACED_SURFACES`, a
    `dataset_resubmit` request or a `trace_id` that is not 32 lowercase hex digits (`skipped`),
    or with `PENDING_CAP` traces already waiting (`dropped`).
    """
    try:
        if not tracing_enabled():
            return TraceOutcome.off, None
        if (
            surface not in TRACED_SURFACES
            or inp.kind == "dataset_resubmit"
            or trace_id is None
            or not _TRACE_ID.fullmatch(trace_id)
        ):
            return TraceOutcome.skipped, None
        payload = {
            "input": trace_input(inp),
            "output": trace_output(response, error_code=error_code),
            "metadata": {"surface": surface},
        }
        if not _PENDING.acquire(blocking=False):
            return TraceOutcome.dropped, None
        try:
            future = _EXECUTOR.submit(_record, trace_id, payload)
        except BaseException:
            _PENDING.release()
            raise
    except Exception as exc:
        logger.warning("resolve trace not queued: %s", type(exc).__name__)
        return TraceOutcome.failed, None
    return TraceOutcome.queued, future


async def await_resolve_trace(future: Future[TraceOutcome]) -> TraceOutcome:
    """Wait up to `EXPORT_WAIT_S` for a queued trace to export, without blocking the loop.

    Returns `sent` or `failed` from the export, or `timed_out` when it outlasts the wait, in which
    case the export carries on. A cancellation of the waiting task propagates and leaves the
    export running.
    """
    try:
        async with asyncio.timeout(EXPORT_WAIT_S):
            return await asyncio.shield(asyncio.wrap_future(future))
    except TimeoutError:
        return TraceOutcome.timed_out
    except Exception as exc:
        logger.warning("resolve trace not awaited: %s", type(exc).__name__)
        return TraceOutcome.failed


async def drain_resolve_traces() -> None:
    """Wait up to `SHUTDOWN_DRAIN_S` for the worker to export every trace already queued. Each
    export flushes the client, so a drained worker leaves nothing buffered."""
    try:
        async with asyncio.timeout(SHUTDOWN_DRAIN_S):
            await asyncio.wrap_future(_EXECUTOR.submit(lambda: None))
    except TimeoutError:
        logger.warning("resolve traces still exporting at shutdown")


def trace_input(inp: ResolveInput) -> dict:
    """The trace's input: the request's whole `input`, every string clipped to `QUERY_CHARS`."""
    return _clipped(inp.model_dump(mode="json", exclude_none=True))


def trace_output(response: ResolveResponse | None, *, error_code: str | None) -> dict:
    """The trace's output: the answer and each part's specs. See the module docstring."""
    if response is None:
        return {"kind": None, "status": None, "parts": [], "error_code": error_code}
    if isinstance(response, Resolution):
        status: str | None = response.status
    elif isinstance(response, DeclinedAnswer):
        status = response.reason
    else:
        status = None
    parts = response.parts if isinstance(response, Resolution) else []
    return {
        "kind": response.kind,
        "status": status,
        "parts": [
            {
                "status": part.status,
                "specs": [
                    {
                        "spec_id": spec.spec_id,
                        "definition": _clip(spec.resolved.rendered_sentence, DEFINITION_CHARS),
                    }
                    for spec in part_specs(part)
                ],
            }
            for part in parts
        ],
        "error_code": error_code,
    }


def _clip(text: str | None, limit: int) -> str | None:
    return text[:limit] if text is not None else None


def _clipped(value: object) -> Any:
    """`value`, a JSON value, with every string in it clipped to `QUERY_CHARS`."""
    if isinstance(value, str):
        return value[:QUERY_CHARS]
    if isinstance(value, list):
        return [_clipped(item) for item in value]
    if isinstance(value, dict):
        return {key: _clipped(item) for key, item in value.items()}
    return value


@functools.cache
def _client() -> Langfuse:
    """The process's Langfuse client, on a tracer provider of its own."""
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider

    return Langfuse(tracer_provider=TracerProvider())


def _record(trace_id: str, payload: dict) -> TraceOutcome:
    """Record one trace on the worker thread and flush it."""
    try:
        from langfuse import propagate_attributes

        client = _client()
        with (
            client.start_as_current_observation(
                name=TRACE_NAME,
                as_type="span",
                trace_context={"trace_id": trace_id},
                input=payload["input"],
                output=payload["output"],
                metadata=payload["metadata"],
            ),
            propagate_attributes(trace_name=TRACE_NAME, metadata=payload["metadata"]),
        ):
            pass
        client.flush()
    except Exception as exc:
        logger.warning("resolve trace not recorded: %s", type(exc).__name__)
        return TraceOutcome.failed
    finally:
        _PENDING.release()
    return TraceOutcome.sent
