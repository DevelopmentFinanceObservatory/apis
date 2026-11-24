"""One Langfuse trace per resolve a route ran, so a grader can read what was asked and what came
back.

`send_resolve_trace` sends a trace named `search.resolve` whose Langfuse trace id is the
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

Only a request whose surface is in `TRACED_SURFACES` (`website` and `mcp`) is traced. A Studio
turn is traced by the harness, and `api`, `direct` and `unknown` requests are mostly crawlers
and scripts. Tracing is off unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are both set.
`LANGFUSE_BASE_URL` names the host, and the SDK reads its other `LANGFUSE_*` settings itself.
The client exports through a tracer provider of its own, so no other span in the process reaches
Langfuse.

The route calls `send_resolve_trace` once its log line is written. It builds the payload and
hands it to one worker thread, which records the span and flushes the client at once. The
service runs with `cpu_idle=true`, so CPU is throttled between requests and a batch left for the
SDK's interval could wait for the next request. Flushing per trace exports it while the request
that queued it is ending. Delivery stays best effort: a trace the flush misses leaves with a later
flush or the SDK's flush at exit, and is lost if the instance dies first. A failure anywhere is
logged at WARNING by type name only, since a message could carry query text, and never reaches
the request.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
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
TRACED_SURFACES: Final = frozenset({"website", "mcp"})

_TRACE_ID: Final = re.compile(r"[0-9a-f]{32}")
_EXECUTOR: Final = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resolve-trace")


def part_specs(part: Part) -> list[Spec]:
    """The specs a part answers with, in rank order, at most `MAX_SPECS`: a definite part's
    spec, a candidates part's specs, none for a no_data or unsupported part."""
    if isinstance(part, DefinitePart):
        return [part.spec]
    if isinstance(part, CandidatesPart):
        return part.candidates.specs[:MAX_SPECS]
    return []


def tracing_enabled() -> bool:
    """Whether both Langfuse keys are set."""
    return all(os.environ.get(name) for name in KEY_ENV_VARS)


def send_resolve_trace(
    *,
    trace_id: str | None,
    inp: ResolveInput,
    response: ResolveResponse | None,
    error_code: str | None,
    surface: str,
) -> None:
    """Queue the `search.resolve` trace of one resolve. It never raises.

    Nothing is sent while tracing is off, for a surface outside `TRACED_SURFACES`, or when
    `trace_id` is not 32 lowercase hex digits.
    """
    try:
        if (
            surface not in TRACED_SURFACES
            or not tracing_enabled()
            or trace_id is None
            or not _TRACE_ID.fullmatch(trace_id)
        ):
            return
        payload = {
            "input": trace_input(inp),
            "output": trace_output(response, error_code=error_code),
            "metadata": {"surface": surface},
        }
        _EXECUTOR.submit(_record, trace_id, payload)
    except Exception as exc:
        logger.warning("resolve trace not queued: %s", type(exc).__name__)


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


def _record(trace_id: str, payload: dict) -> None:
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
