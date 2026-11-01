"""Search engine refusal and infrastructure error types.

Input errors map to 4xx. Infrastructure failures propagate as 502/503 and count as
evaluation skips. A grounding miss means the graph returned 200 without a node.
Role-compatibility errors are separate from graph errors. Semantic outcomes return
`NoDataPart` or `UnsupportedPart`.

HTTP error bodies use the shared RFC 9457 `ProblemDetails` type from
`dfo_platform.problem`.
"""

from __future__ import annotations

import logging

from fastapi.exceptions import RequestValidationError

import dc_graph
from dc_sdmx import SdmxClientError
from dfo_platform import Alternative, Cause, ProblemSpec, refusal
from dfo_platform.validation import error_field_loc, pointer_for_error, validation_problem
from materialise.errors import UpstreamResponseError, UpstreamVersionChangedError
from materialise.rate_tables import RateTableInvalidError


class EngineInputError(Exception):
    """Deterministic input refusal, HTTP 400 by default, separate from infrastructure errors.

    `message` is the text the caller reads, and a coded refusal serves it as the problem
    `detail`. `code` selects the registry entry. An omitted code falls back to
    `request_invalid` with the generic detail, which keeps the message off the wire.
    Unknown-shape routing refusals omit it to prevent shape probing. `status` sets the
    HTTP status. Optional `pointer`, `alternatives`, and `cause` map to problem-detail
    extensions.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int = 400,
        pointer: str | None = None,
        alternatives: list[Alternative] | None = None,
        cause: Cause | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.pointer = pointer
        self.alternatives = alternatives
        self.cause = cause


class EngineInfraError(Exception):
    """LLM transport/schema or graph transport/status/JSON/timeout failure.

    Propagate these errors to avoid false no-data results.
    """


RATE_TABLE_REFUSALS = (
    RateTableInvalidError,
    UpstreamVersionChangedError,
)
"""Conversion-table failures that refuse the whole request on every door: an
inadmissible table or base year (501 `capability_unavailable`) or a table version
that moved mid-load (503, retryable). `classify_refusal` maps them, and a
degrading fan-out re-raises them as it does `EngineInfraError`."""

UPSTREAM_READ_ERRORS = (
    UpstreamResponseError,
    dc_graph.GraphReadError,
    SdmxClientError,
)
"""Upstream failures of a materialise read, graph or SDMX, past its own retries: a
graph read, a deadline the graph read crossed among them (`GraphDeadlineExceededError`),
an SDMX read, or an upstream answer materialise could not use. A preview read counts them
as failed reads and omits what they leave unread, beside `TimeoutError` for the shared
SDMX deadline (`observations.fetch`, `observations.bulk_conversion`)."""


def classify_refusal(exc: BaseException) -> ProblemSpec | None:
    """Search's one dispatch over `exc`, or `None` to defer to the platform. It logs nothing:
    a refusal that wants a line names it (`ProblemSpec.log`), which its reader logs
    (`dfo_platform.problem.log_problem`)."""
    if isinstance(exc, EngineInputError):
        # Code-less routing failures keep generic code and detail to prevent ID probing.
        code = exc.code if exc.code is not None else "request_invalid"
        detail = None if exc.code is None else str(exc)
        return ProblemSpec(
            status=exc.status,
            code=code,
            detail=detail,
            pointer=exc.pointer,
            alternatives=exc.alternatives,
            cause=exc.cause,
            retryable=False,
        )
    if isinstance(exc, RATE_TABLE_REFUSALS):
        # The materialise library refuses an inadmissible rate table with 501
        # `capability_unavailable` and a table version that moved mid-load with 503, retryable.
        return ProblemSpec(
            status=exc.status,
            code=exc.code,
            detail=exc.detail,
            retryable=exc.retryable,
            log=f"Rate table refusal {exc.code}: {exc.detail}",
            log_level=logging.WARNING,
        )
    if isinstance(exc, EngineInfraError):
        # Upstream 4xx responses (e.g. graph returned 404) map to 502 Bad Gateway
        # so callers can distinguish client-side upstream errors from transient failures.
        upstream_status = getattr(exc, "upstream_status", None)
        client_side = isinstance(upstream_status, int) and 400 <= upstream_status < 500
        problem = (
            ProblemSpec(status=502, code="upstream_error", retryable=True)
            if client_side
            else ProblemSpec(status=503, code="service_unavailable", retryable=True)
        )
        return problem._replace(log=f"Engine infrastructure error: {exc}", log_traceback=True)
    if isinstance(exc, RequestValidationError):
        # The platform maps body errors to 400 and schema-version errors to 422. Search adds
        # a field-specific refinement below.
        problem = validation_problem(exc)
        errors = exc.errors()
        # Unsupported identity versions and recognised future facet axes also map to 422.
        # Match the nested field path or Pydantic error type. A top-level schema-version error
        # already maps to 422 in the platform.
        if problem.code == "request_invalid":
            version_error = next(
                (
                    e
                    for e in errors
                    if error_field_loc(e)[-1:] == ["identity_version"]
                    or e.get("type") == "facet_axis_higher_schema_version"
                ),
                None,
            )
            if version_error is not None:
                problem = problem._replace(
                    code="unsupported_schema_version",
                    status=422,
                    pointer=pointer_for_error(exc, version_error),
                    detail=None,
                )
        return problem
    if isinstance(exc, TimeoutError):
        return refusal.render_timeout_refusal()._replace(log="Request hit its hard deadline")
    return None


def served_problem(exc: BaseException) -> ProblemSpec:
    """The refusal the route serves for `exc`, Search's own else the platform's."""
    return classify_refusal(exc) or refusal.default_problem(exc)


class LLMInfraError(EngineInfraError):
    """LLM-specific infrastructure failure (transport, API error, schema validation)."""


class LLMOutputRefused(LLMInfraError):
    """The model answered with JSON its response schema refuses on purpose, such as a
    binding that names no value (`clients.llm.SCHEMA_REFUSAL`). Gemini leaves the parsed
    response empty for it. A caller whose schema refuses an answer on purpose can answer
    for it; every other caller sees an `LLMInfraError`. `usage` is the refused call's token
    usage, `None` when the model reported none."""

    def __init__(self, message: str, *, usage: dict | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class GraphInfraError(EngineInfraError):
    """Graph transport, status, JSON, or timeout failure.

    `upstream_status` is `None` when unavailable. Upstream 4xx maps to 502.
    Other failures map to 503.
    """

    def __init__(self, *args, upstream_status: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.upstream_status = upstream_status


class GraphReadFailure(GraphInfraError, dc_graph.GraphReadError):
    """Graph failure caught as both `GraphInfraError` and `GraphReadError`.

    `upstream_status` is `None` for transport failures.
    """

    def __init__(self, *args: object, upstream_status: int | None = None) -> None:
        Exception.__init__(self, *args)
        self.upstream_status = upstream_status


class GraphDeadlineFailure(GraphInfraError, dc_graph.GraphDeadlineExceededError):
    """Deadline failure caught as both `GraphInfraError` and `GraphReadError`.

    The timeout has no upstream response status.
    """

    def __init__(self, *args: object, upstream_status: int | None = None) -> None:
        Exception.__init__(self, *args)
        self.upstream_status = upstream_status


class RoleNotDeclaredError(Exception):
    """Requested role has no declared `observationProperties` edge.

    Separate from `GraphInfraError` so callers can return `role_incompatible`.
    Raised by SDMX client implementations.
    """


class RoleSignatureError(Exception):
    """Role signatures conflict or exceed one mixer query's capacity.

    Separate from `GraphInfraError`. Raised by SDMX clients and `group_roles`.
    """


class FamilyNotResolvableError(Exception):
    """The mint's member walk refuses a family for a fault in one of its members: a
    constraint with two values, a degenerate role arc, a member that is no StatVar. A
    resubmit's walk also raises it for a family it cannot settle: a query with no measured
    property, or a candidate its reads leave undismissed.

    Raised by `retrieval.bound_values.RoleClassWalks`. The mint refuses the identity
    `member_not_resolvable`, so Search answers nothing for it.
    """


class MembershipTooBroadError(Exception):
    """A resubmit's member walk meets a family the mint refuses as too broad: a filtered
    set past the subject ceiling, or more members than the member cap.

    Raised by `retrieval.bound_values.RoleClassWalks` in its resubmit mode. The resubmit
    answers the part `scope_too_broad`.
    """


class DeclaredDefaultUnsupportedError(EngineInputError):
    """The graph declares a default facet policy for a StatVar axis.

    Search coverage, the headline and the preview route elect facets by ranking
    (`retrieval.coverage`), which models no declared default, while materialise serves a
    declared default before ranking. `retrieval.facet_policy.refuse_declared_defaults`
    raises it at resolve time, before the shape's coverage is built and before any sentence
    of the part is sent (the pipeline holds sentences until the answer stands), as the
    registered `facet_binding_unsupported` refusal (501, not retryable): a facet axis this
    deployment recognises but cannot honour. The fan-out re-raises it rather than degrade it
    to no data. A part whose elected answer leaves terms unmet answers nothing on the
    series, so it refuses `qualifier_not_bound` instead (`region.pipeline._run_phases`).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, code="facet_binding_unsupported", status=501)


class DefaultScopeIndexSourcedError(Exception):
    """A subject-only shape whose population the code index sources would take a default
    place scope.

    The default-scope presence read covers a shape's `sv_arc_facts`, and an index-sourced
    shape is probed over its index population, which can hold members the read never
    covered. Raised where the read's StatVars are chosen
    (`region.place_context.default_scope_stat_vars`) for the shape being answered, the
    elected shape or the shape a resubmit posts, which refuses the request as the unhandled
    default (500 `internal_error`) before any member meets an unread StatVar, unless the
    elected answer leaves terms unmet and the part refuses `qualifier_not_bound` instead
    (`region.pipeline._run_phases`). An alternate
    in that position is dropped from the offer, and the answer proceeds without it
    (`region.phase_place_materialise._bind_alternates`).
    """


class RoleEntityUnplaced(Exception):
    """Entity axes do not match the shape's declared roles.

    Separate from `GraphInfraError` because this is a data compatibility refusal.
    Raised by `engine.shape.entities_by_role`.
    """


class GraphCapExceededError(dc_graph.GraphCapExceededError):
    """Decoded bytes exceeded `max_bytes` while reading, before facet parsing.

    `cap_kind` is `bytes`. `cap` is the configured limit, and `observed` is the
    consumed byte count at abort, a lower bound on response size. This caps the
    wire fetch. Callers apply row limits after fetch. Separate from
    `GraphInfraError`. Raised by `observation_facets_batch`.
    """


class GroundingMiss(Exception):
    """Graph returned 200 but the proposed dcid has no node.

    Grounding callers drop the dcid or return no-data. This is a semantic outcome.
    """

    def __init__(self, dcid: str, message: str = ""):
        self.dcid = dcid
        super().__init__(message or f"Node absent in graph: {dcid!r}")
