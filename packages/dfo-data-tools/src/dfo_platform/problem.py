"""RFC 9457 problem-details bodies and the `Envelope` that builds them.

`ProblemSpec` holds classified refusals as data. FastAPI loads only during response rendering.
"""

from __future__ import annotations

import http
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, NamedTuple

from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter

from dfo_platform.codes import CodeCatalogue, registered_status
from dfo_platform.trace import get_trace_id

if TYPE_CHECKING:
    from fastapi.responses import JSONResponse

PROBLEM_JSON = "application/problem+json"


class Alternative(BaseModel):
    """One value that would satisfy the request in place of what was posted."""

    ref: str = Field(description="the alternative's own reference (e.g. a code or id)")
    label: str = Field(description="the alternative's human-readable label")


class Cause(BaseModel):
    """The `cause` extension: why a refusal fired, beneath its own top-level
    `code` -- present only on a refusal that carries one (`recipe_not_declared`,
    `rate_coverage_gap`, `operation_failed`, or a Presentation refusal of a frame). On
    `operation_failed`, `code` is the failed operation's refusal code, and on a
    frame's refusal the reason the frame does not load. Mirrors `alternatives`/`truncated`'s
    own extension style: `entities` names what the cause concerns (entity or
    `entity@year` keys), capped, with `truncated` set when the cap cut items
    short.
    """

    code: str = Field(
        description=(
            "the cause code: a sub-cause such as nominal_undeclared, recipe_undeclared, or "
            "rate_missing, on operation_failed the failed operation's own refusal code, or on "
            "a Presentation refusal of a frame the reason the frame does not load"
        )
    )
    entities: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "the entity or entity@year keys this cause concerns, in the order given, "
            "capped at 10; empty for a cause that names no entity"
        ),
    )
    truncated: bool = Field(
        default=False,
        description="true when the entities cap cut items short",
    )


class ProblemIssue(BaseModel):
    """One failure among several a single refusal can report at once.

    Each issue carries its own pointer, the member or axis it concerns, its
    own alternatives and truncation flag, and the machine code and message of
    the failure it stands for -- mirroring the same-named top-level
    ProblemDetails fields, but scoped to one entry among many.
    Fields default to None and are omitted from the wire body when absent
    (the envelope's omit-when-absent rule).
    """

    pointer: str | None = Field(
        default=None, description="a JSON Pointer into the request body naming the field at fault"
    )
    member: str | None = Field(
        default=None,
        description="the member this issue concerns, when the refusal is member-scoped",
    )
    axis: str | None = Field(
        default=None, description="the axis this issue concerns, when the refusal is axis-scoped"
    )
    alternatives: list[Alternative] | None = Field(
        default=None, description="values that would satisfy this issue, when known"
    )
    truncated: bool | None = Field(
        default=None,
        description="present iff alternatives is present; whether the cap cut items",
    )
    code: str | None = Field(
        default=None,
        description="the stable machine-readable code of the failure this issue stands for",
    )
    detail: str | None = Field(
        default=None,
        description="human-readable explanation of the failure this issue stands for",
    )


class ProblemDetails(BaseModel):
    """RFC 9457 problem-details body for every non-2xx response.

    Standard base fields (type, title, status, detail) plus the platform
    registry's stable extensions (code, pointer, alternatives, truncated,
    issues, constraint_id, trace_id, capability_version, retryable). Optional
    members (pointer, alternatives, truncated, issues, constraint_id) are
    omitted from the wire body when absent -- callers should serialize with
    ``model_dump(mode="json", exclude_none=True)``, which ``to_json`` does.
    """

    type: str = Field(description="a relative URI reference to the code's registry entry")
    title: str = Field(description="short, human-readable summary of the problem type")
    status: int = Field(description="the HTTP status code")
    detail: str = Field(description="human-readable explanation of this occurrence")
    code: str = Field(description="the stable machine-readable code from the platform registry")
    pointer: str | None = Field(
        default=None, description="a JSON Pointer into the request body naming the field at fault"
    )
    alternatives: list[Alternative] | None = Field(
        default=None, description="values that would satisfy this request, when known"
    )
    truncated: bool | None = Field(
        default=None,
        description="present iff alternatives is present; whether the cap cut items",
    )
    issues: list[ProblemIssue] | None = Field(
        default=None,
        description="one entry per failure, for a refusal code that reports several at once",
    )
    cause: Cause | None = Field(
        default=None,
        description="why the refusal fired, for the codes that carry one -- omitted otherwise",
    )
    constraint_id: str | None = Field(
        default=None,
        description="the machine-keyable id of the constraint this refusal violates, when known",
    )
    ceiling_kind: str | None = Field(
        default=None,
        description=(
            "for over_ceiling: which ceiling was crossed -- member_cardinality, "
            "candidate_window, rows, or bytes"
        ),
    )
    count: int | None = Field(
        default=None,
        description=(
            "for over_ceiling: the observed count, in ceiling_kind's own unit, that crossed "
            "cap -- omitted only when the count itself is unmeasured, never when it is "
            "merely large. Under candidate_window it is the candidates the read actually "
            "returned, at most cap: a read that fills its window returns exactly cap, and "
            "one that stops early on a surviving continuation token returns fewer"
        ),
    )
    cap: int | None = Field(
        default=None,
        description=(
            "for over_ceiling: the ceiling, in ceiling_kind's own unit, that count crossed. "
            "Under candidate_window it is the candidate window the read was allowed"
        ),
    )
    trace_id: str = Field(
        description="this request's trace_id, shared with diagnostics and the SSE done event"
    )
    capability_version: str = Field(
        description="the deployment capability version that produced this response"
    )
    retryable: bool = Field(description="whether retrying the same request might succeed")

    def to_json(self) -> dict[str, object]:
        """This body's wire form: optional members omitted when absent (envelope rule)."""
        return self.model_dump(mode="json", exclude_none=True)


class UnregisteredCode(Exception):
    """Raised when `Envelope.details` receives an unregistered code.

    Every code rendered by an envelope must appear in its catalogue."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(
            f"code {code!r} is not registered in this envelope's own catalogue. A "
            "surface's codes catalogue must contain every code it renders through "
            "Envelope.details/.response -- use merge_catalogues to build a view "
            "that also includes the platform's own catalogue if this code belongs "
            "there instead."
        )


def title_for_status(status: int) -> str:
    """The standard HTTP reason phrase for `status`, e.g. 404 -> "Not Found"."""
    try:
        return http.HTTPStatus(status).phrase
    except ValueError:
        return "Error"


class ProblemSpec(NamedTuple):
    """A classified refusal as data, ready for `Envelope.render`.

    Surface classifiers return this or `None` to defer to the platform default."""

    status: int
    code: str
    retryable: bool
    detail: str | None = None
    pointer: str | None = None
    alternatives: list[Alternative] | None = None
    truncated: bool = False
    issues: list[ProblemIssue] | None = None
    cause: Cause | None = None
    constraint_id: str | None = None
    ceiling_kind: str | None = None
    count: int | None = None
    cap: int | None = None
    headers: Mapping[str, str] | None = None
    # What `log_problem` logs for a refusal a surface's classifier or admission check returns,
    # so both stay pure, with the classified exception's traceback when `log_traceback`. None
    # reaches the wire or `to_canonical_data`.
    log: str | None = None
    log_level: int = logging.ERROR
    log_traceback: bool = False

    def to_canonical_data(self) -> dict[str, object]:
        """Return JSON-safe fields, dumping nested models and sorting headers for stable hashes."""
        return {
            "status": self.status,
            "code": self.code,
            "retryable": self.retryable,
            "detail": self.detail,
            "pointer": self.pointer,
            "alternatives": (
                [a.model_dump(mode="json") for a in self.alternatives]
                if self.alternatives is not None
                else None
            ),
            "truncated": self.truncated,
            "issues": (
                [i.model_dump(mode="json") for i in self.issues]
                if self.issues is not None
                else None
            ),
            "cause": self.cause.model_dump(mode="json") if self.cause is not None else None,
            "constraint_id": self.constraint_id,
            "ceiling_kind": self.ceiling_kind,
            "count": self.count,
            "cap": self.cap,
            "headers": dict(sorted(self.headers.items())) if self.headers is not None else None,
        }

    @classmethod
    def from_canonical_data(cls, data: Mapping[str, object]) -> ProblemSpec:
        """Rebuild the spec `to_canonical_data` dumped, validating every field.

        The exact inverse for every key that method writes. A key it does not write, or a
        value of the wrong type, raises `pydantic.ValidationError`. An absent key takes the
        field's default, so data dumped before a field existed still loads."""
        return _PROBLEM_SPEC_ADAPTER.validate_python(data)


def _without_log_fields(data: object) -> object:
    """`data`, refused if it names a field `to_canonical_data` does not write."""
    if isinstance(data, Mapping) and data.keys() & {"log", "log_level", "log_traceback"}:
        raise ValueError("canonical problem data holds no log fields")
    return data


_PROBLEM_SPEC_ADAPTER: TypeAdapter[ProblemSpec] = TypeAdapter(
    Annotated[ProblemSpec, BeforeValidator(_without_log_fields)]
)


# The headers of a 503 that refuses a request because a pool or permit is full. A full pool
# refuses at once, and a running read or work thread frees within a request, so a client
# waits the shortest `Retry-After`, one second, before it retries.
BACKPRESSURE_HEADERS: Mapping[str, str] = MappingProxyType({"Retry-After": "1"})


def refused(
    code: str,
    *,
    retryable: bool,
    detail: str | None = None,
    pointer: str | None = None,
    alternatives: list[Alternative] | None = None,
    truncated: bool = False,
    issues: list[ProblemIssue] | None = None,
    cause: Cause | None = None,
    constraint_id: str | None = None,
    headers: Mapping[str, str] | None = None,
    log: str | None = None,
    log_level: int = logging.ERROR,
    log_traceback: bool = False,
) -> ProblemSpec:
    """`code`'s problem at the status the code registry gives it on the bound endpoint, so a
    classifier cannot state one apart from the other.

    Raises:
        LookupError: The registry maps no one status for `code` there (`registered_status`).
    """
    return ProblemSpec(
        status=registered_status(code),
        code=code,
        retryable=retryable,
        detail=detail,
        pointer=pointer,
        alternatives=alternatives,
        truncated=truncated,
        issues=issues,
        cause=cause,
        constraint_id=constraint_id,
        headers=headers,
        log=log,
        log_level=log_level,
        log_traceback=log_traceback,
    )


def log_problem(
    spec: ProblemSpec,
    exc: BaseException | None,
    *,
    logger: logging.Logger,
    context: str | None,
) -> None:
    """Log the line `spec` names (`ProblemSpec.log`) at its level, after `context` when one is
    given, with `exc`'s traceback when `spec.log_traceback`. A spec naming no line logs
    nothing. The request path calls this once per classified request, and a reader that
    classifies outside a request calls it itself."""
    if spec.log is None:
        return
    exc_info = exc if spec.log_traceback else None
    if context is None:
        logger.log(spec.log_level, "%s", spec.log, exc_info=exc_info)
    else:
        logger.log(spec.log_level, "%s: %s", context, spec.log, exc_info=exc_info)


class Refusal(Exception):
    """A refusal a surface renders as a problem from the attributes it carries.

    A subclass sets `code` and, when a retry can succeed, `retryable`. Every other member is
    absent by default, and an instance sets the ones it carries. `detail` falls back to the
    exception's message. The status is the code registry's for `code` on the endpoint the
    request's route serves (`registered_status`).
    `strip_pointer` removes the top-level pointer and every issue's pointer, for a route whose
    request has no body the pointers could name. A refusal with a `cause` extension renders
    through its surface's own classifier, since the materialise conversion refusals name their
    cause as a bare code.
    """

    code: str
    retryable: bool = False
    detail: str | None = None
    pointer: str | None = None
    alternatives: list[Alternative] | None = None
    truncated: bool = False
    issues: Sequence[ProblemIssue] | None = None
    constraint_id: str | None = None
    ceiling_kind: str | None = None
    count: int | None = None
    cap: int | None = None
    headers: Mapping[str, str] | None = None
    strip_pointer: bool = False


def classify_refusal(exc: BaseException) -> ProblemSpec | None:
    """Render a `Refusal` from its attributes, naming a log line with its traceback for a
    server-side one a retry cannot clear. `count` and `cap` belong to the ceiling extension and
    render only beside `ceiling_kind`. Return `None` for any other exception."""
    if not isinstance(exc, Refusal):
        return None
    status = registered_status(exc.code)
    server_fault = status >= http.HTTPStatus.INTERNAL_SERVER_ERROR and not exc.retryable
    pointer, issues = exc.pointer, list(exc.issues) if exc.issues else None
    if exc.strip_pointer:
        pointer = None
        issues = [issue.model_copy(update={"pointer": None}) for issue in issues or ()] or None
    ceiling = exc.ceiling_kind is not None
    return ProblemSpec(
        status=status,
        code=exc.code,
        retryable=exc.retryable,
        detail=exc.detail or str(exc),
        pointer=pointer,
        alternatives=exc.alternatives,
        truncated=exc.truncated,
        issues=issues,
        constraint_id=exc.constraint_id,
        ceiling_kind=exc.ceiling_kind,
        count=exc.count if ceiling else None,
        cap=exc.cap if ceiling else None,
        headers=exc.headers,
        log=f"Refusal {exc.code}: {exc}" if server_fault else None,
        log_traceback=server_fault,
    )


@dataclass(frozen=True)
class Envelope:
    """Build consistent problem-details bodies from a platform or surface catalogue."""

    catalogue: CodeCatalogue
    capability_version: str

    def title(self, code: str, status: int) -> str:
        """Return the catalogue title, or the standard HTTP phrase when absent."""
        entry = self.catalogue.get(code)
        return entry[0] if entry is not None else title_for_status(status)

    def detail(self, code: str, detail: str | None) -> str:
        """Use the supplied detail, catalogue default, or generic fallback, in that order."""
        if detail is not None:
            return detail
        entry = self.catalogue.get(code)
        return entry[1] if entry is not None else "Bad request"

    def details(
        self,
        *,
        status: int,
        code: str,
        detail: str | None = None,
        pointer: str | None = None,
        alternatives: list[Alternative] | None = None,
        truncated: bool = False,
        issues: list[ProblemIssue] | None = None,
        cause: Cause | None = None,
        constraint_id: str | None = None,
        ceiling_kind: str | None = None,
        count: int | None = None,
        cap: int | None = None,
        retryable: bool,
    ) -> ProblemDetails:
        """Build one problem body. Raise `UnregisteredCode` for unknown codes.

        `truncated` is included only when `alternatives` is present."""
        if code not in self.catalogue:
            raise UnregisteredCode(code)
        return ProblemDetails(
            type=f"/codes/{code}",
            title=self.title(code, status),
            status=status,
            detail=self.detail(code, detail),
            code=code,
            pointer=pointer,
            alternatives=alternatives,
            truncated=(truncated if alternatives is not None else None),
            issues=issues,
            cause=cause,
            constraint_id=constraint_id,
            ceiling_kind=ceiling_kind,
            count=count,
            cap=cap,
            trace_id=get_trace_id(),
            capability_version=self.capability_version,
            retryable=retryable,
        )

    def response(
        self,
        *,
        status: int,
        code: str,
        detail: str | None = None,
        pointer: str | None = None,
        alternatives: list[Alternative] | None = None,
        truncated: bool = False,
        issues: list[ProblemIssue] | None = None,
        cause: Cause | None = None,
        constraint_id: str | None = None,
        ceiling_kind: str | None = None,
        count: int | None = None,
        cap: int | None = None,
        retryable: bool,
        headers: Mapping[str, str] | None = None,
    ) -> JSONResponse:
        """Return `details()` as the non-2xx JSON response.

        Set `Cache-Control: no-store` because refusal statuses can be heuristically
        cacheable. Caller headers such as `Allow` are preserved. Any caller cache-control
        value is removed case-insensitively before setting `no-store`."""
        return self._problem_response(
            self.details(
                status=status,
                code=code,
                detail=detail,
                pointer=pointer,
                alternatives=alternatives,
                truncated=truncated,
                issues=issues,
                cause=cause,
                constraint_id=constraint_id,
                ceiling_kind=ceiling_kind,
                count=count,
                cap=cap,
                retryable=retryable,
            ),
            headers=headers,
        )

    def _problem_response(
        self, problem: ProblemDetails, *, headers: Mapping[str, str] | None
    ) -> JSONResponse:
        """`problem` as the non-2xx JSON response `response` documents."""
        from fastapi.responses import JSONResponse

        merged_headers = {
            key: value for key, value in (headers or {}).items() if key.lower() != "cache-control"
        }
        merged_headers["Cache-Control"] = "no-store"
        return JSONResponse(
            status_code=problem.status,
            content=problem.to_json(),
            media_type=PROBLEM_JSON,
            headers=merged_headers,
        )

    def problem(self, spec: ProblemSpec) -> ProblemDetails:
        """The body of a classified refusal, every field of `spec`, which a JSON failure and
        an SSE terminal error both carry."""
        return self.details(
            status=spec.status,
            code=spec.code,
            detail=spec.detail,
            pointer=spec.pointer,
            alternatives=spec.alternatives,
            truncated=spec.truncated,
            issues=spec.issues,
            cause=spec.cause,
            constraint_id=spec.constraint_id,
            ceiling_kind=spec.ceiling_kind,
            count=spec.count,
            cap=spec.cap,
            retryable=spec.retryable,
        )

    def render(self, spec: ProblemSpec) -> JSONResponse:
        """Render a classified refusal as the non-2xx JSON response."""
        return self._problem_response(self.problem(spec), headers=spec.headers)
