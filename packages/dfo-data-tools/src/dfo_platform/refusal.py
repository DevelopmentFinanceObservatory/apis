"""Platform refusal policy shared by `dfo_platform.app` and `dfo_platform.version`.

`classify_default` maps exceptions to `ProblemSpec`. Renderers build fixed outcomes. The
version hashes policy data and representative rendered responses. This module stays below
`app` and `version` in the import graph.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from dfo_platform.problem import Envelope, ProblemSpec, title_for_status
from dfo_platform.validation import validation_problem

if TYPE_CHECKING:
    from fastapi.responses import JSONResponse

_logger = logging.getLogger(__name__)

# Routing-generated 404s and 405s have dedicated codes.
STATUS_CODE_MAP: dict[int, str] = {
    404: "not_found",
    405: "method_not_allowed",
}

# Unmapped 4xx statuses use the client fallback. All others use the server fallback.
CLIENT_ERROR_FALLBACK_CODE = "request_invalid"
SERVER_ERROR_FALLBACK_CODE = "internal_error"
CLIENT_ERROR_STATUS_LOW = 400
CLIENT_SERVER_BOUNDARY = 500


def code_for_status(status: int) -> str:
    """The refusal code for a hand-raised or routing-generated
    StarletteHTTPException at `status`: `STATUS_CODE_MAP`'s own entry when
    one exists, else the client/server fallback split at
    `CLIENT_SERVER_BOUNDARY`.
    """
    if status in STATUS_CODE_MAP:
        return STATUS_CODE_MAP[status]
    if CLIENT_ERROR_STATUS_LOW <= status < CLIENT_SERVER_BOUNDARY:
        return CLIENT_ERROR_FALLBACK_CODE
    return SERVER_ERROR_FALLBACK_CODE


@dataclass(frozen=True)
class RefusalDefault:
    """Fixed status, code, and retryability for a platform-default refusal."""

    status: int
    code: str
    retryable: bool


TIMEOUT_DEFAULT = RefusalDefault(status=504, code="timeout", retryable=True)
UNHANDLED_DEFAULT = RefusalDefault(status=500, code="internal_error", retryable=False)

# HTTPException status and code come from the raise site.
# Its retryable value is false.
HTTP_EXCEPTION_DEFAULT_RETRYABLE = False


@dataclass(frozen=True)
class CodeNotFoundDefault:
    """Fixed outcome for an unknown `GET /codes/{code}` request.

    `detail_template` is formatted with the requested code when rendered. This response carries
    `PLATFORM_CAPABILITY_VERSION`, so every field here, template included, feeds that hash."""

    status: int
    code: str
    retryable: bool
    detail_template: str


CODE_NOT_FOUND_DEFAULT = CodeNotFoundDefault(
    status=404,
    code="code_not_found",
    retryable=False,
    detail_template="No registry entry exists for code {code!r}.",
)


def as_canonical_data() -> dict[str, object]:
    """Return JSON-safe policy data for `PLATFORM_CAPABILITY_VERSION` hashing.

    Integer status keys are converted to strings for JSON."""
    return {
        "status_code_map": {str(status): code for status, code in STATUS_CODE_MAP.items()},
        "client_error_fallback_code": CLIENT_ERROR_FALLBACK_CODE,
        "server_error_fallback_code": SERVER_ERROR_FALLBACK_CODE,
        "client_error_status_low": CLIENT_ERROR_STATUS_LOW,
        "client_server_boundary": CLIENT_SERVER_BOUNDARY,
        "timeout_default": {
            "status": TIMEOUT_DEFAULT.status,
            "code": TIMEOUT_DEFAULT.code,
            "retryable": TIMEOUT_DEFAULT.retryable,
        },
        "unhandled_default": {
            "status": UNHANDLED_DEFAULT.status,
            "code": UNHANDLED_DEFAULT.code,
            "retryable": UNHANDLED_DEFAULT.retryable,
        },
        "http_exception_default_retryable": HTTP_EXCEPTION_DEFAULT_RETRYABLE,
        "code_not_found_default": {
            "status": CODE_NOT_FOUND_DEFAULT.status,
            "code": CODE_NOT_FOUND_DEFAULT.code,
            "retryable": CODE_NOT_FOUND_DEFAULT.retryable,
            "detail_template": CODE_NOT_FOUND_DEFAULT.detail_template,
        },
    }


def render_timeout_refusal() -> ProblemSpec:
    """Return the platform's default timeout refusal."""
    return ProblemSpec(
        status=TIMEOUT_DEFAULT.status,
        code=TIMEOUT_DEFAULT.code,
        retryable=TIMEOUT_DEFAULT.retryable,
    )


def render_unhandled_refusal() -> ProblemSpec:
    """Return the catch-all refusal. `classify_default` logs exception details."""
    return ProblemSpec(
        status=UNHANDLED_DEFAULT.status,
        code=UNHANDLED_DEFAULT.code,
        retryable=UNHANDLED_DEFAULT.retryable,
    )


def render_code_not_found_refusal(envelope: Envelope, *, code: str) -> JSONResponse:
    """Render the unknown-code refusal, inserting `code` into its detail."""
    return envelope.response(
        status=CODE_NOT_FOUND_DEFAULT.status,
        code=CODE_NOT_FOUND_DEFAULT.code,
        detail=CODE_NOT_FOUND_DEFAULT.detail_template.format(code=code),
        retryable=CODE_NOT_FOUND_DEFAULT.retryable,
    )


def render_http_exception_refusal(
    *,
    status: int,
    detail: object,
    headers: Mapping[str, str] | None,
) -> ProblemSpec:
    """Map an HTTP exception to `ProblemSpec`, preserving its headers.

    Only a string detail distinct from the HTTP reason phrase reaches the wire. FastAPI
    permits structured details, but `ProblemDetails.detail` is a string, so other values
    use the catalogue default."""
    code = code_for_status(status)
    rendered_detail = (
        detail if isinstance(detail, str) and detail != title_for_status(status) else None
    )
    return ProblemSpec(
        status=status,
        code=code,
        detail=rendered_detail,
        retryable=HTTP_EXCEPTION_DEFAULT_RETRYABLE,
        # Routing supplies `Allow` for 405s. Raised exceptions may supply other headers.
        headers=headers,
    )


def default_problem(exc: BaseException) -> ProblemSpec:
    """The refusal for validation, timeout, and HTTP errors, and the catch-all for any other.
    It logs nothing."""
    if isinstance(exc, RequestValidationError):
        return validation_problem(exc)
    if isinstance(exc, asyncio.TimeoutError):
        return render_timeout_refusal()
    if isinstance(exc, StarletteHTTPException):
        return render_http_exception_refusal(
            status=exc.status_code, detail=exc.detail, headers=exc.headers
        )
    return render_unhandled_refusal()


def classify_default(exc: BaseException) -> ProblemSpec:
    """`default_problem`, logging a timeout and any exception the catch-all takes.

    Surface classifiers use this when they return `None`."""
    if isinstance(exc, asyncio.TimeoutError):
        _logger.error("Request timed out")
    elif not isinstance(exc, _DEFAULT_EXCEPTION_TYPES):
        _logger.error("Unhandled platform error: %s", exc, exc_info=exc)
    return default_problem(exc)


_DEFAULT_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    RequestValidationError,
    asyncio.TimeoutError,
    StarletteHTTPException,
)

# One exemplar per classification branch, plus one for the catch-all.
_DEFAULT_EXEMPLARS: tuple[BaseException, ...] = (
    RequestValidationError(
        errors=[
            {
                "type": "string_type",
                "loc": ("body", "exemplar_field"),
                "msg": "Input should be a valid string",
                "input": None,
            }
        ],
        body={"exemplar_field": None},
    ),
    RequestValidationError(
        errors=[
            {
                "type": "missing",
                "loc": ("query", "exemplar_query_param"),
                "msg": "Field required",
                "input": None,
            }
        ],
        body=None,
    ),
    RequestValidationError(
        errors=[
            {
                "type": "json_invalid",
                "loc": ("body", "exemplar_field"),
                "msg": "JSON decode error",
                "input": "not valid json",
            }
        ],
        body="not valid json",
    ),
    RequestValidationError(
        errors=[
            {
                "type": "literal_error",
                "loc": ("body", "schema_version"),
                "msg": "Input should be '1'",
                "input": "99",
            }
        ],
        body={"schema_version": "99"},
    ),
    TimeoutError(),
    StarletteHTTPException(status_code=404, detail="Not Found", headers=None),
    StarletteHTTPException(status_code=405, detail=None, headers={"Allow": "GET"}),
    StarletteHTTPException(
        status_code=405,
        detail=None,
        headers={"Allow": "GET, POST", "cache-control": "public, max-age=60"},
    ),
    StarletteHTTPException(status_code=403, detail="You may not do that here.", headers=None),
    StarletteHTTPException(status_code=301, detail=None, headers=None),
    RuntimeError("exemplar unhandled error"),
)
