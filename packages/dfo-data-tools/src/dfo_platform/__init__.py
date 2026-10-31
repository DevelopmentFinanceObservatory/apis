"""FastAPI-free exports for shared HTTP contract types and helpers.

App composition, capability versions, static responses, and configuration live
in their own `dfo_platform` submodules.
"""

from __future__ import annotations

from dfo_platform.codes import (
    PLATFORM_CODE_CATALOGUE,
    CodeCatalogue,
    CodeCollision,
    merge_catalogues,
)
from dfo_platform.diagnostics import WarningSeverity
from dfo_platform.health import HealthResponse, health_response
from dfo_platform.problem import (
    Alternative,
    Cause,
    Envelope,
    ProblemDetails,
    ProblemIssue,
    ProblemSpec,
    UnregisteredCode,
    refused,
    title_for_status,
)
from dfo_platform.roles import RoleAxis
from dfo_platform.trace import (
    TRACE_ID_HEADER,
    get_trace_id,
    new_trace_id,
    reset_trace_id,
    set_trace_id,
)
from dfo_platform.validation import (
    error_field_loc,
    pointer_for_error,
    reset_request_body_model,
    set_request_body_model,
    validation_problem,
)

__all__ = [
    "PLATFORM_CODE_CATALOGUE",
    "TRACE_ID_HEADER",
    "Alternative",
    "Cause",
    "CodeCatalogue",
    "CodeCollision",
    "Envelope",
    "HealthResponse",
    "ProblemDetails",
    "ProblemIssue",
    "ProblemSpec",
    "RoleAxis",
    "UnregisteredCode",
    "WarningSeverity",
    "error_field_loc",
    "get_trace_id",
    "health_response",
    "merge_catalogues",
    "new_trace_id",
    "pointer_for_error",
    "refused",
    "reset_request_body_model",
    "reset_trace_id",
    "set_request_body_model",
    "set_trace_id",
    "title_for_status",
    "validation_problem",
]
