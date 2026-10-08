"""Fingerprint of platform-owned responses, separate from each surface's token.

The hash covers `refusal.as_canonical_data()`, the wire-visible `ProblemDetails` schema,
`PLATFORM_CODE_CATALOGUE`, and representative rendered outcomes from `classify_default`,
HTTP-exception handling, unknown-code refusals, and successful code lookups. Policy data covers
unsampled constants. Rendered outcomes cover changes to dispatch and response logic.

The sample matrix is representative. A new case outside it can change wire behaviour without
changing this token.

The token excludes field descriptions, mounted-surface selection, app-level exception dispatch,
and surface-owned exception exemplars. Those inputs belong to schema docs, app composition, or a
surface's own token. `app` imports this module's token, so including app-level dispatch here would
create a cycle. The platform token covers code only and has no environment override.

`HTTPStatus` reason phrases depend on the Python version, so the runtime and dependencies are
pinned."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
from collections.abc import Mapping

from dfo_spec_kit.registry import REGISTRY
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from dfo_platform import codes, refusal
from dfo_platform.codes import PLATFORM_CODE_CATALOGUE, CodeCatalogue
from dfo_platform.problem import Alternative, Envelope, ProblemDetails, ProblemIssue
from dfo_platform.trace import set_trace_id


def _strip_wire_invisible_keys(value: object) -> object:
    """Recursively remove schema-only `description` and `title` keys before hashing.

    `ProblemDetails.to_json()` sends field values. Property keys and nested `$defs` retain schema
    identity."""
    if isinstance(value, dict):
        return {
            key: _strip_wire_invisible_keys(val)
            for key, val in value.items()
            if key not in ("description", "title")
        }
    if isinstance(value, list):
        return [_strip_wire_invisible_keys(item) for item in value]
    return value


# Fixed exemplar values keep hashes independent of request context and the token being computed.
_EXEMPLAR_TRACE_ID = "0" * 32
_EXEMPLAR_CAPABILITY_VERSION = "plat_exemplar_placeholder"


def _render_body_exemplars(catalogue: CodeCatalogue) -> dict[str, dict[str, object]]:
    """Render every catalogue code, populating all optional fields on the first sorted code.

    This adds their wire names to the hash. Other codes render bare."""
    envelope = Envelope(catalogue=catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    sorted_codes = sorted(catalogue)
    exemplars: dict[str, dict[str, object]] = {}
    for index, code in enumerate(sorted_codes):
        if index == 0:
            problem = envelope.details(
                status=400,
                code=code,
                detail="exemplar detail",
                pointer="/exemplar/pointer",
                alternatives=[Alternative(ref="exemplar-ref", label="exemplar-label")],
                truncated=True,
                issues=[
                    ProblemIssue(
                        pointer="/exemplar/issue-pointer",
                        member="exemplar-member",
                        axis="exemplar-axis",
                        alternatives=[Alternative(ref="issue-ref", label="issue-label")],
                        truncated=False,
                        code=code,
                        detail="exemplar issue detail",
                    )
                ],
                retryable=True,
            )
        else:
            problem = envelope.details(status=400, code=code, retryable=False)
        exemplars[code] = problem.to_json()
    return exemplars


def _response_headers_digest_input(response: JSONResponse) -> dict[str, str]:
    """Return lowercase response headers for hashing, omitting derived `content-length`."""
    return {
        key.lower(): value
        for key, value in sorted(response.headers.items())
        if key.lower() != "content-length"
    }


def _rendered_response_digest_input(response: JSONResponse) -> dict[str, object]:
    """Return the wire body and canonical headers used by response exemplars."""
    return {
        "body": json.loads(bytes(response.body)),
        "headers": _response_headers_digest_input(response),
    }


def _render_response_exemplars(catalogue: CodeCatalogue) -> list[dict[str, str]]:
    """Hash headers from bare and merged responses, including `Allow` and conflicting
    cache-control.

    This pins `Envelope.response`'s header merge and `no-store` precedence."""
    envelope = Envelope(catalogue=catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    bare = envelope.response(status=400, code="request_invalid", retryable=False)
    merged = envelope.response(
        status=405,
        code="method_not_allowed",
        retryable=False,
        headers={"Allow": "GET, POST", "cache-control": "public, max-age=60"},
    )
    return [_response_headers_digest_input(bare), _response_headers_digest_input(merged)]


def _render_code_not_found_exemplar(catalogue: CodeCatalogue) -> dict[str, object]:
    """Render the unknown-code response through the same renderer used by the route."""
    envelope = Envelope(catalogue=catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    return _rendered_response_digest_input(
        refusal.render_code_not_found_refusal(envelope, code="exemplar_unknown_code")
    )


# These samples call the renderer with `None`, dict, and list details.
# Starlette converts `None` to the status phrase. FastAPI accepts structured details at runtime,
# though `StarletteHTTPException.detail` is annotated `str | None`.
_HTTP_EXCEPTION_DIRECT_CALL_INPUTS: tuple[tuple[int, object, Mapping[str, str] | None], ...] = (
    (503, None, None),  # detail absent
    (403, {"reason": "denied"}, None),  # structured dict detail
    (422, ["a", "b"], None),  # structured list detail
)


def _render_http_exception_exemplars(catalogue: CodeCatalogue) -> list[dict[str, object]]:
    """Render HTTP-exception edge cases with details passed straight to the renderer."""
    envelope = Envelope(catalogue=catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    exemplars: list[dict[str, object]] = []
    for status, detail, headers in _HTTP_EXCEPTION_DIRECT_CALL_INPUTS:
        spec = refusal.render_http_exception_refusal(status=status, detail=detail, headers=headers)
        exemplars.append(_rendered_response_digest_input(envelope.render(spec)))
    return exemplars


def _render_default_classification_exemplars(catalogue: CodeCatalogue) -> list[dict[str, object]]:
    """Render default exception exemplars through `classify_default` to pin its dispatch."""
    envelope = Envelope(catalogue=catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    return [
        _rendered_response_digest_input(envelope.render(refusal.classify_default(exc)))
        for exc in refusal._DEFAULT_EXEMPLARS
    ]


class _SyntheticValidationTimeoutException(RequestValidationError, asyncio.TimeoutError):
    """Exercise validation-before-timeout dispatch. Only validation state is required."""

    def __init__(self) -> None:
        RequestValidationError.__init__(
            self,
            errors=[
                {
                    "type": "missing",
                    "loc": ("body", "field"),
                    "msg": "Field required",
                    "input": None,
                }
            ],
            body={"field": None},
        )


class _SyntheticValidationHttpException(RequestValidationError, StarletteHTTPException):
    """Exercises validation-before-HTTP dispatch with both exceptions' required state."""

    def __init__(self) -> None:
        RequestValidationError.__init__(
            self,
            errors=[
                {
                    "type": "missing",
                    "loc": ("body", "field"),
                    "msg": "Field required",
                    "input": None,
                }
            ],
            body={"field": None},
        )
        StarletteHTTPException.__init__(
            self, status_code=422, detail="synthetic dispatch-order fixture"
        )


class _SyntheticTimeoutHttpException(asyncio.TimeoutError, StarletteHTTPException):
    """Exercises timeout-before-HTTP dispatch with HTTP exception state available."""

    def __init__(self) -> None:
        StarletteHTTPException.__init__(
            self, status_code=504, detail="synthetic dispatch-order fixture"
        )


# One fixture per pair of classifier branches pins their `isinstance` order.
# TimeoutError constructors take no arguments because Python 3.11 aliases it to the built-in
# OSError, whose `__new__` dispatches on errno-shaped positional arguments.
_DISPATCH_ORDER_EXEMPLARS: tuple[BaseException, ...] = (
    _SyntheticValidationTimeoutException(),
    _SyntheticValidationHttpException(),
    _SyntheticTimeoutHttpException(),
)


def _render_dispatch_order_exemplars(catalogue: CodeCatalogue) -> list[dict[str, object]]:
    """Render mixed-exception fixtures through `classify_default` to pin branch order."""
    envelope = Envelope(catalogue=catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    return [
        _rendered_response_digest_input(envelope.render(refusal.classify_default(exc)))
        for exc in _DISPATCH_ORDER_EXEMPLARS
    ]


def _sorted_first(class_name: str, class_codes: list[str]) -> str:
    """Return the first sorted code, or raise an assertion naming an empty registry class."""
    assert class_codes, f"no registry code belongs to the '{class_name}' class any more"
    return sorted(class_codes)[0]


def _render_registered_code_exemplar(catalogue: CodeCatalogue) -> dict[str, object]:
    """Render a code present in both sources, populating catalogue and registry fields."""
    code = _sorted_first("catalogued and registered", [c for c in catalogue if c in REGISTRY])
    return codes.registry_entry_body(
        code=code, registry_entry=REGISTRY[code], catalogue_entry=catalogue[code]
    ).model_dump()


def _render_warning_code_exemplar(catalogue: CodeCatalogue) -> dict[str, object]:
    """Render a warning code absent from the catalogue, so title and detail are null."""
    code = _sorted_first(
        "warning", [entry.code for entry in REGISTRY if entry.classification == "warning"]
    )
    return codes.registry_entry_body(
        code=code, registry_entry=REGISTRY[code], catalogue_entry=catalogue.get(code)
    ).model_dump()


def _render_no_data_reason_refusal_code_exemplar(catalogue: CodeCatalogue) -> dict[str, object]:
    """Render a refusal code used as a no-data reason, with no problem-details mapping."""
    code = _sorted_first(
        "refusal with no problem_details channel",
        [
            entry.code
            for entry in REGISTRY
            if entry.classification == "refusal"
            and not any(mapping.channel == "problem_details" for mapping in entry.mappings)
        ],
    )
    return codes.registry_entry_body(
        code=code, registry_entry=REGISTRY[code], catalogue_entry=catalogue.get(code)
    ).model_dump()


def _render_multi_channel_code_exemplar(catalogue: CodeCatalogue) -> dict[str, object]:
    """Render a code with mappings across multiple channels."""
    code = _sorted_first(
        "more than one channel",
        [entry.code for entry in REGISTRY if len({m.channel for m in entry.mappings}) > 1],
    )
    return codes.registry_entry_body(
        code=code, registry_entry=REGISTRY[code], catalogue_entry=catalogue.get(code)
    ).model_dump()


def _render_unregistered_code_exemplar(catalogue: CodeCatalogue) -> dict[str, object]:
    """Render a catalogue-only entry with a synthetic unregistered code."""
    first_catalogue_code = sorted(catalogue)[0]
    return codes.registry_entry_body(
        code="exemplar_unregistered_code",
        registry_entry=None,
        catalogue_entry=catalogue[first_catalogue_code],
    ).model_dump()


def _render_codes_success_body_exemplars(catalogue: CodeCatalogue) -> dict[str, object]:
    """Render combined, registry-only, and catalogue-only successful lookup bodies."""
    return {
        "registered": _render_registered_code_exemplar(catalogue),
        "warning": _render_warning_code_exemplar(catalogue),
        "no_data_reason_refusal": _render_no_data_reason_refusal_code_exemplar(catalogue),
        "multi_channel": _render_multi_channel_code_exemplar(catalogue),
        "unregistered": _render_unregistered_code_exemplar(catalogue),
    }


def _render_exemplars(catalogue: CodeCatalogue) -> dict[str, object]:
    """Render all exemplars in an isolated context with refusal logs suppressed.

    Synthetic trace IDs keep hashes stable. Restore the original logger level afterward."""
    set_trace_id(_EXEMPLAR_TRACE_ID)
    refusal_logger = logging.getLogger(refusal.__name__)
    previous_level = refusal_logger.level
    refusal_logger.setLevel(logging.CRITICAL + 1)
    try:
        return {
            "body_exemplars": _render_body_exemplars(catalogue),
            "response_header_exemplars": _render_response_exemplars(catalogue),
            "code_not_found_exemplar": _render_code_not_found_exemplar(catalogue),
            "default_classification_exemplars": _render_default_classification_exemplars(catalogue),
            "dispatch_order_exemplars": _render_dispatch_order_exemplars(catalogue),
            "http_exception_direct_call_exemplars": _render_http_exception_exemplars(catalogue),
            "codes_success_body_exemplars": _render_codes_success_body_exemplars(catalogue),
        }
    finally:
        refusal_logger.setLevel(previous_level)


def _compute_platform_capability_version(
    *,
    problem_details_schema: Mapping[str, object] | None = None,
    platform_codes: CodeCatalogue = PLATFORM_CODE_CATALOGUE,
    refusal_policy: Mapping[str, object] | None = None,
) -> str:
    """Return the deterministic token for the platform schema, codes, policy, and outcomes.

    Inputs default to live platform data and can be overridden to test that each moves the
    digest. Render exemplars in a copied context so their trace ID cannot leak to the caller.
    Convert catalogue tuples to lists before canonical JSON serialization for stable hashing.
    The `plat_` prefix distinguishes this token from surface `cap_` tokens and schema versions."""
    if problem_details_schema is None:
        problem_details_schema = ProblemDetails.model_json_schema()
    if refusal_policy is None:
        refusal_policy = refusal.as_canonical_data()

    exemplars = contextvars.copy_context().run(_render_exemplars, platform_codes)

    payload = {
        "problem_details_schema": _strip_wire_invisible_keys(problem_details_schema),
        "platform_codes": {code: list(entry) for code, entry in platform_codes.items()},
        "refusal_policy": refusal_policy,
        **exemplars,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"plat_{digest}"


# Token for `/codes/{code}` refusals and platform-default exception responses.
PLATFORM_CAPABILITY_VERSION: str = _compute_platform_capability_version()
