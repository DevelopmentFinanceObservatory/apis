"""The `skills_` capability token adds 16 SHA-256 hex characters over wire-visible refusal
bodies, the warning catalogue, and response schemas. `strip_schema_annotations` removes
`description` and `title` annotations and keeps fields of those names. The token omits index
limits because the Skills API publishes no capability document. Inputs are injectable so tests
can prove each one changes the digest.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Mapping

from dfo_platform import Alternative, CodeCatalogue, Envelope, HealthResponse, ProblemIssue
from dfo_platform.schema_annotations import strip_schema_annotations
from dfo_platform.trace import set_trace_id
from skills_api.codes import SKILLS_CODE_CATALOGUE, SKILLS_WARNING_CATALOGUE
from skills_api.models import SkillBody, SkillListResponse, SkillResourceContent

# Stable trace id keeps refusal-body hashes deterministic across runs.
_EXEMPLAR_TRACE_ID = "0" * 32

# Placeholder avoids hashing a token that depends on itself.
_EXEMPLAR_CAPABILITY_VERSION = "skills_exemplar_placeholder"


def _render_body_exemplars(catalogue: CodeCatalogue) -> dict[str, dict[str, object]]:
    """Render one problem body per code. The first sorted code fills every optional field.

    Call in a fresh context with `_EXEMPLAR_TRACE_ID` bound for a stable digest, leaving the
    caller's context unchanged.
    """
    set_trace_id(_EXEMPLAR_TRACE_ID)
    envelope = Envelope(catalogue=catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    sorted_codes = sorted(catalogue)
    exemplars: dict[str, dict[str, object]] = {}
    for index, code in enumerate(sorted_codes):
        if index == 0:
            problem = envelope.details(
                status=404,
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
            problem = envelope.details(status=404, code=code, retryable=False)
        exemplars[code] = problem.to_json()
    return exemplars


def served_wire_schemas() -> dict[str, dict[str, object]]:
    """The JSON Schema of each response body, keyed by name, in serialization mode, which
    describes the body the server emits and lists every field as required."""
    return {
        "health_response": HealthResponse.model_json_schema(mode="serialization"),
        "skill_list_response": SkillListResponse.model_json_schema(mode="serialization"),
        "skill_body": SkillBody.model_json_schema(mode="serialization"),
        "skill_resource_content": SkillResourceContent.model_json_schema(mode="serialization"),
    }


def _compute_skills_capability_version(
    *,
    code_catalogue: CodeCatalogue = SKILLS_CODE_CATALOGUE,
    warning_catalogue: Mapping[str, tuple[str, str, str]] = SKILLS_WARNING_CATALOGUE,
    wire_schemas: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """Return the deterministic hash of the wire-visible inputs."""
    if wire_schemas is None:
        wire_schemas = served_wire_schemas()

    payload = {
        "refusal_body_exemplars": contextvars.copy_context().run(
            _render_body_exemplars, code_catalogue
        ),
        "warning_catalogue": {
            code: list(entry) for code, entry in sorted(warning_catalogue.items())
        },
        "wire_schemas": {
            name: strip_schema_annotations(schema) for name, schema in sorted(wire_schemas.items())
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"skills_{digest}"


# This surface's own `capability_version`, stamped on every skills_api
# response (see `dfo_platform.app.create_app`'s per-surface envelope).
SKILLS_CAPABILITY_VERSION: str = _compute_skills_capability_version()
