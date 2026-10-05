"""`compute_presentation_capability_version` returns `pres_` plus 16 SHA-256 hex digits for
the release this surface serves. The hash covers the release document with a fixed
`catalog_version`, strict intent schemas, formatter profile, grammar members, encoding classes,
constraints, rendered refusal exemplars, and frame-unavailable outcomes. Wording in the release
schemas affects the hash.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Mapping

from assembler.catalogue import CatalogueRelease
from dfo_platform import Alternative, CodeCatalogue, Envelope, ProblemIssue
from dfo_platform.trace import set_trace_id
from presentation_api.codes import CODE_RETRY_MODE, PRESENTATION_CODE_CATALOGUE
from presentation_api.doors.release import release_document
from presentation_api.refusals import _FRAME_UNAVAILABLE_OUTCOME

__all__ = ["compute_presentation_capability_version"]

# Fixed values keep exemplar hashes independent of request context and the computed digest.
_EXEMPLAR_TRACE_ID = "0" * 32
_EXEMPLAR_CAPABILITY_VERSION = "pres_exemplar_placeholder"
_EXEMPLAR_CATALOG_VERSION = "cv_exemplar_placeholder"


def _render_refusal_exemplars(
    code_catalogue: CodeCatalogue, retry_modes: Mapping[str, str]
) -> dict[str, dict[str, object]]:
    """Render one sorted `ProblemDetails` body per code. The first includes every optional
    field, so removing a field changes the hash. Run in a fresh context with the fixed trace ID
    to keep results deterministic without changing the caller's context.
    """
    set_trace_id(_EXEMPLAR_TRACE_ID)
    envelope = Envelope(catalogue=code_catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    sorted_codes = sorted(code_catalogue)
    exemplars: dict[str, dict[str, object]] = {}
    for index, code in enumerate(sorted_codes):
        retryable = retry_modes[code] != "do_not_retry"
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
                    )
                ],
                retryable=retryable,
            )
        else:
            problem = envelope.details(status=400, code=code, retryable=retryable)
        exemplars[code] = problem.to_json()
    return exemplars


def compute_presentation_capability_version(
    *,
    release: CatalogueRelease,
    code_catalogue: CodeCatalogue = PRESENTATION_CODE_CATALOGUE,
    retry_modes: Mapping[str, str] = CODE_RETRY_MODE,
    frame_unavailable_outcome: Mapping[str, tuple[int, bool]] = _FRAME_UNAVAILABLE_OUTCOME,
) -> str:
    """Return the capability hash for `release` and the supplied refusal data."""
    refusal_exemplars = contextvars.copy_context().run(
        _render_refusal_exemplars, code_catalogue, retry_modes
    )
    pinned_release_document = {
        **release_document(release, retry_modes=retry_modes),
        "catalog_version": _EXEMPLAR_CATALOG_VERSION,
    }
    payload = {
        "intent_grammar": dict(release.intent_schemas),
        "formatter_profile": dict(release.formatter_profile),
        "grammar_members": sorted(release.grammar_members),
        "encoding_classes": dict(release.encoding_classes),
        "constraints": sorted(
            (spec.constraint_id, list(spec.encoding_classes), list(spec.component_types))
            for spec in release.constraints
        ),
        "refusal_exemplars": refusal_exemplars,
        "frame_unavailable_outcome": {
            reason: list(outcome) for reason, outcome in frame_unavailable_outcome.items()
        },
        "release_document": pinned_release_document,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"pres_{digest}"
