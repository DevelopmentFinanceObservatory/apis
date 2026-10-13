"""`graph_` plus 16 SHA-256 hex digits over canonical JSON.
The response determines the token, so an environment variable cannot misstate it.


The hash covers three inputs:

1. The capability document with `capability_version`, `build` and
   `data_builds` pinned. Published limits and door liveness affect the
   token. Rebuilds with unchanged values, and new data builds, keep the
   token stable.
2. Rendered refusal bodies for every code in `GRAPH_CODE_CATALOGUE`.
3. The schemas `GET /api/graph/schema` serves
   (`graph_api.doors.schema.served_wire_schemas`), sorted by name, with
   `description` and `title` annotations removed by
   `strip_schema_annotations`. Field names, types,
   `enum`, `const` and `default` values affect the token; annotation prose
   does not.

Inputs are injectable for perturbation tests. Capability values are hashed
as served, so deployments with different limits have different tokens.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Mapping

from dfo_platform import Alternative, CodeCatalogue, Envelope, ProblemIssue
from dfo_platform.schema_annotations import strip_schema_annotations
from dfo_platform.trace import set_trace_id
from graph_api.capabilities import DataBuilds, build_capabilities
from graph_api.codes import GRAPH_CODE_CATALOGUE
from graph_api.doors.schema import served_wire_schemas

# Fix the trace id so rendered refusal bodies hash deterministically.
_EXEMPLAR_TRACE_ID = "0" * 32

# Fixed value for exemplar bodies and the hashed capability document.
_EXEMPLAR_CAPABILITY_VERSION = "graph_exemplar_placeholder"

# Exclude deployment-specific build ids from the capability hash.
_EXEMPLAR_BUILD = "graph-exemplar-build-placeholder"


def _pin_capability_version_and_build(document: Mapping[str, object]) -> dict[str, object]:
    """Pin deployment-specific fields on a copy of the capability document."""
    pinned = dict(document)
    pinned["capability_version"] = _EXEMPLAR_CAPABILITY_VERSION
    pinned["build"] = _EXEMPLAR_BUILD
    pinned["data_builds"] = DataBuilds().model_dump(mode="json")
    return pinned


def _render_body_exemplars(catalogue: CodeCatalogue) -> dict[str, dict[str, object]]:
    """Render one refusal body per code in sorted order. The first fills every optional
    field so schema changes affect the hash. The caller binds a fixed trace id
    in a fresh context to keep results deterministic.
    """
    set_trace_id(_EXEMPLAR_TRACE_ID)
    envelope = Envelope(catalogue=catalogue, capability_version=_EXEMPLAR_CAPABILITY_VERSION)
    sorted_codes = sorted(catalogue)
    exemplars: dict[str, dict[str, object]] = {}
    for index, code in enumerate(sorted_codes):
        if index == 0:
            problem = envelope.details(
                status=422,
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
            problem = envelope.details(status=422, code=code, retryable=False)
        exemplars[code] = problem.to_json()
    return exemplars


def _compute_graph_capability_version(
    *,
    capability_document: Mapping[str, object] | None = None,
    code_catalogue: CodeCatalogue = GRAPH_CODE_CATALOGUE,
    wire_schemas: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """Deterministic `capability_version` token for this surface's own
    responses: a short content hash over every input named in this
    module's own docstring.
    """
    if capability_document is None:
        capability_document = build_capabilities(
            capability_version=_EXEMPLAR_CAPABILITY_VERSION, build=_EXEMPLAR_BUILD
        ).model_dump(mode="json")
    if wire_schemas is None:
        wire_schemas = served_wire_schemas()

    payload = {
        "capability_document": _pin_capability_version_and_build(capability_document),
        "refusal_body_exemplars": contextvars.copy_context().run(
            _render_body_exemplars, code_catalogue
        ),
        "wire_schemas": {
            name: strip_schema_annotations(schema) for name, schema in sorted(wire_schemas.items())
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"graph_{digest}"


# This surface's own `capability_version`, stamped on every graph_api
# response (see `dfo_platform.app.create_app`'s per-surface envelope).
GRAPH_CAPABILITY_VERSION: str = _compute_graph_capability_version()
