"""`dset_` plus 16 SHA-256 hex digits over canonical JSON. This token is
stored in durable version records, so it has no environment override.

The hash covers four inputs:

1. The capability document, excluding `capability_version` (self-reference),
   `build` (redeploys can change it without changing behaviour), and the
   content-derived `code_index`, `decode_code_systems`, `summable_axes`, and
   `groupby_level_axes`.
   Every other field, including caps, liveness flags, and sync budgets,
   affects the token.
2. Rendered refusal bodies for every code. The first exemplar fills every
   optional field, so rendering and catalogue wording changes are detected.
3. The schema-endpoint payload without `description` or `title` fields.
   Schema structure affects the token. The hash strips description and title text.
4. The warning catalogue, sorted by code, including each title, detail,
   and severity.

Each input is injectable for perturbation tests. Behaviour changes outside
these published inputs do not affect the token.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Mapping

from code_index.types import CodeSystemDeclaration, IndexStamp
from datasets_api.capabilities import build_capabilities
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.config import DatasetsBuildSettings
from datasets_api.schema import build_schema_document
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from dfo_platform import Alternative, CodeCatalogue, Envelope, ProblemIssue
from dfo_platform.config import PublicSiteSettings
from dfo_platform.trace import set_trace_id

# Exclude the self-referential token and deployment build from the hash.
_EXEMPLAR_CAPABILITY_VERSION = "dset_exemplar_placeholder"
_EXEMPLAR_BUILD = "dset-exemplar-build-placeholder"

# The capability document does not carry the website origin, so the import-time build
# supplies a placeholder and the module loads without `DFO_PUBLIC_SITE_URL`.
_EXEMPLAR_SITE = PublicSiteSettings(public_site_url="https://exemplar.invalid")

# Index-derived fields are pinned before hashing. `summability_declarations_live` stays
# in the payload because it reports live behaviour, not index content.
_EXEMPLAR_CODE_INDEX_STAMP = IndexStamp(
    graph_endpoint="",
    exported_at="",
    code_counts={},
    statvar_shape_counts={},
    content_hash="",
)
_EXEMPLAR_CODE_SYSTEM_DECLARATIONS: dict[str, CodeSystemDeclaration] = {}
_EXEMPLAR_CODE_INDEX: dict[str, object] = {
    "content_hash": "dset-exemplar-code-index-placeholder",
    "exported_at": "",
    "code_counts": {},
}
_EXEMPLAR_DECODE_CODE_SYSTEMS: list[str] = []
_EXEMPLAR_SUMMABLE_AXES: list[str] = []
_EXEMPLAR_GROUPBY_LEVEL_AXES: list[str] = []

# Fix the trace id so refusal-body exemplars hash deterministically.
_EXEMPLAR_TRACE_ID = "0" * 32


def _strip_wire_invisible_keys(value: object) -> object:
    """Remove `description` and `title` keys recursively before hashing a schema.

    The local copy keeps the Datasets token independent of the platform.
    """
    if isinstance(value, dict):
        return {
            key: _strip_wire_invisible_keys(val)
            for key, val in value.items()
            if key not in ("description", "title")
        }
    if isinstance(value, list):
        return [_strip_wire_invisible_keys(item) for item in value]
    return value


def _pin_deployment_identity_fields(document: Mapping[str, object]) -> dict[str, object]:
    pinned = dict(document)
    pinned["capability_version"] = _EXEMPLAR_CAPABILITY_VERSION
    pinned["build"] = _EXEMPLAR_BUILD
    pinned["code_index"] = _EXEMPLAR_CODE_INDEX
    pinned["decode_code_systems"] = _EXEMPLAR_DECODE_CODE_SYSTEMS
    pinned["summable_axes"] = _EXEMPLAR_SUMMABLE_AXES
    pinned["groupby_level_axes"] = _EXEMPLAR_GROUPBY_LEVEL_AXES
    return pinned


def _render_body_exemplars(catalogue: CodeCatalogue) -> dict[str, dict[str, object]]:
    """Render one refusal body per code in sorted order. The first fills every optional
    field so omitted fields affect the hash. Run in a fresh context with the fixed trace id
    to keep the output deterministic.
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


def _compute_datasets_capability_version(
    *,
    capability_document: Mapping[str, object] | None = None,
    code_catalogue: CodeCatalogue = DATASETS_CODE_CATALOGUE,
    schema_document: Mapping[str, object] | None = None,
    warning_catalogue: Mapping[str, tuple[str, str, str]] = DATASETS_WARNING_CATALOGUE,
) -> str:
    """Deterministic `capability_version` token for this surface's own
    responses: a short content hash over every input named in this module's
    own docstring.
    """
    if capability_document is None:
        capability_document = build_capabilities(
            capability_version=_EXEMPLAR_CAPABILITY_VERSION,
            build=_EXEMPLAR_BUILD,
            code_index_stamp=_EXEMPLAR_CODE_INDEX_STAMP,
            declarations=_EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
            settings=DatasetsBuildSettings(site=_EXEMPLAR_SITE),
        ).model_dump(mode="json")
    if schema_document is None:
        schema_document = build_schema_document()

    payload = {
        "capability_document": _pin_deployment_identity_fields(capability_document),
        "refusal_body_exemplars": contextvars.copy_context().run(
            _render_body_exemplars, code_catalogue
        ),
        "schema_document": _strip_wire_invisible_keys(schema_document),
        "warning_catalogue": {
            code: list(entry) for code, entry in sorted(warning_catalogue.items())
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"dset_{digest}"


# This surface's own `capability_version`, stamped on every datasets_api
# response (see `dfo_platform.app.create_app`'s per-surface envelope).
DATASETS_CAPABILITY_VERSION: str = _compute_datasets_capability_version()
