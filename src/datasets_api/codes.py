"""Merge the Datasets refusal codes by door family.

`request_invalid`, `unsupported_schema_version`, `service_unavailable`, and `code_not_found` use
platform entries. Redeclaring them would raise `CodeCollision` in `merge_catalogues`.
"""

from __future__ import annotations

from datasets_api.codes_core import CORE_CODE_CATALOGUE
from datasets_api.codes_materialise import MATERIALISE_DOOR_CODE_CATALOGUE
from datasets_api.codes_publish import PUBLISH_DOOR_CODE_CATALOGUE
from datasets_api.codes_snapshot import SNAPSHOT_DOOR_CODE_CATALOGUE
from dfo_platform.codes import PLATFORM_CODE_CATALOGUE, merge_catalogues

DATASETS_CODE_CATALOGUE: dict[str, tuple[str, str]] = merge_catalogues(
    CORE_CODE_CATALOGUE,
    MATERIALISE_DOOR_CODE_CATALOGUE,
    SNAPSHOT_DOOR_CODE_CATALOGUE,
    PUBLISH_DOOR_CODE_CATALOGUE,
)

# Include platform codes handled by this surface's classifiers.
CODE_CATALOG: dict[str, tuple[str, str]] = merge_catalogues(
    PLATFORM_CODE_CATALOGUE, DATASETS_CODE_CATALOGUE
)

# Platform codes this surface returns for request, upstream, capacity, timeout, store, lookup,
# facet, role, capability, and upstream-version refusals. Four entries are contract-reserved.
# `recipe_not_declared` and `rate_coverage_gap` are raised by `materialise.recipes` during
# currency conversion and classified by the mint doors (`doors/conversion_errors.py`).
# Registry-binding tests use this served-code set.
SHARED_PLATFORM_CODES: frozenset[str] = frozenset(
    {
        "request_invalid",
        "unsupported_schema_version",
        "upstream_error",
        "service_unavailable",
        "timeout",
        "internal_error",
        "code_not_found",
        "facet_not_available",
        "facet_binding_unsupported",
        "resolution_ambiguous",
        "invalid_role_ref",
        "role_incompatible",
        "role_signature_incompatible",
        "recipe_not_declared",
        "rate_coverage_gap",
        "capability_unavailable",
        "upstream_version_changed",
    }
)
if not frozenset(PLATFORM_CODE_CATALOGUE) >= SHARED_PLATFORM_CODES:
    raise AssertionError(
        f"SHARED_PLATFORM_CODES names code(s) "
        f"{sorted(SHARED_PLATFORM_CODES - frozenset(PLATFORM_CODE_CATALOGUE))} absent from "
        "PLATFORM_CODE_CATALOGUE"
    )
