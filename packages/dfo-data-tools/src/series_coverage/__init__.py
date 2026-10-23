"""Public API for coverage schemas, lookups, and the served coverage.

Exports backed by numpy-dependent modules resolve on first access, since
`numpy` is installed only with the `coverage` extra. Other types and schema
names load eagerly, so the package imports without that extra.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from series_coverage.coverage_types import (
    BoundRoles,
    Estimate,
    FacetCoverage,
    Headline,
    Role,
    VariableCoverage,
    YearCount,
)
from series_coverage.facets import FacetCandidate, FacetMetadata, facet_candidates, rank_candidates
from series_coverage.schema import CoverageManifest, RoleNames, YearHorizon

if TYPE_CHECKING:
    from series_coverage.index import CoverageIndex
    from series_coverage.loader import CoverageArtifactError
    from series_coverage.shared import get_coverage_service

# Export names resolved on first access.
_LAZY: dict[str, tuple[str, str]] = {
    "CoverageIndex": ("series_coverage.index", "CoverageIndex"),
    "CoverageArtifactError": ("series_coverage.loader", "CoverageArtifactError"),
    "get_coverage_service": ("series_coverage.shared", "get_coverage_service"),
}


def __getattr__(name: str) -> Any:
    """Imports and caches a lazy export.

    Unknown names raise `AttributeError`.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BoundRoles",
    "CoverageArtifactError",
    "CoverageIndex",
    "CoverageManifest",
    "Estimate",
    "FacetCandidate",
    "FacetCoverage",
    "FacetMetadata",
    "Headline",
    "Role",
    "RoleNames",
    "VariableCoverage",
    "YearCount",
    "YearHorizon",
    "facet_candidates",
    "get_coverage_service",
    "rank_candidates",
]
