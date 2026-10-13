"""Joins coverage counts to facet metadata and ranks candidates.

Search coverage reports all confirmed facets. Materialisation preview
selection uses `materialise.facets.rank_candidates`. This implementation
applies the same order to coverage-derived `n_obs` and `last_year` values. The coverage table
stores only `facet_id`; provenance, measurement method, unit, and scaling factor come from
`CoverageIndex.facet_metadata`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from series_coverage.coverage_types import FacetCoverage
from series_coverage.schema import FacetMetadata

_NO_METADATA = FacetMetadata()


@dataclass(frozen=True, slots=True)
class FacetCandidate:
    """Coverage counts and optional metadata for one facet.

    Coverage guarantees `facet_id`, `n_obs`, `first_year`, and `last_year`; `years` lists the
    years with data, from the rows' year masks, and is empty when the coverage carries no
    per-year counts. `first_series_end` is the year its first series to end ends
    (`FacetCoverage.first_series_end`). Metadata may be absent when the caller's map lacks
    that facet.
    """

    facet_id: str
    n_obs: int
    first_year: int
    last_year: int
    years: tuple[int, ...] = ()
    first_series_end: int | None = None
    provenance_id: str | None = None
    measurement_method: str | None = None
    unit: str | None = None
    scaling_factor: str | None = None
    observation_period: str | None = None


def facet_candidates(
    coverage: Sequence[FacetCoverage], *, metadata: Mapping[str, FacetMetadata]
) -> list[FacetCandidate]:
    """Joins coverage rows to metadata in input order. Missing metadata
    fields become `None` and rank last, so the lookup returns the candidate.
    """
    return [
        FacetCandidate(
            facet_id=f.facet_id,
            n_obs=f.n_obs,
            first_year=f.first_year,
            last_year=f.last_year,
            years=tuple(sorted(year for year, count in f.year_counts.items() if count)),
            first_series_end=f.first_series_end,
            provenance_id=metadata.get(f.facet_id, _NO_METADATA).provenance_id,
            measurement_method=metadata.get(f.facet_id, _NO_METADATA).measurement_method,
            unit=metadata.get(f.facet_id, _NO_METADATA).unit,
            scaling_factor=metadata.get(f.facet_id, _NO_METADATA).scaling_factor,
            observation_period=metadata.get(f.facet_id, _NO_METADATA).observation_period,
        )
        for f in coverage
    ]


def rank_candidates(candidates: Sequence[FacetCandidate]) -> tuple[FacetCandidate, ...]:
    """Ranks by descending `n_obs`, descending `last_year`, then
    `provenance_id`, `measurement_method`, `unit`, and `facet_id`. Stable
    sorts preserve earlier keys. The unique `facet_id` makes the order total.
    """
    ranked = list(candidates)
    ranked.sort(key=lambda c: c.facet_id)
    ranked.sort(key=lambda c: _none_last(c.unit))
    ranked.sort(key=lambda c: _none_last(c.measurement_method))
    ranked.sort(key=lambda c: _none_last(c.provenance_id))
    ranked.sort(key=lambda c: c.last_year, reverse=True)
    ranked.sort(key=lambda c: c.n_obs, reverse=True)
    return tuple(ranked)


def _none_last(value: str | None) -> tuple[bool, str]:
    return (value is None, value or "")


__all__ = [
    "FacetCandidate",
    "FacetMetadata",
    "facet_candidates",
    "rank_candidates",
]
