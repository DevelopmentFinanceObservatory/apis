"""Output types and role bindings for `CoverageIndex` lookups.

Keeping these types independent of numpy and pyarrow lets type-only consumers
avoid importing `series_coverage.index` and its bit arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["donor", "recipient"]
# Missing role keys leave that axis open. An empty frozenset matches no rows.
BoundRoles = Mapping[Role, frozenset[str]]


@dataclass(frozen=True, slots=True)
class FacetCoverage:
    """Coverage for one `facet_id`, summed across surviving donor/recipient rows.

    `year_counts` counts active rows per year, so two pairs in one year count
    twice. Its values sum to `n_obs`, and keys fall within the year bounds.
    Directly constructed test values may leave it empty.

    `first_series_end` is the earliest of the rows' last years with data: the year the
    donor/recipient series that ends first ends. Directly constructed test values may leave
    it `None`, which reads as `last_year`.
    """

    facet_id: str
    n_obs: int
    first_year: int
    last_year: int
    year_counts: Mapping[int, int] = field(default_factory=dict)
    first_series_end: int | None = None


@dataclass(frozen=True, slots=True)
class FacetSpan:
    """One `facet_id`'s observation count and first and last years with data, summed across
    surviving donor/recipient rows: `FacetCoverage`'s `n_obs`, `first_year`, `last_year` and
    `first_series_end` without its year counts."""

    n_obs: int
    first_year: int
    last_year: int
    first_series_end: int


@dataclass(frozen=True, slots=True)
class VariableCoverage:
    """One variable's surviving coverage, with facets sorted by `facet_id`.

    Variables with no surviving rows are omitted.
    """

    variable: str
    facets: tuple[FacetCoverage, ...]
    n_obs: int
    first_year: int
    last_year: int


@dataclass(frozen=True, slots=True)
class YearCount:
    """Counts for one calendar year: the series holding an observation in it
    and the distinct donor and recipient dcids those series carry. `observations`
    equals `series`, because each row has at most one observation per year, and a
    year with no data counts zero everywhere."""

    year: int
    observations: int
    series: int
    role_counts: Mapping[Role, int]


@dataclass(frozen=True, slots=True)
class Headline:
    """Counts for the year a `latest` read serves, or for a requested year window.

    `year` is the latest counted year with data inside a window, and under
    `latest` the newest year a variable elected. In the latest-year form
    `observations` equals `series`, because each row has at most one
    observation per year; across a window `observations` can exceed
    `series`. `role_counts` counts distinct donor and recipient dcids. Use
    `CoverageIndex.role_names` for graph names.

    `recent_years` holds the same counts for three calendar years, newest first,
    a year with no data kept with zero counts. For a requested window they end at
    `year`, and the first entry repeats `observations`, `series` and
    `role_counts`. For a `latest` read they end at the latest year any counted row
    holds up to the index's build year, or at the latest year any holds when none
    holds a year that early, and count every row. `year` is one of the three: the
    newest year a variable of the read heads, the year it elected
    (`CoverageIndex.elected_years`) or, electing none, its latest year up to the
    build year. When no variable holds a year that early, each heads its latest
    year. The list stops at the index's epoch year, so it holds one or two
    entries within two years of it.
    The earlier years are counted whether or not a requested window reaches them.
    """

    year: int
    observations: int
    series: int
    role_counts: Mapping[Role, int]
    recent_years: tuple[YearCount, ...]


@dataclass(frozen=True, slots=True)
class ServedExtent:
    """What a read serves across its whole window (`CoverageIndex.served_extent`):
    `rows` counts one point per series and year, `series` the `(variable,
    recipient, donor)` series holding one, and `first_year` and `last_year`
    bound the years of those points."""

    rows: int
    series: int
    first_year: int
    last_year: int


@dataclass(frozen=True, slots=True)
class Estimate:
    """Estimated fetch cost before network I/O. `rows` counts observations
    and `requests` counts SDMX GETs for covered data. `CoverageIndex.estimate`
    leaves `uncovered_requests` at zero. Admission code counts requests whose
    StatVars have no coverage row.
    """

    rows: int
    requests: int
    uncovered_requests: int = 0


__all__ = [
    "BoundRoles",
    "Estimate",
    "FacetCoverage",
    "FacetSpan",
    "Headline",
    "Role",
    "ServedExtent",
    "VariableCoverage",
    "YearCount",
]
