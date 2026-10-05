from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .grammar import TimeWindow

# ---------------------------------------------------------------------------
# Coverage arms
# ---------------------------------------------------------------------------


class BreadthDim(StrictModel):
    """One breadth dimension with its cardinality."""

    label: str = Field(description="the dimension, e.g. donors, years, methods, sources")
    count: int = Field(description="e.g. 36, 22, 3")


class CoverageExact(StrictModel):
    """Observation footprint with an exact count for its resolved window."""

    kind: Literal["exact"] = Field(default="exact", description="coverage kind discriminator")
    has_data: bool = Field(
        description=(
            "whether the exact observation_count has observations in the resolved window, "
            "or in full history when window is absent"
        )
    )
    observation_count: int = Field(
        description=(
            "exact distinct (date, facet) observation pairs for resolved stat_vars and "
            "entities in the resolved window, or full history when window is absent"
        )
    )
    evidence: Literal["facet_metadata", "observed"] = Field(
        default="facet_metadata",
        description=(
            "provenance of observation_count. 'facet_metadata': the count comes from the "
            "graph's facet index and no observation series was delivered for this spec in "
            "this payload. 'observed': a preview series was delivered for this spec in this "
            "payload, and observation_count is still the facet-index count, now backed by "
            "fetchable data. A streaming response's `response` event ships before any "
            "observation fetch, so it always carries 'facet_metadata'; its trailing "
            "`observations` event carries the series. dfo-apis wire field outside the "
            "spec-kit contract."
        ),
    )
    dimensions: list[BreadthDim] | None = Field(
        default=None,
        description=(
            "optional per-facet breadth dimensions; the observations dimension is "
            "window-narrowed to agree with observation_count when window is set, "
            "lifetime otherwise"
        ),
    )
    window: TimeWindow | None = Field(
        default=None, description="the date window this count applies to"
    )


class CoverageBreadth(StrictModel):
    """Observation footprint expressed as structured breadth dimensions."""

    kind: Literal["breadth"] = Field(default="breadth", description="coverage kind discriminator")
    has_data: bool = Field(description="whether the region has any observations")
    dimensions: Annotated[list[BreadthDim], Field(min_length=1)] = Field(
        description="non-empty on breadth; structured dimension counts"
    )
    window: TimeWindow | None = Field(
        default=None, description="the date window this breadth applies to"
    )


class CoverageBare(StrictModel):
    """Observation footprint with no count or dimensions; an honest could-not-count."""

    kind: Literal["bare"] = Field(default="bare", description="coverage kind discriminator")
    has_data: bool = Field(description="whether the region has any observations")
    window: TimeWindow | None = Field(default=None, description="the date window, when known")


Coverage = Annotated[
    CoverageExact | CoverageBreadth | CoverageBare,
    Field(discriminator="kind"),
]


class HeadlineYear(StrictModel):
    """The exact counts for one calendar year of a role-bearing answer's served
    population: observations and series with an observation in it, and how many
    distinct dcids each role carries. A year with no data counts zero."""

    year: int = Field(description="the calendar year these counts are for")
    observations: int = Field(description="exact observation count across served series in year")
    series: int = Field(description="exact series count with an observation in year")
    role_counts: dict[str, int] = Field(
        description=(
            "distinct dcid count per role active in year, keyed by the graph's own role "
            "property dcid; a role with no dcid in the year is absent"
        )
    )


class Headline(StrictModel):
    """The exact, latest-year summary of a role-bearing answer's own served
    population: the year `CoverageIndex.headline` (series_coverage) found
    the most recent observation in, whether that year was the one the
    question named, the exact observation and series count in that year
    alone, and how many distinct dcids each role carries in it. `series_coverage
    .index.Headline`'s own docstring is the source of truth for why
    `observations` and `series` are always equal under this schema.

    The counts are those of the latest year the answer's read plan serves inside the
    years it reads: the latest year with data, or under a named window the latest year
    with data in that window, so a question naming 2015 to 2020 counts the latest of
    those years with data. `named_window` carries the years it named.

    Attached to every role-bearing answer (named-destination confirmed and
    open-axis discovered alike), never to a non-role-bearing one: coverage
    only exists for role-bearing series (series_coverage.schema's own scope
    note), and computing this is a free in-memory bitmask operation, not a
    network read, so there is no population-size reason to withhold it from
    a narrow answer the way the per-series preview stays capped at 20
    regardless of how broad the answer is.
    """

    year: int = Field(
        description=(
            "the latest year any served series has an observation, inside the named "
            "window when the request names one"
        )
    )
    year_named: bool = Field(
        description=(
            "whether the request itself named a year or a range of years, rather than "
            "defaulting to one, independent of whether those years include the shown `year`"
        )
    )
    named_window: TimeWindow | None = Field(
        default=None,
        description=(
            "the year or years the request named: equal bounds for one year, an absent "
            "bound for an open range; null when no year was named. The counts are for "
            "`year` alone, whatever this window holds"
        ),
    )
    observations: int = Field(description="exact observation count across served series in year")
    series: int = Field(description="exact series count with an observation in year")
    role_counts: dict[str, int] = Field(
        description=(
            "distinct dcid count per role active in year, keyed by the graph's own role "
            "property dcid (e.g. financingProvider, financingRecipient) -- never by a "
            "donor/recipient label, so the wire stays aligned with Data Commons' own names"
        )
    )

    recent_years: list[HeadlineYear] = Field(
        min_length=1,
        description=(
            "the counts for `year` and the two calendar years before it, newest first, so a "
            "latest year with far fewer observations than the years before it reads as "
            "incomplete. A year with no data is kept with zero counts. The list stops at the "
            "first year the coverage artifact represents, so it can hold fewer than three "
            "entries. The first entry repeats `observations`, `series` and `role_counts`. The "
            "earlier years count whether or not `named_window` reaches them"
        ),
    )

    @model_validator(mode="after")
    def _named_window_only_when_a_year_was_named(self) -> Headline:
        if self.named_window is not None and not self.year_named:
            raise ValueError("named_window is set only when year_named is true")
        return self

    @model_validator(mode="after")
    def _year_inside_the_named_window(self) -> Headline:
        """The counted year is one the answer read, so it lies inside the window named."""
        window = self.named_window
        if window is not None and not (
            (window.start_year is None or window.start_year <= self.year)
            and (window.end_year is None or self.year <= window.end_year)
        ):
            raise ValueError(f"year {self.year} lies outside the named window {window!r}")
        return self


def coverage_is_empty(coverage: Coverage) -> bool:
    """Return true when an exact count is zero or ``has_data`` is false."""
    return (isinstance(coverage, CoverageExact) and coverage.observation_count == 0) or (
        not coverage.has_data
    )
