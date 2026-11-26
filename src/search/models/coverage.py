from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .entity import _is_none
from .grammar import HeldKind, TimeWindow

# ---------------------------------------------------------------------------
# Coverage arms
# ---------------------------------------------------------------------------


class BreadthDim(StrictModel):
    """One breadth dimension with its cardinality."""

    label: str = Field(description="the dimension, e.g. providers, years, methods, sources")
    count: int = Field(description="e.g. 36, 22, 3")
    residual_count: int | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "on an open role's dimension, how many of `count` are residual members "
            "(party_kind residual), counted as RoleAxisFact.residual_count counts them, so "
            "count minus residual_count counts the countable parties. Absent on every other "
            "dimension and where the role holds no residual member"
        ),
    )


class ReportedShare(StrictModel):
    """How much of a partly reported year its source has reported so far, against the
    year before it."""

    basis: Literal["value", "series"] = Field(
        description=(
            "'value': the year's summed reported value over the year before's, 0 or more "
            "and above 1 when the year already exceeds the year before; 'series': the "
            "series holding the year over those holding the year before"
        )
    )
    share: float = Field(description="the share reported so far, e.g. 0.12")
    against: int = Field(description="the year compared with, the year before")


class HeldBack(StrictModel):
    """One run of years a read holds back unless its window opts in to them
    (`TimeWindow.include`): a source's partly reported period, or the projections
    continuing a series past its newest actual year."""

    kind: HeldKind = Field(
        description=(
            "'partial': years the source has reported only in part; 'projected': "
            "projected years after the newest actual year"
        )
    )
    served: bool = Field(
        description=(
            "true when the request's window opted in and these years are in the answer; "
            "false when the answer leaves them out"
        )
    )
    source: str = Field(description="the provenance dcid that marks or continues these years")
    source_name: str | None = Field(
        description="the provenance's display name, e.g. 'OECD CRS'; null when unknown"
    )
    start_year: int = Field(description="the first year of the run")
    end_year: int = Field(description="the last year of the run")
    reported_share: ReportedShare | None = Field(
        description=(
            "for a partial run, how much of its first year is reported so far, when known; "
            "null for a projected run"
        )
    )


def _no_held_back(value: object) -> bool:
    return not value


_HELD_BACK_DESCRIPTION = (
    "every run of partly reported or projected years the answer's sources hold, whatever "
    "the request's window. Each states whether the answer serves them (`served`, when the "
    "window's `include` names their kind and the window holds a year of them) or leaves "
    "them out; absent when the sources hold none"
)


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
            "entities in the resolved window, or full history when window is absent. "
            "A role-bearing series holds one observation per year, so a bulk read counts "
            "its years with data"
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
    held_back: list[HeldBack] | None = Field(
        default=None, exclude_if=_no_held_back, description=_HELD_BACK_DESCRIPTION
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
    held_back: list[HeldBack] | None = Field(
        default=None, exclude_if=_no_held_back, description=_HELD_BACK_DESCRIPTION
    )


class CoverageBare(StrictModel):
    """Observation footprint with no count or dimensions; an honest could-not-count."""

    kind: Literal["bare"] = Field(default="bare", description="coverage kind discriminator")
    has_data: bool = Field(description="whether the region has any observations")
    window: TimeWindow | None = Field(default=None, description="the date window, when known")
    held_back: list[HeldBack] | None = Field(
        default=None, exclude_if=_no_held_back, description=_HELD_BACK_DESCRIPTION
    )


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
    """The exact, one-year summary of a role-bearing answer's own served
    population: the year `CoverageIndex.headline` (series_coverage) heads the
    answer with, whether that year was the one the question named, the exact
    observation and series count in that year alone, and how many distinct dcids
    each role carries in it. `series_coverage.index.Headline`'s own docstring is
    the source of truth for why `observations` and `series` are always equal
    under this schema.

    The counts are those of the year the answer's read plan serves. Under a `latest`
    window that is the year its variables elect (`CoverageIndex.elected_years`), which
    the preview and a mint of the Spec serve too. Under a named window it is the latest
    year with data in that window, so a question naming 2015 to 2020 counts the latest
    of those years with data. `named_window` carries the years it named.

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
            "the year the counts are for. For a latest window it is the elected year: of "
            "the three calendar years ending at the newest year a variable's series hold, "
            "at or before the year the coverage artifact was built, the newest one that at "
            "least half as many series hold as the fullest of the three, and with several "
            "variables the newest year any of them elects. For a named window it is the "
            "latest year any served series has an observation inside the window"
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
            "the counts for three calendar years, newest first, so a newest year with far "
            "fewer series than the years before it reads as incomplete. For a latest window "
            "they are raw counts over every series the answer holds, ending at the newest "
            "year any of them holds at or before the year the coverage artifact was built, or "
            "at the newest year any holds when none holds a year that early; `year` is one of "
            "the three: the newest year a variable heads, which is the year it elected or, for "
            "a variable that elects none, its newest year at or "
            "before the build year. When no variable holds a year that early, each heads its "
            "newest year. `observations`, `series` and `role_counts` count only the variables "
            "heading it. For a named window they end at "
            "`year`, the first entry repeats `observations`, `series` and `role_counts`, and "
            "the earlier years count whether or not `named_window` reaches them. A year with "
            "no data is kept with zero counts. The list stops at the first year the coverage "
            "artifact represents, so it can hold fewer than three entries"
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


def _no_years(value: object) -> bool:
    return not value


class YearsWithData(StrictModel):
    """The calendar years an answer's served series hold data, over each series' whole
    span, whatever window the request named. A client offering the reader other years of
    the same answer reads them here.

    The series are those the answer serves: its StatVars, entities, constraints and facets,
    each counted in every year it holds, inside the request's window or outside it. The
    years are those a request naming no window would be served: a source's partly reported
    period and projected years are left out (`coverage.held_back` lists them), whether or
    not this request opted in to them, and under a unit request a year the conversion
    cannot convert is left out, as the download leaves it out. A role-bearing answer reads
    them from the coverage artifact (`CoverageIndex.series_years`), and any other answer
    from its series' facet metadata, where a facet listing no complete dates counts every
    year of its span."""

    start_year: int = Field(description="the first year any served series holds data")
    end_year: int = Field(description="the last year any served series holds data")
    gap_years: list[int] | None = Field(
        default=None,
        exclude_if=_no_years,
        description=(
            "the years after start_year and before end_year that no served series holds, "
            "in order; absent when every year between them holds data"
        ),
    )

    @model_validator(mode="after")
    def _gaps_inside_the_span(self) -> YearsWithData:
        if self.end_year < self.start_year:
            raise ValueError(f"end_year {self.end_year} precedes start_year {self.start_year}")
        if any(not self.start_year < year < self.end_year for year in self.gap_years or ()):
            raise ValueError("gap_years lie strictly between start_year and end_year")
        return self


def coverage_is_empty(coverage: Coverage) -> bool:
    """Return true when an exact count is zero or ``has_data`` is false."""
    return (isinstance(coverage, CoverageExact) and coverage.observation_count == 0) or (
        not coverage.has_data
    )
