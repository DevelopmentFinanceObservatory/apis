"""The omission of a preview read in the years a composite's component serves
(`read_plan.read_in_years`) that holds no point in them, shared by the per-series
(`fetch`) and bulk (`bulk_role_series`) reads."""

from __future__ import annotations

from search.engine.region.codes import OBSERVATIONS_NO_POINTS
from search.models import SpecObservations, omitted_observations


def year_count(years: frozenset[int]) -> str:
    """`years` counted with its noun: "1 year", "3 years"."""
    return f"{len(years)} year" if len(years) == 1 else f"{len(years)} years"


def no_points_in_served_years(
    years: frozenset[int],
    *,
    dispatched: int,
    empty: int,
    hard_failed: int | None = None,
    facet_rejected: int | None = None,
) -> SpecObservations:
    """The `observations_no_points` omission of a preview none of whose series holds a point
    in `years`, the years its narrowed plan serves (`ServedYears.years`), having dispatched
    `dispatched` series of which `empty` came back empty, `hard_failed` failed and
    `facet_rejected` the facet filter rejected."""
    return omitted_observations(
        OBSERVATIONS_NO_POINTS,
        f"the observation preview shows no series: none of the {dispatched} series it "
        f"read holds a point in the {year_count(years)} it previews."
        if dispatched
        else "the observation preview read no rows: no series of the answer holds a point "
        f"in the {year_count(years)} it previews.",
        dispatched=dispatched,
        empty=empty,
        hard_failed=hard_failed,
        facet_rejected=facet_rejected,
    )
