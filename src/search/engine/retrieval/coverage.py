"""Build an answer's coverage and headline from its read plan.

Coverage counts and dates what the plan's preview serves, over the whole population, never
the part a preview delivers (`read_plan`):

- A per-series plan (`PairPlan`) serves each served series' group. Materialise collapses the
  group's rows to one per date and observation period (`materialise.facet_collapse`), so a
  series counts its distinct points (`read_plan.series_points`). Under `latest` each StatVar
  serves its one latest period, one point from each series holding it (`PairPlan.served`).
  A range counts the points inside it. A scope-defaulted plan serves its one facet at every
  place.
- A bulk role plan (`BulkPlan`) serves its elected facet across every pair of each member
  in that member's years (`BulkPlan.member_years`). Under `latest` those are the member's
  own latest year alone, the newest any of its pairs reaches.

The window a range or `latest` answer states spans the years of the dates the plan serves
(`read_plan.served_dates`), the dates each StatVar is dated from.

The headline counts the plan's served rows (`served_rows`) inside the plan's years. Option
coverage plans each option's facets the same way. This module performs no I/O beyond the
coverage index lookups the headline and bulk counts make.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from search.engine.clients.graph import Facet
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PairPlan,
    ReadPlan,
    dated_without_complete_dates,
    facets_served,
    series_dates,
    series_points,
    served_dates,
    year_of,
)
from search.models import (
    BreadthDim,
    CoverageBare,
    CoverageBreadth,
    CoverageExact,
    Headline,
    HeadlineYear,
    TimeWindow,
)

if TYPE_CHECKING:
    from search.engine.extract import DateRequest
    from series_coverage.index import BoundRoles, CoverageIndex
    from series_coverage.index import Role as CoverageRole

# Shared label for served-variable coverage.
MEMBER_VARIABLES_LABEL = "variables"

type ServedRow = tuple[str, str, str | None, str | None]
"""A served coverage row pattern: variable, facet id, recipient and donor, where `None`
matches any dcid on that role."""


@dataclass(frozen=True)
class CoverageForm:
    """How an answer states its coverage: the labels of its source and observation
    dimensions, whether the first dimension counts the places served (`counts_places`, for a
    place set) instead of citable sources, whether a `variables` dimension counts the
    StatVars served, and for a discovered role scope the open axis's label and coverage role
    (`open_axis`), which makes the coverage a breadth summary led by the members served."""

    facet_label: str = "sources"
    obs_label: str = "observations"
    counts_places: bool = False
    variables: bool = False
    open_axis: tuple[str, CoverageRole] | None = None


def citable_source_ids(facets: Sequence[Facet]) -> list[str]:
    """Return ordered distinct `provenance_id` values.

    Facets without one contribute nothing. Duplicates, including facets with
    different `facet_id` values, count as one source.
    """
    return list(dict.fromkeys(f.provenance_id for f in facets if f.provenance_id))


def stated_window(plan: ReadPlan) -> TimeWindow | None:
    """The window coverage states for `plan`: the span of the years of the dates it serves
    (`read_plan.served_dates`), earliest to latest, whether a range named its bounds or left
    them open. A range that serves nothing states its own years, a `latest` read with no
    dated point states none, and an every-date read states none."""
    if plan.years is None:
        return None
    years = [
        year
        for dates in served_dates(plan).values()
        for date in dates
        if (year := year_of(date)) is not None
    ]
    if not years:
        return plan.years if isinstance(plan.years, TimeWindow) else None
    return TimeWindow(start_year=min(years), end_year=max(years))


def scope_members(plan: ReadPlan, form: CoverageForm) -> list[str]:
    """The members of a defaulted scope that `plan` serves.

    For a discovered role scope (`form.open_axis`) they are the open role's dcids, sorted:
    those of the served series, or in bulk those the elected facet holds data for inside
    each member's years (`BulkPlan.member_pairs`). Otherwise they are the places of the
    served series, in series order. Coverage counts its breadth dimension from this list, and an answer lists it as
    its scope (`ResolvedSpec.scope`) and counts it in its `role_scope_defaulted` warning, so
    the three agree."""
    if isinstance(plan, BulkPlan):
        members: set[str] = set()
        for _sv, c, (first, last) in plan.member_pairs():
            members |= c.pairs.members_with_data(first_year=first, last_year=last)
        return sorted(members)
    if form.open_axis is not None:
        _label, role = form.open_axis
        index = 0 if role == "recipient" else 1
        return sorted(
            {member for s in plan.served if s.rows is not None and (member := s.rows[index])}
        )
    return list(dict.fromkeys(s.role_tuple[0][1] for s in plan.served))


def _variables_dim(count: int) -> BreadthDim:
    return BreadthDim(label=MEMBER_VARIABLES_LABEL, count=count)


def answer_coverage(
    plan: ReadPlan, form: CoverageForm
) -> CoverageExact | CoverageBreadth | CoverageBare:
    """The coverage of what `plan` serves, stated in `form`. Evidence is `facet_metadata`.

    A plan with no candidate returns `CoverageBare` without data. A range or `latest` window
    over a served data-bearing facet without complete dates returns `CoverageBreadth` with
    lifetime counts in its observation dimension. The observation dimension is the largest
    per-series count.
    """
    if isinstance(plan, BulkPlan):
        return _bulk_coverage(plan, form)
    return _pair_coverage(plan, form)


def _pair_coverage(
    plan: PairPlan, form: CoverageForm
) -> CoverageExact | CoverageBreadth | CoverageBare:
    if not any(s.candidates for s in plan.series):
        return CoverageBare(has_data=False)
    served = plan.served
    facets = facets_served(plan)
    read = plan.years if isinstance(plan.years, TimeWindow) else None
    window = stated_window(plan)
    facet_dim = BreadthDim(
        label=form.facet_label,
        count=(
            len(scope_members(plan, form))
            if form.counts_places
            else len(citable_source_ids(facets))
        ),
    )
    variables = [_variables_dim(len({s.stat_var for s in served}))] if form.variables else []

    if window is not None and dated_without_complete_dates(facets):
        return CoverageBreadth(
            kind="breadth",
            has_data=any(f.obs_count > 0 for f in facets),
            dimensions=[
                facet_dim,
                BreadthDim(
                    label=form.obs_label, count=max((f.obs_count for f in facets), default=0)
                ),
                *variables,
            ],
            window=window,
        )

    if plan.years == "latest" and window is not None:
        # Each served series serves one point, its StatVar's latest period.
        per_series = [1 for s in served if series_dates(s, plan.years) is not None]
    else:
        per_series = [series_points(s, read) for s in served]
    observation_count, obs_dim_count = sum(per_series), max(per_series, default=0)

    if form.open_axis is not None:
        label, _role = form.open_axis
        return CoverageBreadth(
            has_data=bool(served),
            window=window,
            dimensions=[
                BreadthDim(label=label, count=len(scope_members(plan, form))),
                facet_dim,
                *variables,
            ],
        )
    return CoverageExact(
        kind="exact",
        has_data=observation_count > 0,
        observation_count=observation_count,
        evidence="facet_metadata",
        dimensions=[facet_dim, BreadthDim(label=form.obs_label, count=obs_dim_count), *variables],
        window=window,
    )


def _bulk_coverage(plan: BulkPlan, form: CoverageForm) -> CoverageExact | CoverageBreadth:
    window = stated_window(plan)
    facet_dim = BreadthDim(
        label=form.facet_label, count=len(citable_source_ids(facets_served(plan)))
    )
    served_svs = {
        sv
        for sv, c, (first, last) in plan.member_pairs()
        if c.pairs.n_with_data(first_year=first, last_year=last)
    }
    variables = [_variables_dim(len(served_svs))] if form.variables else []
    if form.open_axis is not None:
        label, _role = form.open_axis
        return CoverageBreadth(
            has_data=plan.population > 0,
            window=window,
            dimensions=[
                BreadthDim(label=label, count=len(scope_members(plan, form))),
                facet_dim,
                *variables,
            ],
        )
    observation_count = plan.observation_count
    obs_dim_count = max(
        (
            c.pairs.max_n_obs(first_year=first, last_year=last)
            for _sv, c, (first, last) in plan.member_pairs()
        ),
        default=0,
    )
    return CoverageExact(
        kind="exact",
        has_data=observation_count > 0,
        observation_count=observation_count,
        evidence="facet_metadata",
        dimensions=[facet_dim, BreadthDim(label=form.obs_label, count=obs_dim_count), *variables],
        window=window,
    )


def served_rows(plan: ReadPlan) -> frozenset[ServedRow]:
    """The coverage rows `plan` serves: each served series' group facets at its recipient
    and donor (`None` on a role it does not declare), or in bulk the elected facet at every
    pair of every member it serves (`BulkPlan.served_members`)."""
    if isinstance(plan, BulkPlan):
        best = plan.elected
        if best is None:
            return frozenset()
        return frozenset((sv, best.facet_id, None, None) for sv in plan.served_members)
    return frozenset(
        (s.stat_var, facet_id, *(s.rows or (None, None)))
        for s in plan.served
        for facet_id in s.group_ids
    )


def answer_headline(
    plan: ReadPlan, coverage_index: CoverageIndex, *, date_request: DateRequest | None
) -> Headline | None:
    """The headline of a role-bearing plan: the latest year its served rows reach inside its
    years, with that year's counts. The rows a conversion leaves out (`ReadPlan.omitted`)
    count for nothing. None for a plan that is not role-bearing."""
    if plan.roles is None:
        return None
    window = plan.years if isinstance(plan.years, TimeWindow) else None
    return headline_from_coverage_index(
        coverage_index,
        variables=plan.members,
        bound=plan.roles.bound,
        role_dcid_by_coverage_role=plan.roles.role_names,
        date_request=date_request,
        served=served_rows(plan),
        excluded=coverage_exclusions(plan),
        first_year=window.start_year if window is not None else None,
        last_year=window.end_year if window is not None else None,
    )


def coverage_exclusions(plan: ReadPlan) -> list[tuple[str, CoverageRole, str]]:
    """The rows a conversion leaves out of a role-bearing `plan` (`ReadPlan.omitted`), as
    `(variable, coverage role, dcid)` for the coverage index to drop. Empty for a plan that
    is not role-bearing."""
    if plan.roles is None:
        return []
    coverage_role = {graph: role for role, graph in plan.roles.role_names.items()}
    return [
        (sv, coverage_role[role], entity)
        for sv, role, entity in plan.omitted
        if role in coverage_role
    ]


def headline_from_coverage_index(
    coverage_index: CoverageIndex,
    *,
    variables: Sequence[str],
    bound: BoundRoles,
    role_dcid_by_coverage_role: Mapping[CoverageRole, str],
    date_request: DateRequest | None,
    served: Iterable[ServedRow],
    excluded: Iterable[tuple[str, CoverageRole, str]] = (),
    first_year: int | None = None,
    last_year: int | None = None,
) -> Headline | None:
    """Build the exact latest-year headline for `variables` under `bound`, counting only the
    `served` rows inside `[first_year, last_year]` less the `excluded` `(variable, role,
    dcid)` rows, so the headline names the year the preview's points reach.

    Return `None` when no candidate survives. Map coverage labels such as
    `donor` and `recipient` to Data Commons role dcids for the wire; omit roles
    absent from the mapping.

    The counts are for the latest year with data inside `[first_year, last_year]`,
    which a read plan sets to the years it reads (`answer_headline`). `named_window` is
    the request's window, a single year, a range or an open range, and `year_named` is
    true whenever there is one; both are unset for a `latest` request and for no
    request.
    """
    result = coverage_index.headline(
        variables=variables,
        bound=bound,
        served=served,
        excluded=excluded,
        first_year=first_year,
        last_year=last_year,
    )
    if result is None:
        return None

    def role_dcid_counts(counts: Mapping[CoverageRole, int]) -> dict[str, int]:
        return {
            role_dcid_by_coverage_role[coverage_role]: count
            for coverage_role, count in counts.items()
            if coverage_role in role_dcid_by_coverage_role
        }

    named_window = date_request.window if date_request is not None else None
    return Headline(
        year=result.year,
        year_named=named_window is not None,
        named_window=named_window,
        observations=result.observations,
        series=result.series,
        role_counts=role_dcid_counts(result.role_counts),
        recent_years=[
            HeadlineYear(
                year=counted.year,
                observations=counted.observations,
                series=counted.series,
                role_counts=role_dcid_counts(counted.role_counts),
            )
            for counted in result.recent_years
        ],
    )
