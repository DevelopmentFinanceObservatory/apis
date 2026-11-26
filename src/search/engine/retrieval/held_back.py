"""The runs of partly reported and projected years an answer's read's sources hold, which
Search states on `resolved.coverage.held_back` whatever the read's window and which decide a
named window's opt-in.

The runs are read from facet metadata by the rules the mint records its cut runs by
(`materialise.held_runs.read_runs`), over the entries the mint reads: each member's
candidates merged across the places or role tuples the answer names, and the facet ids the
plan dispatches of them. A per-series read takes each member's probe (`PairPlan.probes`,
the probe a mint builds from its bound places), and a role-bearing one without a probe the
candidates of the coverage probe a mint reads (`materialise.facets.coverage_probe`), read
alone, with the group's selection and continuation for a bulk read. Every bulk plan is
role-bearing (`read_plan.plan_role_read`).

A `partial` run is a served source's partly reported period through the newest year its
served facets hold. A `projected` run holds the years a projection facet continues a served
series past its actual years. Only the mint, which reads the rows, states a partial run's
reported share, so Search states none.

Every run is stated, whatever the read's window. It is served when the window's `include`
names its kind and the window holds a year of it. Several answers merged into one state the
union of their runs (`merged_held_back`).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

from materialise.facets import bound_coverage_facets
from materialise.held_back import HELD_KINDS, cut_catalogue
from materialise.held_runs import HeldRun, merged_runs, read_runs
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PairPlan,
    PlanYears,
    ReadPlan,
    opt_in,
    read_range,
)
from search.models import HeldBack, HeldKind, IdentityWindow, IdentityWindowRange, TimeWindow

if TYPE_CHECKING:
    from search.engine.clients.graph import Facet
    from series_coverage.index import CoverageIndex

__all__ = [
    "HeldRun",
    "contained_kinds",
    "held_back",
    "held_runs",
    "merged_held_back",
    "plan_runs",
    "window_held_back",
]


def _member_reads(
    plan: ReadPlan, coverage_index: CoverageIndex | None
) -> Iterator[tuple[Sequence[Facet], Collection[str]]]:
    """Each member of `plan`'s candidates and the facet ids the plan dispatches of them: its
    probe's (`PairPlan.probes`, else for a role-bearing plan those of the coverage probe a
    mint reads, without the probe's admission and breadth, which no run reads) and its
    served groups, or in bulk the group's selection and continuation."""
    roles = plan.roles
    for sv in plan.members:
        probe = plan.probes.get(sv) if isinstance(plan, PairPlan) else None
        if probe is not None:
            candidates: Sequence[Facet] = probe.candidates
        elif coverage_index is None or roles is None:
            continue
        else:
            candidates = bound_coverage_facets(
                stat_var=sv, bound=roles.bound, coverage=coverage_index
            )
        if isinstance(plan, BulkPlan):
            dispatched = {*plan.selected.get(sv, ()), *plan.continuing}
        else:
            dispatched = {
                *plan.selections.get(sv, ()),
                *(f for s in plan.series if s.stat_var == sv for f in s.group_ids),
            }
        yield candidates, dispatched


def held_runs(plan: ReadPlan, *, coverage_index: CoverageIndex | None) -> tuple[HeldRun, ...]:
    """Every run of held-back years `plan`'s sources hold past its served actual years,
    whatever its window (module docstring), partial runs first.

    Raises:
        CatalogueNotBound: no origin catalogue is bound, so partial marks cannot be read.
    """
    cut_catalogue()
    return read_runs(_member_reads(plan, coverage_index))


def plan_runs(plan: ReadPlan, *, coverage_index: CoverageIndex | None) -> tuple[HeldRun, ...]:
    """The runs `plan` holds as the answer stating it read them (`ReadPlan.held_runs`, set
    by `region.facets.plan_answer`), else read now (`held_runs`), for a plan no answer
    stated, such as one a preview derived from it."""
    if plan.held_runs is not None:
        return plan.held_runs
    return held_runs(plan, coverage_index=coverage_index)


def _overlaps(years: PlanYears, run: HeldRun) -> bool:
    """Whether a read of `years` holds a year of `run`: every date, `latest` (which serves
    each series' newest year, a held run's when the read opts in to it), or a range sharing
    a year with it."""
    if not isinstance(years, TimeWindow):
        return True
    return (years.end_year is None or years.end_year >= run.start_year) and (
        years.start_year is None or years.start_year <= run.end_year
    )


def contained_kinds(window: TimeWindow, runs: Iterable[HeldRun]) -> frozenset[HeldKind]:
    """The kinds of `runs` holding a year inside `window`, the opt-in a window naming those
    years needs to serve them."""
    return frozenset(run.kind for run in runs if _overlaps(window, run))


def held_back(plan: ReadPlan, runs: Iterable[HeldRun]) -> list[HeldBack]:
    """Every run of `runs`, whatever the read's window, as Search states them, each served
    when the window's `include` names its kind (`read_plan.opt_in`) and the window holds a
    year of it, and named by the origin catalogue's provenance table."""
    return _stated(runs, years=plan.years, include=opt_in(plan.years), names=_catalogue_names())


def window_held_back(window: IdentityWindow, runs: Iterable[HeldRun]) -> list[HeldBack]:
    """Every run of `runs`, a composite's (merged from its component plans by
    `region.composite_answer.read_composite_plans`), as Search states them under the
    identity's `window`, as a mint of the composite states them
    (`datasets_api.manifest.held_back_coverage`): each served when the window's `include`
    names its kind and the years a read of the window covers hold a year of it
    (`_window_read`), and named by the origin catalogue's provenance table."""
    years, include = _window_read(window)
    return _stated(runs, years=years, include=include, names=_catalogue_names())


def merged_held_back(window: IdentityWindow, stated: Iterable[HeldBack]) -> list[HeldBack]:
    """The runs several answers state (`stated`, each answer's `Coverage.held_back`) as the
    answer merging them under the identity's `window` states them: merged per kind and
    source (`held_runs.merged_runs`), as one read of the answers' union merges its sources'
    runs, each served as `window_held_back` serves it, and named as the answers named it."""
    held = list(stated)
    names = {h.source: h.source_name for h in held}
    runs = merged_runs(
        HeldRun(kind=h.kind, source=h.source, start_year=h.start_year, end_year=h.end_year)
        for h in held
    )
    years, include = _window_read(window)
    return _stated(runs, years=years, include=include, names=names.get)


def _window_read(window: IdentityWindow) -> tuple[PlanYears, frozenset[HeldKind]]:
    """The years a read of the identity's `window` covers and the kinds it serves: a range
    ending where a read of it ends (`read_plan.read_range`), so an open end stops at the
    clock year as the mint's election does, or every date for an `all` or `latest` window,
    and the kinds its `include` names. A range holding no year serves none."""
    include = frozenset(window.include or ())
    if not isinstance(window, IdentityWindowRange):
        return None, include
    years = read_range(TimeWindow(start_year=window.start_year, end_year=window.end_year))
    return (years, include) if years is not None else (None, frozenset())


def _catalogue_names() -> Callable[[str], str | None]:
    """The name the bound origin catalogue's provenance table gives each source."""
    catalogue = cut_catalogue()
    return lambda source: row.name if (row := catalogue.provenance(source)) else None


def _stated(
    runs: Iterable[HeldRun],
    *,
    years: PlanYears,
    include: frozenset[HeldKind],
    names: Callable[[str], str | None],
) -> list[HeldBack]:
    """`runs` as Search states them, each served when `include` names its kind and a read of
    `years` holds a year of it (`_overlaps`), and named by `names`."""
    assert include <= HELD_KINDS, include
    return [
        HeldBack(
            kind=run.kind,
            served=run.kind in include and _overlaps(years, run),
            source=run.source,
            source_name=names(run.source),
            start_year=run.start_year,
            end_year=run.end_year,
            reported_share=None,
        )
        for run in runs
    ]
