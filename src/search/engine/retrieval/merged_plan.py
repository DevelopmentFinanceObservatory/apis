"""Rebuild a same-shape merged answer's read plan in memory from its members' plans.

A merged answer elects as one answer does: one facet group across every StatVar it reads
(`read_plan.PairPlan`, a bulk plan's lead facet), and under `latest` one year for each
StatVar across every entity it names (`PairPlan.elected_years`). Its members' counts
therefore do not add up to its own. `merged_read_plan` builds the plan a resolve of the
merged identity builds and runs those elections over it, from what the members' plans and
the request's coverage index hold, so the merged Spec's download, coverage and headline
come from the functions a single answer's come from. It makes no graph read.

- A role-bearing per-series plan reads its series and probes from the coverage index under
  the union of the members' bound roles (`role_scope.pair_series`, `role_scope.member_probes`),
  as a resolve reads them. Each StatVar then keeps the facets its members' plans kept under
  their elected datasets and the request's facet filter (`read_plan.in_elected_datasets`,
  `ReadPlan.narrowed`), each of which keeps or drops a facet by the facet alone. A unit
  request's units, which an answer elects over its own group, are not such a filter, and the
  caller rebuilds no plan for members narrowed by them (`same_shape._merged_counts`).
- A standard per-series plan unions the members' series, already filtered, over the union of
  their places, and elects over probes built from them (`read_plan.series_probes`), as
  `read_plan.plan_standard` does.
- A bulk plan unions the members' candidates, each StatVar's pairs its own, and elects again
  (`read_plan.role_election`).

The rows a conversion leaves out and the members a unit decision leaves out carry over
(`ReadPlan.omitting`, `ReadPlan.leaving_out`).

The plan is None where it cannot be rebuilt without a read the members did not make, or
where the members' plans do not decide it:

- plans of different kinds, years, places-scope, or role names;
- a role-bearing per-series plan with a role it does not bind, an open axis, whose preview
  mode counts members the role-class election kept that no plan holds;
- a role-bearing per-series plan whose StatVars times bound entities, or whose series, pass
  `obs_preview_max_pairs`, which a resolve reads in bulk;
- a role-bearing StatVar holding a facet at the merged entities that no member's plan of it
  held, whose filtering no plan states;
- a standard merge across places whose members read different StatVars, whose series at
  each other's places no plan holds;
- a standard plan leaving rows out for a conversion, whose group was elected over the
  series it then left out (`PairPlan.omitting`), which the plan does not hold;
- a standard merge at one place whose election ties on a variable's whole breadth
  (`materialise.facets.PendingBreadth`), which only a graph read settles;
- bulk plans at different entities or year lists, sharing a StatVar, or leaving rows out for
  a conversion, which a bulk plan elects before it leaves them out (`BulkPlan.omitting`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, NoReturn

from materialise.facets import bound_variable_breadth
from search.engine.retrieval.facet_election import RoleBulkCandidate
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PairPlan,
    PlanRoles,
    PlanSeries,
    ReadPlan,
    RoleYears,
    plan_pairs,
    role_election,
    series_probes,
)
from search.engine.retrieval.role_scope import member_probes, pair_series
from search.observations import observation_settings

if TYPE_CHECKING:
    from materialise.facets import MemberProbe
    from series_coverage.index import BoundRoles, CoverageIndex


def merged_read_plan(
    plans: Sequence[ReadPlan], *, coverage_index: CoverageIndex
) -> ReadPlan | None:
    """The read plan of the same-shape merge of the answers `plans` read, one plan each, or
    None where the module docstring says it cannot be rebuilt. One plan is its own merge."""
    first, *rest = plans
    if not rest:
        return first
    if any(type(plan) is not type(first) or plan.years != first.years for plan in rest):
        return None
    role_names = first.roles.role_names if first.roles is not None else None
    if any(
        (plan.roles.role_names if plan.roles is not None else None) != role_names for plan in rest
    ):
        return None
    if isinstance(first, BulkPlan):
        return _merged_bulk([plan for plan in plans if isinstance(plan, BulkPlan)])
    pairs = [plan for plan in plans if isinstance(plan, PairPlan)]
    merged = (
        _merged_standard(pairs)
        if first.roles is None
        else _merged_role_pairs(pairs, coverage_index=coverage_index)
    )
    if merged is None:
        return None
    return merged.omitting(frozenset().union(*(plan.omitted for plan in pairs))).leaving_out(
        frozenset().union(*(plan.left_out for plan in pairs))
    )


def _members(plans: Sequence[ReadPlan]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(sv for plan in plans for sv in plan.members))


def _merged_role_pairs(
    plans: Sequence[PairPlan], *, coverage_index: CoverageIndex
) -> PairPlan | None:
    """The role-bearing per-series plan of the merge, read from coverage under the union of
    the members' bound roles and kept to the facets the members' plans kept."""
    roles = [plan.roles for plan in plans if plan.roles is not None]
    role_names = roles[0].role_names
    if any(set(r.bound) != set(role_names) for r in roles):
        return None
    bound: BoundRoles = {
        role: frozenset().union(*(r.bound[role] for r in roles)) for role in role_names
    }
    members = _members(plans)
    cap = observation_settings().obs_preview_max_pairs
    if len(members) * math.prod(len(dcids) for dcids in bound.values()) > cap:
        return None
    series = pair_series(
        coverage_index,
        variables=members,
        bound=bound,
        role_names=role_names,
        years=RoleYears(years=plans[0].years, listed=None),
    )
    if len(series) > cap:
        return None
    probes = member_probes(coverage_index, variables=members, bound=bound)
    kept: dict[str, set[str]] = {}
    seen: dict[str, set[str]] = {}
    for plan, plan_roles in zip(plans, roles, strict=True):
        unfiltered = member_probes(coverage_index, variables=plan.members, bound=plan_roles.bound)
        for sv in plan.members:
            seen.setdefault(sv, set()).update(_facet_ids(unfiltered[sv]))
            kept.setdefault(sv, set()).update(
                _facet_ids(plan.probes[sv]) if sv in plan.probes else ()
            )
    if any(not _facet_ids(probe) <= seen.get(sv, set()) for sv, probe in probes.items()):
        return None
    merged = plan_pairs(
        series,
        members=members,
        years=plans[0].years,
        roles=PlanRoles(bound=bound, role_names=role_names),
        probes=probes,
        index_sourced=all(plan.index_sourced for plan in plans),
        elect=plans[0].elect,
    )
    return merged.narrowed(lambda sv, candidate: candidate.facet_id in kept.get(sv, ()))


def _facet_ids(probe: MemberProbe) -> set[str]:
    return {c.facet_id for c in probe.candidates if c.facet_id is not None}


class _BreadthNeeded(Exception):
    """An election of the merged plan tied on a variable's whole breadth."""


def _refuse_breadth(stat_vars: Sequence[str], *, deadline: float | None) -> NoReturn:
    raise _BreadthNeeded


def _merged_standard(plans: Sequence[PairPlan]) -> PairPlan | None:
    """The standard per-series plan of the merge, built from the members' series over the
    union of their places and elected over probes built from them. Each probe's whole breadth
    refuses to read (`_refuse_breadth`), so an election that would read it leaves no plan. The plan
    states the merged answer's counts and is not elected again, so its probes hold no
    breadth a later election could read."""
    first = plans[0]
    if any(plan.scope != first.scope or plan.omitted for plan in plans):
        return None
    places = tuple(dict.fromkeys(place for plan in plans for place in plan.places))
    if any(set(plan.places) != set(places) for plan in plans) and any(
        set(plan.members) != set(first.members) for plan in plans
    ):
        return None
    series: dict[tuple[str, str], PlanSeries] = {}
    for plan in plans:
        for s in plan.series:
            series.setdefault(s.key, s)
    try:
        with bound_variable_breadth(_refuse_breadth, refuse_unread=False):
            return plan_pairs(
                list(series.values()),
                members=_members(plans),
                years=first.years,
                scope=first.scope,
                places=places,
                probes=series_probes(list(series.values()), places=() if first.scope else places),
                index_sourced=all(plan.index_sourced for plan in plans),
            )
    except _BreadthNeeded:
        return None


def _merged_bulk(plans: Sequence[BulkPlan]) -> BulkPlan | None:
    """The bulk plan of the merge of `plans`, which read disjoint StatVars at the same
    entities over the same year list. Their candidates are unioned by facet, each StatVar's
    pairs stay its own, and the facet is elected again."""
    first = plans[0]
    if any(
        plan.roles != first.roles or plan.read_years != first.read_years or plan.omitted
        for plan in plans
    ):
        return None
    members = _members(plans)
    if len(members) != sum(len(plan.members) for plan in plans):
        return None
    candidates: dict[str, RoleBulkCandidate] = {}
    for plan in plans:
        for facet_id, candidate in plan.candidates.items():
            held = candidates.get(facet_id)
            candidates[facet_id] = (
                replace(candidate, by_sv=dict(candidate.by_sv), lifetime=dict(candidate.lifetime))
                if held is None
                else replace(
                    held,
                    by_sv={**held.by_sv, **candidate.by_sv},
                    pair_count=held.pair_count + candidate.pair_count,
                    latest_date=max(
                        (d for d in (held.latest_date, candidate.latest_date) if d), default=None
                    ),
                    lifetime={**held.lifetime, **candidate.lifetime},
                )
            )
    index_sourced = all(plan.index_sourced for plan in plans)
    election = role_election(
        candidates, members=members, read_years=first.read_years, index_sourced=index_sourced
    )
    return BulkPlan(
        members=members,
        years=first.years,
        read_years=first.read_years,
        **election._asdict(),
        candidates=candidates,
        roles=first.roles,
        index_sourced=index_sourced,
        left_out=frozenset().union(*(plan.left_out for plan in plans)),
        elect=first.elect,
    )
