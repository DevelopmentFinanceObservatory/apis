"""Decide whether an answer serves a bound unit for every resolved StatVar.

The single-shape path, the candidate path and the resubmit path share this gate. The
answer's read plan, narrowed to the elected units, is the record of what its preview reads
(`retrieval.read_plan`), so every route decides from the plan alone and makes no fetch to
decide. A per-series read (`PairPlan`) confirms the StatVars of the series it serves, and a
bulk read (`BulkPlan`) the members whose elected facet holds a pair with data in their
years (`BulkPlan.member_years`).
The preview, when one is read, reports its own omission and warning beside the verdict.
`bound_unit_gate_reason` is the verdict a fresh request and a resubmit both return.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence

from search.engine.retrieval.read_plan import BulkPlan, ReadPlan


@dataclasses.dataclass(frozen=True)
class UnitEvidence:
    """The StatVars an answer's read plan serves under their elected units within its years.

    The evidence is internal and never reaches the wire.
    """

    confirmed: frozenset[str]

    def covers(self, sv_dcids: Collection[str]) -> bool:
        """Whether every StatVar is confirmed. An empty list is vacuously covered."""
        return set(sv_dcids) <= self.confirmed


def _bulk_served(plan: BulkPlan) -> frozenset[str]:
    """The members whose pairs under the elected facet hold an observation in their years
    (`BulkPlan.member_pairs`)."""
    return frozenset(
        sv
        for sv, c, (first, last) in plan.member_pairs()
        if c.pairs.n_with_data(first_year=first, last_year=last)
    )


def plan_unit_evidence(plan: ReadPlan | None) -> UnitEvidence:
    """The bound-unit evidence `plan`, narrowed to the elected units, gives.

    A bulk read confirms the members it serves (`_bulk_served`). A per-series read confirms
    the StatVars of the series it serves (`PairPlan.served`), those whose group holds a
    point in its years, the same predicate the bulk read applies per pair, and under
    `latest` those at their StatVar's latest period, which every StatVar with data has.
    Either also confirms the StatVars whose rows in its years a conversion left out: their unit holds
    data there, and the omission warning names the rows they lose. A per-series plan omits
    only rows of the series it serves (`ReadPlan.omitted`); a bulk plan records the members
    whose omitted rows fall in their own years (`BulkPlan.omitted_served`). An answer
    without a plan confirms nothing.
    """
    if plan is None:
        return UnitEvidence(confirmed=frozenset())
    if isinstance(plan, BulkPlan):
        return UnitEvidence(confirmed=_bulk_served(plan) | plan.omitted_served)
    omitted = frozenset(sv for sv, _role, _entity in plan.omitted)
    return UnitEvidence(confirmed=frozenset(s.stat_var for s in plan.served) | omitted)


def bound_unit_gate_reason(
    plan: ReadPlan | None,
    obs_entities: Mapping[str, Sequence[str]],
    sv_dcids: Collection[str],
) -> str | None:
    """The no-data reason a bound unit request answers over `plan`, or None when honoured.

    A request with no probe entity cannot honour a unit filter, and otherwise every resolved
    StatVar needs evidence under its elected unit (`plan_unit_evidence`): coverage cannot
    pool across StatVars. Either failure answers ``facet_not_available``.
    """
    if not obs_entities or not plan_unit_evidence(plan).covers(sv_dcids):
        return "facet_not_available"
    return None
