"""Materialise the alternates the status rule offers beside the elected shape
(`phase_place_materialise._plan_admission`). Each alternate that passes the admission guards
ends admitted, rejected or unevaluated (`_AlternatesAdmission`).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

from search.engine.degraded_reads import carry
from search.engine.region.availability import _shape_group_key
from search.engine.region.candidates import _Alternate, _Candidate, _materialise_alt_shapes
from search.engine.region.codes import ALTERNATES_UNEVALUATED
from search.engine.region.phase_place_materialise import (
    _plan_admission,
    _plan_status,
    _read_alternates_presence,
    _unsettled_places,
    _walked_alternates,
    merge_presence,
    record_unevaluated,
)
from search.engine.region.state import AlternateUnevaluatedReason, ResolveState
from search.engine.retrieval.retrieve import Materialised
from search.models import Warning

_UNEVALUATED_MESSAGES: dict[AlternateUnevaluatedReason, str] = {
    "no_capacity": (
        "Another reading of this question could not be checked because the service was at "
        "capacity, so this answer may not be the only one."
    ),
    "upstream_error": (
        "Another reading of this question could not be checked because a data source "
        "failed, so this answer may not be the only one."
    ),
    "timed_out": (
        "Another reading of this question could not be checked because a data source timed "
        "out, so this answer may not be the only one."
    ),
}


@dataclass(frozen=True)
class _AlternatesAdmission:
    """What admission made of each alternate that passed the guards and the candidate
    budget. An alternate is admitted (materialised with data), rejected (materialised
    without data) or unevaluated, with the reason its presence read did not complete.
    Unread data is no evidence against an unevaluated alternate: it is not offered, and the
    answer states it (`alternates_unevaluated`)."""

    admitted: list[_Candidate] = field(default_factory=list)
    rejected: list[int] = field(default_factory=list)
    unevaluated: dict[int, AlternateUnevaluatedReason] = field(default_factory=dict)
    # Whether the candidate budget clipped the pool.
    candidate_budget_truncated: bool = False
    # Whether a capped alternate blocks the candidate path; `state.mat_result` holds its
    # refusal.
    blocked: bool = False

    @property
    def evaluated(self) -> bool:
        """Whether any alternate was materialised, so the candidate path has shapes to
        ground beside the elected one."""
        return bool(self.admitted or self.rejected)


async def _materialise_candidates(
    state: ResolveState, indices: list[int], *, group: Sequence[int]
) -> tuple[dict[int, _Candidate], dict[int, AlternateUnevaluatedReason], frozenset[int], bool]:
    """Materialise the capped shapes at `indices`, each in its own place context. Returns
    the candidates with data by index, in `indices` order, each with the basis the status
    rule offers it on (`ResolveState.offer_bases`), the indices left unevaluated for
    want of a presence read, each with the reason, the indices dropped for the bound on the
    places a Spec binds, and whether a capped shape blocked the fan-out, in which case
    `state.mat_result` holds its refusal.

    A stand-in that status rule 1 offered is taken as re-election materialised it
    (`state.sibling_candidates`) and is not read again: re-election probed it in the same
    place context, over the same presence read, beside the same elected shape with no
    data. The rest have their role class elected first (`_walked_alternates`), so the
    presence read and the place context belong to the shape each is answered through. The
    presence they need and the request has not read is read here, over those shapes and
    `group` (`_read_alternates_presence`), so only a shape that passed the admission guards
    and is about to be materialised can ask for a read. A shape whose read could not be
    completed is left unevaluated and is neither materialised nor offered. Each shape
    settles its own place sets on its own read (`_materialise_alt_shapes`). A shape whose
    places exceed the bound on the places a Spec binds is dropped, as a shape the admission
    guards reject is. It is not offered or counted as evaluated, since no read of the
    places it would bind found it empty.
    """
    candidates = {i: state.sibling_candidates[i] for i in indices if i in state.sibling_candidates}
    walked = await _walked_alternates(
        state,
        {i: state.capped_shapes[i] for i in indices if i not in candidates},
        elected_has_data=state.elected_has_data,
    )
    shapes = {
        i: (alternate.shape, alternate.alternate.bindings, alternate.roles)
        for i, alternate in walked.items()
    }
    unevaluated = await _read_alternates_presence(state, shapes, group=group)
    unread = [i for i in walked if i not in unevaluated]
    over_bound: frozenset[int] = frozenset()
    if unread:
        places = await _unsettled_places(
            state,
            [(walked[i].shape, walked[i].alternate.bindings, walked[i].roles) for i in unread],
            state.default_presence,
        )
        refused: list[_Alternate] = []
        read = await asyncio.to_thread(
            _materialise_alt_shapes,
            [walked[i].alternate for i in unread],
            places=places,
            over_bound=refused,
            presence=state.default_presence,
            presence_failure=state.alternates_presence_failure,
            named_place_dcids=state.standard_place_dcids,
            graph=state.graph,
            sdmx=state.sdmx,
            date_request=state.date_request,
            deadline=state.deadline,
            coverage=state.coverage,
        )
        merge_presence(state, read.presence)
        # Every alternate the admission reads shapes the offer.
        for _alternate, record in read.consulted:
            carry(record)
        # An alternate whose presence read after unbinding did not complete is unevaluated,
        # as one the read above left without presence is.
        dropped = {
            i: reason
            for i in unread
            for alternate, reason in read.unevaluated
            if walked[i].alternate is alternate
        }
        record_unevaluated(state, dropped)
        unevaluated = {**unevaluated, **dropped}
        over_bound = frozenset(
            i for i in unread if any(walked[i].alternate is alternate for alternate in refused)
        )
        if read.blocking is not None:
            # A capped alternate keeps the ambiguous refusal intact across candidate ordering.
            state.mat_result = read.blocking
            return {}, unevaluated, over_bound, True
        # The role-class election can replace a shape, so a result maps back by its group.
        index_by_group = {_shape_group_key(walked[i].shape): i for i in unread}
        candidates.update(
            {index_by_group[_shape_group_key(cand.shape)]: cand for cand in read.candidates}
        )
    return (
        {
            i: dataclasses.replace(candidates[i], basis=state.offer_bases.get(i))
            for i in indices
            if i in candidates
        },
        unevaluated,
        over_bound,
        False,
    )


def alternates_unevaluated_warning(reason: AlternateUnevaluatedReason) -> Warning:
    """The `alternates_unevaluated` warning, stating the cause `reason` stands for in plain
    words. The machine reason stays on the admission result and in the log line."""
    return Warning(
        code=ALTERNATES_UNEVALUATED, severity="info", message=_UNEVALUATED_MESSAGES[reason]
    )


async def admit_alternates(state: ResolveState) -> _AlternatesAdmission:
    """Materialise the alternates the status rule offers beside the elected shape. Sets
    `state.elected_has_data` for collapse.

    The status rule is planned again now the elected shape's read and the presence reads
    have settled more of D (`_plan_status`), then the admission guards and the candidate
    budget are applied, with no request (`_plan_admission`), so no read is made for an
    alternate they reject and the failure of one cannot reach the answer. Each surviving
    alternate is materialised in its own place context (`place_context`): a subject-only
    alternate with no place takes the default scope of its own StatVars. A subject-only
    elected shape's presence read covers that scope; beside a role-bearing elected shape,
    the read started beside its materialisation does (`_phase_place_bindings`). Presence
    either lacks is read here before the alternates are materialised, on the alternates'
    own pool. A stand-in re-election offered is taken as re-election materialised it
    (`_materialise_candidates`).

    When that read does not complete, for want of a worker, on a source failure or on a
    timeout, the alternates it would cover are unevaluated and the elected answer is served
    with `alternates_unevaluated`. So is an alternate whose values matched no member together
    and whose presence read under the relaxed bindings does not complete
    (`candidates._presence_after_unbinding`).

    An alternate whose places exceed the bound on the places a Spec binds is dropped unread
    and is not listed as rejected.
    """
    state.elected_has_data = (
        isinstance(state.mat_result, Materialised) and state.mat_result.has_data
    )
    state.status_plan = _plan_status(state)
    plan = _plan_admission(state)
    alt_indices = plan.alt_indices
    # Record clipping by the candidate budget, so `CandidateSet.truncated` reflects it even
    # when later assembly keeps fewer candidates.
    candidate_budget_truncated = len(plan.eligible) > len(alt_indices)

    admitted: dict[int, _Candidate] = {}
    unevaluated: dict[int, AlternateUnevaluatedReason] = {}
    evaluated: list[int] = []
    if alt_indices:
        admitted, unevaluated, over_bound, blocked = await _materialise_candidates(
            state, alt_indices, group=alt_indices
        )
        if blocked:
            return _AlternatesAdmission(unevaluated=unevaluated, blocked=True)
        evaluated = [i for i in alt_indices if i not in unevaluated and i not in over_bound]
    return _AlternatesAdmission(
        admitted=list(admitted.values()),
        rejected=[i for i in evaluated if i not in admitted],
        unevaluated=unevaluated,
        candidate_budget_truncated=candidate_budget_truncated,
    )
