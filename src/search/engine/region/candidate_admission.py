"""Admit `also_plausible` shapes and widen a confirmed near-neighbour tie when its alternate lacks
data. Each alternate that passes the admission guards ends admitted, rejected or unevaluated
(`_AlternatesAdmission`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from search.engine.region.availability import _shape_group_key
from search.engine.region.candidates import _Alternate, _Candidate, _materialise_alt_shapes
from search.engine.region.codes import ALTERNATES_UNEVALUATED, NEAR_NEIGHBOR_POOL_WIDENED
from search.engine.region.near_neighbor import NEAR_NEIGHBOR_UNRESOLVED_MESSAGE
from search.engine.region.phase_place_materialise import (
    _plan_admission,
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
    the candidates with data by index, in `indices` order, the indices left unevaluated for
    want of a presence read, each with the reason, the indices dropped for the bound on the
    places a Spec binds, and whether a capped shape blocked the fan-out, in which case
    `state.mat_result` holds its refusal.

    A sibling that zero-observation re-election offered is taken as re-election materialised
    it (`state.sibling_candidates`) and is not read again: re-election probed it in the
    same place context, over the same presence read, beside the same elected shape with no
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
    return {i: candidates[i] for i in indices if i in candidates}, unevaluated, over_bound, False


def _drop_near_neighbor_tie(state: ResolveState) -> None:
    """Remove the near-neighbour tie's warning and index, so the tied reading is neither
    stated as a candidate nor searched beyond."""
    if state.near_neighbor_warning is not None:
        state.warnings.remove(state.near_neighbor_warning)
    state.near_neighbor_warning = None
    state.near_neighbor_fired_index = None


def alternates_unevaluated_warning(reason: AlternateUnevaluatedReason) -> Warning:
    """The `alternates_unevaluated` warning, stating the cause `reason` stands for in plain
    words. The machine reason stays on the admission result and in the log line."""
    return Warning(
        code=ALTERNATES_UNEVALUATED, severity="info", message=_UNEVALUATED_MESSAGES[reason]
    )


def _keep_tie_unresolved(state: ResolveState) -> None:
    """Give a fired tie's warning the message of a tie that stands unresolved. Candidate
    assembly restores the served labels when both tied readings are offered."""
    if state.near_neighbor_warning is not None:
        unresolved = state.near_neighbor_warning.model_copy(
            update={"message": NEAR_NEIGHBOR_UNRESOLVED_MESSAGE}
        )
        state.warnings[state.warnings.index(state.near_neighbor_warning)] = unresolved
        state.near_neighbor_warning = unresolved


async def admit_alternates(state: ResolveState) -> _AlternatesAdmission:
    """Guard and materialise alternates, widening a confirmed tie when its alternate has no data.
    Sets `state.elected_has_data` for collapse.

    The admission guards and the candidate budget are applied first, with no request
    (`_plan_admission`), so no read is made for an alternate they reject and the failure of
    one cannot reach the answer. Each surviving alternate is materialised in its own place
    context (`place_context`): a subject-only alternate with no place takes the default
    scope of its own StatVars. A subject-only elected shape's presence read covers that
    scope; beside a role-bearing elected shape, the read started beside its
    materialisation does (`_phase_place_bindings`). Presence either lacks is read here
    before the alternates are materialised, with the widened search's shapes when the
    search can run, on the alternates' own pool. A sibling re-election offered is taken as
    re-election materialised it (`_materialise_candidates`).

    When that read does not complete, for want of a worker, on a source failure or on a
    timeout, the alternates it would cover are unevaluated and the elected answer is served
    with `alternates_unevaluated`. So is an alternate whose values matched no member together
    and whose presence read under the relaxed bindings does not complete
    (`candidates._presence_after_unbinding`). A fired tie then stays: its warning and its effect on
    confidence hold (`ResolveState.tie_unresolved`). No search widens past an unevaluated
    tied reading, since nothing read its data.

    An alternate whose places exceed the bound on the places a Spec binds is dropped unread
    and is not listed as rejected. A tie whose other reading is dropped this way is dropped
    with it (`_drop_near_neighbor_tie`).
    """
    state.elected_has_data = (
        isinstance(state.mat_result, Materialised) and state.mat_result.has_data
    )
    plan = _plan_admission(state)
    if state.near_neighbor_fired_index in plan.inadmissible:
        _drop_near_neighbor_tie(state)
    alt_indices = plan.alt_indices
    # Record clipping by either candidate fan-out, so `CandidateSet.truncated` reflects the pool
    # budget even when later assembly keeps fewer candidates.
    _candidate_budget_truncated = len(plan.eligible) > len(alt_indices)

    # A confirmed tie remains ambiguous when its alternate has no data. When the elected shape has
    # data, search beyond `SEARCH_NEAR_MARGIN` for semantically distinct candidates. The no-data
    # path handles an elected shape without data. This search runs only when a tie fired.
    widens = state.near_neighbor_fired_index is not None and state.elected_has_data
    cascade_indices = plan.cascade_indices if widens else []

    admitted: dict[int, _Candidate] = {}
    unevaluated: dict[int, AlternateUnevaluatedReason] = {}
    evaluated: list[int] = []
    if alt_indices:
        admitted, unevaluated, over_bound, blocked = await _materialise_candidates(
            state, alt_indices, group=[*alt_indices, *cascade_indices]
        )
        if blocked:
            return _AlternatesAdmission(unevaluated=unevaluated, blocked=True)
        evaluated = [i for i in alt_indices if i not in unevaluated and i not in over_bound]
        if state.near_neighbor_fired_index in unevaluated:
            widens = False
        if state.near_neighbor_fired_index in over_bound:
            # Nothing read the tied reading, so no search widens past it for want of data.
            _drop_near_neighbor_tie(state)
            widens = False

    if widens and state.near_neighbor_fired_index not in admitted:
        # The cascade has a separate remaining budget and can independently mark the pool
        # truncated.
        _candidate_budget_truncated = _candidate_budget_truncated or (
            len(plan.eligible_cascade) > len(cascade_indices)
        )
        cascade: dict[int, _Candidate] = {}
        if cascade_indices:
            (
                cascade,
                cascade_unevaluated,
                cascade_over_bound,
                blocked,
            ) = await _materialise_candidates(state, cascade_indices, group=cascade_indices)
            unevaluated = {**unevaluated, **cascade_unevaluated}
            if blocked:
                return _AlternatesAdmission(unevaluated=unevaluated, blocked=True)
        if cascade:
            # Later narrowing and deduplication may remove these draft shapes. Candidate labels
            # come from the surviving set, so the warning names the widened search.
            state.warnings.append(
                Warning(
                    code=NEAR_NEIGHBOR_POOL_WIDENED,
                    severity="info",
                    message=(
                        "the chosen reading's near-neighbor alternate had no data; "
                        "the search was widened, and real data was found among the "
                        "candidates shown."
                    ),
                )
            )
            admitted = {**admitted, **cascade}
            evaluated = [
                *evaluated,
                *(
                    i
                    for i in cascade_indices
                    if i not in unevaluated and i not in cascade_over_bound
                ),
            ]

    if unevaluated:
        _keep_tie_unresolved(state)
    return _AlternatesAdmission(
        admitted=list(admitted.values()),
        rejected=[i for i in evaluated if i not in admitted],
        unevaluated=unevaluated,
        candidate_budget_truncated=_candidate_budget_truncated,
    )
