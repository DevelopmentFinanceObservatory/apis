"""Select one shared facet per axis for a `materialise_group` call.

Index-sourced groups agree on `(provenance_id, unit)` and may vary in
`measurementMethod`. Other groups require the same `facet_id`. Members with
no candidate contribute zero rows. Populated members must pass joint checks.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from materialise.errors import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationError,
    MaterialisationIssue,
    MaterialisationUsageError,
)
from materialise.facets import (
    # Reuse facets.py's selection helpers so group and single-member rules
    # share axis matching, alternatives, merging, and currency preference.
    # Narrow bound axes across members before applying currency preference once.
    _MAX_ALTERNATIVES_PER_ISSUE,
    DEPLOYMENT_FACET_POLICY,
    DEPLOYMENT_FACET_POLICY_VERSION,
    FACET_AXES,
    FacetBinding,
    FacetSelection,
    MemberProbe,
    _alternatives,
    _decided_by_precedence,
    _match_policy,
    _narrow_by_bound_axes_only,
    _narrow_by_currency_native_request,
    _preferred_native_unit_selection,
    _require_currency_native_unit,
    _selection,
    dataset_candidates,
    facet_axis_value,
    joint_dispatch_facet_ids,
    merge_facets,
    rank_pool,
    select_bound_facet,
)
from materialise.graph import Facet, FacetDefaultPolicy, GraphReader
from materialise.window import RequestedWindow

if TYPE_CHECKING:
    from materialise.pipeline import MaterialisationRequest


# Index-sourced groups agree on provenance and unit. The index declares one
# CRS provenance, and CRS can split a shared population across facet IDs that
# differ only in measurement method. Other groups require an exact facet ID.
_GROUP_AGREEMENT_AXES: frozenset[str] = frozenset({"provenance", "unit"})


def _group_is_index_sourced(requests: Sequence[MaterialisationRequest]) -> bool:
    """Index-sourced groups use relaxed facet agreement."""
    return all(request.index_sourced for request in requests)


def _facet_id_key(candidate: Facet) -> object:
    return candidate.facet_id


def _agreement_key(candidate: Facet) -> object:
    """Index-sourced groups agree on `(provenance_id, unit)`.
    Merging still requires the same `facet_id`.
    """
    return (candidate.provenance_id, candidate.unit)


def _populated_mask(candidates: Sequence[Sequence[Facet]]) -> list[bool]:
    """Marks members with a candidate `facet_id`, in request order.

    Unpopulated members dispatch no rows. The shared mask keeps candidate
    intersection, policy agreement, and row acceptance aligned.
    """
    return [
        any(c.facet_id is not None for c in member_candidates) for member_candidates in candidates
    ]


def _group_shared_candidates(
    requests: Sequence[MaterialisationRequest],
    candidates: Sequence[Sequence[Facet]],
    *,
    graph: GraphReader,
) -> list[Facet]:
    """Builds the group's unranked shared candidate pool.

    Each populated member's bound axes are checked (`facets.select_bound_facet`) and
    then applied jointly. A failed bound ref adds a member-tagged issue. Members with no candidates
    are exempt from the bound check, and empty probes contribute no keys. Populated members narrowed to an empty
    set make the group refuse. Non-index groups intersect `facet_id`s. Index-sourced
    groups intersect `(provenance_id, unit)` pairs, allowing different
    `measurementMethod` values while retaining shared provenance and units.

    An empty intersection raises one group-level issue per facet axis, with
    alternatives from all candidates. Matching candidates are merged by exact
    `facet_id` (`facets.merge_facets`), summing counts and points. Currency-native preference
    runs after merging to preserve shared facets that member-level currency
    preferences could discard. Results stay unranked and in first-seen order.
    """
    bound_issues: list[MaterialisationIssue] = []
    for request, member_candidates in zip(requests, candidates, strict=True):
        if not member_candidates:
            continue
        for axis in FACET_AXES:
            binding = request.facet_bindings.get(axis, FacetBinding(state="unbound"))
            if binding.state != "bound":
                continue
            try:
                select_bound_facet(axis, binding.ref, candidates=member_candidates, graph=graph)
            except MaterialisationError as exc:
                issue = _as_issue(axis, exc)
                issue.member = request.stat_var
                bound_issues.append(issue)
    if bound_issues:
        raise FacetNotAvailableError(
            "one or more group members lack an observation on a bound facet axis",
            issues=tuple(bound_issues),
        )

    dispatchable = [
        [c for c in member_candidates if c.facet_id is not None] for member_candidates in candidates
    ]
    viable = [
        _narrow_by_bound_axes_only(member_dispatchable, request.facet_bindings)
        for request, member_dispatchable in zip(requests, dispatchable, strict=True)
    ]
    index_sourced = _group_is_index_sourced(requests)
    agreement_key = _agreement_key if index_sourced else _facet_id_key
    # Empty probes contribute no key set. A populated member narrowed to no
    # candidates contributes an empty set and makes the group refuse.
    populated = _populated_mask(candidates)
    populated_keys = [
        {agreement_key(c) for c in member_viable}
        for member_viable, is_populated in zip(viable, populated, strict=True)
        if is_populated
    ]
    shared_keys = set.intersection(*populated_keys) if populated_keys else set()
    if not shared_keys:
        combined_dispatchable = [
            c for member_dispatchable in dispatchable for c in member_dispatchable
        ]
        issues = []
        for axis in FACET_AXES:
            alternatives, truncated = _alternatives(combined_dispatchable, axis, graph)
            exc = FacetNotAvailableError(
                pointer=f"/facets/{axis}",
                alternatives=alternatives,
                alternatives_truncated=truncated,
            )
            issues.append(_as_issue(axis, exc))
        raise FacetNotAvailableError(
            "no probed facet serves every member of the group at once",
            issues=tuple(issues),
        )

    shared: dict[str, list[Facet]] = {}
    for member_viable in viable:
        for candidate in member_viable:
            facet_id = candidate.facet_id
            if facet_id is None or agreement_key(candidate) not in shared_keys:
                continue
            shared.setdefault(facet_id, []).append(candidate)
    merged_pool = [merge_facets(facets) for facets in shared.values()]
    return _narrow_by_currency_native_request(merged_pool, requests[0].facet_bindings)


def _group_declared_policies(
    requests: Sequence[MaterialisationRequest],
    candidates: Sequence[Sequence[Facet]],
    *,
    graph: GraphReader,
) -> dict[str, FacetDefaultPolicy | None]:
    """Each axis the group leaves to selection, mapped to the graph policy every populated
    member declares, or `None` when they declare none or disagree. Bound axes, and an
    index-sourced group's unbound `measurementMethod`, are absent."""
    bindings = requests[0].facet_bindings
    index_sourced = _group_is_index_sourced(requests)
    populated_requests = [
        request
        for request, is_populated in zip(requests, _populated_mask(candidates), strict=True)
        if is_populated
    ]
    policies: dict[str, FacetDefaultPolicy | None] = {}
    for axis in FACET_AXES:
        state = bindings.get(axis, FacetBinding(state="unbound")).state
        if state == "bound" or (index_sourced and axis not in _GROUP_AGREEMENT_AXES):
            continue
        declared = {
            graph.default_facet_policy(stat_var=request.stat_var, axis=axis)
            for request in populated_requests
        }
        policies[axis] = next(iter(declared)) if len(declared) == 1 else None
    return policies


def _resolve_group_axis_selections(
    requests: Sequence[MaterialisationRequest],
    top: Facet,
    shared_pool: Sequence[Facet],
    shared_candidates: Sequence[Facet],
    policies: Mapping[str, FacetDefaultPolicy | None],
    *,
    window: RequestedWindow | None,
    precedence: frozenset[str],
    breadth: Mapping[str, int] | None,
) -> dict[str, FacetSelection]:
    """Resolves every axis from the group's ranked shared candidate.

    `shared_candidates` is `shared_pool`'s fallback pool (`facets.fallback_pool`), so each
    selection reason describes the narrowed pool.

    For unbound axes, the policy in `policies` (`_group_declared_policies`) wins when its
    ref occurs in `shared_pool`: a declared default ignores the window, as it does in
    `select_facet`. Otherwise the selected candidate supplies the value. A missing value remains a `None` selection
    so rows without that axis stay filtered. Provenance must resolve to a ref
    or raises `DefaultFacetUndefinedError`.

    Index-sourced groups omit unbound `measurementMethod` from selections so
    dispatch and row filtering allow members to use different methods. A bound
    method remains a hard constraint. Only populated members affect policy
    agreement. A request binding outside `unit` raises
    `MaterialisationUsageError`.

    A unit request first seeks a native-currency selection in the bound-narrowed
    pool, then uses the ordinary policy and ranking rules. The result must be a
    currency ref (`native_currency_iso`) or `NonCurrencyUnitError` is raised.
    """
    # Bindings agree across members by construction. The representative only
    # supplies selection basis/reason; candidates remain member-specific.
    representative_bindings = requests[0].facet_bindings
    index_sourced = _group_is_index_sourced(requests)
    selections: dict[str, FacetSelection] = {}
    for axis in FACET_AXES:
        binding = representative_bindings.get(axis, FacetBinding(state="unbound"))
        if index_sourced and axis not in _GROUP_AGREEMENT_AXES and binding.state != "bound":
            continue
        value = facet_axis_value(top, axis)
        if binding.state == "bound":
            selections[axis] = _selection(
                axis, value, basis="bound", dispatch_facet_id=top.facet_id
            )
            continue
        # Preserve request metadata on `FacetSelection` for the wire and
        # currency-native validation.
        if binding.state == "request" and axis != "unit":
            raise MaterialisationUsageError(
                f"a request-state facet binding only applies to the unit axis, not {axis!r}"
            )
        request = binding.request if binding.state == "request" else None
        if request is not None:
            preferred = _preferred_native_unit_selection(
                axis,
                request,
                shared_candidates,
                window=window,
                precedence=precedence,
                breadth=breadth,
            )
            if preferred is not None:
                selections[axis] = preferred
                continue
        policy = policies.get(axis)
        if policy is not None:
            match = _match_policy(policy, shared_pool, axis)
            if match is not None:
                selections[axis] = _selection(
                    axis,
                    policy.resolved_ref,
                    basis="declared_default",
                    dispatch_facet_id=match.facet_id,
                    default_policy=policy.policy_node,
                    default_policy_version=policy.policy_version,
                    selection_reason=policy.selection_reason,
                    request=request,
                )
                if request is not None:
                    _require_currency_native_unit(selections[axis].ref)
                continue
        if value is None:
            if axis == "provenance":
                raise DefaultFacetUndefinedError(
                    "facet axis 'provenance' is unbound, no default policy is declared, and "
                    "the winning candidate carries no provenance",
                    pointer="/facets/provenance",
                )
            reason = "absent_on_selected_facet"
        elif len(shared_candidates) == 1:
            reason = "sole_candidate"
        elif all(facet_axis_value(c, axis) == value for c in shared_candidates):
            reason = "most_observations"
        elif _decided_by_precedence(top, shared_candidates, precedence):
            reason = "dfo_precedence"
        else:
            reason = "most_observations_over_disagreement"
        selections[axis] = _selection(
            axis,
            value,
            basis="deployment_fallback",
            dispatch_facet_id=top.facet_id,
            default_policy=DEPLOYMENT_FACET_POLICY,
            default_policy_version=DEPLOYMENT_FACET_POLICY_VERSION,
            request=request,
            selection_reason=reason,
        )
        if request is not None:
            _require_currency_native_unit(selections[axis].ref)

    return selections


def _check_group_joint_coherence(
    requests: Sequence[MaterialisationRequest],
    candidates: Sequence[Sequence[Facet]],
    selections: Mapping[str, FacetSelection],
    *,
    graph: GraphReader,
) -> None:
    """Refuses selections that no populated member can dispatch jointly.

    Declared defaults resolve axes independently, so their combined refs may
    lack a joint candidate. Such members add an issue per facet axis tagged
    with `issue.member`. Empty candidate pools are exempt and contribute zero
    rows. `select_group_facets` calls this check so the currency signal path
    gets the same refusal. `_group_signal` maps it to `None` and rebuilds.
    """
    issues: list[MaterialisationIssue] = []
    for request, member_candidates in zip(requests, candidates, strict=True):
        if not member_candidates or joint_dispatch_facet_ids(member_candidates, selections):
            continue
        dispatchable = [c for c in member_candidates if c.facet_id is not None]
        for axis in FACET_AXES:
            alternatives, truncated = _alternatives(dispatchable, axis, graph)
            issue = _as_issue(
                axis,
                FacetNotAvailableError(
                    pointer=f"/facets/{axis}",
                    alternatives=alternatives,
                    alternatives_truncated=truncated,
                ),
            )
            issue.member = request.stat_var
            issues.append(issue)
    if issues:
        raise FacetNotAvailableError(
            "the group's own shared facet selection resolved, but one or more "
            "members' probed candidates carry no facet matching it jointly",
            issues=tuple(issues),
        )


def group_breadth(
    probes: Sequence[MemberProbe], *, window: RequestedWindow | None
) -> dict[str, int] | None:
    """Each facet id's series count under `window` summed across the group's members
    (`MemberProbe.breadth_in`), which leads the group's ranking as a member's count leads
    its own (`facets.rank_candidates`). None when no member counts series."""
    total: dict[str, int] = {}
    counted = False
    for probe in probes:
        member = probe.breadth_in(window)
        if member is None:
            continue
        counted = True
        for facet_id, count in member.items():
            total[facet_id] = total.get(facet_id, 0) + count
    return total if counted else None


def select_group_facets(
    requests: Sequence[MaterialisationRequest],
    candidates: Sequence[Sequence[Facet]],
    *,
    graph: GraphReader,
    breadth: Mapping[str, int] | None = None,
) -> dict[str, FacetSelection]:
    """Resolves the group's shared selections from its member candidates.

    The function merges shared candidates, resolves each declared default the
    members agree on, keeps the shared candidates carrying those defaults and
    narrows them to the requested window (`facets.fallback_pool`, the pool member
    selection ranks too), ranks them with DFO's first when any member is in the
    origin catalogue's overlap set (`facets.dfo_precedence`, computed on the
    narrowed pool) and led by `breadth`, each facet's series count across the members
    (`group_breadth`), resolves each remaining axis from the top candidate, then
    checks joint coherence before returning. Members share one identity and so
    one window; the first request's window stands for the group, as its bindings
    do. `candidates` follows `requests` order and may come from a precomputed
    probe. The currency signal calls this function too, so it uses the
    dispatch's same selection and refusal rules.

    A member of an index-resolved family contributes only the candidates its
    elected dataset publishes (`facets.dataset_candidates`), so the group's
    provenance is the elected one and its ranking reads only that dataset's
    facets, and a member holding observations under another dataset alone is
    unpopulated. `DatasetDisagreementError` reports a member the probe finds
    only under provenances the code index does not record for it.
    """
    candidates = [
        dataset_candidates(request, member_candidates)
        for request, member_candidates in zip(requests, candidates, strict=True)
    ]
    window = requests[0].window
    shared_pool = _group_shared_candidates(requests, candidates, graph=graph)
    policies = _group_declared_policies(requests, candidates, graph=graph)
    declared = {
        axis: policy.resolved_ref
        for axis, policy in policies.items()
        if policy is not None and _match_policy(policy, shared_pool, axis) is not None
    }
    ranked_pool = rank_pool(
        shared_pool,
        stat_vars=[request.stat_var for request in requests],
        window=window,
        declared=declared,
        breadth=breadth,
    )
    shared_candidates, precedence = ranked_pool.pool, ranked_pool.precedence
    top = ranked_pool.ranked[0]
    selections = _resolve_group_axis_selections(
        requests,
        top,
        shared_pool,
        shared_candidates,
        policies,
        window=window,
        precedence=precedence,
        breadth=breadth,
    )
    _check_group_joint_coherence(requests, candidates, selections, graph=graph)
    return selections


def _as_issue(axis: str, exc: MaterialisationError) -> MaterialisationIssue:
    """Copies the error pointer, alternatives, code, and message into one aggregated issue.

    `truncated` appears only with alternatives, matching the response wire
    shape. It is true when `facets._alternatives` or this backstop applies
    `_MAX_ALTERNATIVES_PER_ISSUE`.
    """
    alternatives = exc.alternatives
    truncated = isinstance(exc, FacetNotAvailableError) and exc.alternatives_truncated
    if alternatives is not None and len(alternatives) > _MAX_ALTERNATIVES_PER_ISSUE:
        alternatives = alternatives[:_MAX_ALTERNATIVES_PER_ISSUE]
        truncated = True
    return MaterialisationIssue(
        axis=axis,
        pointer=exc.pointer,
        alternatives=alternatives,
        truncated=(truncated if alternatives is not None else None),
        code=exc.code,
        detail=str(exc) or None,
    )
