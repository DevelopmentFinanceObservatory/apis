"""Elect one group of facets for a definition's members (`elect_group`), which the mint
(`select_group_facets`) and a read of members alone (`group_served`) take.

Index-sourced groups agree on `(provenance_id, unit)` and may vary in `measurementMethod`.
Other groups require the same `facet_id`. The group is the one serving the most populated
members, and a populated member it does not serve is named. Members with no candidate
contribute zero rows. The members the group serves must pass joint checks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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
    FacetElection,
    FacetSelection,
    MemberProbe,
    RankedPool,
    SeriesCounts,
    VariableBreadth,
    _alternatives,
    _decided_by_precedence,
    _match_policy,
    _narrow_by_bound_axes_only,
    _narrow_by_currency_native_request,
    _preferred_native_unit_selection,
    _require_currency_native_unit,
    _selection,
    cadences_by_member,
    continued_election,
    continuing_facet_ids,
    dataset_candidates,
    facet_axis_value,
    facet_election,
    joined_breadth,
    joint_dispatch_facet_ids,
    joint_matches,
    merge_facets,
    rank_pool,
    ranked_values,
    select_bound_facet,
    summed_counts,
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


def _group_requires_provenance(requests: Sequence[MaterialisationRequest]) -> bool:
    """Whether the group's product must cite a source. The requests share one definition,
    so they agree (`pipeline.MaterialisationRequest.requires_provenance`)."""
    flags = {request.requires_provenance for request in requests}
    assert len(flags) == 1
    return flags.pop()


def _facet_id_key(candidate: Facet) -> object:
    return candidate.facet_id


def _agreement_key(candidate: Facet) -> object:
    """Index-sourced groups agree on `(provenance_id, unit)`.
    Merging still requires the same `facet_id`.
    """
    return (candidate.provenance_id, candidate.unit)


class _Identified(Protocol):
    @property
    def facet_id(self) -> str | None: ...


def is_populated(candidates: Iterable[_Identified]) -> bool:
    """Whether a member holding `candidates` is populated: one carries a `facet_id`. An
    unpopulated member dispatches no rows and is not named unserved (`GroupElection`), and
    Search names a member it leaves out by the same test (`read_plan.populated_members`)."""
    return any(c.facet_id is not None for c in candidates)


def _populated_mask(candidates: Sequence[Sequence[Facet]]) -> list[bool]:
    """Marks members with a candidate `facet_id` (`is_populated`), in request order.

    The shared mask keeps policy agreement and the unserved members aligned.
    """
    return [is_populated(member_candidates) for member_candidates in candidates]


def _check_bound_axes(
    requests: Sequence[MaterialisationRequest],
    candidates: Sequence[Sequence[Facet]],
    *,
    graph: GraphReader,
) -> None:
    """Refuses a group whose populated member lacks an observation on a bound facet axis
    (`facets.select_bound_facet`), with one member-tagged issue per failed bound ref. Members
    with no candidate are exempt."""
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


def _group_pool(
    candidates: Sequence[Sequence[Facet]],
    *,
    bindings: Mapping[str, FacetBinding],
    index_sourced: bool,
) -> tuple[list[Facet], dict[object, frozenset[int]]]:
    """The group's unranked candidate pool, and the members holding each agreement key.

    Each member's dispatchable candidates are narrowed to the bound refs jointly
    (`facets._narrow_by_bound_axes_only`). An index-sourced group's candidates agree on
    `(provenance_id, unit)`, allowing different `measurementMethod` values under one source,
    and any other group's on `facet_id`. The pool keeps the candidates whose key the most
    members hold, so a key every populated member holds wins whenever one exists, and
    otherwise the keys serving the most members compete. A member left with no candidate
    holds no key. The kept candidates are merged by exact `facet_id`
    (`facets.merge_facets`), summing counts and points, in first-seen order, and then
    narrowed to the request's currency (`facets._narrow_by_currency_native_request`), which
    runs after merging so a facet the members share is not lost to one member's own
    currency preference. Empty when no member holds a candidate."""
    agreement_key = _agreement_key if index_sourced else _facet_id_key
    viable = [
        _narrow_by_bound_axes_only([c for c in member if c.facet_id is not None], bindings)
        for member in candidates
    ]
    holders: dict[object, set[int]] = {}
    for member, member_viable in enumerate(viable):
        for candidate in member_viable:
            holders.setdefault(agreement_key(candidate), set()).add(member)
    if not holders:
        return [], {}
    most = max(len(members) for members in holders.values())
    by_id: dict[str | None, list[Facet]] = {}
    for member_viable in viable:
        for candidate in member_viable:
            if len(holders[agreement_key(candidate)]) == most:
                by_id.setdefault(candidate.facet_id, []).append(candidate)
    pool = [merge_facets(facets) for facets in by_id.values()]
    return (
        _narrow_by_currency_native_request(pool, bindings),
        {key: frozenset(members) for key, members in holders.items()},
    )


def _no_group_refusal(
    candidates: Sequence[Sequence[Facet]], *, graph: GraphReader
) -> FacetNotAvailableError:
    """The refusal of a group no member of which holds a candidate surviving its bound refs:
    one group-level issue per facet axis, with alternatives from every member's
    candidates."""
    dispatchable = [c for member in candidates for c in member if c.facet_id is not None]
    issues = []
    for axis in FACET_AXES:
        alternatives, truncated = _alternatives(dispatchable, axis, graph)
        exc = FacetNotAvailableError(
            facet_pointer=f"/facets/{axis}",
            alternatives=alternatives,
            alternatives_truncated=truncated,
        )
        issues.append(_as_issue(axis, exc))
    return FacetNotAvailableError(
        "no probed facet serves any member of the group", issues=tuple(issues)
    )


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
    ranked_pool: RankedPool,
    shared_pool: Sequence[Facet],
    policies: Mapping[str, FacetDefaultPolicy | None],
    *,
    bindings: Mapping[str, FacetBinding],
    index_sourced: bool,
    requires_provenance: bool,
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None,
    variable_breadth: VariableBreadth,
) -> dict[str, FacetSelection]:
    """Resolves every axis from the group's top-ranked shared candidate, the top of
    `ranked_pool`.

    `ranked_pool.pool` is `shared_pool`'s fallback pool (`facets.fallback_pool`), so each
    selection reason describes the narrowed pool.

    For unbound axes, the policy in `policies` (`_group_declared_policies`) wins when its
    ref occurs in `shared_pool`: a declared default ignores the window, as it does in
    `select_facet`. Otherwise the selected candidate supplies the value. A missing value remains a `None` selection
    so rows without that axis stay filtered. Provenance must resolve to a ref, or raises
    `DefaultFacetUndefinedError`, under `requires_provenance`
    (`pipeline.MaterialisationRequest.requires_provenance`).

    Index-sourced groups omit unbound `measurementMethod` from selections so
    dispatch and row filtering allow members to use different methods. A bound
    method remains a hard constraint. Only populated members affect policy
    agreement. A request binding outside `unit` raises
    `MaterialisationUsageError`.

    A unit request first seeks a native-currency selection in the bound-narrowed
    pool, then uses the ordinary policy and ranking rules. The result must be a
    currency ref (`native_currency_iso`) or `NonCurrencyUnitError` is raised.
    """
    shared_candidates, precedence = ranked_pool.pool, ranked_pool.precedence
    top = ranked_pool.ranked[0]
    selections: dict[str, FacetSelection] = {}
    for axis in FACET_AXES:
        binding = bindings.get(axis, FacetBinding(state="unbound"))
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
                variable_breadth=variable_breadth,
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
            if axis == "provenance" and requires_provenance:
                raise DefaultFacetUndefinedError(
                    "facet axis 'provenance' is unbound, no default policy is declared, and "
                    "the winning candidate carries no provenance",
                    facet_pointer="/facets/provenance",
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
            ranked_values=ranked_values(
                ranked_pool, axis, reason=reason, window=window, breadth=breadth
            ),
        )
        if request is not None:
            _require_currency_native_unit(selections[axis].ref)

    return selections


def _coherence_refusal(
    requests: Sequence[MaterialisationRequest],
    candidates: Sequence[Sequence[Facet]],
    members: Sequence[int],
    *,
    graph: GraphReader,
) -> FacetNotAvailableError:
    """The refusal of selections that `members`, members the elected group serves
    (`GroupElection.incoherent`), cannot dispatch jointly: declared defaults resolve axes
    independently, so their combined refs may lack a joint candidate. Each such member adds
    an issue per facet axis tagged with `issue.member`."""
    issues: list[MaterialisationIssue] = []
    for member in members:
        dispatchable = [c for c in candidates[member] if c.facet_id is not None]
        for axis in FACET_AXES:
            alternatives, truncated = _alternatives(dispatchable, axis, graph)
            issue = _as_issue(
                axis,
                FacetNotAvailableError(
                    facet_pointer=f"/facets/{axis}",
                    alternatives=alternatives,
                    alternatives_truncated=truncated,
                ),
            )
            issue.member = requests[member].stat_var
            issues.append(issue)
    return FacetNotAvailableError(
        "the group's own shared facet selection resolved, but one or more "
        "members' probed candidates carry no facet matching it jointly",
        issues=tuple(issues),
    )


def group_breadth(
    probes: Sequence[MemberProbe], *, window: RequestedWindow | None
) -> SeriesCounts | None:
    """Each facet id's series count under `window` summed across the group's members
    (`MemberProbe.breadth_in`, `facets.summed_counts`), which leads the group's ranking as a
    member's count leads its own (`facets.rank_candidates`). None when no member counts
    series."""
    counted = [b for probe in probes if (b := probe.breadth_in(window)) is not None]
    return summed_counts(counted) if counted else None


def group_variable_breadth(probes: Sequence[MemberProbe]) -> VariableBreadth:
    """Each facet id's places across the whole of each member's variable, summed across the
    group's members (`MemberProbe.variable_breadth`), which breaks the group's tie before the
    unit as a member's own does (`facets.rank_candidates`), joined without reading it
    (`facets.joined_breadth`): pending members stay one pending read, which the group's tie
    alone starts. A `BreadthUnread` when any member's is, so a group needing it acts on it as
    a member would; otherwise None unless every member carries one, so a member without one
    leaves the group's tie to the unit's name."""
    return joined_breadth([probe.variable_breadth for probe in probes])


@dataclass(frozen=True)
class GroupElection:
    """One group's election (`elect_group`): the selections and cadence every member
    dispatches under (`election`), and, per member in request order, the facet ids its
    dispatch serves (`dispatch_ids`, `facets.joint_dispatch_facet_ids`), those of them that
    continue the selected series past its last year (`continuing_ids`,
    `facets.continuing_facet_ids`), the populated members holding no facet that matches the
    selections, which serve nothing (`unserved`), and the members holding the elected
    group's agreement key that hold no such facet either (`incoherent`), which the mint
    refuses (`select_group_facets`)."""

    election: FacetElection
    dispatch_ids: tuple[tuple[str, ...], ...]
    continuing_ids: tuple[tuple[str, ...], ...]
    unserved: tuple[int, ...] = ()
    incoherent: tuple[int, ...] = ()

    def selected(self, member: int) -> frozenset[str]:
        """The facet ids member `member` dispatches for the selection itself: its dispatch
        ids less those continuing the selected series."""
        return frozenset(self.dispatch_ids[member]) - frozenset(self.continuing_ids[member])

    def served(self, member: int, candidates: Sequence[Facet]) -> tuple[Facet, ...]:
        """Member `member`'s `candidates` its dispatch serves, in serving order: the
        selection's facets, then those continuing them, each in candidate order."""
        continuing = set(self.continuing_ids[member])
        selected = self.selected(member)
        return tuple(c for c in candidates if c.facet_id in selected) + tuple(
            c for c in candidates if c.facet_id in continuing
        )


def elect_group(
    members: Sequence[Sequence[Facet]],
    *,
    stat_vars: Sequence[str],
    window: RequestedWindow | None,
    index_sourced: bool,
    requires_provenance: bool,
    bindings: Mapping[str, FacetBinding] | None = None,
    policies: Mapping[str, FacetDefaultPolicy | None] | None = None,
    electable: Sequence[Sequence[Facet]] | None = None,
    breadth: Mapping[str, int] | None = None,
    variable_breadth: VariableBreadth = None,
) -> GroupElection | None:
    """Elects the one group a definition's members serve under, or None when no member
    holds a candidate surviving `bindings`. It reads nothing but a pending
    `variable_breadth` a tie needs.

    `members` holds each member's candidates as its dispatch reads them, in `stat_vars`
    order, and `electable` those the election may rank (`facets.dataset_candidates`),
    `members` when None. The group is the agreement key (`_group_pool`) held by the most
    members: every populated member when one key serves them all, and otherwise the key
    serving the most, the rest named in `GroupElection.unserved`. Among those keys' merged
    candidates, `bindings` and the declared defaults of `policies` (`_group_declared_policies`)
    found among them are kept and narrowed to `window` (`facets.fallback_pool`, the pool
    member selection ranks too), then ranked with DFO's first when any member is in the
    origin catalogue's overlap set and DFO's broadest facet holds at least as many of the
    members' series as any other facet (`facets.dfo_precedence`), led by `breadth`, each
    facet's series count summed across the members (`group_breadth`), with
    `variable_breadth`, each facet's places across the members' whole variables
    (`group_variable_breadth`), breaking a tie that remains before the unit. Each remaining
    axis resolves from the top candidate (`_resolve_group_axis_selections`).

    The group's cadence is the one at which the most members hold a candidate matching every
    selected ref, counting under a range window only candidates with a point in it when any
    member has one. Ties go to the cadence of the facet a counted member holds that the same
    ranking puts first among the members' joint matches merged by id, then to the earliest
    member (`facets.dispatch_cadence`, `_merged_joint_matches`). Every member dispatches only
    its facets at that cadence, so one group serves one observation period, and a member
    holding none serves nothing. Each member's copies of a facet id are judged apart
    (`facets.cadences_by_member`), so an id stating no period that one member dates by year
    and another by month dispatches only for the members whose copy is at the elected
    cadence. A range window running past the group's served series takes the same source's
    estimates for the later years, as a member's election does
    (`facets.continued_election`)."""
    bindings = bindings or {}
    policies = policies or {}
    electable = members if electable is None else electable
    pool, holders = _group_pool(electable, bindings=bindings, index_sourced=index_sourced)
    if not pool:
        return None
    declared = {
        axis: policy.resolved_ref
        for axis, policy in policies.items()
        if policy is not None and _match_policy(policy, pool, axis) is not None
    }
    ranked_pool = rank_pool(
        pool,
        stat_vars=stat_vars,
        window=window,
        declared=declared,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )
    selections = _resolve_group_axis_selections(
        ranked_pool,
        pool,
        policies,
        bindings=bindings,
        index_sourced=index_sourced,
        requires_provenance=requires_provenance,
        window=window,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )
    agreement_key = _agreement_key if index_sourced else _facet_id_key
    incoherent = tuple(
        member
        for member in sorted(holders[agreement_key(ranked_pool.ranked[0])])
        if not joint_matches(electable[member], selections)
    )
    election = facet_election(
        selections,
        matching=_merged_joint_matches(members, selections),
        member_matches=[joint_matches(member, selections) for member in members],
        cadence_of=cadences_by_member(members),
        stat_vars=stat_vars,
        window=window,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )
    election = continued_election(election, members, window=window, breadth=breadth)
    return GroupElection(
        election=election,
        dispatch_ids=tuple(
            joint_dispatch_facet_ids(member, election, member=index)
            for index, member in enumerate(members)
        ),
        continuing_ids=tuple(
            continuing_facet_ids(member, election, member=index)
            for index, member in enumerate(members)
        ),
        unserved=tuple(
            index
            for index, (member, populated) in enumerate(
                zip(members, _populated_mask(electable), strict=True)
            )
            if populated and not joint_matches(member, election.selections)
        ),
        incoherent=incoherent,
    )


def rank_group(
    members: Sequence[Sequence[Facet]],
    *,
    stat_vars: Sequence[str],
    window: RequestedWindow | None,
    index_sourced: bool,
    breadth: Mapping[str, int] | None = None,
    variable_breadth: VariableBreadth = None,
) -> RankedPool | None:
    """The ranking `elect_group` runs over `members` when no axis is bound and no default is
    declared: the merged candidates of the agreement keys held by the most members
    (`_group_pool`), ranked by `rank_pool`. Its top is the facet such an election leads
    with. None when no member holds a candidate. A caller serving one facet across every
    member reads its top without the cadence and continuation an election adds."""
    pool, _holders = _group_pool(members, bindings={}, index_sourced=index_sourced)
    if not pool:
        return None
    return rank_pool(
        pool,
        stat_vars=stat_vars,
        window=window,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )


def elect_probes(
    probes: Mapping[str, MemberProbe],
    *,
    window: RequestedWindow | None,
    index_sourced: bool = False,
) -> GroupElection | None:
    """The one group the members of `probes`, keyed by StatVar, serve under `window` when no
    axis is bound and the graph declares no default (`elect_group`, members in `probes`
    order), led by the series each facet holds (`group_breadth`), with a tie broken on each
    facet's places across the whole variables (`group_variable_breadth`). A top carrying no
    provenance serves, since such a read need not cite a source. None when no candidate
    carries a `facet_id`."""
    return elect_group(
        [probe.candidates for probe in probes.values()],
        stat_vars=list(probes),
        window=window,
        index_sourced=index_sourced,
        requires_provenance=False,
        breadth=group_breadth(list(probes.values()), window=window),
        variable_breadth=group_variable_breadth(list(probes.values())),
    )


def group_served(
    probes: Mapping[str, MemberProbe],
    *,
    window: RequestedWindow | None,
    index_sourced: bool = False,
) -> dict[str, tuple[Facet, ...]]:
    """The facets each member of `probes`, keyed by StatVar, serves under `window` in the
    group `elect_probes` elects, in serving order (`GroupElection.served`). A member holding
    no facet of the group serves none, and every member serves none when no candidate
    carries a `facet_id`."""
    group = elect_probes(probes, window=window, index_sourced=index_sourced)
    if group is None:
        return dict.fromkeys(probes, ())
    return {
        stat_var: group.served(index, probe.candidates)
        for index, (stat_var, probe) in enumerate(probes.items())
    }


def select_group_facets(
    requests: Sequence[MaterialisationRequest],
    candidates: Sequence[Sequence[Facet]],
    *,
    graph: GraphReader,
    breadth: Mapping[str, int] | None = None,
    variable_breadth: VariableBreadth = None,
) -> GroupElection:
    """The mint's election of one group across `requests`' members (`elect_group`), from
    their candidates in request order, which may come from a precomputed probe. Members
    share one identity and so one window and one set of bindings; the first request's stand
    for the group.

    A member of an index-resolved family contributes only the candidates its elected dataset
    publishes (`facets.dataset_candidates`), so the group's provenance is the elected one and
    its ranking reads only that dataset's facets, and a member holding observations under
    another dataset alone is unpopulated. `DatasetDisagreementError` reports a member the
    probe finds only under provenances the code index does not record for it.

    Raises `FacetNotAvailableError` when a populated member lacks an observation on a bound
    axis (`_check_bound_axes`), when no member holds a candidate carrying the bound refs
    jointly (`_no_group_refusal`), and when a member holding the elected group carries no
    facet matching its selections jointly (`_coherence_refusal`)."""
    electable = [
        dataset_candidates(request, member_candidates)
        for request, member_candidates in zip(requests, candidates, strict=True)
    ]
    _check_bound_axes(requests, electable, graph=graph)
    group = elect_group(
        candidates,
        stat_vars=[request.stat_var for request in requests],
        window=requests[0].window,
        index_sourced=_group_is_index_sourced(requests),
        requires_provenance=_group_requires_provenance(requests),
        bindings=requests[0].facet_bindings,
        policies=_group_declared_policies(requests, electable, graph=graph),
        electable=electable,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )
    if group is None:
        raise _no_group_refusal(electable, graph=graph)
    if group.incoherent:
        raise _coherence_refusal(requests, electable, group.incoherent, graph=graph)
    return group


def _merged_joint_matches(
    candidates: Sequence[Sequence[Facet]], selections: Mapping[str, FacetSelection]
) -> list[Facet]:
    """Every member's `joint_matches`, merged by facet id (`merge_facets`) in first-seen
    order: the facets the group's members dispatch from. The shared pool can hold none of
    them when declared defaults resolve apart, while each member still holds its own."""
    by_id: dict[str | None, list[Facet]] = {}
    for member_candidates in candidates:
        for facet in joint_matches(member_candidates, selections):
            by_id.setdefault(facet.facet_id, []).append(facet)
    return [merge_facets(facets) for facets in by_id.values()]


def _as_issue(axis: str, exc: MaterialisationError) -> MaterialisationIssue:
    """Copies the error's facet pointer, alternatives, code, and message into one aggregated
    issue.

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
        pointer=exc.facet_pointer,
        alternatives=alternatives,
        truncated=(truncated if alternatives is not None else None),
        code=exc.code,
        detail=str(exc) or None,
    )
