"""Resolve role-bearing shapes by filtering members and confirming observations."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from code_index import absent_readings
from code_index.shared import get_index
from dc_graph import declared_role_refresh_deadline
from materialise.roles import roles_by_axis
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.errors import GraphInfraError, RoleSignatureError
from search.engine.retrieval.bound_values import (
    bound_values_from_bindings,
    declared_role_signature,
    held_family_exact,
    index_open_role_keys,
    shape_member_facts,
    shape_query,
    undeclared_bound_roles,
)
from search.engine.retrieval.coverage import (
    CoverageForm,
    answer_coverage,
    answer_headline,
    plan_election,
)
from search.engine.retrieval.elected_dataset import elected_datasets, published
from search.engine.retrieval.facet_election import merge_role_candidates
from search.engine.retrieval.facet_policy import refuse_declared_defaults
from search.engine.retrieval.member_facts import (
    IndexMemberFacts,
    MemberFacts,
)
from search.engine.retrieval.read_plan import (
    NoYearsToRead,
    PlanRoles,
    ReadPlan,
    RoleYears,
    asked_years,
    in_elected_datasets,
    plan_role_read,
    role_years,
    serves_nothing,
    window_gap,
)
from search.engine.retrieval.role_scope import (
    FACET_AXES,
    check_minted_role_members,
    discover_role_scope,
    facet_axis_filters_from_bound_values,
    facet_from_coverage,
    member_probes,
    pair_series,
    pair_series_count,
    role_index_for_named_entities,
    scope_too_broad,
)
from search.engine.retrieval.standard import _bind_aware_order
from search.engine.shape import ShapeDraft, is_role_bearing, role_names_by_axis
from search.models import CoverageBare
from search.scope_window import WindowTooLong
from statvar_members import UNRESOLVED_NODE, constraints_resolve, elect_role_class

if TYPE_CHECKING:
    from code_index.members import FamilyQuery
    from materialise.roles import RoleAxis
    from search.engine.bind import SlotBindingDraft
    from search.engine.clients.sdmx import EngineSdmxClient
    from search.engine.extract import DateRequest
    from search.engine.retrieval.retrieve import (
        BoundValuesUnrealisable,
        Materialised,
        MemberConflict,
        NoDataDraft,
        UnsupportedDraft,
    )
    from series_coverage.index import BoundRoles, CoverageIndex, Role
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ScopeDecision:
    """Discovery axis and any companion axis defaulted by the same lookup."""

    discovering: bool
    open_axis: RoleAxis | None
    companion_open_axis: RoleAxis | None
    role_open_origin: bool


def _anchor_axis(names: Mapping[RoleAxis, str]) -> RoleAxis:
    """The flow axis a role-bearing answer is confirmed at: destination when the shape places
    a role there, else origin, the member axis of `role_scope.role_index_for_named_entities`.
    A shape whose one role is an origin role, such as a provider-only flow, is confirmed at
    its bound origin."""
    return "destination" if "destination" in names else "origin"


def _binds_an_entity(
    names: Mapping[RoleAxis, str], *, subject_dcids: Sequence[str], origin_dcids: Sequence[str]
) -> bool:
    """Whether the request binds an entity confirmation can start from: one on the shape's
    anchor axis (`_anchor_axis`), or a destination entity, which a shape with no destination
    role refuses `role_incompatible` at confirmation (`_confirm_role_bearing`)."""
    return bool(subject_dcids) or (_anchor_axis(names) == "origin" and bool(origin_dcids))


def _may_bind_an_entity(
    names: Mapping[RoleAxis, str], *, subject_dcids: Sequence[str], origin_dcids: Sequence[str]
) -> bool:
    """Whether the request can bind an entity (`_binds_an_entity`) under some subset of the
    roles in `names`: a destination entity, or an origin entity where `names` places an
    origin role. The members admission elects can declare fewer roles than the shape, and a
    donor-only total of a shape with an optional recipient is confirmed at its bound
    origin."""
    return bool(subject_dcids) or ("origin" in names and bool(origin_dcids))


def _decide_scope(
    *,
    discover_scope: bool,
    names: dict[RoleAxis, str],
    subject_dcids: list[str],
    origin_dcids: Sequence[str],
) -> _ScopeDecision:
    """Choose one open axis to discover when enabled.

    A shape with both roles and no bound entities discovers destination; origin
    defaults as its companion. A bound destination with open origin reads that
    population from the same coverage lookup.
    """
    discovering = False
    open_axis: RoleAxis | None = None
    companion_open_axis: RoleAxis | None = None
    role_open_origin = False
    if discover_scope:
        if "destination" in names and not subject_dcids:
            discovering = True
            open_axis = "destination"
            if "origin" in names and not origin_dcids:
                companion_open_axis = "origin"
        elif (
            "destination" not in names
            and "origin" in names
            and not origin_dcids
            and not subject_dcids
        ):
            discovering = True
            open_axis = "origin"
        elif "destination" in names and subject_dcids and "origin" in names and not origin_dcids:
            role_open_origin = True
    return _ScopeDecision(discovering, open_axis, companion_open_axis, role_open_origin)


def _constraints_resolve(member: MemberFacts, *, excluding: Collection[str] = ()) -> bool:
    """Whether every constraint property `member` declares, apart from those in
    `excluding`, carries a value that resolves to a node, by the mint's walk's own test
    (`statvar_members.constraints_resolve`)."""
    values = {
        **dict.fromkeys(member.read_unresolved_constraints(), UNRESOLVED_NODE),
        **member.read_constraints(),
    }
    declared = [p for p in member.read_declared_constraint_properties() if p not in excluding]
    return constraints_resolve(declared, values)


def _mint_admits(member: MemberFacts, bound_values: Mapping[str, Sequence[str]]) -> bool:
    """Whether the mint's walk can admit `member` under `bound_values`: every constraint it
    declares carries a value that resolves to a node (`_constraints_resolve`), and it
    asserts every bound property at a bound value.

    A declared constraint with no value arc fails, as does one whose value node does not
    resolve, bound or not, and a bound property the member does not assert or holds at
    another value. The index filter (`CodeIndex.filter`), the mint and the resubmit apply
    the same rule, so an answer never lists a member the mint does not serve."""
    if not _constraints_resolve(member):
        return False
    constraints = member.read_constraints()
    return all(constraints.get(prop) in values for prop, values in bound_values.items())


def _mint_conflicts(
    member_facts: Mapping[str, MemberFacts], bound_values: Mapping[str, Sequence[str]]
) -> tuple[MemberConflict, ...]:
    """Each member's conflict with `bound_values` under the mint's rule (`_mint_admits`),
    in population order and without repeats: the bound properties the member asserts at
    another value or does not assert, since the mint serves no member silent on a bound
    property. Unbinding a member's set admits it.

    A member with a declared constraint that does not resolve (`_constraints_resolve`) has
    no conflict here: no unbinding admits it. The result is empty when that is every member.
    """
    from search.engine.retrieval.retrieve import MemberConflict

    conflicts: dict[MemberConflict, None] = {}
    for member in member_facts.values():
        if not _constraints_resolve(member):
            continue
        constraints = member.read_constraints()
        conflicts[
            MemberConflict(
                properties=frozenset(
                    prop
                    for prop, values in bound_values.items()
                    if constraints.get(prop) not in values
                ),
                fully_bound=all(prop in constraints for prop in bound_values),
            )
        ] = None
    return tuple(conflicts)


def _constraint_values(bound_values: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """The bound values admission judges members on. Facet axes describe observations and
    no member asserts one, so they are left out."""
    return {prop: values for prop, values in bound_values.items() if prop not in FACET_AXES}


def unrealisable_bound_values(
    shape: ShapeDraft, query: FamilyQuery, bindings: list[SlotBindingDraft]
) -> BoundValuesUnrealisable | None:
    """The refusal `_admit_population` gives `shape`'s population under `bindings` when no
    member asserts every bound value and unbinding some admits one, else None.

    `query` is the shape's `shape_query` under `bindings`. A caller that learns before any
    place is read that the family holds no member under the query (`MembershipResolution.no_member`)
    unbinds by this and settles the family again, so the members it then reads are the
    family of the bindings it answers under."""
    from search.engine.retrieval.retrieve import BoundValuesUnrealisable

    constraint_values = _constraint_values(bound_values_from_bindings(bindings))
    member_facts, _index_sourced = shape_member_facts(shape, query)
    if any(_mint_admits(member, constraint_values) for member in member_facts.values()):
        return None
    conflicts = _mint_conflicts(member_facts, constraint_values)
    return BoundValuesUnrealisable(conflicts=conflicts) if conflicts else None


def _no_subject_outcome(*, names: dict[RoleAxis, str]) -> NoDataDraft:
    """Name the outcome for a request with no subject and no axis to discover."""
    from search.engine.retrieval.retrieve import NoDataDraft

    if "destination" not in names and "origin" in names:
        # The variable resolved, but this deployment cannot place its role on a flow axis.
        return NoDataDraft(reason="role_not_resolved")
    return NoDataDraft(reason="variable_not_resolved")


def _filter_and_rank_surviving(
    member_facts: dict[str, MemberFacts],
    bound_values: dict[str, list[str]],
) -> list[str]:
    """Keep the members the mint's walk can admit (`_mint_admits`) and rank by fewest
    unbound constraints.

    Index rows already passed the bound filter in ``shape_member_facts``. No
    graph reads occur here.
    """
    surviving_svs: list[str] = []
    for sv_dcid, member in member_facts.items():
        if _mint_admits(member, bound_values):
            surviving_svs.append(sv_dcid)
        else:
            logger.debug(
                "graph_confirm_resolve: dropped sv %r, which does not assert every bound "
                "property at a value in %r",
                sv_dcid,
                bound_values,
            )

    surviving_svs.sort(
        key=lambda dcid: sum(
            1 for prop in member_facts[dcid].read_constraints() if prop not in bound_values
        )
    )
    return surviving_svs


def _refresh_declared_roles(
    surviving_svs: list[str],
    member_facts: dict[str, MemberFacts],
    *,
    index_sourced: bool,
    minted_role_keys: list[str],
    open_role_keys: list[str],
    graph: EngineGraphClient,
    deadline: float | None,
) -> bool:
    """Refresh index roles from the graph; return whether they remain unverified.

    Recall-sourced members and populations that elect under no role key, bound or open,
    need no refresh. The client answers each member the catalogue's arc tables hold from
    them, so a family of any size refreshes; only members outside the tables are read
    live, in the graph client's chunks. On success, live roles replace frozen index
    roles. On failure or timeout, frozen roles remain and the result is marked
    unverified. The deadline-aware scheduler stops dispatching when the request budget
    expires, so partial reads are never treated as verified.
    """
    if not (index_sourced and (minted_role_keys or open_role_keys)):
        return False

    try:
        with declared_role_refresh_deadline(deadline):
            live_properties = graph.node_observation_properties_batch(surviving_svs)
    except (GraphInfraError, TimeoutError):
        logger.warning(
            "graph_confirm_resolve: the declared-role refresh over %d index-sourced "
            "member(s) failed or ran past its deadline; keeping the roles frozen "
            "at export",
            len(surviving_svs),
        )
        return True

    for sv_dcid in surviving_svs:
        member = member_facts[sv_dcid]
        if not isinstance(member, IndexMemberFacts):
            # Index sourcing is all-or-nothing. Leave unexpected members unchanged.
            logger.warning(
                "graph_confirm_resolve: index-sourced population carries a "
                "non-index member %r; its declared roles stay as read",
                sv_dcid,
            )
            continue
        roles = live_properties.get(sv_dcid) or ("observationAbout",)
        member_facts[sv_dcid] = replace(
            member, row=replace(member.row, observation_properties=roles)
        )
    return False


def _drop_undeclared_role_members(
    surviving_svs: list[str],
    member_facts: dict[str, MemberFacts],
    minted_role_keys: list[str],
) -> list[str]:
    """Drop members missing any role this resolve will mint (step 1 of the role rule)."""
    undeclared = undeclared_bound_roles(
        minted_role_keys, {sv_dcid: member_facts[sv_dcid] for sv_dcid in surviving_svs}
    )
    if not undeclared:
        return surviving_svs
    logger.info(
        "graph_confirm_resolve: dropped %d member(s) missing a bound role of %r: %r",
        len(undeclared),
        minted_role_keys,
        undeclared,
    )
    return [sv_dcid for sv_dcid in surviving_svs if sv_dcid not in undeclared]


def _elect_role_class(
    surviving_svs: list[str],
    member_facts: dict[str, MemberFacts],
    role_keys: Collection[str],
    open_role_keys: Collection[str],
) -> list[str] | None:
    """Reduce admitted members to the role class the mint elects, or return `None` on a
    tie.

    The rule is `statvar_members.elect_role_class`, the one the mint, the code-index
    election and the graph walk run: the constraint antichain under each code system's
    `absent_reading`, then the role-twin collapse, so a breakdown over an unbound
    constraint or role gives way to the total that the mint serves. A member whose
    constraints are unknown takes no part in the election and is kept. Admission
    (`_filter_and_rank_surviving`) already keeps only members the mint can admit
    (`_mint_admits`), and `_drop_undeclared_role_members` only members declaring every
    bound role. `open_role_keys` are the open axes no entity binds, which join `role_keys`
    softly (`statvar_members.electing_role_keys`).
    """
    electing = [sv for sv in surviving_svs if member_facts[sv].constraints_known()]
    election = elect_role_class(
        {sv: member_facts[sv].read_constraints() for sv in electing},
        {sv: member_facts[sv].read_declared_roles() for sv in electing},
        electing,
        role_keys=role_keys,
        open_role_keys=open_role_keys,
        readings=absent_readings(get_index()),
    )
    if election.tied:
        return None
    elected = frozenset(election.members)
    dropped = frozenset(electing) - elected
    kept = [sv for sv in surviving_svs if sv not in dropped]
    if dropped:
        logger.info(
            "graph_confirm_resolve: dropped %d member(s) the role-class election reduces: %r",
            len(surviving_svs) - len(kept),
            [sv for sv in surviving_svs if sv in dropped],
        )
    return kept


@dataclass(frozen=True)
class _RoleBearingConfirmation:
    """The role-bearing destination confirmation's non-terminal outcome: a
    real, bound destination confirmed against coverage, ready for the
    coverage/headline assembly that follows in graph_confirm_resolve.
    """

    confirmed_svs: list[str]
    # None when the years asked lie outside the coverage horizon, so nothing is read.
    plan: ReadPlan | None
    facets_by_sv: dict[str, list[Facet]]
    # None when the shape declares no destination role, so none is checked.
    destination_confirmed: set[str] | None
    coverage_index: CoverageIndex
    # When a range read serves no point, each member's lifetime facets at the bound places:
    # data the answer holds outside the years asked.
    held_outside_years: dict[str, list[Facet]]
    # The plan the answer builds over those members with the window removed, or None when
    # none holds data.
    dateless_plan: ReadPlan | None


def _role_plan(
    confirmed_svs: Sequence[str],
    *,
    member_facts: Mapping[str, MemberFacts],
    names: Mapping[RoleAxis, str],
    subject_dcids: Sequence[str],
    origin_dcids: Sequence[str],
    bound_roles: BoundRoles,
    years: RoleYears,
    coverage_index: CoverageIndex,
) -> ReadPlan:
    """Plan a role answer's preview read (`read_plan.plan_role_read`) over `confirmed_svs`,
    members the role-class election kept (`_elect_role_class`), at the entities bound on the
    shape's anchor axis (`_anchor_axis`) and any bound on the other.

    `years` are the ones confirmation used (`read_plan.role_years`). A per-pair plan reads
    each role pair holding a point in those years with its candidate facets from that pair's
    coverage rows (`role_scope.pair_series`), built only once those pairs are counted and fit
    the pair cap (`role_scope.pair_series_count`), and elects each member's one group from
    the coverage probe a mint reads (`role_scope.member_probes`). A member of an index-resolved
    family keeps only its elected dataset's facets (`read_plan.in_elected_datasets`). A bulk
    plan elects its one facet over the
    role index the bulk read builds: the members at the named entities, inside the read's
    year list. Coverage counts the planned facet across every pair of the members, and the
    bulk read reuses it.
    """
    # A pair binds, and the headline names, every role the shape places, bound or open,
    # so an open origin reads one series per donor, as the bulk read serves it.
    by_axis: dict[RoleAxis, Sequence[str]] = {
        "destination": subject_dcids,
        "origin": origin_dcids,
    }
    coverage_role: dict[RoleAxis, Role] = {"destination": "recipient", "origin": "donor"}
    role_names: dict[Role, str] = {
        coverage_role[axis]: names[axis] for axis in ("destination", "origin") if axis in names
    }
    entity_lists = [by_axis[axis] for axis in ("destination", "origin") if by_axis[axis]]
    entities = {
        names[axis]: list(by_axis[axis]) for axis in ("destination", "origin") if axis in names
    }
    time_period = list(years.listed) if years.listed is not None else None
    plan = plan_role_read(
        confirmed_svs,
        entity_lists,
        years=years.years,
        read_years=years.listed,
        candidates=lambda members: merge_role_candidates(
            role_index_for_named_entities(
                members, entities, time_period=time_period, coverage_index=coverage_index
            ),
            members,
            lifetime=coverage_index.facet_spans(variables=members, bound=bound_roles),
        ),
        series_count=lambda members: pair_series_count(
            coverage_index,
            variables=members,
            bound=bound_roles,
            role_names=role_names,
            years=years,
        ),
        pair_series=lambda members: pair_series(
            coverage_index,
            variables=members,
            bound=bound_roles,
            role_names=role_names,
            years=years,
        ),
        member_probes=lambda members: member_probes(
            coverage_index, variables=members, bound=bound_roles
        ),
        roles=PlanRoles(bound=bound_roles, role_names=role_names),
        elect=plan_election(coverage_index),
    )
    return in_elected_datasets(plan, member_facts)


def _confirm_role_bearing(
    *,
    names: dict[RoleAxis, str],
    subject_dcids: list[str],
    origin_dcids: Sequence[str],
    surviving_svs: list[str],
    bindings_admitted_dcids: tuple[str, ...],
    member_facts: dict[str, MemberFacts],
    member_population_exact: bool,
    roles_unverified: bool,
    bound_values: dict[str, list[str]],
    discovering: bool,
    open_axis: RoleAxis | None,
    companion_open_axis: RoleAxis | None,
    date_request: DateRequest | None,
    coverage_snapshot: CoverageSnapshot,
    facet_label: str,
) -> Materialised | NoDataDraft | UnsupportedDraft | _RoleBearingConfirmation:
    """Confirm members against coverage's fixed recipient/donor role labels.

    Members are confirmed at the entities bound on the shape's anchor axis (`_anchor_axis`)
    and any bound on the other axis. Returns a terminal result when a role cannot be placed
    or an open axis is discovered. Otherwise, a matching coverage row confirms an observation
    for that variable and facet. All matching facets are retained; this local
    table lookup has no read cap.
    """
    # Import here to avoid a circular import at module level (retrieve imports role_resolve).
    from search.engine.retrieval.retrieve import Materialised, NoDataDraft

    destination = names.get("destination")
    if destination is None:
        if discovering and open_axis == "origin":
            # Origin-only shapes can discover that axis when it is unbound.
            return discover_role_scope(
                surviving_svs=surviving_svs,
                admitted_dcids=bindings_admitted_dcids,
                member_facts=member_facts,
                member_population_exact=member_population_exact,
                role_declaration_unverified=roles_unverified,
                open_flow_axis="origin",
                role_name=names["origin"],
                opposite_role_name=None,
                opposite_dcids=(),
                facet_filters=facet_axis_filters_from_bound_values(bound_values),
                date_request=date_request,
                coverage_snapshot=coverage_snapshot,
                facet_label=facet_label,
            )
        if subject_dcids or not origin_dcids:
            return NoDataDraft(reason="role_incompatible")
    elif not subject_dcids:
        destination_result = discover_role_scope(
            surviving_svs=surviving_svs,
            admitted_dcids=bindings_admitted_dcids,
            member_facts=member_facts,
            member_population_exact=member_population_exact,
            role_declaration_unverified=roles_unverified,
            open_flow_axis="destination",
            role_name=destination,
            opposite_role_name=names.get("origin"),
            opposite_dcids=origin_dcids,
            facet_filters=facet_axis_filters_from_bound_values(bound_values),
            date_request=date_request,
            coverage_snapshot=coverage_snapshot,
            facet_label=facet_label,
        )
        if companion_open_axis is not None and isinstance(destination_result, Materialised):
            destination_result.also_defaulted_axis = companion_open_axis
        return destination_result

    coverage_index = coverage_snapshot.index
    bound_roles: BoundRoles = {}
    if subject_dcids:
        bound_roles = {"recipient": frozenset(subject_dcids)}
    if origin_dcids:
        bound_roles = {**bound_roles, "donor": frozenset(origin_dcids)}

    # Confirm on the years the plan reads (`read_plan.role_years`), so a member with data
    # only outside them is not confirmed. Latest and absent dates confirm on every year;
    # coverage resolves latest from confirmed facet years.
    years: RoleYears | None
    try:
        years = role_years(date_request, coverage_index.horizon)
    except NoYearsToRead:
        if asked_years(date_request) is None:
            # The range asks for no year at all.
            return NoDataDraft(reason="no_observations", sv_dcids=surviving_svs)
        # The years asked lie outside the coverage horizon, so nothing is read.
        years = None
    except WindowTooLong as too_long:
        return scope_too_broad(axis=_anchor_axis(names), detail=str(too_long))

    # Metadata is keyed by facet_id; missing entries leave fields unknown.
    facet_metadata = coverage_index.facet_metadata
    # A member of an index-resolved family is confirmed by, and judged on, the facets its
    # elected dataset publishes, the ones a mint serves it from. A member the mint refuses
    # over its lifetime facets is refused here before any window narrows them.
    datasets = elected_datasets(member_facts)
    lifetime = check_minted_role_members(
        coverage_index, minted=surviving_svs, member_facts=member_facts, bound=bound_roles
    )
    confirmed_svs: list[str] = []
    facets_by_sv: dict[str, list[Facet]] = {}
    destination_confirmed: set[str] = set()
    plan: ReadPlan | None = None
    if years is not None:
        first_year, last_year = years.bounds
        coverage_by_sv = {
            candidate.variable: candidate
            for candidate in coverage_index.filter_candidates(
                variables=surviving_svs,
                bound=bound_roles,
                first_year=first_year,
                last_year=last_year,
            )
        }
        for sv_dcid in surviving_svs:
            candidate = coverage_by_sv.get(sv_dcid)
            if candidate is None:
                continue
            facets = published(
                datasets,
                sv_dcid,
                (
                    facet_from_coverage(facet, facet_metadata.get(facet.facet_id))
                    for facet in candidate.facets
                ),
            )
            if not facets:
                continue
            destination_confirmed.add(sv_dcid)
            confirmed_svs.append(sv_dcid)
            facets_by_sv[sv_dcid] = facets
        plan = _role_plan(
            confirmed_svs,
            member_facts=member_facts,
            names=names,
            subject_dcids=subject_dcids,
            origin_dcids=origin_dcids,
            bound_roles=bound_roles,
            years=years,
            coverage_index=coverage_index,
        )

    held_outside_years: dict[str, list[Facet]] = {}
    if asked_years(date_request) is not None and (
        plan is None or not confirmed_svs or serves_nothing(plan)
    ):
        held_outside_years = {
            sv: facets
            for sv, found in lifetime.items()
            if (facets := published(datasets, sv, found))
        }
    dateless_plan = (
        _role_plan(
            [sv for sv in surviving_svs if sv in held_outside_years],
            member_facts=member_facts,
            names=names,
            subject_dcids=subject_dcids,
            origin_dcids=origin_dcids,
            bound_roles=bound_roles,
            years=role_years(None, coverage_index.horizon),
            coverage_index=coverage_index,
        )
        if held_outside_years
        else None
    )
    return _RoleBearingConfirmation(
        held_outside_years=held_outside_years,
        dateless_plan=dateless_plan,
        confirmed_svs=confirmed_svs,
        plan=plan,
        facets_by_sv=facets_by_sv,
        destination_confirmed=destination_confirmed if destination is not None else None,
        coverage_index=coverage_index,
    )


@dataclass
class _AdmittedPopulation:
    """The bound-value-filtered, declared-role-verified member population,
    ready for confirmation. `names` maps flow axes to the survivors' agreed
    roles.
    """

    surviving_svs: list[str]
    member_facts: dict[str, MemberFacts]
    names: dict[RoleAxis, str]
    bindings_admitted: int
    bindings_admitted_dcids: tuple[str, ...]
    roles_unverified: bool
    member_population_exact: bool
    open_axes: frozenset[RoleAxis]
    """The open axes the answer's identity names as its open roles: those that joined the
    code index's election of the family, or, for a family the index does not elect, the
    open axes asked for that the survivors declare a role on."""


def _admit_population(
    shape: ShapeDraft,
    *,
    query: FamilyQuery,
    bound_values: dict[str, list[str]],
    names: dict[RoleAxis, str],
    subject_dcids: list[str],
    origin_dcids: Sequence[str],
    open_role_keys: list[str],
    graph: EngineGraphClient,
    deadline: float | None,
) -> NoDataDraft | _AdmittedPopulation:
    """Source, filter, rank, and role-check this shape's members.

    Every member must assert each bound constraint property at a bound value.
    A role-bearing population elects one role class under the axes this resolve
    binds: members must declare every bound role, then the mint's election
    (`_elect_role_class`) under the bound and open axes drops a
    member refining another over an unbound constraint and a breakdown twin over unnamed
    roles, and refuses a tie. ``names`` is recomputed from that
    set, so a shape's optional role that no survivor declares drops out.

    The population read here is one family already: the index's elected rows, a walked
    family, or the recalled members of one role set (`shape_member_facts`). Electing among
    it joins every open axis its role set places. The election that chose the family among
    every candidate can have set one of them aside (`statvar_members.electing_role_keys`),
    and the mint is sure to elect the same dataset and members only under the open keys
    that joined that election (`code_index.members.elect_dataset`). The identity's open
    roles (`_AdmittedPopulation.open_axes`) are therefore those keys where the index
    elected the family (`index_open_role_keys`), and the open axes the survivors declare
    a role on where a walk or the recall did.

    Empty populations return ``NoDataDraft`` with one of four reasons:
    ``variable_not_resolved`` (no members, or none whose unbound constraints all
    resolve), ``binding_mismatch`` (no member asserts every bound value, and some member's
    unbound constraints all resolve; as ``BoundValuesUnrealisable``, naming each member's
    conflict (`_mint_conflicts`), when unbinding values admits one), ``role_incompatible`` (no member declares every bound role, including a
    bound axis the shape declares no role on), or
    ``role_signature_incompatible`` (survivors disagree on role names, or their
    roles share a flow axis).
    """
    from search.engine.retrieval.retrieve import BoundValuesUnrealisable, NoDataDraft

    constraint_values = _constraint_values(bound_values)

    # Use the full index population when available, otherwise the shape's arcs.
    member_facts, index_sourced = shape_member_facts(shape, query)
    if not member_facts:
        return NoDataDraft(reason="variable_not_resolved")

    # Thread exactness because discovery does not receive the ShapeDraft.
    member_population_exact = (
        index_sourced or shape.sv_arc_facts_complete or held_family_exact(shape)
    )

    surviving_svs = _filter_and_rank_surviving(member_facts, constraint_values)
    if not surviving_svs:
        if conflicts := _mint_conflicts(member_facts, constraint_values):
            return BoundValuesUnrealisable(conflicts=conflicts)
        # No unbinding admits a member. A bound value is still mismatched when some member
        # fails only on bound properties: its unbound constraints all resolve.
        return NoDataDraft(
            reason=(
                "binding_mismatch"
                if any(
                    _constraints_resolve(member, excluding=constraint_values)
                    for member in member_facts.values()
                )
                else "variable_not_resolved"
            )
        )

    # Mint a role key for every bound axis of a role-bearing shape, including one
    # the shape declares no role on: its members then fail step 1 of the role rule
    # and the shape is role_incompatible. Discovery handles open axes and excludes
    # candidates that do not declare their role.
    minted_role_keys = [
        axis for axis, dcids in (("destination", subject_dcids), ("origin", origin_dcids)) if dcids
    ]

    roles_unverified = _refresh_declared_roles(
        surviving_svs,
        member_facts,
        index_sourced=index_sourced,
        minted_role_keys=minted_role_keys,
        open_role_keys=open_role_keys,
        graph=graph,
        deadline=deadline,
    )

    # Admission precedes role filtering, so snapshot its count and dcids here.
    bindings_admitted = len(surviving_svs)
    bindings_admitted_dcids = tuple(surviving_svs)

    surviving_svs = _drop_undeclared_role_members(surviving_svs, member_facts, minted_role_keys)
    if not surviving_svs:
        return NoDataDraft(reason="role_incompatible")

    elected = _elect_role_class(surviving_svs, member_facts, minted_role_keys, open_role_keys)
    if elected is None:
        return NoDataDraft(reason="role_signature_incompatible")
    surviving_svs = elected
    try:
        names = roles_by_axis(
            declared_role_signature({sv_dcid: member_facts[sv_dcid] for sv_dcid in surviving_svs})
        )
    except (RoleSignatureError, ValueError):
        return NoDataDraft(reason="role_signature_incompatible")

    joined = index_open_role_keys(query) if index_sourced else None
    minted_open_keys = open_role_keys if joined is None else joined
    return _AdmittedPopulation(
        surviving_svs=surviving_svs,
        member_facts=member_facts,
        names=names,
        bindings_admitted=bindings_admitted,
        bindings_admitted_dcids=bindings_admitted_dcids,
        roles_unverified=roles_unverified,
        member_population_exact=member_population_exact,
        open_axes=frozenset(axis for axis in names if axis in minted_open_keys),
    )


def role_members_served(
    member_facts: dict[str, MemberFacts],
    bound_values: Mapping[str, Sequence[str]],
    role_keys: Collection[str],
    open_role_keys: Collection[str],
) -> list[str]:
    """The members of a role-bearing population the code index does not decide that a
    role-bearing answer lists under `bound_values`, `role_keys` and `open_role_keys`, in
    population order:
    those the mint's walk can admit (`_mint_admits`), reduced to the role class the
    mint elects (`_elect_role_class`). Members the election ties all stay. For a population
    the mint's walk settled (`elected_role_class_shape`), these are the walked family's
    elected members, the ones the mint serves."""
    holding = [sv for sv, member in member_facts.items() if _mint_admits(member, bound_values)]
    return _elect_role_class(holding, member_facts, role_keys, open_role_keys) or holding


def _zero_observation_carrier(
    *,
    surviving_svs: list[str],
    member_facts: dict[str, MemberFacts],
    bound_values: dict[str, list[str]],
    date_request: DateRequest | None,
    open_origin_dcids: frozenset[str] | None,
    roles_unverified: bool,
    bindings_admitted: int,
    bindings_admitted_dcids: tuple[str, ...],
    member_population_exact: bool,
    held_outside_years: Mapping[str, list[Facet]],
    dateless_plan: ReadPlan | None,
    open_axes: frozenset[RoleAxis],
) -> Materialised:
    """Build the carrier for an admitted population with no point in the years read.

    With no data at the place it has one bind-aware representative, no facets, and
    ``available_at_place=False``. With data there in other years (`held_outside_years`, each
    member's lifetime facets), it carries the first such member, its facets and the gap
    `date_request` leaves in `dateless_plan`, the plan the answer builds with its window
    removed (`read_plan.window_gap`), and ``available_at_place=True``:
    the region answers ``no_observations`` for the window. An open-origin result retains its
    discovered members so callers can report the defaulted axis even without observations.
    """
    from search.engine.retrieval.retrieve import Materialised, MemberPopulation

    # Only a range leaves data outside the years read: a latest or every-date read serves any
    # data held, so held data under one makes no window answer.
    years = asked_years(date_request)
    held = {
        sv: list(facets)
        for sv, facets in held_outside_years.items()
        if years is not None and any(f.obs_count > 0 for f in facets)
    }
    representative_sv = (
        next(iter(held))
        if held
        else _bind_aware_order(surviving_svs, member_facts, bound_values)[0]
    )
    return Materialised(
        sv_dcids=[representative_sv],
        facets_by_sv={representative_sv: held[representative_sv]} if held else None,
        has_data=False,
        coverage=CoverageBare(has_data=False, window=date_request.window if date_request else None),
        available_at_place=bool(held),
        window_gap=window_gap(dateless_plan, years) if held and years is not None else None,
        open_origin_dcids=open_origin_dcids,
        open_axes=open_axes,
        role_declaration_unverified=roles_unverified,
        member_facts=member_facts,
        population=MemberPopulation(
            admitted=bindings_admitted,
            admitted_dcids=bindings_admitted_dcids,
            confirmed=0,
            exact=member_population_exact,
        ),
    )


def _assemble_materialised(
    *,
    confirmed_svs: list[str],
    plan: ReadPlan,
    facets_by_sv: dict[str, list[Facet]],
    destination_confirmed: set[str] | None,
    open_origin_dcids: frozenset[str] | None,
    roles_unverified: bool,
    member_facts: dict[str, MemberFacts],
    bindings_admitted: int,
    bindings_admitted_dcids: tuple[str, ...],
    member_population_exact: bool,
    date_request: DateRequest | None,
    facet_label: str,
    obs_label: str,
    coverage_index: CoverageIndex,
    coverage_snapshot: CoverageSnapshot,
    open_axes: frozenset[RoleAxis],
) -> Materialised:
    """Build coverage, headline, and population fields for a data-bearing result.

    Facets retain artifact metadata. The result carries a headline from the
    confirmation's coverage index and every confirmed SV; no narrower winner is
    elected.
    """
    from search.engine.retrieval.retrieve import Materialised, MemberPopulation

    form = CoverageForm(facet_label=facet_label, obs_label=obs_label, variables=True)
    coverage = answer_coverage(plan, form)

    headline = answer_headline(plan, coverage_index, date_request=date_request)

    return Materialised(
        sv_dcids=confirmed_svs,
        has_data=True,
        coverage=coverage,
        coverage_form=form,
        facets_by_sv=facets_by_sv,
        read_plan=plan,
        destination_confirmed=destination_confirmed,
        open_origin_dcids=open_origin_dcids,
        open_axes=open_axes,
        role_declaration_unverified=roles_unverified,
        member_facts=member_facts,
        headline=headline,
        coverage_stamp=coverage_snapshot.stamp,
        population=MemberPopulation(
            admitted=bindings_admitted,
            admitted_dcids=bindings_admitted_dcids,
            confirmed=len(confirmed_svs),
            exact=member_population_exact,
        ),
    )


def graph_confirm_resolve(
    *,
    shape: ShapeDraft,
    bindings: list[SlotBindingDraft],
    subject_dcids: list[str],
    origin_dcids: Sequence[str] | None = None,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    date_request: DateRequest | None = None,
    deadline: float | None = None,
    facet_label: str = "sources",
    obs_label: str = "observations",
    discover_scope: bool = False,
    open_axes: frozenset[RoleAxis] = frozenset(),
    coverage_snapshot: CoverageSnapshot,
) -> Materialised | NoDataDraft | UnsupportedDraft:
    """Filter a role-bearing shape's members by bound values, then confirm observations.

    Member facts come from the release's code index when available, otherwise from
    the shape's fetched arcs. A member must assert every bound constraint
    property at a bound value; a silent, unresolved or out-of-bound value
    fails. Confirmation reads this request's coverage snapshot with fixed
    recipient/donor labels. ``sdmx`` remains in the signature for callers but is
    not read here. A shape declaring no role raises ``ValueError``: a standard shape
    resolves through ``standard.standard_resolve``.

    ``discover_scope`` lets an unbound role axis use coverage-based discovery;
    by default, the resolver reports the corresponding unresolved-role outcome.
    The axes open to discovery come from the admitted members' agreed roles,
    so a shape's optional role opens an axis only when a survivor declares it.
    The same ``coverage_snapshot`` supplies the index and result stamp, avoiding
    reads across an artifact swap.

    ``open_axes`` are the flow axes the request asks for every member of. Each one no
    entity binds joins the role-class election softly, whatever roles the recalled shape
    declares, since recall is a sample of the family. A family with a by-creditor twin
    answers with the twin, and one whose members all lack the axis elects as it would
    without it (`statvar_members.electing_role_keys`). The result's ``open_axes`` holds
    the identity's open roles (`_AdmittedPopulation.open_axes`), as does a no-data
    outcome over the elected members (`NoDataDraft.open_axes`).

    Returns ``Materialised``, ``NoDataDraft``, or ``UnsupportedDraft``. A bound
    destination with no observations returns an unavailable ``Materialised``
    with empty facets and ``population.confirmed == 0``; the other empty-member
    cases return named no-data outcomes.
    """
    # Import here to avoid a circular import at module level (retrieve imports role_resolve).
    from search.engine.retrieval.retrieve import Materialised, NoDataDraft

    if not is_role_bearing(shape):
        raise ValueError(
            f"shape {shape.shape_id!r} declares no role; a standard shape resolves through "
            "standard_resolve"
        )
    origin_dcids = list(origin_dcids or [])
    names = role_names_by_axis(shape)
    # The shape's roles include optional ones the survivors of admission may lack. Its
    # scope can only widen theirs, and their anchor axis can be origin where the shape's is
    # destination, so only a request that binds no entity the survivors could be confirmed
    # at, and has nothing to discover here, is refused before admission.
    if (
        not _may_bind_an_entity(names, subject_dcids=subject_dcids, origin_dcids=origin_dcids)
        and not _decide_scope(
            discover_scope=discover_scope,
            names=names,
            subject_dcids=subject_dcids,
            origin_dcids=origin_dcids,
        ).discovering
    ):
        return _no_subject_outcome(names=names)

    # Both resolvers share bound-value parsing but apply different member filters.
    bound_values = bound_values_from_bindings(bindings)

    bound_axes: list[RoleAxis] = [
        axis
        for axis, dcids in (("destination", subject_dcids), ("origin", origin_dcids))
        if axis in names and dcids
    ]
    open_role_keys: list[str] = sorted(frozenset(open_axes).difference(bound_axes))
    admitted = _admit_population(
        shape,
        query=shape_query(shape, bindings, bound_axes, open_role_keys=open_role_keys),
        bound_values=bound_values,
        names=names,
        subject_dcids=subject_dcids,
        origin_dcids=origin_dcids,
        open_role_keys=open_role_keys,
        graph=graph,
        deadline=deadline,
    )
    if not isinstance(admitted, _AdmittedPopulation):
        return admitted
    # The open axes the identity names. The others stay unnamed.
    elected_open_axes = admitted.open_axes
    surviving_svs = admitted.surviving_svs
    refuse_declared_defaults(graph, surviving_svs)
    member_facts = admitted.member_facts
    names = admitted.names
    # Scope reads the survivors' roles: an optional role only a dropped
    # breakdown declared opens no axis.
    scope = _decide_scope(
        discover_scope=discover_scope,
        names=names,
        subject_dcids=subject_dcids,
        origin_dcids=origin_dcids,
    )
    if (
        not _binds_an_entity(names, subject_dcids=subject_dcids, origin_dcids=origin_dcids)
        and not scope.discovering
    ):
        return _no_subject_outcome(names=names)
    bindings_admitted = admitted.bindings_admitted
    bindings_admitted_dcids = admitted.bindings_admitted_dcids
    roles_unverified = admitted.roles_unverified
    member_population_exact = admitted.member_population_exact

    confirmation = _confirm_role_bearing(
        names=names,
        subject_dcids=subject_dcids,
        origin_dcids=origin_dcids,
        surviving_svs=surviving_svs,
        bindings_admitted_dcids=bindings_admitted_dcids,
        member_facts=member_facts,
        member_population_exact=member_population_exact,
        roles_unverified=roles_unverified,
        bound_values=bound_values,
        discovering=scope.discovering,
        open_axis=scope.open_axis,
        companion_open_axis=scope.companion_open_axis,
        date_request=date_request,
        coverage_snapshot=coverage_snapshot,
        facet_label=facet_label,
    )
    if not isinstance(confirmation, _RoleBearingConfirmation):
        # The elected members declare the open axes whether or not they hold data, so a
        # no-data outcome over them (`NoDataDraft.sv_dcids`) carries the axes too.
        if isinstance(confirmation, Materialised) or (
            isinstance(confirmation, NoDataDraft) and confirmation.sv_dcids
        ):
            confirmation.open_axes = elected_open_axes
        return confirmation
    confirmed_svs = confirmation.confirmed_svs
    plan = confirmation.plan
    coverage_index = confirmation.coverage_index

    # Reuse the destination lookup's index. Read all surviving SVs because
    # donor membership can exist without destination confirmation; retain it
    # on both data-bearing results and the zero-observation carrier.
    open_origin_dcids: frozenset[str] | None = None
    if scope.role_open_origin:
        open_origin_dcids = coverage_index.open_axis(
            variables=surviving_svs, axis="donor", bound={"recipient": frozenset(subject_dcids)}
        )

    # No member confirmed, or none holds a point in the years the plan reads. Data the
    # members hold at the place in other years makes the answer one for the window.
    if plan is None or not confirmed_svs or serves_nothing(plan):
        return _zero_observation_carrier(
            surviving_svs=surviving_svs,
            member_facts=member_facts,
            bound_values=bound_values,
            date_request=date_request,
            open_origin_dcids=open_origin_dcids,
            roles_unverified=roles_unverified,
            bindings_admitted=bindings_admitted,
            bindings_admitted_dcids=bindings_admitted_dcids,
            member_population_exact=member_population_exact,
            held_outside_years=confirmation.held_outside_years,
            dateless_plan=confirmation.dateless_plan,
            open_axes=elected_open_axes,
        )

    return _assemble_materialised(
        confirmed_svs=confirmed_svs,
        plan=plan,
        facets_by_sv=confirmation.facets_by_sv,
        destination_confirmed=confirmation.destination_confirmed,
        open_origin_dcids=open_origin_dcids,
        roles_unverified=roles_unverified,
        member_facts=member_facts,
        bindings_admitted=bindings_admitted,
        bindings_admitted_dcids=bindings_admitted_dcids,
        member_population_exact=member_population_exact,
        date_request=date_request,
        facet_label=facet_label,
        obs_label=obs_label,
        coverage_index=coverage_index,
        coverage_snapshot=coverage_snapshot,
        open_axes=elected_open_axes,
    )
