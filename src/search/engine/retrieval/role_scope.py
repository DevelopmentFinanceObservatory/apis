"""Discover open role scopes and build role indexes from coverage.

Named-entity reads use the same index builder. Coverage lookups are local table
filters. They cannot trip member caps, so
scope refusals apply only to requested year windows. Unit, source, and method
filters are recorded for echo because coverage has no per-row facet metadata.

Role indexes group real ``pair_rows`` by variable and facet. Real pairs avoid
inventing donor-recipient combinations. A fully open query with 104 x 166
members produced a 13.6 million-pair cross product; a bare ODA query has
1.64 million real pairs, held as Arrow rows. The index carries facet metadata;
previews read observation values separately. Data-bearing results include an
exact latest-year headline.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from dfo_platform.roles import GENERIC_ROLE_WORDING
from materialise.facets import MemberProbe, coverage_probe
from materialise.roles import RoleAxis, axis_for_role
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import RoleFacetCandidate
from search.engine.retrieval.coverage import (
    CoverageForm,
    answer_coverage,
    answer_headline,
    plan_election,
    scope_members,
)
from search.engine.retrieval.elected_dataset import (
    check_elected_datasets,
    elected_datasets,
    published,
    published_by_stat_var,
)
from search.engine.retrieval.facet_election import merge_role_candidates
from search.engine.retrieval.read_plan import (
    NoYearsToRead,
    PlanRoles,
    PlanSeries,
    ReadPlan,
    RoleYears,
    asked_years,
    in_elected_datasets,
    plan_role_read,
    role_years,
    window_gap,
)
from search.models import CoverageBare, EntityRoleKey
from search.scope_window import WindowTooLong
from series_coverage.schema import ABSENT_ENTITY

if TYPE_CHECKING:
    from search.engine.clients.sdmx import RoleTuple
    from search.engine.extract import DateRequest
    from search.engine.retrieval.member_facts import MemberFacts
    from search.engine.retrieval.retrieve import Materialised, NoDataDraft, UnsupportedDraft
    from series_coverage.index import BoundRoles, CoverageIndex, FacetCoverage, PairRows
    from series_coverage.index import Role as CoverageRole
    from series_coverage.schema import FacetMetadata
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

# Facet axes accepted by role-scope discovery. Unit and measurementMethod
# currently have no bind-time slot axis. They describe observations, so no
# StatVar asserts them and member admission skips them.
FACET_AXES = ("unit", "provenance", "measurementMethod")


# Public label for the discovered scope's citable-source dimension.
DEFAULT_FACET_LABEL = "sources"


# Coverage indexes call the destination recipient and the origin donor.
_COVERAGE_ROLE_FOR_AXIS: dict[RoleAxis, CoverageRole] = {
    "destination": "recipient",
    "origin": "donor",
}


def _opposite_flow_axis(axis: RoleAxis) -> RoleAxis:
    return "origin" if axis == "destination" else "destination"


@dataclass(frozen=True)
class DiscoveredScope:
    """Resolved scope and filters for an open role axis.

    ``axis`` is origin or destination; ``role_name`` is the graph role sent on
    the wire. ``member_dcids`` are the axis's members holding data in the years
    read, dcid-sorted: the population a reader can name on the axis, which later
    probes and the preview bind. The members the answer serves are the read
    plan's (``coverage.scope_members``), on ``Materialised.scope_member_dcids``,
    a subset when the plan elects one facet in bulk. ``filters`` records bound roles and
    facets for echo, but only the opposite-role filter narrows coverage. The
    preview's read plan, with its years and the sources the answer cites, travels
    on ``Materialised.read_plan``.
    """

    axis: EntityRoleKey
    role_name: str
    member_dcids: list[str]
    filters: dict[str, list[str]]


def facet_axis_filters_from_bound_values(
    bound_values: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """The subset of a bound_values-shaped dict (graph_confirm_resolve's own
    property_dcid -> bound value_dcids map, built from `bindings`) that
    names a facet axis (unit, provenance, measurementMethod) role-scope
    discovery's envelope can filter an availability read by. An axis absent
    from bound_values is omitted here too.
    """
    return {axis: list(bound_values[axis]) for axis in FACET_AXES if bound_values.get(axis)}


def open_role_axis_members(
    role_index_by_sv: Mapping[str, Sequence[RoleFacetCandidate]],
    *,
    role_name: str,
    svs: Sequence[str],
) -> frozenset[str]:
    """Return each ``role_name`` dcid in the indexed candidates for ``svs``.

    Role tuples carry both axes, so this reads the opposite-axis members at
    no extra SDMX cost. Results are complete only when the role index is
    complete; the named-destination builder raises on a cap trip.
    """
    members: set[str] = set()
    for sv in svs:
        for candidate in role_index_by_sv.get(sv, ()):
            members.update(candidate.members_by_role.get(role_name, ()))
    return frozenset(members)


def _entity_role_key(role_name: str) -> EntityRoleKey:
    """Map a role name to its flow axis; callers must pass a placed role."""
    flow_axis = axis_for_role(role_name)
    if flow_axis is None:
        raise AssertionError(
            f"role name {role_name!r} reached role-scope discovery with no flow axis; "
            "a discovery caller must only ever pass a role_names_by_axis-placed name"
        )
    return flow_axis


def _bound_role_filter(role_name: str | None, dcids: Sequence[str]) -> dict[str, list[str]]:
    """Return a role filter only when both the role and its dcids are present."""
    if role_name and dcids:
        return {role_name: list(dcids)}
    return {}


_SCOPE_TOO_BROAD_REMEDY = (
    "name an entity on this axis, narrow the requested time window, or bind "
    "a source/unit/method facet to narrow the scope"
)


def scope_too_broad(*, axis: RoleAxis, detail: str) -> UnsupportedDraft:
    """The role routes' window refusal: a requested window whose year list passes
    `scope_max_window_years` (`search.scope_window.WindowTooLong`), reported before any coverage
    lookup runs."""
    from search.engine.retrieval.retrieve import UnsupportedDraft

    return UnsupportedDraft(
        code="scope_too_broad",
        pointer=f"/definition/identity/entities/{axis}",
        detail=f"{detail}; {_SCOPE_TOO_BROAD_REMEDY}",
    )


def _role_tuple_for_pair(
    *, role_name: str, member: str, opposite_role_name: str | None, opposite: str
) -> RoleTuple:
    """Build a sorted tuple for one real pair, or just ``member`` if unpaired.

    An unbound opposite role still names the dcid in this observed pair.
    """
    if opposite_role_name:
        return tuple(sorted(((role_name, member), (opposite_role_name, opposite))))
    return ((role_name, member),)


@dataclass(frozen=True)
class _CoverageRolePairs:
    """clients.sdmx.RolePairs over one (variable, facet_id)'s coverage rows
    in a PairRows lookup, turned into role tuples only when read."""

    rows: PairRows
    variable: str
    facet_id: str
    role_name: str
    opposite_role_name: str | None
    member_is_donor: bool

    def head(
        self,
        limit: int,
        *,
        first_year: int | None,
        last_year: int | None,
        with_data: bool = False,
    ) -> list[tuple[RoleTuple, int]]:
        """Decode only the requested ``limit`` rows from a pair group."""
        has_opposite = self.opposite_role_name is not None
        member_sorts_first = (
            self.opposite_role_name is None or self.role_name < self.opposite_role_name
        )
        pairs = self.rows.head_pairs(
            variable=self.variable,
            facet_id=self.facet_id,
            limit=limit,
            first_year=first_year,
            last_year=last_year,
            member_is_donor=self.member_is_donor,
            has_opposite=has_opposite,
            member_sorts_first=member_sorts_first,
            with_data=with_data,
        )
        return [(self._role_tuple(member, opposite), n_obs) for member, opposite, n_obs in pairs]

    def year_counts(
        self, *, first_year: int | None, last_year: int | None
    ) -> list[tuple[RoleTuple, tuple[tuple[int, int], ...]]]:
        pairs = self.rows.pair_year_counts(
            variable=self.variable,
            facet_id=self.facet_id,
            first_year=first_year,
            last_year=last_year,
            member_is_donor=self.member_is_donor,
            has_opposite=self.opposite_role_name is not None,
        )
        return [(self._role_tuple(member, opposite), counts) for member, opposite, counts in pairs]

    def _role_tuple(self, member: str, opposite: str | None) -> RoleTuple:
        """The role tuple of a row binding `member` on this pairs' role and `opposite`, or
        no opposite, on the other."""
        if opposite is None:
            return ((self.role_name, member),)
        return _role_tuple_for_pair(
            role_name=self.role_name,
            member=member,
            opposite_role_name=self.opposite_role_name,
            opposite=opposite,
        )

    def n_obs(self, *, first_year: int | None, last_year: int | None) -> int:
        return self.rows.n_obs(
            variable=self.variable,
            facet_id=self.facet_id,
            first_year=first_year,
            last_year=last_year,
        )

    def n_with_data(self, *, first_year: int | None, last_year: int | None) -> int:
        return self.rows.rows_with_data(
            variable=self.variable,
            facet_id=self.facet_id,
            first_year=first_year,
            last_year=last_year,
        )

    def max_n_obs(self, *, first_year: int | None, last_year: int | None) -> int:
        return self.rows.max_n_obs(
            variable=self.variable,
            facet_id=self.facet_id,
            first_year=first_year,
            last_year=last_year,
        )

    def year_span(self, *, first_year: int | None, last_year: int | None) -> tuple[int, int] | None:
        return self.rows.year_span(
            variable=self.variable,
            facet_id=self.facet_id,
            first_year=first_year,
            last_year=last_year,
        )

    def members_with_data(
        self, *, first_year: int | None, last_year: int | None, role: str | None = None
    ) -> frozenset[str]:
        on_member_role = role is None or role == self.role_name
        return self.rows.members_with_data(
            variable=self.variable,
            facet_id=self.facet_id,
            member_is_donor=self.member_is_donor if on_member_role else not self.member_is_donor,
            first_year=first_year,
            last_year=last_year,
        )

    def excluding(self, *, role: str, dcids: frozenset[str]) -> _CoverageRolePairs:
        """These pairs over rows without those binding one of `dcids` on `role`
        (`PairRows.excluding`). A role these pairs do not carry leaves them unchanged."""
        if role == self.role_name:
            member_is_donor = self.member_is_donor
        elif role == self.opposite_role_name:
            member_is_donor = not self.member_is_donor
        else:
            return self
        column: CoverageRole = "donor" if member_is_donor else "recipient"
        return replace(self, rows=self.rows.excluding(role=column, dcids=dcids))


def _role_index_from_coverage(
    confirmed_svs: Sequence[str],
    rows: PairRows,
    *,
    facet_metadata: Mapping[str, FacetMetadata],
    role_name: str,
    opposite_role_name: str | None,
    member_is_donor: bool,
) -> dict[str, list[RoleFacetCandidate]]:
    """Build candidates by variable and facet from real coverage pairs.

    Each candidate carries its exact members, pair count, and facet metadata.
    Pair groups stay columnar; ``head`` decodes only its requested rows. A
    single-role shape counts members instead of pairs, and different facets
    remain separate candidates. ``member_is_donor`` selects the open-axis column.
    """
    role_index_by_sv: dict[str, list[RoleFacetCandidate]] = {sv: [] for sv in confirmed_svs}
    for group in rows.groups():
        candidates = role_index_by_sv.get(group.variable)
        if candidates is None:
            continue
        members, opposites = (
            (group.donors, group.recipients)
            if member_is_donor
            else (group.recipients, group.donors)
        )
        members_by_role = {role_name: members}
        if opposite_role_name:
            members_by_role[opposite_role_name] = opposites
        meta = facet_metadata.get(group.facet_id)
        candidates.append(
            RoleFacetCandidate(
                facet_id=group.facet_id,
                members_by_role=members_by_role,
                pair_count=group.pair_count if opposite_role_name else len(members),
                latest_date=str(group.last_year),
                pairs=_CoverageRolePairs(
                    rows=rows,
                    variable=group.variable,
                    facet_id=group.facet_id,
                    role_name=role_name,
                    opposite_role_name=opposite_role_name,
                    member_is_donor=member_is_donor,
                ),
                provenance_id=meta.provenance_id if meta else None,
                measurement_method=meta.measurement_method if meta else None,
                unit=meta.unit if meta else None,
                scaling_factor=meta.scaling_factor if meta else None,
                observation_period=meta.observation_period if meta else None,
            )
        )
    return role_index_by_sv


def role_index_for_named_entities(
    sv_dcids: Sequence[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    time_period: Sequence[str] | None,
    coverage_index: CoverageIndex,
) -> dict[str, list[RoleFacetCandidate]]:
    """Build a role index for caller-named entities from coverage.

    ``obs_entities`` maps declared role names to resolved dcids. Each non-empty
    role filters coverage; empty roles stay unbound. ``pair_rows`` supplies
    observed pairs to the shared index builder. An optional year list narrows
    the lookup by its first and last years. Reuse the request's
    ``coverage_index`` so the result shares its artifact snapshot.
    """
    role_by_axis: dict[RoleAxis, str] = {}
    for role_name in obs_entities:
        axis = axis_for_role(role_name)
        if axis is not None:
            role_by_axis[axis] = role_name
    member_axis: RoleAxis = "destination" if "destination" in role_by_axis else "origin"
    assert member_axis in role_by_axis, (
        f"obs_entities {sorted(obs_entities)!r} names no role this deployment's flow-axis "
        "table places on either the origin or destination axis"
    )
    opposite_axis = _opposite_flow_axis(member_axis)
    member_role = role_by_axis[member_axis]
    opposite_role = role_by_axis.get(opposite_axis)
    member_coverage_role = _COVERAGE_ROLE_FOR_AXIS[member_axis]

    first_year = int(time_period[0]) if time_period is not None else None
    last_year = int(time_period[-1]) if time_period is not None else None

    # Omit empty roles; an empty BoundRoles set would match no rows.
    bound: dict[CoverageRole, frozenset[str]] = {}
    member_set = frozenset(obs_entities.get(member_role) or ())
    if member_set:
        bound[member_coverage_role] = member_set
    if opposite_role is not None:
        opposite_set = frozenset(obs_entities.get(opposite_role) or ())
        if opposite_set:
            bound[_COVERAGE_ROLE_FOR_AXIS[opposite_axis]] = opposite_set
    rows = coverage_index.pair_rows(
        variables=list(sv_dcids), bound=bound, first_year=first_year, last_year=last_year
    )
    return _role_index_from_coverage(
        sv_dcids,
        rows,
        facet_metadata=coverage_index.facet_metadata,
        role_name=member_role,
        opposite_role_name=opposite_role,
        member_is_donor=member_coverage_role == "donor",
    )


def lifetime_role_facets(
    coverage_index: CoverageIndex, *, variables: Sequence[str], bound: BoundRoles
) -> dict[str, list[Facet]]:
    """Every facet each of `variables` holds at the `bound` roles over every year, as a
    mint's coverage probe reads them (`materialise.facets.coverage_probe`): no window
    narrows them. A variable holding none is absent."""
    facet_metadata = coverage_index.facet_metadata
    return {
        candidate.variable: [
            facet_from_coverage(facet, facet_metadata.get(facet.facet_id))
            for facet in candidate.facets
        ]
        for candidate in coverage_index.filter_candidates(variables=variables, bound=bound)
    }


def check_minted_role_members(
    coverage_index: CoverageIndex,
    *,
    minted: Sequence[str],
    member_facts: Mapping[str, MemberFacts],
    bound: BoundRoles,
) -> dict[str, list[Facet]]:
    """Check each of `minted`, the members a mint of a role-bearing answer serves, as the
    mint checks it (`check_elected_datasets`), over its lifetime facets at `bound`, and
    return those facets."""
    lifetime = lifetime_role_facets(coverage_index, variables=minted, bound=bound)
    check_elected_datasets(elected_datasets(member_facts), lifetime)
    return lifetime


def facet_from_coverage(facet: FacetCoverage, meta: FacetMetadata | None) -> Facet:
    """Convert a coverage candidate to a dated facet.

    ``obs_count`` is the candidate's observations and ``dates`` its years with data, one
    entry per year whatever the year's count, as a mint's coverage probe carries them
    (`materialise.facets.coverage_probe`). A candidate summed across every role tuple of a
    family therefore holds at most one date per horizon year. An election ranks it by
    ``obs_count`` and a window check reads the years. Facet metadata, the observation period
    included, is copied from the release's coverage view when present. The years are coverage's,
    whatever the facet's cadence, so they are no evidence that it is annual
    (``dates_coarsened``).
    """
    dates = tuple(str(year) for year, count in sorted(facet.year_counts.items()) if count)
    return Facet(
        earliest_date=str(facet.first_year),
        latest_date=str(facet.last_year),
        obs_count=facet.n_obs,
        dates=dates,
        dates_complete=True,
        dates_coarsened=True,
        facet_id=facet.facet_id,
        provenance_id=meta.provenance_id if meta else None,
        measurement_method=meta.measurement_method if meta else None,
        unit=meta.unit if meta else None,
        scaling_factor=meta.scaling_factor if meta else None,
        observation_period=meta.observation_period if meta else None,
    )


def pair_series(
    coverage_index: CoverageIndex,
    *,
    variables: Sequence[str],
    bound: BoundRoles,
    role_names: Mapping[CoverageRole, str],
    years: RoleYears,
) -> list[PlanSeries]:
    """Each role pair's series as the per-pair preview binds it, for the pairs holding a
    point in `years` (`pair_series_count` counts them): its variable at one dcid on each role
    in `role_names` (coverage role to graph role, destination first) that the variable
    declares, its candidate facets over every year from that pair's coverage rows, which the
    plan elects under `years`, and its recipient and donor (`None` for a role it does not
    bind). A variable declaring no donor holds `ABSENT_ENTITY` there, which it neither binds
    nor names."""
    facet_metadata = coverage_index.facet_metadata
    pair_roles = tuple(role_names)
    first_year, last_year = years.bounds
    per_pair = coverage_index.pair_facet_candidates(
        variables=variables,
        bound=bound,
        pair_roles=pair_roles,
        first_year=first_year,
        last_year=last_year,
    )
    series: list[PlanSeries] = []
    for (variable, dcids), coverages in sorted(per_pair.items()):
        by_role = {
            role: dcid
            for role, dcid in zip(pair_roles, dcids, strict=True)
            if dcid != ABSENT_ENTITY
        }
        series.append(
            PlanSeries(
                stat_var=variable,
                role_tuple=tuple(
                    sorted((role_names[role], dcid) for role, dcid in by_role.items())
                ),
                candidates=tuple(
                    facet_from_coverage(facet, facet_metadata.get(facet.facet_id))
                    for facet in coverages
                ),
                rows=(by_role.get("recipient"), by_role.get("donor")),
            )
        )
    return series


def pair_series_count(
    coverage_index: CoverageIndex,
    *,
    variables: Sequence[str],
    bound: BoundRoles,
    role_names: Mapping[CoverageRole, str],
    years: RoleYears,
) -> int:
    """How many series `pair_series` builds for the same arguments: the pairs holding a
    point in `years`, counted from coverage without building one (`CoverageIndex.pair_count`).
    A per-pair plan serves at most these series."""
    first_year, last_year = years.bounds
    return coverage_index.pair_count(
        variables=variables,
        bound=bound,
        pair_roles=tuple(role_names),
        first_year=first_year,
        last_year=last_year,
    )


def member_probes(
    coverage_index: CoverageIndex, *, variables: Sequence[str], bound: BoundRoles
) -> dict[str, MemberProbe]:
    """Each variable's election input for a per-pair plan: the coverage probe a mint of the
    answer reads under the same bound roles (`materialise.facets.coverage_probe`), its
    candidates across every role tuple and year and each facet's series count."""
    return {
        variable: coverage_probe(stat_var=variable, bound=bound, coverage=coverage_index)
        for variable in variables
    }


@dataclass(frozen=True)
class _ScopeRead:
    """An open axis read under one set of years: its members holding data in them, the
    variables with a role candidate, their role index and the preview's read plan."""

    members: list[str]
    confirmed_svs: list[str]
    role_index_by_sv: dict[str, list[RoleFacetCandidate]]
    plan: ReadPlan | None


def _read_scope(
    coverage_index: CoverageIndex,
    *,
    surviving_svs: list[str],
    member_facts: Mapping[str, MemberFacts],
    axis: CoverageRole,
    bound: BoundRoles,
    role_name: str,
    opposite_role_name: str | None,
    opposite_dcids: Sequence[str],
    years: RoleYears,
) -> _ScopeRead:
    """Read the open `axis` under `years` and plan its preview once: the members the
    role-class election kept (`surviving_svs`) that hold data, read per pair within the pair
    cap, one bulk facet elected over the discovered index above it. Each variable elects one
    group across its pairs from its facets over every year, as a named read does, and a pair
    with a point in the years serves the facets of that group it holds. `plan` is None when
    no member or no variable holds data in the years."""
    opposite_coverage_role: CoverageRole = "donor" if axis == "recipient" else "recipient"
    # A pair binds, and the headline names, every role the shape places, bound or open.
    role_names: dict[CoverageRole, str] = {axis: role_name}
    if opposite_role_name:
        role_names[opposite_coverage_role] = opposite_role_name
    first_year, last_year = years.bounds
    members = sorted(
        coverage_index.open_axis(
            variables=surviving_svs,
            axis=axis,
            bound=bound,
            first_year=first_year,
            last_year=last_year,
        )
    )
    if not members:
        return _ScopeRead(members, [], {}, None)
    rows = coverage_index.pair_rows(
        variables=surviving_svs, bound=bound, first_year=first_year, last_year=last_year
    )
    # A member of an index-resolved family is read from its elected dataset's facets alone.
    surviving_index = published_by_stat_var(
        elected_datasets(member_facts),
        _role_index_from_coverage(
            surviving_svs,
            rows,
            facet_metadata=coverage_index.facet_metadata,
            role_name=role_name,
            opposite_role_name=opposite_role_name,
            member_is_donor=axis == "donor",
        ),
    )
    confirmed_svs = [sv for sv in surviving_svs if surviving_index[sv]]
    if not confirmed_svs:
        return _ScopeRead(members, [], {}, None)
    role_index_by_sv = {sv: surviving_index[sv] for sv in confirmed_svs}
    entity_lists = [members, list(opposite_dcids)] if opposite_dcids else [members]
    plan = plan_role_read(
        confirmed_svs,
        entity_lists,
        years=years.years,
        read_years=years.listed,
        candidates=lambda read: merge_role_candidates(
            role_index_by_sv,
            read,
            lifetime=coverage_index.facet_spans(variables=read, bound=bound),
        ),
        series_count=lambda read: pair_series_count(
            coverage_index, variables=read, bound=bound, role_names=role_names, years=years
        ),
        pair_series=lambda read: pair_series(
            coverage_index, variables=read, bound=bound, role_names=role_names, years=years
        ),
        member_probes=lambda read: member_probes(coverage_index, variables=read, bound=bound),
        roles=PlanRoles(bound=bound, role_names=role_names),
        elect=plan_election(coverage_index),
    )
    plan = in_elected_datasets(plan, member_facts)
    return _ScopeRead(members, confirmed_svs, role_index_by_sv, plan)


def _no_point_in_years(
    coverage_index: CoverageIndex,
    *,
    surviving_svs: list[str],
    admitted_dcids: Sequence[str],
    axis: CoverageRole,
    bound: BoundRoles,
    role_name: str,
    opposite_role_name: str | None,
    opposite_dcids: Sequence[str],
    date_request: DateRequest | None,
    member_facts: dict[str, MemberFacts],
    member_population_exact: bool,
    role_declaration_unverified: bool,
) -> Materialised | None:
    """The carrier for an open axis with members only outside the years a range asks for, as
    a named destination's (`role_resolve._zero_observation_carrier`): the first variable
    holding data, its lifetime facets across those members and the gap `date_request` leaves
    in the plan the same scope builds with the window removed (`read_plan.window_gap` over
    `_read_scope` on every year). None for a latest or dateless request, or when no member
    holds data in any year. The population states `admitted_dcids`, as `discover_role_scope`
    does."""
    from search.engine.retrieval.retrieve import Materialised, MemberPopulation

    years = asked_years(date_request)
    if years is None:
        return None
    members = coverage_index.open_axis(variables=surviving_svs, axis=axis, bound=bound)
    if not members:
        return None
    held = coverage_index.filter_candidates(variables=surviving_svs, bound={**bound, axis: members})
    facet_metadata = coverage_index.facet_metadata
    datasets = elected_datasets(member_facts)
    by_sv = {
        c.variable: facets
        for c in held
        if (
            facets := published(
                datasets,
                c.variable,
                (facet_from_coverage(f, facet_metadata.get(f.facet_id)) for f in c.facets),
            )
        )
    }
    sv = next((sv for sv in surviving_svs if sv in by_sv), None)
    if sv is None:
        return None
    dateless = _read_scope(
        coverage_index,
        surviving_svs=surviving_svs,
        member_facts=member_facts,
        axis=axis,
        bound=bound,
        role_name=role_name,
        opposite_role_name=opposite_role_name,
        opposite_dcids=opposite_dcids,
        years=role_years(None, coverage_index.horizon),
    )
    return Materialised(
        sv_dcids=[sv],
        facets_by_sv={sv: by_sv[sv]},
        has_data=False,
        coverage=CoverageBare(has_data=False, window=date_request.window if date_request else None),
        available_at_place=True,
        window_gap=window_gap(dateless.plan, years),
        role_declaration_unverified=role_declaration_unverified,
        member_facts=member_facts,
        population=MemberPopulation(
            admitted=len(admitted_dcids),
            admitted_dcids=tuple(admitted_dcids),
            confirmed=0,
            exact=member_population_exact,
        ),
    )


def discover_role_scope(
    *,
    surviving_svs: list[str],
    admitted_dcids: Sequence[str],
    member_facts: dict[str, MemberFacts],
    member_population_exact: bool,
    role_declaration_unverified: bool = False,
    open_flow_axis: RoleAxis,
    role_name: str,
    opposite_role_name: str | None,
    opposite_dcids: Sequence[str],
    facet_filters: Mapping[str, Sequence[str]],
    date_request: DateRequest | None,
    coverage_snapshot: CoverageSnapshot,
    facet_label: str = DEFAULT_FACET_LABEL,
) -> Materialised | NoDataDraft | UnsupportedDraft:
    """Discover members on an open role axis and resolve them from coverage.

    The caller supplies the already-filtered ``member_facts``, population
    exactness, and role-refresh status; this function narrows only by coverage.
    ``surviving_svs`` are the members read: those the role-class election kept.
    ``admitted_dcids`` are the members the bindings admit before that election: the
    family the Datasets mint resolves for the identity where the code index decides the
    shape's membership, and the recalled members the bindings admit elsewhere. The
    population states them (``MemberPopulation.admitted``), as a bound destination's does.
    It returns the window carrier (`_no_point_in_years`) when the axis has members
    only outside the years asked, including years outside the coverage horizon, and
    ``NoDataDraft`` when the axis has no member in any year, the range asks for no
    year, or no SV has a role candidate. A window above the configured year cap
    returns ``UnsupportedDraft(code="scope_too_broad")`` before lookup.

    The request's single ``coverage_snapshot`` supplies both index and stamp.
    The lookup, the read plan and its year list share one set of years
    (`read_plan.role_years`): floored at ``max(scope_min_year, horizon.epoch_year)`` and
    capped at ``horizon.max_year``, which carries projection years. Coverage, the window it
    reports and the headline are stated from the plan (`coverage.answer_coverage`). The
    lookup confirms only members holding a point in those years, so the plan always serves
    one.

    Opposite-role and time filters narrow the lookup. Bound facet filters are
    recorded for echo only because coverage has no per-row facet metadata.
    Candidates come from real ``pair_rows`` with facet metadata; each confirmed
    SV has at least one candidate. The returned scope preserves the caller's
    member facts and role-verification status.
    """
    from search.engine.retrieval.retrieve import (
        Materialised,
        MemberPopulation,
        NoDataDraft,
    )

    years: RoleYears | None
    try:
        years = role_years(date_request, coverage_snapshot.index.horizon)
    except NoYearsToRead:
        # The years asked lie outside the coverage horizon, or the range holds none: no
        # member is read in them.
        years = None
    except WindowTooLong as too_long:
        return scope_too_broad(axis=open_flow_axis, detail=str(too_long))

    filters: dict[str, list[str]] = {
        **_bound_role_filter(opposite_role_name, opposite_dcids),
        **{axis: list(dcids) for axis, dcids in facet_filters.items() if dcids},
    }

    coverage_index = coverage_snapshot.index
    discover_coverage_role = _COVERAGE_ROLE_FOR_AXIS[open_flow_axis]
    opposite_coverage_role = _COVERAGE_ROLE_FOR_AXIS[_opposite_flow_axis(open_flow_axis)]
    bound: BoundRoles = (
        {opposite_coverage_role: frozenset(opposite_dcids)} if opposite_dcids else {}
    )
    check_minted_role_members(
        coverage_index, minted=surviving_svs, member_facts=member_facts, bound=bound
    )
    read = (
        _read_scope(
            coverage_index,
            surviving_svs=surviving_svs,
            member_facts=member_facts,
            axis=discover_coverage_role,
            bound=bound,
            role_name=role_name,
            opposite_role_name=opposite_role_name,
            opposite_dcids=opposite_dcids,
            years=years,
        )
        if years is not None
        else None
    )
    if read is None or not read.members:
        return _no_point_in_years(
            coverage_index,
            surviving_svs=surviving_svs,
            admitted_dcids=admitted_dcids,
            axis=discover_coverage_role,
            bound=bound,
            role_name=role_name,
            opposite_role_name=opposite_role_name,
            opposite_dcids=opposite_dcids,
            date_request=date_request,
            member_facts=member_facts,
            member_population_exact=member_population_exact,
            role_declaration_unverified=role_declaration_unverified,
        ) or NoDataDraft(reason="no_observations", sv_dcids=surviving_svs)
    if read.plan is None:
        return NoDataDraft(reason="no_observations", sv_dcids=surviving_svs)
    members, confirmed_svs, plan = read.members, read.confirmed_svs, read.plan
    role_index_by_sv = read.role_index_by_sv
    form = CoverageForm(
        facet_label=facet_label,
        variables=True,
        open_axis=(GENERIC_ROLE_WORDING.noun(open_flow_axis).plural, discover_coverage_role),
    )
    coverage = answer_coverage(plan, form)
    # The latest-year headline is another local coverage lookup over the served rows.
    headline = answer_headline(plan, coverage_index, date_request=date_request)

    discovered = DiscoveredScope(
        axis=_entity_role_key(role_name),
        role_name=role_name,
        member_dcids=members,
        filters=filters,
    )
    return Materialised(
        sv_dcids=confirmed_svs,
        has_data=True,
        coverage=coverage,
        scope_member_dcids=scope_members(plan, form),
        discovered_scope=discovered,
        role_index_by_sv=role_index_by_sv,
        read_plan=plan,
        coverage_form=form,
        role_declaration_unverified=role_declaration_unverified,
        member_facts=member_facts,
        headline=headline,
        coverage_stamp=coverage_snapshot.stamp,
        # Confirmation does not elect a narrower set, so sv_dcids and confirmed have the
        # same size.
        population=MemberPopulation(
            admitted=len(admitted_dcids),
            admitted_dcids=tuple(admitted_dcids),
            confirmed=len(confirmed_svs),
            exact=member_population_exact,
        ),
    )
