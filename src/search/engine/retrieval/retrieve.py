"""Materialise stage: thin resolve dispatcher.

``materialise`` forks on ``search.engine.shape.is_role_bearing`` alone, to
``retrieval.role_resolve.graph_confirm_resolve`` for a role-bearing shape and to
``retrieval.standard.standard_resolve`` for every other shape.

Result types:
  Materialised           - confirmed SV dcids, their facets, the read plan and coverage.
  NoDataDraft            - named reason why no data can be returned.
  BoundValuesUnrealisable - the bound values match no member together.
  MaterialisedCandidates - multiple Materialised results, one per surviving shape.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING

from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.errors import RoleEntityUnplaced
from search.engine.retrieval.read_plan import preview_years
from search.engine.retrieval.role_resolve import graph_confirm_resolve
from search.engine.retrieval.standard import FacetBatch, standard_resolve
from search.engine.shape import ShapeDraft, is_role_bearing, role_names_by_axis
from search.models import Coverage, Headline, NoDataReason

if TYPE_CHECKING:
    from materialise.place_scope import DefaultPlaceScope
    from materialise.roles import RoleAxis
    from search.engine.clients.sdmx import RoleFacetCandidate
    from search.engine.extract import DateRequest
    from search.engine.place import ConsultedPresence
    from search.engine.retrieval.coverage import CoverageForm
    from search.engine.retrieval.member_facts import MemberFacts
    from search.engine.retrieval.read_plan import PlanYears, ReadPlan, WindowGap
    from search.engine.retrieval.role_scope import DiscoveredScope
    from search.models import IdentityFacetBinding
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemberPopulation:
    """Population measured behind a resolve's ``sv_dcids``.

    ``admitted`` counts members allowed by bindings before observation probes.
    The standard path uses filtered ``member_facts``; role paths use the members
    that pass the bound-value checks, before the role-class election. No-probe standard paths count
    only members that explicitly assert every bound property. ``admitted_dcids``
    lists the counted members in source order; digest callers sort them.

    ``confirmed`` counts admitted members with positive observations. It equals
    ``admitted`` when no probe runs and is zero when a probe finds no data. Role
    paths return all confirmed members; standard resolution may elect fewer.

    ``exact`` is true for index-sourced or complete recall populations. A capped
    recall sample gives a lower bound, and member-digest creation requires an
    exact population.

    ``confirmed_exact`` is false when the probe stopped once the answer's member was
    confirmed (`standard._probe_and_scan`), so ``confirmed`` counts only the members read.
    """

    admitted: int
    admitted_dcids: tuple[str, ...]
    confirmed: int
    exact: bool
    confirmed_exact: bool = True

    @property
    def member_count_exact(self) -> bool:
        """Whether ``confirmed`` is a true count, the Spec's
        ``stat_vars.member_count_exact``: the population is exact and every member was
        probed."""
        return self.exact and self.confirmed_exact


@dataclass(frozen=True)
class FamilyTotalAbsent:
    """The single member a result names is not its family's total, or the result shows a
    family with no total by its one breakdown.

    ``shown`` is the member the result names. ``total`` is the family's total when
    the family holds one, which then had no point at ``places`` in the years the read
    plan reads, and ``None`` when the family holds none. ``places`` are the probed
    place dcids, empty when no probe ran. ``in_years_asked`` is true when the plan
    read a year range, so the total can hold data at the places in other years.
    ``breakdown`` names the slots a result on a family with no total shows every member
    by, its one breakdown signature (`standard.Narrow.lone_signature`); ``shown`` is then
    its first member. Empty when the result names one member.
    """

    total: str | None
    shown: str
    places: tuple[str, ...]
    in_years_asked: bool = False
    breakdown: tuple[str, ...] = ()


@dataclass
class Materialised:
    """Confirmed SV dcids, their facets and the read plan for a resolved query."""

    sv_dcids: list[str]
    has_data: bool
    coverage: Coverage
    # Per-SV confirmed facets, served or not; None when the resolver does not populate
    # them. They feed unit elections and facet refusals. No answer states anything from
    # them: an answer's sources and dates are those of the facets its plan serves
    # (`region.facets.cited_facets`, `read_plan.served_dates`).
    facets_by_sv: dict[str, list[Facet]] | None = None
    # The answer's preview read plan, built once at resolve time; coverage, the headline,
    # the preview dispatch and the cited sources all read it (`read_plan`). None without
    # facet reads.
    read_plan: ReadPlan | None = None
    # How `coverage` states the plan, so a filtered answer rebuilds it the same way
    # (`coverage.CoverageForm`). None without a plan.
    coverage_form: CoverageForm | None = None
    # Role-bearing SVs confirmed at a bound destination; None if unassessed.
    destination_confirmed: set[str] | None = None
    # Place probe result; None means unassessed, False means no member had data.
    available_at_place: bool | None = None
    # Standard path: the single member in ``sv_dcids`` is not its family's
    # total. The region pipeline surfaces this as a warning.
    family_total_absent: FamilyTotalAbsent | None = None
    # Data held at the place with no point in the years read (`read_plan.window_gap`): the
    # region answers `no_observations` for the window, departing as `year_unavailable`.
    window_gap: WindowGap | None = None
    # Role path: live role refresh failed or exceeded its member cap; the
    # region pipeline surfaces the retained index roles as unverified.
    role_declaration_unverified: bool = False
    # Role path: the members are recalled ones the mint's member walk did not settle
    # (`bound_values.MembershipAuthority`). Set on an alternate's result; the region
    # pipeline surfaces it as a warning.
    membership_unverified: bool = False
    # Standard plural probes: each entity's confirmed facets, before facet
    # filters. A defaulted plural scope reads its preview in bulk when they are set.
    facets_by_entity: dict[str, list[Facet]] | None = None
    # The members of a defaulted scope the read plan serves (`coverage.scope_members`):
    # the places of a defaulted place scope (["Earth"] for a worldwide one), or the open
    # axis's members of a discovered role scope. A filtered answer lists the filtered
    # plan's. Empty on a carrier with no plan; None when no scope was defaulted.
    scope_member_dcids: list[str] | None = None
    # The default place scope a standard resolve derived from a presence read: that of the
    # StatVars in `sv_dcids`, the scope a resubmit of them derives. None when no presence
    # read was given, or when no member asserts every bound property.
    default_scope: DefaultPlaceScope | None = None
    # Scope discovered for an unbound role axis; None when no discovery ran.
    discovered_scope: DiscoveredScope | None = None
    # Discovered-scope role candidates by SV; empty lists mean no coverage row.
    role_index_by_sv: dict[str, list[RoleFacetCandidate]] | None = None
    # Companion role axis also defaulted during discovery, without a second read.
    also_defaulted_axis: RoleAxis | None = None
    # Open origin members from the destination confirmation's coverage index.
    open_origin_dcids: frozenset[str] | None = None
    # Role path: the flow axes the request left open that the elected members declare,
    # which the minted identity states as its open roles. Empty when none was asked.
    open_axes: frozenset[RoleAxis] = frozenset()
    # Source facts for this resolve's members, before observation confirmation.
    # None on Materialised values constructed outside a resolver.
    member_facts: dict[str, MemberFacts] | None = None
    # Population counts behind this result; standard resolution may elect
    # fewer SVs than the confirmed count. None on caller-built results.
    population: MemberPopulation | None = None
    # Latest-year summary from the role coverage index; None without role data.
    headline: Headline | None = None
    # Coverage view stamp used for this result's population and headline.
    coverage_stamp: str | None = None

    @property
    def preview_years(self) -> PlanYears:
        """`read_plan.preview_years` of this answer's plan."""
        return preview_years(self.read_plan)

    @property
    def served_members(self) -> Sequence[str]:
        """The members the answer's identity serves: the admitted population, or
        `sv_dcids` on a result built without one."""
        return self.population.admitted_dcids if self.population is not None else self.sv_dcids


@dataclass
class UnsupportedDraft:
    """Refusal for a valid request whose discovered scope exceeds a bound."""

    code: str  # "scope_too_broad"
    pointer: str
    detail: str
    # Facet bindings tied to the refusal, so its pointer resolves in the response.
    facets: dict[str, IdentityFacetBinding] | None = None


@dataclass
class NoDataDraft:
    """Named no-data outcome from the materialise stage."""

    reason: NoDataReason
    # Confirmed SVs when identity was checked before the no-data outcome.
    sv_dcids: list[str] | None = None
    # Role path: the flow axes the request left open that the elected members declare, as
    # on `Materialised.open_axes`, set when confirmation finds no data
    # (`role_resolve.graph_confirm_resolve`). A refusal that flags those members states them
    # as its open roles. Empty everywhere else.
    open_axes: frozenset[RoleAxis] = frozenset()


@dataclass(frozen=True)
class MemberConflict:
    """The bound properties that keep one member out of the answer, and whether the member
    asserts every bound property, as a probe requires
    (`bound_values._fully_bound_members_facts`).

    The standard path names the properties the member's asserted values contradict
    (`bound_values._bound_value_conflicts`). The role path names those and the bound
    properties the member does not assert (`role_resolve._mint_conflicts`), since the mint
    serves no member silent on a bound property."""

    properties: frozenset[str]
    fully_bound: bool


@dataclass
class BoundValuesUnrealisable(NoDataDraft):
    """The bound values match no member together (`binding_mismatch`), and unbinding some
    of them admits one.

    `conflicts` holds each member's conflict, in population order and without repeats.
    Each set is non-empty: every member it describes is kept out by some bound value. A
    caller that can unbind values unbinds a minimal set (`relaxation`) and resolves again
    (`region.unrealisable`); any other caller reads the outcome as the `binding_mismatch`
    refusal it is.
    """

    reason: str = "binding_mismatch"
    conflicts: tuple[MemberConflict, ...] = ()

    def relaxation(
        self,
        *,
        engine_chosen: AbstractSet[str] = frozenset(),
        unnamed: AbstractSet[str] = frozenset(),
    ) -> frozenset[str]:
        """The bound properties to unbind next.

        Unbinding a member's conflict set lifts every conflict that member carries, so the
        candidates are the members' sets, ranked by the fewest properties the request named
        (those in neither class below), then the fewest `unnamed` ones, then the fewest
        `engine_chosen` ones (a declared default or the elected measure partition), then a
        member that asserts every bound property, then the earliest member in population
        order. The counts make the pick inclusion-minimal: a strict subset of a set is
        smaller in one count and no larger in the others.

        The classes give way in that order. A set of engine-chosen values alone is unbound
        first. Otherwise every `unnamed` value in any member's set is unbound at once: such
        a value binds because its slot offered no other, a claim that every member asserts
        it, and a member conflicting on it refutes the claim. Otherwise the ranked set goes,
        which then holds a value the request named. A caller with no such split passes
        nothing, so the smallest set wins.

        Unbinding is not guaranteed to admit a member. The next resolve can draw a
        different population and refuse again, which is why callers loop.

        Raises:
            ValueError: there is no conflict, or the chosen set is empty, so unbinding it
                would resolve the same bindings again.
        """
        if not self.conflicts:
            raise ValueError("binding_mismatch names no member conflict")
        unnamed = unnamed - engine_chosen
        _, chosen = min(
            enumerate(self.conflicts),
            key=lambda ranked: (
                len(ranked[1].properties - engine_chosen - unnamed),
                len(ranked[1].properties & unnamed),
                len(ranked[1].properties & engine_chosen),
                not ranked[1].fully_bound,
                ranked[0],
            ),
        )
        if not chosen.properties:
            raise ValueError("binding_mismatch names a member no bound value contradicts")
        if chosen.properties <= engine_chosen:
            return chosen.properties
        refuted = frozenset().union(*(conflict.properties & unnamed for conflict in self.conflicts))
        return refuted or chosen.properties


@dataclass
class MaterialisedCandidates:
    """One materialised result per surviving shape."""

    candidates: list[Materialised]


def materialise(
    shape: ShapeDraft,
    bindings: list[SlotBindingDraft],
    entities: Mapping[str, list[str]],
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    date_request: DateRequest | None = None,
    deadline: float | None = None,
    place_dcids: list[str] | None = None,
    default_presence: ConsultedPresence | None = None,
    discover_scope: bool = False,
    open_axes: frozenset[RoleAxis] = frozenset(),
    coverage: CoverageSnapshot,
    facet_batch: FacetBatch | None = None,
) -> Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft:
    """Resolve a shape through the standard or role-bearing path.

    Role-bearing shapes use ``graph_confirm_resolve`` with entities keyed by
    their declared role names; others use ``standard_resolve``. ``entities`` is the
    shape's own binding of the query's places (`region.place_roles.bind_place_roles`).
    An unplaced role entity becomes ``role_incompatible``. Roles absent from this
    deployment's flow-axis vocabulary yield ``role_not_resolved``. An open
    destination, or an unbound origin on an origin-only shape, is discovered
    only when ``discover_scope`` is true. A bound origin with no destination
    remains ``role_not_resolved``.
    ``open_axes`` are the role path's open flow axes (`graph_confirm_resolve`).

    Returns a materialised result, surviving candidates, an unsupported-scope
    refusal, or a named no-data outcome. ``coverage`` is the request's snapshot
    and is read only by the role path. ``sdmx`` and ``discover_scope`` are also
    role-path inputs. ``place_dcids``, ``default_presence`` and ``facet_batch`` are used
    only by the standard path, where ``default_presence`` derives a default place scope in
    place of ``place_dcids`` and ``facet_batch`` shares the facet reads of a Spec whose
    place sets are settling (`standard_resolve`).
    """
    if is_role_bearing(shape):
        names = role_names_by_axis(shape)
        subject_dcids = list(entities.get(names.get("destination", ""), ()))
        origin_dcids = list(entities.get(names.get("origin", ""), ()))
        if "destination" not in names and not (discover_scope and "origin" in names):
            logger.warning(
                "materialise: shape %r declares roles %r, none of which this "
                "deployment's axis vocabulary places on a flow axis materialise "
                "can resolve against",
                shape.shape_id,
                shape.declared_roles,
            )
            return NoDataDraft(reason="role_not_resolved")
        try:
            return graph_confirm_resolve(
                shape=shape,
                bindings=bindings,
                subject_dcids=subject_dcids,
                origin_dcids=origin_dcids,
                graph=graph,
                sdmx=sdmx,
                date_request=date_request,
                deadline=deadline,
                discover_scope=discover_scope,
                open_axes=open_axes,
                coverage_snapshot=coverage,
            )
        except RoleEntityUnplaced:
            return NoDataDraft(reason="role_incompatible")
    return standard_resolve(
        shape=shape,
        bindings=bindings,
        entities=entities,
        graph=graph,
        date_request=date_request,
        place_dcids=place_dcids,
        default_presence=default_presence,
        facet_batch=facet_batch,
    )
