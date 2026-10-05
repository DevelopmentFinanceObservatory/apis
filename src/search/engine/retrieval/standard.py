"""Resolve standard shapes by probing observations through the injected graph client."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

from code_index import absent_readings
from code_index.shared import get_index
from materialise.place_scope import DefaultPlaceScope, DefaultPresence
from search.engine.clients.materialise_bridge import _FACET_PROBE_MAX_PAIRS_PER_POST
from search.engine.place import is_worldwide_scope
from search.engine.retrieval.bound_values import (
    _bound_value_conflicts,
    _bound_value_filter_facts,
    _collapse_bindings,
    _fully_bound_members_facts,
    bound_values_from_bindings,
    shape_member_facts,
    shape_query,
)
from search.engine.retrieval.elected_dataset import (
    NO_DATASETS,
    ElectedDatasets,
    check_elected_datasets,
    elected_datasets,
    published,
)
from search.engine.retrieval.facet_policy import refuse_declared_defaults
from search.engine.retrieval.facts import read_constraints
from search.engine.retrieval.member_facts import asserted_constraint_properties
from search.engine.retrieval.read_plan import (
    PairPlan,
    PlanYears,
    asked_years,
    has_points,
    in_elected_datasets,
    plan_standard,
    plan_years,
    serves_nothing,
    window_gap,
)
from search.models import SlotOption, SlotOptions, TimeWindow
from statvar_members import antichain_minimal_ids

if TYPE_CHECKING:
    from materialise import GraphReader
    from search.engine.bind import SlotBindingDraft
    from search.engine.clients.graph import EngineGraphClient, Facet
    from search.engine.extract import DateRequest
    from search.engine.retrieval.member_facts import MemberFacts
    from search.engine.retrieval.retrieve import (
        FamilyTotalAbsent,
        Materialised,
        MaterialisedCandidates,
        MemberPopulation,
        NoDataDraft,
    )
    from search.engine.shape import ShapeDraft

# SV, its facets, and positive probe entities (None for single-entity probes).
type _ConfirmedMember = tuple[str, list[Facet], set[str] | None]


def _bind_aware_order(
    svs: list[str],
    member_facts: Mapping[str, MemberFacts],
    bound_values: dict[str, list[str]],
) -> list[str]:
    """Rank explicit bound-property matches first, preserving order within ties.

    Shared by probe ordering and representative selection. Reads constraints
    through ``MemberFacts`` so recall- and index-sourced members rank alike.
    """
    if not bound_values:
        return list(svs)
    return sorted(
        svs,
        key=lambda sv: (
            -sum(1 for prop in bound_values if prop in member_facts[sv].read_constraints())
        ),
    )


@dataclass(frozen=True)
class _Family:
    """One eligible family and the election among its members.

    ``members`` are the eligible members in recall order. ``order`` holds the ones
    the datasets mint keeps, in election order. ``totals`` are the members that
    stand for the whole family.
    """

    members: tuple[str, ...]
    order: tuple[str, ...]
    totals: frozenset[str]

    def total_absent(
        self, shown: str, places: Sequence[str], *, years: PlanYears = None
    ) -> FamilyTotalAbsent | None:
        """Return the fact a result carries when ``shown`` is not the family's total.
        ``years`` are the years the read plan reads, under which the total held no point."""
        from search.engine.retrieval.retrieve import FamilyTotalAbsent

        if shown in self.totals:
            return None
        first = self.order[0]
        return FamilyTotalAbsent(
            total=first if first in self.totals else None,
            shown=shown,
            places=tuple(places),
            in_years_asked=isinstance(years, TimeWindow),
        )


class _Assertions(NamedTuple):
    """What one member asserts, read against the query's bindings."""

    # Every asserted constraint property; None when the member's constraints are unknown.
    properties: frozenset[str] | None
    # The resolved (property, value) pairs on unbound properties.
    open_items: tuple[tuple[str, str], ...]
    # How many unbound properties the member asserts, resolved or not.
    open_count: int
    # The constraints are known, every unbound one is resolved, and each bound
    # property holds the one value its binding states.
    can_be_total: bool


def _read_assertions(facts: MemberFacts, stated_values: Mapping[str, list[str]]) -> _Assertions:
    properties = asserted_constraint_properties(facts)
    if properties is None:
        return _Assertions(properties=None, open_items=(), open_count=0, can_be_total=False)
    constraints = facts.read_constraints()
    open_props = [prop for prop in properties if prop not in stated_values]
    open_items = tuple((prop, constraints[prop]) for prop in open_props if prop in constraints)
    return _Assertions(
        properties=properties,
        open_items=open_items,
        open_count=len(open_props),
        can_be_total=len(open_items) == len(open_props)
        and all(
            len(values) == 1 and constraints.get(prop) == values[0]
            for prop, values in stated_values.items()
        ),
    )


def _elect_family(
    svs: Sequence[str],
    member_facts: Mapping[str, MemberFacts],
    stated_values: Mapping[str, list[str]],
) -> _Family:
    """Reduce `svs` to the members the datasets mint keeps and order them for election.

    `stated_values` maps each bound property to the values its binding states, before
    any widening to descendants. The mint keeps a family's antichain-minimal members:
    one member dominates another when its constraint properties are a proper subset
    and each missing property reads `aggregate` in the code index. A member the mint
    drops cannot be downloaded, so Search elects only among the kept ones. A member
    with an unresolved constraint keeps its property keys and can dominate, as in the
    mint. A member whose constraints are unknown has no keys and dominates nothing.

    A member is the family's total when it narrows the family on nothing:

    - every member of `svs` asserts each (property, value) pair it asserts on an
      unbound property. `undata/unicef/CME_MR.AGE--Y0T4` is the total of a family
      whose other member is `undata/unicef/CME_MR.AGE--Y0T4__SEX--M`, and of no
      family that also holds `undata/unicef/CME_MR.AGE--M0`;
    - on each bound property it asserts the value the binding states. A member
      asserting a descendant of a hierarchical value, or one value of a binding that
      states several, is narrower than the binding;
    - its constraints are known and its unbound ones are resolved.

    Two narrowed answers pass as totals. A family with a single member has that member
    as its total unless its constraints are not known, an unbound one is not resolved, or
    it narrows a bound property, so a recall sample holding one member narrow only on
    unbound properties raises nothing. An
    alternate or re-elected shape binds every constraint of its representative, so the
    representative is a total and the narrowing shows in the Spec's bound slots.

    Kept members sort totals first, then by the recall position of the first member
    whose unbound assertions include their own, then by how many unbound properties
    they assert.
    """
    read = {sv: _read_assertions(member_facts[sv], stated_values) for sv in svs}
    # The positions of the members asserting each unbound (property, value) pair.
    positions: dict[tuple[str, str], set[int]] = {}
    for i, sv in enumerate(svs):
        for item in read[sv].open_items:
            positions.setdefault(item, set()).add(i)

    totals = frozenset(
        sv
        for sv in svs
        if read[sv].can_be_total
        and all(len(positions[item]) == len(svs) for item in read[sv].open_items)
    )
    index_of = {sv: i for i, sv in enumerate(svs)}

    def rank(sv: str) -> tuple[bool, int, int]:
        member = read[sv]
        if member.properties is None or len(member.open_items) < member.open_count:
            first = index_of[sv]
        elif member.open_items:
            first = min(set.intersection(*(positions[item] for item in member.open_items)))
        else:
            first = 0
        return sv not in totals, first, member.open_count

    keys = {sv: member.properties for sv, member in read.items() if member.properties is not None}
    kept = antichain_minimal_ids(keys, svs, readings=absent_readings(get_index()))
    return _Family(members=tuple(svs), order=tuple(sorted(kept, key=rank)), totals=totals)


# engine.bind_repair imports this helper by name.
def _is_degenerate_binding(b: SlotBindingDraft) -> bool:
    """Return true for value/set bindings with no values.

    ``property_dcid`` can be absent on a valid constraint, so only kind and
    ``value_dcids`` determine degeneracy.
    """
    return b.kind in ("value", "set") and not b.value_dcids


def _standard_probe_key(
    members: Sequence[str],
    place_dcid: str | None,
    date_request: DateRequest | None,
    place_dcids: list[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], bool] | None:
    """Return the standard probe cache key, or ``None`` when no probe runs.

    The key contains every confirmed member, the place set (or single place),
    and whether dates are needed. No members or no place yields ``None``.
    Candidate pre-warming uses a separate speculative population because it
    runs before alternative shapes have bindings.
    """
    if not members:
        return None
    if place_dcids:
        probe_entities: tuple[str, ...] = tuple(place_dcids)
    elif place_dcid is not None:
        probe_entities = (place_dcid,)
    else:
        return None
    return (tuple(members), probe_entities, date_request is not None)


@dataclass(frozen=True)
class _AdmittedStandardMembers:
    """The bound-value-filtered member population a standard-path resolve
    probes, ready for the default-scope short-circuit and the ordinary
    member-probe alike.
    """

    bound_values: dict[str, list[str]]
    member_facts: dict[str, MemberFacts]
    exact: bool
    # The members asserting every bound property, and the election among them.
    family: _Family
    representative_sv: str


def _admit_standard_members(
    shape: ShapeDraft, bindings: list[SlotBindingDraft]
) -> NoDataDraft | _AdmittedStandardMembers:
    """Source members and apply the bound-value filter.

    An empty shape, absent source population, or degenerate value/set binding
    returns ``NoDataDraft(reason="variable_not_resolved")``. Missing constraint
    values pass, and mismatched asserted values are filtered. Bound values that
    together exclude every member return ``BoundValuesUnrealisable``, naming each
    member's conflicting bound properties, so the caller can unbind a minimal set
    and resolve again. ``exact`` marks a
    complete index/recall population; capped recall is a floor. Members keep
    recall order, then index order for members recall did not return. The
    representative, which a result names when no member confirms, is the first
    member in election order among those asserting every bound property
    (``_elect_family``), or the bind-aware first member when none does.
    """
    # Local imports avoid the retrieve/role_resolve cycle.
    from search.engine.retrieval.retrieve import (
        BoundValuesUnrealisable,
        MemberConflict,
        NoDataDraft,
    )

    if not shape.sv_arc_facts:
        return NoDataDraft(reason="variable_not_resolved")
    if any(_is_degenerate_binding(b) for b in bindings):
        return NoDataDraft(reason="variable_not_resolved")

    bound_values = bound_values_from_bindings(bindings)

    # Use the full index population when available, otherwise the shape's arcs.
    # Downstream readers share this filtered population; ShapeDraft.sv_arc_facts
    # remains the recall-sourced pool when index members back the result.
    member_facts, index_sourced = shape_member_facts(shape, shape_query(shape, bindings))
    if not member_facts:
        return NoDataDraft(reason="variable_not_resolved")

    # Recall order first, so the family order starts from recall's reading of the
    # question; index members recall did not return follow in index order. A refusal's
    # conflicts keep the same order.
    recalled = [sv for sv in shape.sv_arc_facts if sv in member_facts]
    member_facts = {sv: member_facts[sv] for sv in [*recalled, *member_facts]}
    filtered = _bound_value_filter_facts(member_facts, bound_values)
    if not filtered:
        asserting = set(_fully_bound_members_facts(list(member_facts), member_facts, bound_values))
        return BoundValuesUnrealisable(
            conflicts=tuple(
                dict.fromkeys(
                    MemberConflict(
                        properties=_bound_value_conflicts(member, bound_values),
                        fully_bound=sv in asserting,
                    )
                    for sv, member in member_facts.items()
                )
            )
        )
    member_facts = filtered
    exact = index_sourced or shape.sv_arc_facts_complete

    fully_bound = _fully_bound_members_facts(list(member_facts), member_facts, bound_values)
    family = _elect_family(fully_bound, member_facts, _collapse_bindings(bindings))
    representative_sv = (
        family.order[0]
        if family.order
        else _bind_aware_order(list(member_facts), member_facts, bound_values)[0]
    )

    return _AdmittedStandardMembers(
        bound_values=bound_values,
        member_facts=member_facts,
        exact=exact,
        family=family,
        representative_sv=representative_sv,
    )


@dataclass(frozen=True)
class _DefaultScopeCheck:
    """Whether this call is a default-scope probe: no place
    named at all, with place_dcids threaded in as the resolved scope and
    place_dcid still None. Only that hook produces this combination. A bound
    query's place_dcids always comes with a real place_dcid. When active,
    terminal carries either that probe's own short-circuit result or None to
    continue to the ordinary member-probe below.
    """

    active: bool
    terminal: Materialised | None


def _default_scope_short_circuit(
    *,
    place_dcid: str | None,
    place_dcids: list[str] | None,
    member_facts: dict[str, MemberFacts],
    family: _Family,
    representative_sv: str,
    date_request: DateRequest | None,
    exact: bool,
) -> _DefaultScopeCheck:
    """Handle default-scope queries with no place dcid and a supplied place list.

    An empty fully-bound population returns unavailable coverage with
    ``available_at_place=None``. An empty ``place_dcids`` list means the
    presence read found no data, so it returns ``available_at_place=False``
    with the fully-bound population admitted and zero confirmed, and
    ``family_total_absent`` when the representative is not the family's total.
    ``["Earth"]`` proceeds to the probe.
    """
    from search.engine.retrieval.retrieve import Materialised, MemberPopulation
    from search.models import CoverageBare

    active = place_dcid is None and place_dcids is not None
    if not active:
        return _DefaultScopeCheck(active=False, terminal=None)

    if not family.members:
        return _DefaultScopeCheck(
            active=True,
            terminal=Materialised(
                sv_dcids=[representative_sv],
                has_data=False,
                coverage=CoverageBare(
                    has_data=False, window=date_request.window if date_request else None
                ),
                scope_member_dcids=[],
                member_facts=member_facts,
                population=MemberPopulation(
                    admitted=0, admitted_dcids=(), confirmed=0, exact=exact
                ),
            ),
        )
    if not place_dcids:
        return _DefaultScopeCheck(
            active=True,
            terminal=Materialised(
                sv_dcids=[representative_sv],
                has_data=False,
                coverage=CoverageBare(
                    has_data=False, window=date_request.window if date_request else None
                ),
                available_at_place=False,
                family_total_absent=family.total_absent(representative_sv, ()),
                scope_member_dcids=[],
                member_facts=member_facts,
                population=MemberPopulation(
                    admitted=len(family.members),
                    admitted_dcids=family.members,
                    confirmed=0,
                    exact=exact,
                ),
            ),
        )
    return _DefaultScopeCheck(active=True, terminal=None)


def _no_entity_disambiguation(
    *,
    member_facts: dict[str, MemberFacts],
    family: _Family,
    representative_sv: str,
    date_request: DateRequest | None,
    exact: bool,
) -> Materialised:
    """Return bare coverage when no place entity is available.

    No observation probe runs. A member must explicitly assert every bound
    property; the first of those members in election order answers
    (``_elect_family``), with ``family_total_absent`` when it is not the family's
    total. Otherwise the result has ``has_data=False`` and ``available_at_place=None``.
    Explicit windows are echoed, while latest requests leave the window unset.
    """
    from search.engine.retrieval.retrieve import Materialised, MemberPopulation
    from search.models import CoverageBare

    no_entity_window = date_request.window if date_request is not None else None
    if not family.members:
        return Materialised(
            sv_dcids=[representative_sv],
            has_data=False,
            coverage=CoverageBare(has_data=False, window=no_entity_window),
            member_facts=member_facts,
            population=MemberPopulation(admitted=0, admitted_dcids=(), confirmed=0, exact=exact),
        )
    return Materialised(
        sv_dcids=[representative_sv],
        has_data=True,
        coverage=CoverageBare(has_data=True, window=no_entity_window),
        family_total_absent=family.total_absent(representative_sv, ()),
        member_facts=member_facts,
        population=MemberPopulation(
            admitted=len(family.members),
            admitted_dcids=family.members,
            confirmed=len(family.members),
            exact=exact,
        ),
    )


def _select_probe_members(
    members: Sequence[str],
    bindings: list[SlotBindingDraft],
    member_facts: dict[str, MemberFacts],
    bound_values: dict[str, list[str]],
) -> tuple[list[str], bool]:
    """Choose probe members and whether a set binding collects several results.

    For ``kind="set"``, collect every member that explicitly matches at least
    one set value and contradicts none of its asserted set axes. Silent axes
    pass, but a sibling with no explicit set value is excluded. Without a set
    binding, probe bind-aware order. Finally, require every returned member to
    explicitly assert every bound property; separate siblings cannot jointly
    satisfy different bound axes.
    """
    set_bound_values: dict[str, list[str]] = {
        b.property_dcid: list(b.value_dcids)
        for b in bindings
        if b.property_dcid and b.kind == "set" and b.value_dcids
    }
    set_members: list[str] = []
    if set_bound_values:
        for sv in members:
            constraints = member_facts[sv].read_constraints()
            consistent = all(
                constraints.get(prop) in vals
                for prop, vals in set_bound_values.items()
                if prop in constraints
            )
            explicit_match = any(
                constraints.get(prop) in vals for prop, vals in set_bound_values.items()
            )
            if consistent and explicit_match:
                set_members.append(sv)
    multi_collect = bool(set_members)
    if multi_collect:
        probe_members = set_members
    elif bound_values:
        probe_members = _bind_aware_order(list(members), member_facts, bound_values)
    else:
        probe_members = list(members)

    probe_members = _fully_bound_members_facts(probe_members, member_facts, bound_values)
    return probe_members, multi_collect


def _member_places(
    members: Sequence[str],
    probe_entities: Sequence[str],
    *,
    own_scope_members: Sequence[str],
    presence: DefaultPresence | None,
) -> dict[str, tuple[str, ...]]:
    """The places each of `members` is read at: each of `own_scope_members` at its own default
    place scope (`DefaultPresence.scope`) when `presence` is given, and every other member at
    `probe_entities`."""
    places = dict.fromkeys(members, tuple(probe_entities))
    if presence is not None:
        places.update((sv, presence.scope(sv).places) for sv in own_scope_members)
    return places


def _facet_reads(svs: Sequence[str], places: Sequence[str]) -> list[list[str]]:
    """The StatVar chunks one facet read of `svs` at `places` issues, one POST each: at most
    `_FACET_PROBE_MAX_PAIRS_PER_POST` pairs, or one StatVar when its places alone exceed
    that."""
    step = max(1, _FACET_PROBE_MAX_PAIRS_PER_POST // len(places))
    return [list(svs[i : i + step]) for i in range(0, len(svs), step)]


def facets_at_own_places(
    graph: EngineGraphClient, places_of: Mapping[str, Sequence[str]], *, needs_dates: bool
) -> dict[str, dict[str, list[Facet]]]:
    """Read each StatVar's facets at its own places (`places_of`), keeping only those.

    Of two read plans, the one with fewer POSTs runs, the per-scope plan on a tie, its reads
    one after another. The per-scope plan reads the StatVars sharing a place set
    together, in first-StatVar order; the union plan reads every StatVar at the union of
    their places, in first-place order. Each read is chunked by `_facet_reads`. Either way a
    StatVar keeps its facets at its own places alone, so both plans return the same result.
    A StatVar with no places maps to `{}` and is not read.
    """
    by_places: dict[tuple[str, ...], list[str]] = {}
    for sv, places in places_of.items():
        if places:
            by_places.setdefault(tuple(places), []).append(sv)
    per_scope = [
        (chunk, places) for places, svs in by_places.items() for chunk in _facet_reads(svs, places)
    ]
    union = list(dict.fromkeys(place for places in by_places for place in places))
    read_svs = [sv for svs in by_places.values() for sv in svs]
    union_reads = (
        [(chunk, tuple(union)) for chunk in _facet_reads(read_svs, union)] if union else []
    )
    reads = union_reads if len(union_reads) < len(per_scope) else per_scope

    facets: dict[str, dict[str, list[Facet]]] = {sv: {} for sv in places_of}
    for chunk, places in reads:
        batch = graph.observation_facets_batch(chunk, list(places), needs_dates=needs_dates)
        for sv in chunk:
            own = set(places_of[sv])
            facets[sv] = {e: f for e, f in batch.get(sv, {}).items() if e in own}
    return facets


def _probe_and_scan(
    *,
    places_of: Mapping[str, Sequence[str]],
    probe_members: Sequence[str],
    needs_dates: bool,
    graph: EngineGraphClient,
    years: PlanYears,
    datasets: ElectedDatasets,
    minted: Collection[str],
) -> tuple[dict[str, dict[str, list[Facet]]], list[_ConfirmedMember], list[_ConfirmedMember]]:
    """Probe every member at its own places (`places_of`, `facets_at_own_places`), then
    scan eligible members for points in `years`, each at its own places.

    Scanning is exhaustive because the reads hold every member's facets. A member's facets
    union across its places, and it is confirmed at a place holding a point in `years` by the
    point rule every plan and election reads (`read_plan.has_points`); only its own places
    count, and election happens later.

    A member of an index-resolved family keeps only the facets its elected dataset
    publishes (`datasets`), the ones a mint serves it from. Each member a mint of the answer
    serves (`minted`) is first checked as the mint checks it, over every facet it holds at
    its places (`check_elected_datasets`).

    Returns the batch facets, the confirmed members, and the members holding data at their
    places in other years only, which make an empty answer one for the window rather than
    the place.
    """
    # Each member is read at its own places, the entities its mint reads it at. A member the
    # mint serves is checked over every facet there, and each keeps the facets its elected
    # dataset publishes.
    own = facets_at_own_places(graph, places_of, needs_dates=needs_dates)
    check_elected_datasets(
        datasets,
        {
            sv: [facet for facets in by_entity.values() for facet in facets]
            for sv, by_entity in own.items()
            if sv in minted
        },
    )
    facets_by_sv_batch = {
        sv: {entity: published(datasets, sv, facets) for entity, facets in by_entity.items()}
        for sv, by_entity in own.items()
    }
    confirmed: list[_ConfirmedMember] = []
    outside_years: list[_ConfirmedMember] = []
    for sv in probe_members:
        facets_by_entity = facets_by_sv_batch.get(sv, {})
        facets = [f for entity in places_of[sv] for f in facets_by_entity.get(entity, [])]
        positive_entities = {
            entity
            for entity in places_of[sv]
            if has_points(facets_by_entity.get(entity, []), years)
        }
        if positive_entities:
            plural = len(places_of[sv]) > 1
            confirmed.append((sv, facets, positive_entities if plural else None))
        elif any(f.obs_count > 0 for f in facets):
            outside_years.append((sv, facets, None))
    return facets_by_sv_batch, confirmed, outside_years


def _coverage_label(
    answer_places: Sequence[str],
    *,
    place_dcids: list[str] | None,
    default_scope: DefaultPlaceScope | None,
) -> tuple[str, bool]:
    """The coverage label and plural flag of an answer read at `answer_places`.

    `default_scope` is the default scope of the members the answer names, on a default-scope
    probe. ``worldwide`` applies to such a scope that is Earth alone, and to
    ``place_dcids == ["Earth"]`` on any other probe. Any other plural answer is labelled
    ``places`` when its default scope is every place with data, which spans every level and
    type, else ``countries``; one place's is ``sources``."""
    is_plural = len(answer_places) > 1
    worldwide = is_worldwide_scope(
        list(answer_places) if default_scope is not None else place_dcids
    )
    if worldwide:
        return "worldwide", is_plural
    if not is_plural:
        return "sources", is_plural
    every_place = default_scope is not None and default_scope.every_place
    return ("places" if every_place else "countries"), is_plural


def _no_data_at_place(
    *,
    representative_sv: str,
    date_request: DateRequest | None,
    default_scope_probe: bool,
    member_facts: dict[str, MemberFacts],
    population: MemberPopulation,
    family_total_absent: FamilyTotalAbsent | None,
) -> Materialised:
    """The flagged-unavailable carrier for a probed population with no
    member confirmed to have data at this place: has_data=False, no
    facets, and available_at_place=False. region/__init__.py turns this
    into a no_data response carrying the representative Spec as
    flagged_match. ``family_total_absent`` is set when the representative is
    not its family's total.
    """
    from search.engine.retrieval.retrieve import Materialised
    from search.models import CoverageBare

    return Materialised(
        sv_dcids=[representative_sv],
        has_data=False,
        coverage=CoverageBare(has_data=False, window=date_request.window if date_request else None),
        available_at_place=False,
        family_total_absent=family_total_absent,
        scope_member_dcids=[] if default_scope_probe else None,
        member_facts=member_facts,
        population=population,
    )


def _standard_plan(
    svs: Sequence[str],
    *,
    facets_by_sv_batch: Mapping[str, Mapping[str, list[Facet]]],
    places_of: Mapping[str, Sequence[str]],
    member_facts: Mapping[str, MemberFacts],
    years: PlanYears,
    scope: bool,
) -> PairPlan:
    """The read plan of an answer serving `svs` under `years`, each at its own places
    (`places_of`). `svs` are members the family election keeps (`_elect_family`, the mint's
    antichain), so the plan reads each of them at each of its places holding facets, a
    member of an index-resolved family from its elected dataset's facets alone
    (`read_plan.in_elected_datasets`). A scope-defaulted place set previews one facet
    across every place (`BulkScope`)."""
    plan = plan_standard(
        {
            (sv, entity): facets
            for sv in svs
            for entity in places_of[sv]
            if (facets := facets_by_sv_batch.get(sv, {}).get(entity, []))
        },
        members=svs,
        years=years,
        scope=scope,
        places=tuple(dict.fromkeys(entity for sv in svs for entity in places_of[sv])),
    )
    narrowed = in_elected_datasets(plan, member_facts)
    assert isinstance(narrowed, PairPlan)
    return narrowed


def _no_data_in_years(
    member: _ConfirmedMember,
    *,
    dateless_plan: PairPlan,
    date_request: DateRequest | None,
    default_scope_probe: bool,
    member_facts: dict[str, MemberFacts],
    population: MemberPopulation,
    family_total_absent: FamilyTotalAbsent | None,
) -> Materialised:
    """The carrier for a population holding data at the place but no point in the years
    read: has_data=False and available_at_place=True, with `member`'s facets and the gap the
    years asked leave in `dateless_plan`, the plan the answer builds with its window removed
    (`read_plan.window_gap`). The region answers `no_observations` for the window, its
    `year_unavailable` departure naming the years asked and the span the series runs. Only
    a range leaves data outside the years read: under a latest or every-date read it is the
    flagged-unavailable carrier (`_no_data_at_place`).
    ``family_total_absent`` is set when `member` is not its family's total."""
    from search.engine.retrieval.retrieve import Materialised
    from search.models import CoverageBare

    sv, facets, _entities = member
    years = asked_years(date_request)
    if years is None:
        return _no_data_at_place(
            representative_sv=sv,
            date_request=date_request,
            default_scope_probe=default_scope_probe,
            member_facts=member_facts,
            population=population,
            family_total_absent=family_total_absent,
        )
    return Materialised(
        sv_dcids=[sv],
        has_data=False,
        coverage=CoverageBare(has_data=False, window=date_request.window if date_request else None),
        facets_by_sv={sv: list(facets)},
        available_at_place=True,
        window_gap=window_gap(dateless_plan, years),
        family_total_absent=family_total_absent,
        scope_member_dcids=[] if default_scope_probe else None,
        member_facts=member_facts,
        population=population,
    )


def _materialise_collected(
    collected: list[_ConfirmedMember],
    *,
    facets_by_sv_batch: dict[str, dict[str, list[Facet]]],
    places_of: Mapping[str, Sequence[str]],
    is_plural: bool,
    coverage_facet_label: str,
    date_request: DateRequest | None,
    default_scope_probe: bool,
    member_facts: dict[str, MemberFacts],
    population: MemberPopulation,
    family_total_absent: FamilyTotalAbsent | None,
) -> Materialised:
    """Build a result from the collected members and their facets.

    A plan that serves no point in its years (`read_plan.serves_nothing`) returns the
    carrier for data outside them (`_no_data_in_years`). A scope-defaulted probe lists the
    places the plan serves as its scope members.

    A single winner supplies the result; a ``kind="set"`` collect emits every
    member and unions their facets, and the read plan reads the collected members. Coverage
    states what the plan serves. Plural probes also retain per-entity facets for each
    emitted member at its own places (`places_of`). This standard subject probe does not
    confirm a directional destination.
    """
    from search.engine.retrieval.coverage import CoverageForm, answer_coverage, scope_members
    from search.engine.retrieval.retrieve import Materialised

    collected_svs = [sv for sv, _, _ in collected]
    plan = _standard_plan(
        collected_svs,
        facets_by_sv_batch=facets_by_sv_batch,
        places_of=places_of,
        member_facts=member_facts,
        years=plan_years(date_request),
        scope=default_scope_probe and is_plural,
    )
    if serves_nothing(plan):
        return _no_data_in_years(
            collected[0],
            dateless_plan=plan.every_date(),
            date_request=date_request,
            default_scope_probe=default_scope_probe,
            member_facts=member_facts,
            population=population,
            family_total_absent=family_total_absent,
        )
    form = CoverageForm(facet_label=coverage_facet_label, counts_places=is_plural, variables=True)
    return Materialised(
        sv_dcids=collected_svs,
        has_data=True,
        coverage=answer_coverage(plan, form),
        coverage_form=form,
        facets_by_sv={sv: facets for sv, facets, _ in collected},
        read_plan=plan,
        facets_by_entity=(
            {
                e: [f for sv in collected_svs for f in facets_by_sv_batch.get(sv, {}).get(e, [])]
                for e in dict.fromkeys(e for sv in collected_svs for e in places_of[sv])
            }
            if is_plural
            else None
        ),
        scope_member_dcids=scope_members(plan, form) if default_scope_probe else None,
        member_facts=member_facts,
        population=population,
        family_total_absent=family_total_absent,
    )


def standard_resolve(
    *,
    shape: ShapeDraft,
    bindings: list[SlotBindingDraft],
    entities: Mapping[str, list[str]],
    graph: EngineGraphClient,
    date_request: DateRequest | None = None,
    place_dcids: list[str] | None = None,
    default_presence: DefaultPresence | None = None,
) -> Materialised | NoDataDraft | MaterialisedCandidates:
    """Resolve a confirmed shape against its subject entities.

    Probe every admitted member at its own places (`facets_at_own_places`). Bindings
    determine eligibility.
    Election reads only the members the datasets mint keeps from the eligible
    family (``_elect_family``): a ``kind="set"`` binding returns each of them with
    a point in the years read, while other bindings elect the first of them, in
    election order, that holds one. With none, a kept member holding data at the place
    in other years answers for the window (``_no_data_in_years``), and with no kept
    member holding data the first in election order is flagged as unavailable. Under
    either binding, a result naming one member, elected or flagged, that is not the
    family's total records it in ``family_total_absent``. A place-set probe unions facets
    across its entities. An empty default scope can return before a probe because its
    presence read already established the result.

    ``default_presence`` defaults the place scope of a shape with no subject entity, in
    place of ``place_dcids``. Each member the result can name (the kept family) is read at
    its own default scope (`DefaultPresence.scope`), which is where a datasets mint fetches
    it, and every other member counts only toward the confirmed population, read at the
    union of the scopes of the kept members that can answer (`DefaultPresence.facet_entities`).
    The result carries the union of the scopes of the StatVars it names as
    ``default_scope``. A presence read that lacks one of those members raises
    `AssertionError`; a subject entity beside it raises too.

    With no place, return bare coverage after requiring an explicit match for
    every bound property. With a place probe and no observations, return the
    unavailable carrier with ``available_at_place=False``. The no-place path
    uses ``None`` because availability was unassessed. Empty or ineligible
    populations return a named ``NoDataDraft``, and bound values that together exclude
    every member return its ``BoundValuesUnrealisable`` form. Shape-level candidate selection
    belongs to the caller; this function resolves one shape.
    """
    from search.engine.retrieval.retrieve import MemberPopulation

    place_dcid = next(iter(entities.get("observationAbout", ())), None)

    admitted = _admit_standard_members(shape, bindings)
    if not isinstance(admitted, _AdmittedStandardMembers):
        return admitted
    bound_values = admitted.bound_values
    member_facts = admitted.member_facts
    exact = admitted.exact
    representative_sv = admitted.representative_sv
    # Refuse a declared default over the admitted population before any return, as
    # `graph_confirm_resolve` does, so no answer path serves a StatVar coverage cannot date.
    refuse_declared_defaults(graph, list(member_facts))

    probe_members, multi_collect = _select_probe_members(
        list(member_facts), bindings, member_facts, bound_values
    )
    # The members the datasets mint keeps; a set binding's eligible members are the ones
    # matching a set value.
    family = (
        _elect_family(probe_members, member_facts, _collapse_bindings(bindings))
        if multi_collect
        else admitted.family
    )
    # The kept members the probe can confirm.
    probed = set(probe_members)
    answering = [sv for sv in family.order if sv in probed] or probe_members
    if default_presence is not None:
        if place_dcid is not None:
            raise AssertionError(
                f"a default place scope was asked for beside the subject entity {place_dcid!r}"
            )
        place_dcids = default_presence.facet_entities(answering)

    # The places the default-scope facet reads covered, once they have been read.
    read_places: set[str] | None = None

    def scoped(result: Materialised) -> Materialised:
        """`result` settled on the default scope of the StatVars it serves. A scope naming a
        place the facet reads did not cover raises: its rows were never read, and taking
        them as empty would answer a place with no data it may hold."""
        if default_presence is None:
            return result
        scope = default_presence.union_scope(result.sv_dcids)
        if read_places is not None and (unread := set(scope.places) - read_places):
            raise AssertionError(
                f"the default scope of {result.sv_dcids!r} settles on {sorted(unread)!r}, "
                "which the facet reads did not cover"
            )
        return replace(result, default_scope=scope)

    scope_check = _default_scope_short_circuit(
        place_dcid=place_dcid,
        place_dcids=place_dcids,
        member_facts=member_facts,
        family=admitted.family,
        representative_sv=representative_sv,
        date_request=date_request,
        exact=exact,
    )
    if scope_check.terminal is not None:
        return scoped(scope_check.terminal)

    probe_key = _standard_probe_key(list(member_facts), place_dcid, date_request, place_dcids)
    if probe_key is None:
        return _no_entity_disambiguation(
            member_facts=member_facts,
            family=admitted.family,
            representative_sv=representative_sv,
            date_request=date_request,
            exact=exact,
        )

    members, probe_entities, needs_dates = probe_key
    places_of = _member_places(
        members,
        probe_entities,
        own_scope_members=family.order,
        presence=default_presence if scope_check.active else None,
    )

    years = plan_years(date_request)
    if default_presence is not None and scope_check.active:
        read_places = {place for places in places_of.values() for place in places}
    facets_by_sv_batch, confirmed, outside_years = _probe_and_scan(
        places_of=places_of,
        probe_members=probe_members,
        needs_dates=needs_dates,
        graph=graph,
        years=years,
        datasets=elected_datasets(member_facts),
        minted=family.order,
    )

    population = MemberPopulation(
        admitted=len(member_facts),
        admitted_dcids=tuple(member_facts),
        confirmed=len(confirmed),
        exact=exact,
    )
    # A member is elected among those holding a point in the years read, the members the
    # plan can serve; a member holding data at the place in other years only makes an empty
    # answer one for the window. Both lists keep only the members the datasets mint keeps.
    if multi_collect:
        kept = set(family.order)
        collected = [member for member in confirmed if member[0] in kept]
        held = [member for member in outside_years if member[0] in kept]
    else:
        by_sv = {member[0]: member for member in confirmed}
        shown = next((sv for sv in family.order if sv in by_sv), None)
        collected = [by_sv[shown]] if shown is not None else []
        outside_by_sv = {member[0]: member for member in outside_years}
        held = [outside_by_sv[sv] for sv in family.order if sv in outside_by_sv][:1]
    # A window answer with no point names its first held member alone.
    named_members = collected or held[:1]
    family_total_absent: FamilyTotalAbsent | None = None
    if family.order and len(named_members) <= 1:
        named = named_members[0][0] if named_members else family.order[0]
        family_total_absent = family.total_absent(named, places_of[family.order[0]], years=years)
    coverage_facet_label, is_plural = _coverage_label(
        tuple(dict.fromkeys(e for sv, _, _ in named_members for e in places_of[sv])),
        place_dcids=place_dcids,
        default_scope=(
            default_presence.union_scope([sv for sv, _, _ in named_members])
            if default_presence is not None and scope_check.active
            else None
        ),
    )

    if not collected and held:
        return scoped(
            _no_data_in_years(
                held[0],
                dateless_plan=_standard_plan(
                    [sv for sv, _, _ in held],
                    facets_by_sv_batch=facets_by_sv_batch,
                    places_of=places_of,
                    member_facts=member_facts,
                    years=None,
                    scope=scope_check.active and is_plural,
                ),
                date_request=date_request,
                default_scope_probe=scope_check.active,
                member_facts=member_facts,
                population=population,
                family_total_absent=family_total_absent,
            )
        )
    if not collected:
        return scoped(
            _no_data_at_place(
                representative_sv=family.order[0] if family.order else representative_sv,
                date_request=date_request,
                default_scope_probe=scope_check.active,
                member_facts=member_facts,
                population=population,
                family_total_absent=family_total_absent,
            )
        )

    return scoped(
        _materialise_collected(
            collected,
            facets_by_sv_batch=facets_by_sv_batch,
            places_of=places_of,
            is_plural=is_plural,
            coverage_facet_label=coverage_facet_label,
            date_request=date_request,
            default_scope_probe=scope_check.active,
            member_facts=member_facts,
            population=population,
            family_total_absent=family_total_absent,
        )
    )


def standard_option_coverage(
    *,
    shape: ShapeDraft,
    slot_options: list[SlotOptions],
    entities: Mapping[str, list[str]],
    materialise_graph: GraphReader | None,
    date_request: DateRequest | None = None,
    place_dcids: list[str] | None = None,
    datasets: ElectedDatasets = NO_DATASETS,
    pick_members: Mapping[str, Mapping[str, Sequence[str] | None]] | None = None,
    facets_pinned: bool = False,
) -> list[SlotOptions]:
    """Probe option values across the shape's sibling members.

    One batch covers matching StatVars from every slot, with each StatVar
    probed once. ``SEARCH_MAX_OPTIONS`` limits option values, not probe size.
    The preview uses recalled ``sv_arc_facts`` because bound members are not
    yet known. A missing ``materialise_graph`` is a no-op. Each option's coverage is
    stated from a plan over its matching facets (`plan_standard`) under the request's
    years: the group the plan elects for each StatVar across the probed places, as an
    answer would serve it. A member with an elected dataset in `datasets` counts only the
    facets that dataset publishes.

    `pick_members`, keyed by slot property then option value, names the members a pick of
    each option reads (`option_coverage._pick_members`). With it each option is probed
    over those members and carries `served_rows`, the rows its plan serves as the download
    counts them (`download.pair_extent`). The count is an upper bound under `facets_pinned`,
    when the answer's identity binds a facet. An option mapped to no member counts 0, and
    one mapped to None is left unprobed and uncounted. Without `pick_members` an option is
    probed over every member carrying its value, and carries coverage alone.
    """
    from search.engine.clients.materialise_bridge import (
        observation_facets_via_materialise,
    )
    from search.engine.retrieval.coverage import CoverageForm, answer_coverage
    from search.engine.retrieval.download import pair_extent

    place_dcid = next(iter(entities.get("observationAbout", ())), None)

    sv_arc_facts = shape.sv_arc_facts or {}
    if not sv_arc_facts or not slot_options:
        return slot_options
    if materialise_graph is None:
        return slot_options

    # Option coverage uses recalled arcs because it runs before binding.
    probe_key = _standard_probe_key(list(sv_arc_facts), place_dcid, date_request, place_dcids)
    if probe_key is None:
        return slot_options
    _members, probe_entities, _needs_dates = probe_key
    is_plural = len(probe_entities) > 1

    # Union matching StatVars so a member serving several slots is probed once.
    resolved: dict[int, SlotOptions] = {}
    pending: list[tuple[int, SlotOptions, dict[str, list[str] | None]]] = []
    union_svs: set[str] = set()
    for i, so in enumerate(slot_options):
        prop_dcid = so.key.property.dcid if so.key.property else None
        if prop_dcid is None:
            resolved[i] = so
            continue

        picks = pick_members.get(prop_dcid) if pick_members is not None else None
        sv_by_option: dict[str, list[str] | None] = {}
        for opt in so.options:
            dcid = opt.value.ref.dcid if opt.value.ref else None
            if dcid is None:
                continue
            if pick_members is not None:
                found = picks.get(dcid) if picks is not None else None
                sv_by_option[dcid] = list(found) if found is not None else None
                continue
            sv_by_option[dcid] = [
                sv
                for sv, arcs in sv_arc_facts.items()
                if read_constraints(arcs).get(prop_dcid) == dcid
            ]

        all_svs = {sv for svs in sv_by_option.values() if svs for sv in svs}
        if not all_svs and pick_members is None:
            resolved[i] = so
            continue

        pending.append((i, so, sv_by_option))
        union_svs.update(all_svs)

    facets_batch: dict[str, dict[str, list[Facet]]] = {}
    if union_svs:
        facets_batch = observation_facets_via_materialise(
            stat_vars=sorted(union_svs),
            entities=list(probe_entities),
            materialise_graph=materialise_graph,
        )

    for i, so, sv_by_option in pending:
        new_options: list[SlotOption] = []
        for opt in so.options:
            dcid = opt.value.ref.dcid if opt.value.ref else None
            svs = sv_by_option.get(dcid) if dcid else None
            if not svs:
                # Preserve fields this wrapper does not own. A pick reading no member
                # serves nothing. A pick whose members are unknown stays uncounted.
                counted = (
                    {"served_rows": 0, "served_rows_exact": True}
                    if svs is not None and pick_members is not None
                    else {}
                )
                new_options.append(opt.model_copy(update={"coverage": None, **counted}))
                continue
            series = {
                (sv, entity): entity_facets
                for sv in svs
                for entity in probe_entities
                if (
                    entity_facets := published(
                        datasets, sv, facets_batch.get(sv, {}).get(entity, [])
                    )
                )
            }
            plan = plan_standard(series, members=svs, years=plan_years(date_request))
            coverage = answer_coverage(
                plan,
                CoverageForm(
                    facet_label="countries" if is_plural else "sources",
                    counts_places=is_plural,
                    variables=True,
                ),
            )
            counted: dict[str, int | bool] = {}
            if pick_members is not None:
                extent = pair_extent(plan, plan.years)
                counted = {
                    "served_rows": extent.rows,
                    "served_rows_exact": extent.exact and not facets_pinned,
                }
            new_options.append(opt.model_copy(update={"coverage": coverage, **counted}))
        resolved[i] = SlotOptions(
            key=so.key,
            options=new_options,
            truncated=so.truncated,
            source=so.source,
        )

    return [resolved[i] for i in range(len(slot_options))]
