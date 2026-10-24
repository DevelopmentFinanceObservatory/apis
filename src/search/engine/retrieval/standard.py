"""Resolve standard shapes by probing observations through the injected graph client."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, NamedTuple

from code_index import absent_readings
from code_index.members import FamilyQuery, SourceNotInFamily, option_families
from code_index.shared import get_index
from materialise.place_scope import DefaultPlaceScope, DefaultPresence
from search.engine.clients.materialise_bridge import _FACET_PROBE_MAX_PAIRS_PER_POST
from search.engine.place import is_worldwide_scope
from search.engine.retrieval.bound_values import (
    RoleKeys,
    _bound_value_conflicts,
    _bound_value_filter_facts,
    _collapse_bindings,
    _fully_bound_members_facts,
    bound_values_from_bindings,
    entity_role_keys,
    held_family_exact,
    query_bound_values,
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
from search.engine.retrieval.member_facts import (
    IndexMemberFacts,
    asserted_constraint_properties,
    member_name,
)
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
from search.engine.shape import is_role_bearing
from search.models import SlotOption, SlotOptions, TimeWindow
from statvar_members import antichain_minimal_ids
from statvar_members.enumerate import dominates_constraint_keys

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


def family_totals(
    member_facts: Mapping[str, MemberFacts], stated_values: Mapping[str, list[str]]
) -> frozenset[str]:
    """The members of `member_facts` that are the family's total under the bindings'
    `stated_values`, by the rule `_elect_family` elects by."""
    return _totals(
        {sv: _read_assertions(facts, stated_values) for sv, facts in member_facts.items()}
    )


def holds_broader_reading(
    member_facts: Mapping[str, MemberFacts],
    stated_values: Mapping[str, list[str]],
    prop: str,
    value: str,
) -> bool:
    """Whether `member_facts` hold a reading of the part without `value` on `prop`: a
    member that states nothing on `prop` and whose every constraint a member asserting
    `value` shares, so the mint keeps it in place of that member while `prop` stays open
    (`dominates_constraint_keys`).

    The part is the members stating one of the `stated_values` on each bound property. A
    member whose constraints are unknown or unresolved is no witness. A family that splits
    on a slot the part leaves open holds no total (`family_totals`), and each side's
    broadest member is the reading an open `prop` serves there: "net transfers on external
    debt" beside the series for long-term debt without IMF credit."""
    readings = absent_readings(get_index())
    part: list[Mapping[str, str]] = []
    for facts in member_facts.values():
        properties = asserted_constraint_properties(facts)
        constraints = facts.read_constraints()
        if properties is None or properties != constraints.keys():
            continue
        if all(constraints.get(bound) in values for bound, values in stated_values.items()):
            part.append(constraints)
    asserting = [member for member in part if member.get(prop) == value]
    return any(
        prop not in broader
        and broader.items() <= narrower.items()
        and dominates_constraint_keys(frozenset(broader), frozenset(narrower), readings=readings)
        for broader in part
        for narrower in asserting
    )


def _totals(read: Mapping[str, _Assertions]) -> frozenset[str]:
    if not read:
        return frozenset()
    shared = set.intersection(*(set(member.open_items) for member in read.values()))
    return frozenset(
        sv
        for sv, member in read.items()
        if member.can_be_total and shared.issuperset(member.open_items)
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

    Kept members sort totals first. With no total, the members narrowing the family least
    sort first: members with known constraints before unknown ones, then fewer asserted
    unbound properties first, so a recall score never puts a narrower slice ahead of a
    broader member. Recall breaks the remaining ties, by the position of the first member
    whose unbound assertions include the member's own. A slot that partitions the family,
    one every member states with two or more values among them, is bound before the part
    resolves when every member recall scored states one value on it
    (`bind_partition_values`).
    """
    read = {sv: _read_assertions(member_facts[sv], stated_values) for sv in svs}
    # The positions of the members asserting each unbound (property, value) pair.
    positions: dict[tuple[str, str], set[int]] = {}
    for i, sv in enumerate(svs):
        for item in read[sv].open_items:
            positions.setdefault(item, set()).add(i)

    totals = _totals(read)
    index_of = {sv: i for i, sv in enumerate(svs)}

    def rank(sv: str) -> tuple[bool, bool, int, int]:
        member = read[sv]
        if member.properties is None or len(member.open_items) < member.open_count:
            first = index_of[sv]
        elif member.open_items:
            first = min(set.intersection(*(positions[item] for item in member.open_items)))
        else:
            first = 0
        return sv not in totals, member.properties is None, member.open_count, first

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


class _Answering(NamedTuple):
    """The members a resolve over admitted members probes (`_select_probe_members`),
    whether a set binding collects several results, the family it elects among, and the
    kept members whose scopes its facet reads union."""

    probe_members: list[str]
    multi_collect: bool
    family: _Family
    answering: list[str]


def _answering(admitted: _AdmittedStandardMembers, bindings: list[SlotBindingDraft]) -> _Answering:
    """The members `standard_resolve` reads over `admitted` under `bindings`. The family is
    the members the datasets mint keeps (`_elect_family`): the admitted family, or for a set
    binding the kept members matching a set value. The answering members are the kept ones
    the probe can confirm, else every probe member."""
    member_facts = admitted.member_facts
    probe_members, multi_collect = _select_probe_members(
        list(member_facts), bindings, member_facts, admitted.bound_values
    )
    family = (
        _elect_family(probe_members, member_facts, _collapse_bindings(bindings))
        if multi_collect
        else admitted.family
    )
    probed = set(probe_members)
    answering = [sv for sv in family.order if sv in probed] or probe_members
    return _Answering(probe_members, multi_collect, family, answering)


def _serving(family: _Family, member_facts: Mapping[str, MemberFacts]) -> tuple[list[str], bool]:
    """The kept members of `family` an answer can serve, in election order, and whether it
    serves every one of them holding a point. A family with no total is served as a mint of
    its definition serves it: every kept member whose constraints are known, since the mint
    drops the rest. A family with a total, or one whose kept constraints are all unknown,
    answers by its first kept member holding data."""
    known = [
        sv for sv in family.order if asserted_constraint_properties(member_facts[sv]) is not None
    ]
    if not family.totals and known:
        return known, True
    return list(family.order), False


def default_scope_members(
    shape: ShapeDraft, bindings: list[SlotBindingDraft], role_keys: Collection[str]
) -> list[str]:
    """The StatVars `standard_resolve` reads at their own default place scope for `shape`
    under `bindings` with its places bound under `role_keys`: the family a result can name,
    as admitted (`_admit_standard_members`) and as elected for a set binding, then the
    members whose scopes its facet reads union (`_answering`), then the representative a
    result names when no member confirms. A default-scope presence read over these covers
    every scope the resolve reads. Empty when the resolve refuses before it reads."""
    admitted = _admit_standard_members(shape, bindings, role_keys)
    if not isinstance(admitted, _AdmittedStandardMembers):
        return []
    answer = _answering(admitted, bindings)
    return list(
        dict.fromkeys(
            [
                *admitted.family.order,
                *answer.family.order,
                *answer.answering,
                admitted.representative_sv,
            ]
        )
    )


def members_served(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    role_keys: RoleKeys,
    holds: Callable[[str], bool | None],
) -> list[str]:
    """The members an answer over `shape` under `bindings`, with its places bound under
    `role_keys`, can serve, by the rule `standard_resolve` serves by, given whether each
    member holds a point where and when the part asks (`holds`, None when no read settled
    it). A set binding collects every kept member holding one, and so does a family with no
    total (`_serving`), so nominal GDP alone answers "GDP" in Brazil among the gross value
    added series no read finds there. Any other family answers by its first kept member
    holding one, so the members are those in election order up to the first that does,
    skipping each that holds none. With no member that may hold one, the answer names one
    of the kept members, and they are all returned. A role-bearing shape serves the members
    a role-bearing answer lists (`_served_under`). Empty when the bindings admit no
    family."""
    if is_role_bearing(shape):
        return list(_served_under(shape, list(bindings), role_keys))
    admitted = _admit_standard_members(shape, list(bindings), role_keys.bound)
    if not isinstance(admitted, _AdmittedStandardMembers):
        return []
    answer = _answering(admitted, list(bindings))
    if answer.multi_collect:
        kept, every = list(answer.family.order), True
    else:
        kept, every = _serving(answer.family, admitted.member_facts)
    served: list[str] = []
    for sv in kept:
        held = holds(sv)
        if held is not False:
            served.append(sv)
        if held and not every:
            break
    return served or kept


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
    shape: ShapeDraft, bindings: list[SlotBindingDraft], role_keys: Collection[str]
) -> NoDataDraft | _AdmittedStandardMembers:
    """Source members, elected under the bindings and `role_keys`, and apply the bound-value
    filter.

    A member asserting a slot a breakdown decision left out of `shape`
    (`ShapeDraft.omitted_slots`) is left out, as the mint of the identity minted from the
    shape leaves it out, so the shape admits the members its identity serves. An empty
    shape, absent source population, or degenerate value/set binding returns
    ``NoDataDraft(reason="variable_not_resolved")``. Missing constraint values pass, and mismatched asserted values are filtered. Bound values that
    together exclude every member return ``BoundValuesUnrealisable``, naming each
    member's conflicting bound properties, so the caller can unbind a minimal set
    and resolve again. ``exact`` marks a complete population: the index's family, an
    exact member set, or a family the catalogue's tables hold whole (`held_family_exact`);
    capped recall and a held family the graph may extend are floors. Members keep
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
    member_facts, index_sourced = shape_member_facts(shape, shape_query(shape, bindings, role_keys))
    # A member asserting a slot the breakdown decision left out is no member of a mint of
    # the identity (`statvar_members.enumerate._confirm_member`).
    if shape.omitted_slots:
        member_facts = {
            sv: member
            for sv, member in member_facts.items()
            if shape.omitted_slots.isdisjoint(asserted_constraint_properties(member) or ())
        }
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
    exact = index_sourced or shape.sv_arc_facts_complete or held_family_exact(shape)

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


def _recalled_partition_values(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], admitted: _AdmittedStandardMembers
) -> dict[str, str]:
    """Each what or how slot property of `shape` that partitions the admitted family, in
    slot-key order, with the one value recall evidences on it. A slot partitions the family
    when `bindings` leave it open and every member states a value on it, with two or more
    values among them; a member that states no value, or whose constraints are unknown,
    leaves it no partition. Recall evidences a value when every member recall scored
    (`ShapeDraft.scored_members`) that the family admits states that value; a slot whose
    scored members state two or more values, or that no scored member reaches, is left
    out."""
    constraints = {
        sv: admitted.member_facts[sv].read_constraints() for sv in admitted.family.members
    }
    recalled = [constraints[sv] for sv in constraints if sv in shape.scored_members]
    values: dict[str, str] = {}
    for prop in _partition_properties(shape, bindings, list(constraints.values())):
        evidenced = {member[prop] for member in recalled}
        if len(evidenced) == 1:
            values[prop] = evidenced.pop()
    return values


def _partition_properties(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    constraints: Sequence[Mapping[str, str]],
    *,
    operation_slots: Collection[str] = (),
) -> list[str]:
    """Each what or how slot property of `shape` that partitions the members whose
    `constraints` these are, in slot-key order: one `bindings` leave open on which every
    member states a value, with two or more values among them. Such members hold no total
    over the slot. A slot `bindings` bind to a value, or leave open as `requested_all` (a
    phrase asking for every value), is not open, and neither is one of `operation_slots`,
    the slots an operation reads, whatever binding it holds."""
    closed = (
        bound_values_from_bindings(bindings).keys()
        | {b.property_dcid for b in bindings if b.evidence == "requested_all"}
        | set(operation_slots)
    )
    return [
        prop
        for slot in shape.slot_keys
        if (prop := slot.property_dcid) is not None
        and slot.axis in ("what", "how")
        and prop not in closed
        and all(prop in member for member in constraints)
        and len({member[prop] for member in constraints}) >= 2
    ]


def _served_under(
    shape: ShapeDraft, bindings: list[SlotBindingDraft], role_keys: RoleKeys
) -> dict[str, MemberFacts]:
    """The members a part over `shape` would serve under `bindings` and `role_keys`, read
    from the population it holds before any walk settles one. A role-bearing shape serves
    the members a role-bearing answer lists (`role_resolve.role_members_served`) of the
    population `shape_member_facts` gives, the read `_admit_population` makes. Any other
    serves from the family `_admit_standard_members` admits. Empty for a role-bearing
    population the code index decides, whose rows carry no name, and for bindings that
    admit no family."""
    from search.engine.retrieval.role_resolve import role_members_served

    if not is_role_bearing(shape):
        admitted = _admit_standard_members(shape, bindings, role_keys.bound)
        if not isinstance(admitted, _AdmittedStandardMembers):
            return {}
        return {sv: admitted.member_facts[sv] for sv in admitted.family.members}
    query = shape_query(shape, bindings, role_keys.bound, open_role_keys=role_keys.open)
    member_facts, index_sourced = shape_member_facts(shape, query)
    if index_sourced:
        return {}
    served = role_members_served(
        member_facts, query_bound_values(query), role_keys.bound, role_keys.open
    )
    return {sv: member_facts[sv] for sv in served}


def _without_spanning_sets(
    shape: ShapeDraft,
    bindings: list[SlotBindingDraft],
    role_keys: RoleKeys,
    *,
    operation_slots: Collection[str],
) -> list[SlotBindingDraft]:
    """`bindings` without each `set` binding that narrows nothing: one whose values every
    member the part would serve without it (`_served_under`) states on its property. Such
    a slot reads as an open one. A set that leaves out a served member's value, or whose
    property some served member does not state, narrows the part and stays. So does a set
    on one of `operation_slots`, a slot an operation reads: its values are the breakdown
    the reader asked for."""
    kept = list(bindings)
    for binding in bindings:
        prop = binding.property_dcid
        if binding.kind != "set" or prop is None or prop in operation_slots:
            continue
        without = [b for b in kept if b is not binding]
        served = _served_under(shape, without, role_keys)
        values = bound_values_from_bindings([binding]).get(prop, ())
        if served and all(
            member.read_constraints().get(prop) in values for member in served.values()
        ):
            kept = without
    return kept


def bind_named_member_values(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    role_keys: RoleKeys,
    *,
    variable: str,
    written: str | None = None,
    operation_slots: Collection[str] = (),
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Bind each open slot that partitions the members a part would serve to the value of
    the one served member the variable phrase names.

    Served members that all state a value on an open slot, with two or more values among
    them, hold no total over it (`_partition_properties`), so the answer lists one series
    for each value. A phrase equal to the graph name of exactly one of those members asks
    for that member: "net transfers on external debt" is the name of the member netting
    servicing payments, beside one netting principal repayments. Every such slot then
    binds to the named member's value. Names compare as `bind.normalised_key` reads them,
    lowercased with only their alphanumeric characters. A slot bound to a set that every
    served member satisfies narrows nothing and reads as open (`_without_spanning_sets`),
    so the named member's value replaces the set. A slot of `operation_slots`, one an
    operation ranks or breaks down over (`phase_rank_bind.operation_slot_phrases`), is the
    reader's breakdown and never binds here, whether it arrives unbound or as a set.

    The variable phrase is extraction's reading of the request and can reorder its words:
    "net flows on external debt for Kenya" arrives as "external debt net flows". Where the
    phrase equals no served member's name, `written` names the member when it equals one
    name, compared the same way: the stretch of the request that writes the phrase
    (`written_phrase.written_phrase`), "net flows on external debt". A single-variable
    raw-text request carries one, and the layer-2 response cache keys it
    (`response_cache_flow.compute_layer2_signals`).

    A phrase and a written stretch equal to no served member's name, or to the names of
    two, bind nothing, and so does a population holding a served member with no name, where
    one match cannot be told from two. A recalled or held member carries its name in its arcs
    (`ArcsMemberFacts`); a code index row carries none. The members are the ones
    `_served_under` reads, and binding a slot can leave a further slot partitioning the
    members still served, so slots bind until none does.

    Returns the bindings with each such slot bound as a `mentioned` value, since the
    phrase named the member that states it, and the bindings added, in the order they
    bound."""
    from search.engine.bind import SlotBindingDraft, normalised_key

    bound = list(bindings)
    added: list[SlotBindingDraft] = []
    asked = normalised_key(variable)
    written_key = normalised_key(written or "")
    axis_of = {slot.property_dcid: slot.axis for slot in shape.slot_keys}
    while asked:
        unnarrowed = _without_spanning_sets(
            shape, bound, role_keys, operation_slots=operation_slots
        )
        served = _served_under(shape, unnarrowed, role_keys)
        constraints = {sv: member.read_constraints() for sv, member in served.items()}
        partitions = _partition_properties(
            shape, unnarrowed, list(constraints.values()), operation_slots=operation_slots
        )
        if not partitions:
            break
        names = {sv: member_name(member) for sv, member in served.items()}
        if any(name is None for name in names.values()):
            break
        named = [sv for sv, name in names.items() if name and normalised_key(name) == asked]
        if not named and written_key:
            named = [
                sv for sv, name in names.items() if name and normalised_key(name) == written_key
            ]
        if len(named) != 1:
            break
        named_bindings = [
            SlotBindingDraft(
                axis=axis_of[prop],
                property_dcid=prop,
                kind="value",
                value_dcids=[constraints[named[0]][prop]],
                evidence="mentioned",
            )
            for prop in partitions
        ]
        bound = [b for b in bound if b.property_dcid not in partitions] + named_bindings
        added.extend(named_bindings)
    return bound, added


def bind_partition_values(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], role_keys: Collection[str]
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Bind each open slot that partitions a standard shape's family to the value recall
    evidences on it.

    A stand-in (`_elect_family`) may narrow the measure asked, but may not cross a slot
    that partitions the family: one the bindings leave open on which every admitted
    member states a value, with two or more values among them. Such a family has no
    total, and any stand-in answers with one of the slot's values, changing the measure.
    The nominal GDP family's members all state `activitySource` (gross domestic
    production, gross or net value added, net domestic production), and the least narrow
    of them is a gross value added series. So the part binds such a slot from recall when
    the evidence is unanimous: every member recall scored (`ShapeDraft.scored_members`)
    that the family admits states the same value on it. For "GDP" the one scored member is
    nominal GDP; the gross value added members are Topic expansions. When the scored
    members state two or more values (a female and a male member both recalled for
    under-five mortality), the evidence picks none, and the slot stays open. Partition
    slots bind one at a time, in slot-key order, until no open slot has unanimous
    evidence. A slot `bindings` bind, by the model's `mentioned` value among others, or
    leave open as `requested_all` is not open. The family is the one
    `_admit_standard_members` admits under `bindings` and `role_keys`, for a shape whose
    family the graph walk decides and the catalogue holds (`ShapeDraft.held_family`), so
    its members are the family read whole. A recall sample is left alone, since the
    family's total may be a member recall did not return. So are a family the index
    decides, whose declarations read a row's missing property as its aggregate and bind
    its defaults, and a role-bearing shape, which `_elect_family` never reads.

    Returns the bindings with each such slot bound as a `sole_option` value, and the
    bindings added, in the order they bound."""
    from search.engine.bind import SlotBindingDraft

    bound = list(bindings)
    added: list[SlotBindingDraft] = []
    if shape.held_family is None or is_role_bearing(shape):
        return bound, added
    axis_of = {slot.property_dcid: slot.axis for slot in shape.slot_keys}
    while True:
        admitted = _admit_standard_members(shape, bound, role_keys)
        if not isinstance(admitted, _AdmittedStandardMembers):
            return bound, added
        values = _recalled_partition_values(shape, bound, admitted)
        if not values:
            return bound, added
        prop, value = next(iter(values.items()))
        binding = SlotBindingDraft(
            axis=axis_of[prop],
            property_dcid=prop,
            kind="value",
            value_dcids=[value],
            evidence="sole_option",
        )
        bound = [b for b in bound if b.property_dcid != prop] + [binding]
        added.append(binding)


@dataclass(frozen=True)
class Keep:
    """The family holds a total, or the decision has nothing to read: the identity carries
    every slot of the shape. `unopened` is set when the slots the request opened leave no
    signature holding them all: the unbound properties outside them that members assert,
    by which the answer shows the family beside the slots it opened, narrower than asked."""

    unopened: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Narrow:
    """The identity carries `properties` and leaves every other what or how slot out, so a
    mint of it serves only the members asserting nothing else. `lone_signature` marks a
    family no bound or opened slot narrowed, whose members all break down by one
    signature: the answer is narrower than asked."""

    properties: frozenset[str]
    lone_signature: bool = False


@dataclass(frozen=True)
class Candidates:
    """The members break down by several `signatures`, or the slots the request opened
    leave no member and every member that asserts them all also asserts a slot it did not
    open. Each signature is the set of unbound properties some member asserts, in member
    order, and holds every opened slot, so each candidate carries the breakdown the
    question named. Each candidate's identity carries the `bound` properties and one
    signature."""

    signatures: tuple[frozenset[str], ...]
    bound: frozenset[str] = frozenset()


type BreakdownDecision = Keep | Narrow | Candidates


def breakdown_decision(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], role_keys: Collection[str]
) -> BreakdownDecision:
    """Which what and how slots the identity of a part over `shape` carries under its settled
    `bindings`, with its places bound under `role_keys`.

    The decision reads the family `_admit_standard_members` admits, before any narrowing. A
    member's signature is the set of its asserted constraint properties no binding binds to
    a value. A family with a total (`family_totals`) keeps every slot, and so do a
    role-bearing shape, which `_elect_family` never reads, a recall sample with no catalogue
    family (`ShapeDraft.held_family`), whose total may be a member recall did not return,
    and bindings that admit no family. The population read is the one
    `bind_partition_values` reads, with the families the code index decides and exact
    member sets beside it. Otherwise, with slots opened as ``requested_all`` (an
    operation's slot, or a breakdown the phrase names), the identity carries the bound and
    opened properties when a member asserts no other (`Narrow`). When none does, the
    signatures holding every opened slot are `Candidates`, and with no such signature the
    family keeps every slot, naming the unopened properties its members assert
    (`Keep.unopened`). With nothing opened, members asserting only bound properties
    narrow to those, one signature narrows to it beside them, and several are `Candidates`.
    A member whose constraints are unknown has no signature, since the mint drops it."""
    if is_role_bearing(shape):
        return Keep()
    admitted = _admit_standard_members(shape, list(bindings), role_keys)
    if not isinstance(admitted, _AdmittedStandardMembers) or not (
        admitted.exact or shape.held_family is not None
    ):
        return Keep()
    if admitted.family.totals:
        return Keep()
    bound = frozenset(admitted.bound_values)
    signatures = tuple(
        dict.fromkeys(
            properties - bound
            for sv in admitted.family.members
            if (properties := asserted_constraint_properties(admitted.member_facts[sv])) is not None
        )
    )
    if not signatures:
        return Keep()
    opened = frozenset(
        prop
        for b in bindings
        if b.evidence == "requested_all" and (prop := b.property_dcid) is not None
    )
    if opened:
        if any(signature <= opened for signature in signatures):
            return Narrow(bound | opened)
        named = tuple(signature for signature in signatures if opened <= signature)
        if named:
            return Candidates(named, bound)
        return Keep(frozenset().union(*signatures) - opened)
    if frozenset() in signatures:
        return Narrow(bound)
    if len(signatures) == 1:
        return Narrow(bound | signatures[0], lone_signature=True)
    return Candidates(signatures, bound)


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
    served: Sequence[str],
    representative_sv: str,
    date_request: DateRequest | None,
    exact: bool,
) -> Materialised:
    """Return bare coverage when no place entity is available.

    No observation probe runs. A member must explicitly assert every bound
    property; the `served` members of those the datasets mint keeps answer, in election
    order (``_elect_family``), with ``family_total_absent`` when they are one member that
    is not the family's total. Otherwise the result has ``has_data=False`` and
    ``available_at_place=None``.
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
        sv_dcids=list(served),
        has_data=True,
        coverage=CoverageBare(has_data=True, window=no_entity_window),
        family_total_absent=(family.total_absent(served[0], ()) if len(served) == 1 else None),
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
    graph: EngineGraphClient,
    places_of: Mapping[str, Sequence[str]],
    *,
    needs_dates: bool,
    settled: Callable[[Mapping[str, dict[str, list[Facet]]]], bool] | None = None,
) -> dict[str, dict[str, list[Facet]]]:
    """Read each StatVar's facets at its own places (`places_of`), keeping only those.

    Of two read plans, the one with fewer POSTs runs, the per-scope plan on a tie, its reads
    one after another. The per-scope plan reads the StatVars sharing a place set
    together, in first-StatVar order; the union plan reads every StatVar at the union of
    their places, in `places_of` order. Each read is chunked by `_facet_reads`. Either way a
    StatVar keeps its facets at its own places alone, so both plans return the same result.
    A StatVar with no places maps to `{}` and is not read.

    `settled`, given the facets read so far, says whether the caller needs no more of
    them. The reads stop after the first read it accepts, and the StatVars no read reached
    are left out of the result. Each read is then split into the groups the graph sends
    apart (`EngineGraphClient.observation_read_groups`), so `settled` is asked after every
    request: the UN System Data Commons reads by GET, about 70 StatVars to a request, and
    a read of a 2,896-member family at one place is 40 of them.
    """
    by_places: dict[tuple[str, ...], list[str]] = {}
    for sv, places in places_of.items():
        if places:
            by_places.setdefault(tuple(places), []).append(sv)
    per_scope = [
        (chunk, places) for places, svs in by_places.items() for chunk in _facet_reads(svs, places)
    ]
    union = list(dict.fromkeys(place for places in by_places for place in places))
    read_svs = [sv for sv, places in places_of.items() if places]
    union_reads = (
        [(chunk, tuple(union)) for chunk in _facet_reads(read_svs, union)] if union else []
    )
    reads = union_reads if len(union_reads) < len(per_scope) else per_scope
    if settled is not None:
        reads = [
            (group, places)
            for chunk, places in reads
            for group in graph.observation_read_groups(chunk, list(places), needs_dates)
        ]

    facets: dict[str, dict[str, list[Facet]]] = {
        sv: {} for sv, places in places_of.items() if not places
    }
    for chunk, places in reads:
        batch = graph.observation_facets_batch(chunk, list(places), needs_dates=needs_dates)
        for sv in chunk:
            own = set(places_of[sv])
            facets[sv] = {e: f for e, f in batch.get(sv, {}).items() if e in own}
        if settled is not None and settled(facets):
            break
    return {sv: facets[sv] for sv in places_of if sv in facets}


@dataclass
class FacetBatch:
    """The facet reads the resolves of one Spec share while its place sets settle.

    The Spec is resolved at a set's node and, where the node holds no data, at the set's
    members. The first resolve reads each StatVar at `places`, every place either reading
    binds, beside its own places, in one read. A later resolve whose places that read
    covered makes no request.
    """

    places: tuple[str, ...]
    # The pairs read, by date need, and the facets the reads returned for them.
    _read: set[tuple[bool, str, str]] = field(default_factory=set)
    _held: dict[tuple[bool, str], dict[str, list[Facet]]] = field(default_factory=dict)

    def facets(
        self, graph: EngineGraphClient, places_of: Mapping[str, Sequence[str]], *, needs_dates: bool
    ) -> dict[str, dict[str, list[Facet]]]:
        """Each StatVar's facets at its own places (`facets_at_own_places`), reading only
        the StatVars with a place no earlier read covered."""
        unread = {
            sv: tuple(dict.fromkeys((*places, *self.places)))
            for sv, places in places_of.items()
            if any((needs_dates, sv, place) not in self._read for place in places)
        }
        if unread:
            read = facets_at_own_places(graph, unread, needs_dates=needs_dates)
            for sv, places in unread.items():
                self._read.update((needs_dates, sv, place) for place in places)
                self._held.setdefault((needs_dates, sv), {}).update(read[sv])
        facets: dict[str, dict[str, list[Facet]]] = {}
        for sv, places in places_of.items():
            held, own = self._held.get((needs_dates, sv), {}), set(places)
            facets[sv] = {place: found for place, found in held.items() if place in own}
        return facets


class _Probe(NamedTuple):
    """What `_probe_and_scan` read and found."""

    # Each member read, to its facets at its own places that its elected dataset publishes.
    facets_by_sv: dict[str, dict[str, list[Facet]]]
    # The members read holding a point in the years read, in probe order.
    confirmed: list[_ConfirmedMember]
    # The members read holding data at their places in other years only.
    outside_years: list[_ConfirmedMember]
    # Whether every member was read, so `confirmed` counts every member holding a point.
    read_all: bool


def _positive_entities(
    by_entity: Mapping[str, list[Facet]], places: Sequence[str], years: PlanYears
) -> set[str]:
    """The places of `places` at which `by_entity` holds a point in `years`."""
    return {entity for entity in places if has_points(by_entity.get(entity, []), years)}


def _probe_and_scan(
    *,
    places_of: Mapping[str, Sequence[str]],
    probe_members: Sequence[str],
    needs_dates: bool,
    graph: EngineGraphClient,
    years: PlanYears,
    datasets: ElectedDatasets,
    minted: Collection[str],
    elects: Sequence[str] | None = None,
    facet_batch: FacetBatch | None = None,
) -> _Probe:
    """Probe the members at their own places (`places_of`, `facets_at_own_places`), then
    scan the members read for points in `years`, each at its own places. With `facet_batch`
    the probe is one of the reads a settling Spec shares (`FacetBatch.facets`).

    `elects` is the election order of an answer that names one member: the first of them
    holding a point in `years` answers. Its reads stop once they settle that member, the
    first holding a point with every member before it read, or once they have read every one
    of them, so a family the tables hold whole is read only as far as its answer. "Latest
    life expectancy data" at the default scope elects the family's total from its first
    read, where reading all 351 members of the held family takes 27 reads of 2 to 3 s each.
    Without `elects`, as for a set binding, a family keeping several members, which the
    answer serves together, or a family whose `minted` members carry an elected dataset,
    every member is read, and so is every member of a `facet_batch` read.

    A member's facets union across its places, and it is confirmed at a place holding a
    point in `years` by the point rule every plan and election reads
    (`read_plan.has_points`); only its own places count, and election happens later.

    A member of an index-resolved family keeps only the facets its elected dataset
    publishes (`datasets`), the ones a mint serves it from. Each member a mint of the answer
    serves (`minted`) is first checked as the mint checks it, over every facet it holds at
    its places (`check_elected_datasets`). A member the reads stopped before is not
    checked, so `elects` is given only when no member of `minted` has an elected dataset.

    Returns the facets of the members read, the confirmed members, the members holding data
    at their places in other years only, which make an empty answer one for the window
    rather than the place, and whether every member was read.
    """

    def settled(read: Mapping[str, dict[str, list[Facet]]]) -> bool:
        assert elects is not None
        for sv in elects:
            if sv not in read:
                return False
            if _positive_entities(
                {entity: published(datasets, sv, facets) for entity, facets in read[sv].items()},
                places_of[sv],
                years,
            ):
                return True
        return True

    # Each member is read at its own places, the entities its mint reads it at. A member the
    # mint serves is checked over every facet there, and each keeps the facets its elected
    # dataset publishes.
    own = (
        facets_at_own_places(
            graph,
            places_of,
            needs_dates=needs_dates,
            settled=settled if elects is not None else None,
        )
        if facet_batch is None
        else facet_batch.facets(graph, places_of, needs_dates=needs_dates)
    )
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
        positive_entities = _positive_entities(facets_by_entity, places_of[sv], years)
        if positive_entities:
            plural = len(places_of[sv]) > 1
            confirmed.append((sv, facets, positive_entities if plural else None))
        elif any(f.obs_count > 0 for f in facets):
            outside_years.append((sv, facets, None))
    return _Probe(
        facets_by_sv=facets_by_sv_batch,
        confirmed=confirmed,
        outside_years=outside_years,
        read_all=all(sv in own for sv in probe_members),
    )


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
        # The request deadline bound to the context bounds the breadth read.
        deadline=None,
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
    facet_batch: FacetBatch | None = None,
) -> Materialised | NoDataDraft | MaterialisedCandidates:
    """Resolve a confirmed shape against its subject entities.

    Probe the admitted members at their own places (`facets_at_own_places`), through
    `facet_batch` when the Spec's place sets are settling (`FacetBatch`), in election order:
    an answer naming one member reads them only until its election settles
    (`_probe_and_scan`), and its confirmed count is then a floor
    (`MemberPopulation.confirmed_exact`). A family the answer serves whole, or one with a
    kept member that has an elected dataset, is read whole, since the mint serves or checks
    each such member. Bindings determine eligibility.
    The answer reads the members the datasets mint keeps from the eligible family
    (``_elect_family``). A family with no total serves every one of them whose constraints
    are known and that holds a point in the years read, as a mint of its definition does,
    which drops the rest. A family with a total, or one whose kept members' constraints are
    all unknown, elects the first of them in election order that holds one, and a
    ``kind="set"`` binding returns each of them, in probe order. With none, a kept member
    holding data at the place in other years answers for the window
    (``_no_data_in_years``), and with no kept member holding data the first in election
    order is flagged as unavailable. A result naming one member, served or flagged, that
    is not the family's total records it in ``family_total_absent``, so an answer whose
    total holds no data says which member it shows. A place-set probe unions facets across
    its entities. An empty default scope can return before a probe because its presence
    read already established the result.

    ``default_presence`` defaults the place scope of a shape with no subject entity, in
    place of ``place_dcids``. Each member the result can name (the kept family) is read at
    its own default scope (`DefaultPresence.scope`), which is where a datasets mint fetches
    it, and every other member counts only toward the confirmed population, read at the
    union of the scopes of the kept members that can answer (`DefaultPresence.facet_entities`).
    The result carries the union of the scopes of the StatVars it names as
    ``default_scope``, and none when no member asserts every bound property, since it then
    serves no StatVar. A presence read that lacks one of those members raises
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

    # A standard shape declares no flow role, so it leaves no axis open.
    admitted = _admit_standard_members(
        shape, bindings, entity_role_keys(entities, open_axes=()).bound
    )
    if not isinstance(admitted, _AdmittedStandardMembers):
        return admitted
    member_facts = admitted.member_facts
    exact = admitted.exact
    representative_sv = admitted.representative_sv
    # Refuse a declared default over the admitted population before any return, as
    # `graph_confirm_resolve` does, so no answer path serves a StatVar coverage cannot date.
    refuse_declared_defaults(graph, list(member_facts))

    probe_members, multi_collect, family, answering = _answering(admitted, bindings)
    served, serves_every = _serving(family, member_facts)
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
        # With no member asserting every bound property the result serves no StatVar: it
        # names its representative only to flag it, and settles on no default scope.
        if not admitted.family.members:
            return scope_check.terminal
        return scoped(scope_check.terminal)

    probe_key = _standard_probe_key(list(member_facts), place_dcid, date_request, place_dcids)
    if probe_key is None:
        return _no_entity_disambiguation(
            member_facts=member_facts,
            family=admitted.family,
            served=served if serves_every else [representative_sv],
            representative_sv=representative_sv,
            date_request=date_request,
            exact=exact,
        )

    members, probe_entities, needs_dates = probe_key
    # Members in election order, so the reads reach the answering member first.
    kept = set(family.order)
    places_of = _member_places(
        [*family.order, *(sv for sv in members if sv not in kept)],
        probe_entities,
        own_scope_members=family.order,
        presence=default_presence if scope_check.active else None,
    )

    years = plan_years(date_request)
    if default_presence is not None and scope_check.active:
        read_places = {place for places in places_of.values() for place in places}
    datasets = elected_datasets(member_facts)
    probe = _probe_and_scan(
        places_of=places_of,
        probe_members=probe_members,
        needs_dates=needs_dates,
        graph=graph,
        years=years,
        datasets=datasets,
        minted=family.order,
        # The mint checks every kept member's elected dataset, so such a family is read
        # whole, and so is one the answer serves every member of.
        elects=None
        if multi_collect or serves_every or not datasets.keys().isdisjoint(family.order)
        else family.order,
        facet_batch=facet_batch,
    )
    facets_by_sv_batch, confirmed, outside_years = (
        probe.facets_by_sv,
        probe.confirmed,
        probe.outside_years,
    )

    population = MemberPopulation(
        admitted=len(member_facts),
        admitted_dcids=tuple(member_facts),
        confirmed=len(confirmed),
        exact=exact,
        confirmed_exact=probe.read_all,
    )
    # The answer serves the `served` members holding a point in the years read, every one
    # of them or the first in election order; a member holding data at the place in other
    # years only makes an empty answer one for the window. Both lists keep only the members
    # the datasets mint keeps, a set binding's in probe order and any other's in election
    # order.
    if multi_collect:
        collected = [member for member in confirmed if member[0] in kept]
        held = [member for member in outside_years if member[0] in kept]
    else:
        by_sv = {member[0]: member for member in confirmed}
        outside_by_sv = {member[0]: member for member in outside_years}
        collected = [by_sv[sv] for sv in served if sv in by_sv]
        held = [outside_by_sv[sv] for sv in served if sv in outside_by_sv]
        if not serves_every:
            collected, held = collected[:1], held[:1]
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


def _option_datasets(
    query: FamilyQuery | None, prop: str, sv_by_option: Mapping[str, Sequence[str] | None]
) -> dict[str, ElectedDatasets]:
    """The elected dataset of each StatVar of `sv_by_option` that is a member of the family
    the code index elects for `query` with `prop` bound to its option, by option. A StatVar
    outside that family is absent from its option's map, which is empty when the family
    holds none. An option the index does not decide is absent, as is every option without
    `query` or when `prop` is not one of its slots."""
    values = [value for value, svs in sv_by_option.items() if svs]
    if query is None or prop not in query.slots or not values:
        return {}
    try:
        families = option_families(get_index(), query, prop, values)
    except SourceNotInFamily:
        return {}
    if families is None:
        return {}
    by_option: dict[str, ElectedDatasets] = {}
    for value, family in families.items():
        svs = set(sv_by_option[value] or ())
        by_option[value] = elected_datasets(
            {
                row.dcid: IndexMemberFacts(row=row, dataset=family.provenance)
                for row in family.rows
                if row.dcid in svs
            }
        )
    return by_option


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
    query: FamilyQuery | None = None,
) -> list[SlotOptions]:
    """Probe option values across the shape's sibling members.

    One batch covers matching StatVars from every slot, with each StatVar
    probed once. ``SEARCH_MAX_OPTIONS`` limits option values, not probe size.
    The preview uses recalled ``sv_arc_facts`` because bound members are not
    yet known. A missing ``materialise_graph`` is a no-op. Each option's coverage is
    stated from a plan over its matching facets (`plan_standard`) under the request's
    years: the group the plan elects for each StatVar across the probed places, as an
    answer would serve it.

    A member counts only the facets its elected dataset publishes. With `query`, the
    answer's family query, the dataset is the one the code index elects once the option is
    bound (`code_index.members.option_families`), so an option reports the table that would
    serve it, and a member outside that table's family counts nothing for it. Without
    `query`, or for an option the index does not decide, it is the member's dataset in
    `datasets`, the answer's own family.

    `pick_members`, keyed by slot property then option value, names the members a pick of
    each option reads (`option_coverage._pick_members`). With it each option is probed
    over those members of the family its election serves and carries `served_rows`, the
    rows its plan serves as the download counts them (`download.pair_extent`). The count is
    an upper bound under `facets_pinned`, when the answer's identity binds a facet. An
    option mapped to no member counts 0, and one mapped to None is left unprobed and
    uncounted. Without `pick_members` an option is probed over every member carrying its
    value, and carries coverage alone.
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
    pending: list[tuple[int, SlotOptions, str, dict[str, list[str] | None]]] = []
    union_svs: set[str] = set()
    for i, so in enumerate(slot_options):
        prop_dcid = so.key.property.ref if so.key.property else None
        if prop_dcid is None:
            resolved[i] = so
            continue

        picks = pick_members.get(prop_dcid) if pick_members is not None else None
        sv_by_option: dict[str, list[str] | None] = {}
        for opt in so.options:
            dcid = opt.value.ref.ref if opt.value.ref else None
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

        pending.append((i, so, prop_dcid, sv_by_option))
        union_svs.update(all_svs)

    facets_batch: dict[str, dict[str, list[Facet]]] = {}
    if union_svs:
        facets_batch = observation_facets_via_materialise(
            stat_vars=sorted(union_svs),
            entities=list(probe_entities),
            materialise_graph=materialise_graph,
        )

    for i, so, prop_dcid, sv_by_option in pending:
        by_option = _option_datasets(query, prop_dcid, sv_by_option)
        new_options: list[SlotOption] = []
        for opt in so.options:
            dcid = opt.value.ref.ref if opt.value.ref else None
            svs = sv_by_option.get(dcid) if dcid else None
            option_datasets = by_option.get(dcid, datasets) if dcid else datasets
            if svs is not None and dcid in by_option:
                # The option's election serves its own family alone.
                svs = [sv for sv in svs if sv in option_datasets]
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
                        option_datasets, sv, facets_batch.get(sv, {}).get(entity, [])
                    )
                )
            }
            plan = plan_standard(
                series,
                members=svs,
                years=plan_years(date_request),
                places=tuple(probe_entities),
                # The request deadline bound to the context bounds the breadth read.
                deadline=None,
            )
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
