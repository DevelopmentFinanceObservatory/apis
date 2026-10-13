"""Build slot, open-role, and facet options for responses or trailing events.

Every slot menu is counted from the series its answer serves (`count_slot_options`) before
the Spec takes it; the coverage probe counts only the menus that leaves uncounted.
Disabled `include_option_coverage` adds no slot-coverage probe and no entity-option label
read. Facet options are built for every definite Spec and candidate either way
(`_maybe_build_facet_options`). The materialised result supplies
open-role membership and counts. Observation previews use a separate dispatch.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, get_args

from code_index.members import (
    FamilyQuery,
    SourceNotInFamily,
    eligible_datasets,
    index_decides,
    option_families,
)
from code_index.shared import get_index
from materialise import GraphReader, facet_axis_value, role_bearing_facet_candidates
from materialise.enrichment import read_party_kinds
from origin_catalogue import recording_un_blocked_places
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.concurrency import create_owned_task
from search.engine.config import settings
from search.engine.errors import GraphInfraError
from search.engine.events import EntityOptionsEvent, FacetOptionsEvent, SlotOptionsEvent
from search.engine.extract import DateRequest
from search.engine.facet_refusals import pin_admitted, pins_admit_facet
from search.engine.region.codes import (
    OPTION_COVERAGE_PROBE_FAILED,
    _OptionEvent,
    _OptionProbeResult,
)
from search.engine.region.facet_probe import NO_PINS, FacetPins, FacetProbe
from search.engine.region.refs import listed_provenance_labels
from search.engine.region.un_system import un_unavailable_warning
from search.engine.retrieval.bound_values import (
    OptionServedFact,
    RoleKeys,
    SeriesRows,
    ServedRows,
    ServedSeries,
    ServedSeriesRows,
    bound_values_from_bindings,
    entity_role_keys,
    held_family_exact,
    identity_query,
    members_by_value,
    query_members,
    served_rows,
    shape_member_facts,
    shape_query,
    synthetic_shape_for_identity,
    without_values,
)
from search.engine.retrieval.download import coverage_kept_years, series_rows
from search.engine.retrieval.elected_dataset import (
    NO_DATASETS,
    ElectedDatasets,
    elected_datasets,
    published,
    publishes,
)
from search.engine.retrieval.member_facts import IndexMemberFacts, MemberFacts
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PairPlan,
    PlanYears,
    ReadPlan,
    minimal_constraint_svs,
    points_in_years,
)
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.role_scope import open_role_axis_members
from search.engine.retrieval.standard import standard_option_coverage
from search.engine.shape import ShapeDraft, is_role_bearing, role_names_by_axis
from search.engine.spec_build import RoleAxisFactInput
from search.engine.unit.conversion import (
    CountConversion,
    CountedConversion,
    counted_exact,
    merged_conversion,
)
from search.models import (
    NULL_REF_FACET_AXES,
    BaseIdentity,
    Entity,
    EntityOptions,
    FacetAxis,
    FacetOptions,
    GraphRef,
    IdentityShapeSchema,
    RoleAxis,
    Slot,
    SlotOption,
    SlotOptions,
    SlotValue,
    TimeWindow,
    ValueKind,
    Warning,
)
from search.observations import facet_value_label, unlisted_provenances
from series_coverage.coverage_types import BoundRoles, Role
from series_coverage.index import CoverageIndex
from series_coverage.schema import RoleNames
from statvar_members import covers_role_keys

logger = logging.getLogger(__name__)


def role_axis_facts_for_materialised(
    mat: Materialised, *, shape: ShapeDraft
) -> list[RoleAxisFactInput]:
    """Return one fact per open directional role axis, reusing `mat`'s resolved populations.

    An open origin uses `open_origin_dcids`. Discovered scope supplies its axis and members.
    When both axes are open, the companion population comes from `role_index_by_sv`.
    Unknown role-index data yields `member_dcids=None`. A known empty population is an empty
    set. Returns `[]` when no role axis is open. These facts add no SDMX reads and feed both
    the resolved spec and entity options.
    """
    if mat.open_origin_dcids is not None:
        return [RoleAxisFactInput(role="origin", member_dcids=mat.open_origin_dcids)]
    if mat.discovered_scope is None:
        return []
    facts: list[RoleAxisFactInput] = []
    discovered_axis = mat.discovered_scope.axis
    if discovered_axis in ("origin", "destination"):
        facts.append(
            RoleAxisFactInput(
                # Materialised.discovered_scope exists only for role-bearing resolves.
                # This check narrows its EntityRoleKey to the two accepted axes.
                role=discovered_axis,
                member_dcids=frozenset(mat.discovered_scope.member_dcids),
            )
        )
    companion_axis = mat.also_defaulted_axis
    if companion_axis is not None:
        companion_dcids: frozenset[str] | None = None
        if mat.role_index_by_sv is not None:
            companion_role_name = role_names_by_axis(shape).get(companion_axis)
            if companion_role_name is not None:
                companion_dcids = open_role_axis_members(
                    mat.role_index_by_sv, role_name=companion_role_name, svs=mat.sv_dcids
                )
        facts.append(RoleAxisFactInput(role=companion_axis, member_dcids=companion_dcids))
    return facts


def _entity_options_from_facts(role_axis_facts: Sequence[RoleAxisFactInput]) -> list[EntityOptions]:
    """Build one unlabeled entry per open axis without a graph call.

    Each entry has `entity_ref=None`. Known members use their dcids as labels. Unknown
    membership (`member_dcids=None`) yields empty options with `truncated=True`.
    """
    result: list[EntityOptions] = []
    for fact in role_axis_facts:
        if fact.member_dcids is None:
            result.append(
                EntityOptions(
                    entity_ref=None, role=fact.role, options=[], truncated=True, source="observed"
                )
            )
            continue
        ranked = sorted(fact.member_dcids)
        truncated = len(ranked) > settings().max_options
        options = [
            SlotOption(value=SlotValue(ref=GraphRef(ref=dcid, label=dcid), value_kind="entity"))
            for dcid in ranked[: settings().max_options]
        ]
        result.append(
            EntityOptions(
                entity_ref=None,
                role=fact.role,
                options=options,
                truncated=truncated,
                source="observed",
            )
        )
    return result


def _entity_options_count_only(
    role_axis_facts: Sequence[RoleAxisFactInput],
) -> list[EntityOptions]:
    """Return empty option lists and mark withheld or unknown membership as truncated.

    A known empty population is complete and has `truncated=False`. Axis facts and member
    counts stay available on `ResolvedSpec.role_axis_facts`.
    """
    return [
        EntityOptions(
            entity_ref=None,
            role=fact.role,
            options=[],
            truncated=not (fact.member_dcids is not None and len(fact.member_dcids) == 0),
            source="observed",
        )
        for fact in role_axis_facts
    ]


def _enrich_entity_options(
    entity_options: list[EntityOptions], *, graph: EngineGraphClient
) -> list[EntityOptions]:
    """Resolve member labels in one batch, using the dcid when a label lookup misses, and give
    each member the party kind the code index lists for it on its axis
    (`materialise.enrichment.read_party_kinds`).

    Membership and counts come from the materialised result, so only labels require a read.
    """
    all_dcids = sorted(
        {o.value.ref.ref for eo in entity_options for o in eo.options if o.value.ref is not None}
    )
    labels = graph.node_labels_batch(all_dcids) if all_dcids else {}
    kinds = read_party_kinds(
        (
            (eo.role, o.value.ref.ref)
            for eo in entity_options
            for o in eo.options
            if o.value.ref is not None
        ),
        parties=get_index().parties,
    )
    return [
        eo.model_copy(
            update={
                "options": [
                    SlotOption(
                        value=SlotValue(
                            ref=GraphRef(
                                ref=o.value.ref.ref,
                                label=labels.get(o.value.ref.ref, o.value.ref.ref),
                            ),
                            value_kind="entity",
                        ),
                        party_kind=kinds.get((eo.role, o.value.ref.ref)),
                    )
                    for o in eo.options
                    if o.value.ref is not None
                ]
            }
        )
        for eo in entity_options
    ]


class _FacetSource(NamedTuple):
    """A coverage facet's provenance, the field an elected dataset's test reads."""

    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class RoleServedFilter:
    """The served test of a role-bearing answer, read in memory from the coverage index.

    A series is a StatVar at one recipient and one donor, as an answer's download counts it
    (`CoverageIndex.served_extent`). It is served when the coverage index holds a row of it
    at the answer's bound roles with a point inside `years` in a facet its elected dataset
    publishes, the rows the role paths elect a facet from
    (`materialise.role_bearing_facets`). Its rows are the years those facets hold, ORed,
    and under `latest` its StatVar's latest year alone, from the series holding it, as the
    download counts them (`CoverageIndex.series_years`). `probe_entities` maps each
    declared role name to its bound entities (`coverage_probe_role_entities`); a role it
    omits stays open. No graph or SDMX request is made.

    A member's rows are exact when its series sit in one collapse group
    (`series_coverage.index.KeptFacets.groups`): a per-series read elects one group per
    member across every role tuple, so of several it can serve fewer. Under `latest`
    they are an upper bound when one of its facets states a period shorter than a year,
    whose latest period the index cannot find within the year. `bulk_read`, for an
    answer read in bulk (`read_plan.BulkPlan`), carries each member's facet ids, so a
    pick whose members sit in more than one facet counts an upper bound
    (`bound_values.served_rows`). `facets_pinned` marks every count an upper bound, for
    an answer whose identity binds a facet value (a unit, a provenance or a method), which
    this test does not apply. A unit request instead keeps of each series the years its
    `conversion` keeps (`unit.conversion.CountedConversion`), and its counts are exact by
    the same rule as the answer's download (`unit.conversion.counted_exact`).
    """

    probe_entities: Mapping[str, Sequence[str]]
    # The flow axes the answer leaves open (`place_roles.PlaceRoles.open_axes`), which its
    # picks elect under beside the roles `probe_entities` binds.
    open_axes: frozenset[RoleAxis]
    years: PlanYears
    coverage: CoverageIndex
    facets_pinned: bool = False
    bulk_read: bool = False
    conversion: CountConversion = None

    def served(self, member_facts: Mapping[str, MemberFacts]) -> Mapping[str, ServedSeriesRows]:
        """The series this filter serves of each member of `member_facts` it serves any of,
        with their rows."""
        window = self.years if isinstance(self.years, TimeWindow) else None
        latest = self.years == "latest"
        datasets = elected_datasets(member_facts)
        metadata = self.coverage.facet_metadata

        def admits(sv: str, facet_id: str) -> bool:
            found = metadata.get(facet_id)
            return publishes(datasets, sv, _FacetSource(found.provenance_id if found else None))

        by_roles: dict[RoleNames, list[str]] = {}
        for sv in member_facts:
            if (names := self.coverage.role_names.get(sv)) is not None:
                by_roles.setdefault(names, []).append(sv)
        conversion = self.conversion
        counted = conversion if isinstance(conversion, CountedConversion) else None
        served: dict[str, dict[ServedSeries, SeriesRows]] = {}
        for names, svs in by_roles.items():
            role_names: dict[Role, str] = {
                role: name
                for role, name in (("recipient", names.recipient), ("donor", names.donor))
                if name is not None
            }
            found = self.coverage.series_years(
                variables=svs,
                bound=self._bound(names),
                admits=admits,
                first_year=window.start_year if window is not None else None,
                last_year=window.end_year if window is not None else None,
                latest=latest,
                kept_years=(
                    coverage_kept_years(counted, role_names) if counted is not None else None
                ),
            )
            if found is None:
                continue
            for (sv, recipient, donor), years in zip(found.keys(), found.year_bits(), strict=True):
                kept = found.variables[sv]
                exact = kept.groups == 1 and not (latest and kept.sub_annual)
                served.setdefault(sv, {})[(sv, f"{recipient}|{donor}")] = SeriesRows(
                    rows=years.bit_count(),
                    exact=counted_exact(
                        exact and not self.facets_pinned,
                        conversion,
                        units=(
                            found_facet.unit if (found_facet := metadata.get(f)) else None
                            for f in kept.facet_ids
                        ),
                    ),
                    years=years,
                    bulk_facets=kept.facet_ids if self.bulk_read else frozenset(),
                )
        return served

    def _bound(self, names: RoleNames) -> BoundRoles:
        roles: tuple[tuple[Role, str | None], ...] = (
            ("donor", names.donor),
            ("recipient", names.recipient),
        )
        return {
            role: frozenset(dcids)
            for role, declared in roles
            if declared is not None and (dcids := self.probe_entities.get(declared))
        }


def role_served_filter(
    shape: ShapeDraft,
    probe_entities: Mapping[str, Sequence[str]],
    *,
    open_axes: frozenset[RoleAxis],
    years: PlanYears,
    coverage: CoverageIndex,
    facets_pinned: bool,
    plan: ReadPlan | None,
    conversion: CountConversion = None,
) -> RoleServedFilter | None:
    """The in-memory served test of an answer over `shape` read by `plan` under
    `conversion` (`RoleServedFilter`), or None for a subject-place family, whose served
    test needs graph reads."""
    if not is_role_bearing(shape) or "observationAbout" in probe_entities:
        return None
    return RoleServedFilter(
        probe_entities=probe_entities,
        open_axes=open_axes,
        years=years,
        coverage=coverage,
        facets_pinned=facets_pinned,
        bulk_read=isinstance(plan, BulkPlan),
        conversion=conversion,
    )


def _plan_series(
    plan: PairPlan, *, facets_pinned: bool, conversion: CountConversion
) -> Mapping[str, ServedSeriesRows]:
    """The series `plan` serves a point of inside its years, by StatVar, each with the rows
    the download counts for it (`download.series_rows`) in the years `conversion` keeps;
    an upper bound under `facets_pinned`, and exact otherwise by the download's rule
    (`unit.conversion.counted_exact`)."""
    counted = conversion if isinstance(conversion, CountedConversion) else None
    found: dict[str, dict[ServedSeries, SeriesRows]] = {}
    for s in plan.served:
        kept = counted.kept_years(dict(s.role_tuple)) if counted is not None else None
        rows, exact = series_rows(s, plan.years, kept)
        if rows:
            found.setdefault(s.stat_var, {})[s.key] = SeriesRows(
                rows=rows,
                exact=counted_exact(
                    exact and not facets_pinned, conversion, units=(f.unit for f in s.group)
                ),
            )
    return found


@dataclass(frozen=True)
class _PickPopulation:
    """The members a pick of some slot's options reads from, before antichain reduction,
    and their constraints."""

    members: Mapping[str, MemberFacts]
    constraints: Mapping[str, Mapping[str, str]]

    def picks(self, prop: str, values: Sequence[str]) -> dict[str, list[str]]:
        """For each of `values`, the members a pick of it on `prop` reads: those carrying it
        or a descendant, reduced to their antichain-minimal members as a mint reduces the
        family that pick resolves."""
        admitted = members_by_value({sv: c.get(prop) for sv, c in self.constraints.items()}, values)
        return {
            value: minimal_constraint_svs(svs, {sv: self.constraints[sv] for sv in svs})
            for value, svs in admitted.items()
        }


def _pick_population(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    served_filter: RoleServedFilter,
    *,
    index_only: bool,
) -> _PickPopulation | None:
    """The members an answer over `shape` under `bindings` reads from (`query_members`).
    None when `index_only` and neither the code index decides the family nor the catalogue
    holds it whole for the shape (`held_family_exact`): a graph-walked family holds the
    answer's own bindings alone, so it cannot stand for bindings that differ from them,
    where a family held whole holds every member whatever the bindings."""
    role_keys = entity_role_keys(served_filter.probe_entities, open_axes=served_filter.open_axes)
    query = shape_query(shape, bindings, role_keys.bound, open_role_keys=role_keys.open)
    members, index_sourced = query_members(shape, query)
    if index_only and not index_sourced and not held_family_exact(shape):
        return None
    return _PickPopulation(
        members=members, constraints={sv: m.read_constraints() for sv, m in members.items()}
    )


type _PickGroup = tuple[Literal["elected", "population"], str | None]
"""The members a pick is counted over (`option_axis_facts`): those of the dataset its
election serves, by provenance, or those of the population it is drawn from, by the bound
property it leaves open, None for the answer's own."""


class _IndexPick(NamedTuple):
    """The members a pick of one option reads where the code index decides the picks, and
    the dataset its election serves them from."""

    dataset: str | None
    members: dict[str, MemberFacts]
    """The antichain-minimal members of the family the pick elects, in pick order."""


def _index_option_picks(
    query: FamilyQuery, prop: str, values: Sequence[str]
) -> dict[str, _IndexPick] | None:
    """For each of `values`, the members a pick of it on `prop` reads when the code index
    decides the picks: the family the index elects for `query` with `prop` bound to the
    value (`code_index.members.option_families`), reduced to its antichain-minimal members,
    as a mint reduces the family that pick resolves. A pick another dataset serves reads
    that dataset's rows. None when the index does not decide the picks, or `query` names a
    dataset that does not publish its five-tuple."""
    try:
        families = option_families(get_index(), query, prop, values)
    except SourceNotInFamily:
        return None
    if families is None:
        return None
    picks: dict[str, _IndexPick] = {}
    for value, family in families.items():
        rows = {row.dcid: row for row in family.rows}
        constraints = {dcid: dict(row.constraint_properties) for dcid, row in rows.items()}
        picks[value] = _IndexPick(
            dataset=family.provenance,
            members={
                sv: IndexMemberFacts(row=rows[sv], dataset=family.provenance)
                for sv in minimal_constraint_svs(list(constraints), constraints)
            },
        )
    return picks


def _family_rows(
    shape: ShapeDraft, query: FamilyQuery, served_filter: RoleServedFilter
) -> ServedRows | None:
    """The rows of the family `query` resolves in the code index a mint serves and
    `served_filter` serves, or None when the index does not decide the family."""
    members, index_sourced = shape_member_facts(shape, query)
    if not index_sourced:
        return None
    minimal = minimal_constraint_svs(
        list(members), {sv: member.read_constraints() for sv, member in members.items()}
    )
    served = served_filter.served({sv: members[sv] for sv in minimal})
    return served_rows(_pick_series(served, minimal))


def option_axis_facts(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    *,
    menus: Mapping[str, Sequence[str]],
    plan: ReadPlan | None,
    member_facts: Mapping[str, MemberFacts],
    served_filter: RoleServedFilter | None,
    facets_pinned: bool,
    conversion: CountConversion = None,
) -> tuple[OptionServedFact, ...]:
    """What an answer's slot menus are counted from: for each what or how constraint slot
    of `shape` with a menu in `menus` (property to option value dcids), the series a pick
    of each option would serve, with their rows.

    A role-bearing family counts in memory (`served_filter`), each option over the members
    its pick reads: the family the code index elects with the option bound where the index
    decides the picks (`_index_option_picks`), so a pick another dataset serves counts that
    dataset's rows, else those of the pick's population, reduced as that pick's family is
    (`_PickPopulation.picks`). Each pick counts the facets of the dataset its own election
    serves, though another pick elects another dataset for a row both publish. An
    unbound slot's pick reads the answer's own family, whether the code index or the mint's
    graph walk decides it, so an answer serving its family total still counts the parts a
    pick would serve. A bound slot's pick reads the family with that slot left unbound and
    every other binding kept, which only the code index resolves without a graph walk; a
    graph-walked family's bound slots get no fact. Under an answer that leaves flow axes
    open, a pick whose members do not declare every open axis counts no series
    (`_serving_open_keys`). A menu counted at zero for every option while members carry
    some of them, under an answer that serves data, means the coverage index and the
    members disagree (a facet without metadata, a population that differs from the one
    materialised) when the slot is unbound or an option overlaps its bound value, so it
    gets no fact and the coverage probe counts it (`_drifted`). A bound root's menu of the
    other roots alone overlaps nothing and may count zero throughout.

    A subject-place family's served test needs graph reads, so an unbound slot reads the
    series `plan` serves inside its years (`_plan_series`), unless none of their StatVars
    declares it: an answer serving its family total leaves the slot to the coverage probe
    (`_maybe_enrich_slot_options`), as it does every bound slot. A family the code index
    decides with more than one eligible dataset gets no fact, since a pick may elect a
    dataset the plan does not read (`_options_may_elect_elsewhere`); the probe counts its
    menus, each option over the dataset its election serves. A subject-place plan is
    always per-series: only a role-bearing read plans in bulk (`read_plan.plan_role_read`),
    and every role-bearing answer has a served filter (`role_served_filter`). Without a
    plan no slot is counted. `facets_pinned` marks the plan's counts upper bounds, as
    `RoleServedFilter.facets_pinned` does the filter's, and the plan's series keep the
    years `conversion` keeps, as `RoleServedFilter.conversion` does the filter's.
    """
    if plan is None:
        return ()
    bound = bound_values_from_bindings(bindings)
    slot_properties = dict.fromkeys(
        prop
        for slot in shape.slot_keys
        if (prop := slot.property_dcid) is not None
        and prop in menus
        and slot.axis in ("what", "how")
    )
    if served_filter is None:
        if not isinstance(plan, PairPlan) or _options_may_elect_elsewhere(shape, bindings):
            return ()
        return _plan_facts(
            [prop for prop in slot_properties if prop not in bound],
            menus,
            plan=plan,
            member_facts=member_facts,
            facets_pinned=facets_pinned,
            conversion=conversion,
        )
    role_keys = entity_role_keys(served_filter.probe_entities, open_axes=served_filter.open_axes)
    # Each pick is counted over the members of the group it reads from: the dataset its
    # election serves, or the population it is drawn from. A row two datasets publish thus
    # counts the facets of each pick's own dataset. The picks of one group overlap, so the
    # coverage index is read once per group.
    groups: dict[_PickGroup, dict[str, MemberFacts]] = {}
    prop_picks: dict[str, dict[str, tuple[_PickGroup, Sequence[str]]]] = {}
    answer_population: _PickPopulation | None = None
    for prop in slot_properties:
        pick_bindings = without_values(bindings, {prop}) if prop in bound else bindings
        index_picks = _index_option_picks(
            shape_query(shape, pick_bindings, role_keys.bound, open_role_keys=role_keys.open),
            prop,
            menus[prop],
        )
        if index_picks is not None:
            prop_picks[prop] = {}
            for value, pick in index_picks.items():
                group = ("elected", pick.dataset)
                groups.setdefault(group, {}).update(pick.members)
                prop_picks[prop][value] = (
                    group,
                    _serving_open_keys(pick.members, list(pick.members), role_keys.open),
                )
            continue
        if prop in bound:
            population = _pick_population(shape, pick_bindings, served_filter, index_only=True)
            group = ("population", prop)
        else:
            if answer_population is None:
                answer_population = _pick_population(
                    shape, bindings, served_filter, index_only=False
                )
            population = answer_population
            group = ("population", None)
        if population is not None:
            groups[group] = dict(population.members)
            prop_picks[prop] = {
                value: (group, _serving_open_keys(population.members, svs, role_keys.open))
                for value, svs in population.picks(prop, menus[prop]).items()
            }
    served = {group: served_filter.served(members) for group, members in groups.items()}
    facts: list[OptionServedFact] = []
    for prop, grouped in prop_picks.items():
        picks = {value: svs for value, (_group, svs) in grouped.items()}
        fact = OptionServedFact(
            property=prop,
            served={
                value: _pick_series(served[group], svs) for value, (group, svs) in grouped.items()
            },
        )
        if not _drifted(fact, picks, current=bound.get(prop), shape_id=shape.shape_id):
            facts.append(fact)
    return tuple(facts)


def _options_may_elect_elsewhere(shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]) -> bool:
    """Whether a pick of an option of an answer over `shape` under `bindings` may elect a
    dataset other than the answer's: the code index decides the family and more than one
    dataset is eligible for it (`code_index.members.eligible_datasets`). The answer's plan
    holds only its own dataset's series, so it cannot count such an option."""
    query = shape_query(shape, bindings, ())
    index = get_index()
    return index_decides(index, query) and len(eligible_datasets(index, query)) > 1


def _pick_series(
    served: Mapping[str, ServedSeriesRows], members: Sequence[str]
) -> ServedSeriesRows:
    """The series a pick reading `members` serves, of those `served` holds by member."""
    return {key: rows for sv in members for key, rows in served.get(sv, {}).items()}


def _serving_open_keys(
    facts: Mapping[str, MemberFacts], members: Sequence[str], open_keys: frozenset[str]
) -> list[str]:
    """The members a pick reading `members` counts under an answer leaving `open_keys`
    open: all of them when each declares a role on every open key (`covers_role_keys`),
    else none. A pick whose elected members leave an open axis undeclared serves the
    total, and its resubmit, which inherits the answer's `open_roles`, refuses
    `open_role_unavailable`, so its option counts no series."""
    if not open_keys:
        return list(members)
    if all(covers_role_keys(facts[sv].read_declared_roles(), open_keys) for sv in members):
        return list(members)
    return []


def _drifted(
    fact: OptionServedFact,
    picks: Mapping[str, Sequence[str]],
    *,
    current: Sequence[str] | None,
    shape_id: str,
) -> bool:
    """Whether `fact` counts no series for any option of its menu while members carry some
    of them (`picks`), logged as a warning. A definite answer serves its bound slot's
    `current` values or, for an unbound slot, the family its options partition. `current`
    holds the bound values as `CodeIndex.expansion` widens them
    (`bound_values_from_bindings`): with their descendants in a system declared
    `disjoint_from_children`, and alone in one whose parent is the total of its children
    (`dac1AidType`, `dac2aAidType`). An empty count is the coverage index disagreeing with
    the members, which would hide the dimension from the reader, when the slot is unbound
    or an option overlaps the current values: it lies under one of them, as the current
    value itself or a child does, or one lies under it, as under the parent. A bound root's
    menu of the other roots alone may count
    zero throughout, as may a menu none of whose options any member carries; each serves
    nothing and is dropped."""
    if any(fact.served.values()) or not any(picks.values()):
        return False
    if current is not None:
        index = get_index()
        bound = set(current)
        below = {d for value in current for d in index.descendants(value)}
        if not any(
            option in below or bound.intersection(index.descendants(option)) for option in picks
        ):
            return False
    logger.warning(
        "option counting found no served series for any option of %r on shape_id=%r; "
        "leaving the menu to the coverage probe",
        fact.property,
        shape_id,
    )
    return True


def _plan_facts(
    properties: Sequence[str],
    menus: Mapping[str, Sequence[str]],
    *,
    plan: PairPlan,
    member_facts: Mapping[str, MemberFacts],
    facets_pinned: bool,
    conversion: CountConversion,
) -> tuple[OptionServedFact, ...]:
    """Each of `properties` counted from the series `plan` serves whose StatVars carry each
    option (`option_axis_facts`), skipping a property none of them declares."""
    served = _plan_series(plan, facets_pinned=facets_pinned, conversion=conversion)
    constraints = {sv: m.read_constraints() for sv, m in member_facts.items() if sv in served}
    facts: list[OptionServedFact] = []
    for prop in properties:
        values = {sv: c.get(prop) for sv, c in constraints.items()}
        # Served StatVars none of which declares the slot are its family total, and an
        # option binds a part the answer does not read: they cannot count it.
        if all(value is None for value in values.values()):
            continue
        admitted = members_by_value(values, menus[prop])
        facts.append(
            OptionServedFact(
                property=prop,
                served={value: _pick_series(served, svs) for value, svs in admitted.items()},
            )
        )
    return tuple(facts)


def _current_values(slots: Sequence[Slot]) -> dict[str, frozenset[str]]:
    """Each bound slot's property and the value dcids it is bound to."""
    current: dict[str, frozenset[str]] = {}
    for slot in slots:
        if slot.key.property is None:
            continue
        binding = slot.binding
        if binding.kind == "value":
            refs = [binding.value.ref]
        elif binding.kind == "set":
            refs = [v.ref for v in binding.values]
        else:
            continue
        current[slot.key.property.ref] = frozenset(r.ref for r in refs if r is not None)
    return current


def count_slot_options(
    slot_options: Sequence[SlotOptions],
    facts: Sequence[OptionServedFact],
    slots: Sequence[Slot],
) -> list[SlotOptions]:
    """Count each slot menu `facts` covers and drop the options it would serve nothing for.

    Every producer of slot menus passes its menus through here before the Spec is built, and
    the coverage probe and trailing `slot_options` events start from the result, so an
    option dropped here reaches none of them. An option's `served_rows` and
    `served_rows_exact` are the rows its slot's fact says a pick of it would serve
    (`OptionServedFact.rows`). The current value of a bound slot in `slots` stays whatever
    its count. A menu no fact covers is returned as given, uncounted; a menu left with no
    option is dropped.
    """
    by_property = {fact.property: fact for fact in facts}
    current = _current_values(slots)
    counted: list[SlotOptions] = []
    for so in slot_options:
        fact = by_property.get(so.key.property.ref) if so.key.property is not None else None
        if fact is None:
            counted.append(so)
            continue
        kept = current.get(fact.property, frozenset())
        options: list[SlotOption] = []
        for o in so.options:
            if o.value.ref is None:
                options.append(o)
                continue
            found = fact.rows(o.value.ref.ref)
            if found.rows or o.value.ref.ref in kept:
                options.append(
                    o.model_copy(
                        update={"served_rows": found.rows, "served_rows_exact": found.exact}
                    )
                )
        if options:
            counted.append(so.model_copy(update={"options": options}))
    return counted


class CountedSlotOptions(NamedTuple):
    """An answer's counted slot menus and the facts they were counted from, which a
    same-shape merge unions to count the merged menus (`RegionResult.option_axis_facts`)."""

    options: list[SlotOptions]
    facts: tuple[OptionServedFact, ...]


def counted_slot_options(
    slot_options: Sequence[SlotOptions],
    *,
    slots: Sequence[Slot],
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    plan: ReadPlan | None,
    member_facts: Mapping[str, MemberFacts],
    served_filter: RoleServedFilter | None,
    facets_pinned: bool,
    conversion: CountConversion = None,
) -> CountedSlotOptions:
    """An answer's grounded slot menus counted from its own facts (`option_axis_facts`,
    `count_slot_options`). `facets_pinned` is whether the answer's identity binds a facet
    value, which makes every count an upper bound, and `conversion` how its unit request
    changes the counts (`unit.conversion.counted_conversion`)."""
    facts = option_axis_facts(
        shape,
        bindings,
        menus={
            so.key.property.ref: [o.value.ref.ref for o in so.options if o.value.ref is not None]
            for so in slot_options
            if so.key.property is not None
        },
        plan=plan,
        member_facts=member_facts,
        served_filter=served_filter,
        facets_pinned=facets_pinned,
        conversion=conversion,
    )
    return CountedSlotOptions(count_slot_options(slot_options, facts, slots), facts)


def measure_served_counter(
    served_filter: RoleServedFilter | None,
) -> Callable[[BaseIdentity], ServedRows | None] | None:
    """The rows a resubmit of a sibling measure's identity would serve, counted as a bound
    slot's options are (`_family_rows`), for `spec_build`'s measure alternatives. None for
    a subject-place family. The counter returns None for an identity whose family the
    index does not decide."""
    if served_filter is None:
        return None

    def count(identity: BaseIdentity) -> ServedRows | None:
        if not isinstance(identity.shape, IdentityShapeSchema):
            return None
        # The index's family alone is counted, so the shape holds no members of its own.
        shape = replace(synthetic_shape_for_identity(identity.shape), sv_arc_facts_complete=False)
        return _family_rows(shape, identity_query(identity), served_filter)

    return count


def merged_served_filter(filters: Sequence[RoleServedFilter | None]) -> RoleServedFilter | None:
    """The served test of a same-shape merge of answers tested by `filters`: each role's
    entities unioned in first-seen order, as the merged identity binds them. The merged
    test pins facets and reads in bulk when any answer's does, and keeps the years their
    conversions keep (`unit.conversion.merged_conversion`). None when an answer has no
    test or the answers read different years, coverage or open axes."""
    if not filters or any(f is None for f in filters):
        return None
    tests = [f for f in filters if f is not None]
    first = tests[0]
    if any(
        t.years != first.years or t.coverage is not first.coverage or t.open_axes != first.open_axes
        for t in tests
    ):
        return None
    entities: dict[str, list[str]] = {}
    for test in tests:
        for role, dcids in test.probe_entities.items():
            entities.setdefault(role, [])
            entities[role].extend(d for d in dcids if d not in entities[role])
    return RoleServedFilter(
        probe_entities=entities,
        open_axes=first.open_axes,
        years=first.years,
        coverage=first.coverage,
        facets_pinned=any(t.facets_pinned for t in tests),
        bulk_read=any(t.bulk_read for t in tests),
        conversion=merged_conversion([t.conversion for t in tests]),
    )


def _uncounted(so: SlotOptions) -> bool:
    """Whether no option of `so` carries served rows, so the coverage probe counts it."""
    return all(o.served_rows is None for o in so.options)


def _pick_members(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    menus: Sequence[SlotOptions],
    role_keys: RoleKeys,
) -> dict[str, dict[str, list[str] | None]]:
    """For each menu's property and each of its options, the members a pick of that option
    reads. Where the code index decides the picks, each reads the family the index elects
    with the option bound (`_index_option_picks`), so a pick another dataset serves reads
    that dataset's rows. Otherwise each reads, as `_PickPopulation.picks` selects them, the
    members of the pick's population carrying the option or a descendant, reduced to their
    antichain-minimal members. An unbound slot's pick reads the answer's own population, a
    bound slot's the population with that slot left unbound (`query_members`).

    An option no member carries maps to an empty list when the population is exact, so the
    pick serves nothing, and to None when it is not, so the option stays uncounted. A
    population is exact when the code index decides it, when the catalogue's tables hold its
    family whole (`held_family_exact`), whatever the bindings, or for an unbound slot when the
    shape holds its family exactly (`ShapeDraft.sv_arc_facts_complete`); a walked family
    holds the answer's own bindings alone, so it cannot rule out an option of a bound slot.
    A pick whose members do not declare every open key of `role_keys` reads none of them
    (`_serving_open_keys`).
    """
    bound = bound_values_from_bindings(bindings)
    picks: dict[str, dict[str, list[str] | None]] = {}
    for so in menus:
        if so.key.property is None:
            continue
        prop = so.key.property.ref
        is_bound = prop in bound
        query = shape_query(
            shape,
            without_values(bindings, {prop}) if is_bound else bindings,
            role_keys.bound,
            open_role_keys=role_keys.open,
        )
        values = [o.value.ref.ref for o in so.options if o.value.ref is not None]
        index_picks = _index_option_picks(query, prop, values)
        if index_picks is not None:
            picks[prop] = {
                value: _serving_open_keys(pick.members, list(pick.members), role_keys.open)
                for value, pick in index_picks.items()
            }
            continue
        members, index_sourced = query_members(shape, query)
        exact = (
            index_sourced
            or held_family_exact(shape)
            or (not is_bound and shape.sv_arc_facts_complete)
        )
        population = _PickPopulation(
            members=members, constraints={sv: m.read_constraints() for sv, m in members.items()}
        )
        picks[prop] = {
            value: counted
            if (counted := _serving_open_keys(members, svs, role_keys.open)) or exact
            else None
            for value, svs in population.picks(prop, values).items()
        }
    return picks


def _counted_from_probe(so: SlotOptions, kept: frozenset[str]) -> SlotOptions:
    """`so` after the coverage probe: each option the probe counted dropped when it serves
    nothing, unless it is the slot's current value (`kept`). An option the probe could not
    count keeps no count."""
    return so.model_copy(
        update={
            "options": [
                o
                for o in so.options
                if o.served_rows != 0 or (o.value.ref is not None and o.value.ref.ref in kept)
            ]
        }
    )


def _maybe_enrich_slot_options(
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    slot_options: list[SlotOptions],
    *,
    entities: Mapping[str, list[str]],
    open_axes: frozenset[RoleAxis],
    date_request: DateRequest | None,
    materialise_graph: GraphReader | None,
    place_dcids: list[str] | None = None,
    include_option_coverage: bool,
    facets_pinned: bool,
) -> tuple[list[SlotOptions], Warning | None]:
    """Count the menus `count_slot_options` left uncounted from the coverage probe, when
    enabled. Graph failures preserve the original options.

    `slot_options` are counted menus (`count_slot_options`). A menu with a served count is
    returned as given: its count is exact and the probe, which conditions on the place
    alone, would only weaken it. Each uncounted menu is probed (`standard_option_coverage`)
    over the members a pick of each option reads (`_pick_members`), each option counted
    from the series that pick serves, and an option serving nothing dropped unless it is
    the slot's current value (`_counted_from_probe`); a menu left empty is dropped.
    Disabled coverage, a missing `materialise_graph`, or nothing to probe returns inputs
    unchanged. A `GraphInfraError` also returns the original options with an
    `OPTION_COVERAGE_PROBE_FAILED` warning, distinguishing failure from an empty result.
    A member counts only the facets of the dataset elected once its option is bound to the
    family query `bindings`, the role keys of `entities` and `open_axes` state, else of the
    family they elect (`standard_option_coverage`). `facets_pinned`, whether the answer's identity binds
    a facet, makes every probed count an upper bound.
    """
    pending = [i for i, so in enumerate(slot_options) if _uncounted(so)]
    if not include_option_coverage or not pending:
        return slot_options, None
    try:
        role_keys = entity_role_keys(entities, open_axes=open_axes)
        query = shape_query(shape_draft, bindings, role_keys.bound, open_role_keys=role_keys.open)
        member_facts, _index_sourced = shape_member_facts(shape_draft, query)
        menus = [slot_options[i] for i in pending]
        probed = standard_option_coverage(
            shape=shape_draft,
            slot_options=menus,
            entities=entities,
            materialise_graph=materialise_graph,
            date_request=date_request,
            place_dcids=place_dcids,
            datasets=elected_datasets(member_facts),
            pick_members=_pick_members(shape_draft, bindings, menus, role_keys),
            facets_pinned=facets_pinned,
            query=query,
        )
    except GraphInfraError:
        logger.warning(
            "option_coverage probe failed for shape_id=%r; returning uncovered slot_options",
            shape_draft.shape_id,
            exc_info=True,
        )
        return slot_options, Warning(
            code=OPTION_COVERAGE_PROBE_FAILED,
            severity="warn",
            message=(
                f"option coverage probe failed for shape {shape_draft.shape_id!r}; "
                "slot options returned without coverage."
            ),
        )
    current = {
        b.property_dcid: frozenset(b.value_dcids)
        for b in bindings
        if b.property_dcid and b.kind in ("value", "set")
    }
    enriched = list(slot_options)
    for i, so in zip(pending, probed, strict=True):
        prop = so.key.property.ref if so.key.property is not None else None
        enriched[i] = _counted_from_probe(so, current.get(prop, frozenset()))
    return [so for so in enriched if so.options], None


def _build_and_enrich_entity_options(
    shape_draft: ShapeDraft,
    role_entities: Mapping[str, list[str]],
    entities: list[Entity],
    sv_dcids: list[str],
    *,
    date_request: DateRequest | None,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    include_option_coverage: bool,
    role_axis_facts: Sequence[RoleAxisFactInput] = (),
    place_dcids: list[str] | None = None,
) -> tuple[list[EntityOptions], list[EntityOptions], Warning | None]:
    """Build open-role options and return raw options, response options, and an optional warning.

    Raw options retain known dcids with dcid labels for streamed probes. Coverage enables real
    labels in response options. Disabled coverage returns empty response lists, while role facts
    and counts are available on `ResolvedSpec`. A label lookup failure returns raw options with
    `OPTION_COVERAGE_PROBE_FAILED`. Empty facts return two empty lists and no warning.
    The shared call signature includes `role_entities`, `entities`, and `sv_dcids`. Confirmed
    populations come from `role_axis_facts`.
    """
    raw = _entity_options_from_facts(role_axis_facts)
    if not raw:
        return [], [], None
    if not include_option_coverage:
        return raw, _entity_options_count_only(role_axis_facts), None
    try:
        return raw, _enrich_entity_options(raw, graph=graph), None
    except GraphInfraError:
        logger.warning(
            "entity_options label probe failed for shape_id=%r; returning unlabelled entity_options",
            shape_draft.shape_id,
            exc_info=True,
        )
        return (
            raw,
            raw,
            Warning(
                code=OPTION_COVERAGE_PROBE_FAILED,
                severity="warn",
                message=(
                    f"entity options label probe failed for shape {shape_draft.shape_id!r}; "
                    "entity options returned with dcid-only labels."
                ),
            ),
        )


def _stream_slot_options_events(
    spec_id: str,
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    slot_options: list[SlotOptions],
    *,
    entities: Mapping[str, list[str]],
    open_axes: frozenset[RoleAxis],
    date_request: DateRequest | None,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    place_dcids: list[str] | None = None,
    entity_options: list[EntityOptions] | None = None,
    facet_probe: FacetProbe | None = None,
    coverage: CoverageIndex,
    deadline: float | None,
    facets_pinned: bool,
    variable_labels: Mapping[str, str],
) -> _OptionProbeResult:
    """Probe options and return trailing events and warnings after the response event.

    A `GraphInfraError` preserves slot/entity options and adds a warning. Facet failures return
    an empty list with a warning. Other errors propagate. Empty slot entries and facet axes are
    omitted, while a nonempty slot whose candidates all have `coverage=None` emits an event. Every
    supplied entity entry emits, including one with unknown membership. Facets follow slot
    probes to reuse graph caches. Places the UN front door refused during the probes carry an
    `un_system_unavailable` warning, since the answer's own warnings are built before this runs.
    Its variables are named from `variable_labels`, the names of the Spec's members.
    """

    with recording_un_blocked_places() as un_blocked:
        enriched, warn = _maybe_enrich_slot_options(
            shape_draft,
            bindings,
            slot_options,
            entities=entities,
            open_axes=open_axes,
            date_request=date_request,
            materialise_graph=materialise_graph,
            place_dcids=place_dcids,
            include_option_coverage=True,
            facets_pinned=facets_pinned,
        )
        enriched_entities: list[EntityOptions] = []
        entity_warn: Warning | None = None
        if entity_options:
            try:
                enriched_entities = _enrich_entity_options(entity_options, graph=graph)
            except GraphInfraError:
                logger.warning(
                    "entity_options label probe failed for spec_id=%r; returning unlabelled "
                    "entity_options",
                    spec_id,
                    exc_info=True,
                )
                enriched_entities = entity_options
                entity_warn = Warning(
                    code=OPTION_COVERAGE_PROBE_FAILED,
                    severity="warn",
                    message=(
                        f"entity options label probe failed for spec {spec_id!r}; entity "
                        "options returned with dcid-only labels."
                    ),
                )

        facet_options: list[FacetOptions] = []
        facet_warn: Warning | None = None
        if facet_probe is not None:
            facet_options, facet_warn = _maybe_build_facet_options(
                facet_probe,
                graph=graph,
                coverage=coverage,
                deadline=deadline,
            )

    events: list[_OptionEvent] = [
        SlotOptionsEvent(spec_id=spec_id, key=so.key, options=so.options, truncated=so.truncated)
        for so in enriched
        if so.options
    ]
    warns = [warn.model_copy(update={"spec_id": spec_id})] if warn is not None else []

    events.extend(
        EntityOptionsEvent(
            spec_id=spec_id,
            entity_ref=eo.entity_ref,
            role=eo.role,
            options=eo.options,
            truncated=eo.truncated,
        )
        for eo in enriched_entities
    )
    if entity_warn is not None:
        warns.append(entity_warn.model_copy(update={"spec_id": spec_id}))

    events.extend(
        FacetOptionsEvent(spec_id=spec_id, axis=fo.axis, options=fo.options, truncated=fo.truncated)
        for fo in facet_options
        if fo.options
    )
    if facet_warn is not None:
        warns.append(facet_warn.model_copy(update={"spec_id": spec_id}))
    refused = un_unavailable_warning(un_blocked, labels=variable_labels)
    if refused is not None:
        warns.append(refused.model_copy(update={"spec_id": spec_id}))

    return tuple(events), tuple(warns)


def _launch_slot_options_task(
    spec_id: str,
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    slot_options: list[SlotOptions],
    *,
    entities: Mapping[str, list[str]],
    open_axes: frozenset[RoleAxis],
    date_request: DateRequest | None,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    place_dcids: list[str] | None = None,
    entity_options: list[EntityOptions] | None = None,
    facet_probe: FacetProbe | None = None,
    coverage: CoverageIndex,
    deadline: float | None,
    facets_pinned: bool,
    variable_labels: Mapping[str, str],
) -> asyncio.Task[_OptionProbeResult]:
    """Run the probe in a worker thread and return its task without awaiting it.

    `create_owned_task` assigns abandoned-task cleanup to the request scope. `resolve_variable`
    returns while the probe runs.
    """
    return create_owned_task(
        asyncio.to_thread(
            _stream_slot_options_events,
            spec_id,
            shape_draft,
            bindings,
            slot_options,
            entities=entities,
            open_axes=open_axes,
            date_request=date_request,
            graph=graph,
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            place_dcids=place_dcids,
            entity_options=entity_options,
            facet_probe=facet_probe,
            coverage=coverage,
            deadline=deadline,
            facets_pinned=facets_pinned,
            variable_labels=variable_labels,
        )
    )


class _ObservedFacets(NamedTuple):
    """`by_sv`: each StatVar's candidate facets, the ones a pin's refusal is judged against.
    `windowed`: every candidate with a point inside the window, its StatVar and its in-window
    points."""

    by_sv: dict[str, list[Facet]]
    windowed: list[tuple[str, Facet, int]]


def _observed_facets(
    sv_dcids: Sequence[str],
    probe_entities: Mapping[str, Sequence[str]],
    *,
    years: PlanYears,
    graph: EngineGraphClient,
    coverage: CoverageIndex,
    datasets: ElectedDatasets = NO_DATASETS,
) -> _ObservedFacets:
    """Read each StatVar's candidate facets from the source its pin refusal reads, a member
    of an index-resolved family keeping only those its elected dataset publishes
    (`datasets`), the facets a mint of the Spec can serve.

    Subject entities (`observationAbout`) read cached graph facets per entity, the facets
    `facet_refusals.facet_refusal` judges unwindowed. A facet is in the window when it holds
    a point under `years`, the answer's read plan years, and counts the points every plan
    election weighs it by (`read_plan.points_in_years`).
    Role-bearing StatVars read `materialise.role_bearing_facets` over the coverage index
    instead: the bound roles' rows within `years`, with a role the map omits left open, the
    rows the role paths' refusals judge. Materialisation selects from the same
    candidates. No graph or SDMX request is made on that path, and a StatVar the index does
    not cover is left out. With no range, or `latest`, every observation counts. The one
    period a `latest` read serves each StatVar (`materialise.window.latest_period`) does not
    narrow a facet's weight.
    """
    year_window = years if isinstance(years, TimeWindow) else None
    subject = list(probe_entities.get("observationAbout") or [])
    by_sv: dict[str, list[Facet]] = {}
    windowed: list[tuple[str, Facet, int]] = []
    if subject:
        facets_batch = graph.observation_facets_batch(list(sv_dcids), subject, needs_dates=True)
        for sv in sv_dcids:
            by_sv[sv] = published(
                datasets,
                sv,
                (f for entity in subject for f in facets_batch.get(sv, {}).get(entity, [])),
            )
            windowed.extend(
                (sv, facet, n) for facet in by_sv[sv] if (n := points_in_years(facet, years))
            )
        return _ObservedFacets(by_sv, windowed)
    for sv in sv_dcids:
        facets = role_bearing_facet_candidates(
            stat_var=sv,
            role_bindings=probe_entities,
            coverage=coverage,
            first_year=year_window.start_year if year_window is not None else None,
            last_year=year_window.end_year if year_window is not None else None,
        )
        if facets is None:
            continue
        by_sv[sv] = published(datasets, sv, facets)
        windowed.extend((sv, facet, facet.obs_count) for facet in by_sv[sv])
    return _ObservedFacets(by_sv, windowed)


def _facet_options(
    sv_dcids: Sequence[str],
    probe_entities: Mapping[str, Sequence[str]],
    *,
    years: PlanYears,
    graph: EngineGraphClient,
    coverage: CoverageIndex,
    pins: FacetPins = NO_PINS,
    datasets: ElectedDatasets = NO_DATASETS,
    deadline: float | None,
) -> list[FacetOptions]:
    """List the facet values a pin serves alongside the spec's other pins.

    `probe_entities` maps each bound role to its entities (`_observed_facets` reads it). A
    value is offered when pinning it together with `pins`' bindings on the other axes is
    not refused, `facet_refusals.pin_admitted` over every StatVar's candidates, and a facet
    carrying it that honours those other pins has an observation inside `years`: binding a
    value with no point in the window returns `no_data` with reason `no_observations`. A
    value only some StatVars carry is withheld, since its pin is refused. The value
    replaces any pin its own axis holds, so a pinned axis lists its alternatives. Values come from `materialise.facet_axis_value`, the rule a pinned ref
    matches. On a `NULL_REF_FACET_AXES` axis, facets that declare no value form one more
    option, with no `ref`, which a null identity ref pins. It is listed only beside a
    declared value, so an axis no facet declares stays omitted. Rank by summed in-window
    observation points, descending, then dcid, ascending, with the ref-less option last
    among equals. Results cap at `settings().max_options` and mark truncation.
    Omit axes without values. Cached facet reads avoid duplicate graph calls. Resolve labels
    in one batch and use the dcid for unnamed nodes. The origin catalogue labels the
    provenances it lists, which join no batch. Source is always `observed`.
    """
    if not sv_dcids:
        return []
    observed = _observed_facets(
        sv_dcids, probe_entities, years=years, graph=graph, coverage=coverage, datasets=datasets
    )
    axis_counts: dict[str, dict[str | None, int]] = {axis: {} for axis in get_args(FacetAxis)}
    for sv, facet, n_in_window in observed.windowed:
        for axis in get_args(FacetAxis):
            dcid = facet_axis_value(facet, axis)
            if (dcid is not None or axis in NULL_REF_FACET_AXES) and pins_admit_facet(
                graph, facet, pins, except_axis=axis, sv=sv, deadline=deadline
            ):
                axis_counts[axis][dcid] = axis_counts[axis].get(dcid, 0) + n_in_window
    for axis, counts in axis_counts.items():
        refused = [
            d
            for d in counts
            if not pin_admitted(graph, axis, d, observed.by_sv, pins, deadline=deadline)
        ]
        for dcid in refused:
            del counts[dcid]
        if list(counts) == [None]:
            counts.clear()

    provenance_dcids = [d for d in axis_counts["provenance"] if d is not None]
    all_dcids = list(
        dict.fromkeys(
            [
                *(
                    d
                    for axis, counts in axis_counts.items()
                    if axis != "provenance"
                    for d in counts
                    if d is not None
                ),
                *unlisted_provenances(provenance_dcids),
            ]
        )
    )
    labels = graph.node_labels_batch(all_dcids) if all_dcids else {}
    labels.update(listed_provenance_labels(provenance_dcids))

    result: list[FacetOptions] = []
    for axis in get_args(FacetAxis):
        counts = axis_counts[axis]
        if not counts:
            continue
        ranked = sorted(
            counts, key=lambda dcid, counts=counts: (-counts[dcid], dcid is None, dcid or "")
        )
        truncated = len(ranked) > settings().max_options
        value_kind: ValueKind = "source" if axis == "provenance" else "enum_value"
        options = [
            SlotOption(
                value=SlotValue(
                    ref=None
                    if dcid is None
                    else GraphRef(ref=dcid, label=facet_value_label(axis, labels.get(dcid, dcid))),
                    value_kind=value_kind,
                )
            )
            for dcid in ranked[: settings().max_options]
        ]
        result.append(
            FacetOptions(axis=axis, options=options, truncated=truncated, source="observed")
        )
    return result


def _maybe_build_facet_options(
    probe: FacetProbe,
    *,
    graph: EngineGraphClient,
    coverage: CoverageIndex,
    deadline: float | None,
) -> tuple[list[FacetOptions], Warning | None]:
    """Build facet options over `probe`. A graph failure returns `[]` with a warning.

    Facet options come only from observed values, so a successful empty probe returns `[]`.
    """
    try:
        return (
            _facet_options(
                probe.sv_dcids,
                probe.probe_entities,
                years=probe.years,
                graph=graph,
                coverage=coverage,
                pins=probe.pins,
                datasets=probe.datasets,
                deadline=deadline,
            ),
            None,
        )
    except GraphInfraError:
        logger.warning(
            "facet_options build failed for sv_dcids=%r entities=%r; returning no facet_options",
            probe.sv_dcids,
            probe.probe_entities,
            exc_info=True,
        )
        return [], Warning(
            code=OPTION_COVERAGE_PROBE_FAILED,
            severity="warn",
            message="facet options build failed; facet_options was not populated.",
        )


# Probe inputs follow _stream_slot_options_events' positional fields, then its
# `facets_pinned` and `variable_labels`. A missing facet probe skips facets.
# `entity_options` carries the raw list for streamed events.
_SlotOptionsProbe = tuple[
    str,
    ShapeDraft,
    list[SlotBindingDraft],
    list[SlotOptions],
    Mapping[str, list[str]],
    frozenset[RoleAxis],
    "list[str] | None",
    "list[EntityOptions] | None",
    "FacetProbe | None",
    bool,
    Mapping[str, str],
]


async def _gather_slot_options_events(
    probes: list[_SlotOptionsProbe],
    *,
    date_request: DateRequest | None,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    coverage: CoverageIndex,
    deadline: float | None,
) -> _OptionProbeResult:
    """Run candidate probes concurrently in worker threads and combine their events and warnings.

    The candidate path returns one `RegionResult` with a single deferred task for all specs.
    """
    groups = await asyncio.gather(
        *[
            asyncio.to_thread(
                _stream_slot_options_events,
                spec_id,
                shape,
                bindings,
                raw_options,
                entities=entities,
                open_axes=open_axes,
                date_request=date_request,
                graph=graph,
                sdmx=sdmx,
                materialise_graph=materialise_graph,
                place_dcids=place_dcids,
                entity_options=entity_options,
                facet_probe=facet_probe,
                coverage=coverage,
                deadline=deadline,
                facets_pinned=facets_pinned,
                variable_labels=variable_labels,
            )
            for (
                spec_id,
                shape,
                bindings,
                raw_options,
                entities,
                open_axes,
                place_dcids,
                entity_options,
                facet_probe,
                facets_pinned,
                variable_labels,
            ) in probes
        ]
    )
    events = tuple(ev for group_events, _ in groups for ev in group_events)
    warns = tuple(w for _, group_warns in groups for w in group_warns)
    return events, warns
