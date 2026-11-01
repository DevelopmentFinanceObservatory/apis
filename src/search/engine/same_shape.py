"""Merge grounded regions that share a shape.

The merge itself makes no LLM read. The caller injects the two graph readers a merge can
need: `set_ref_for` names a merged value set, and `facet_options_for` judges the merged spec's
facet options over its merged `FacetProbe`. A merged answer's preview reads its merged
identity, a composite's components or else one resubmit, through the request clients the
caller passes (`preview_reads`): a streamed merge launches it as a trailing task, and a
buffered one reads it in the calling thread (`_with_merged_preview`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, NamedTuple

from code_index.composites import composite_ref
from code_index.shared import get_index
from dfo_platform.request_deadline import bounded_deadline, request_deadline
from search.engine.config import settings
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.envelope import now_ms
from search.engine.extract import DateRequest
from search.engine.region.codes import (
    DEFLATOR_PROJECTED,
    ELECTED_SHAPE_NO_DATA,
    UN_SYSTEM_UNAVAILABLE,
)
from search.engine.region.facet_probe import FacetProbe, merge_facet_probes
from search.engine.region.observations.dispatch import _attach_observations
from search.engine.region.option_coverage import (
    count_slot_options,
    measure_served_counter,
    merged_served_filter,
)
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    OpenAxisFact,
    OptionServedFact,
    ServedSeriesRows,
    StatedMembers,
    WalkedMembers,
    identity_query,
    union_served,
    walk_key,
)
from search.engine.retrieval.coverage import answer_coverage, answer_headline
from search.engine.retrieval.merged_plan import merged_read_plan
from search.engine.spec_build import (
    RoleAxisFactInput,
    build_spec,
    entity_from_resolved,
    spec_named_groups,
)
from search.engine.unit.conversion import with_unit_conversion
from search.models import (
    BindingSet,
    BindingUnbound,
    BindingValue,
    EntityOptions,
    IdentityShapeSchema,
    ProvenanceRef,
    SlotValue,
    StatVar,
)

if TYPE_CHECKING:
    from search.engine.region import RegionResult
    from search.engine.region.composite_answer import (
        ComponentReads,
        CompositeRelaunch,
        IdentityPreview,
    )
    from search.engine.region.option_coverage import RoleServedFilter
    from search.engine.retrieval.member_facts import MemberFacts
    from search.engine.retrieval.read_plan import ReadPlan
    from search.models import (
        Coverage,
        Entity,
        FacetOptions,
        GraphRef,
        Headline,
        PartScope,
        ResolvedFacet,
        ResolvedQualifier,
        RoleAxis,
        Slot,
        SlotOptions,
        Spec,
        SpecObservations,
        Warning,
    )
    from series_coverage.index import CoverageIndex

ScopeReader = Callable[["RegionResult", "Sequence[RegionResult]"], "PartScope | None"]
"""Build the scope of a merged region from it and the group it folds
(`region.pipeline.merged_scope_choices` bound to the request's coverage snapshot)."""

FacetOptionsReader = Callable[[FacetProbe], "tuple[list[FacetOptions], Warning | None]"]
"""Build facet options over a probe, with a warning when the read fails
(`option_coverage._maybe_build_facet_options` bound to the request's clients)."""


def five_tuple_key(
    spec: Spec,
) -> tuple[str | None, str | None, str | None, str | None, str | None, bool]:
    """Return the structural key used to group same-shape specs.

    The five dimensions are population type, measured property, stat type,
    qualifier, and denominator. The sixth component distinguishes dangling
    arcs from resolved targets with the same dcid. Each may be ``None`` when
    the StatVar does not assert it.
    """
    sh = spec.resolved.shape
    return (
        sh.population_type.ref if sh.population_type else None,
        sh.measured_property.ref if sh.measured_property else None,
        sh.stat_type.ref if sh.stat_type else None,
        sh.measurement_qualifier.ref if sh.measurement_qualifier else None,
        sh.measurement_denominator.ref if sh.measurement_denominator else None,
        sh.measurement_denominator_unresolved,
    )


def fold_key(region: RegionResult) -> tuple[object, ...]:
    """Return the key that decides which definite regions fold into one answer.

    Regions fold when they share a five-tuple (`five_tuple_key`). A region compared with
    the others (`RegionResult.separate`) keys on its own position, so it never folds and
    no other region folds into it. `collapse_same_shape` and the combiner's grouping
    (`conjoin._group_definite_regions`) both group by this key.
    """
    if region.separate:
        return ("separate", region.earliest_index)
    return five_tuple_key(region.spec)


def _slot_key_t(slot) -> tuple[str, str | None]:
    """Return a slot's axis/property key for Slot or SlotOptions."""
    return (slot.key.axis, slot.key.property.ref if slot.key.property else None)


def _slot_signature(slots_by_key: Mapping, key: tuple[str, str | None]) -> tuple[str | None, ...]:
    """Return a comparable slot binding, treating absent and unbound as equal.

    Value and set signatures retain their dcids, so bound/unbound differences
    remain conflicts.
    """
    s = slots_by_key.get(key)
    if s is None or s.binding.kind == "unbound":
        return ("unbound",)
    if s.binding.kind == "value":
        ref = s.binding.value.ref
        return ("value", ref.ref if ref else None)
    return ("set", *sorted(v.ref.ref for v in s.binding.values if v.ref))


def _union_slot_options(
    group: list[RegionResult], merged_slots: Sequence[Slot]
) -> tuple[list[SlotOptions], tuple[OptionServedFact, ...]]:
    """Union options by slot key, deduplicating values in group order, and count the union
    from the members' unioned counting facts (`_union_option_axis_facts`,
    `option_coverage.count_slot_options`).

    Member-specific options can include resolved recipients, so taking only the
    first member would omit values in a merged BindingSet. A menu the unioned facts do not
    cover loses its members' counts, which overlapping populations cannot sum. The union is
    capped at ``max_options``; ``truncated`` combines member flags and overflow. First-seen
    order and source are retained. Returns the menus and the facts they were counted from.
    """
    by_key: dict[tuple[str, str | None], SlotOptions] = {}
    seen_dcids: dict[tuple[str, str | None], set[str]] = {}
    for r in group:
        for so in r.spec.resolved.slot_options:
            key = _slot_key_t(so)
            if key not in by_key:
                by_key[key] = so
                seen_dcids[key] = {o.value.ref.ref for o in so.options if o.value.ref}
                continue
            existing = by_key[key]
            seen = seen_dcids[key]
            merged_options = list(existing.options)
            for o in so.options:
                dcid = o.value.ref.ref if o.value.ref else None
                if dcid is not None and dcid not in seen:
                    seen.add(dcid)
                    merged_options.append(o)
            by_key[key] = existing.model_copy(
                update={
                    "options": merged_options,
                    "truncated": existing.truncated or so.truncated,
                }
            )
    facts = _union_option_axis_facts(group)
    covered = {fact.property for fact in facts}
    unioned = [
        so
        if so.key.property is not None and so.key.property.ref in covered
        else so.model_copy(
            update={
                "options": [
                    o.model_copy(update={"served_rows": None, "served_rows_exact": None})
                    for o in so.options
                ]
            }
        )
        for so in by_key.values()
    ]
    counted = count_slot_options(unioned, facts, merged_slots)
    return [
        so.model_copy(
            update={
                "options": so.options[: settings().max_options],
                "truncated": so.truncated or len(so.options) > settings().max_options,
            }
        )
        for so in counted
    ], facts


def _union_option_axis_facts(group: list[RegionResult]) -> tuple[OptionServedFact, ...]:
    """Union each property's counting fact across the group, option by option, so a series
    several members serve counts once, its year masks ORed (`bound_values.union_served`).
    A property some member has no fact for is left out: that member's menu was not
    counted, so the union cannot be."""
    shared = [
        fact.property
        for fact in group[0].option_axis_facts
        if all(any(f.property == fact.property for f in r.option_axis_facts) for r in group)
    ]
    merged: list[OptionServedFact] = []
    for prop in shared:
        served: dict[str, ServedSeriesRows] = {}
        for r in group:
            for fact in r.option_axis_facts:
                if fact.property == prop:
                    for value, found in fact.served.items():
                        served[value] = union_served(served.get(value, {}), found)
        merged.append(OptionServedFact(property=prop, served=served))
    return tuple(merged)


def _merge_facet_options(
    group: list[RegionResult],
    merged_spec: Spec,
    facet_options_for: FacetOptionsReader | None,
) -> tuple[Spec, FacetProbe | None, Warning | None]:
    """Judge the merged spec's facet options over the members' merged probe.

    The merged probe covers every member's StatVars and entities, the merged spec's preview
    window and the shared pins, so an offered value is one a pin on the merged spec serves.
    A value every merged StatVar carries is offered by at least one member, so a group whose
    members offer nothing offers nothing and needs no read. Returns the spec with its
    options, the merged probe, and the read's warning. A group that offers options without
    a probe on every member, or without `facet_options_for`, is a caller error.
    """
    probes = [r.facet_probe for r in group if r.facet_probe is not None]
    merged_probe = (
        # Same-shape members share one identity, so one window and the years it reads.
        merge_facet_probes(probes, years=probes[0].years) if probes else None
    )
    if not any(r.spec.resolved.facet_options for r in group):
        return merged_spec, merged_probe, None
    if merged_probe is None or len(probes) != len(group) or facet_options_for is None:
        raise ValueError(
            "a same-shape merge whose members offer facet options needs every member's "
            "facet_probe and a facet_options_for reader"
        )
    facet_options, warning = facet_options_for(merged_probe)
    merged_spec = merged_spec.model_copy(
        update={
            "resolved": merged_spec.resolved.model_copy(update={"facet_options": facet_options})
        }
    )
    if warning is not None:
        warning = warning.model_copy(update={"spec_id": merged_spec.spec_id})
    return merged_spec, merged_probe, warning


def _union_equivalent_readings(group: list[RegionResult]) -> list[Spec] | None:
    """Union member twins by ``spec_id`` in group order, or return ``None``.

    The merged spec spans every member's StatVars, so it needs each member's twins.
    """
    seen: set[str] = set()
    merged: list[Spec] = []
    for r in group:
        for twin in r.spec.resolved.equivalent_readings or []:
            if twin.spec_id not in seen:
                seen.add(twin.spec_id)
                merged.append(twin)
    return merged or None


def _union_query_qualifiers(group: list[RegionResult]) -> list[ResolvedQualifier]:
    """Union qualifier records in group order, one per text, offsets, operation and
    selected codes.

    Records alike in those five are one record, and first-seen keeps the base member's
    verdict. Records of one text that sit at different offsets, or that a union
    or refinement joined to different codes (`ResolvedQualifier.selected`), all survive,
    so the merged trace keeps every member's operation.
    """
    seen: set[tuple[object, ...]] = set()
    merged: list[ResolvedQualifier] = []
    for r in group:
        for q in r.spec.resolved.trace.query_qualifiers:
            key = (q.text, q.start, q.end, q.operation, tuple(ref.ref for ref in q.selected))
            if key not in seen:
                seen.add(key)
                merged.append(q)
    return merged


def _simplify_merged_union(refs: list[GraphRef]) -> list[GraphRef]:
    """Drop each of `refs` that descends from another of them.

    A fold reads its members as a union, as the bind of one part reads codes a union
    connective joins (`phase_rank_bind._combine_matches`): HIV/AIDS (13040) folded with
    reproductive health (130) binds 130, since 130 already holds 13040. Codes that do not
    nest keep their own values, so health (120) and HIV/AIDS (13040) stay a set.
    """
    index = get_index()
    nested = {
        descendant
        for ref in refs
        for descendant in index.descendants(ref.ref)
        if descendant != ref.ref
    }
    return [ref for ref in refs if ref.ref not in nested]


def _union_open_axis_facts(group: list[RegionResult]) -> tuple[OpenAxisFact, ...]:
    """Union each property's per-member values across the group.

    A where-axis merge can serve one StatVar in several members, so the union is keyed
    by member dcid and counts each served StatVar once. A member with no fact for a
    property serves no StatVar asserting it, so its served StatVars count as undeclared
    there. Properties keep first-seen group order.
    """
    order: list[str] = []
    for r in group:
        for fact in r.open_axis_facts:
            if fact.property not in order:
                order.append(fact.property)
    merged: list[OpenAxisFact] = []
    for prop in order:
        member_values: dict[str, str | None] = {}
        value_names: dict[str, str] = {}
        for r in group:
            fact = next((f for f in r.open_axis_facts if f.property == prop), None)
            if fact is None:
                member_values.update(dict.fromkeys(_served_member_dcids(r)))
            else:
                member_values.update(fact.member_values)
                value_names |= fact.value_names
        merged.append(
            OpenAxisFact(property=prop, member_values=member_values, value_names=value_names)
        )
    return tuple(merged)


def _served_member_dcids(region: RegionResult) -> list[str]:
    """The StatVars ``region`` serves.

    Every open-axis fact keys all served members. A region with no fact reads its
    Spec's members.
    """
    if region.open_axis_facts:
        return list(region.open_axis_facts[0].member_values)
    return [sv.ref.ref for sv in region.spec.resolved.members]


def _merge_option_dcids(opts: Sequence[EntityOptions]) -> tuple[list, set[str]]:
    """Merge entity options by dcid in first-seen order, returning values and dcids."""
    seen: set[str] = set()
    merged: list = []
    for eo in opts:
        for o in eo.options:
            dcid = o.value.ref.ref if o.value.ref else None
            if dcid is not None and dcid not in seen:
                seen.add(dcid)
                merged.append(o)
    return merged, seen


def _union_role_axis_facts(
    group: list[RegionResult],
) -> tuple[list[RoleAxisFactInput], list[EntityOptions]]:
    """Merge role facts and entity options together so their counts agree.

    Open role populations can overlap across members, so the merged count is the size of
    the union of the members' populations (`RegionResult.role_axis_facts`). It is known
    only when every member records the role's population. The role's options are the
    members' lists merged by dcid in group order when every member lists the role in
    full. Otherwise they are withheld as a single answer withholds them
    (`option_coverage._entity_options_count_only`): an empty list, truncated unless the
    merged population is known to be empty. Entity options without role facts are
    bound-axis seed lists; union and cap them without creating a role fact.

    Returns one role-fact input and matching option entry per named role, in
    first-seen group order.
    """
    order: list[RoleAxis] = []
    populations: dict[RoleAxis, list[frozenset[str] | None]] = {}
    options_by_role: dict[RoleAxis, list[EntityOptions]] = {}
    for r in group:
        for fact in r.role_axis_facts:
            if fact.role not in populations:
                order.append(fact.role)
            populations.setdefault(fact.role, []).append(fact.member_dcids)
        for eo in r.spec.resolved.entity_options:
            if eo.role not in options_by_role and eo.role not in order:
                order.append(eo.role)
            options_by_role.setdefault(eo.role, []).append(eo)

    role_inputs: list[RoleAxisFactInput] = []
    merged_options: list[EntityOptions] = []
    cap = settings().max_options
    for role in order:
        members = populations.get(role)
        opts = options_by_role.get(role, [])

        if members is None:
            # Bound-axis seed options have no open-role population to count.
            merged_slot_options, _ = _merge_option_dcids(opts)
            merged_options.append(
                opts[0].model_copy(
                    update={
                        "options": merged_slot_options[:cap],
                        "truncated": any(eo.truncated for eo in opts)
                        or len(merged_slot_options) > cap,
                    }
                )
            )
            continue

        member_dcids: frozenset[str] | None = None
        if len(members) == len(group) and all(m is not None for m in members):
            member_dcids = frozenset().union(*(m for m in members if m is not None))
        role_inputs.append(RoleAxisFactInput(role=role, member_dcids=member_dcids))

        if len(opts) == len(group) and not any(eo.truncated for eo in opts):
            merged_slot_options, _ = _merge_option_dcids(opts)
            options = merged_slot_options[:cap]
            truncated = len(merged_slot_options) > cap
        else:
            options = []
            truncated = member_dcids is None or len(member_dcids) > 0
        merged_options.append(
            EntityOptions(
                entity_ref=None, role=role, options=options, truncated=truncated, source="observed"
            )
        )
    return role_inputs, merged_options


def _bound_facet_axes(group: list[RegionResult]) -> set[str]:
    """Return request-bound facet axes across the group.

    Bound means ``basis == "bound"`` or a unit facet has ``request_basis``.
    Merge gates compare these values because they come from the request, not
    observation enrichment.
    """
    axes: set[str] = set()
    for r in group:
        for axis, facet in r.spec.resolved.facets.items():
            if facet.basis == "bound" or facet.request_basis is not None:
                axes.add(axis)
    return axes


def _merge_resolved_facets(
    group: list[RegionResult], bound_axes: set[str]
) -> dict[str, ResolvedFacet]:
    """Merge resolved facets, keeping request-bound axes unchanged.

    Other axes retain one reference only when every member reports the same
    scalar selection. Missing values, differing refs, or any ``per_series``
    selection produce a merged ``per_series`` facet, which on the unit axis lists every
    member's units in group order.
    """
    from search.models import ResolvedFacet

    all_axes: set[str] = set()
    for r in group:
        all_axes |= set(r.spec.resolved.facets)

    result: dict[str, ResolvedFacet] = {}
    for axis in all_axes:
        facets = [r.spec.resolved.facets.get(axis) for r in group]
        if axis in bound_axes:
            result[axis] = next(f for f in facets if f is not None)
            continue

        present = [f for f in facets if f is not None]
        basis = present[0].basis
        heterogeneous = len(present) != len(group)
        distinct: dict[str, ResolvedFacet] = {}
        for f in present:
            if f.selection == "per_series" or f.ref is None:
                heterogeneous = True
                continue
            distinct.setdefault(f.ref, f)
        if not heterogeneous and len(distinct) == 1:
            shared = next(iter(distinct.values()))
            result[axis] = ResolvedFacet(
                basis=basis,
                ref=shared.ref,
                label=shared.label,
                source=shared.source,
                data_commons=shared.data_commons,
                documentation=shared.documentation,
                currency=shared.currency,
                account=shared.account,
            )
        else:
            refs = (ref for f in present for ref in ([f.ref] if f.ref else f.refs or ()))
            result[axis] = ResolvedFacet(
                basis=basis,
                selection="per_series",
                refs=list(dict.fromkeys(refs)) if axis == "unit" else None,
            )
    return result


def _union_entities(group: list[RegionResult]) -> list[Entity]:
    """Union displayed entities by ``(role, dcid)`` in group order.

    One dcid can appear under different roles across members.
    """
    seen_role_dcid: set[tuple[str, str]] = set()
    entities: list[Entity] = []
    for r in group:
        for re in r.spec.resolved.entities:
            key = (re.role, re.ref)
            if key not in seen_role_dcid:
                seen_role_dcid.add(key)
                entities.append(entity_from_resolved(re))
    return entities


def _union_sources(group: list[RegionResult]) -> list[ProvenanceRef]:
    """Union the members' cited sources by dcid in group order. Each member cites the
    sources its own read plan serves, and the merged answer serves every member's."""
    sources: dict[str, ProvenanceRef] = {}
    for r in group:
        for source in r.spec.resolved.trace.resolved_sources:
            sources.setdefault(source.ref, source)
    return list(sources.values())


def _identity_entity_bucket_diff(group: list[RegionResult]) -> set[str]:
    """Return identity role buckets whose dcid sets differ across members.

    Missing buckets count as empty. The merge gate uses this to catch entity
    differences hidden from slot comparisons. Non-where merges require equal
    buckets; a where merge may differ in one, but a second signals another axis.
    """
    base = {
        role: frozenset(dcids) for role, dcids in group[0].spec.definition.identity.entities.items()
    }
    all_roles: set[str] = set(base)
    for r in group[1:]:
        all_roles |= set(r.spec.definition.identity.entities)

    diff: set[str] = set()
    for role in all_roles:
        base_bucket = base.get(role, frozenset())
        for r in group[1:]:
            bucket = frozenset(r.spec.definition.identity.entities.get(role, ()))
            if bucket != base_bucket:
                diff.add(role)
                break
    return diff


def _union_identity_entities(group: list[RegionResult]) -> list[Entity]:
    """Union identity entities from the buckets that fed each member's spec id.

    Use definition buckets rather than displayed entities: the dev-finance
    ``place_as_constraint=False`` path collapses displayed roles to a subject,
    while identity still distinguishes provider from recipient. Deduplicate by
    ``(role_key, dcid)`` in group order. Rebuilt display fields do not affect the
    hash, which reads only role and dcid.
    """
    from search.models import ResolvedEntity

    seen_role_dcid: set[tuple[str, str]] = set()
    entities: list[Entity] = []
    for r in group:
        for role_key, dcids in r.spec.definition.identity.entities.items():
            for dcid in dcids:
                key = (role_key, dcid)
                if key not in seen_role_dcid:
                    seen_role_dcid.add(key)
                    entities.append(
                        entity_from_resolved(
                            ResolvedEntity(
                                ref=dcid,
                                label=None,
                                entity_type=None,
                                role=role_key,  # ty: ignore[invalid-argument-type]
                                confidence=None,
                            )
                        )
                    )
    return entities


def collapse_same_shape(
    definite: list[RegionResult],
    *,
    set_ref_for: Callable[[list[str]], GraphRef | None] | None = None,
    facet_options_for: FacetOptionsReader | None = None,
    scope_for: ScopeReader | None = None,
    preview_reads: ComponentReads | None = None,
) -> tuple[list[RegionResult], list[tuple[str | None, int]]]:
    """Merge definite regions with the same fold key (`fold_key`) when their identities
    align.

    A multi-member group must share its window, scope, identity facets,
    ``unit_lexicon_version``, ``declaration_version``, and request-bound facet
    values. Unbound resolved facets may vary.
    Slot comparison treats an absent slot like ``BindingUnbound``.

    The group merges in either case:
    - No binding differs and at least one unbound slot appears in only some
      members. Union the slot rosters without adding a ``BindingSet``.
    - Exactly one slot differs, every member binds it to a value or, off the where axis,
      a set (a value reads as a set of one), members binding different sets share no
      ref, the union holds at least two distinct refs, and entity buckets meet the
      dimensionality rule. The refs drop each one nested under another
      (`_simplify_merged_union`), and one ref left binds a value.

    A ``where`` difference may widen one identity-entity bucket. Roster-only
    and other slot merges require identical buckets; otherwise the merge would
    claim combinations no member resolved. Differing resolved facets merge as
    ``per_series`` when members select different values. Failed gates, exact duplicates, or
    multiple differing slots keep the first region and return the rest as
    ``SAME_SHAPE_MULTISLOT`` residuals.

    Merges rebuild the spec id and slot filters, union slot and entity options, judge
    facet options over the merged spec (`_merge_facet_options`), preserve shared
    identity facets, and set ``earliest_index`` to the group's minimum. A merged region
    states the scope ``scope_for`` builds over the merged identity, none without it. A group
    of composite answers merges into the composite over the merged identity
    (`composite_answer.as_merged_composite`), and refuses without ``preview_reads``, the
    request's clients. Where its members preview, a merged region previews the read of its
    merged identity through ``preview_reads`` (`_with_merged_preview`). The merged regions
    whose members embed their previews read theirs together in the calling thread
    (`_read_merged_previews`). Returns
    effective regions and residual ``(variable_text, earliest_index)`` pairs;
    ``variable_text`` may be ``None`` for dataset-resubmit entries.
    """
    if not definite:
        return [], []

    groups: dict[tuple, list[RegionResult]] = {}
    for r in definite:
        groups.setdefault(fold_key(r), []).append(r)

    effective: list[RegionResult] = []
    residual_texts: list[tuple[str | None, int]] = []
    # The merged regions whose members embed their previews, which read theirs here.
    buffered: list[RegionResult] = []
    for group in groups.values():
        group_effective, group_residual = _collapse_group(
            group, set_ref_for, facet_options_for, scope_for, preview_reads
        )
        merged = group_effective[0]
        if merged is not group[0] and any(r.spec.resolved.observations is not None for r in group):
            buffered.append(merged)
        effective.extend(group_effective)
        residual_texts.extend(group_residual)
    if buffered:
        read = dict(zip(map(id, buffered), _read_merged_previews(buffered), strict=True))
        effective = [read.get(id(r), r) for r in effective]
    return effective, residual_texts


class _RefuseMerge(Exception):
    """Internal signal that a group fails a merge gate."""


def _collapse_group(
    group: list[RegionResult],
    set_ref_for: Callable[[list[str]], GraphRef | None] | None,
    facet_options_for: FacetOptionsReader | None,
    scope_for: ScopeReader | None,
    preview_reads: ComponentReads | None,
) -> tuple[list[RegionResult], list[tuple[str | None, int]]]:
    """Merge one group; keep the first member and residualise the rest on refusal. A group
    of composite answers, which merges into the composite over the merged identity
    (`_with_merged_preview`), refuses without `preview_reads`. The merged region states the
    scope `scope_for` builds over its identity, none without it."""
    if len(group) == 1:
        return [group[0]], []
    try:
        if preview_reads is None and _binds_composite(group[0].spec):
            raise _RefuseMerge
        merged_region = _merge_group(group, set_ref_for, facet_options_for, preview_reads)
    except _RefuseMerge:
        residual = [(r.variable_text, r.earliest_index) for r in group[1:]]
        return [group[0]], residual
    scope = scope_for(merged_region, group) if scope_for is not None else None
    return [dataclasses.replace(merged_region, scope_choices=scope)], []


def _binds_composite(spec: Spec) -> bool:
    """Whether `spec`'s identity binds a declared composite provenance (`composite_ref`)."""
    facets = spec.definition.identity.model_dump(mode="json", include={"facets"})
    return composite_ref(facets) is not None


def _check_merge_gates(group: list[RegionResult], base_spec: Spec) -> None:
    """Reject differences in shared window, scope, identity facets and open roles, trace
    versions, or request-bound facet values, and a composite answer beside one that is not
    (`_binds_composite`).

    These fields become group-wide after merging, so a difference would be lost.
    """
    if any(
        r.spec.resolved.trace.requested_window != base_spec.resolved.trace.requested_window
        for r in group[1:]
    ):
        raise _RefuseMerge
    if any(r.spec.resolved.scope != base_spec.resolved.scope for r in group[1:]):
        raise _RefuseMerge

    base_facets = base_spec.definition.identity.facets
    if any(r.spec.definition.identity.facets != base_facets for r in group[1:]):
        raise _RefuseMerge
    # A composite answer merges only with composites, so the merge restates the composite.
    if len({_binds_composite(r.spec) for r in group}) > 1:
        raise _RefuseMerge

    base_open_roles = base_spec.definition.identity.open_roles
    if any(r.spec.definition.identity.open_roles != base_open_roles for r in group[1:]):
        raise _RefuseMerge

    base_unit_lexicon_version = base_spec.resolved.unit_lexicon_version
    if any(r.spec.resolved.unit_lexicon_version != base_unit_lexicon_version for r in group[1:]):
        raise _RefuseMerge

    base_declaration_version = base_spec.resolved.declaration_version
    if any(r.spec.resolved.declaration_version != base_declaration_version for r in group[1:]):
        raise _RefuseMerge

    bound_facet_axes = _bound_facet_axes(group)
    if any(
        r.spec.resolved.facets.get(axis) != base_spec.resolved.facets.get(axis)
        for axis in bound_facet_axes
        for r in group[1:]
    ):
        raise _RefuseMerge


def _resolve_diff_key(
    group: list[RegionResult],
    base_slots: Mapping[tuple[str, str | None], object],
) -> tuple[str, str | None] | None:
    """Return the sole differing key, or ``None`` for an asymmetric unbound roster.

    Absent and explicit unbound slots compare equal. Refuse exact duplicates
    and groups with multiple differing keys.
    """
    member_slots_by_key: list[Mapping[tuple[str, str | None], object]] = [base_slots]
    all_keys: set[tuple[str, str | None]] = set(base_slots)
    for r in group[1:]:
        r_slots = {_slot_key_t(s): s for s in r.spec.resolved.slots}
        member_slots_by_key.append(r_slots)
        all_keys |= set(r_slots)

    # Presence distinguishes exact duplicates from differing unbound rosters.
    differing_keys: set[tuple[str, str | None]] = set()
    asymmetric_roster = False
    for k in all_keys:
        present_count = sum(1 for m in member_slots_by_key if k in m)
        if 0 < present_count < len(member_slots_by_key):
            asymmetric_roster = True
        signatures = {_slot_signature(m, k) for m in member_slots_by_key}
        if len(signatures) > 1:
            differing_keys.add(k)

    if len(differing_keys) == 0 and asymmetric_roster:
        return None
    if len(differing_keys) == 1:
        return next(iter(differing_keys))
    raise _RefuseMerge


def _build_merged_slots(
    group: list[RegionResult],
    base_spec: Spec,
    base_slots: Mapping[tuple[str, str | None], object],
    diff_key: tuple[str, str | None] | None,
    set_ref_for: Callable[[list[str]], GraphRef | None] | None,
) -> list[Slot]:
    """Build merged slots from a roster union or the union of one differing binding.

    A ``None`` key unions unbound slots after the base roster. Otherwise every
    member must bind the differing slot to graph refs, a value as a set of one and a set
    only off the where axis. Members binding different sets must share no ref, and the
    sets must union to at least two distinct refs. Each ref nested under another drops
    (`_simplify_merged_union`). One ref left binds a ``BindingValue``, more bind a
    ``BindingSet``. The set keeps the ``set_ref`` of a member whose set it equals, else
    ``set_ref_for`` labels it, on what/how only because ``isPartOf`` also represents
    geographic containment.
    """
    if diff_key is None:
        merged_slots = list(base_spec.resolved.slots)
        seen_keys = set(base_slots)
        for r in group[1:]:
            for sl in r.spec.resolved.slots:
                k = _slot_key_t(sl)
                if k not in seen_keys:
                    seen_keys.add(k)
                    merged_slots.append(sl)
        return merged_slots

    bindings: list[BindingValue | BindingSet] = []
    for r in group:
        s = next((sl for sl in r.spec.resolved.slots if _slot_key_t(sl) == diff_key), None)
        if s is None or isinstance(s.binding, BindingUnbound):
            raise _RefuseMerge
        if isinstance(s.binding, BindingSet) and diff_key[0] == "where":
            # A place set reads a region, and this merge reads no containment of places.
            raise _RefuseMerge
        bindings.append(s.binding)
    member_values = [[b.value] if isinstance(b, BindingValue) else b.values for b in bindings]
    member_refs = [[value.ref for value in values if value.ref] for values in member_values]
    if any(
        len(refs) < len(values) for refs, values in zip(member_refs, member_values, strict=True)
    ):
        # A literal or a window names no code to union.
        raise _RefuseMerge
    member_codes = [frozenset(ref.ref for ref in refs) for refs in member_refs]
    # Members that share a code would count it in two parts of one answer.
    distinct_codes = set(member_codes)
    if sum(len(codes) for codes in distinct_codes) != len(frozenset().union(*distinct_codes)):
        raise _RefuseMerge

    diff_refs: list[GraphRef] = []
    seen_dcids: set[str] = set()
    for ref in (ref for refs in member_refs for ref in refs):
        if ref.ref not in seen_dcids:
            seen_dcids.add(ref.ref)
            diff_refs.append(ref)
    if len(diff_refs) < 2:
        raise _RefuseMerge
    diff_refs = _simplify_merged_union(diff_refs)

    # Slot values may be enum values, so retain the source value_kind.
    source_vk = member_values[0][0].value_kind
    values = [SlotValue(ref=ref, value_kind=source_vk) for ref in diff_refs]
    binding: BindingValue | BindingSet
    if len(values) == 1:
        binding = BindingValue(value=values[0])
    else:
        # A member whose set the merge leaves whole keeps naming its group.
        merged_codes = frozenset(ref.ref for ref in diff_refs)
        merged_set_ref = next(
            (
                b.set_ref
                for b, codes in zip(bindings, member_codes, strict=True)
                if isinstance(b, BindingSet) and b.set_ref is not None and codes == merged_codes
            ),
            None,
        )
        if merged_set_ref is None and diff_key[0] in ("what", "how") and set_ref_for is not None:
            merged_set_ref = set_ref_for([r.ref for r in diff_refs])
        binding = BindingSet(values=values, set_ref=merged_set_ref)
    return [
        sl.model_copy(update={"binding": binding}) if _slot_key_t(sl) == diff_key else sl
        for sl in base_spec.resolved.slots
    ]


def _merge_stat_vars(
    group: list[RegionResult],
    is_where: bool,
    diff_key: tuple[str, str | None] | None,
) -> tuple[list[StatVar], int, bool]:
    """Union the members' served StatVars by ref and derive the merged population count.

    Where merges can overlap, the larger of the union size and
    ``max(member_count)`` is a lower bound. Roster-only merges share
    one identity, so the maximum is exact when any member's count is exact. Other
    slot differences can also overlap when a StatVar omits that constraint. Their
    union count is exact only when every member count is exact and the member serves
    that many StatVars; otherwise use the larger of the union size and the largest
    member count as a lower bound.
    """
    all_stat_vars: list[StatVar] = []
    seen_sv_dcids: set[str] = set()
    for r in group:
        for sv in r.spec.resolved.members:
            if sv.ref.ref not in seen_sv_dcids:
                seen_sv_dcids.add(sv.ref.ref)
                all_stat_vars.append(sv)

    if is_where:
        return (
            all_stat_vars,
            max(len(all_stat_vars), max(r.spec.resolved.stat_vars.member_count for r in group)),
            False,
        )
    if diff_key is None:
        return (
            all_stat_vars,
            max(r.spec.resolved.stat_vars.member_count for r in group),
            any(r.spec.resolved.stat_vars.member_count_exact for r in group),
        )
    if all(
        r.spec.resolved.stat_vars.member_count_exact
        and len(r.spec.resolved.members) >= r.spec.resolved.stat_vars.member_count
        for r in group
    ):
        return all_stat_vars, len(all_stat_vars), True
    return (
        all_stat_vars,
        max(len(all_stat_vars), max(r.spec.resolved.stat_vars.member_count for r in group)),
        False,
    )


def _merge_shape(group: list[RegionResult], base_spec: Spec, is_where: bool):
    """Recompute shape population metadata across the merged group.

    Count the union of admitted dcids only when every population is present and exact,
    and the merge is not on the where axis. Otherwise report the largest member count as a
    lower bound. `build_spec` states the merged Spec's `member_digest`, and labels the
    shape for the merged members from the label the base's shape was built with
    (`ResolvedSpec.built_shape_label`).
    """
    if not is_where and all(
        r.member_population is not None and r.member_population.exact for r in group
    ):
        admitted_dcids: set[str] = set()
        for r in group:
            assert r.member_population is not None  # the all() check above guarantees this
            admitted_dcids.update(r.member_population.admitted_dcids)
        merged_member_count = len(admitted_dcids)
    else:
        merged_member_count = max(r.spec.resolved.shape.member_count for r in group)
    return base_spec.resolved.shape.model_copy(
        update={
            "label": base_spec.resolved.built_shape_label,
            "member_count": merged_member_count,
        }
    )


def _merged_walked(group: list[RegionResult]) -> WalkedMembers | None:
    """What the mint states of the members every merged Spec holds, when each states the
    same served count and family digest for the same walk (`walk_key`). A where-axis merge
    binds one identity's members at more places, so `build_spec` keeps the statement.
    It states none for a merge whose identity the parts' walk does not resolve
    (`stated_members`)."""
    statements = {
        (r.spec.resolved.stat_vars.served_count, r.spec.resolved.shape.member_digest) for r in group
    }
    identities = [r.spec.definition.identity for r in group]
    if len(statements) != 1 or not all(
        isinstance(identity.shape, IdentityShapeSchema) for identity in identities
    ):
        return None
    ((served_count, digest),) = statements
    keys = {walk_key(identity_query(identity)) for identity in identities}
    if served_count is None or len(keys) != 1:
        return None
    return WalkedMembers(
        key=keys.pop(), stated=StatedMembers(served_count=served_count, digest=digest)
    )


def _merge_slot_flags(
    group: list[RegionResult], merged_slots: list[Slot]
) -> tuple[list[str], list[str]]:
    """Return the merged spec's (inapplicable_slots, unresolved_slots), sorted.

    Inapplicable properties intersect across members; unresolved properties
    union, then exclude properties bound in the merged slots.
    """
    merged_inapplicable = set(group[0].spec.resolved.inapplicable_slots)
    for r in group[1:]:
        merged_inapplicable &= set(r.spec.resolved.inapplicable_slots)

    merged_bound_props = {
        sl.key.property.ref
        for sl in merged_slots
        if sl.key.axis in ("what", "how")
        and sl.key.property is not None
        and sl.binding.kind != "unbound"
    }
    merged_unresolved: set[str] = set()
    for r in group:
        merged_unresolved.update(r.spec.resolved.unresolved_slots)
    merged_unresolved -= merged_bound_props

    return sorted(merged_inapplicable), sorted(merged_unresolved)


class _MergedCounts(NamedTuple):
    """The merged answer's read plan (`merged_plan.merged_read_plan`), and the coverage and
    headline stated from it, read from the request's coverage index."""

    plan: ReadPlan
    coverage: Coverage
    headline: Headline | None
    coverage_index: CoverageIndex


def _merged_counts(
    group: list[RegionResult],
    base_spec: Spec,
    *,
    diff_key: tuple[str, str | None] | None,
    reads: ComponentReads | None,
) -> _MergedCounts | None:
    """The read plan a resolve of the merged identity builds, rebuilt in memory from the
    members' plans, and the coverage and headline a single answer states from its plan
    (`coverage.answer_coverage` in the members' `RegionResult.coverage_form`,
    `coverage.answer_headline`), so `build_spec` states the merged answer's download from it
    too. A merge that differs in no slot reads one identity, whose plan every member holds,
    so it reads the base's plan alone.

    A member's plan narrowed to its twin cluster's currency (`facets.cluster_currency_filter`)
    is narrowed as the merged one is: that narrowing binds the identity's unit to the units it
    keeps (`facets.cluster_currency_binding`), the merge gate holds every member to one
    identity facet binding, and a resolve of the merged identity keeps a facet by that
    binding, which keeps exactly the facets each member's narrowing kept.

    The result is None, and the merged spec states the coverage, unserved entities and
    members, data years and continuation the members state (`_merge_coverage`,
    `_merged_unserved`) and no download, where

    - `reads`, the request's clients, is absent;
    - the merge is a composite, which states its components' download
      (`composite_answer.as_merged_composite`);
    - a member's plan was narrowed to the units a unit request elected over that member's own
      group (`RegionResult.units_elected`), which a resolve of the merged identity elects over
      the merged group;
    - a member holds no single plan, or the members state coverage in different forms;
    - `merged_read_plan` cannot rebuild the plan without a read."""
    if (
        reads is None
        or _binds_composite(base_spec)
        or any(region.units_elected for region in group)
    ):
        return None
    forms = {region.coverage_form for region in group}
    if len(forms) != 1 or any(len(region.served_plans) != 1 for region in group):
        return None
    (form,) = forms
    if form is None:
        return None
    read = group if diff_key is not None else group[:1]
    coverage_index = reads.coverage.index
    plan = merged_read_plan(
        [region.served_plans[0] for region in read], coverage_index=coverage_index
    )
    if plan is None:
        return None
    window = base_spec.resolved.trace.requested_window
    date_request = (
        None
        if window is None
        else DateRequest(window=None, latest=True)
        if window == "latest"
        else DateRequest(window=window, latest=False)
    )
    return _MergedCounts(
        plan=plan,
        coverage=answer_coverage(plan, form),
        headline=answer_headline(plan, coverage_index, date_request=date_request),
        coverage_index=coverage_index,
    )


def _merge_coverage(group: list[RegionResult], base_spec: Spec):
    """Downgrade exact coverage because its count covers only the base member, for a merge
    whose plan `_merged_counts` does not rebuild.

    Preserve positive ``has_data`` from any member; this pure merge cannot
    recount observations over the union.
    """
    from search.models import CoverageBare, CoverageBreadth

    base_cov = base_spec.resolved.coverage
    if base_cov.kind != "exact":
        return base_cov
    merged_has_data = any(r.spec.resolved.coverage.has_data for r in group)
    if base_cov.dimensions:
        return CoverageBreadth(
            has_data=merged_has_data, dimensions=base_cov.dimensions, window=base_cov.window
        )
    return CoverageBare(has_data=merged_has_data, window=base_cov.window)


def _merge_slot_filter_properties(
    group: list[RegionResult],
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    """Union slot-filter properties by provenance across the group: the text-recognised,
    the grouped, the measure-elected and the defaulted.

    Query-recognized properties outrank measure elections, which outrank
    defaults. Subtraction keeps each property in one set, as ``build_spec``
    requires. A property every member that names it binds by a sector concept's declared
    grouping is grouped on the merge, and text-recognised too, as ``build_spec`` requires.
    A slot filter states one provenance, so a property one member grouped and another
    named by its code ("health and education") is text-recognised alone, which holds for
    every value it binds.
    """
    grouped_somewhere = frozenset(
        sf.key.property.ref
        for r in group
        for sf in r.spec.resolved.trace.slot_filters
        if sf.provenance == "declared_grouping" and sf.key.property is not None
    )
    named_somewhere = frozenset(
        sf.key.property.ref
        for r in group
        for sf in r.spec.resolved.trace.slot_filters
        if sf.provenance == "text_recognized" and sf.key.property is not None
    )
    grouped = grouped_somewhere - named_somewhere
    text_recognized = grouped_somewhere | named_somewhere
    measure_elected = (
        frozenset(
            sf.key.property.ref
            for r in group
            for sf in r.spec.resolved.trace.slot_filters
            if sf.provenance == "measure_elected" and sf.key.property is not None
        )
        - text_recognized
    )
    defaulted = (
        frozenset(
            sf.key.property.ref
            for r in group
            for sf in r.spec.resolved.trace.slot_filters
            if sf.provenance == "declared_default" and sf.key.property is not None
        )
        - text_recognized
        - measure_elected
    )
    return text_recognized, grouped, measure_elected, defaulted


def _merge_measure_partition(
    group: list[RegionResult],
) -> tuple[MeasurePartition | None, tuple[MeasurePartition, ...]]:
    """Keep the partition only when every member agrees on its values.

    ``build_spec`` re-mints siblings against the merged identity.
    """
    partition_values = {
        r.measure_partition.values if r.measure_partition is not None else None for r in group
    }
    if len(partition_values) == 1 and group[0].measure_partition is not None:
        return group[0].measure_partition, group[0].measure_siblings
    return None, ()


def _dedupe_merged_warnings(warnings: Sequence[Warning]) -> tuple[Warning, ...]:
    """Keep one warning per (code, severity, message, spec_id), in first-seen key order.

    Members' ``elected_shape_no_data`` warnings share a key but name their own readings,
    so the kept warning's ``no_data`` and ``elected`` become the first-seen unions.
    """
    merged: dict[tuple[str, str, str, str | None], Warning] = {}
    for w in warnings:
        key = (w.code, w.severity, w.message, w.spec_id)
        kept = merged.get(key)
        if kept is None or w.code != ELECTED_SHAPE_NO_DATA:
            merged[key] = w
            continue
        merged[key] = kept.model_copy(
            update={
                "params": {
                    name: list(dict.fromkeys([*_labels(kept, name), *_labels(w, name)]))
                    for name in ("no_data", "elected")
                }
            }
        )
    return tuple(merged.values())


def _labels(warning: Warning, name: str) -> list[str]:
    """The shape labels an ``elected_shape_no_data`` warning lists under ``name``."""
    value = (warning.params or {}).get(name)
    return value if isinstance(value, list) else []


def _finalize_merged_region(
    group: list[RegionResult],
    merged_spec: Spec,
    *,
    preview_reads: ComponentReads | None,
    facet_probe: FacetProbe | None,
    facet_warning: Warning | None,
    measure_partition: MeasurePartition | None,
    measure_siblings: tuple[MeasurePartition, ...],
    open_axis_facts: tuple[OpenAxisFact, ...],
    option_axis_facts: tuple[OptionServedFact, ...],
    role_axis_facts: Sequence[RoleAxisFactInput],
    served_filter: RoleServedFilter | None,
    served_members: tuple[MemberFacts, ...],
    served_plans: tuple[ReadPlan, ...],
) -> RegionResult:
    """Assemble the merged region and preserve group-wide trace data.

    Copy the agreed lexicon and declaration versions from the base spec.
    Clear pending option tasks so enrichment can relaunch them for the new
    spec id, and retain place-option relaunch state. Remap group warnings to the merged spec
    id so they remain attached to their result, and add the facet-options read's warning.
    The merged region is one answer, so a warning several members raise appears once. Its
    preview is a read of the merged identity (`_with_merged_preview`).
    """
    base_resolved = group[0].spec.resolved
    merged_spec = merged_spec.model_copy(
        update={
            "resolved": merged_spec.resolved.stating_unit_lexicon_version(
                base_resolved.unit_lexicon_version
            ).stating_declaration_version(base_resolved.declaration_version)
        }
    )

    group_spec_ids = {r.spec.spec_id for r in group}
    remapped = (
        w.model_copy(update={"spec_id": merged_spec.spec_id}) if w.spec_id in group_spec_ids else w
        for r in group
        for w in r.warnings
    )
    facet_warnings = (facet_warning,) if facet_warning is not None else ()
    merged = dataclasses.replace(
        group[0],
        variable_text=" and ".join(r.variable_text for r in group if r.variable_text),
        specs=(merged_spec,),
        warnings=_dedupe_merged_warnings((*remapped, *facet_warnings)),
        timing_by_step=dict(group[0].timing_by_step),
        earliest_index=min(r.earliest_index for r in group),
        slot_options_task=None,
        place_options_task=None,
        facet_probe=facet_probe,
        # Any member may admit a phrase that suppresses a request-wide warning.
        admitted_qualifier_phrases=frozenset(
            p for r in group for p in r.admitted_qualifier_phrases
        ),
        every_member_axes=frozenset(axis for r in group for axis in r.every_member_axes),
        # A fact survives with its warning, so every member's facts carry over.
        departure_facts={key: fact for r in group for key, fact in r.departure_facts.items()},
        measure_partition=measure_partition,
        measure_siblings=measure_siblings,
        open_axis_facts=open_axis_facts,
        option_axis_facts=option_axis_facts,
        role_axis_facts=tuple(role_axis_facts),
        served_filter=served_filter,
        served_members=served_members,
        served_plans=served_plans,
    )
    return _with_merged_preview(merged, group, reads=preview_reads)


def _with_merged_preview(
    merged: RegionResult, group: list[RegionResult], *, reads: ComponentReads | None
) -> RegionResult:
    """`merged`, the region `group` merged into, restated as the merged identity answers and
    with the relaunch that previews it when the members preview. The merged identity's
    composite binding (`_binds_composite`, which the merge gate holds every member to)
    chooses between them. A group of composite answers is restated as the composite over
    the merged identity (`composite_answer.as_merged_composite`), whose `CompositeRelaunch`
    reads the components it serves. Any other merge keeps its Spec and previews by one
    resubmit of its identity (`composite_answer.IdentityPreview`). Either reads through
    `reads`, the request's clients. The members' own preview warnings leave the merged
    answer, which states its own read's. Where the members do not preview, the merged region
    reads no preview.

    Where the members stream their previews, `enrichment._reconcile_after_collapse` launches
    the relaunch and cancels the members' tasks. Where they embed them,
    `collapse_same_shape` reads it (`_read_merged_previews`).

    Raises `ValueError` when the members are composite answers or preview and `reads` is
    None: every door that combines previewing regions passes its clients, and
    `_collapse_group` refuses a composite group without them."""
    streamed = any(r.observations_relaunch is not None for r in group)
    embedded = any(r.spec.resolved.observations is not None for r in group)
    previews = streamed or embedded
    composite = _binds_composite(group[0].spec)
    unread = dataclasses.replace(
        merged, observations_task=None, observations_relaunch=None, preview_read=None
    )
    if not previews and not composite:
        return unread
    if reads is None:
        raise ValueError(
            "a merge of composite or previewing regions needs the request's preview reads"
        )
    # composite_answer reads through resubmit, which imports conjoin, which imports this module.
    from search.engine.region.composite_answer import (
        PREVIEW_WARNING_CODES,
        IdentityPreview,
        as_merged_composite,
    )

    # The codes only a preview read raises, without `un_system_unavailable`, which a member's
    # resolve can raise too.
    member_codes = (PREVIEW_WARNING_CODES - {UN_SYSTEM_UNAVAILABLE}) | {DEFLATOR_PROJECTED}
    unread = dataclasses.replace(
        unread, warnings=tuple(w for w in merged.warnings if w.code not in member_codes)
    )
    if not composite:
        return dataclasses.replace(
            unread, observations_relaunch=IdentityPreview(spec=merged.spec, reads=reads)
        )
    restated = as_merged_composite(unread, group, reads=reads)
    return restated if previews else dataclasses.replace(restated, observations_relaunch=None)


_COMBINE_TAIL_S = 0.25
"""How long before the deadline of the reads' clients (`ComponentReads.deadline`, the bound
the door awaits the combine under) the merged reads end, leaving the combine time to embed
them and build the response in memory."""


def _read_merged_previews(pending: Sequence[RegionResult]) -> list[RegionResult]:
    """Each of `pending`, merged regions whose members embedded their previews, embedding the
    preview its relaunch reads (`_with_merged_preview`), as one answer's buffered preview is
    (`dispatch._attach_observations`), with its relaunch dropped.

    The reads run together on an event loop of their own in the calling thread, a worker
    thread off the request's loop (`combine_regions`), which takes no permit of the request
    semaphore, since that belongs to the request's loop. The loop's tasks copy the thread's
    context, so the request deadline, the catalogue and the bound below reach every read.
    A merged preview is its answer's own preview, a full observation read, so like a single
    answer's buffered preview it reads to the deadline and takes no fail-open budget: the
    latest of the reads' own deadlines (`ComponentReads.deadline`) less `_COMBINE_TAIL_S`,
    so the reads end before the door's bound on the combine. A read still running at that
    bound is cut: it previews its omission (`cut_merged_preview`) and notes a degraded
    read, as does a read that swallowed one, so no cache stores its region. Abandoned
    worker threads run on, and the loop does not wait for them. The reads are timed as the
    `merged_preview` step. A code fault in a read propagates."""
    from search.engine.region.composite_answer import (
        CompositeRelaunch,
        IdentityPreview,
        cut_merged_preview,
    )

    started = now_ms()
    relaunches: list[CompositeRelaunch | IdentityPreview] = []
    for region in pending:
        relaunch = region.observations_relaunch
        assert isinstance(relaunch, CompositeRelaunch | IdentityPreview)
        relaunches.append(relaunch)
    # A composite's reads run to its latest member's (`as_merged_composite`), which may pass
    # the request clients' own.
    deadlines = [r.reads.deadline for r in relaunches if r.reads.deadline is not None]
    bound = bounded_deadline(max(deadlines) - _COMBINE_TAIL_S if deadlines else None)
    with request_deadline(bound) if bound is not None else nullcontext():
        bounded = [
            dataclasses.replace(relaunch, reads=dataclasses.replace(relaunch.reads, deadline=bound))
            for relaunch in relaunches
        ]
        outcomes = _run_together(bounded, deadline=bound)
    elapsed = now_ms() - started
    read: list[RegionResult] = []
    for region, outcome in zip(pending, outcomes, strict=True):
        if outcome is None:
            observations, raised = cut_merged_preview()
            degraded = True
        else:
            observations, raised, degraded = outcome
        spec = region.spec
        warnings = list(region.warnings)
        keyed = (w.model_copy(update={"spec_id": spec.spec_id}) for w in raised)
        warnings.extend(w for w in keyed if w not in warnings)
        spec = _attach_observations(spec, (observations, None), warnings)
        read.append(
            dataclasses.replace(
                region,
                specs=(spec,),
                warnings=tuple(warnings),
                degraded=region.degraded or degraded,
                observations_relaunch=None,
                timing_by_step={**region.timing_by_step, "merged_preview": elapsed},
            )
        )
    return read


type _MergedRead = tuple[SpecObservations, tuple[Warning, ...], bool]


def _run_together(
    relaunches: Sequence[CompositeRelaunch | IdentityPreview], *, deadline: float | None
) -> list[_MergedRead | None]:
    """Each of `relaunches` read (`read`), with whether it swallowed a read, all together on
    a new event loop run in the calling thread until `deadline` (to the end with none), and
    None for a read still running then, which is cancelled. The loop is closed without
    waiting for the worker threads its reads started, so a stalled thread holds nothing past
    `deadline`."""

    async def one(relaunch: CompositeRelaunch | IdentityPreview) -> _MergedRead:
        with recording_degraded_reads() as lost:
            observations, raised = await relaunch.read(semaphore=None)
        return observations, raised, lost.degraded

    async def every() -> list[_MergedRead | None]:
        tasks = [asyncio.ensure_future(one(relaunch)) for relaunch in relaunches]
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        _done, running = await asyncio.wait(tasks, timeout=timeout)
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        return [None if task in running else task.result() for task in tasks]

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(every())
    finally:
        # Shuts the loop's default executor down without joining its threads.
        loop.close()


def _merged_unserved(group: list[RegionResult], merged_spec: Spec) -> tuple[tuple[str, str], ...]:
    """The merged Spec's `unserved_entities`: the `(role, dcid)` pairs some member's read
    plan serves no point at (`ResolvedSpec.unserved_entities`) and no member serves, in the
    merged Spec's entity order. A member serves each of its role entities outside its own
    unserved ones, so a place one part's series holds data for on a role is served there
    by the merged answer."""
    unserved = {pair for region in group for pair in region.spec.resolved.unserved_entities}
    served = {
        (entity.role, entity.ref)
        for region in group
        for entity in region.spec.resolved.entities
        if (entity.role, entity.ref) not in region.spec.resolved.unserved_entities
    }
    return tuple(
        (entity.role, entity.ref)
        for entity in merged_spec.resolved.entities
        if (entity.role, entity.ref) in unserved - served
    )


def _merged_unserved_members(group: list[RegionResult]) -> tuple[GraphRef, ...]:
    """The merged Spec's `unserved_members`: each member's own (`ResolvedSpec.unserved_members`)
    that no member's answer serves, in group order, once each."""
    served = {
        member.ref.ref
        for region in group
        for member in region.spec.resolved.members
        if member.ref.ref not in {ref.ref for ref in region.spec.resolved.unserved_members}
    }
    named: dict[str, GraphRef] = {}
    for region in group:
        for ref in region.spec.resolved.unserved_members:
            if ref.ref not in served:
                named.setdefault(ref.ref, ref)
    return tuple(named.values())


def _merged_data_years(group: list[RegionResult]) -> tuple[int, int] | None:
    """The merged Spec's `data_years`: the earliest first and latest last year over the
    members' own (`ResolvedSpec.data_years`), `None` when no member states them."""
    spans = [years for region in group if (years := region.spec.resolved.data_years)]
    if not spans:
        return None
    return min(first for first, _last in spans), max(last for _first, last in spans)


def _merge_group(
    group: list[RegionResult],
    set_ref_for: Callable[[list[str]], GraphRef | None] | None,
    facet_options_for: FacetOptionsReader | None,
    preview_reads: ComponentReads | None,
) -> RegionResult:
    """Merge a group of at least two regions or raise ``_RefuseMerge``."""
    base_spec = group[0].spec
    _check_merge_gates(group, base_spec)

    base_slots = {_slot_key_t(s): s for s in base_spec.resolved.slots}
    diff_key = _resolve_diff_key(group, base_slots)

    # Identity slots omit where, so entity differences need a separate gate.
    # Only a where merge may widen one entity bucket; other merges require none.
    diff_buckets = _identity_entity_bucket_diff(group)
    is_where = diff_key is not None and diff_key[0] == "where"
    merge_allowed = len(diff_buckets) <= 1 if is_where else len(diff_buckets) == 0
    if not merge_allowed:
        raise _RefuseMerge

    merged_slots = _build_merged_slots(group, base_spec, base_slots, diff_key, set_ref_for)
    merged_slot_options, option_axis_facts = _union_slot_options(group, merged_slots)
    all_stat_vars, stat_vars_member_count, stat_vars_member_count_exact = _merge_stat_vars(
        group, is_where, diff_key
    )
    merged_shape = _merge_shape(group, base_spec, is_where)
    inapplicable_slots, unresolved_slots = _merge_slot_flags(group, merged_slots)
    counted = _merged_counts(group, base_spec, diff_key=diff_key, reads=preview_reads)
    text_recognized, grouped, measure_elected, defaulted = _merge_slot_filter_properties(group)
    measure_partition, measure_siblings = _merge_measure_partition(group)
    open_axis_facts = _union_open_axis_facts(group)
    role_axis_facts, entity_options = _union_role_axis_facts(group)
    bound_facet_axes = _bound_facet_axes(group)
    served_filter = merged_served_filter([r.served_filter for r in group])
    served_members = tuple(member for r in group for member in r.served_members)
    served_plans = tuple(plan for r in group for plan in r.served_plans)

    merged_spec = build_spec(
        shape=merged_shape,
        slots=merged_slots,
        stat_vars=all_stat_vars,
        stat_vars_member_count=stat_vars_member_count,
        stat_vars_member_count_exact=stat_vars_member_count_exact,
        walked=_merged_walked(group),
        entities=_union_entities(group),
        # Identity slots omit where; its entity buckets define the merged identity.
        # Use each member's hashed buckets, not its displayed entity projection.
        identity_entities=_union_identity_entities(group),
        # The merge gate holds every member to the base's open roles.
        open_roles=base_spec.definition.identity.open_roles or (),
        # All members share this request window; use the base's hashed value.
        requested_window=base_spec.resolved.trace.requested_window,
        coverage=counted.coverage if counted is not None else _merge_coverage(group, base_spec),
        headline=counted.headline if counted is not None else None,
        plan=counted.plan if counted is not None else None,
        coverage_index=counted.coverage_index if counted is not None else None,
        resolved_sources=_union_sources(group),
        slot_options=merged_slot_options,
        entity_options=entity_options,
        scope=base_spec.resolved.scope,
        inapplicable_slots=inapplicable_slots,
        unresolved_slots=unresolved_slots,
        # Shared identity facets pass through. build_spec derives provenance from merged slots.
        facet_bindings=base_spec.definition.identity.facets,
        facets=_merge_resolved_facets(group, bound_facet_axes),
        facet_options=[],
        equivalent_readings=_union_equivalent_readings(group),
        text_recognized_properties=text_recognized,
        grouped_properties=grouped,
        query_qualifiers=_union_query_qualifiers(group),
        # Every member comes from one leg, which applied one default reading.
        default_reading=base_spec.resolved.trace.default_reading,
        measure_elected_properties=measure_elected,
        defaulted_properties=defaulted,
        measure_partition=measure_partition,
        measure_siblings=measure_siblings,
        open_axis_facts=open_axis_facts,
        role_axis_facts=role_axis_facts,
        # A member's group survives on a role the merge leaves unchanged.
        named_groups=[named for region in group for named in spec_named_groups(region.spec)],
        # The merged identity's alternatives read every member's entities.
        measure_served_rows=measure_served_counter(served_filter),
    )
    merged_spec = with_unit_conversion(merged_spec, members=served_members, plans=served_plans)
    if counted is None:
        # With no merged plan, `build_spec` states none of these, so they come from the
        # members. A rebuilt plan states them as its download counts them.
        merged_spec = merged_spec.model_copy(
            update={
                "resolved": merged_spec.resolved.stating_unserved(
                    _merged_unserved(group, merged_spec)
                )
                .stating_unserved_members(_merged_unserved_members(group))
                .stating_data_years(_merged_data_years(group))
                .stating_continued(any(region.spec.resolved.continued for region in group))
            }
        )

    merged_spec, merged_probe, facet_warning = _merge_facet_options(
        group, merged_spec, facet_options_for
    )
    return _finalize_merged_region(
        group,
        merged_spec,
        preview_reads=preview_reads,
        facet_probe=merged_probe,
        facet_warning=facet_warning,
        measure_partition=measure_partition,
        measure_siblings=measure_siblings,
        open_axis_facts=open_axis_facts,
        option_axis_facts=option_axis_facts,
        role_axis_facts=role_axis_facts,
        served_filter=served_filter,
        served_members=served_members,
        served_plans=served_plans,
    )
