"""Build grounded Slot, Shape, StatVar, and Spec objects from draft components,
confirmed GraphRefs, and resolved graph facts.

Pure assembly: no LLM, no graph calls.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

from code_index.labels import (
    PRICE_BASIS_AXIS,
    PRICES_BY_QUALIFIER,
    compose_measure_label,
)
from code_index.shared import get_index, get_measure_index
from code_index.types import label_for_pairs
from origin_catalogue import ORIGINS, current_catalogue
from search.engine.bind import SlotBindingDraft
from search.engine.config import settings
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    StatedMembers,
    WalkedMembers,
    apply_slot_summability,
    bound_source,
    identity_dataset,
    measure_without_price_basis,
    mint_sibling_identity,
    price_basis_sibling,
    stated_members,
)
from search.engine.retrieval.derive import upstream_source
from search.engine.retrieval.download import download_facts
from search.engine.retrieval.member_facts import ArcsMemberFacts, asserted_constraint_properties
from search.engine.retrieval.read_plan import data_years, served_dates, unserved_entities
from search.engine.shape import (
    ShapeDraft,
    SlotKeyDraft,
    UpstreamSource,
    identity_shape_for,
    is_ratio_backed_sv,
)
from search.engine.spec_id import (
    build_identity_projection,
    build_spec_identity,
    entity_role_key,
    slots_in_identity_order,
)
from search.engine.unit.conversion import counted_conversion
from search.engine.unit.conversion_catalogue import current_conversion_catalogue
from search.heading.price_basis import price_basis_option_label
from search.models import (
    SCHEMA_VERSION,
    BaseIdentity,
    BindingSet,
    BindingUnbound,
    BindingValue,
    Confidence,
    Coverage,
    DataCommonsRef,
    DateRange,
    Entity,
    EntityGroup,
    EntityOptions,
    EntityRole,
    EntityRoleDirectional,
    EntityRoleSubject,
    FacetOptions,
    GraphRef,
    Headline,
    IdentityFacetBinding,
    MeasureAxisValue,
    OpenAxisValueCount,
    OpenConstraintAxisFact,
    ProvenanceRef,
    RequestedWindow,
    ResolutionTrace,
    ResolvedEntity,
    ResolvedFacet,
    ResolvedFilter,
    ResolvedQualifier,
    ResolvedScope,
    ResolvedSpec,
    ResolvedStatVars,
    RoleAxisFact,
    SentenceToken,
    Shape,
    ShapeUpstream,
    Slot,
    SlotKey,
    SlotOptions,
    SlotValue,
    SourceRef,
    SpecObservations,
    StatVar,
    StatVarSlotValue,
    TimeWindow,
    TraceDefaultReading,
    served_date_range,
)
from search.render.labels import _humanize_ref

if TYPE_CHECKING:
    from materialise.roles import RoleAxis
    from search.engine.retrieval.bound_values import OpenAxisFact, ServedRows
    from search.engine.retrieval.member_facts import MemberFacts
    from search.engine.retrieval.read_plan import ReadPlan
    from search.models import (  # circular at runtime, safe under TYPE_CHECKING
        BindingKind,
        BindingProvenance,
        Spec,
        SpecAlternativeMeasure,
    )
    from series_coverage.index import CoverageIndex


def pair_slot_drafts(
    slot_keys: Sequence[SlotKeyDraft], bindings: Sequence[SlotBindingDraft]
) -> list[tuple[SlotKeyDraft, SlotBindingDraft | None]]:
    """Pair each slot key with its draft in `bindings`, or None, in slot-key order.

    A slot with a constraint property takes the draft on that property. A where, when or
    source slot has none and takes the property-less draft on its own axis, so a source
    binding reaches the source slot alone.
    """
    by_property = {b.property_dcid: b for b in bindings if b.property_dcid is not None}
    by_axis = {b.axis: b for b in bindings if b.property_dcid is None}
    return [
        (
            slot_key,
            by_property.get(slot_key.property_dcid)
            if slot_key.property_dcid is not None
            else by_axis.get(slot_key.axis),
        )
        for slot_key in slot_keys
    ]


def _source_slot_binding(binding_draft: SlotBindingDraft | None) -> BindingValue | BindingUnbound:
    """The source slot's binding: the dataset a source-axis value draft names, under its
    declared label, else unbound. `retrieval.bound_values.shape_query` reads the same
    draft for the family election, so the identity names the dataset the election used."""
    dcid = bound_source([binding_draft] if binding_draft is not None else [])
    if dcid is None:
        return BindingUnbound()
    dataset = get_index().datasets.get(dcid)
    ref = GraphRef(dcid=dcid, label=dataset.label if dataset is not None else dcid)
    return BindingValue(value=SlotValue(ref=ref, value_kind="source"))


def build_slot(
    slot_key_draft: SlotKeyDraft,
    binding_draft: SlotBindingDraft | None,
    grounded_values: list[GraphRef],
    *,
    property_ref: GraphRef | None = None,
    set_ref: GraphRef | None = None,
) -> Slot:
    """Build a grounded slot, falling back to unbound when values fail to resolve.

    The source slot binds from its own draft (`_source_slot_binding`) and reads no
    grounded value."""
    # Preserve graph labels for binding. Humanise only dcid echoes. Indexed code
    # systems supply their declared plain_name.
    declaration = (
        get_index().declarations.get(slot_key_draft.property_dcid)
        if slot_key_draft.property_dcid is not None
        else None
    )
    if declaration is not None:
        slot_label = declaration.plain_name
    elif slot_key_draft.property_dcid is not None:
        slot_label = _humanize_ref(slot_key_draft.property_dcid, slot_key_draft.label)
    else:
        slot_label = slot_key_draft.label
    slot_key = SlotKey(
        axis=slot_key_draft.axis,
        property=property_ref,
        label=slot_label,
    )

    # bind_when_slot binds when afterward.
    if slot_key_draft.axis == "when":
        return Slot(key=slot_key, binding=BindingUnbound())
    if slot_key_draft.axis == "source":
        return Slot(key=slot_key, binding=_source_slot_binding(binding_draft))

    if binding_draft is None:
        return Slot(key=slot_key, binding=BindingUnbound())

    kind = binding_draft.kind

    # where binds entities. what/how bind taxonomy values.
    vk = "entity" if slot_key_draft.axis == "where" else "enum_value"

    if kind == "value":
        if grounded_values:
            sv = SlotValue(ref=grounded_values[0], value_kind=vk)
        else:
            return Slot(key=slot_key, binding=BindingUnbound())
        return Slot(key=slot_key, binding=BindingValue(value=sv))

    if kind == "set":
        # Preserve one resolved where value as BindingValue so its identity survives.
        # Role-confirmed refs echo dcids on label misses. Other set members remain ungrounded.
        if slot_key_draft.axis == "where" and len(grounded_values) == 1:
            sv = SlotValue(ref=grounded_values[0], value_kind="entity")
            return Slot(key=slot_key, binding=BindingValue(value=sv))
        if len(grounded_values) < 2:
            # A set binding needs at least two grounded values.
            return Slot(key=slot_key, binding=BindingUnbound())
        slot_values = [SlotValue(ref=gv, value_kind=vk) for gv in grounded_values]
        return Slot(key=slot_key, binding=BindingSet(values=slot_values, set_ref=set_ref))

    if kind == "unbound":
        return Slot(key=slot_key, binding=BindingUnbound())

    if kind == "absent":
        # "absent" collapses to unbound on the wire. Graph applicability is reported
        # separately through ResolvedSpec.inapplicable_slots.
        return Slot(key=slot_key, binding=BindingUnbound())


def inapplicable_property_dcid(
    slot_key_draft: SlotKeyDraft, binding_draft: SlotBindingDraft | None
) -> str | None:
    """An ``absent`` draft yields its property dcid. Other drafts yield None.

    The wire binding is unbound, so this preserves graph applicability in
    ``ResolvedSpec.inapplicable_slots``.
    """
    if binding_draft is None or binding_draft.kind != "absent":
        return None
    return slot_key_draft.property_dcid


def _resolved_entities(entities: list[Entity]) -> list[ResolvedEntity]:
    """Project entities with the shared role mapping. Confidence is unset."""
    return [
        ResolvedEntity(
            ref=e.ref.dcid,
            label=e.ref.label,
            entity_type=e.entity_type.dcid if e.entity_type else None,
            role=entity_role_key(e),
            confidence=None,
        )
        for e in entities
    ]


@dataclass(frozen=True)
class NamedGroup:
    """A group the request named ("EU member states") and the entity dcids it stands for.

    `label` is the group's name as the request gave it. `dcid` is its own graph node, when
    the graph holds one.
    """

    label: str
    dcid: str | None
    member_dcids: frozenset[str]


def spec_named_groups(spec: Spec) -> tuple[NamedGroup, ...]:
    """The groups `spec` serves, each standing for its role's entities in the identity.

    The members come from ``definition.identity.entities``, since ``resolved.entities`` is
    the display projection and can collapse origin and destination.
    """
    entities = spec.definition.identity.entities
    return tuple(
        NamedGroup(label=group.label, dcid=group.dcid, member_dcids=frozenset(entities[role]))
        for role, group in (spec.resolved.entity_groups or {}).items()
    )


def role_entity_groups(
    groups: Iterable[NamedGroup], entities: Iterable[Entity]
) -> dict[str, EntityGroup]:
    """The group each role's entities stand for, keyed by identity role key.

    A role takes a group when its bound dcids are exactly the group's members, so a role
    whose entities were capped, widened or edited reads as its members instead. The first
    matching group wins.
    """
    dcids_by_role: dict[str, set[str]] = {}
    for entity in entities:
        dcids_by_role.setdefault(entity_role_key(entity), set()).add(entity.ref.dcid)
    groups = tuple(groups)
    result: dict[str, EntityGroup] = {}
    for role, dcids in dcids_by_role.items():
        group = next((g for g in groups if g.member_dcids == dcids), None)
        if group is not None:
            result[role] = EntityGroup(dcid=group.dcid, label=group.label)
    return result


def entity_from_resolved(re: ResolvedEntity) -> Entity:
    """Rebuild an Entity from its wire projection for same-shape merges.

    Directional role-property provenance is absent on the wire and restored as a
    placeholder. Identity uses the role axis and entity dcid instead.
    """
    ref = GraphRef(dcid=re.ref, label=re.label or re.ref)
    entity_type = GraphRef(dcid=re.entity_type, label=re.entity_type) if re.entity_type else None
    role: EntityRole
    if re.role == "observationAbout":
        role = EntityRoleSubject()
    else:
        role = EntityRoleDirectional(role=GraphRef(dcid=re.role, label=re.role), axis=re.role)
    return Entity(ref=ref, entity_type=entity_type, role=role)


def _resolved_stat_vars(
    stat_vars: Sequence[StatVar],
    *,
    member_count: int | None,
    member_count_exact: bool,
    stated: StatedMembers | None,
) -> ResolvedStatVars:
    """The wire's counts of `stat_vars` and the dates they serve.

    ``member_count`` is ``len(stat_vars)`` when None and can report a wider confirmed
    population. ``member_count_exact`` tells a total from a lower bound. ``served_count`` is
    the mint's own count when Search knows it (``stated``)."""
    return ResolvedStatVars(
        member_count=member_count if member_count is not None else len(stat_vars),
        member_count_exact=member_count_exact,
        served_count=stated.served_count if stated is not None else None,
        data_date_range=served_date_range(stat_vars),
    )


def bind_when_slot(slots: list[Slot], *, window: TimeWindow | None) -> list[Slot]:
    """Bind the when-axis slot to a time_window value when a window is available.

    No-op when window is None. Slot order is preserved.
    """
    if window is None:
        return slots
    result = []
    for slot in slots:
        if slot.key.axis == "when":
            sv = SlotValue(ref=None, value_kind="time_window", time_window=window)
            slot = slot.model_copy(update={"binding": BindingValue(value=sv)})
        result.append(slot)
    return result


def elected_sources(sources: list[ProvenanceRef], identity: BaseIdentity) -> list[ProvenanceRef]:
    """Narrow an answer's sources to the dataset elected for its family.

    A family the index resolves holds one dataset's members (`identity_dataset`), and its
    read serves that dataset's facets alone (`read_plan.in_elected_datasets`), so the
    answer names that dataset's ref among `sources`, under its declared label, the label a
    bound source slot carries. The ref keeps its source and Data Commons. An answer that
    served nothing from the dataset names no source. `sources` is returned unchanged when
    the index elects no dataset.
    """
    dataset = identity_dataset(identity)
    if dataset is None:
        return sources
    observed = next((source for source in sources if source.dcid == dataset), None)
    if observed is None:
        return []
    return [observed.model_copy(update={"label": get_index().datasets[dataset].label})]


def shape_data_commons(members: Iterable[str]) -> list[DataCommonsRef]:
    """Every Data Commons the bound origin catalogue says holds data for one of
    `members`, in wire order."""
    catalogue = current_catalogue()
    held = {origin.id for member in members for origin in catalogue.data_commons(member)}
    return [
        DataCommonsRef(id=origin.id, label=origin.label)
        for origin in ORIGINS.values()
        if origin.id in held
    ]


def _shape_upstream(upstream: UpstreamSource | None) -> ShapeUpstream | None:
    if upstream is None:
        return None
    return ShapeUpstream(
        source=SourceRef(ref=upstream.source.ref, label=upstream.source.label),
        data_commons=DataCommonsRef(id=upstream.data_commons.id, label=upstream.data_commons.label),
    )


def build_shape_model(
    shape_draft: ShapeDraft,
    slot_keys: list[SlotKey],
    five_tuple_refs: dict[str, GraphRef],
    member_count: int,
    *,
    members: Sequence[str] | None,
    suppress_unresolved_population_type_ref: bool = False,
) -> Shape:
    """Build a grounded Shape, using the caller's member count. `build_spec` states its
    `member_digest` (`stated_members`).

    ``members`` are the StatVars the answer serves. ``data_commons`` lists the origins
    holding data for one of them, and ``upstream`` names the one source all of them come
    through (`derive.upstream_source`). ``None`` marks a shape built before materialise,
    whose members are not yet known, and states neither.
    ``suppress_unresolved_population_type_ref`` omits an unlabelled population type
    in the pre-bind shape event, where a dcid shown as a label would read as a confirmed name.
    """

    measure_value_labels = {row.dcid: row.label for row in get_measure_index().values}

    def _tuple_ref(dcid: str | None) -> GraphRef | None:
        """Return a tuple ref, or None when the SV leaves the component unasserted.

        Missing graph labels use the packaged measure index for declared axes. Other
        dcids fall back to their own text. No ref is fabricated for absent arcs.
        """
        if not dcid:
            return None
        if ref := five_tuple_refs.get(dcid):
            return ref
        if (measure_label := measure_value_labels.get(dcid)) is not None:
            return GraphRef(dcid=dcid, label=measure_label)
        return GraphRef(dcid=dcid, label=dcid)

    pop_ref = (
        None
        if suppress_unresolved_population_type_ref
        and shape_draft.pop_type_dcid not in five_tuple_refs
        else _tuple_ref(shape_draft.pop_type_dcid)
    )
    meas_ref = _tuple_ref(shape_draft.meas_prop_dcid)
    stat_ref = _tuple_ref(shape_draft.stat_type_dcid)
    qual_ref = _tuple_ref(shape_draft.meas_qual_dcid)
    denom_ref = _tuple_ref(shape_draft.meas_denom_dcid)

    # Ratio suffixes fall outside sibling construction's token-per-constraint grammar.
    # Refinement is disabled for those shapes.
    is_standard_ratio = any(is_ratio_backed_sv(sv) for sv in (shape_draft.sv_arc_facts or {}))
    refine_supported = not is_standard_ratio

    return Shape(
        shape_id=shape_draft.shape_id,
        label=shape_draft.label,
        population_type=pop_ref,
        measured_property=meas_ref,
        stat_type=stat_ref,
        measurement_qualifier=qual_ref,
        measurement_denominator=denom_ref,
        measurement_denominator_unresolved=shape_draft.meas_denom_unresolved,
        slot_keys=slot_keys,
        member_count=member_count,
        refine_supported=refine_supported,
        data_commons=shape_data_commons(members or ()),
        upstream=_shape_upstream(upstream_source(members or ())),
    )


def identity_shape_from_wire(shape: Shape) -> dict:
    """Build the identity shape member from the grounded Shape's dcids."""
    return identity_shape_for(
        pop_type_dcid=shape.population_type.dcid if shape.population_type else None,
        meas_prop_dcid=shape.measured_property.dcid if shape.measured_property else None,
        stat_type_dcid=shape.stat_type.dcid if shape.stat_type else None,
        meas_qual_dcid=shape.measurement_qualifier.dcid if shape.measurement_qualifier else None,
        meas_denom_dcid=shape.measurement_denominator.dcid
        if shape.measurement_denominator
        else None,
        meas_denom_unresolved=shape.measurement_denominator_unresolved,
    )


def build_stat_vars(
    sv_refs: list[GraphRef],
    shape_id: str,
    slots: list[Slot],
    *,
    plan: ReadPlan | None = None,
    destination_confirmed: set[str] | None = None,
    arcs_by_sv: Mapping[str, dict | None] | None = None,
    member_facts: Mapping[str, MemberFacts] | None = None,
) -> list[StatVar]:
    """Build StatVars with the bound what/how values each member asserts and every
    constraint property it asserts.

    Candidate values come from slot bindings and are retained only when the member's
    own constraints match. when/source carry observation context, while where names an
    entity relationship. Neither becomes a StatVar slot value. Exactly one of
    ``arcs_by_sv`` or ``member_facts`` is required, or ValueError is raised. A StatVar
    with no entry there, such as a member outside a capped recall sample, or with an
    empty arcs mapping, carries no slot values and null ``constraint_properties``,
    since an empty list would read as the family's root. A non-empty arcs mapping
    without a ``constraintProperties`` arc serves ``[]``: the graph read lists only
    the arcs a node has, so the wire cannot tell a dropped arc from an absent one.
    ``plan`` is the answer's read plan under the request's facet filter: each StatVar's
    date range spans the dates the plan serves for it (``read_plan.served_dates``), and a
    StatVar the plan serves nothing of, or an answer with no plan, carries none.
    ``destination_confirmed`` populates destination
    confirmation.
    """
    if (arcs_by_sv is None) == (member_facts is None):
        raise ValueError("build_stat_vars requires exactly one of arcs_by_sv or member_facts")
    candidate_slot_values: list[tuple[SlotKey, SlotValue]] = []
    for slot in slots:
        if slot.key.axis in ("when", "source", "where"):
            continue
        binding = slot.binding
        if isinstance(binding, BindingValue):
            candidate_slot_values.append((slot.key, binding.value))
        elif isinstance(binding, BindingSet):
            candidate_slot_values.extend((slot.key, sv) for sv in binding.values)

    served = served_dates(plan) if plan is not None else {}
    stat_vars: list[StatVar] = []
    for sv_ref in sv_refs:
        date_range: DateRange | None = None
        confirmed_at_destination: bool | None = None

        if (dates := served.get(sv_ref.dcid)) is not None:
            date_range = DateRange(start=dates[0], end=dates[1])

        if destination_confirmed is not None:
            confirmed_at_destination = sv_ref.dcid in destination_confirmed

        if member_facts is not None:
            sv_member = member_facts.get(sv_ref.dcid)
        else:
            sv_arcs = (arcs_by_sv or {}).get(sv_ref.dcid)
            sv_member = ArcsMemberFacts(sv_arcs) if sv_arcs is not None else None
        sv_constraints = sv_member.read_constraints() if sv_member is not None else {}
        asserted = asserted_constraint_properties(sv_member) if sv_member is not None else None
        sv_constraint_properties = sorted(asserted) if asserted is not None else None
        sv_slot_values = [
            StatVarSlotValue(key=key, value=value)
            for key, value in candidate_slot_values
            if key.property is not None
            and value.ref is not None
            and sv_constraints.get(key.property.dcid) == value.ref.dcid
        ]

        stat_vars.append(
            StatVar(
                ref=sv_ref,
                shape_id=shape_id,
                slot_values=sv_slot_values,
                constraint_properties=sv_constraint_properties,
                data_date_range=date_range,
                data_confirmed_at_destination=confirmed_at_destination,
            )
        )

    return stat_vars


def _menu_siblings(
    partition: MeasurePartition, siblings: Sequence[MeasurePartition]
) -> tuple[list[MeasurePartition], list[tuple[MeasurePartition, str]]]:
    """Split `siblings` into the measure menu and the unit token's price-basis options.

    A sibling that differs from `partition` only in price basis is a unit option, paired
    with its price-basis value. The others form the measure menu, one entry per measure
    whatever its price basis. Of two siblings that differ only in price basis, the one at
    `partition`'s price basis stays, else the first in `siblings` order.
    """
    elected_basis = dict(partition.values).get(PRICE_BASIS_AXIS)
    menu: dict[tuple[tuple[str, str | None], ...], MeasurePartition] = {}
    unit_options: list[tuple[MeasurePartition, str]] = []
    for sibling in siblings:
        if (basis := price_basis_sibling(partition, sibling)) is not None:
            unit_options.append((sibling, basis))
            continue
        measure = measure_without_price_basis(sibling)
        kept = menu.get(measure)
        if kept is None or (
            dict(sibling.values).get(PRICE_BASIS_AXIS) == elected_basis
            and dict(kept.values).get(PRICE_BASIS_AXIS) != elected_basis
        ):
            menu[measure] = sibling
    return list(menu.values()), unit_options


def _measure_alternatives(
    identity: BaseIdentity,
    partition: MeasurePartition | None,
    siblings: Sequence[MeasurePartition],
    served_rows: Callable[[BaseIdentity], ServedRows | None] | None = None,
) -> tuple[list[SpecAlternativeMeasure] | None, MeasureLabel]:
    """Publish alternatives and labels for the partition the served rows actually use.

    ``partition`` is None when no single measure partition describes the served rows,
    and the result is then ``(None, NO_MEASURE_LABEL)``. Siblings without authored labels are
    omitted. A sibling that differs only in price basis is published with axis "unit"
    and labelled by its prices; the rest are the measure menu, axis "what" (see
    ``_menu_siblings``). The main label is `measure_label_for_partition`'s. Each
    alternative carries its own identity definition so resubmission can recover that
    measure family. ``served_rows`` counts the rows a resubmit of a sibling's identity
    would serve (`option_coverage.measure_served_counter`), and omits a sibling it counts
    none for.
    """
    from search.models import (
        SpecAlternativeMeasure,
        SpecDefinition,
    )

    if partition is None:
        return None, NO_MEASURE_LABEL
    measure_index = get_measure_index()
    axes, labels, values = measure_index.axes, measure_index.partition_labels, measure_index.values
    menu, unit_options = _menu_siblings(partition, siblings)
    alternatives: list[SpecAlternativeMeasure] = []
    entries: list[tuple[MeasurePartition, Literal["what", "unit"], str | None]] = [
        *((sibling, "what", None) for sibling in menu),
        *((sibling, "unit", basis) for sibling, basis in unit_options),
    ]
    for sibling_partition, axis, basis in entries:
        sibling = mint_sibling_identity(identity, sibling_partition, axes, labels)
        if sibling.label is None:
            continue
        counted = served_rows(sibling.identity) if served_rows is not None else None
        if counted is not None and counted.rows == 0:
            continue
        alternatives.append(
            SpecAlternativeMeasure(
                definition=SpecDefinition(spec_id=sibling.spec_id, identity=sibling.identity),
                label=(
                    compose_measure_label(sibling.label, values)
                    if basis is None
                    else price_basis_option_label(PRICES_BY_QUALIFIER[basis])
                ),
                axis=axis,
                measure_axis_values=measure_axis_values(sibling_partition),
                served_rows=counted.rows if counted is not None else None,
                served_rows_exact=counted.exact if counted is not None else None,
            )
        )
    return alternatives, measure_label_for_partition(partition)


class MeasureLabel(NamedTuple):
    """A measure partition's composed label (``ResolvedSpec.measure_label``), the
    partition axes it states (``measure_label_axes``) and the code pairs its authored
    label names (``measure_label_named_values``)."""

    text: str | None
    axes: list[str]
    named_values: tuple[tuple[str, str], ...]


NO_MEASURE_LABEL = MeasureLabel(None, [], ())


def measure_label_for_partition(partition: MeasurePartition | None) -> MeasureLabel:
    """Return the partition's composed label, or `NO_MEASURE_LABEL` without one.

    The label names no price basis (``compose_measure_label``), whatever unit the request
    asks for, and the byline's unit token states it. The axes list every partition axis
    with a value, the price-basis axis included, so no how clause repeats the native
    value. The named values are the matched label's `MeasurePartitionLabel.named_values`.
    """
    if partition is None:
        return NO_MEASURE_LABEL
    measure_index = get_measure_index()
    partition_match = label_for_pairs(partition.values, measure_index.partition_labels)
    if partition_match is None:
        return NO_MEASURE_LABEL
    return MeasureLabel(
        compose_measure_label(partition_match, measure_index.values),
        [axis for axis, value in partition.values if value is not None],
        partition_match.named_values,
    )


def measure_axis_values(partition: MeasurePartition) -> list[MeasureAxisValue]:
    """Return each axis value the served partition holds, labelled from the measure index.

    The order follows ``measure_label_axes``. A value the measure index never declared
    raises ``KeyError``, as in ``compose_measure_label``.
    """
    labels = {value.dcid: value.label for value in get_measure_index().values}
    return [
        MeasureAxisValue(axis=axis, value=GraphRef(dcid=value, label=labels[value]))
        for axis, value in partition.values
        if value is not None
    ]


def _open_axis_facts(
    axes: Sequence[OpenAxisFact],
) -> list[OpenConstraintAxisFact] | None:
    """Project open constraint axes. Return None when there are no axes.

    An axis appears even with no declared values. A single declared value is
    published only for constant axes. Every declared value is published with its
    member count, highest first. Labels come from the code index, then the graph
    name a served member's arc carries, then the dcid.
    """
    if not axes:
        return None
    index = get_index()

    def _value_ref(axis: OpenAxisFact, dcid: str) -> GraphRef:
        code_row = index.code_by_dcid(dcid)
        label = code_row.label if code_row is not None else axis.value_names.get(dcid, dcid)
        return GraphRef(dcid=dcid, label=label)

    facts: list[OpenConstraintAxisFact] = []
    for axis in axes:
        varies = len(axis.values) > 1
        counts = axis.value_counts()
        value = None
        if not varies and counts:
            (only_value,) = counts
            value = _value_ref(axis, only_value)
        facts.append(
            OpenConstraintAxisFact(
                property=axis.property,
                varies=varies,
                value=value,
                distinct_count=len(counts),
                undeclared_count=axis.undeclared_count,
                values=[
                    OpenAxisValueCount(value=_value_ref(axis, dcid), count=count)
                    for dcid, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                ],
            )
        )
    return facts


@dataclass(frozen=True, slots=True)
class RoleAxisFactInput:
    """Input for an open directional axis from destination confirmation or role-scope reads.

    ``member_dcids`` is None when deriving the population would require another graph
    read.
    """

    role: RoleAxis
    member_dcids: frozenset[str] | None


def _role_axis_facts(
    facts: Sequence[RoleAxisFactInput],
) -> list[RoleAxisFact] | None:
    """Project open role axes, returning None when no facts were supplied.

    A missing member population yields ``member_count=None`` and
    ``members_available=False``.
    """
    if not facts:
        return None
    return [
        RoleAxisFact(
            role=fact.role,
            member_count=len(fact.member_dcids) if fact.member_dcids is not None else None,
            members_available=fact.member_dcids is not None,
        )
        for fact in facts
    ]


def _compose_wire_shape_label(
    measure_label: str | None,
    measure_label_axes: Sequence[str],
    slots: Sequence[Slot],
    open_axis_facts: Sequence[OpenConstraintAxisFact] | None,
) -> str | None:
    """Compose a label from properties the served group actually shares.

    An authored measure label supplies the head. Single bound values and constant
    open-axis values add qualifiers unless the label already covers their axes.
    Without a measure label, qualifiers form the
    whole label. Varying axes and set bindings are omitted, and duplicate label text
    is removed case-insensitively.
    """
    covered = set(measure_label_axes)
    qualifiers: list[str] = []
    seen_text: set[str] = set()

    def _add_qualifier(label: str) -> None:
        folded = label.casefold()
        if folded not in seen_text:
            qualifiers.append(label)
            seen_text.add(folded)

    for slot in slots:
        prop = slot.key.property
        if prop is None or prop.dcid in covered:
            continue
        binding = slot.binding
        if isinstance(binding, BindingValue) and binding.value.ref is not None:
            covered.add(prop.dcid)
            _add_qualifier(binding.value.ref.label)
    for fact in open_axis_facts or ():
        if fact.property in covered or fact.varies or fact.value is None:
            continue
        covered.add(fact.property)
        _add_qualifier(fact.value.label)
    if not qualifiers:
        return measure_label
    joined = ", ".join(qualifiers)
    if measure_label is None:
        return joined
    if measure_label.endswith(")"):
        return f"{measure_label[:-1]}, {joined})"
    return f"{measure_label} ({joined})"


def _check_provenance_disjoint(
    text_recognized_properties: frozenset[str],
    measure_elected_properties: frozenset[str],
    defaulted_properties: frozenset[str],
) -> None:
    """Require disjoint provenance sets. Overlap makes a slot's trace ambiguous."""
    overlap = text_recognized_properties & measure_elected_properties
    if overlap:
        raise ValueError(
            "text_recognized_properties and measure_elected_properties overlap "
            f"on {sorted(overlap)!r}; the sets are disjoint by construction and "
            "an overlap is a caller bug"
        )
    overlap = defaulted_properties & (text_recognized_properties | measure_elected_properties)
    if overlap:
        raise ValueError(
            "defaulted_properties overlaps text_recognized_properties or "
            f"measure_elected_properties on {sorted(overlap)!r}; the three sets "
            "are disjoint by construction and an overlap is a caller bug"
        )


def _wire_shape_label(
    shape: Shape,
    *,
    measure_partition: MeasurePartition | None,
    measure_label: str | None,
    measure_label_axes: Sequence[str],
    slots: list[Slot],
    resolved_open_axis_facts: list[OpenConstraintAxisFact] | None,
    stat_vars: list[StatVar],
) -> Shape:
    """Return the served-group label. Fall back to a confirmed member's name when empty.

    Without one measure partition, return ``shape`` unchanged.
    """
    if measure_partition is None:
        return shape
    composed_label = _compose_wire_shape_label(
        measure_label, measure_label_axes, slots, resolved_open_axis_facts
    )
    if composed_label is not None:
        return shape.model_copy(update={"label": composed_label})
    partition_members = set(measure_partition.members)
    served_donor = next((sv for sv in stat_vars if sv.ref.dcid in partition_members), None)
    if served_donor is not None:
        return shape.model_copy(update={"label": served_donor.ref.label})
    return shape


def _build_slot_filters(
    slots: list[Slot],
    *,
    text_recognized_properties: frozenset[str],
    measure_elected_properties: frozenset[str],
    defaulted_properties: frozenset[str],
) -> list[ResolvedFilter]:
    """Build slot filters with refs and provenance."""
    slot_filters: list[ResolvedFilter] = []
    for slot in slots:
        binding = slot.binding
        refs: list[GraphRef] = []
        if isinstance(binding, BindingValue) and binding.value.ref:
            refs = [binding.value.ref]
        elif isinstance(binding, BindingSet):
            refs = [sv.ref for sv in binding.values if sv.ref]

        kind: BindingKind = binding.kind  # type: ignore[assignment]
        prop_dcid = slot.key.property.dcid if slot.key.property is not None else None
        provenance: BindingProvenance
        if prop_dcid is not None and prop_dcid in text_recognized_properties:
            provenance = "text_recognized"
        elif prop_dcid is not None and prop_dcid in measure_elected_properties:
            provenance = "measure_elected"
        elif prop_dcid is not None and prop_dcid in defaulted_properties:
            provenance = "declared_default"
        else:
            provenance = "binder"

        slot_filters.append(
            ResolvedFilter(key=slot.key, binding_kind=kind, refs=refs, provenance=provenance)
        )
    return slot_filters


def build_spec(
    shape: Shape,
    slots: list[Slot],
    stat_vars: list[StatVar],
    entities: list[Entity],
    coverage: Coverage,
    *,
    headline: Headline | None = None,
    coverage_stamp: str | None = None,
    resolved_sources: list[ProvenanceRef] | None = None,
    available_at_place: bool | None = None,
    confidence: Confidence | None = None,
    slot_options: list[SlotOptions] | None = None,
    entity_options: list[EntityOptions] | None = None,
    facet_options: list[FacetOptions] | None = None,
    observations: SpecObservations | None = None,
    requested_window: RequestedWindow | None = None,
    identity_entities: list[Entity] | None = None,
    open_roles: Collection[RoleAxis],
    inapplicable_slots: list[str] | None = None,
    unresolved_slots: list[str] | None = None,
    rendered_sentence: str | None = None,
    sentence_plan: list[SentenceToken] | None = None,
    facet_bindings: dict[str, IdentityFacetBinding] | None = None,
    facets: dict[str, ResolvedFacet] | None = None,
    equivalent_readings: list[Spec] | None = None,
    scope: ResolvedScope | None = None,
    text_recognized_properties: frozenset[str] = frozenset(),
    query_qualifiers: list[ResolvedQualifier] | None = None,
    default_reading: TraceDefaultReading | None = None,
    measure_elected_properties: frozenset[str],
    defaulted_properties: frozenset[str] = frozenset(),
    measure_partition: MeasurePartition | None = None,
    measure_siblings: Sequence[MeasurePartition] = (),
    open_axis_facts: Sequence[OpenAxisFact] = (),
    role_axis_facts: Sequence[RoleAxisFactInput] = (),
    stat_vars_member_count: int | None = None,
    stat_vars_member_count_exact: bool = True,
    walked: WalkedMembers | None = None,
    plan: ReadPlan | None = None,
    named_groups: Iterable[NamedGroup] = (),
    coverage_index: CoverageIndex | None = None,
    measure_served_rows: Callable[[BaseIdentity], ServedRows | None] | None = None,
) -> Spec:
    """Assemble a Spec with a deterministic spec_id.

    Args:
        plan: The answer's read plan under the request's facet filter. The entities among
            ``entities`` that it was asked to read and serves no point at on their role
            (``read_plan.unserved_entities``) become ``ResolvedSpec.unserved_entities``,
            and the years its served series hold over every date
            (``read_plan.data_years``) become ``ResolvedSpec.data_years``. The part's
            departures state both. They never feed identity.
        coverage_index: The request's coverage index, which ``ResolvedSpec.download``
            counts a role-bearing plan from (``download.download_facts``). Without it, or
            without a plan, the Spec states no download facts.
        requested_window: The caller's requested date window, separate from the
            coverage-bound when slot. It is hashed and echoed for resubmission.
        identity_entities: Optional hash-only entity list that preserves directional
            identity when displayed entities collapse origin and destination.
        open_roles: The flow axes the identity asks for member by member, none bound by
            the hashed entities.
        named_groups: The groups the request named. A role whose identity entities are
            exactly one group's members carries that group on
            ``ResolvedSpec.entity_groups`` (`role_entity_groups`).
        facet_bindings: Measurement-method and unit bindings included in identity.
        inapplicable_slots and unresolved_slots: Graph constraint facts reported in
            ``ResolvedSpec`` and excluded from identity.
        text_recognized_properties, measure_elected_properties, defaulted_properties:
            Slot provenance sources. They must be disjoint.
        measure_partition and measure_siblings: Served partition and alternatives for
            the response label.
        measure_served_rows: Counts the rows a resubmit of each alternative measure would
            serve (`SpecAlternativeMeasure.served_rows`). An alternative counted at none
            is omitted. None leaves the alternatives uncounted.
        stat_vars: The served members. ``ResolvedSpec.members`` holds them all, and
            ``ResolvedSpec.sentence_facts`` what the renderer reads of them. The wire
            states only the date range they serve (``ResolvedStatVars.data_date_range``).
        stat_vars_member_count and stat_vars_member_count_exact: Population count and
            whether it is exact or a lower bound.
        walked: What the mint states of the members Search's walk settled for the query it
            walked. The Spec states the mint's served count and family digest
            (``stated_members``) for an identity the code index decides and one ``walked``
            walked, and neither otherwise.
        Other optional arguments populate response metadata, including render data,
            facets, observations, scope, qualifiers, and open-axis facts. Slot provenance
            properties, scope, qualifiers, open-axis facts, and role-axis facts never feed
            identity.

    """
    from search.models import (
        Spec,
        SpecDefinition,
    )

    _check_provenance_disjoint(
        text_recognized_properties, measure_elected_properties, defaulted_properties
    )

    # spec_id and identity derive from one projection through build_spec_identity.
    hashed_entities = identity_entities if identity_entities is not None else entities
    entity_groups = role_entity_groups(named_groups, hashed_entities)
    spec_id, identity_doc = build_spec_identity(
        identity_shape_from_wire(shape),
        slots,
        entities=hashed_entities,
        open_roles=open_roles,
        requested_window=requested_window,
        explicit_facets=facet_bindings,
    )
    definition = SpecDefinition(spec_id=spec_id, identity=identity_doc)
    slots = slots_in_identity_order(slots, identity_doc)
    stated = stated_members(identity_doc, walked)
    digest = stated.digest if stated is not None else None
    if shape.member_digest != digest:
        shape = shape.model_copy(update={"member_digest": digest})

    alternative_measures, measure_label = _measure_alternatives(
        identity_doc, measure_partition, measure_siblings, measure_served_rows
    )
    elected_axis_values = (
        measure_axis_values(measure_partition)
        if measure_partition is not None and measure_label.text is not None
        else []
    )
    resolved_open_axis_facts = _open_axis_facts(open_axis_facts)
    resolved_role_axis_facts = _role_axis_facts(role_axis_facts)

    # Identity is minted before the display label is composed. See _wire_shape_label.
    wire_shape = _wire_shape_label(
        shape,
        measure_partition=measure_partition,
        measure_label=measure_label.text,
        measure_label_axes=measure_label.axes,
        slots=slots,
        resolved_open_axis_facts=resolved_open_axis_facts,
        stat_vars=stat_vars,
    )

    resolved_entity_refs = [e.ref for e in entities]
    resolved_sources = elected_sources(resolved_sources or [], identity_doc)

    slot_filters = _build_slot_filters(
        slots,
        text_recognized_properties=text_recognized_properties,
        measure_elected_properties=measure_elected_properties,
        defaulted_properties=defaulted_properties,
    )

    trace = ResolutionTrace(
        resolved_entities=resolved_entity_refs,
        resolved_sources=resolved_sources,
        slot_filters=slot_filters,
        query_qualifiers=query_qualifiers or [],
        default_reading=default_reading,
        requested_window=requested_window,
        capability_version=settings().capability_version,
    )

    provenance = (facets or {}).get("provenance")
    download = (
        download_facts(
            plan,
            coverage_index,
            requested_window=requested_window,
            conversion=counted_conversion(
                plan, identity_doc.facets, catalogue=current_conversion_catalogue()
            ),
            provenance=provenance.ref if provenance is not None else None,
        )
        if plan is not None and coverage_index is not None
        else None
    )
    unserved = unserved_entities(plan)
    return Spec(
        spec_id=spec_id,
        definition=definition,
        resolved=ResolvedSpec(
            shape=wire_shape,
            slots=apply_slot_summability(slots, identity_doc),
            inapplicable_slots=inapplicable_slots or [],
            unresolved_slots=unresolved_slots or [],
            slot_options=slot_options or [],
            entity_options=entity_options or [],
            facet_options=facet_options or [],
            stat_vars=_resolved_stat_vars(
                stat_vars,
                member_count=stat_vars_member_count,
                member_count_exact=stat_vars_member_count_exact,
                stated=stated,
            ),
            entities=_resolved_entities(entities),
            entity_groups=entity_groups or None,
            coverage=coverage,
            headline=headline,
            download=download,
            coverage_stamp=coverage_stamp,
            scope=scope,
            facets=facets or {},
            trace=trace,
            available_at_place=available_at_place,
            confidence=confidence,
            observations=observations,
            rendered_sentence=rendered_sentence,
            sentence_plan=sentence_plan,
            equivalent_readings=equivalent_readings,
            alternative_measures=alternative_measures,
            measure_label=measure_label.text,
            measure_label_axes=measure_label.axes,
            measure_axis_values=elected_axis_values,
            open_axis_facts=resolved_open_axis_facts,
            role_axis_facts=resolved_role_axis_facts,
        )
        .stating_unserved(
            tuple(
                (entity_role_key(e), e.ref.dcid)
                for e in entities
                if (entity_role_key(e), e.ref.dcid) in unserved
            )
        )
        .stating_data_years(data_years(plan))
        .stating_members(stat_vars)
        .stating_measure_label_named_values(measure_label.named_values),
    )


def build_unsupported_definition(
    shape: Mapping[str, object],
    slots: list[Slot],
    *,
    entities: list[Entity] | None = None,
    open_roles: Collection[RoleAxis] = (),
    requested_window: RequestedWindow | None = None,
    explicit_facets: dict[str, IdentityFacetBinding] | None = None,
) -> dict[str, object]:
    """Build and validate an unsupported definition without minting a spec id.

    ``open_roles`` keeps the flow axes the request asks for member by member, and
    ``explicit_facets`` a completed but unservable facet request, in the definition.
    Validation errors propagate before the payload is returned.
    """
    projection = build_identity_projection(
        shape,
        slots,
        entities=entities,
        open_roles=open_roles,
        requested_window=requested_window,
        explicit_facets=explicit_facets,
    )
    BaseIdentity.model_validate(projection)
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": projection,
    }
