"""Build pre-bind shape and post-bind sentence events for streaming resolves.

`resolve_variable` calls these builders only when `emit` is set. The first
sentence is built while materialisation runs, before the answer's members, years, unit and
sources are known: its heading states what bind elected, with no flow type an open axis
settles and no period, and it has no byline. Over a graph that can declare a default facet
the pipeline holds both sentences until the part's answer stands. A definite result emits a
final sentence under the same `variable_index`, with the full heading and its byline; a
multi-spec candidate response carries each final sentence in its own spec. The final
`spec_id` differs from the first when re-election or a single-survivor collapse swaps the Spec.
The builders add events without changing the returned `RegionResult`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from code_index import LevelOption
from code_index.members import datasets_carry
from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.events import SentenceEvent, ShapeEvent
from search.engine.extract import DateRequest
from search.engine.region.grounding import (
    EntityRoles,
    _build_entity,
    _ground_answer,
    build_identity_entities,
    requested_window_from_date_request,
)
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    apply_slot_summability,
    shape_part_datasets,
    shape_pins,
    shape_population_arcs,
    shape_query,
)
from search.engine.shape import ShapeDraft, identity_shape_for
from search.engine.spec_build import (
    NamedGroup,
    build_shape_model,
    measure_label_for_partition,
    role_entity_groups,
)
from search.engine.spec_id import build_spec_identity, slots_in_identity_order
from search.heading.pins import identity_pinned_properties
from search.models import (
    BindingUnbound,
    GraphRef,
    IdentityFacetBinding,
    IdentityFacetValueRequest,
    RoleAxis,
    Slot,
    SlotKey,
    SlotOption,
    SlotOptions,
    SlotValue,
    Spec,
)
from search.render.heading import heading_from_parts
from search.render.labels import _humanize_ref
from search.render.plan import render_sentence_plan
from search.render.tokens import plan_tokens

# Values shown per slot in the shape-skeleton event's slot_options, same order
# of magnitude as the bind prompt's own _MAX_TAXONOMY_VALUES_SHOWN (bind.py).
_SHAPE_EVENT_TAXONOMY_CAP = 15


async def _member_count(shape_draft: ShapeDraft) -> int:
    """The number of members `shape_draft` stands for (`shape_population_arcs`), the
    population the answer reads from. A held family's members are read off the event loop."""
    if shape_draft.held_family is None:
        return len(shape_draft.sv_arc_facts or {})
    return len(await asyncio.to_thread(shape_population_arcs, shape_draft))


def _empty_slots_for(slot_keys: list[SlotKey]) -> list[Slot]:
    """Build unbound slots for the pre-bind shape skeleton."""
    return [Slot(key=sk, binding=BindingUnbound()) for sk in slot_keys]


async def _build_shape_event(
    shape_draft: ShapeDraft,
    *,
    variable_index: int,
    graph: EngineGraphClient,
) -> ShapeEvent:
    """Build a best-effort, pre-bind skeleton for the top-ranked shape.

    Bind may elect another shape. Callers run this alongside bind, hiding the
    label batch call behind the LLM request. Index-sourced slots show
    classification roots without their children, for a slot whose property the index rows
    of the datasets the part may answer from carry (`ShapeDraft.datasets`, else every
    publisher of the five-tuple; `code_index.members.datasets_carry`). The final Spec
    carries the full menu. Recall-sourced slots show at most 15 sampled values.
    """
    taxonomy = shape_draft.slot_taxonomy or {}
    index_sourced = shape_draft.slot_taxonomy_index_sourced
    level_options_by_key: dict[str, tuple[LevelOption, ...]] = {}
    option_dcids: set[str] = set()
    if index_sourced:
        index = get_index()
        five_tuple = shape_query(shape_draft, (), ()).five_tuple
        datasets = shape_part_datasets(shape_draft, index)
        for skd in shape_draft.slot_keys:
            if skd.property_dcid is None:
                continue
            key = f"{skd.axis}:{skd.property_dcid}"
            if not datasets_carry(index, datasets, five_tuple, skd.property_dcid):
                continue
            opts = tuple(
                o
                for o in index.level_scoped_options(skd.property_dcid, None)
                if o.relation == "root"
            )
            level_options_by_key[key] = opts
            option_dcids.update(o.dcid for o in opts)
    else:
        for values in taxonomy.values():
            option_dcids.update(values[:_SHAPE_EVENT_TAXONOMY_CAP])
    labels = (
        await asyncio.to_thread(graph.node_labels_batch, list(option_dcids)) if option_dcids else {}
    )

    slot_keys: list[SlotKey] = []
    slot_options: list[SlotOptions] = []
    for skd in shape_draft.slot_keys:
        prop_ref = GraphRef(ref=skd.property_dcid, label=skd.label) if skd.property_dcid else None
        slot_label = (
            _humanize_ref(skd.property_dcid, skd.label)
            if skd.property_dcid is not None
            else skd.label
        )
        sk = SlotKey(axis=skd.axis, property=prop_ref, label=slot_label)
        slot_keys.append(sk)
        if skd.property_dcid is None:
            continue
        key = f"{skd.axis}:{skd.property_dcid}"
        value_kind = "entity" if skd.axis == "where" else "enum_value"
        if index_sourced:
            options = [
                SlotOption(
                    value=SlotValue(
                        ref=GraphRef(ref=o.dcid, label=labels[o.dcid]), value_kind=value_kind
                    ),
                    relation=o.relation,
                    member_count=o.member_count,
                )
                for o in level_options_by_key.get(key, ())[:_SHAPE_EVENT_TAXONOMY_CAP]
                if o.dcid in labels
            ]
        else:
            dcids = taxonomy.get(key, [])[:_SHAPE_EVENT_TAXONOMY_CAP]
            options = [
                SlotOption(
                    value=SlotValue(ref=GraphRef(ref=d, label=labels[d]), value_kind=value_kind)
                )
                for d in dcids
                if d in labels
            ]
        if options:
            slot_options.append(
                SlotOptions(
                    key=sk,
                    options=options,
                    truncated=True,
                    source="observed",
                )
            )

    slots = _empty_slots_for(slot_keys)
    member_count = await _member_count(shape_draft)
    # Pre-bind: no confirmed five-tuple ref lookup has run yet. Do not project
    # an unresolved population-type dcid into the sentence plan: it can otherwise
    # be split and title-cased as if it were a confirmed label. Other five-tuple
    # fallbacks remain useful (for example, ``fertilityRate`` becomes "Fertility
    # rate"). The final `sentence` event carries confirmed component labels.
    minimal_shape = build_shape_model(
        shape_draft,
        slot_keys,
        five_tuple_refs={},
        member_count=member_count,
        members=None,
        suppress_unresolved_population_type_ref=True,
    )
    plan = plan_tokens(
        heading_from_parts(
            slots,
            [],
            minimal_shape,
            window=None,
            pinned_properties=frozenset(shape_pins(shape_draft)),
        )
    )

    return ShapeEvent(
        variable_index=variable_index,
        shape_id=minimal_shape.shape_id,
        label=minimal_shape.label,
        slot_keys=slot_keys,
        sentence_plan=plan,
        slot_options=slot_options,
        refine_supported=minimal_shape.refine_supported,
    )


async def _build_sentence_event(
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    roles: EntityRoles,
    *,
    variable_index: int,
    date_request: DateRequest | None,
    graph: EngineGraphClient,
    open_axes: frozenset[RoleAxis],
    explicit_facets: Mapping[str, IdentityFacetBinding] | None = None,
    query_entity_text_by_dcid: Mapping[str, str] | None = None,
    measure_partition: MeasurePartition | None = None,
    named_groups: Sequence[NamedGroup] = (),
) -> SentenceEvent:
    """Build the sentence after binding and before materialisation completes.

    `_ground_answer` supplies slots and entities. With no StatVars, facets or open-axis
    facts yet, the heading has no period and no flow type an open axis settles, and the
    byline is empty. This work runs alongside materialisation, and the final event fills
    them. A role whose entities are exactly a group the query named
    (``named_groups``, `role_entity_groups`) reads by the group's name, as in the final
    sentence.

    `measure_partition` supplies the bind-time label through the same helper
    used by the final Spec, and its members' StatVar dcids the flow family,
    so the heading words and orders its roles as the final one does. Materialised members
    may elect another partition.
    Identity uses the requested date window, grounded entities, and the same
    `explicit_facets` as `build_spec`, so ids agree when this shape survives.
    `open_axes` are the flow axes the part asks for member by member, which grounding reads.
    """
    grounding = await asyncio.to_thread(
        _ground_answer,
        shape_draft,
        bindings,
        [],
        roles,
        graph,
        None,
        query_entity_text_by_dcid=query_entity_text_by_dcid,
        open_axes=open_axes,
    )
    (
        five_tuple_refs,
        _role_refs,
        slots,
        slot_key_models,
        _sv_refs,
        entity_data,
        _source_refs,
        _slot_options,
        _inapplicable_slots,
        _,  # warnings: discarded, SentenceEvent has no warnings field
        _labels,
    ) = grounding
    member_count = await _member_count(shape_draft)
    shape_model = build_shape_model(
        shape_draft,
        slot_key_models,
        five_tuple_refs,
        member_count=member_count,
        members=None,
    )
    entities = [_build_entity(rd, er, etr, _role_refs) for rd, er, etr in entity_data]
    # The `when` slot stays unbound: the years an answer serves are known only after
    # materialise, and an answer that ends `no_data` or refused sends no final sentence,
    # so a requested window stated here could be the last one a client holds.
    # Identity hashes the requested window, not the coverage-resolved window.
    # Build identity entities with the same role dispatch as build_spec.
    identity_entities = build_identity_entities(entity_data, _role_refs)
    entity_groups = role_entity_groups(named_groups, identity_entities)
    spec_id, identity = build_spec_identity(
        identity_shape_for(
            pop_type_dcid=shape_draft.pop_type_dcid,
            meas_prop_dcid=shape_draft.meas_prop_dcid,
            stat_type_dcid=shape_draft.stat_type_dcid,
            meas_qual_dcid=shape_draft.meas_qual_dcid,
            meas_denom_dcid=shape_draft.meas_denom_dcid,
            meas_denom_unresolved=shape_draft.meas_denom_unresolved,
        ),
        slots,
        entities=identity_entities,
        # The final identity states the open axes its elected members declare. Until
        # materialise elects them, the axes the part leaves open stand in, and the final
        # sentence event replaces this one.
        open_roles=open_axes,
        requested_window=requested_window_from_date_request(date_request),
        explicit_facets=explicit_facets,
    )
    slots = apply_slot_summability(slots_in_identity_order(slots, identity), identity)
    measure_label = measure_label_for_partition(measure_partition)
    unit_binding = (explicit_facets or {}).get("unit")
    plan = plan_tokens(
        heading_from_parts(
            slots,
            entities,
            shape_model,
            None,
            pinned_properties=identity_pinned_properties(identity),
            measure_label=measure_label.text,
            measure_label_axes=measure_label.axes,
            measure_label_named_values=measure_label.named_values,
            unit_request=(
                unit_binding.request
                if isinstance(unit_binding, IdentityFacetValueRequest)
                else None
            ),
            entity_groups=entity_groups,
            member_dcids=measure_partition.members if measure_partition is not None else (),
        )
    )
    return SentenceEvent(
        variable_index=variable_index,
        spec_id=spec_id,
        sentence_plan=plan,
        slots=slots,
    )


def _build_definite_sentence_event(spec: Spec, *, variable_index: int) -> SentenceEvent:
    """Build the final sentence event for a definite Spec, with its heading and byline.

    The rendered plan matches the terminal response event's plan. The warnings an election
    raises (`slot_defaulted`, `family_total_absent`) stay on the part and out of the
    sentence, so a resubmit, which elects nothing, states the same sentence. A single
    surviving Spec emits this event after assembly; its `spec_id` names the
    winner, which may differ from the bind-time shape after re-election or
    candidate collapse. Streaming clients replace the first event by
    `variable_index`. Multi-spec candidate responses carry sentences per spec.
    """
    return SentenceEvent(
        variable_index=variable_index,
        spec_id=spec.spec_id,
        sentence_plan=render_sentence_plan(spec),
        slots=spec.resolved.slots,
    )
