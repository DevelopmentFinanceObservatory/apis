"""Build a part's shape and sentence events for streaming resolves.

The shape event goes out while bind runs and carries the top admissible reading, marked
provisional. A resolution that runs sends each part's one sentence event just ahead of its
response, built from the response's own part (`answer_events`), so it carries the
`spec_id` the response states. A response served from the response cache describes a
resolution that did not run and goes out without them. Neither changes the returned
`RegionResult`.
"""

from __future__ import annotations

import asyncio

from code_index import LevelOption
from code_index.members import datasets_carry
from code_index.shared import get_index
from search.engine.clients.graph import EngineGraphClient
from search.engine.events import ResponseEvent, SentenceEvent, ShapeEvent
from search.engine.retrieval.bound_values import (
    shape_part_datasets,
    shape_pins,
    shape_population_arcs,
    shape_query,
)
from search.engine.shape import ShapeDraft
from search.engine.spec_build import build_shape_model
from search.models import (
    BindingUnbound,
    CandidatesPart,
    DefinitePart,
    GraphRef,
    NoDataPart,
    Part,
    Resolution,
    ResolveResponse,
    Slot,
    SlotKey,
    SlotOption,
    SlotOptions,
    SlotValue,
    Spec,
)
from search.render.heading import heading_from_parts
from search.render.labels import _humanize_ref
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


def _answer_spec(part: Part) -> Spec | None:
    """The Spec a part answers with: a definite part's Spec, a candidates part's first
    candidate, or the flagged match a no-data part carries. `None` for any other part."""
    if isinstance(part, DefinitePart):
        return part.spec
    if isinstance(part, CandidatesPart):
        return part.candidates.specs[0]
    if isinstance(part, NoDataPart):
        return part.no_data.flagged_match
    return None


def response_sentence_events(response: ResolveResponse) -> list[SentenceEvent]:
    """The one sentence event of each part of `response`, in part order: the sentence plan
    of the Spec the part answers with (`_answer_spec`). A part with no such Spec sends none.

    Built from the response itself, so each event's `spec_id` and plan are the ones the
    response states for its part, after the conjunction's same-shape merge has minted the
    Spec two parts share. Every Spec a part shows carries its plan
    (`ResolvedSpec.sentence_plan`)."""
    if not isinstance(response, Resolution):
        return []
    events: list[SentenceEvent] = []
    for index, part in enumerate(response.parts):
        spec = _answer_spec(part)
        if spec is not None and spec.resolved.sentence_plan is not None:
            events.append(
                SentenceEvent(
                    variable_index=index,
                    spec_id=spec.spec_id,
                    sentence_plan=spec.resolved.sentence_plan,
                    slots=spec.resolved.slots,
                )
            )
    return events


def answer_events(response: ResolveResponse) -> list[SentenceEvent | ResponseEvent]:
    """The events a resolution that ran sends once its response stands: each part's one
    sentence (`response_sentence_events`), then the response."""
    return [*response_sentence_events(response), ResponseEvent(response=response)]
