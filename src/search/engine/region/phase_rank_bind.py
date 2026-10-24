"""Rank shapes, admit query qualifiers, and run the universal choose-and-bind call."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from functools import cached_property
from typing import Literal, NamedTuple

from code_index.labels import without_property_suffix
from code_index.members import ProvenanceScope, is_opt_in
from code_index.query import CodeIndex, fold_word
from code_index.shared import get_index, get_measure_index
from code_index.tokens import Token, text_matches, tokenize, word_tokens
from code_index.types import CodeRow, DatasetDeclaration, MeasureAxis, StatedWord
from origin_catalogue import current_catalogue
from search.engine.bind import (
    CandidateMenu,
    SlotBindingDraft,
    _slot_name,
    _UniversalBindOutput,
    rank_elected,
    split_stray_bindings,
)
from search.engine.bind_repair import (
    _label_has_negated_occurrence,
    _label_named_in_text,
    _label_negated_in_text,
)
from search.engine.binders import BindRequest, BindResult, QuantityRequest
from search.engine.binders.base import PolicyContractError
from search.engine.binders.house_policy import (
    POLICY_ORDER,
    Context,
    PolicyChoice,
    PolicyName,
    PolicyPool,
    ShapeFacts,
    apply_house_policies,
    chooses_on,
    context_of,
    shape_facts,
)
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read
from search.engine.departures import qualifier_property
from search.engine.envelope import _emit_step, _sum_usage, now_ms
from search.engine.errors import GraphInfraError, LLMOutputRefused
from search.engine.events import FrameEvent
from search.engine.frame import Frame, SlotDefault, with_recall
from search.engine.function_words import (
    AMOUNT_WORDS,
    COUNT_WORDS,
    DATA_WORDS,
    FLOW_VERBS,
    FUNCTION_WORDS,
    PERSON_WORDS,
    RATIO_WORDS,
)
from search.engine.interpretation import Interpretation, InterpretationSet
from search.engine.qualifier_match import (
    BREAKDOWN_FILLER_NOUNS,
    CONNECTIVES,
    UNION_CONNECTIVES,
    CodedQualifier,
    GoverningConnective,
    QualifierMatch,
    QualifierSpan,
    QualifierTie,
    TextReading,
    _all_normalized_text_ranges,
    _match_source,
    _normalize,
    _ranges_overlap,
    _row_system,
    bindable_codes,
    breakdown_spans,
    governing_connectives,
    joins,
    match_unsupported_qualifier_phrases,
    names_plain_name,
    plural_forms,
    read_by_vocabulary,
    slot_noun_named,
)
from search.engine.quantity import (
    PER_HEAD_RATE_BASE_RE,
    DenominatorKind,
    DenominatorRead,
    NamedDenominator,
    names_only_form,
    normalisation_class,
    read_denominators,
    statvar_denominators,
    with_stated_form,
)
from search.engine.quantity_read import QuantityOutput
from search.engine.region.admissibility import (
    judge_measure_and_roles,
    judge_pinned_parts,
    pinned_code_records,
)
from search.engine.region.alternate_guard import (
    leaves_price_basis_unread,
    measure_class,
    shape_measure_class,
)
from search.engine.region.availability import (
    has_data_where_asked,
    member_has_data_where_asked,
    menu_read_places,
    read_menu_availability,
    series_at_places,
)
from search.engine.region.codes import AVAILABILITY_PROBE_FAILED, DEGENERATE_BINDING_DETECTED
from search.engine.region.decisions import PartyReadingWithdrawn
from search.engine.region.grounding import _shape_with_taxonomy
from search.engine.region.place_roles import (
    PlaceAssignment,
    bind_place_roles,
    held_party_words,
    names_a_place,
    part_states,
    parties_of_kind,
    without_open_axis_gaps,
    without_unasked_axes,
)
from search.engine.region.qualifier_readings import match_query_readings, reading_for_shape
from search.engine.region.result import RegionResult
from search.engine.region.seating import is_expansion, seat_menu
from search.engine.region.sentence_events import _build_shape_event
from search.engine.region.source_choice import (
    idle_exclusion_warnings,
    scope_records,
    source_record,
    with_idle_exclusions,
)
from search.engine.region.state import (
    Requirement,
    ResolveState,
    ValueRule,
    for_shape,
    merge_lists,
    shape_vocabulary,
    unapplied_departure,
    with_requirement,
)
from search.engine.retrieval.bound_values import (
    DefaultReading,
    RoleKeys,
    applies_one_of,
    applies_value,
    bound_realizable_members,
    bound_values_from_bindings,
    collapse_measure_families,
    declared_datasets,
    elect_five_tuple_family,
    five_tuple_member_facts,
    held_family_exact,
    index_carries_shape,
    other_families_properties,
    part_datasets,
    part_member_rows,
    phrases_within,
    price_basis_order,
    query_phrases_read,
    read_held_families,
    shape_constraint_values,
    shape_datasets,
    shape_five_tuple,
    shape_for_part,
    shape_listed_rows,
    shape_population_arcs,
)
from search.engine.retrieval.facts import read_constraints, read_five_tuple, read_name
from search.engine.retrieval.standard import members_served
from search.engine.role_text import role_axis_for_text
from search.engine.shape import (
    ShapeDraft,
    SlotKeyDraft,
    counts_persons,
    is_role_bearing,
    measures_count,
    role_names_by_axis,
)
from search.engine.source_code import names_only_code
from search.engine.source_scope import SourceScope
from search.engine.time_bind import parse_time_text
from search.heading.measure_noun import denominator_noun
from search.models import (
    GraphRef,
    Operation,
    ResolvedQualifier,
    TraceDefaultReading,
    Warning,
)
from search.render.labels import _humanize_ref

logger = logging.getLogger(__name__)


def _part_codes(shape: ShapeDraft, index: CodeIndex) -> dict[str, frozenset[str]] | None:
    """Each constraint property the rows of `shape`'s five-tuple that its part's datasets
    publish carry (`part_member_rows`), mapped to the values they hold on it. `None` for a
    shape no candidacy stamped or the index does not carry, whose slot keys state its code
    systems and whose values the index does not record."""
    if shape.datasets is None:
        return None
    rows = part_member_rows(shape, index)
    if not rows:
        return None
    held: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for prop, value in row.constraint_properties.items():
            held[prop].add(value)
    return {prop: frozenset(values) for prop, values in held.items()}


def _rank_key(shape: ShapeDraft) -> tuple[bool, bool, float, int, int, str]:
    """Order our pool's shapes ahead of the UN System Data Commons', since each resolver
    scores on its own scale (`shape.Pool`), then by recall or topic-root rank, breadth,
    country breadth, then ``shape_id``.

    Unscored shapes sort last. Breadth is the confirmed recall sample. It breaks
    flat-score ties, broadest first. Country breadth (`_country_breadth`) breaks the
    ties left, so of two readings recalled on one sentence the one with data in more
    countries leads: World Bank GDP per capita before Eurostat's European regions. The
    same key selects the cross-population anchor.
    """
    return (
        shape.pool != "ours",
        shape.rank_score is None,
        -(shape.rank_score or 0.0),
        -(len(shape.sv_arc_facts) if shape.sv_arc_facts else 1),
        -_country_breadth(shape),
        shape.shape_id,
    )


def _country_breadth(shape: ShapeDraft) -> int:
    """The most `country/` places the bound catalogue lists for one of `shape`'s
    members (`Catalogue.base_places`), 0 when it lists none of them. DFO and UN System
    variables are outside the base variable table, so their shapes count 0."""
    catalogue = current_catalogue()
    return max(
        (
            sum(place.startswith("country/") for place in places)
            for sv in shape.sv_arc_facts or ()
            if (places := catalogue.base_places(sv)) is not None
        ),
        default=0,
    )


def _role_bearing_outranks(eligible_shapes: Sequence[ShapeDraft]) -> bool:
    """Whether a role-bearing shape among `eligible_shapes` outranks the subject-only ones:
    the top-ranked shape (`_rank_key`) is role-bearing, or a role-bearing shape meets the
    score floor on its own measured score. A role-bearing shape the index carries passes
    the floor whatever its score, so its presence alone says nothing of the query: two
    flows recalled at 0.61 for "GDP of Ethiopia" do not outrank GDP shapes above them."""
    if not eligible_shapes:
        return False
    floor = settings().unknown_score_threshold
    return is_role_bearing(min(eligible_shapes, key=_rank_key)) or any(
        is_role_bearing(s) and s.measured_score is not None and s.measured_score >= floor
        for s in eligible_shapes
    )


async def _read_denominators(state: ResolveState, shapes: list[ShapeDraft]) -> None:
    """Read the kind and name of every StatVar denominator the shapes divide by into
    ``state.denominator_reads``, in one batch, and hold their kinds in
    ``state.denominator_kinds`` when the leg names a form.

    The quantity read lists the denominators by name, so the model can say which one a
    question names ("debt to GDP"). Materialise and the alternate guard read the kinds: a series divided
    by `Count_Person_15OrMoreYears` is per person, one divided by GNI is not. A menu with
    no StatVar denominator reads nothing. The menu keeps every shape: an off-topic
    per-capita series must not displace the shape the query names ("teacher count per
    capita" keeps the teacher count, which refuses ``denominator_not_available`` at
    materialise).

    A leg that names no form reads the denominators for the quantity read's list alone, so
    a graph read that fails or omits a dcid (``GraphInfraError``) leaves the reads empty and
    the list without them (`_quantity_denominators`), and is held on
    ``state.denominator_read_failure``: the request answers as one that named no
    denominator would unless the quantity read states a form or names a denominator, which
    raise it
    (`_read_quantity_answer`). A leg that names a form needs the kinds to judge its shapes,
    so the failure raises.
    """
    try:
        state.denominator_reads = await asyncio.to_thread(
            read_denominators, statvar_denominators(shapes), graph=state.graph
        )
    except GraphInfraError as exc:
        if state.quantity_requirement.named:
            raise
        logger.warning(
            "denominator read failed; the quantity read lists no denominator", exc_info=True
        )
        state.denominator_reads = {}
        state.denominator_read_failure = exc
        return
    if state.quantity_requirement.named:
        state.denominator_kinds = _kinds_of(state.denominator_reads)


def _kinds_of(reads: Mapping[str, DenominatorRead]) -> dict[str, DenominatorKind]:
    return {denominator: read.kind for denominator, read in reads.items()}


def _quantity_denominators(state: ResolveState, shapes: Sequence[ShapeDraft]) -> list[GraphRef]:
    """The StatVar denominators `shapes` divide by and the graph was read for
    (`_read_denominators`), in shape order, each under its graph name, else its dcid.
    The quantity read lists them (`quantity_read.read_stated_quantity`)."""
    reads = state.denominator_reads
    return [
        GraphRef(ref=dcid, label=reads[dcid].name or dcid)
        for dcid in statvar_denominators(shapes)
        if dcid in reads
    ]


async def _read_stated_quantity(
    state: ResolveState, shapes: Sequence[ShapeDraft]
) -> RegionResult | None:
    """Read the form and the base the variable phrase states, from its own words, over the
    denominators `shapes` divide by (`binders.Binder.read_quantity`), and record them
    (`_read_quantity_answer`). Return a terminal result for a malformed reading.

    The read runs after the measure and role rules and before the form rule, so a form
    only the model reads refuses a shape as one a wording pattern names does
    (`admissibility.judge_pinned_parts`), and the bind call elects among what is left. Its
    usage opens the part's LLM usage (`ResolveState.var_usage`)."""
    denominators = _quantity_denominators(state, shapes)
    request = QuantityRequest(
        variable=_quantity_phrase(state),
        denominators=denominators,
        bind_variable=_bind_variable(state, shapes),
        query_text=state.query_text or state.role_query,
        part=state.variable_index,
        pool=state.pool,
    )
    quantity, state.var_usage = await state.binder.read_quantity(request, llm=state.llm)
    if _read_quantity_answer(state, quantity, denominators, shapes):
        return None
    await _record_shape_step(state)
    return state.no_data("variable_not_resolved", llm_usage=state.var_usage)


def _entity_spans(state: ResolveState, tokens: Sequence[Token]) -> list[tuple[int, int]]:
    """The character ranges of the variable phrase, read as `tokens`, that the part's
    entities name, by their names and by the spans their door roles carry
    (`role_text.DoorRole.span`, the query's own wording of the place): other readers own
    them ("German" in "German aid" is the origin Germany). A role's cue word ("to" in "aid
    to Kenya") stays in the phrase."""
    return [
        occurrence
        for text in (
            *state.entities,
            *(role.span for roles in state.entity_roles.values() for role in roles),
        )
        for occurrence in _token_occurrences(tokens, text)
    ]


def _quantity_phrase(state: ResolveState) -> str:
    """The variable phrase the quantity read reads: the variable less the words the part's
    entities own (`_entity_spans`), so a place named inside the phrase is never read as a
    base."""
    variable = state.variable
    words: list[str] = []
    at = 0
    for start, end in sorted(_entity_spans(state, tokenize(variable))):
        words.append(variable[at:start] if start > at else "")
        at = max(at, end)
    words.append(variable[at:])
    return " ".join("".join(words).split())


def _names_base(state: ResolveState, words: str | None, shapes: Sequence[ShapeDraft]) -> bool:
    """Whether `words`, the quantity read's quote of the base its denominator names, occur
    in the variable phrase with a word left beyond form wording and function words
    (`quantity.names_only_form`) once the spans the part's entities own (`_entity_spans`)
    and the spans the labels of `shapes`' values hold (`_value_label_spans`) are taken
    out. `shapes` are the shapes the read lists the denominators of. A base the
    model names without quoting the phrase is one the phrase does not name: "maternal
    mortality ratio" names a form, and no base. A quote of form words alone, "ratio" in
    "poverty headcount ratio" or "share" in "debt as a share of GDP", names no base
    either, while "share of GDP" names GDP. Nor does a quote of a qualifier and function
    words: "private creditors" and "to private creditors" in "debt service to private
    creditors" leave nothing or "to" outside the span of the creditor type a shape's slot
    holds. "total government spending" leaves "total spending" outside the value
    "Government", and "GDP of private creditors" leaves "GDP of", so both name a base. The
    quote matches with hyphens read as spaces in both, so "GDP" grounds in "debt-to-GDP
    ratio", which the tokeniser holds as one word. The swap keeps every offset, so the
    owned spans still apply."""
    if not words:
        return False
    tokens = tokenize(state.variable.replace("-", " "))
    owned = [
        *_entity_spans(state, tokenize(state.variable)),
        *_value_label_spans(shapes, tokens),
    ]
    return any(
        not names_only_form(
            " ".join(
                token.text
                for token in tokens
                if token.kind == "word"
                and start <= token.start
                and token.end <= end
                and not any(_ranges_overlap((token.start, token.end), span) for span in owned)
            )
        )
        for start, end in _token_occurrences(tokens, words.replace("-", " "))
    )


def _value_label_spans(
    shapes: Sequence[ShapeDraft], tokens: Sequence[Token]
) -> list[tuple[int, int]]:
    """The character ranges of the variable phrase, read as `tokens` with hyphens as
    spaces, where the graph label (`ShapeDraft.value_labels`) of a constraint value a
    member of one of `shapes` holds occurs, among the members each shape stands for
    (`shape_constraint_values`). The label is
    read without the trailing name of its property, which `value_labels` also holds and
    the graph adds to tell code lists apart (`code_index.labels.without_property_suffix`); a code-index
    label carries none. "private creditors (IDS creditor type)" owns "private creditors"
    in "debt service to private creditors", and "holders of GDP-linked bonds" owns nothing
    in "debt to GDP"."""
    spans: list[tuple[int, int]] = []
    for shape in shapes:
        labels = shape.value_labels or {}
        for prop, value in shape_constraint_values(shape):
            label = labels.get(value)
            if label is not None:
                own = without_property_suffix(label, labels.get(prop))
                spans += _token_occurrences(tokens, own.replace("-", " "))
    return spans


def _read_quantity_answer(
    state: ResolveState,
    quantity: QuantityOutput,
    denominators: list[GraphRef],
    shapes: Sequence[ShapeDraft],
) -> bool:
    """Record what the quantity read found the question asks for: the form it states,
    joined to the requirement when no phrase named one (`quantity.with_stated_form`), and
    the denominator it chose from ``denominators``, the last index meaning "none of these".
    The denominator stands only when the read quotes the phrase's words that name it
    (`_names_base`, over `shapes`, the shapes whose denominators the list holds); otherwise the phrase names no base, whatever the index. A stated
    rate or share is a division by a base, so a quote that names none drops it too:
    "debt service to private creditors" read as a rate on "private creditors", a
    qualifier, states no form. An unquoted rate or share stands, since a share of people
    quotes no base by design ("HIV prevalence among young women" reads as a share), as
    does a per-person, per-head or per-area form. Return whether the output is well
    formed: a grounded index outside the list is a malformed output, as an out-of-range
    shape index is, and the part refuses ``variable_not_resolved``. When the denominator
    read failed (``state.denominator_read_failure``), a stated form or any grounded
    index, "none of these" included, would be judged on reads that are missing, so the
    read's error is raised.
    """
    index = quantity.denominator_index
    stated_form = quantity.stated_form
    if not _names_base(state, quantity.base_words, shapes):
        if index is not None:
            logger.info(
                "quantity read names denominator_index %d on ungrounded words %r; no base read",
                index,
                quantity.base_words,
            )
            index = None
        if quantity.base_words and stated_form in ("rate", "share"):
            logger.info(
                "quantity read states %s on quoted words %r that name no base; no form read",
                stated_form,
                quantity.base_words,
            )
            stated_form = None
    failure = state.denominator_read_failure
    if failure is not None and (stated_form is not None or index is not None):
        raise failure
    requirement = with_stated_form(state.quantity_requirement, stated_form)
    if requirement != state.quantity_requirement:
        state.quantity_requirement = requirement
        state.stated_form = stated_form
        state.denominator_kinds = _kinds_of(state.denominator_reads)
    if index is None:
        return True
    base_words = quantity.base_words or ""
    if 0 <= index < len(denominators):
        state.named_denominator = NamedDenominator(ref=denominators[index], base_words=base_words)
    elif index == len(denominators):
        state.named_denominator = NamedDenominator(ref=None, base_words=base_words)
    else:
        logger.warning(
            "quantity read returned denominator_index %d outside %d entries",
            index,
            len(denominators) + 1,
        )
        return False
    return True


def _slot_defaults(shapes: Sequence[ShapeDraft]) -> list[SlotDefault]:
    """The declared default of each constraint slot `shapes` hold: the code index's, then
    each one it declares for a population type among them (`CodeIndex.shape_slot_defaults`)."""
    vocabulary = frozenset().union(*(shape_vocabulary(shape) for shape in shapes))
    index = get_index()
    declarations = index.declarations
    defaults = [
        SlotDefault(property=prop, value=default.code)
        for prop in sorted(vocabulary)
        if (declaration := declarations.get(prop)) is not None
        and (default := declaration.default) is not None
    ]
    population_types = {shape.pop_type_dcid for shape in shapes}
    defaults.extend(
        SlotDefault(property=default.property_dcid, value=default.value_dcid)
        for default in index.shape_slot_defaults
        if default.population_type in population_types and default.property_dcid in vocabulary
    )
    return defaults


def _completed_frame(
    state: ResolveState, eligible_shapes: Sequence[ShapeDraft], read: _QueryQualifiers
) -> Frame:
    """The part's frame completed once `read`, the codes and ties the offered readings'
    vocabularies find in its words, is in (`frame.with_recall`), with the slot defaults
    `eligible_shapes` declare."""
    return with_recall(
        state.frame,
        dropped_phrases=state.dropped_qualifier_phrases,
        matches=read.matches,
        ties=merge_lists(read.ties.values()),
        readings=state.query_qualifier_readings,
        requirement=state.quantity_requirement,
        stated_form=state.stated_form,
        named_denominator=state.named_denominator,
        measure_class=measure_class(state.variable),
        place_sets=state.place_sets,
        role_bearing=state.role_bearing_outranks,
        slot_defaults=_slot_defaults(eligible_shapes),
    )


async def _send_frame(state: ResolveState, eligible_shapes: Sequence[ShapeDraft]) -> None:
    """Send the part's completed frame with the denominators `eligible_shapes` carry
    (`FrameEvent`). A part sends it once: with its seated menu, when a source exclusion
    removes a reading and what it leaves seats nothing, or with the empty menu of a part
    that offers none (`_refused_menu`)."""
    if state.emit is not None:
        await state.emit(
            FrameEvent(
                variable_index=state.variable_index,
                frame=state.frame,
                denominators_on_offer=sorted(
                    {s.meas_denom_dcid for s in eligible_shapes if s.meas_denom_dcid is not None}
                ),
            )
        )


@dataclass(frozen=True)
class _Offer:
    """What the phase reads for the readings it offers, held apart from the part until the
    menu seats one of them (`_hold_offer`): the completed frame and the shapes whose
    denominators it is sent with, the query's codes, ties and matched phrases in the offered
    vocabularies, the measure matches, and the default-reading collapse."""

    frame: Frame
    framed_shapes: tuple[ShapeDraft, ...]
    matches: dict[frozenset[str], tuple[QualifierMatch, ...]]
    ties: dict[frozenset[str], tuple[QualifierTie, ...]]
    matched_phrases: frozenset[str]
    measure_matches: tuple[QualifierMatch, ...]
    measure_conflict_phrases: frozenset[str]
    unread_price_basis: bool
    default_reading: TraceDefaultReading
    eligible_shapes: list[ShapeDraft]


async def _hold_offer(state: ResolveState, offer: _Offer) -> None:
    """Write `offer` to the part and send its frame."""
    state.frame = offer.frame
    state.query_qualifier_matches = offer.matches
    state.query_qualifier_ties = offer.ties
    state.variable_matched_qualifier_phrases = offer.matched_phrases
    state.measure_query_matches = offer.measure_matches
    state.measure_qualifier_conflict_phrases = offer.measure_conflict_phrases
    state.unread_price_basis = offer.unread_price_basis
    state.default_reading = offer.default_reading
    state.eligible_shapes = offer.eligible_shapes
    await _send_frame(state, offer.framed_shapes)


def _read_offer(
    state: ResolveState,
    eligible_shapes: Sequence[ShapeDraft],
    sources: _SourceReadings,
    reading_owned_phrases: frozenset[str],
) -> tuple[_Offer, Sequence[ShapeDraft]]:
    """What the phase reads for `eligible_shapes`, the readings it may offer, and those
    readings narrowed, ranked and collapsed, ready to seat. Nothing is written to the
    part (`_hold_offer`). The query-aware measure collapse and bind admission share the
    eligible shapes' matches and ties."""
    query_qualifiers = _match_query_qualifiers(
        sources,
        qualifiers=state.qualifiers,
        vocabularies=(shape_vocabulary(s) for s in eligible_shapes),
    )
    frame = _completed_frame(state, eligible_shapes, query_qualifiers)
    framed_shapes = tuple(eligible_shapes)
    # Measure values are read in every system together, so the measure matches come from
    # the union of every vocabulary's matches.
    query_qualifier_matches = merge_lists(query_qualifiers.matches.values())
    # Families of one five-tuple and role class are rival readings, and the one reading the
    # most of the query reaches the election; of families reading alike, one with no series
    # at the named places leaves for one with a series there. On a part read with a word
    # naming a party, the shapes binding it are the pool first, or the reading is withdrawn
    # (`with_party_reading`). Then a role-bearing shape publishing on an axis the query
    # does not ask for leaves for one reading as much without it (`without_unasked_axes`).
    phrases_read = _phrases_reader(state, query_qualifiers)
    eligible_shapes = with_party_reading(state, eligible_shapes, phrases_read)
    eligible_shapes = elect_five_tuple_family(
        eligible_shapes,
        phrases_read,
        series_at_places=series_at_places(state),
        stated_words=_stated_words_reader(state),
    )
    eligible_shapes = without_unasked_axes(state, eligible_shapes, phrases_read)
    measure_axes = get_measure_index().axes
    measure_query_matches, measure_qualifier_conflict_phrases = _measure_query_matches(
        query_qualifier_matches, measure_axes
    )
    # Only free-axis matches steer family election. Basis matches remain reportable only.
    role_free_axes = {axis.axis for axis in measure_axes if axis.role == "free"}
    role_free_matches = tuple(
        match for match in measure_query_matches if match.code_system in role_free_axes
    )

    # Preserve ranked order: the status rule offers rivals in menu order.
    ranked_shapes = sorted(eligible_shapes, key=_rank_key)
    # Collapse measure-only variants before seating. Newly freed seats follow their own
    # rank. Shapes the index does not carry take the declared price-basis order, the query's
    # named readings first, unless the leg left a price-basis phrase unread.
    unread_price_basis = leaves_price_basis_unread(
        state.dropped_qualifier_phrases,
        operation_phrases=[operation.phrase for operation in state.operations],
    )
    collapsed = collapse_measure_families(
        ranked_shapes,
        get_index(),
        measure_axes,
        named=role_free_matches,
        reading=(
            None
            if unread_price_basis
            else _default_reading(state, role_free_matches, eligible_shapes)
        ),
    )
    offer = _Offer(
        frame=frame,
        framed_shapes=framed_shapes,
        matches=query_qualifiers.matches,
        ties=query_qualifiers.ties,
        # Dropped-qualifier warnings cover matches, ties and two-reading spans alike.
        matched_phrases=_outside_unit_text(
            query_qualifiers.phrases | reading_owned_phrases, state.unit_text
        ),
        measure_matches=measure_query_matches,
        measure_conflict_phrases=measure_qualifier_conflict_phrases,
        unread_price_basis=unread_price_basis,
        default_reading=_default_reading_trace(
            ranked_shapes, collapsed, applied=not unread_price_basis
        ),
        eligible_shapes=eligible_shapes,
    )
    return offer, collapsed


def _outside_unit_text(phrases: frozenset[str], unit_text: str | None) -> frozenset[str]:
    """`phrases` without those `unit_text` holds. The unit request owns its words, as
    raw-text entry owns a dropped phrase's: it converts the answer to the price basis it
    names ("in real terms"), or departs for its phrase when the answer cannot take it
    (`phase_shared.withdraw_typed_unit`), so no measure-value warning repeats them."""
    if unit_text is None:
        return phrases
    unit = f" {_normalize(unit_text)} "
    return frozenset(phrase for phrase in phrases if f" {phrase} " not in unit)


async def _phase_prune_and_rank(state: ResolveState) -> RegionResult | None:
    """Judge every reading of the part, then rank and cap the admissible ones into the bind
    menu. Return a terminal result when none is admissible.

    Dataset candidacy (`_dataset_candidates`) and the score floor run first. A part whose
    scope (`ResolveState.scope`) names a source takes candidacy first, so naming a table
    can elect a family the unscoped ranking puts lower. It refuses ``qualifier_not_bound``,
    naming its sources (`_out_of_scope_records`), when the scope leaves none of the
    recalled shapes, before any rule. A code term that none of the scope's rows holds
    refuses once the bindings are final, with the elected shape as its near match
    (`_unavailable_qualifiers`).

    The admissibility rules then judge each shape under the frame
    (`region.admissibility`): measure class, a declared role on every axis the question
    asks, every form it names, and every code it names on a slot the shape holds. Between
    the role and form rules the denominators the shapes divide by are read
    (`_read_denominators`), and the quantity read states the form and base the phrase
    names (`_read_stated_quantity`), so the form rule judges a form only the model reads. Every judged shape is held in `ResolveState.interpretations`.
    When none is admissible and some are refused only on a pinned part, the menu is empty
    and those readings are its not-electable list (`_refused_menu`): bind runs so the model
    can name the nearest of them, which the part refuses with as its flagged match. When
    none is refused on a pinned part, the part refuses without a bind call
    (`_refusal_before_bind`), flagging nothing.

    The steps run in this order, and any binder that reads the quantity late keeps it:
    measure and roles; the quantity read; the form rule; pinned codes; the refusals
    written to `ResolveState.interpretations`; open-axis preference, family election and
    collapse; seating; the availability read; bind. The form rule runs before seating, so
    a shape it refuses holds no seat, and a menu it empties ends in a refusal because the
    refusals are already recorded.

    The shapes are ours, with `SEARCH_UN_ON_FIRST_MENU` joined by the UN System Data
    Commons' (`phase_intake._merge_un_pool`), or the UN System Data Commons' alone on the
    pass that runs when ours ends no_data (`ResolveState.pool`). Each shape meets its own
    pool's score floor (`_meets_score_floor`). The open-axis preference, five-tuple family election, the
    unasked-axes rule and the measure-family collapse narrow the admissible shapes, and
    the menu seats them in rank order under ``bind_top_k``, a topic root with every leaf
    or none (`seating.seat_menu`). What the phase reads for the readings it may offer
    (`_Offer`) is written to the part, and its frame sent, once the menu seats one. A part
    with no exclusion that removed a reading, whose admissible readings seat nothing, the
    leaves of roots larger than the menu, offers none and holds none of it: it takes the
    refused-menu path when it refused a reading (`_refuse_without_a_menu`), and otherwise
    refuses as `_declined_terminal` does. The menu availability read then states whether
    each menu shape has data at the places the part names. The shapes refused only on a
    pinned part, seated by the same rule under the same seat count, are listed beside the
    menu as not electable
    (`InterpretationSet.refused_on_pinned`).

    A part that only excludes sources chooses among the readings the unscoped question
    reaches: every rule and the ranking run unscoped, and candidacy then filters the
    ranked menu before the cap (`_in_scope`). Rules judge each shape against the best
    reading, so removing that reading first would let through readings they drop. A table
    the index does not declare stays, and the scoped read serves none of an excluded
    source's data, so a reading whose only data is the excluded source has none where the
    question asks. The filter removes a reading when the menu would have offered it
    without the exclusion and does not offer it with it: a reading the filter took, or one
    a root displaced that seats only once the filter took some of its leaves. A reading
    offered neither way, a leaf of a root larger than the menu or a shape past the last
    seat, is no loss to the part (`ResolveState.scope_removed_readings`). When the filter
    removes a reading and what it leaves seats nothing, no shape or only the leaves of
    roots larger than the menu, the part refuses
    ``qualifier_not_bound`` naming the excluded sources, which marks the scope as settling
    the answer (`exclusion_records`, `ResolveState.scope_settled`), so the UN pass does not
    run, as it does not when the bind model declines what the exclusion left; it holds what
    its eligible readings read, less the leaves of roots it could not seat. A part the
    prunes after admissibility leave with no reading otherwise refuses as
    `_declined_terminal` does: ``qualifier_not_bound`` naming
    its sources when its scope removed a reading, at candidacy for a part naming a source
    and from the seated menu for one that only excludes, else ``variable_not_resolved``,
    as for an excluding part whose filter removed nothing the menu would have offered.
    Falling
    through leaves ``_phase_bind`` to call the model. An excluding source mention that
    publishes none of the recalled shapes is marked idle on the scope
    (`with_idle_exclusions`) and warns that it removed nothing (`idle_exclusion_warnings`)
    on every later outcome, and no refusal names it.
    """
    state.scope = with_idle_exclusions(state.scope, state.shapes_found)
    state.warnings.extend(idle_exclusion_warnings(state.scope))
    only_excludes = state.scope is not None and not state.scope.named
    candidates = _dataset_candidates(state.shapes_found, None if only_excludes else state.scope)
    state.scope_removed_readings = len(candidates) < len(state.shapes_found)
    # An exclusion that removes nothing leaves the part as unscoped.
    excluding = state.scope if only_excludes and _excludes_some(state.scope) else None
    if not candidates and state.scope is not None and state.shapes_found:
        await _record_shape_step(state)
        return state.no_data(
            "qualifier_not_bound",
            unbound_sources=_out_of_scope_records(state.shapes_found, state.scope),
        )
    # One reading of each text source, in every code system and in each candidate's
    # vocabulary (`qualifier_match.read_by_vocabulary`), serves the named-dataset check, the
    # matches and ties, and the elected shape's qualifier records. Each vocabulary, a
    # shape's slot properties, reads the text on its own: "ODA" binds the CRS flow type on a
    # CRS shape and the DAC1 aid type on a DAC1 shape, and a DAC1 label such as
    # "multilateral ODA" never hides "ODA" from a CRS shape.
    sources = _read_sources(
        state.variable,
        state.dropped_qualifier_phrases,
        query_text=state.query_text,
        vocabularies=(shape_vocabulary(shape) for shape in candidates),
    )
    state.variable_spans_in_every_system = sources.variable.everywhere.spans
    state.variable_spans_by_vocabulary = {
        vocabulary: reading.spans for vocabulary, reading in sources.variable.by_vocabulary.items()
    }
    # Retrieval score bounds the menu and orders it. The model chooses among the seated
    # menu.
    #
    # The floor filters candidates on their measured recall score, a topic expansion on its
    # root's, against its pool's floor (`_meets_score_floor`). A role-bearing
    # shape the index carries is exempt: its recall score describes the flow's direction,
    # and the index vouches for its members and offers the family's full option sets to
    # bind from. A role-bearing shape the index does not carry, a debt series, holds only
    # its recalled members, so a weak score is the whole of its claim to the question and
    # the floor applies.
    floored = [
        s
        for s in candidates
        if (is_role_bearing(s) and index_carries_shape(s)) or _meets_score_floor(s)
    ]
    # When the query's places name both flow ends and a role-bearing shape outranks the
    # subject-only ones (`_role_bearing_outranks`), admissibility refuses every subject-only
    # shape `role_incompatible` (`subject_shapes_refused`). Shapes refused on their measure
    # class rank nothing. A role-bearing shape the floor only exempts, ranked below a
    # subject-only one, evicts nothing: "GDP of Ethiopia" beside "aid from Norway to
    # Ethiopia" keeps its GDP shapes.
    named_class = measure_class(state.variable)
    state.role_bearing_outranks = _role_bearing_outranks(
        [s for s in floored if named_class is None or shape_measure_class(s) in (None, named_class)]
    )
    judged = judge_measure_and_roles(state, floored)
    admissible = [i.shape for i in judged if i.admissible]
    # The kinds of the denominators the shapes divide by class each one's form, and the
    # quantity read lists them by name to read the form and base the phrase states.
    await _read_denominators(state, admissible)
    if admissible and (malformed := await _read_stated_quantity(state, admissible)) is not None:
        return malformed
    # The question's codes read in each vocabulary the pinned parts are judged in. A reading
    # refused on a pinned part keeps its vocabulary's matches, which settle it when it
    # stands in for a declined election (`_stand_in_declined`).
    pinned_part_read = _match_query_qualifiers(
        sources,
        qualifiers=state.qualifiers,
        vocabularies=(shape_vocabulary(i.shape) for i in judged if i.admissible),
    )
    state.pinned_part_matches = pinned_part_read.matches
    judged = judge_pinned_parts(state, judged, state.pinned_part_matches)
    state.interpretations = InterpretationSet(interpretations=tuple(judged))
    # Spans naming a code and a price basis ("PPP") are read per shape, so the collapse, the
    # guard and bind admission each keep the reading their shape carries.
    query_qualifier_readings, reading_owned_phrases = match_query_readings(
        state.variable, state.dropped_qualifier_phrases, query_text=state.query_text
    )
    state.query_qualifier_readings = query_qualifier_readings
    admitted = [i.shape for i in judged if i.admissible]
    if not admitted and state.interpretations.refusal_reason() is not None:
        return await _refuse_without_a_menu(
            state, _refused_on_pinned(judged, excluding), pinned_part_read
        )
    # An axis the query asks every member of is a preference among those shapes: a shape
    # the index carries that declares no role on it leaves once a role-bearing shape that
    # does survives (`without_open_axis_gaps`).
    eligible_shapes = without_open_axis_gaps(state, admitted)
    if not eligible_shapes:
        # Nothing is admitted and nothing was refused, so no candidate cleared the score
        # floor; a refused reading takes the branch above.
        await _record_shape_step(state)
        return state.no_data("variable_not_resolved", llm_usage=state.var_usage)
    # From here to the seated menu the phase reads the readings it may offer apart from
    # the part (`_read_offer`), and writes them to it only once the menu seats one, so a
    # part that offers none holds nothing they read.
    offer, ranked_shapes = _read_offer(state, eligible_shapes, sources, reading_owned_phrases)
    refused_on_pinned = _refused_on_pinned(judged, excluding)
    top_k = settings().bind_top_k
    # The menu seats the hits in rank order under ``bind_top_k``, a topic root with every
    # leaf or none (`seat_menu`), which is the order the status rule offers alternatives in.
    menu = seat_menu(ranked_shapes, top_k)
    if excluding is not None:
        in_scope = _dataset_candidates(ranked_shapes, excluding)
        # The exclusion removed a reading when the menu would have offered it without the
        # exclusion and does not offer it with it: one the exclusion took, or one a root
        # displaced that seats only once the exclusion took some of its leaves. A reading
        # offered neither way, a leaf of a root larger than the menu or a shape past the
        # last seat, is no loss to the part.
        unexcluded, menu = menu, seat_menu(in_scope, top_k)
        seated = {shape.shape_id for shape in menu}
        state.scope_removed_readings = any(shape.shape_id not in seated for shape in unexcluded)
        ranked_shapes = in_scope
        refuses = state.scope_removed_readings and not menu
        if refuses and ranked_shapes:
            # What the exclusion left is the leaves of roots larger than the menu. The part
            # does not offer them, so they leave nothing on it: the offer is read from the
            # eligible shapes less every leaf of those roots.
            unseated = {shape.expansion_root for shape in ranked_shapes}
            offer, _ = _read_offer(
                state,
                [
                    shape
                    for shape in eligible_shapes
                    if not (is_expansion(shape) and shape.expansion_root in unseated)
                ],
                sources,
                reading_owned_phrases,
            )
        offer = replace(
            offer, eligible_shapes=_dataset_candidates(offer.eligible_shapes, excluding)
        )
        if refuses:
            # The exclusion removed a reading and what it left seats nothing: no shape, or
            # the leaves of roots larger than the menu. The part refuses on the exclusion.
            await _hold_offer(state, offer)
            await _record_shape_step(state)
            return state.no_data(
                "qualifier_not_bound",
                unbound_sources=exclusion_records(state, excluding),
                llm_usage=state.var_usage,
            )
    if not menu:
        # Every admissible reading is a leaf of a root larger than the menu, so the part
        # offers none: it refuses as a part refusing every reading does when it refused
        # some, and as a part its prunes left with no reading does otherwise.
        if state.interpretations.refusal_reason() is not None:
            return await _refuse_without_a_menu(state, refused_on_pinned, pinned_part_read)
        await _record_shape_step(state)
        return _declined_terminal(state, llm_usage=state.var_usage)
    state.ranked_shapes = list(ranked_shapes)
    await _hold_offer(state, offer)
    capped_shapes = [_shape_with_taxonomy(s) for s in menu]
    # The availability read: whether each menu shape has data where the part asks. Derive
    # read the named countries; a part naming any other place reads the menu at every
    # place it names, in one read. A menu of one shape has no alternative to read: the
    # elected shape's own read settles its data.
    places = menu_read_places(state.place_sets, state.recall_only_place_dcids)
    if places and len(capped_shapes) > 1:
        capped_shapes = await _read_availability(state, capped_shapes, places)
    state.capped_shapes = capped_shapes
    state.interpretations = replace(
        state.interpretations,
        admissible=tuple(Interpretation(shape=shape) for shape in capped_shapes),
        refused_on_pinned=tuple(refused_on_pinned),
    )
    await _record_shape_step(state)
    return None


async def _read_availability(
    state: ResolveState, shapes: list[ShapeDraft], places: list[str]
) -> list[ShapeDraft]:
    """`shapes` with their availability at `places` (`read_menu_availability`). A failed
    read leaves them as they were and warns that D stayed unknown."""
    read, failed = await asyncio.to_thread(
        read_menu_availability,
        shapes,
        graph=state.graph,
        places=places,
        deadline=state.deadline,
    )
    if failed:
        state.warnings.append(
            Warning(
                code=AVAILABILITY_PROBE_FAILED,
                severity="warn",
                message=(
                    "the read of whether each reading has data at the named places "
                    "failed; no reading was set aside for lack of data."
                ),
            )
        )
    return read


def _meets_score_floor(shape: ShapeDraft) -> bool:
    """Whether `shape`'s measured recall score clears its own pool's floor:
    `SEARCH_UNKNOWN_SCORE_THRESHOLD` for ours, `SEARCH_UN_UNKNOWN_SCORE_THRESHOLD` for the
    UN System Data Commons' (`ShapeDraft.pool`), since each resolver scores on its own
    scale. A shape with no score passes; it was not retrieved against the query."""
    score = shape.measured_score
    if score is None:
        return True
    floor = (
        settings().un_unknown_score_threshold
        if shape.pool == "un_system"
        else settings().unknown_score_threshold
    )
    return score >= floor


def _refused_on_pinned(
    judged: Sequence[Interpretation], excluding: SourceScope | None
) -> list[Interpretation]:
    """The not-electable list: the interpretations refused only on a pinned part, less
    those an exclusion removes, seated as the menu is under the same seat count
    (`seat_menu`), so a topic root is listed with every leaf or none and the leaves of one
    large root do not fill the list ahead of a scored reading."""
    ranked = sorted(
        (i for i in judged if i.pinned_failure is not None), key=lambda i: _rank_key(i.shape)
    )
    refused = ranked if excluding is None else _excluded_out(ranked, excluding)
    by_shape = {id(i.shape): i for i in refused}
    seated = seat_menu([i.shape for i in refused], settings().bind_top_k)
    return [by_shape[id(shape)] for shape in seated]


async def _refuse_without_a_menu(
    state: ResolveState, refused: list[Interpretation], pinned_part_read: _QueryQualifiers
) -> RegionResult | None:
    """End the phase of a part that offers no reading and has refused some. With a
    not-electable list the menu is empty and bind runs, so the model can name the nearest
    refused reading (`_refused_menu`); with none the part refuses here
    (`_refusal_before_bind`)."""
    if refused:
        return await _refused_menu(state, refused, pinned_part_read)
    await _record_shape_step(state)
    return _refusal_before_bind(state, llm_usage=state.var_usage)


async def _refused_menu(
    state: ResolveState, refused: list[Interpretation], pinned_part_read: _QueryQualifiers
) -> None:
    """Set an empty bind menu with `refused`, the seated readings refused only on a pinned
    part (`_refused_on_pinned`), as its not-electable list. The frame takes its codes and
    ties from `pinned_part_read`, the vocabularies the pinned parts were judged in, refused
    readings' included, and no denominator, since no shape is on offer."""
    state.capped_shapes = []
    state.frame = _completed_frame(state, (), pinned_part_read)
    await _send_frame(state, ())
    state.interpretations = replace(state.interpretations, refused_on_pinned=tuple(refused))
    await _record_shape_step(state)


def _refusal_before_bind(state: ResolveState, *, llm_usage: dict | None = None) -> RegionResult:
    """The refusal of a part no interpretation of which is admissible and none of whose
    refused readings the bind model named: the reason `NO_DATA_RESOLUTION_ORDER` elects
    among the refused interpretations' reasons (`InterpretationSet.refusal_reason`),
    flagging nothing. A `qualifier_not_bound` refusal names the codes each reading refused
    on them cannot bind."""
    judged = state.interpretations.interpretations
    reason = state.interpretations.refusal_reason()
    if reason is None:
        # A menu emptied without recording why is a fault upstream; refuse the part loudly
        # rather than answer from a menu nobody judged.
        logger.error("part %r reached a refusal with no refused reading recorded", state.variable)
        return state.no_data("variable_not_resolved", llm_usage=llm_usage)
    return state.no_data(
        reason,
        unbound_qualifiers=tuple(
            record
            for interpretation in judged
            if reason == "qualifier_not_bound" and interpretation.pinned_codes
            for record in pinned_code_records(interpretation)
        ),
        llm_usage=llm_usage,
    )


def _excludes_some(scope: SourceScope | None) -> bool:
    """Whether an excluding mention of `scope` removes some recalled shape, so it is not
    marked idle (`source_choice.with_idle_exclusions`)."""
    return scope is not None and any(
        mention.polarity == "exclude" and position not in scope.idle
        for position, mention in enumerate(scope.mentions)
    )


def _excluded_out(
    interpretations: Sequence[Interpretation], scope: SourceScope
) -> list[Interpretation]:
    """`interpretations` whose shapes a part excluding sources under `scope` may still
    answer from (`_dataset_candidates`), each with its shape stamped for the part."""
    index = get_index()
    return [
        replace(i, shape=shape_for_part(i.shape, scope, index))
        for i in interpretations
        if _in_scope(i.shape, scope, index)
    ]


async def _record_shape_step(state: ResolveState) -> None:
    """Record the shape timing from the start of shape recall and emit the shape step."""
    state.timing["shape"] = now_ms() - state.shape_t0
    await _emit_step(
        state.emit,
        state.variable_index,
        "shape",
        ms=state.timing["shape"],
    )


def _dataset_candidates(
    shapes: Sequence[ShapeDraft], scope: ProvenanceScope | None
) -> list[ShapeDraft]:
    """The shapes a part under the source scope `scope` may answer from
    (`_in_scope`), each stamped with the datasets the part may answer it from and the
    dataset it names (`shape_for_part`). With no scope every shape stays.
    """
    index = get_index()
    return [
        shape_for_part(shape, scope, index)
        for shape in shapes
        if scope is None or _in_scope(shape, scope, index)
    ]


def _in_scope(shape: ShapeDraft, scope: ProvenanceScope, index: CodeIndex) -> bool:
    """Whether a part under `scope` may answer `shape`.

    A shape of a family the index declares stays when a dataset publishing it is eligible
    under the scope (`part_datasets`, `code_index.members.eligible_among`). On a family a
    dataset that is not opt-in publishes, an opt-in publisher answers only a part naming a
    source. A shape whose five-tuple the index lists without declaring a dataset
    (`shape_listed_rows`) stays when a listed row is held by a provenance the scope allows.
    Any other shape, a table the index does not declare, stays when the scope allows a
    provenance the index declares no dataset for, and the read applies the scope to it
    (`origin_catalogue.Catalogue.scoped`): a provenance the part excludes serves none of
    its data, and the availability read sees none of it. A declared dataset publishes only
    the index's rows, so a scope of declared datasets alone holds none of its data. A UN
    System Data Commons shape reaches here only when the scope allows a UN provenance
    (`phase_intake._un_shapes`)."""
    if shape_datasets(shape, index):
        return bool(part_datasets(shape, scope, index))
    listed = shape_listed_rows(shape, index)
    if listed:
        return any(not scope.allowed.isdisjoint(row.provenances) for row in listed)
    return not scope.allowed <= index.datasets.keys()


def _declined_terminal(state: ResolveState, *, llm_usage: dict | None = None) -> RegionResult:
    """The refusal of a part left with no reading it accepts. A part whose source scope
    removed a reading (`ResolveState.scope_removed_readings`) refuses
    ``qualifier_not_bound`` naming its sources: readings exist outside them and none inside
    is accepted. When the scope only excludes sources that refusal settles the answer, so
    the UN pass does not run (`exclusion_records`); a part naming a source is left to
    `source_choice.allows_un_system`. Any other part refuses ``variable_not_resolved``,
    including one whose prunes, run unscoped under an excluding scope, left no reading."""
    scope = state.scope
    if scope is not None and state.scope_removed_readings:
        return state.no_data(
            "qualifier_not_bound",
            unbound_sources=(
                _out_of_scope_records(state.shapes_found, scope)
                if scope.named
                else exclusion_records(state, scope)
            ),
            llm_usage=llm_usage,
        )
    return state.no_data("variable_not_resolved", llm_usage=llm_usage)


def exclusion_records(state: ResolveState, excluding: SourceScope) -> tuple[ResolvedQualifier, ...]:
    """The sources a part refuses ``qualifier_not_bound`` on when it only excludes sources
    (`_out_of_scope_records`), marking the scope as settling the answer
    (`ResolveState.scope_settled`): our catalogue holds the measure and the part left its
    publishers out, so the UN pass does not run. It is the one place that sets the mark,
    and three refusals name their sources through here: the exclusion left no reading that
    seats, the model declined what it left (`_declined_terminal`), or the scoped read found
    the elected data excluded (`pipeline._excluded_from_the_place`). A refusal of the
    election for codes or rows the part's tables lack settles nothing, though it may name
    an excluded dataset (`phase_election._unserved_sources`)."""
    state.scope_settled = True
    return _out_of_scope_records(state.shapes_found, excluding)


def _out_of_scope_records(
    shapes: Sequence[ShapeDraft], scope: SourceScope
) -> tuple[ResolvedQualifier, ...]:
    """What a part whose scope leaves none of `shapes` cannot bind: the sources the scope
    was compiled from (`scope_records`), then each opt-in dataset the scope allows that
    publishes one of the shapes, which a part answers from only by naming it."""
    index = get_index()
    opt_in = dict.fromkeys(
        dataset
        for shape in shapes
        for dataset in shape_datasets(shape, index)
        if dataset in scope.allowed and is_opt_in(index, dataset)
    )
    return (*scope_records(scope), *(source_record(dataset) for dataset in opt_in))


def _bound_code_records(bindings: Sequence[SlotBindingDraft]) -> tuple[ResolvedQualifier, ...]:
    """Each code `bindings` bind on a what or how slot, as an unbound record under its
    label: what a part cannot bind when no table it may answer from holds a row with all
    of them."""
    index = get_index()
    records: list[ResolvedQualifier] = []
    for binding in bindings:
        if binding.axis not in ("what", "how") or binding.kind not in ("value", "set"):
            continue
        for dcid in binding.value_dcids:
            row = index.code_by_dcid(dcid)
            label = row.label if row is not None else dcid
            records.append(
                ResolvedQualifier(
                    text=label,
                    named=[GraphRef(ref=dcid, label=label)],
                    slot=None,
                    disposition="unbound",
                )
            )
    return tuple(records)


def _word_ranges(variable: str, words: AbstractSet[str]) -> list[tuple[int, int]]:
    """The character range of each word of `variable` that is one of the casefolded
    `words`."""
    return [
        (word.start, word.end) for word in word_tokens(variable) if word.text.casefold() in words
    ]


def _unread_words(
    variable: str, span: QualifierSpan, read: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The character ranges of `span`'s words in `variable` that no range in `read` covers,
    in query order: the words of a term a shape's own reading leaves unread."""
    return [
        (span.start + word.start, span.start + word.end)
        for word in word_tokens(variable[span.start : span.end])
        if not any(
            start <= span.start + word.start and span.start + word.end <= end for start, end in read
        )
    ]


class _UnreadRun(NamedTuple):
    """A run of a code term's words the elected shape's reading leaves unread: its
    character range in the variable and the term it belongs to."""

    start: int
    end: int
    span: QualifierSpan


def _unread_qualifier_runs(
    state: ResolveState, recorded: Sequence[ResolvedQualifier]
) -> list[_UnreadRun]:
    """The words of the variable's code terms, read in every code system together, that the
    elected shape's own reading leaves unread where it reads others, each run of adjacent
    words as one run, in query order.

    "multilateral ODA" is DAC1 aid type 2000 in every system together; a CRS shape reads
    "ODA" alone, so "multilateral" is unread there. Every span of the shape's own reading
    counts as read, a measure value's included. A term the shape reads none of names no
    code the answer applies in part: "gross national income" is a CRS type of finance and
    the name of the series a GNI shape measures. A run overlapping a term in `recorded`,
    such as a phrase the bind model listed as unplaced, is left to that record. A word the
    series states counts as read (`_series_words`, `_holds_word`): "grant" in "ODA grant
    equivalents", DAC2a aid type 201 "ODA grant" in every system together, is read on the
    grant-equivalent shape and is a run on an amount shape. So does a word the shape's
    datasets state of the rows the part reads (`_dataset_stated_words`): "bilateral" in
    "bilateral ODA", DAC1 aid type 1015 in every system together, is read on a CRS shape,
    whose every row is bilateral. On a shape with a denominator,
    function words count as read too: "as a share of GNI" in "ODA as a share of GNI", type
    of finance 2 in every system together, is read on the GNI share shape and is a run on
    an amount shape.
    """
    variable = state.variable
    stated = _series_words(state.shape_draft) | _dataset_stated_words(
        state.shape_draft, get_index(), state
    )
    if _ratio_words(state.shape_draft):
        stated |= FUNCTION_WORDS
    read = [
        (span.start, span.end)
        for span in for_shape(state.variable_spans_by_vocabulary, state.shape_draft)
    ]
    found = [
        occurrence
        for qualifier in recorded
        for occurrence in _all_normalized_text_ranges(qualifier.text, variable)
    ]
    unread_runs: list[_UnreadRun] = []
    for span in state.variable_spans_in_every_system:
        if not all(isinstance(row, CodeRow) for row in span.rows):
            continue
        unread = _unread_words(variable, span, read)
        if len(unread) == len(_unread_words(variable, span, ())):
            continue
        unread = [
            (start, end)
            for start, end in unread
            if not _holds_word(stated, variable[start:end].casefold())
        ]
        runs: list[tuple[int, int]] = []
        for start, end in unread:
            if runs and not variable[runs[-1][1] : start].strip():
                runs[-1] = (runs[-1][0], end)
            else:
                runs.append((start, end))
        unread_runs.extend(
            _UnreadRun(start, end, span)
            for start, end in runs
            if not any(_ranges_overlap((start, end), occurrence) for occurrence in found)
        )
    return unread_runs


def _unread_qualifier_phrases(
    state: ResolveState, withheld: Sequence[ResolvedQualifier] = ()
) -> tuple[str, ...]:
    """The text of each run `_unread_qualifier_runs` finds beside the recorded terms of
    `state.query_qualifiers` and the `withheld` ones (`_withheld_qualifiers`), which depart
    on their own as the words a log-mode part leaves unapplied."""
    return tuple(
        state.variable[run.start : run.end]
        for run in _unread_qualifier_runs(state, (*state.query_qualifiers, *withheld))
    )


def _undeclared_qualifiers(
    variable: str,
    spans: Sequence[QualifierSpan],
    shapes: Sequence[ShapeDraft],
    own_spans: Mapping[frozenset[str], tuple[QualifierSpan, ...]],
    chosen: Sequence[tuple[int, int]] = (),
    state: ResolveState | None = None,
) -> tuple[ResolvedQualifier, ...]:
    """The variable's code terms, `spans` read in every code system, that no shape in
    `shapes` can hold, each as an unbound record with evidence ``"unavailable"``.

    A part bound to a named dataset elects from that dataset's rows alone, so a code counts
    as held on a shape only when a row the dataset publishes of its five-tuple holds the
    code or one of its descendants (`_part_codes`), whatever other publishers of the
    five-tuple hold. Such a term can bind on none of them: "grants" on imputed multilateral
    aid, whose rows carry no type of finance, or a sector only another publisher of the
    five-tuple reports. A shape the index does not carry holds any code of a code system it
    declares a slot for. A span holding a measure value is the measure's to read and is
    left alone.

    `own_spans` holds `variable` read in each shape's own vocabulary
    (`qualifier_match.read_by_vocabulary`). A shape also holds a term when spans of its own
    reading, each holding a code on it, cover every word of the term. "OOF budget support"
    names DAC1 aid type 2210 in every code system together, which no CRS row holds, while
    a CRS shape reads "OOF" and "budget support" and holds both. "multilateral ODA" stays
    refused on a CRS part: the CRS reading holds "ODA" and leaves "multilateral" unread.
    A word the shape's datasets state of the rows the part reads (`_dataset_stated_words`,
    which reads the text and places of `state`) is read: "bilateral" in "bilateral ODA
    grants" on a CRS part.
    The phrases the bind model's menu choices read (`chosen`, `_menu_choices`) count as
    such spans: "population" in "population policies" yields to the purpose chosen for
    the phrase.
    """
    index = get_index()
    readings = [(shape, _part_codes(shape, index)) for shape in shapes]

    def held_on(
        shape: ShapeDraft, codes: Mapping[str, frozenset[str]] | None, row: CodeRow
    ) -> bool:
        if codes is None:
            return row.code_system in shape_vocabulary(shape)
        return bool(
            codes.get(row.code_system, frozenset()) & frozenset(index.descendants(row.dcid))
        )

    def read_on(
        shape: ShapeDraft, codes: Mapping[str, frozenset[str]] | None
    ) -> list[tuple[int, int]]:
        """The ranges of `shape`'s own spans that hold a code on it."""
        return [
            (own.start, own.end)
            for own in for_shape(own_spans, shape)
            if any(isinstance(row, CodeRow) and held_on(shape, codes, row) for row in own.rows)
        ]

    covered = [
        [
            *read_on(shape, codes),
            *chosen,
            *_word_ranges(variable, _dataset_stated_words(shape, index, state)),
        ]
        for shape, codes in readings
    ]

    def held(span: QualifierSpan) -> bool:
        code_rows = [row for row in span.rows if isinstance(row, CodeRow)]
        if any(held_on(shape, codes, row) for shape, codes in readings for row in code_rows):
            return True
        return any(not _unread_words(variable, span, ranges) for ranges in covered)

    undeclared: list[ResolvedQualifier] = []
    for span in spans:
        code_rows = [row for row in span.rows if isinstance(row, CodeRow)]
        if len(code_rows) < len(span.rows) or held(span):
            continue
        undeclared.append(_unavailable_record(span, code_rows))
    return tuple(undeclared)


def _unbindable_qualifiers(
    variable: str,
    spans: Sequence[QualifierSpan],
    shape: ShapeDraft,
    own_spans: Mapping[frozenset[str], tuple[QualifierSpan, ...]],
    chosen: Sequence[tuple[int, int]] = (),
) -> tuple[ResolvedQualifier, ...]:
    """The variable's code terms, `spans` read in every code system, that name only codes
    no shape can bind (`qualifier_match.bindable_codes`), one of them of a system `shape`
    declares, and whose words `shape`'s own reading (`own_spans`) leaves partly
    unread, each as an unbound record with evidence ``"unavailable"``.

    A shape's own reading skips such codes, so "population", `CRSTypeOfFinance4`, neither
    binds nor hides a purpose reading on a CRS shape. Where nothing else reads the word,
    "population ODA" refuses on it. A term every word of which the shape reads
    otherwise yields to that reading, and so does one inside a phrase the bind model's
    menu choice reads (`chosen`, `_menu_choices`): "population policies" read as purpose
    13010 records no unavailable "population".
    """
    bindable = bindable_codes(get_index())
    vocabulary = shape_vocabulary(shape)
    read = [(own.start, own.end) for own in for_shape(own_spans, shape)] + list(chosen)
    unbindable: list[ResolvedQualifier] = []
    for span in spans:
        code_rows = [row for row in span.rows if isinstance(row, CodeRow)]
        if (
            not code_rows
            or len(code_rows) < len(span.rows)
            or any(row.dcid in bindable for row in code_rows)
            or not any(row.code_system in vocabulary for row in code_rows)
            or not _unread_words(variable, span, read)
        ):
            continue
        unbindable.append(_unavailable_record(span, code_rows))
    return tuple(unbindable)


def _unavailable_qualifiers(state: ResolveState) -> list[ResolvedQualifier]:
    """The variable's code terms the elected shape cannot apply that the part refuses on,
    each as an unbound record with evidence ``"unavailable"``, in query order, with the
    elected shape as its near match.

    On a part naming a source (`_names_source`) they are the terms none of its tables'
    rows holds (`_undeclared_qualifiers`) and the terms naming
    only codes no shape can bind (`_unbindable_qualifiers`), in every coverage mode:
    "multilateral ODA" names DAC1 aid type 2000, which a CRS part cannot apply, so the CRS
    ODA answer is the near match. On a part naming no source they are the unbindable
    terms under ``SearchSettings.qualifier_coverage="refuse"`` only. In log mode
    `_withheld_qualifiers` holds them.
    """
    variable = state.variable
    spans = state.variable_spans_in_every_system
    chosen = _chosen_ranges(state)
    records = list(
        _unbindable_qualifiers(
            variable, spans, state.shape_draft, state.variable_spans_by_vocabulary, chosen
        )
    )
    if not _names_source(state):
        return records if settings().qualifier_coverage == "refuse" else []
    recorded = {(record.start, record.end) for record in records}
    records += [
        record
        for record in _undeclared_qualifiers(
            variable,
            spans,
            [state.shape_draft],
            state.variable_spans_by_vocabulary,
            chosen,
            state=state,
        )
        if (record.start, record.end) not in recorded
    ]
    return sorted(records, key=lambda record: record.start or 0)


def _unelected_qualifiers(state: ResolveState) -> tuple[ResolvedQualifier, ...]:
    """The variable's code terms no shape the bind call was offered holds
    (`_undeclared_qualifiers`), on a part naming a source whose call elected no shape;
    empty on any other part.

    With no elected shape, no final bindings exist for `_unavailable_qualifiers` to read,
    and the term is the reason the call found nothing to elect: "education aid" on a part
    naming IDS, whose debt rows hold no CRS purpose.
    """
    if not _names_source(state):
        return ()
    return _undeclared_qualifiers(
        state.variable,
        state.variable_spans_in_every_system,
        state.capped_shapes,
        state.variable_spans_by_vocabulary,
        state=state,
    )


def _names_source(state: ResolveState) -> bool:
    """Whether the part's source scope includes a source (`SourceScope.named`), so its
    answer comes from those sources' tables alone."""
    return state.scope is not None and state.scope.named


def _withheld_qualifiers(state: ResolveState) -> tuple[ResolvedQualifier, ...]:
    """The unavailable records `_unavailable_qualifiers` leaves out of the part's records:
    in log mode, on a part naming no source, the terms naming only codes no shape can bind
    (`_unbindable_qualifiers`). The part answers without them and departs each as not
    applied: "ODA goods donations from France to Kenya" answers ODA and says "goods
    donations" was not applied. Empty otherwise."""
    if _names_source(state) or settings().qualifier_coverage == "refuse":
        return ()
    return _unbindable_qualifiers(
        state.variable,
        state.variable_spans_in_every_system,
        state.shape_draft,
        state.variable_spans_by_vocabulary,
        _chosen_ranges(state),
    )


def _unavailable_record(span: QualifierSpan, code_rows: Sequence[CodeRow]) -> ResolvedQualifier:
    """The unbound record of the term `span`, naming `code_rows`, that the part cannot
    apply."""
    return ResolvedQualifier(
        text=span.matched_text,
        named=[GraphRef(ref=row.dcid, label=row.label) for row in code_rows],
        slot=None,
        disposition="unbound",
        start=span.start,
        end=span.end,
        evidence="unavailable",
    )


def _unheld_bound_codes(
    bindings: Sequence[SlotBindingDraft], shape: ShapeDraft
) -> tuple[ResolvedQualifier, ...]:
    """The codes `bindings` bind on `shape`'s what and how slots that a row of another
    publisher of its family holds and no row of the datasets its part answers from
    (`ShapeDraft.datasets`) holds, itself or as a descendant (`_part_codes`), each as an
    unbound record under its label.

    Candidacy offers the bind model only the codes of a scoped part's tables, but a code
    the query text names is forced into the bindings after the model's
    (`_force_qualifier_bindings`), and one only a table outside the scope holds can bind on
    none of the part's rows. A code no publisher holds is the bindings' own matter, as it
    is for a part with no scope. A shape the index does not carry has no rows to check.
    """
    index = get_index()
    codes = _part_codes(shape, index)
    family = _part_codes(replace(shape, datasets=shape_datasets(shape, index)), index)
    if codes is None or family is None:
        return ()
    unheld: list[ResolvedQualifier] = []
    for binding in bindings:
        if binding.axis not in ("what", "how") or binding.property_dcid is None:
            continue
        held = codes.get(binding.property_dcid, frozenset())
        elsewhere = family.get(binding.property_dcid, frozenset())
        for dcid in binding.value_dcids if binding.kind in ("value", "set") else ():
            reads = frozenset(index.descendants(dcid))
            if held & reads or not elsewhere & reads:
                continue
            row = index.code_by_dcid(dcid)
            label = row.label if row is not None else dcid
            unheld.append(
                ResolvedQualifier(
                    text=label,
                    named=[GraphRef(ref=dcid, label=label)],
                    slot=None,
                    disposition="unbound",
                )
            )
    return tuple(unheld)


def _default_reading_trace(
    before: Sequence[ShapeDraft], after: Sequence[ShapeDraft], *, applied: bool
) -> TraceDefaultReading:
    """What the default-reading collapse did to the menu. It alone removes members of shapes
    the index does not carry, so the members of those shapes no surviving shape holds are
    the ones it removed."""
    kept = {sv for shape in after for sv in shape.sv_arc_facts or {}}
    removed = {
        sv
        for shape in before
        if not index_carries_shape(shape)
        for sv in shape.sv_arc_facts or {}
        if sv not in kept
    }
    return TraceDefaultReading(applied=applied, removed_stat_vars=sorted(removed))


def _default_reading(
    state: ResolveState,
    free_matches: Sequence[QualifierMatch],
    eligible_shapes: Sequence[ShapeDraft],
) -> DefaultReading:
    """The price-basis order and requested years the default-reading collapse applies.

    A price basis is named by a free-axis match, or by a two-reading span ("PPP") whose
    measure reading one of `eligible_shapes` carries alone (`reading_for_shape`).
    `price_basis_order` ignores values of other axes.
    """
    named = [match.code_dcid for match in free_matches]
    named += [
        readings.measure.code_dcid
        for readings in state.query_qualifier_readings
        if any(reading_for_shape(readings, s) is readings.measure for s in eligible_shapes)
    ]
    request = state.date_request
    window = None if request is None else ("latest" if request.latest else request.window)
    return DefaultReading(order=price_basis_order(get_measure_index(), named), window=window)


def _filter_admitted_matches(
    matches: tuple[QualifierMatch, ...], declared: set[str] | frozenset[str]
) -> tuple[QualifierMatch, ...]:
    """Keep matches on declared constraint properties."""
    return tuple(m for m in matches if m.code_system in declared)


def _measure_query_matches(
    matches: tuple[QualifierMatch, ...], axes: Sequence[MeasureAxis]
) -> tuple[tuple[QualifierMatch, ...], frozenset[str]]:
    """Keep one query-named value per declared measure axis and return refused phrases.

    Conflicting values on one axis refuse that axis. Matches on undeclared axes are dropped.
    This reporting set includes free and basis axes. The caller separately limits which
    roles can steer measure collapse.
    """
    declared_axes = {axis.axis for axis in axes}
    by_axis: dict[str, list[QualifierMatch]] = defaultdict(list)
    for match in matches:
        if match.code_system in declared_axes:
            by_axis[match.code_system].append(match)
    named: list[QualifierMatch] = []
    refused_phrases: set[str] = set()
    for axis_matches in by_axis.values():
        if len({m.code_dcid for m in axis_matches}) == 1:
            named.append(axis_matches[0])
        else:
            refused_phrases.update(_normalize(m.matched_text) for m in axis_matches)
    return tuple(named), frozenset(refused_phrases)


@dataclass(frozen=True, slots=True)
class _SourceReadings:
    """Each text source read once in every code system and in each vocabulary
    (`qualifier_match.read_by_vocabulary`): the variable, then the query text when given,
    then each dropped phrase. The region path owns the first `owned`; the raw-text path
    owns the dropped phrases' warnings."""

    texts: tuple[TextReading, ...]
    owned: int

    @property
    def variable(self) -> TextReading:
        return self.texts[0]


def _read_sources(
    variable: str,
    dropped_qualifier_phrases: Sequence[str] = (),
    *,
    query_text: str | None = None,
    vocabularies: Iterable[frozenset[str]],
) -> _SourceReadings:
    """Read ``variable``, optional ``query_text`` and each dropped phrase separately, so a
    match cannot span sources, each in every one of `vocabularies`."""
    texts = [variable, *([query_text] if query_text else []), *dropped_qualifier_phrases]
    return _SourceReadings(
        texts=read_by_vocabulary(
            texts, get_index(), get_measure_index(), vocabularies=vocabularies
        ),
        owned=1 + bool(query_text),
    )


def _in_vocabulary(code_system: str, vocabulary: frozenset[str]) -> bool:
    """Whether a sidecar match or tie row on `code_system` belongs to `vocabulary`: a code
    system's own row when the vocabulary holds the system, a measure value always, as
    `qualifier_match.read_by_vocabulary` reads measure values in every vocabulary."""
    return code_system in vocabulary or code_system not in get_index().declarations


@dataclass(frozen=True, slots=True)
class _QueryQualifiers:
    """The query's qualifier matches and ties, one list per vocabulary, and the normalised
    phrases this region path owns."""

    matches: dict[frozenset[str], tuple[QualifierMatch, ...]]
    ties: dict[frozenset[str], tuple[QualifierTie, ...]]
    phrases: frozenset[str]


def _sidecar_ties(qualifiers: Sequence[CodedQualifier]) -> list[QualifierTie]:
    """Each sidecar span naming two or more ``CodeRow`` values, as a tie."""
    ties: list[QualifierTie] = []
    for coded in qualifiers:
        for span in coded.spans:
            if len(span.rows) < 2 or not all(isinstance(row, CodeRow) for row in span.rows):
                continue
            candidate_norm = _normalize(span.matched_text)
            ties.append(
                QualifierTie(
                    matched_text=span.matched_text,
                    rows=tuple(
                        QualifierMatch(
                            code_system=_row_system(row),
                            code_dcid=row.dcid,
                            label=row.label,
                            source=_match_source(row, candidate_norm),
                            matched_text=span.matched_text,
                        )
                        for row in span.rows
                    ),
                )
            )
    return ties


def _phrases_reader(
    state: ResolveState, query: _QueryQualifiers
) -> Callable[[ShapeDraft], frozenset[str]]:
    """The query phrases each shape's family reads (`query_phrases_read`), in the shape's
    own vocabulary's matches and ties in `query`, each shape read once."""
    read: dict[int, frozenset[str]] = {}

    def phrases_read(shape: ShapeDraft) -> frozenset[str]:
        if id(shape) not in read:
            read[id(shape)] = query_phrases_read(
                state.held_codes_of(shape),
                for_shape(query.matches, shape),
                for_shape(query.ties, shape),
            )
        return read[id(shape)]

    return phrases_read


def _stated_words_reader(state: ResolveState) -> Callable[[ShapeDraft], frozenset[str]]:
    """The words of the part each shape's datasets state of the rows it would read
    (`_dataset_stated_words`), each shape read once."""
    index = get_index()
    stated: dict[int, frozenset[str]] = {}

    def stated_words(shape: ShapeDraft) -> frozenset[str]:
        if id(shape) not in stated:
            stated[id(shape)] = _dataset_stated_words(shape, index, state)
        return stated[id(shape)]

    return stated_words


def _unlocated(match: QualifierMatch) -> QualifierMatch:
    """`match` without offsets, for a match read from text other than the variable."""
    return replace(match, start=None, end=None)


def _match_query_qualifiers(
    sources: _SourceReadings,
    *,
    qualifiers: Sequence[CodedQualifier] | None = None,
    vocabularies: Iterable[frozenset[str]],
) -> _QueryQualifiers:
    """Collect raw qualifier matches and ties per vocabulary, and the phrases owned by this
    region path, from `sources` read in each of `vocabularies`.

    Return matches and ties before shape admission. The owned phrases are those any
    vocabulary matches or ties in the variable and query text, and the typed sidecar's;
    dropped phrases are left out because the raw-text path already owns their warnings.

    Sidecar matches and tie rows join each vocabulary holding their system
    (`_in_vocabulary`). Matches deduplicate by ``(code_system, code_dcid)`` and ties by
    their row sets; a sidecar phrase stays owned even when its match is a duplicate. Full
    qualifier spans are collected separately.
    """
    sidecar_matches = [match for coded in qualifiers or () for match in coded.matches]
    sidecar_ties = _sidecar_ties(qualifiers or ())
    phrases = {_normalize(match.matched_text) for match in sidecar_matches}
    phrases |= {_normalize(tie.matched_text) for tie in sidecar_ties}
    matches: dict[frozenset[str], tuple[QualifierMatch, ...]] = {}
    ties: dict[frozenset[str], tuple[QualifierTie, ...]] = {}
    for vocabulary in dict.fromkeys(vocabularies):
        readings = [text.by_vocabulary[vocabulary] for text in sources.texts]
        for reading in readings[: sources.owned]:
            phrases.update(_normalize(match.matched_text) for match in reading.matches)
            phrases.update(_normalize(tie.matched_text) for tie in reading.ties)

        # Only the variable's matches keep their offsets (`QualifierMatch.start`).
        read_matches = readings[0].matches + tuple(
            _unlocated(match) for reading in readings[1:] for match in reading.matches
        )
        seen_codes = {(m.code_system, m.code_dcid) for m in read_matches}
        joined_matches: list[QualifierMatch] = []
        for match in sidecar_matches:
            key = (match.code_system, match.code_dcid)
            if key in seen_codes or not _in_vocabulary(match.code_system, vocabulary):
                continue
            seen_codes.add(key)
            joined_matches.append(match)
        matches[vocabulary] = read_matches + tuple(joined_matches)

        read_ties = readings[0].ties + tuple(
            replace(tie, rows=tuple(_unlocated(row) for row in tie.rows))
            for reading in readings[1:]
            for tie in reading.ties
        )
        seen_row_sets = {
            frozenset((row.code_system, row.code_dcid) for row in tie.rows) for tie in read_ties
        }
        joined_ties: list[QualifierTie] = []
        for tie in sidecar_ties:
            rows = tuple(row for row in tie.rows if _in_vocabulary(row.code_system, vocabulary))
            key = frozenset((row.code_system, row.code_dcid) for row in rows)
            if not rows or key in seen_row_sets:
                continue
            seen_row_sets.add(key)
            joined_ties.append(QualifierTie(matched_text=tie.matched_text, rows=rows))
        ties[vocabulary] = read_ties + tuple(joined_ties)
    return _QueryQualifiers(matches=matches, ties=ties, phrases=frozenset(phrases))


def _admit_qualifier_matches(
    shapes: list[ShapeDraft],
    matches: Mapping[frozenset[str], tuple[QualifierMatch, ...]],
    also: tuple[QualifierMatch, ...] = (),
) -> tuple[QualifierMatch, ...]:
    """Keep each candidate shape's own matches (`for_shape`), with `also`, on the constraint
    properties that shape declares, merged in shape order (`merge_lists`).

    Election vetting narrows them to the chosen shape. The caller supplies the shared raw
    matches so each variable's text is recognised once per vocabulary.
    """
    return merge_lists(
        _filter_admitted_matches((*for_shape(matches, shape), *also), shape_vocabulary(shape))
        for shape in shapes
    )


class _ShapeTies(NamedTuple):
    """One reading of a shape's ties (`_admit_qualifier_ties`): the rows that promote to
    matches, the ties bind chooses among, and the ties the shape declares a system of but
    whose codes its family holds none of, which bind is not offered and election reports
    as dropped."""

    promoted: tuple[QualifierMatch, ...]
    admitted: tuple[QualifierTie, ...]
    unheld: tuple[QualifierTie, ...]


def _admit_qualifier_ties(
    shapes: list[ShapeDraft],
    ties: Mapping[frozenset[str], tuple[QualifierTie, ...]],
    *,
    holds: Callable[[ShapeDraft], Callable[[QualifierMatch], bool]],
) -> _ShapeTies:
    """Narrow each candidate shape's own ties (`for_shape`) to rows whose properties that
    shape declares and whose codes its family holds (`holds`, the reading the five-tuple
    family election compares; `ResolveState.held_codes_of` keeps one per shape). "ODA"
    ties the CRS flow type and DAC2a's ODA code on ONE's imputed sectors table, whose rows
    hold only the flow type.

    Drop a tie of no declared system, promote a single surviving row to an ordinary match,
    and pass ties with two or more rows to the model for election, vetted after shape
    choice. A tie of a declared system whose codes the family holds none of, though rows
    of another family do, is `unheld`. A shape's own reading skips codes no shape can bind
    (`qualifier_match.bindable_codes`), so they form no tie.
    """
    promoted: list[list[QualifierMatch]] = []
    admitted: list[list[QualifierTie]] = []
    unheld: list[list[QualifierTie]] = []
    for shape in shapes:
        declared = shape_vocabulary(shape)
        holds_code = holds(shape)
        promoted.append([])
        admitted.append([])
        unheld.append([])
        for tie in for_shape(ties, shape):
            in_vocabulary = tuple(row for row in tie.rows if row.code_system in declared)
            narrowed = tuple(row for row in in_vocabulary if holds_code(row))
            if not narrowed:
                if in_vocabulary:
                    unheld[-1].append(
                        QualifierTie(matched_text=tie.matched_text, rows=in_vocabulary)
                    )
                continue
            if len(narrowed) == 1:
                promoted[-1].append(narrowed[0])
                continue
            admitted[-1].append(QualifierTie(matched_text=tie.matched_text, rows=narrowed))
    return _ShapeTies(merge_lists(promoted), merge_lists(admitted), merge_lists(unheld))


def _shape_qualifiers(
    state: ResolveState, shape: ShapeDraft
) -> tuple[tuple[QualifierMatch, ...], tuple[QualifierTie, ...], tuple[QualifierTie, ...]]:
    """`shape`'s own admitted qualifiers and ties: its vocabulary's matches on the properties
    it declares, the code reading of each two-reading span it reads as the code
    (`reading_for_shape`), then the tie rows it alone holds, promoted; its ties with two or
    more rows it holds; and its unheld ties (`_ShapeTies.unheld`)."""
    reading_codes = tuple(
        readings.code
        for readings in state.query_qualifier_readings
        if reading_for_shape(readings, shape) is readings.code
    )
    matches = _admit_qualifier_matches([shape], state.query_qualifier_matches, also=reading_codes)
    ties = _admit_qualifier_ties([shape], state.query_qualifier_ties, holds=state.held_codes_of)
    return matches + ties.promoted, ties.admitted, ties.unheld


def _admit_qualifier_matches_against_elected_shape(
    matches: tuple[QualifierMatch, ...], shape_draft: ShapeDraft
) -> tuple[QualifierMatch, ...]:
    """Keep only matches whose constraint property the elected shape declares."""
    return _filter_admitted_matches(matches, shape_vocabulary(shape_draft))


class _Operation(NamedTuple):
    """How a match's codes combined with another match's on its code system, and the codes
    the two asked for before the combination simplified them, in variable order."""

    operation: Literal["union", "refinement"]
    selected: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Combination:
    """How the admitted matches of one variable combine, per code system.

    `values` holds each code system's codes once combined; a system whose matches leave
    each other unresolved, or whose every match an exclusion removes, has none.
    `operations` holds the operation of each located match that combined with another, by
    its offsets. `unsupported` holds each exclusion or comparison phrase, with whether it
    compares two codes of one system that do not nest.
    """

    values: dict[str, tuple[str, ...]]
    operations: dict[tuple[int, int], _Operation]
    unsupported: tuple[tuple[GoverningConnective, bool], ...]


def _offsets(match: QualifierMatch) -> tuple[int, int] | None:
    """`match`'s offsets in the variable, or `None` for a match read from other text."""
    if match.start is None or match.end is None:
        return None
    return match.start, match.end


def _nests_under(codes: Iterable[str], broader: Sequence[str], index: CodeIndex) -> bool:
    """Whether every one of `codes` is one of `broader` or a descendant of one."""
    return all(any(code in index.descendants(b) for b in broader) for code in codes)


def _combine_matches(
    variable: str, matches: Sequence[QualifierMatch], index: CodeIndex
) -> _Combination:
    """Combine the admitted matches on each code system by the connective between each
    pair, read from `variable` (`qualifier_match.joins`).

    Matches a union connective joins form a union: each code that nests under another of
    the union drops, so "HIV/AIDS and population policy, programme, and reproductive
    health" binds 130, and codes that do not nest bind a set, as "health and education"
    binds {120, 110}. Two unions joined any other way, or by nothing, are a refinement when
    one's codes nest under the other's, and the narrower binds: "malaria control within
    health" binds 12262. Otherwise they leave each other unresolved and the system is not
    forced. A match without offsets joins its neighbours by nothing, and takes no part when
    a located match names its code, so "health and education ODA" with the query text
    "health and education ODA to Kenya" still binds {120, 110}. Codes a comparison
    connective governs form no union with each other. A match an exclusion connective
    governs, and each match a union joins to it, drops from the combination, so "health
    excluding malaria control" binds 120, the code the phrase before the connective names.
    A system whose every match drops is not forced, so "ODA excluding health" keeps the
    model's binding, and the excluded code's record judges it.
    """
    governing = governing_connectives(variable, matches)
    excluded = {connective.governed for connective in governing if connective.kind == "exclusion"}
    by_system: dict[str, list[QualifierMatch]] = defaultdict(list)
    for match in matches:
        by_system[match.code_system].append(match)

    values: dict[str, tuple[str, ...]] = {}
    operations: dict[tuple[int, int], _Operation] = {}
    compared: set[tuple[int, int]] = set()
    for code_system, system_matches in by_system.items():
        located = sorted(
            (m for m in system_matches if _offsets(m) is not None), key=lambda m: m.start or 0
        )
        kinds = joins(variable, located, governing)
        unions: list[list[QualifierMatch]] = []
        for position, match in enumerate(located):
            kind = kinds[position - 1] if position else None
            if kind == "exclusion" or _offsets(match) in excluded:
                continue
            if kind == "union" and unions:
                unions[-1].append(match)
                continue
            if kind == "comparison" and unions:
                previous = unions[-1][-1]
                if not (
                    _nests_under([match.code_dcid], [previous.code_dcid], index)
                    or _nests_under([previous.code_dcid], [match.code_dcid], index)
                ):
                    compared.update(span for m in (previous, match) if (span := _offsets(m)))
            unions.append([match])
        # A match of the query text or a dropped phrase is a union of its own, unless it
        # copies a code the variable names, which combines at the variable's offsets.
        named = {m.code_dcid for m in located}
        unions += [[m] for m in system_matches if _offsets(m) is None and m.code_dcid not in named]

        combined: tuple[str, ...] = ()
        refined: list[str] = []
        for union in unions:
            asked = tuple(dict.fromkeys(m.code_dcid for m in union))
            codes = tuple(
                code
                for code in asked
                if not any(other != code and code in index.descendants(other) for other in asked)
            )
            if len(asked) > 1:
                for m in union:
                    if (span := _offsets(m)) is not None:
                        operations[span] = _Operation("union", asked)
            if not combined:
                combined = codes
            elif _nests_under(codes, combined, index):
                refined += (*combined, *codes)
                combined = codes
            elif _nests_under(combined, codes, index):
                refined += (*combined, *codes)
            else:
                break
        else:
            if combined:
                values[code_system] = combined
            selected = tuple(dict.fromkeys(refined))
            if len(selected) > 1:
                for m in system_matches:
                    if (span := _offsets(m)) is not None and m.code_dcid in selected:
                        operations.setdefault(span, _Operation("refinement", selected))

    unsupported = tuple(
        (connective, connective.kind == "comparison" and connective.governed in compared)
        for connective in governing
    )
    return _Combination(values=values, operations=operations, unsupported=unsupported)


def _force_qualifier_bindings(
    bindings: list[SlotBindingDraft],
    matches: tuple[QualifierMatch, ...],
    shape_draft: ShapeDraft,
    *,
    variable: str,
    chosen: tuple[QualifierMatch, ...] = (),
) -> tuple[list[SlotBindingDraft], frozenset[str]]:
    """Replace each admitted property's binding with the codes its matches and the bind
    model's menu choices (`chosen`, `_menu_choices`) combine into (`_combine_matches`), a
    value for one code and a set for several, and return the bindings with the properties
    so replaced.

    "refugee costs in donor countries ODA" binds DAC1's in-donor refugee costs, which nest
    under ODA, in either word order. A property whose matches leave each other unresolved
    keeps the model's binding, which the qualifier dispositions then judge. A menu choice
    combines as a match at its phrase, so "population policies and reproductive health"
    binds 130 whether the model chose 13010 or 130 for "population policies". The caller
    leaves out the terms a choice reads in place of (`_unchosen_matches`), and a code of
    the model's binding that no menu offered drops from a property the choices bind.
    """
    axis_by_property = {
        slot.property_dcid: slot.axis for slot in shape_draft.slot_keys if slot.property_dcid
    }
    combination = _combine_matches(variable, (*matches, *chosen), get_index())
    forced = {
        code_system: SlotBindingDraft(
            axis=axis_by_property[code_system],
            property_dcid=code_system,
            kind="value" if len(codes) == 1 else "set",
            value_dcids=list(codes),
        )
        for code_system, codes in combination.values.items()
        if code_system in axis_by_property
    }
    if not forced:
        return bindings, frozenset()
    kept = [b for b in bindings if b.property_dcid not in forced]
    return kept + list(forced.values()), frozenset(forced)


def _log_admitted_qualifiers(matches: tuple[QualifierMatch, ...]) -> None:
    """Log admitted matches. Keep raw ``matched_text`` in traces, outside prompts."""
    for match in matches:
        logger.info(
            "qualifier_match admitted code_system=%s code_dcid=%s source=%s matched_text=%r",
            match.code_system,
            match.code_dcid,
            match.source,
            match.matched_text,
        )


def _admitted_qualifier_phrases(matches: tuple[QualifierMatch, ...]) -> frozenset[str]:
    """Normalise admitted phrases with strip and casefold for exact warning comparisons.

    Substring matching could hide a warning for a broader dropped phrase.
    """
    return frozenset(_normalize(m.matched_text) for m in matches)


def _qualifier_disposition(
    rows: Sequence[tuple[str, str]],
    bound_values: Mapping[str, Sequence[str]],
    index: CodeIndex,
    union: Sequence[str] = (),
) -> tuple[Literal["bound", "unbound"], str | None]:
    """Return bound when a final bound value names a recorded code or a descendant of one,
    or names an ancestor of a recorded code that `union`, the codes a recorded union
    selected, also holds.

    "HIV" in "HIV/AIDS and population policy, programme, and reproductive health" is
    bound when 130 binds, since the union asked for both. Any other ancestor leaves the
    code unbound. Spans and ties share this rule.
    """
    for code_system, code_dcid in rows:
        bound = set(bound_values.get(code_system, ()))
        if bound & set(index.descendants(code_dcid)):
            return "bound", code_system
        if code_dcid in union and any(
            code_dcid in index.descendants(value) for value in bound & set(union)
        ):
            return "bound", code_system
    return "unbound", None


def _names_carried_per_head_rate(
    phrase: str, shape: ShapeDraft, kinds: Mapping[str, DenominatorKind]
) -> bool:
    """Whether `phrase` is a per-person rate base that `shape`'s denominator carries.

    No constraint slot holds a denominator, so the bind model can list "per 100 people" as
    unplaced on a shape whose measurementDenominator is a count of persons
    (`quantity.normalisation_class` under the leg's `kinds`).
    """
    return (
        PER_HEAD_RATE_BASE_RE.fullmatch(phrase.strip()) is not None
        and normalisation_class(shape, kinds=kinds) == "per_person"
    )


def _names_answering_dataset(phrase: str, state: ResolveState, index: CodeIndex) -> bool:
    """Whether every word of `phrase` occurs in the declared label of the dataset the part
    answers from, or in the words that dataset states (`_dataset_stated_words`): the settled
    dataset (`ResolveState.settled_dataset`), else each dataset the elected shape answers
    from (`ShapeDraft.datasets`), since election may serve any of them.

    Election applied such a word by choosing the table: "imputed" in "imputed education aid"
    picks "ONE Imputed Multilateral Sectors", whose name describes every row it serves. A
    stated word is true of every row the part reads: "bilateral" on the CRS, and "bilateral
    OECD" holds on the CRS by its label and its stated word together.
    """
    words = _casefolded_words(phrase)
    label_words = _answering_dataset_label_words(state, index)
    if not words or label_words is None:
        return False
    return words <= label_words | _dataset_stated_words(state.shape_draft, index, state)


def _casefolded_words(text: str) -> frozenset[str]:
    """The casefolded word tokens of `text`."""
    return frozenset(word.text.casefold() for word in word_tokens(text))


def _answering_dataset_label_words(
    state: ResolveState, index: CodeIndex, shape: ShapeDraft | None = None
) -> frozenset[str] | None:
    """The casefolded words every declared label of the datasets the part answers from
    holds (`_names_answering_dataset`), or `None` when the part answers from none the
    index declares. They are read for `shape`, the elected shape by default, as
    `ResolveState.settled_dataset` reads the elected one. That is the dataset `shape`
    names, else each dataset `shape` answers from."""
    labels = [dataset.label for dataset in _answering_datasets(shape or state.shape_draft, index)]
    if not labels:
        return None
    return frozenset.intersection(*(_casefolded_words(label) for label in labels))


def _answering_datasets(shape: ShapeDraft, index: CodeIndex) -> list[DatasetDeclaration]:
    """The declared datasets `shape` answers from: the dataset it names
    (`ShapeDraft.named_dataset`), else each dataset it answers from (`ShapeDraft.datasets`),
    as `ResolveState.settled_dataset` reads the elected shape."""
    settled = shape.named_dataset
    datasets = (settled,) if settled is not None else shape.datasets or ()
    return [index.datasets[dataset] for dataset in datasets if dataset in index.datasets]


def _dataset_stated_words(
    shape: ShapeDraft, index: CodeIndex, state: ResolveState | None
) -> frozenset[str]:
    """The words of the part that every dataset `shape` answers from states of the rows the
    part reads on it (`DatasetDeclaration.stated_words`). The datasets are those the shape
    names or candidacy stamped on it, and before candidacy, as in the family election, those
    publishing its family (`bound_values.declared_datasets`), none for a shape the index
    does not carry.

    A dataset states a word of the rows with one measured property, and where it names a
    kind of party, of those whose party on that flow axis is of the kind. "bilateral" holds
    on a CRS `amount` shape, every row of which is a flow from one provider to one
    recipient, and on a DAC2a one when the part binds its destination to countries only
    (`parties_of_kind`), since DAC2a's flows to a multilateral organisation are core
    contributions. It holds on no DAC2a shape whose recipient the part leaves open.

    The part states a word where its variable or a typed qualifier phrase holds it as a
    modifier of the measure (`place_roles.part_states`): "bilateral ODA", "net bilateral
    ODA". A word inside another ("non-bilateral"), one an occurrence negates or excludes
    in those texts or in the verbatim query (`bind_repair.stated_words_negated`, which
    reaches a negation the extraction left out of the variable, as "not bilateral ODA", and
    keys the response cache), and one standing past a function word or a connective ("ODA
    net of bilateral flows") names something else and is not stated. A part holding none
    of the words the datasets declare reads no place, and so does a call with no `state`,
    as a resubmit makes.

    A word naming a party `shape` binds (`place_roles.held_party_words`) is read too:
    "multilateral" in "multilateral ODA to Kenya" on a DAC2a shape, whose provider it binds
    to the multilaterals total."""
    if state is None:
        return frozenset()
    where_holds: dict[tuple[tuple[str, str], ...], bool] = {}

    def holds(stated: StatedWord) -> bool:
        if stated.measured_property != shape.meas_prop_dcid:
            return False
        if not part_states(state, stated.word):
            return False
        if not stated.where:
            return True
        key = tuple(sorted(stated.where.items()))
        if key not in where_holds:
            where_holds[key] = parties_of_kind(state, shape, stated.where)
        return where_holds[key]

    declared = [
        frozenset(stated.word for stated in dataset.stated_words if holds(stated))
        for dataset in declared_datasets(shape, index)
    ]
    stated = frozenset.intersection(*declared) if declared else frozenset()
    return stated | {named.word for named in held_party_words(state, shape)}


def with_party_reading(
    state: ResolveState,
    shapes: Sequence[ShapeDraft],
    phrases_read: Callable[[ShapeDraft], AbstractSet[str]],
) -> list[ShapeDraft]:
    """`shapes`, or, on a part read with a word naming a party (`ResolveState.party_sets`),
    the shapes that bind the party (`place_roles.held_party_words`) and read every phrase
    any other pooled shape reads.

    A shape binding the party reads the word as it reads a phrase of its codes
    (`phrases_read`, `phrases_within`). "Multilateral ODA to Kenya" reads "ODA" on the CRS
    and "ODA" and "multilateral" on DAC2a, so DAC2a's shapes are the pool and every answer
    comes from the multilaterals total. The party reading is withdrawn
    (`PartyReadingWithdrawn`) where no pooled shape binds the party, where another shape
    reads a phrase no binding shape reads, as the CRS reads the sector of "multilateral aid
    for education to Kenya", and where another shape reads the word as a value of its own,
    in a phrase of its codes or by a graph label (`_holds_value_labelled`), as a World Bank
    IDS creditor type in "multilateral loans to Kenya"."""
    if not state.party_sets:
        return list(shapes)
    index = get_index()
    words = {
        id(shape): frozenset(named.word for named in held_party_words(state, shape))
        for shape in shapes
    }

    def keeps_the_reading(binder: ShapeDraft) -> bool:
        read = phrases_read(binder) | words[id(binder)]
        return all(
            phrases_within(phrases_read(other), read)
            and not any(
                phrases_within({word}, phrases_read(other))
                or _holds_value_labelled(other, word, index)
                for word in words[id(binder)] - words[id(other)]
            )
            for other in shapes
            if other is not binder
        )

    kept = [shape for shape in shapes if words[id(shape)] and keeps_the_reading(shape)]
    if not kept:
        raise PartyReadingWithdrawn
    return kept


def _holds_value_labelled(shape: ShapeDraft, word: str, index: CodeIndex) -> bool:
    """Whether a member `shape` stands for holds a constraint value
    (`shape_constraint_values`) whose graph label (`ShapeDraft.value_labels`) holds the
    casefolded `word`. A value the code index knows is read in a phrase of its code and
    counts for nothing here, as in `value_requirements`."""
    labels = shape.value_labels or {}
    return any(
        index.code_by_dcid(value) is None and word in _casefolded_words(labels.get(value, ""))
        for _prop, value in shape_constraint_values(shape)
    )


# Four digits standing alone as a word, a year the part's window reads.
_YEAR_RE = re.compile(r"\d{4}")

# The words of a camelCase dcid ("FinancialFlow", "measuredValue").
_CAMEL_WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


# A possessive ending, which a word token holds ("France's").
_POSSESSIVE_RE = re.compile(r"['\u2019]s$", re.IGNORECASE)


def _token_occurrences(tokens: Sequence[Token], text: str) -> list[tuple[int, int]]:
    """The character range of each span of `tokens` that `text` matches as a code label
    matches a query span (`code_index.tokens.text_matches`): the same words, casefolded,
    and each separator the span holds where `text` holds it, so "Hong Kong SAR, China"
    holds the entity "Hong Kong SAR, China". The span's last word may add a possessive
    ending: "France's" holds the entity "France"."""
    count = len(word_tokens(text))
    if not count:
        return []
    word_at = [position for position, token in enumerate(tokens) if token.kind == "word"]
    occurrences: list[tuple[int, int]] = []
    for first in range(len(word_at) - count + 1):
        span = tokens[word_at[first] : word_at[first + count - 1] + 1]
        last = span[-1]
        bare = Token(
            text=_POSSESSIVE_RE.sub("", last.text), kind="word", start=last.start, end=last.end
        )
        if text_matches(text, span) or text_matches(text, (*span[:-1], bare)):
            occurrences.append((span[0].start, last.end))
    return occurrences


def _ratio_words(shape: ShapeDraft) -> frozenset[str]:
    """The words a shape that divides by a denominator states by its division: the ratio
    words (`RATIO_WORDS`) and the words of the denominator's noun
    (`measure_noun.denominator_noun`, "GNI" for a GNI denominator), casefolded. A shape with
    no denominator has none, and "as a share of GNI" restricts it."""
    if shape.meas_denom_dcid is None:
        return frozenset()
    return RATIO_WORDS | _casefolded_words(denominator_noun(shape.meas_denom_dcid) or "")


def _series_words(shape: ShapeDraft) -> frozenset[str]:
    """The casefolded words the series `shape` measures states: its population type name,
    each camelCase word of one of its five dcids ("grant" and "equivalent" of the measured
    property `grantEquivalent`), and its ratio words (`_ratio_words`)."""
    words = set(_casefolded_words(shape.pop_type_name or "")) | _ratio_words(shape)
    for dcid in (
        shape.pop_type_dcid,
        shape.meas_prop_dcid,
        shape.stat_type_dcid,
        shape.meas_qual_dcid,
        shape.meas_denom_dcid,
    ):
        words |= {word.casefold() for word in _CAMEL_WORD_RE.findall(dcid or "")}
    return frozenset(words)


def _holds_word(words: AbstractSet[str], word: str) -> bool:
    """Whether `words` hold the casefolded `word`, or every part of it a hyphen joins:
    "grant-equivalent" is read where "grant" and "equivalent" are."""
    return word in words or ("-" in word and all(part in words for part in word.split("-")))


def _conversational_words(
    state: ResolveState, index: CodeIndex, shape: ShapeDraft | None = None
) -> Callable[[str], bool]:
    """Whether a casefolded word names nothing the part restricts beyond what `shape`, the
    elected shape by default, already states: a function word (`FUNCTION_WORDS`), an
    amount word (`AMOUNT_WORDS`: "spending", "flows"), a word naming data itself
    (`DATA_WORDS`: "data" in "health data"), a flow verb (`FLOW_VERBS`:
    "received" in "net ODA received by Kenya"), a word the series states
    (`_series_words`: "financial", "flow", "amount", and on a shape with a denominator a
    ratio word or a word of the denominator's noun), a word every label of the datasets
    the part answers from holds, a word those datasets state of the rows the part reads
    (`_dataset_stated_words`: "bilateral" on the CRS, and on DAC2a to a country), the
    declared plain name of a slot the shape declares,
    singular or plural, a word that slot's declared default reads
    (`DeclaredDefault.when_named`: "aid" reads ODA on the CRS flow type), or a year. A
    hyphenated word is read when each of its parts is (`_holds_word`), so
    "grant-equivalent" is read on a grant-equivalent shape and refuses on an amount shape.

    The shape's own label reads nothing. It names the representative member, whose
    constraint values ("gross disbursements (population policies and reproductive health,
    ...)") would otherwise read the very words a restriction holds.
    """
    shape = shape or state.shape_draft
    known = set(FUNCTION_WORDS | AMOUNT_WORDS | DATA_WORDS | FLOW_VERBS | _series_words(shape))
    known |= _answering_dataset_label_words(state, index, shape) or frozenset()
    known |= _dataset_stated_words(shape, index, state)
    declarations = [
        index.declarations[prop] for prop in shape_vocabulary(shape) if prop in index.declarations
    ]
    noun_words = [
        word.text for declaration in declarations for word in word_tokens(declaration.plain_name)
    ]
    known |= {
        word.casefold()
        for declaration in declarations
        if declaration.default is not None
        for word in declaration.default.when_named
    }

    def conversational(word: str) -> bool:
        return (
            _holds_word(known, word)
            or _YEAR_RE.fullmatch(word) is not None
            or any(names_plain_name(word, noun) for noun in noun_words)
        )

    return conversational


# The words a run of unread words reaches past to its neighbours
# (`_unread_runs`). Articles and the two prepositions below are
# transparent only while walking backward from a post-code restriction. They
# can separate that restriction from the code it narrows without introducing a
# semantic operation or a clause boundary.
_TRANSPARENT_NEIGHBOUR_GAPS = frozenset({"a", "an", "the", "for", "with"})
_SKIPPED_NEIGHBOURS = AMOUNT_WORDS | DATA_WORDS | FLOW_VERBS | _TRANSPARENT_NEIGHBOUR_GAPS


class _UnreadWords(NamedTuple):
    """A run of adjacent words of the variable no reader covers, by token position: the
    code spans its neighbouring tokens lie in (`adjacent`), else those a union connective
    beside it joins it to (`joined`). Both are empty for a run beside no code span."""

    first: int
    last: int
    adjacent: tuple[QualifierSpan, ...]
    joined: tuple[QualifierSpan, ...]


# A reader of the variable's words, under the name its verdict gives the words it reads
# (`_unread_runs`): the character ranges it reads, or a test of one casefolded word.
type _RangeReader = tuple[str, Sequence[tuple[int, int]]]
type _WordReader = tuple[str, Callable[[str], bool]]


class _Reading(NamedTuple):
    """The runs of the variable's words no reader covers (`_UnreadWords`), in query order,
    and each word's verdict by token position: the name of the first reader that reads it,
    ``"code"`` inside a code span's reach, ``"connective"`` for a connective between two
    code spans, ``"date"`` in a run that forms a date phrase, else ``"unread"``."""

    runs: list[_UnreadWords]
    verdicts: dict[int, str]


def _code_span_reach(
    tokens: Sequence[Token], span: QualifierSpan
) -> tuple[int, int, QualifierSpan]:
    """The character range `span` reads in `tokens`: the span, widened over each run of
    adjacent words on either side whose every word the label or a synonym of each code row
    of the span holds, casefolded. Function words and connectives do not widen it, so the
    reach stops at "and" or "excluding" and joins no span to its neighbour."""
    named = [
        _casefolded_words(" ".join((row.label, *row.synonyms)))
        for row in span.rows
        if isinstance(row, CodeRow)
    ]
    words = frozenset.intersection(*named) - FUNCTION_WORDS - CONNECTIVES if named else frozenset()

    def widens(position: int) -> bool:
        token = tokens[position]
        return token.kind == "word" and token.text.casefold() in words

    first = next(position for position, token in enumerate(tokens) if token.end > span.start)
    last = max(position for position, token in enumerate(tokens) if token.start < span.end)
    while first > 0 and widens(first - 1):
        first -= 1
    while last + 1 < len(tokens) and widens(last + 1):
        last += 1
    return tokens[first].start, tokens[last].end, span


def _unread_runs(
    variable: str,
    tokens: Sequence[Token],
    code_spans: Sequence[QualifierSpan],
    ranges: Sequence[_RangeReader],
    words: Sequence[_WordReader],
) -> _Reading:
    """The runs of the variable's words that no reader in `ranges` or `words` and no
    connective between two of `code_spans` covers, and no date phrase forms
    (`time_bind.parse_time_text`), with the code spans beside each (`_UnreadWords`), and
    each word's verdict (`_Reading`). A run is adjacent words with no separator between
    them. Its neighbours lie past the amount words, data words and flow verbs beside it,
    so "maternal" in "maternal data ODA" and "water" in "water spending ODA" lie beside
    "ODA".

    A code span reaches over the words next to it that the label or a synonym of every
    code it names holds (`_code_span_reach`), so they are read and the span's neighbours
    lie past them. "multilateral" beside "imputed", DAC2a aid type 106 "imputed
    multilateral ODA", is read, and "grants" in "imputed multilateral grants" lies beside
    the span."""
    reach = [_code_span_reach(tokens, span) for span in code_spans]

    def code_span_at(position: int) -> QualifierSpan | None:
        if not 0 <= position < len(tokens):
            return None
        token = tokens[position]
        return next(
            (span for start, end, span in reach if start <= token.start and token.end <= end),
            None,
        )

    def reader_of(position: int) -> str | None:
        token = tokens[position]
        for name, read in ranges:
            if any(start <= token.start and token.end <= end for start, end in read):
                return name
        if code_span_at(position) is not None:
            return "code"
        word = token.text.casefold()
        for name, reads in words:
            if reads(word):
                return name
        if (
            word in CONNECTIVES
            and code_span_at(position - 1) is not None
            and code_span_at(position + 1) is not None
        ):
            return "connective"
        return None

    def joined(position: int, step: int) -> QualifierSpan | None:
        """The code span a union connective at `position` joins the run to, reading away
        from the run by `step`."""
        if not 0 <= position < len(tokens):
            return None
        if tokens[position].text.casefold() not in UNION_CONNECTIVES:
            return None
        return code_span_at(position + step)

    def neighbour(position: int, step: int) -> int:
        """The position of the first token from `position` by `step` that is not a skipped
        word outside a code span. "maternal" in "ODA for maternal care" lies beside
        "ODA", while "and" in "girls and health" stays the token beside "girls"."""
        skipped = _SKIPPED_NEIGHBOURS - (_TRANSPARENT_NEIGHBOUR_GAPS if step > 0 else frozenset())
        position += step
        while (
            0 <= position < len(tokens)
            and tokens[position].kind == "word"
            and code_span_at(position) is None
            and tokens[position].text.casefold() in skipped
        ):
            position += step
        return position

    verdicts: dict[int, str] = {}
    runs: list[tuple[int, int]] = []
    for position, token in enumerate(tokens):
        if token.kind != "word":
            continue
        if (reader := reader_of(position)) is not None:
            verdicts[position] = reader
            continue
        if runs and runs[-1][1] == position - 1:
            runs[-1] = (runs[-1][0], position)
        else:
            runs.append((position, position))

    current_year = date.today().year
    unread: list[_UnreadWords] = []
    for first, last in runs:
        dates, warning = parse_time_text(
            variable[tokens[first].start : tokens[last].end], current_year=current_year
        )
        read_as_date = bool(dates) and warning is None
        for position in range(first, last + 1):
            if tokens[position].kind == "word":
                verdicts[position] = "date" if read_as_date else "unread"
        if read_as_date:
            continue
        before, after = neighbour(first, -1), neighbour(last, 1)
        adjacent = tuple(span for span in (code_span_at(before), code_span_at(after)) if span)
        joined_spans = tuple(span for span in (joined(before, -1), joined(after, 1)) if span)
        unread.append(_UnreadWords(first, last, adjacent, joined_spans))
    return _Reading(unread, verdicts)


def _candidate_menus(state: ResolveState, shape: ShapeDraft) -> tuple[CandidateMenu, ...]:
    """The candidate menu of each phrase of the variable `shape`'s own reading leaves partly
    unread, in query order. A menu lists what the bind model may read the phrase as.

    The phrase's unread words are a run beside a code span `_unread_runs` finds, read
    before any bind output exists by the spans of the shape's own reading, the part's entity texts, the words
    `_conversational_words` accepts for `shape`, a connective between two code spans and a
    date phrase. A code term read in every code system together that names a code another
    shape the part recalled binds (`ResolveState.shapes_found`, before a named dataset
    narrows them), one declaring the code's system whose family holds it
    (`ResolveState.held_codes_of`), is read too, and `shape` leaves its words to the
    coverage rule. "imputed" in "imputed education aid" is DAC2a aid type 106, which the
    imputed multilateral shape binds, so a CRS shape is offered no "imputed student cost"
    for it, and "multilateral" in "multilateral ODA", DAC1 aid type 2000, stays an
    unavailable term on a part naming the CRS. "ODA loans" is DAC2a aid type 218, and
    with no recalled shape holding it "loans" keeps its menu, which offers type of
    finance 420. "population", type of finance 4, binds on no shape and keeps its menu.
    Its candidates are `CodeIndex.candidate_codes` over the run's words, with
    the words of the code terms beside it as context and their codes as seeds, in the code
    systems `shape` declares, less the codes its family does not hold
    (`ResolveState.held_codes_of`). The held test follows the index's bounded menu, so a
    shape holding fewer codes is offered a subset of the same menu. A phrase has a menu
    only when one of its held codes is a lexical hit, sharing a word of the run: no CRS
    purpose shares "maternal", so "maternal health" is read as it is with no menu. A code
    system keeps its candidates when it holds one other than the seeds.

    The phrase spans its run and each code term beside it whose code system kept its
    candidates. "infant health" is "infant" beside "health", and its menu offers the hit
    12240, then 120 and the relatives of each. "contraceptive services ODA" leaves "ODA"
    outside the phrase "contraceptive services", whose menu leads with 13030, since the menu
    offers no flow type but ODA itself. "population policies and reproductive health" holds
    the phrase "population policies", which "and" joins to 130.

    A shape the query's places do not bind to (`bind_place_roles` refuses) has no menu. It
    refuses whatever the model binds on it, and a menu would lift it in the model's
    election: on "how much do African countries spend on their own health care?" the CRS
    flow shape, which takes "African countries" in no role, would offer "general
    government and civil society support" for "domestic general government health", and
    the model would elect it over the expenditure series that answers.
    """
    if isinstance(bind_place_roles(state, shape), str):
        return ()
    index = get_index()
    variable = state.variable
    tokens = tokenize(variable)
    spans = for_shape(state.variable_spans_by_vocabulary, shape)
    code_spans = [span for span in spans if any(isinstance(row, CodeRow) for row in span.rows)]
    read = [(span.start, span.end) for span in spans]
    read += [
        occurrence for text in state.entities for occurrence in _token_occurrences(tokens, text)
    ]
    others = [other for other in state.shapes_found if other.shape_id != shape.shape_id]

    def bound_elsewhere(row: CodeRow) -> bool:
        return any(
            row.code_system in shape_vocabulary(other)
            and state.held_codes_of(other)(_code_match(row))
            for other in others
        )

    read += [
        (span.start, span.end)
        for span in state.variable_spans_in_every_system
        if any(isinstance(row, CodeRow) and bound_elsewhere(row) for row in span.rows)
    ]
    reading = _unread_runs(
        variable,
        tokens,
        code_spans,
        [("reading", read)],
        [("conversational", _conversational_words(state, index, shape))],
    )
    runs = [run for run in reading.runs if run.adjacent or run.joined]
    vocabulary = shape_vocabulary(shape)
    holds = state.held_codes_of(shape)
    menus: list[CandidateMenu] = []
    for run in runs:
        context = [
            word.text
            for span in run.adjacent
            for word in word_tokens(span.matched_text)
            if word.text.casefold() not in FUNCTION_WORDS
        ]
        seeds = tuple(
            dict.fromkeys(
                row.dcid for span in run.adjacent for row in span.rows if isinstance(row, CodeRow)
            )
        )
        menu = index.candidate_codes(
            (tokens[position].text for position in range(run.first, run.last + 1)),
            code_systems=vocabulary,
            context=context,
            seeds=seeds,
        )
        offered = [
            row
            for dcid in menu.codes
            if (row := index.code_by_dcid(dcid)) is not None and holds(_code_match(row))
        ]
        if not any(row.dcid in menu.hits for row in offered):
            continue
        systems = {row.code_system for row in offered if row.dcid not in seeds}
        inside = [
            span
            for span in run.adjacent
            if any(isinstance(row, CodeRow) and row.code_system in systems for row in span.rows)
        ]
        start = min([tokens[run.first].start, *(span.start for span in inside)])
        end = max([tokens[run.last].end, *(span.end for span in inside)])
        candidates = tuple(row for row in offered if row.code_system in systems)
        menus.append(
            CandidateMenu(
                text=variable[start:end],
                start=start,
                end=end,
                candidates=candidates,
                hits=tuple(row.dcid for row in candidates if row.dcid in menu.hits),
            )
        )
    return tuple(menus)


def _code_match(row: CodeRow, menu: CandidateMenu | None = None) -> QualifierMatch:
    """The code `row` as a match: located at `menu`'s phrase, which a model choice from that
    menu reads, or unlocated, for the held-code test. Its source is the code's label."""
    return QualifierMatch(
        code_system=row.code_system,
        code_dcid=row.dcid,
        label=row.label,
        source="label",
        matched_text=menu.text if menu is not None else row.label,
        start=menu.start if menu is not None else None,
        end=menu.end if menu is not None else None,
    )


def _menu_choices(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    *,
    terms: Sequence[QualifierMatch],
) -> tuple[QualifierMatch, ...]:
    """The codes `bindings` bind that the bind model chose from a candidate menu of `shape`
    (`_candidate_menus`) and that read its phrase, each as a match located at the phrase,
    in binding order.

    A code is chosen only on the code system its binding names and on exactly one of the
    menus. A code on two menus belongs to neither. Every menu code is one `shape`'s family
    holds. A chosen code reads its phrase only when it is a lexical hit of the menu
    (`CandidateMenu.hits`) or a descendant of one; vetting then forces it in place of the
    terms inside the phrase on its code system (`_force_qualifier_bindings`), and the part
    records the phrase with evidence ``"model"`` (`_model_records`). The terms' own codes,
    the ancestors of hits and the children of the terms' codes read nothing, so the terms
    force as they would with no menu, so 121 chosen for "infant health", whose menu's one
    hit is 12240, leaves "health" to bind 120.

    A chosen hit reads the phrase as the furthest of its ancestors among the menu's hits
    whose label or synonyms share every word of the phrase its own share
    (`_broadest_reading`), because the phrase names no narrower kind than that ancestor.

    A choice refines the terms of `terms` inside its phrase on its code system. It reads
    the phrase only when, for each such term, its code is the term's code or lies under it,
    or its label or a synonym holds every content word of the term (`_holds_every_word`).
    41081, "environmental education and training", chosen for
    "environment education" reads the phrase in place of 110, the code "education" names,
    though it lies under 410. 15180, "ending violence against women and girls", chosen for
    "girls education" reads nothing, because it lies outside 110 and holds no "education",
    so "education" keeps 110 and "girls" stays unread.
    """
    menus = _candidate_menus(state, shape)
    if not menus:
        return ()
    index = get_index()
    chosen: list[QualifierMatch] = []
    for binding in bindings:
        if binding.kind not in ("value", "set") or binding.property_dcid is None:
            continue
        for dcid in binding.value_dcids:
            offering = [
                (menu, row) for menu in menus for row in menu.candidates if row.dcid == dcid
            ]
            if len(offering) != 1:
                continue
            menu, row = offering[0]
            if row.code_system != binding.property_dcid:
                continue
            if not any(dcid in index.descendants(hit) for hit in menu.hits):
                continue
            match = _code_match(_broadest_reading(row, menu, index), menu)
            if not all(
                match.code_dcid in index.descendants(term.code_dcid)
                or _holds_every_word(index, match.code_dcid, term.matched_text)
                for term in terms
                if _reads_in_place_of(match, term)
            ):
                continue
            if match not in chosen:
                chosen.append(match)
    return tuple(chosen)


def _holds_every_word(index: CodeIndex, dcid: str, text: str) -> bool:
    """Whether the label or a synonym of the code `dcid` holds every content word of
    `text`, less `FUNCTION_WORDS`, compared as `CodeIndex.words_shared` compares them."""
    words = [word.text for word in word_tokens(text) if word.text.casefold() not in FUNCTION_WORDS]
    return bool(words) and all(index.words_shared(dcid, [word]) for word in words)


def _broadest_reading(row: CodeRow, menu: CandidateMenu, index: CodeIndex) -> CodeRow:
    """The code a choice of `row` from `menu` reads the phrase as: the furthest ancestor of
    `row` among the menu's hits whose label or synonyms share every word of the phrase that
    `row`'s share (`CodeIndex.words_shared`), else `row`. A code outside the hits, the
    model's refinement below one, is itself.

    Type of finance 420 (synonym "loan instrument") and its child 421 ("standard loan")
    share only "loans" with "education loans", so 421 chosen reads as 420. The phrase
    names no kind of loan.
    """
    if row.dcid not in menu.hits:
        return row
    words = [word.text for word in word_tokens(menu.text)]
    shared = index.words_shared(row.dcid, words)
    broadest = row
    for hit in menu.hits:
        if (
            hit != broadest.dcid
            and broadest.dcid in index.descendants(hit)
            and shared <= index.words_shared(hit, words)
            and (ancestor := index.code_by_dcid(hit)) is not None
        ):
            broadest = ancestor
    return broadest


def _unchosen_matches(
    matches: Sequence[QualifierMatch], chosen: Sequence[QualifierMatch]
) -> tuple[QualifierMatch, ...]:
    """`matches` less each term a code in `chosen` (`_menu_choices`) reads in place of
    (`_reads_in_place_of`)."""
    return tuple(
        match
        for match in matches
        if not any(_reads_in_place_of(choice, match) for choice in chosen)
    )


def _reads_in_place_of(choice: QualifierMatch, term: QualifierMatch) -> bool:
    """Whether the menu choice `choice` reads in place of `term`: a term on the choice's
    code system inside the choice's phrase."""
    return (
        term.code_system == choice.code_system
        and term.start is not None
        and term.end is not None
        and choice.start is not None
        and choice.end is not None
        and choice.start <= term.start
        and term.end <= choice.end
    )


def _chosen_spans(state: ResolveState, index: CodeIndex) -> list[QualifierSpan]:
    """Each phrase a menu choice reads (`ResolveState.menu_choices`) as a span naming the
    chosen code."""
    return [
        QualifierSpan(
            matched_text=choice.matched_text, start=choice.start, end=choice.end, rows=(row,)
        )
        for choice in state.menu_choices
        if choice.start is not None
        and choice.end is not None
        and (row := index.code_by_dcid(choice.code_dcid)) is not None
    ]


def _chosen_ranges(state: ResolveState) -> list[tuple[int, int]]:
    """The offsets of each phrase a menu choice reads (`ResolveState.menu_choices`)."""
    return [
        (choice.start, choice.end)
        for choice in state.menu_choices
        if choice.start is not None and choice.end is not None
    ]


class _CoverageReading(NamedTuple):
    """The coverage rule's records (`_coverage_records`) and each word of the variable
    with its verdict, in query order (`_coverage_reading`)."""

    records: list[ResolvedQualifier]
    verdicts: tuple[tuple[str, str], ...]


def _coverage_records(
    state: ResolveState, recorded: Sequence[ResolvedQualifier]
) -> list[ResolvedQualifier]:
    """The variable's words no reader covers on the elected shape, each run as one record
    with evidence ``"uncovered"``. They catch restrictions the answer would otherwise drop
    silently. A run beside a recognised code is ``unbound``, naming that code's rows:
    "maternal" in "maternal health ODA" binds nothing beside health, sector 120. In any
    other run the readers only nominate, and the binder rates how likely the shape's
    series states each word (`ResolveState.stated_word_ratings`). Each stretch of words it
    rates below `SearchSettings.stated_word_bar` is ``no_vocabulary``, naming none:
    "refugee" in "refugee population" on the count of all persons, "tax" in "tax revenue
    as a percent of GDP" on total revenue. A word it rates at the bar or above, or does
    not rate, is read. Either way a record refuses the part, and the reading it found is
    its flagged match. A shape none of whose members has a name states no words of its
    measure, so on it only a run beside a code is a record.

    The readers are the spans of the elected shape's own reading (code and measure
    values), the terms in `recorded`, a phrase a menu choice reads among them
    (`_model_records`), the words naming a value a shape on the menu holds, which each
    answer is judged on instead (`value_requirements`), the part's entity texts and the
    spans of their door roles ("German" for Germany as origin), its operation phrases, its unit text, the series code it names (`ResolveState.source_code`),
    a date phrase (`time_bind.parse_time_text`), the words `_conversational_words`
    accepts ("total" among them), the words of the label and synonyms of each code the
    final bindings apply that no span of the reading and no menu choice names
    (`_unnamed_bound_code_words`), so "private" in "private grants" is read when the
    answer binds flow type 30, "private development finance", the words naming the
    shape's measure in its members' names (`_member_measure_words`), so "refugees" is read
    on a series named "Refugee population", on a shape measuring a count
    (`shape.measures_count`) the nouns for what it counts (`COUNT_WORDS`), so "number" is
    read on "Refugees, by host country", on one counting persons (`shape.counts_persons`)
    the nouns for people (`PERSON_WORDS`), so "population" is read on "Number of refugees,
    by host country" and "people" is left for the binder on a count of households, and a
    noun naming a role on an axis the shape declares (`role_text.role_axis_for_text`), so
    "creditors" is read on debt by creditor. The form wording (`_form_ranges`: "rate", "share",
    "percent", "per capita") and the question's words for the denominator the quantity
    read names (`NamedDenominator.base_words`) are read, since the materialise gate holds
    the elected shape to both (`quantity.serves_quantity`) and states a denominator it
    cannot verify (`denominator_unverified`). So are the words of each value the bind
    drafted on a slot the shape lacks, which each answer is judged on
    (`ResolveState.stray_requirements`), and the phrase of each slot the final bindings open
    (``requested_all``, the bind's claim or `_open_operation_slots` and
    `_open_breakdown_slots`), whose every value the answer serves: "quintile" in "by
    wealth quintile" on the "Wealth quantile" slot. A connective
    (`qualifier_match.CONNECTIVES`) between two code spans is read. A run of unread words
    is adjacent words with no separator between them (`_unread_runs`). Its neighbours lie
    in a code span when the token beside it on either side does, or is a union connective
    (`UNION_CONNECTIVES`) with a code span beyond it ("health and girls"). A phrase a menu
    choice reads is a code span naming the chosen code (`_chosen_spans`), so "maternal" in
    "maternal and infant health ODA" is a record when the model chose 12240 for "infant
    health". A record names the rows of the code spans beside it, else of those a union
    connective joins it to.

    A run of the words of a code term read in every code system together that the shape
    reads only in part (`_unread_qualifier_runs`) is a record too, naming that term's rows,
    as "multilateral" in "multilateral ODA" on a CRS shape, which reads "ODA" alone. Such a
    run is no record when the name of an unnamed bound code holds every word of it, as
    "loans" in "ODA loans", DAC2a aid type 218, under type of finance 420.

    The exclusion and comparison phrases the coverage mode decides
    (`_unsupported_records` with `gated`) come first, each ``no_vocabulary`` with evidence
    ``"unsupported"``, and are read like the terms in `recorded`.
    """
    return _coverage_reading(state, recorded).records


def _coverage_reading(
    state: ResolveState, recorded: Sequence[ResolvedQualifier]
) -> _CoverageReading:
    """`_coverage_records` with the verdict of each word of the variable, in query order:
    the name of the reader that reads it (`_Reading.verdicts`), else, in a run beside no
    code span on a shape with a named member, ``"stated"`` when the binder rates the
    shape's series stating it at `SearchSettings.stated_word_bar` or above
    (`ResolveState.stated_word_ratings`), ``"unrated"`` when it gives the word no rating,
    which counts as stated, and ``"unread"`` when it rates it below, else ``"unread"``."""
    index = get_index()
    variable = state.variable
    shape = state.shape_draft
    combination = _combine_matches(
        variable, (*state.admitted_qualifiers, *state.menu_choices), index
    )
    unsupported = _unsupported_records(state, combination, recorded, gated=True)
    recorded = (*recorded, *unsupported)
    tokens = tokenize(variable)
    spans = for_shape(state.variable_spans_by_vocabulary, shape)
    code_spans = [span for span in spans if any(isinstance(row, CodeRow) for row in span.rows)]
    code_spans += _chosen_spans(state, index)
    free = _free_readers(state, shape, tokens, spans, index)
    ranges: list[_RangeReader] = [
        free.reading,
        ("recorded", _occurrences(tokens, (qualifier.text for qualifier in recorded))),
        free.entity,
        free.operation,
        free.unit,
        free.source_code,
        (
            "value",
            [
                (requirement.term.start, requirement.term.end)
                for requirement in value_requirements(state, recorded)
                if requirement.term.start is not None and requirement.term.end is not None
            ],
        ),
        free.form,
        free.base,
        (
            "stray",
            _occurrences(
                tokens, (requirement.term.text for requirement in state.stray_requirements)
            ),
        ),
        (
            "opened",
            _occurrences(
                tokens,
                (
                    binding.requested_all_phrase
                    for binding in state.bindings
                    if binding.evidence == "requested_all"
                ),
            ),
        ),
    ]
    named = {
        *(row.dcid for span in code_spans for row in span.rows if isinstance(row, CodeRow)),
        *(choice.code_dcid for choice in state.menu_choices),
    }
    applied = _unnamed_bound_code_words(state.bindings, named, index, shape.value_labels or {})
    measure_words = _member_measure_words(state)
    reading = _unread_runs(
        variable,
        tokens,
        code_spans,
        ranges,
        [
            ("bound_code", applied),
            free.conversational,
            ("measure", _measure_reader(measure_words or frozenset())),
            free.count,
            free.role,
        ],
    )
    verdicts = dict(reading.verdicts)
    ratings = state.stated_word_ratings
    bar = settings().stated_word_bar
    records: list[ResolvedQualifier] = []
    for run in reading.runs:
        if run.adjacent or run.joined:
            start, end = tokens[run.first].start, tokens[run.last].end
            records.append(
                _uncovered_record(variable[start:end], start, end, run.adjacent or run.joined)
            )
            continue
        if measure_words is None:
            continue
        # A word beside no code is unread only on the binder's rating, so each stretch of
        # the run it rates below the bar is one record.
        unread: list[tuple[int, int]] = []
        for position in range(run.first, run.last + 1):
            rating = ratings.get(tokens[position].text.casefold())
            if rating is None or rating >= bar:
                verdicts[position] = "unrated" if rating is None else "stated"
            elif unread and unread[-1][1] == position - 1:
                unread[-1] = (unread[-1][0], position)
            else:
                unread.append((position, position))
        for first, last in unread:
            start, end = tokens[first].start, tokens[last].end
            records.append(_uncovered_record(variable[start:end], start, end, ()))

    records.extend(
        _uncovered_record(variable[run.start : run.end], run.start, run.end, [run.span])
        for run in _unread_qualifier_runs(state, (*recorded, *records))
        if not all(
            applied(word.text.casefold()) for word in word_tokens(variable[run.start : run.end])
        )
    )
    return _CoverageReading(
        [*unsupported, *records],
        tuple((tokens[position].text, verdicts[position]) for position in sorted(verdicts)),
    )


def _occurrences(tokens: Sequence[Token], texts: Iterable[str | None]) -> list[tuple[int, int]]:
    """The character range of each occurrence among `tokens` of each text of `texts`
    (`_token_occurrences`), skipping an empty or absent one."""
    return [occurrence for text in texts if text for occurrence in _token_occurrences(tokens, text)]


class _FreeReaders(NamedTuple):
    """The readers of the variable's words on one shape that read no binding
    (`_free_readers`), each under the name its verdict gives the words it reads."""

    reading: _RangeReader
    entity: _RangeReader
    operation: _RangeReader
    unit: _RangeReader
    source_code: _RangeReader
    form: _RangeReader
    base: _RangeReader
    conversational: _WordReader
    count: _WordReader
    role: _WordReader


def _free_readers(
    state: ResolveState,
    shape: ShapeDraft,
    tokens: Sequence[Token],
    spans: Sequence[QualifierSpan],
    index: CodeIndex,
) -> _FreeReaders:
    """The readers of the variable's words, `tokens`, that read neither a binding nor the
    bind's reply, on `shape`: the spans `spans` of its own reading, the part's entity
    texts and the spans of their door roles, its operation phrases, its unit text, the
    series code it names, the form wording (`_form_ranges`), the question's words for the
    denominator the quantity read names, the words `_conversational_words` accepts for
    `shape`, on a shape measuring a count (`shape.measures_count`) the nouns for what it
    counts (`COUNT_WORDS`) and, on one counting persons (`shape.counts_persons`), the nouns
    for people (`PERSON_WORDS`), and a noun naming a role on an axis `shape` declares
    (`role_text.role_axis_for_text`). The coverage rule reads them after the bind
    (`_coverage_reading`), and the nomination of each menu shape's words before it
    (`_nominated_words`)."""
    named_base = state.named_denominator
    counts = measures_count(shape)
    persons = counts_persons(shape)
    try:
        axes = frozenset(role_names_by_axis(shape))
    except ValueError:
        axes = frozenset()
    return _FreeReaders(
        reading=("reading", [(span.start, span.end) for span in spans]),
        entity=(
            "entity",
            _occurrences(
                tokens,
                [
                    *state.entities,
                    *(role.span for roles in state.entity_roles.values() for role in roles),
                ],
            ),
        ),
        operation=(
            "operation",
            _occurrences(tokens, (operation.phrase for operation in state.operations)),
        ),
        unit=("unit", _occurrences(tokens, [state.unit_text])),
        source_code=(
            "source_code",
            _occurrences(tokens, [state.source_code.code if state.source_code else None]),
        ),
        form=("form", _form_ranges(state.variable, tokens)),
        base=("base", _occurrences(tokens, [named_base.base_words if named_base else None])),
        conversational=("conversational", _conversational_words(state, index, shape)),
        count=(
            "count",
            lambda word: (counts and word in COUNT_WORDS) or (persons and word in PERSON_WORDS),
        ),
        role=("role", lambda word: role_axis_for_text(fold_word(word)) in axes),
    )


def _measure_reader(words: AbstractSet[str]) -> Callable[[str], bool]:
    """Whether a casefolded word is one of the folded name words `words`
    (`_member_measure_words`), folded as the code index folds words (`fold_word`), or a
    hyphenated word each of whose parts is. Only equal folded words read each other, so
    "literacy" stays unread on a series named "Population: Literate", and "coefficient" on
    one named "Gini index", and the binder rates each (`_coverage_reading`):
    a reader that reads too much hides a restriction the series drops, and one that reads
    too little costs a rating."""

    def measure(word: str) -> bool:
        return fold_word(word) in words or (
            "-" in word and all(fold_word(part) in words for part in word.split("-"))
        )

    return measure


def _member_names(shape: ShapeDraft) -> dict[str, frozenset[str]]:
    """The words of the name of each named member `shape` stands for
    (`shape_population_arcs`, `_label_words`), so a held family's members outside the
    recall sample read their names too."""
    return {
        dcid: _label_words(name)
        for dcid, arcs in shape_population_arcs(shape).items()
        if (name := read_name(arcs)) is not None
    }


def _shape_measure_words(shape: ShapeDraft) -> frozenset[str] | None:
    """The words every named member `shape` stands for holds in its name
    (`_member_names`), which the words of any selection of its members hold too
    (`_member_measure_words`). None when no member has a name."""
    names = _member_names(shape).values()
    return frozenset.intersection(*names) if names else None


def _nominated_words(state: ResolveState, shape: ShapeDraft) -> tuple[str, ...]:
    """The casefolded words of the variable that the readers reading no binding leave
    unread on menu shape `shape` before the bind (`_free_readers`, and the words every
    named member of `shape` holds in its name, `_shape_measure_words`), once each, in
    query order. Empty on a shape none of whose members has a name, where only a run
    beside a code is a record.

    Every word the coverage rule leaves unrated on `shape` once elected
    (`_unrated_words`) is among them. The rule reads with these readers and more: the
    bindings' code words, the recorded terms, the value and stray requirements, the code
    spans the bind's menu choices add, and the names of a subset of these members, which
    hold every word all of them hold."""
    words = _shape_measure_words(shape)
    if words is None:
        return ()
    variable = state.variable
    tokens = tokenize(variable)
    spans = for_shape(state.variable_spans_by_vocabulary, shape)
    code_spans = [span for span in spans if any(isinstance(row, CodeRow) for row in span.rows)]
    free = _free_readers(state, shape, tokens, spans, get_index())
    reading = _unread_runs(
        variable,
        tokens,
        code_spans,
        [
            free.reading,
            free.entity,
            free.operation,
            free.unit,
            free.source_code,
            free.form,
            free.base,
        ],
        [free.conversational, ("measure", _measure_reader(words)), free.count, free.role],
    )
    return tuple(
        dict.fromkeys(
            tokens[position].text.casefold()
            for position, verdict in sorted(reading.verdicts.items())
            if verdict == "unread"
        )
    )


def _unrated_words(state: ResolveState) -> tuple[str, ...]:
    """The casefolded words the coverage rule leaves unrated on the elected shape over the
    final bindings (`_coverage_reading` over `_coverage_terms`), once each, in query
    order: the words of a run beside no code span that no reader reads, which the binder
    rates (`phase_election._rate_unread_words`)."""
    reading = _coverage_reading(state, _coverage_terms(state))
    return tuple(
        dict.fromkeys(word.casefold() for word, verdict in reading.verdicts if verdict == "unrated")
    )


def _coverage_terms(state: ResolveState) -> tuple[ResolvedQualifier, ...]:
    """The terms the coverage rule reads as recorded: the part's query records before the
    rule's own (`_query_records`) and, in log mode, the withheld ones
    (`_withheld_qualifiers`)."""
    return (*_query_records(state), *_withheld_qualifiers(state))


def _form_ranges(variable: str, tokens: Sequence[Token]) -> list[tuple[int, int]]:
    """The character ranges of the form wording in `variable`, whose tokens are `tokens`:
    each word that names only a form (`quantity.names_only_form`: "rate", "share",
    "percent", "capita", "density"), and each numbered per-head base
    (`PER_HEAD_RATE_BASE_RE`: "per 100,000 people")."""
    ranges = [
        (token.start, token.end)
        for token in tokens
        if token.kind == "word"
        and token.text.casefold() not in FUNCTION_WORDS
        and names_only_form(token.text)
    ]
    ranges += [match.span() for match in PER_HEAD_RATE_BASE_RE.finditer(variable.lower())]
    return ranges


def _member_measure_words(state: ResolveState) -> frozenset[str] | None:
    """The words that name the elected shape's measure in the names of the members the
    answer serves over the bindings (`_label_words`), so "refugees" is among the words of
    "Refugee population". The members are those `served_member_names` reads: "GDP" in
    Brazil reads "Nominal gross domestic product", the one member of a family with no total
    holding data there. A family's total answers before its breakdowns, so "refugee" in
    "refugee population" is left for the binder on a family holding the total and a refugee
    member, and a held family's total reads its name though recall found the refugee member
    alone. The words are those every served member holds, which the words the answer reads
    hold whichever of them it shows. When a served member has no name, as a code index row
    recall did not return has none, or none is served, they are the words every named
    member holds (`_shape_measure_words`), the fewest any name of the family reads, so the
    named members alone clear no word. None when no member has a name."""
    shape = state.shape_draft
    if not _member_names(shape):
        return None
    served = served_member_names(state).values()
    if not served or None in served:
        return _shape_measure_words(shape)
    return frozenset.intersection(*(_label_words(name) for name in served if name is not None))


def served_member_names(state: ResolveState) -> dict[str, str | None]:
    """The name of each member the answer over the elected shape and the final bindings
    can serve, None for a member with no name, by dcid in serving order: the members
    `standard.members_served` gives by the rule the resolve serves by, read over the data
    the part's availability reads found for each (`member_has_data_where_asked`). A member
    no read covered is kept, so a failed read costs a rating and clears no word. The places
    bind under the assignment place binding settled (`ResolveState.elected_place_roles`),
    a direction among them, else under the one binding repair took
    (`ResolveState.query_place_roles`). The members are read once for each shape, bindings
    and assignment (`ResolveState.served_member_memo`), as every coverage reading and the
    rating it asks for read them."""
    shape, bindings = state.shape_draft, tuple(state.bindings)
    assignment = state.elected_place_roles or state.query_place_roles
    memo = state.served_member_memo
    if memo is not None and memo[0] is shape and memo[1] == bindings and memo[2] is assignment:
        return memo[3]
    role_keys = assignment.role_keys if isinstance(assignment, PlaceAssignment) else RoleKeys()
    arcs = shape_population_arcs(shape)
    names = {
        dcid: read_name(arcs.get(dcid, {}))
        for dcid in members_served(
            shape, bindings, role_keys, member_has_data_where_asked(state, shape)
        )
    }
    state.served_member_memo = (shape, bindings, assignment, names)
    return names


def _label_words(label: str) -> frozenset[str]:
    """The words of `label`, folded as the code index folds words (`fold_word`), with the
    initials of each run of two or more of its content words (`_initialisms`), so "GDP" is
    among the words of "Nominal gross domestic product"."""
    return frozenset(fold_word(word.text) for word in word_tokens(label)) | _initialisms(label)


def _initialisms(name: str) -> frozenset[str]:
    """The casefolded initials of each run of two or more adjacent content words of `name`,
    less `FUNCTION_WORDS`: "gdp" and "dp" in "Gross Domestic Product", "oda" in "Official
    Development Assistance"."""
    initials = [
        word.text[0].casefold()
        for word in word_tokens(name)
        if word.text.casefold() not in FUNCTION_WORDS and word.text[0].isalpha()
    ]
    return frozenset(
        "".join(initials[first:last])
        for first in range(len(initials))
        for last in range(first + 2, len(initials) + 1)
    )


def model_only_values(
    state: ResolveState, prop: str, values: Sequence[str], labels: Mapping[str, str]
) -> tuple[str, ...]:
    """The `values` bound on `prop` that no recorded term of the request supports, so only
    the bind model read them from the words. A term supports every value on `prop` when it
    is an admitted qualifier match on `prop` or a query record a binding on it covers, and
    supports one value when one of its names (`_value_names`) is a run of the variable,
    a dropped qualifier phrase or the single-variable query, so "women" names Female by
    its synonym. Each source is read separately. A value with no name is the model's. An
    answer that cannot apply a supported value refuses `qualifier_not_bound` on it, and one
    that cannot apply a model-only value discloses it (`unrealisable.unbind_unrealisable`,
    `_stray_requirements`)."""
    if any(match.code_system == prop for match in state.admitted_qualifiers) or any(
        record.slot == prop and record.disposition == "bound" for record in state.query_qualifiers
    ):
        return ()
    texts = [state.variable, *state.dropped_qualifier_phrases]
    if query_text := state.query_text:
        texts.append(query_text)
    sources = [tokenize(text) for text in dict.fromkeys(texts)]
    return tuple(
        value
        for value in values
        if not any(
            _token_occurrences(tokens, name)
            for tokens in sources
            for name in _value_names(prop, value, labels.get(value))
        )
    )


def names_value(phrase: str, *, prop: str, value: str, label: str | None) -> bool:
    """Whether `phrase` names `value` on `prop`: one of the value's names (`_value_names`)
    holds the phrase's words (`holds_words`). "bilateral" names DAC1 "bilateral ODA", and
    "concessional" names no "official non-concessional flow"."""
    return any(holds_words(name, phrase) for name in _value_names(prop, value, label))


def holds_words(text: str, phrase: str) -> bool:
    """Whether `text` holds `phrase`'s words as a run, read as a code label reads a query
    span (`_token_occurrences`): "imputed multilateral aid" holds "multilateral", and
    "non-concessional flows" holds no "concessional", since a hyphenated word is one
    token."""
    return bool(_token_occurrences(tokenize(text), phrase))


def _value_names(prop: str, value: str, label: str | None) -> tuple[str, ...]:
    """Every name the qualifier matcher reads for `value` on `prop`, and its graph `label`:
    the code index's label and synonyms of the code, and the label and synonyms of each
    matchable measure value on `prop` that spells its reading, aliases included."""
    code = get_index().code_by_dcid(value)
    measure_index = get_measure_index()
    reading = measure_index.reading_set(value)
    return (
        *(() if label is None else (label,)),
        *(() if code is None else (code.label, *code.synonyms)),
        *(
            text
            for row in measure_index.values
            if row.axis == prop and row.matchable and row.dcid in reading
            for text in (row.label, *row.synonyms)
        ),
    )


def value_requirements(
    state: ResolveState, recorded: Sequence[ResolvedQualifier]
) -> tuple[Requirement, ...]:
    """Each run of the variable's words that names, by its graph label
    (`ShapeDraft.value_labels`), a constraint value a member of a shape on the bind menu
    (`ResolveState.capped_shapes`) or of the elected shape holds (`shape_constraint_values`),
    as a requirement each
    final answer is judged against (`judge_value_requirements`). "female" in "female
    literacy rate" names Female, which a gender-by-literacy shape on the menu holds,
    whatever the bind model lists as unplaced. Its term is the ``no_vocabulary`` record a
    refusal names on an answer whose shape has no slot for the value
    (`requirements._value_term`), and it departs `measure_not_applied` under the run's own
    words. A value whose label the holding shape's series states (`_series_words`), and
    which every member of that shape's whole family holds (`_family_constraints`), names
    the shape's measure and restricts nothing: Skill Literacy, the one skill of the UN
    "Youth/adult literacy rate", names no value of "literacy rate", which a series
    measuring the literate share answers. A value the family splits on restricts it even
    where its series states the label, as "In extreme poverty" on the ECLAC "Population
    living in extreme poverty and poverty", whose members hold In poverty too, and so does
    a value of a shape whose whole family no source lists, since its recalled sample can
    show one value of a split.

    Words other readers own name nothing here: the elected shape's own reading, the terms
    in `recorded`, the part's entities and their spans (`_entity_spans`: "German" in
    "German aid" is the origin Germany), and every measure term the measure index reads. A value
    the code index knows (`CodeIndex.code_by_dcid`) is the code readers', so "health" is
    the reading of purpose 120 alone. A requirement keeps the code systems of the code
    terms its words overlap, read in every code system, for the judge. A value a negated or
    excluded occurrence names (`_label_has_negated_occurrence`) is left to the exclusion
    readers. A value whose words lie strictly inside the words of another value of its
    property is no requirement (`_outside_wider_names`): "public or publicly guaranteed"
    names that value, and not also Publicly guaranteed.
    """
    index = get_index()
    variable = state.variable
    tokens = tokenize(variable)
    asked: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for menu_shape in (*state.capped_shapes, state.shape_draft):
        labels = menu_shape.value_labels or {}
        measured = _series_words(menu_shape)
        family: list[dict[str, str]] | None = None
        for prop, value in shape_constraint_values(menu_shape):
            label = labels.get(value)
            if label is None or index.code_by_dcid(value) is not None:
                continue
            if all(_holds_word(measured, word) for word in _casefolded_words(label)):
                if family is None:
                    family = _family_constraints(menu_shape)
                if family and all(member.get(prop) == value for member in family):
                    continue
            asked[label].add((prop, value))
    if not asked:
        return ()

    read = [
        (span.start, span.end)
        for span in for_shape(state.variable_spans_by_vocabulary, state.shape_draft)
    ]
    read += [
        occurrence
        for qualifier in recorded
        for occurrence in _token_occurrences(tokens, qualifier.text)
    ]
    read += _entity_spans(state, tokens)
    code_spans: list[tuple[tuple[int, int], frozenset[str]]] = []
    for span in state.variable_spans_in_every_system:
        if all(isinstance(row, CodeRow) for row in span.rows):
            systems = frozenset(row.code_system for row in span.rows if isinstance(row, CodeRow))
            code_spans.append(((span.start, span.end), systems))
        else:
            read.append((span.start, span.end))

    named: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    for label, values in asked.items():
        occurrences = [
            occurrence
            for occurrence in _token_occurrences(tokens, label)
            if not any(_ranges_overlap(occurrence, other) for other in read)
        ]
        if occurrences and not _label_has_negated_occurrence(label, variable):
            for occurrence in occurrences:
                named[occurrence] |= values
    named = _outside_wider_names(named)
    return tuple(
        Requirement(
            term=ResolvedQualifier(
                text=variable[start:end],
                named=[],
                slot=None,
                disposition="no_vocabulary",
                start=start,
                end=end,
            ),
            origin="value",
            rule=ValueRule(
                values=frozenset(values),
                code_systems=frozenset().union(
                    *(
                        systems
                        for span, systems in code_spans
                        if _ranges_overlap((start, end), span)
                    )
                ),
            ),
            departure=unapplied_departure(
                variable[start:end], asked=variable[start:end], property_dcid=None
            ),
        )
        for (start, end), values in sorted(named.items())
    )


def _outside_wider_names(
    named: Mapping[tuple[int, int], set[tuple[str, str]]],
) -> dict[tuple[int, int], set[tuple[str, str]]]:
    """`named` without each value whose run lies strictly inside the run of another value
    of the same property: the words of "public or publicly guaranteed" name that value
    alone, and the "publicly guaranteed" inside them name no second one. The run names a
    value of another property still. Runs that end empty are dropped."""

    def inside(run: tuple[int, int], wider: tuple[int, int]) -> bool:
        return run != wider and wider[0] <= run[0] and run[1] <= wider[1]

    kept: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for run, values in named.items():
        wider_props = {
            prop
            for other, other_values in named.items()
            if inside(run, other)
            for prop, _value in other_values
        }
        own = {(prop, value) for prop, value in values if prop not in wider_props}
        if own:
            kept[run] = own
    return kept


def _family_constraints(shape: ShapeDraft) -> list[dict[str, str]] | None:
    """The constraints of every member of `shape`'s family, or `None` when no source lists
    them all. The sources: the shape's own members when they are complete
    (`ShapeDraft.sv_arc_facts_complete`), the code index's rows of the family its part
    may answer from, or the family the catalogue holds whole for the shape
    (`held_family_exact`), both read by `five_tuple_member_facts`, else the origin
    catalogue's UN tables,
    which hold every admitted member of a UN series (`Catalogue.un_series_members`) with
    all its arcs, narrowed to the shape's five-tuple. A recalled sample is no source."""
    if shape.sv_arc_facts_complete:
        return [read_constraints(arcs) for arcs in (shape.sv_arc_facts or {}).values()]
    rows, index_sourced = five_tuple_member_facts(shape)
    if index_sourced or held_family_exact(shape):
        return [row.read_constraints() for row in rows.values()]
    catalogue = current_catalogue()
    five_tuple = shape_five_tuple(shape)
    members = [
        arcs
        for member in catalogue.un_series_members(shape.pop_type_dcid)
        if (variable := catalogue.un_variable(member)) is not None
        and read_five_tuple(arcs := variable.wire_arcs()) == five_tuple
    ]
    return [read_constraints(arcs) for arcs in members] or None


def judge_value_requirements(
    requirements: Sequence[Requirement],
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    state: ResolveState | None = None,
) -> tuple[Requirement, ...]:
    """The requirements of `requirements` judged by the values their words name
    (`ValueRule`, `value_requirements`) that an answer of `shape` read under `bindings`
    leaves unmet.

    The answer is the members of `shape` its bindings select (`_selected_members`). It
    applies a value its bindings bind, or one every selected member holds, as every member
    of a female population shape holds gender Female, where a code's descendant applies it
    as a binding of the code reads (`applies_value`). The values the bind drafted on a
    property the elected reading lacks (origin ``stray``) are a set, applied by membership
    as a set bound on a slot reads: a binding holds one of them, or every selected member
    does (`applies_one_of`). One value of such a set (origin ``set_value``) is applied when
    a binding or some selected member holds it. Both are judged by values alone: a word of
    a member's name states neither. Any other requirement is also met by a run every word
    of which its series states (`_series_words`), as a Student population states "student", or each
    selected member's name holds, as "Infant Mortality Rate", the name of
    `Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth`, holds "infant" though
    the member binds age 0 and not the age group Infant. A run overlapping a code term of
    a system `shape` holds (`shape_vocabulary`) is its code readers'. Any other run is
    unmet: "female" when the answering literacy rate has no gender slot. A word the
    answer's datasets state of the rows the part reads (`_dataset_stated_words`) is stated
    too: "bilateral", the creditor type of a debt series on the menu, is met by a CRS
    answer. A broad reading that fired on the part (`SlotDefaultDisclosure.reading`) applies
    its narrow value wherever the bindings hold its broader one: "publicly guaranteed" is
    applied by public and publicly guaranteed debt, which the part's note says it read. A
    resubmit gives no `state` and reads none.
    """
    if not requirements:
        return ()
    held = shape_vocabulary(shape)
    bound = {
        (binding.property_dcid, value)
        for binding in bindings
        if binding.kind in ("value", "set")
        for value in binding.value_dcids
    }
    selected = _selected_members(shape, bindings)
    constraints = [read_constraints(arcs) for arcs in selected]
    names = [_casefolded_words(read_name(arcs) or "") for arcs in selected]
    stated = _series_words(shape) | _dataset_stated_words(shape, get_index(), state)
    read_broadly = {
        (reading.property_dcid, reading.value_dcid)
        for disclosure in (state.slot_defaults.values() if state is not None else ())
        if (reading := disclosure.reading) is not None
        and (reading.property_dcid, reading.reads_as) in bound
    }

    def applied(prop: str, value: str) -> bool:
        return (
            (prop, value) in read_broadly
            or any(bound_prop == prop and applies_value(value, held) for bound_prop, held in bound)
            or (
                bool(constraints)
                and all(applies_value(value, member.get(prop)) for member in constraints)
            )
        )

    def applied_as_a_set(values: frozenset[tuple[str, str]]) -> bool:
        # The values the bind drafted on one property the elected reading lacks.
        prop = next(iter(values))[0]
        named = [value for _prop, value in values]
        return any(
            bound_prop == prop and applies_one_of(named, held) for bound_prop, held in bound
        ) or (
            bool(constraints)
            and all(applies_one_of(named, member.get(prop)) for member in constraints)
        )

    def held_by_some(prop: str, value: str) -> bool:
        return any(
            bound_prop == prop and applies_value(value, held) for bound_prop, held in bound
        ) or any(applies_value(value, member.get(prop)) for member in constraints)

    def states(words: frozenset[str]) -> bool:
        return all(_holds_word(stated, word) for word in words) or (
            bool(names) and all(words <= name for name in names)
        )

    def met(requirement: Requirement, rule: ValueRule) -> bool:
        if requirement.origin == "stray":
            return applied_as_a_set(rule.values)
        if requirement.origin == "set_value":
            return any(held_by_some(prop, value) for prop, value in rule.values)
        return any(applied(prop, value) for prop, value in rule.values) or states(
            _casefolded_words(requirement.term.text)
        )

    return tuple(
        requirement
        for requirement in requirements
        if isinstance(rule := requirement.rule, ValueRule)
        and not rule.code_systems & held
        and not met(requirement, rule)
    )


def _selected_members(shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]) -> list[dict]:
    """The arcs of each member `shape` stands for (`shape_population_arcs`) that
    `bindings` select, as materialisation selects them (`bound_values_from_bindings`): one
    holding a value of each property they bind as a value or set, or a descendant of one, and
    none of each property they bind as absent."""
    bound = {
        prop: frozenset(values) for prop, values in bound_values_from_bindings(bindings).items()
    }
    absent = {
        binding.property_dcid
        for binding in bindings
        if binding.property_dcid is not None and binding.kind == "absent"
    }
    selected: list[dict] = []
    for arcs in shape_population_arcs(shape).values():
        member = read_constraints(arcs)
        if all(member.get(prop) in values for prop, values in bound.items()) and not (
            absent & member.keys()
        ):
            selected.append(arcs)
    return selected


def _unnamed_bound_code_words(
    bindings: Sequence[SlotBindingDraft],
    named: AbstractSet[str],
    index: CodeIndex,
    value_labels: Mapping[str, str],
) -> Callable[[str], bool]:
    """Whether the label or a synonym of a code `bindings` bind as a value or a set holds a
    casefolded word (`CodeIndex.words_shared`), counting only codes outside `named`. `named`
    holds the codes a span of the query's own reading or a menu choice names. A value the
    index holds no row for reads by the words of its graph label (`value_labels`, else its
    dcid split into words, `_label_words`), so a bound `Female` reads "female".

    Those codes come from the bind model or a declared default, with no phrase of the
    query standing for them, so their names read the words the answer applies: flow type
    30, "private development finance", reads "private" in "private grants", and type of
    finance 420, synonym "loan instrument", reads "loans". A code a span names reads only
    its span, so 130 for "reproductive health" leaves "policy" in "education policy"
    unread, though its label holds "policy".
    """
    unnamed = [
        dcid
        for binding in bindings
        if binding.kind in ("value", "set")
        for dcid in binding.value_dcids
        if dcid not in named
    ]

    unindexed = frozenset().union(
        *(
            _label_words(_humanize_ref(dcid, value_labels.get(dcid)))
            for dcid in unnamed
            if index.code_by_dcid(dcid) is None
        )
    )

    def applied(word: str) -> bool:
        return fold_word(word) in unindexed or any(
            index.words_shared(dcid, (word,)) for dcid in unnamed
        )

    return applied


def _code_ref(dcid: str, index: CodeIndex) -> GraphRef:
    """The `GraphRef` of the code `dcid`, under its index label."""
    row = index.code_by_dcid(dcid)
    return GraphRef(ref=dcid, label=row.label if row is not None else dcid)


def _unsupported_records(
    state: ResolveState,
    combination: _Combination,
    recorded: Sequence[ResolvedQualifier],
    *,
    gated: bool,
) -> list[ResolvedQualifier]:
    """Each exclusion or comparison phrase of the variable (`_Combination.unsupported`)
    that `recorded` holds no record of, as a ``no_vocabulary`` record with evidence
    ``"unsupported"``. No binding expresses "excluding malaria control" or "versus
    education".

    `gated` selects the phrases the coverage mode decides (`_coverage_records`): every
    exclusion, and a comparison other than one of two codes of one system that do not
    nest. Without them a nested comparison answers its narrower code and a phrase naming
    one code answers it, while an exclusion still refuses on the excluded code's unbound
    record (`_combine_matches` binds the code before the connective). The rest refuse in
    every mode, since "compare aid to health and education" would otherwise bind the set
    {120, 110} as a union.
    """
    already = {_normalize(record.text) for record in recorded}
    records: list[ResolvedQualifier] = []
    for connective, compares_two in combination.unsupported:
        if compares_two == gated:
            continue
        text = state.variable[connective.start : connective.end]
        if _normalize(text) in already:
            continue
        already.add(_normalize(text))
        records.append(
            ResolvedQualifier(
                text=text,
                named=[],
                slot=None,
                disposition="no_vocabulary",
                start=connective.start,
                end=connective.end,
                evidence="unsupported",
            )
        )
    return records


def _uncovered_record(
    text: str, start: int, end: int, spans: Sequence[QualifierSpan]
) -> ResolvedQualifier:
    """The record of the uncovered words `text` at `start`..`end`: ``unbound``, naming the
    code rows of `spans` once each, or ``no_vocabulary`` when `spans` name none."""
    named = {
        row.dcid: GraphRef(ref=row.dcid, label=row.label)
        for span in spans
        for row in span.rows
        if isinstance(row, CodeRow)
    }
    return ResolvedQualifier(
        text=text,
        named=list(named.values()),
        slot=None,
        disposition="unbound" if named else "no_vocabulary",
        start=start,
        end=end,
        evidence="uncovered",
    )


def _resolve_query_qualifiers(state: ResolveState) -> list[ResolvedQualifier]:
    """Build qualifier dispositions from final bindings without mutating state.

    The part's sources have no record here. Candidacy refused the part when its scope
    left no recalled shape, and the read and the identity's pin apply it otherwise.
    Ten sources contribute records:

    1. The variable's code terms the elected shape cannot apply (`_unavailable_qualifiers`),
       each ``unbound`` with evidence ``"unavailable"``. They are terms naming only codes no
       shape can bind whose words the shape's own reading leaves unread and, on a part
       naming a source, terms none of its rows holds. A part naming a source records them
       in every coverage mode. A part naming none records the unbindable terms in refuse
       mode only. A span of the shape's own reading at the same offsets, an
       unplaced phrase overlapping one, and a tie or sidecar span of the same text record
       nothing more.
    2. Variable spans, as the elected shape's own vocabulary reads them
       (`ResolveState.variable_spans_by_vocabulary`), contribute their ``CodeRow`` values,
       all of systems the shape declares, so a CRS answer's "ODA" names the CRS flow type,
       not the DAC1 aid type of the same text. A word that reading leaves unread is no
       recognised span, so it can return as an ``unplaced`` phrase (source 4).
       Measure-value-only spans are handled by measure warnings. Disposition uses
       ``_qualifier_disposition``, and a span whose admitted match combined with another
       (`_combine_matches`) records the operation and the codes the two selected; a code a
       union joined to its ancestor is bound when the ancestor binds.
    3. ``state.admitted_ties`` records model-offered ambiguities unresolved by the final
       bindings. Keep ties with a row property declared by the elected shape, apply the
       shared disposition rule, and deduplicate normalized phrases against variable spans.
    4. An ``unplaced`` phrase must occur in ``variable`` outside recognised spans, must not
       be a per-person rate base the elected shape's denominator carries, and must not be, as
       a whole, the declared plain name of a slot the elected shape declares, singular or
       plural (`names_plain_name`): "sector" on a shape with `crsPurpose` names a distinction
       the data records, and its slot stays as bound. A longer phrase holding the noun ("space sector", "by sector") still refuses.
       Nor may every word of it occur in the label of a dataset the elected shape answers
       from, or in the words its datasets state (`_names_answering_dataset`): "imputed" on the
       imputed table chose that table, and "bilateral" on the CRS states what every row is.
       Surviving phrases get ``no_vocabulary`` with no named codes.
    5. Match the words each code system the shape declares has no vocabulary for
       (``CodeSystemDeclaration.no_vocabulary_terms``) in variable text and optional query
       text. Record ``no_vocabulary`` and refuse the qualifier.
    6. Typed-sidecar spans use the same admission and disposition rules, then deduplicate
       against text spans and ties. Their offsets belong to each sidecar phrase and stay out
       of the overlap check on ``variable``.
    7. An exclusion or comparison phrase that compares two codes of one system that do not
       nest (`_unsupported_records`), ``no_vocabulary`` with evidence ``"unsupported"``.
    8. Each phrase the bind model read through a code it chose from the phrase's candidate
       menu (`ResolveState.menu_choices`, `_model_records`), with evidence ``"model"``,
       naming the chosen codes. A variable span inside the phrase on a chosen code's
       system records nothing more, and no unavailable term inside it is recorded.
    9. Under ``SearchSettings.qualifier_coverage="refuse"``, the variable's words no reader
       covers beside a recognised code, each run ``unbound`` with evidence
       ``"uncovered"``, the words beside none that the binder rates unread, each stretch
       ``no_vocabulary`` with that evidence, and every other exclusion or comparison
       phrase (`_coverage_records`). In log mode election vetting logs them instead
       (`phase_election._repair_bindings`) and the answer reads without them.
    10. Each value the bind drafted on a property the elected shape has no slot for and a
       recorded term supports (`ResolveState.stray_requirements`), ``unbound`` with
       evidence ``"model"``, in every coverage mode, unless a record above already names
       that text.

    The words naming a value a shape on the bind menu holds are no record here: election
    vetting records them as value requirements (`value_requirements`), which each final
    answer is judged against on its own shape and bindings.
    """
    resolved = _query_records(state)
    if settings().qualifier_coverage == "refuse":
        resolved.extend(_coverage_records(state, resolved))
    already_recorded = {_normalize(q.text) for q in resolved}
    resolved.extend(
        requirement.term
        for requirement in state.stray_requirements
        if requirement.refuses and _normalize(requirement.term.text) not in already_recorded
    )
    return resolved


def _query_records(state: ResolveState) -> list[ResolvedQualifier]:
    """The records of sources 1 to 8 of `_resolve_query_qualifiers`, those the coverage
    rule reads as recorded (`_coverage_terms`)."""
    index = get_index()
    declared = shape_vocabulary(state.shape_draft)
    foreign = other_families_properties(state.shape_draft, index)
    # Each property's final value or set binding.
    bound_values: dict[str, Sequence[str]] = {
        b.property_dcid: b.value_dcids
        for b in state.bindings
        if b.property_dcid is not None and b.kind in ("value", "set")
    }

    variable_spans = for_shape(state.variable_spans_by_vocabulary, state.shape_draft)
    resolved = _unavailable_qualifiers(state)
    unavailable = {(record.start, record.end) for record in resolved}
    occupied_ranges = [(span.start, span.end) for span in variable_spans] + [
        (start, end) for start, end in unavailable if start is not None and end is not None
    ]
    occupied_ranges += _chosen_ranges(state)
    chosen = state.menu_choices
    combination = _combine_matches(state.variable, (*state.admitted_qualifiers, *chosen), index)
    for span in variable_spans:
        code_rows = [row for row in span.rows if isinstance(row, CodeRow)]
        if not code_rows or (span.start, span.end) in unavailable:
            continue
        if any(
            (choice.start or 0) <= span.start
            and span.end <= (choice.end or 0)
            and any(row.code_system == choice.code_system for row in code_rows)
            for choice in chosen
        ):
            continue
        named = [GraphRef(ref=row.dcid, label=row.label) for row in code_rows]
        operation = combination.operations.get((span.start, span.end))
        disposition, slot = _qualifier_disposition(
            [(row.code_system, row.dcid) for row in code_rows],
            bound_values,
            index,
            union=operation.selected if operation and operation.operation == "union" else (),
        )
        resolved.append(
            ResolvedQualifier(
                text=span.matched_text,
                named=named,
                slot=slot,
                disposition=disposition,
                operation=operation.operation if operation else None,
                selected=[_code_ref(dcid, index) for dcid in operation.selected]
                if operation
                else [],
            )
        )
    resolved.extend(_model_records(chosen, combination, bound_values, index))

    already_recorded = frozenset(_normalize(q.text) for q in resolved)
    for tie in state.admitted_ties:
        if _normalize(tie.matched_text) in already_recorded:
            continue
        if not any(row.code_system in declared for row in tie.rows):
            continue
        named = [GraphRef(ref=row.code_dcid, label=row.label) for row in tie.rows]
        disposition, slot = _qualifier_disposition(
            [(row.code_system, row.code_dcid) for row in tie.rows], bound_values, index
        )
        resolved.append(
            ResolvedQualifier(
                text=tie.matched_text, named=named, slot=slot, disposition=disposition
            )
        )

    # Sidecar offsets index each coded phrase. ``variable`` ranges stay separate.
    already_recorded = {_normalize(q.text) for q in resolved}
    for coded in state.qualifiers or ():
        for span in coded.spans:
            normalized_phrase = _normalize(span.matched_text)
            if normalized_phrase in already_recorded:
                continue
            code_rows = [
                row
                for row in span.rows
                if isinstance(row, CodeRow) and row.code_system not in foreign
            ]
            if not code_rows or not any(row.code_system in declared for row in code_rows):
                continue
            named = [GraphRef(ref=row.dcid, label=row.label) for row in code_rows]
            disposition, slot = _qualifier_disposition(
                [(row.code_system, row.dcid) for row in code_rows], bound_values, index
            )
            resolved.append(
                ResolvedQualifier(
                    text=span.matched_text, named=named, slot=slot, disposition=disposition
                )
            )
            already_recorded.add(normalized_phrase)

    slot_nouns = [
        index.declarations[prop].plain_name for prop in declared if prop in index.declarations
    ]
    for phrase in state.bind_result.unplaced:
        if _names_carried_per_head_rate(phrase, state.shape_draft, state.denominator_kinds):
            continue
        # A phrase as long as the noun that names it is the noun, singular or plural.
        if any(
            len(phrase.split()) == len(noun.split()) and names_plain_name(phrase, noun)
            for noun in slot_nouns
        ):
            continue
        if _names_answering_dataset(phrase, state, index):
            continue
        phrase_ranges = _all_normalized_text_ranges(phrase, state.variable)
        if not phrase_ranges:
            continue
        if any(
            _ranges_overlap(phrase_range, occ)
            for phrase_range in phrase_ranges
            for occ in occupied_ranges
        ):
            continue
        resolved.append(
            ResolvedQualifier(text=phrase, named=[], slot=None, disposition="no_vocabulary")
        )

    already_recorded = {_normalize(q.text) for q in resolved}
    no_vocabulary_terms = frozenset(
        term
        for prop in declared
        if prop in index.declarations
        for term in index.declarations[prop].no_vocabulary_terms
    )
    if no_vocabulary_terms:
        for text in (state.variable, state.query_text):
            if text is None:
                continue
            for phrase in match_unsupported_qualifier_phrases(text, terms=no_vocabulary_terms):
                normalized_phrase = _normalize(phrase)
                if normalized_phrase in already_recorded:
                    continue
                resolved.append(
                    ResolvedQualifier(text=phrase, named=[], slot=None, disposition="no_vocabulary")
                )
                already_recorded.add(normalized_phrase)

    resolved.extend(_unsupported_records(state, combination, resolved, gated=False))
    return resolved


def _model_records(
    chosen: Sequence[QualifierMatch],
    combination: _Combination,
    bound_values: Mapping[str, Sequence[str]],
    index: CodeIndex,
) -> list[ResolvedQualifier]:
    """One record per phrase the bind model's menu choices read (`_menu_choices`), with
    evidence ``"model"``, naming the codes chosen for it. Its disposition follows
    `_qualifier_disposition` against the final bindings, as a recognised term's does, and
    it records the operation that combined its codes with another term's."""
    by_phrase: dict[tuple[int, int], list[QualifierMatch]] = defaultdict(list)
    for choice in chosen:
        if choice.start is not None and choice.end is not None:
            by_phrase[(choice.start, choice.end)].append(choice)
    records: list[ResolvedQualifier] = []
    for (start, end), choices in by_phrase.items():
        operation = combination.operations.get((start, end))
        disposition, slot = _qualifier_disposition(
            [(choice.code_system, choice.code_dcid) for choice in choices],
            bound_values,
            index,
            union=operation.selected if operation and operation.operation == "union" else (),
        )
        records.append(
            ResolvedQualifier(
                text=choices[0].matched_text,
                named=[GraphRef(ref=choice.code_dcid, label=choice.label) for choice in choices],
                slot=slot,
                disposition=disposition,
                start=start,
                end=end,
                evidence="model",
                operation=operation.operation if operation else None,
                selected=[_code_ref(dcid, index) for dcid in operation.selected]
                if operation
                else [],
            )
        )
    return records


def operation_slot_phrases(
    operations: Sequence[Operation], shape_draft: ShapeDraft
) -> dict[str, str]:
    """Each slot property of `shape_draft` an operation over `category` reads, with the
    phrase of the first operation that names it. A category slot's name is its code
    system's declared `plain_name`, and an operation reads the slot its phrase names
    exactly one of (`slot_noun_named`). The reader asked for that slot's values as a
    breakdown, however the bindings hold it."""
    declarations = get_index().declarations
    plain_names = {declaration.plain_name for declaration in declarations.values()}
    phrases: dict[str, str] = {}
    for operation in operations:
        if operation.over != "category":
            continue
        noun = slot_noun_named(operation.phrase, plain_names)
        if noun is None:
            continue
        for slot in shape_draft.slot_keys:
            prop = slot.property_dcid
            if prop is None or prop in phrases or prop not in declarations:
                continue
            if declarations[prop].plain_name == noun:
                phrases[prop] = operation.phrase
    return phrases


def _open_operation_slots(
    bindings: list[SlotBindingDraft],
    operations: Sequence[Operation],
    shape_draft: ShapeDraft,
) -> list[SlotBindingDraft]:
    """Open each slot an operation over `category` names, as a valid ``requested_all``.

    A category slot's name is its code system's declared `plain_name`, the label the part
    shows ("flow", "sector"). An operation whose phrase names exactly one
    (`slot_noun_named`: "top 5 flow types") needs every value of that slot, so the slot
    binds ``unbound`` with ``requested_all`` evidence and the operation's phrase, which
    keeps a declared default from narrowing it, whether or not bind claimed it. A phrase
    naming two ("top 5 sectors by aid flows") opens neither. A slot bind gave a value
    stays as bound. Origin, destination, place and year need no rule here: no stage
    narrows them without a value the query names.
    """
    axis_of = {slot.property_dcid: slot.axis for slot in shape_draft.slot_keys}
    opened = {
        prop: SlotBindingDraft(
            axis=axis_of[prop],
            property_dcid=prop,
            kind="unbound",
            evidence="requested_all",
            requested_all_phrase=phrase,
        )
        for prop, phrase in operation_slot_phrases(operations, shape_draft).items()
    }
    return _with_opened_slots(bindings, opened)


def _with_opened_slots(
    bindings: list[SlotBindingDraft], opened: Mapping[str, SlotBindingDraft]
) -> list[SlotBindingDraft]:
    """`bindings` with each slot of `opened` they leave open replaced by its
    ``requested_all`` binding, and each they hold no binding for added. A slot they bind
    to a value stays."""
    if not opened:
        return bindings

    def is_open(binding: SlotBindingDraft) -> bool:
        return binding.kind in ("unbound", "absent") and not binding.value_dcids

    result = [
        opened[prop]
        if (prop := b.property_dcid) is not None and prop in opened and is_open(b)
        else b
        for b in bindings
    ]
    bound = {b.property_dcid for b in bindings}
    result.extend(binding for prop, binding in opened.items() if prop not in bound)
    return result


def _open_breakdown_slots(
    bindings: list[SlotBindingDraft],
    variable_text: str,
    dropped_phrases: Sequence[str],
    shape_draft: ShapeDraft,
    *,
    entities: Iterable[str],
) -> list[SlotBindingDraft]:
    """Open each slot of `shape_draft` the variable phrase or a dropped phrase names as a
    breakdown, as a valid ``requested_all``.

    A breakdown span is the words after "by", "across" or "for each", up to the next such
    connective, a number or one of `entities`, less function words and filler nouns
    (`qualifier_match.breakdown_spans`). A span names a slot when one of its words is a
    word of the slot's name (`bind._slot_name`) and each other word is a word of that
    name or of the label of a value in the slot's taxonomy (`_slot_vocabulary`). "By
    wealth quintile in Kenya" names the "Wealth quantile" slot through "wealth", with
    "quintile" from its value "Highest quintile". "Across Africa" names no slot, whatever
    a value's label holds, and "per capita" is no breakdown. A span naming two slots opens
    neither. The slot then binds ``unbound`` with ``requested_all`` evidence and the
    span's phrase, so `bind_partition_values`, which runs after, leaves it open. A slot
    the bindings give a value stays as bound (`_with_opened_slots`).
    """
    entities = tuple(entities)
    vocabularies = {
        slot.property_dcid: _slot_vocabulary(slot, shape_draft)
        for slot in shape_draft.slot_keys
        if slot.property_dcid is not None
    }
    axis_of = {slot.property_dcid: slot.axis for slot in shape_draft.slot_keys}
    opened: dict[str, SlotBindingDraft] = {}
    for text in (variable_text, *dropped_phrases):
        for span in breakdown_spans(text, stops=entities):
            named = [
                prop
                for prop, (name, labels) in vocabularies.items()
                if any(_has_word(word, name) for word in span.words)
                and all(_has_word(word, name | labels) for word in span.words)
            ]
            if len(named) == 1 and named[0] not in opened:
                opened[named[0]] = SlotBindingDraft(
                    axis=axis_of[named[0]],
                    property_dcid=named[0],
                    kind="unbound",
                    evidence="requested_all",
                    requested_all_phrase=span.phrase,
                )
    return _with_opened_slots(bindings, opened)


def _slot_vocabulary(
    slot: SlotKeyDraft, shape_draft: ShapeDraft
) -> tuple[frozenset[str], frozenset[str]]:
    """The casefolded words of `slot`'s name (`bind._slot_name`) and of its values' labels,
    less function words and filler nouns. A value's label is its code row's on a shape
    whose taxonomy the index sources, else its graph label (`ShapeDraft.value_labels`)
    without its property's name (`code_index.labels.without_property_suffix`)."""
    index = get_index()
    graph_labels = shape_draft.value_labels or {}
    values = (shape_draft.slot_taxonomy or {}).get(f"{slot.axis}:{slot.property_dcid}", [])
    labels: list[str] = []
    for value in values:
        if shape_draft.slot_taxonomy_index_sourced:
            if (row := index.code_by_dcid(value)) is not None:
                labels.append(row.label)
        elif (label := graph_labels.get(value)) is not None:
            labels.append(
                without_property_suffix(label, graph_labels.get(slot.property_dcid or ""))
            )

    def content_words(text: str) -> frozenset[str]:
        return frozenset(
            folded
            for token in word_tokens(text)
            if (folded := token.text.casefold()) not in FUNCTION_WORDS
            and folded not in BREAKDOWN_FILLER_NOUNS
        )

    return content_words(_slot_name(slot)), frozenset().union(*map(content_words, labels))


def _has_word(word: str, words: AbstractSet[str]) -> bool:
    """Whether `word` is one of `words`, singular or plural, in the forms
    `qualifier_match.names_plain_name` reads a name in (`plural_forms`)."""
    return any(word in plural_forms(name) for name in words)


def _bind_variable(state: ResolveState, shapes: Sequence[ShapeDraft]) -> str:
    """The variable phrase the bind model reads: the part's own, or the graph name of its
    series code's StatVars when the code is all the part's variable text writes
    (`ResolveState.source_code`), so the model reads the measure in place of an opaque code.
    The name is that of the code member declaring the fewest roles among `shapes`, the
    code's aggregate: the admissible shapes for the quantity read, which runs before the
    menu is seated, and the seated menu for the bind call. A part none of whose members has
    a name reads its own phrase."""
    code = state.source_code
    if code is None or not names_only_code(state.variable, code.code):
        return state.variable
    named = [
        (len(arcs.get("observationProperties", {}).get("nodes", [])), name)
        for shape in shapes
        for dcid, arcs in (shape.sv_arc_facts or {}).items()
        if dcid in code.dcids and (name := read_name(arcs)) is not None
    ]
    return min(named)[1] if named else state.variable


@dataclass(kw_only=True)
class _HousePolicyStep:
    """A part's house-policy step (`BindRequest.apply_policies`): the house policies over a
    binder's pool (`house_policy.apply_house_policies`), the pool's first index the binder's
    pick, read against the part's menu, frame, each shape's admitted qualifiers and
    availability, and the code index's house default datasets, with the binder's ratings of
    the pool shapes and its report of the pool shapes that hold what it recognised on its
    pick. `calls` counts the binder's calls. Each menu shape's facts are read once, and the
    context once per pool, however often the binder asks `chooses_on`."""

    shapes: Sequence[ShapeDraft]
    frame: Frame
    admitted_qualifiers: Sequence[Sequence[QualifierMatch]]
    availability: Sequence[bool | None]
    code_index: CodeIndex
    calls: int = 0
    _contexts: dict[frozenset[int], Context] = field(default_factory=dict, init=False, repr=False)

    def __call__(
        self,
        pool: PolicyPool,
        *,
        ratings: Mapping[int, float] | None = None,
        holds: Mapping[int, bool] | None = None,
    ) -> PolicyChoice:
        self.calls += 1
        return apply_house_policies(
            self._facts_of(pool.measuring),
            pool,
            self._context(pool.measuring),
            chosen=pool.measuring[0],
            ratings=ratings,
            holds=holds,
        )

    def chooses_on(
        self, indices: Sequence[int], axis: str, *, policies: Sequence[PolicyName] = POLICY_ORDER
    ) -> bool:
        """Whether one of `policies` would choose on measure axis `axis` among the menu
        shapes at `indices` (`house_policy.chooses_on`). Not a call of the step."""
        return chooses_on(self._facts_of(indices), self._context(indices), axis, policies=policies)

    @cached_property
    def _facts(self) -> tuple[ShapeFacts, ...]:
        """The facts of every menu shape, by menu index."""
        return shape_facts(
            self.shapes, self.availability, range(len(self.shapes)), code_index=self.code_index
        )

    def _facts_of(self, indices: Sequence[int]) -> tuple[ShapeFacts, ...]:
        return tuple(self._facts[i] for i in indices)

    def _context(self, pool: Sequence[int]) -> Context:
        """The context over `pool`, which reads the pool as a set."""
        key = frozenset(pool)
        if key not in self._contexts:
            self._contexts[key] = context_of(
                self.frame,
                self.admitted_qualifiers,
                pool,
                wanted_datasets=self.code_index.house_defaults(),
                case_systems=self.code_index.house_case_systems(),
            )
        return self._contexts[key]


def _elected_index(state: ResolveState) -> int:
    """The menu index of the elected reading: the first shape the bind menu states as the
    reading the model chose, in the rank that orders the menu (`bind.rank_elected`), and
    the model's own choice where the menu tells every higher-ranked shape apart from it.
    The rank reads only the shapes the house policies left standing in their pool
    (`BindResult.policy_choice`), so it breaks a tie among the survivors and leaves
    a policy's move in place. The model's bindings and nominations stand as drafted: the bindings
    read the elected shape unchanged, and a nominated shape that is elected here is no
    alternative to itself (`phase_place_materialise._plan_status`). A reading the model
    declined every menu shape for stands in as it is (`_stand_in_declined`)."""
    bind_result = state.bind_result
    if state.declined is not None:
        return bind_result.chosen_shape_index
    decision = state.bind_decision
    among = None
    if decision is not None and decision.policy_pool is not None:
        dropped = decision.policy_choice.dropped if decision.policy_choice is not None else ()
        among = frozenset(decision.policy_pool.measuring) - frozenset(dropped)
    return rank_elected(
        state.capped_shapes, bind_result, availability=state.menu_availability, among=among
    )


def _log_bind_decision(state: ResolveState, decision: BindResult | None) -> None:
    """One line per bind call, ``bind_decision`` and a JSON object: the part and pass, the
    binder, the menu's shape ids, the shape the binder chose (``chosen``), the one the house
    policies elect (``elected``) and the one the part answers with after the menu-rank
    tie-break (``answered``, `_elected_index`), each None when the reply elects none, the
    policy pool, the policies that narrowed, the reply's
    ratings as it wrote them (None on a refusal), the binder's own scores
    (`BindResult.scores`), the facts the policies read of each pool shape, and whether the
    schema refused the reply (`decision` None)."""
    pool = decision.policy_pool if decision is not None else None
    choice = decision.policy_choice if decision is not None else None
    output = decision.output if decision is not None else None
    record = {
        "part": state.variable_index,
        "pass": state.pool,
        "binder": decision.binder if decision is not None else state.binder.name,
        "menu": [shape.shape_id for shape in state.capped_shapes],
        "chosen": pool.measuring[0] if pool is not None else None,
        "elected": choice.survivor if choice is not None else None,
        "answered": _elected_index(state) if choice is not None else None,
        "pool": list(pool.measuring) if pool is not None else [],
        "fired": list(choice.fired) if choice is not None else [],
        "ratings": [r.model_dump() for r in output.measuring] if output is not None else None,
        "scores": dict(decision.scores) if decision is not None else {},
        "facts": [asdict(f) for f in choice.facts] if choice is not None else [],
        "refused": decision is None,
    }
    logger.info("bind_decision %s", json.dumps(record, separators=(",", ":")))


async def _phase_bind(state: ResolveState) -> RegionResult | None:
    """Run one choose-and-bind call through the part's binder (`ResolveState.binder`).
    Return a terminal result for an ask that names no nearest reading, or an invalid index.

    The call elects a shape of the admissible menu and binds its slots, and the house
    policies choose among the shapes the binder elects from (`BindRequest.apply_policies`,
    `_HousePolicyStep`), once on a reply that elects a shape and never on one that does
    not. The binder's answer is `ResolveState.bind_decision`, logged as one
    ``bind_decision`` line (`_log_bind_decision`). A binder that runs the step any other
    number of times, or raises `PolicyContractError`, breaks that contract. The breach is
    logged at ERROR in place of the line and the part refuses ``variable_not_resolved``
    with the call's usage. The part's frame
    opens the prompt, and the readings refused only on a pinned part follow the menu as not
    electable (`InterpretationSet.refused_on_pinned`). Each shape's own admitted qualifiers
    become its fixed context (`_shape_qualifiers`), less the terms inside a phrase its own
    reading leaves partly unread, which the shape offers a candidate menu for
    (`_candidate_menus`). Ties with one row become matches, while larger ties remain closed
    alternatives. Declared constraint-measure axes are withheld from index-sourced shapes
    for deterministic repair. Election vetting checks ``requested_all`` claims against the
    chosen shape.

    Each menu shape states whether it has data at the places the part names
    (`has_data_where_asked`), when it names any, and the words of the variable the readers
    that read no binding leave unread on it (`_nominated_words`), which the binder may
    rate once the bindings settle (`Binder.rate_words_stated`). The menu's held families,
    whose members' names those words read, are read first, off the event loop
    (`read_held_families`).

    An answer the schema refuses at parse (`LLMOutputRefused`: a binding that names no
    value) refuses ``variable_not_resolved`` with the call's usage and warns
    ``degenerate_binding_detected``. A value or set draft on a property the elected shape
    has no slot for leaves the output (`split_stray_bindings`), and its value is recorded
    in the ledger as a requirement (`_stray_requirements`). Every answer is held to a value
    a recorded term supports. The part answers from the readings that apply it, and refuses
    ``qualifier_not_bound`` on it, with the elected reading flagged, when none does. An
    answer that leaves a value only the model read unapplied stands and discloses it.

    A model that asks declines every admissible reading. When it names the nearest refused
    reading, that reading stands in as the elected shape (`_stand_in_declined`), so the
    later phases settle its bindings and places and the part refuses with it as the flagged
    match before materialise. A part whose menu is empty, every reading refused, and whose
    model names none refuses as it would have without the call (`_refusal_before_bind`).
    A call that elects no shape and names no refused reading on a part naming a source
    refuses ``qualifier_not_bound`` on the code terms no offered shape holds
    (`_unelected_qualifiers`), and otherwise as `_declined_terminal` refuses, flagging
    nothing.

    The quantity read ran before admissibility (`_read_stated_quantity`), so the election
    never carries it, and the call's usage adds to the read's.
    """
    await read_held_families(state.capped_shapes)
    per_shape = [_shape_qualifiers(state, shape) for shape in state.capped_shapes]
    candidate_menus = [_candidate_menus(state, shape) for shape in state.capped_shapes]
    admitted_qualifiers = [matches for matches, _ties, _unheld in per_shape]
    admitted_ties = [ties for _matches, ties, _unheld in per_shape]
    state.admitted_qualifiers = merge_lists(admitted_qualifiers)
    state.admitted_ties = merge_lists(admitted_ties)
    withheld_measure_axes = frozenset(
        axis.axis for axis in get_measure_index().axes if axis.kind == "constraint"
    )
    availability: list[bool | None] = []
    if names_a_place(state):
        series = series_at_places(state)
        availability = [
            has_data_where_asked(state, shape, series=series) for shape in state.capped_shapes
        ]
    state.menu_availability = availability
    policies = _HousePolicyStep(
        shapes=state.capped_shapes,
        frame=state.frame,
        admitted_qualifiers=admitted_qualifiers,
        availability=availability,
        code_index=get_index(),
    )
    request = BindRequest(
        variable=_bind_variable(state, state.capped_shapes),
        shapes=state.capped_shapes,
        admitted_qualifiers=admitted_qualifiers,
        admitted_ties=admitted_ties,
        withheld_measure_axes=withheld_measure_axes,
        dropped_qualifier_phrases=tuple(state.dropped_qualifier_phrases),
        candidate_menus=candidate_menus if any(candidate_menus) else (),
        nominated_words=[_nominated_words(state, shape) for shape in state.capped_shapes],
        frame=state.frame,
        refused=state.interpretations.refused_on_pinned,
        availability=availability,
        query_text=state.query_text or state.role_query,
        apply_policies=policies,
        policy_chooses_on=policies.chooses_on,
        part=state.variable_index,
        pool=state.pool,
    )
    t0 = now_ms()
    bind_result: _UniversalBindOutput
    _bind_usage: dict | None = None
    bind_task = asyncio.create_task(state.binder.bind(request, llm=state.llm))
    refusal: LLMOutputRefused | None = None
    breach: str | None = None
    try:
        if state.emit is not None and state.capped_shapes:
            # Emit the top-ranked candidate while the bind call runs. It may differ from the choice.
            shape_event = await _build_shape_event(
                state.capped_shapes[0], variable_index=state.variable_index, graph=state.graph
            )
            await state.emit(shape_event)
        try:
            decision, _bind_usage = await bind_task
            state.bind_decision = decision
            bind_result = decision.output
        except LLMOutputRefused as exc:
            # The schema refuses a binding that names no value, so a model that answers
            # with one has bound nothing it can name: the variable is not resolved. Gemini
            # leaves a schema-refused answer unparsed, which would otherwise read as an
            # infrastructure failure.
            logger.info("bind answer refused at parse for %r: %s", state.variable, exc)
            refusal = exc
            if policies.calls:
                breach = f"house policies ran {policies.calls} times on a refusal"
        except PolicyContractError as exc:
            breach = str(exc)
        else:
            elects = not bind_result.ask and 0 <= bind_result.chosen_shape_index < len(
                state.capped_shapes
            )
            if policies.calls != int(elects):
                breach = (
                    f"house policies ran {policies.calls} times on a bind call that "
                    f"{'elects' if elects else 'elects no'} shape"
                )
        if breach is None and refusal is None:
            bind_result, stray = split_stray_bindings(bind_result, state.capped_shapes)
            state.bind_result = bind_result
            if stray:
                state.stray_requirements = await _stray_requirements(state, stray)
    finally:
        # Cancel and await an unfinished bind task so cancellation cannot detach it.
        if not bind_task.done():
            bind_task.cancel()
            await asyncio.gather(bind_task, return_exceptions=True)
    state.timing["bind"] = now_ms() - t0
    # A call that broke the house-policy contract logs its breach in place of a decision.
    if breach is None:
        _log_bind_decision(state, None if refusal is not None else state.bind_decision)
    await _emit_step(
        state.emit,
        state.variable_index,
        "bind",
        ms=state.timing["bind"],
    )
    if breach is not None:
        # A binder that breaks the house-policy contract has an election nobody can
        # vouch for, so the part refuses loudly.
        logger.error(
            "part %r: the binder broke the house-policy contract: %s", state.variable, breach
        )
        usage = refusal.usage if refusal is not None else _bind_usage
        return state.no_data("variable_not_resolved", llm_usage=_sum_usage(state.var_usage, usage))
    if refusal is not None:
        state.warnings.append(
            Warning(
                code=DEGENERATE_BINDING_DETECTED,
                severity="warn",
                message=(
                    "the binding the search drafted named no value, so the question was "
                    "not resolved."
                ),
            )
        )
        return state.no_data(
            "variable_not_resolved", llm_usage=_sum_usage(state.var_usage, refusal.usage)
        )
    _var_usage = state.var_usage = _sum_usage(state.var_usage, _bind_usage)

    # An ask that names a refused reading stands that reading in as the elected shape. Any
    # other ask, or an out-of-range shape index, ends the part here: on code terms no shape
    # can hold, else as `_declined_terminal` refuses. Entity errors were checked before
    # shape derivation.
    nearest = (
        state.interpretations.nearest_refused(bind_result.nearest_refused_index)
        if bind_result.ask
        else None
    )
    if nearest is not None:
        _stand_in_declined(state, nearest)
    elif not state.capped_shapes:
        return _refusal_before_bind(state, llm_usage=_var_usage)
    declined = nearest is None and (
        bool(bind_result.ask) or not 0 <= bind_result.chosen_shape_index < len(state.capped_shapes)
    )
    if declined:
        if unelected := _unelected_qualifiers(state):
            return state.no_data(
                "qualifier_not_bound", unbound_qualifiers=unelected, llm_usage=_var_usage
            )
        return _declined_terminal(state, llm_usage=_var_usage)
    # The bind elected a shape, so each value it drafted on a slot that shape lacks enters
    # the ledger here, before any refusal of settlement can judge or name it.
    for requirement in state.stray_requirements:
        state.requirements = with_requirement(state.requirements, requirement)
    return None


async def _stray_requirements(
    state: ResolveState, stray: Sequence[SlotBindingDraft]
) -> tuple[Requirement, ...]:
    """One requirement per property `stray` binds values on: the values the bind drafted on
    properties the elected shape has no slot for (`split_stray_bindings`). A value no
    recorded term supports (`model_only_values`) is a requirement of its own that does not
    refuse (`Requirement.refuses`). An answer that leaves it unapplied stands and departs
    `measure_not_applied` under the value's name. The other values on one property are a
    set the request names (`ValueRule`), applied by membership: an
    answer that binds one of them, or every member of which holds one, applies it, and the
    elected answer, which drops the draft, leaves it unmet unless every member holds one.
    Its term is ``unbound`` with evidence ``"model"``, named by the values' labels joined
    by "and", each a menu shape's (`ShapeDraft.value_labels`), else the code
    index's, else the graph's (`ResolveState.label_dcids`). When the graph's read fails,
    the values it was to label are named by their dcid's tail in words (`_humanize_ref`)
    and the lost read is noted (`note_degraded_read`).

    A set of two or more values adds one requirement per value (origin ``set_value``),
    named by its label: an answer that meets the set but whose selected members hold not
    one value discloses that value, and neither refuses nor drops the answer for it
    (`requirements.judge`)."""
    values = list(
        dict.fromkeys(
            (draft.property_dcid, value)
            for draft in stray
            if draft.property_dcid is not None
            for value in draft.value_dcids
        )
    )
    index = get_index()
    labels: dict[str, str] = {}
    for shape in state.capped_shapes:
        labels.update((shape.value_labels or {}).items())
    for _prop, value in values:
        if value not in labels and (row := index.code_by_dcid(value)) is not None:
            labels[value] = row.label
    unlabelled = list(dict.fromkeys(value for _prop, value in values if value not in labels))
    if unlabelled:
        try:
            read = await asyncio.to_thread(state.label_dcids, unlabelled)
        except GraphInfraError as exc:
            logger.warning("stray value labels read failed for %s: %s", unlabelled, exc)
            note_degraded_read("stray value labels")
            read = [_humanize_ref(value, None) for value in unlabelled]
        labels.update(zip(unlabelled, read, strict=True))
    by_prop: dict[str, list[str]] = {}
    for prop, value in values:
        by_prop.setdefault(prop, []).append(value)
    requirements: list[Requirement] = []
    for prop, values_on in by_prop.items():
        model = model_only_values(state, prop, values_on, labels)
        requirements.extend(
            Requirement(
                term=ResolvedQualifier(
                    text=labels[value],
                    named=[GraphRef(ref=value, label=labels[value])],
                    slot=None,
                    disposition="unbound",
                    evidence="model",
                ),
                origin="stray",
                rule=ValueRule(values=frozenset({(prop, value)})),
                departure=unapplied_departure(
                    labels[value], asked=labels[value], property_dcid=prop
                ),
                refuses=False,
            )
            for value in model
        )
        drafted = [value for value in values_on if value not in model]
        if not drafted:
            continue
        named = [GraphRef(ref=value, label=labels[value]) for value in drafted]
        label = " and ".join(ref.label for ref in named)
        requirements.append(
            Requirement(
                term=ResolvedQualifier(
                    text=label,
                    named=named,
                    slot=None,
                    disposition="unbound",
                    evidence="model",
                ),
                origin="stray",
                rule=ValueRule(values=frozenset((prop, value) for value in drafted)),
                departure=unapplied_departure(
                    label, asked=label, property_dcid=qualifier_property(named)
                ),
            )
        )
        if len(named) > 1:
            # Each value of a set, which an answer meeting the set by membership may leave
            # out and then discloses (`requirements.judge`).
            requirements.extend(
                Requirement(
                    term=ResolvedQualifier(
                        text=ref.label,
                        named=[ref],
                        slot=None,
                        disposition="unbound",
                        evidence="model",
                    ),
                    origin="set_value",
                    rule=ValueRule(values=frozenset({(prop, ref.ref)})),
                    departure=unapplied_departure(
                        ref.label, asked=ref.label, property_dcid=qualifier_property([ref])
                    ),
                )
                for ref in named
            )
    return tuple(requirements)


def _stand_in_declined(state: ResolveState, declined: Interpretation) -> None:
    """Make `declined`, the refused reading the model named as nearest when it asked, the
    elected shape: appended to the menu with its slot menu, chosen with no drafted bindings
    and no nominations. `ResolveState.declined` marks the part to refuse with it.

    Code settles it as it settles an elected reading, so the flagged match names the
    reading the question asked for and not its whole family: each slot takes the one value
    whose name the variable phrase states (`_named_value_bindings`), the codes the question
    names in its vocabulary join the part's matches (`ResolveState.pinned_part_matches`),
    which vetting forces where it can bind them, and repair applies the declared defaults
    and the values its family shares."""
    state.declined = declined
    vocabulary = shape_vocabulary(declined.shape)
    if vocabulary not in state.query_qualifier_matches:
        state.query_qualifier_matches = {
            **state.query_qualifier_matches,
            vocabulary: state.pinned_part_matches.get(vocabulary, ()),
        }
    state.capped_shapes.append(_shape_with_taxonomy(declined.shape))
    state.bind_result = state.bind_result.model_copy(
        update={
            "chosen_shape_index": len(state.capped_shapes) - 1,
            "bindings": _named_value_bindings(declined.shape, state.variable),
            "also_plausible": None,
        }
    )


def _named_value_bindings(shape: ShapeDraft, variable: str) -> list[SlotBindingDraft]:
    """A binding for each slot of `shape` one of whose members' values the variable phrase
    names by its label, as a whole word outside a negation or exclusion, when exactly one
    value of the slot is named. The members are the shape's population
    (`shape_population_arcs`), its family's when it is held. Empty when the named values
    match no member together, since a binding no member holds names no reading."""
    labels = shape.value_labels or {}
    population = shape_population_arcs(shape)
    bindings: list[SlotBindingDraft] = []
    for slot in shape.slot_keys:
        prop = slot.property_dcid
        if prop is None:
            continue
        values = dict.fromkeys(
            value for member in population.values() if (value := read_constraints(member).get(prop))
        )
        named = [
            value
            for value in values
            if (label := labels.get(value))
            and _label_named_in_text(label, variable)
            and not _label_negated_in_text(label, variable)
        ]
        if len(named) == 1:
            bindings.append(
                SlotBindingDraft(
                    axis=slot.axis, property_dcid=prop, kind="value", value_dcids=named
                )
            )
    if bindings and not bound_realizable_members(
        list(population), population, bound_values_from_bindings(bindings)
    ):
        return []
    return bindings
