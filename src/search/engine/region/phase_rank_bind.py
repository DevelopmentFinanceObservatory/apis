"""Rank shapes, admit query qualifiers, and run the universal choose-and-bind call."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import CodeRow, MeasureAxis
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput, bind_universal
from search.engine.bind_repair import (
    _label_named_in_text,
    _label_negated_in_text,
    _unscoped_property_values,
)
from search.engine.config import settings
from search.engine.envelope import _emit_step, now_ms
from search.engine.qualifier_match import (
    _WORD_RE,
    CodedQualifier,
    QualifierMatch,
    QualifierSpan,
    QualifierTie,
    TextReading,
    _all_normalized_text_ranges,
    _match_source,
    _normalize,
    _ranges_overlap,
    _row_system,
    match_unsupported_qualifier_phrases,
    names_plain_name,
    read_by_vocabulary,
    slot_noun_named,
)
from search.engine.quantity import (
    PER_HEAD_RATE_BASE_RE,
    PER_PERSON_FORMS,
    STRONG_RATE_REQUEST_RE,
    DenominatorKind,
    denominator_kinds,
    normalisation_class,
)
from search.engine.region.alternate_guard import (
    is_cross_population_false_friend,
    leaves_price_basis_unread,
    measure_class,
    shape_measure_class,
)
from search.engine.region.availability import (
    _prune_availability_siblings,
    _resolve_sibling_availability,
    _sibling_probe_places,
)
from search.engine.region.codes import AVAILABILITY_PROBE_FAILED
from search.engine.region.grounding import _shape_with_taxonomy
from search.engine.region.phase_intake import _unresolved_entity_terminal
from search.engine.region.place_roles import (
    bind_place_roles,
    lacks_query_axis,
    subject_shapes_refused,
    without_open_axis_gaps,
    without_unasked_axes,
)
from search.engine.region.qualifier_readings import match_query_readings, reading_for_shape
from search.engine.region.rate_head import (
    _labels_share_rate_head,
    _rate_head,
    _rate_head_compatible,
)
from search.engine.region.result import RegionResult
from search.engine.region.sentence_events import _build_shape_event
from search.engine.region.state import ResolveState, for_shape, merge_lists, shape_vocabulary
from search.engine.retrieval.bound_values import (
    DefaultReading,
    collapse_measure_families,
    elect_five_tuple_family,
    index_carries_shape,
    other_families_properties,
    part_datasets,
    part_member_rows,
    price_basis_order,
    query_phrases_read,
    shape_datasets,
    shape_for_part,
)
from search.engine.retrieval.representative import _shape_disagreement_props
from search.engine.retrieval.role_scope import _COVERAGE_ROLE_FOR_AXIS
from search.engine.shape import (
    ShapeDraft,
    is_role_bearing,
    measures_count,
)
from search.models import (
    GraphRef,
    Operation,
    ResolvedQualifier,
    TraceDefaultReading,
    Warning,
)

logger = logging.getLogger(__name__)

_COUNT_REQUEST_RE = re.compile(r"\b(?:counts?|numbers?)\b|\bhow\s+many\b", re.IGNORECASE)

# Flow/aid terms required before a role-bearing anchor can prune cross-population shapes.
_FLOW_FRAMING_WORDS = (
    "oda",
    "aid",
    "assistance",
    "commitments?",
    "disbursements?",
    "loans?",
    "grants?",
)
_FLOW_FRAMING_RE = re.compile(r"\b(?:" + "|".join(_FLOW_FRAMING_WORDS) + r")\b", re.IGNORECASE)


def _has_strong_rate_intent(variable: str) -> bool:
    """Whether an explicit rate term rules out an unnormalised raw count. The term is `rate`,
    `rates` or a share term (`quantity.STRONG_RATE_REQUEST_RE`), and no count term
    may appear."""
    return (
        STRONG_RATE_REQUEST_RE.search(variable) is not None
        and _COUNT_REQUEST_RE.search(variable) is None
    )


def _names_flow_framing(variable: str) -> bool:
    """Whether the query names aid or flow framing.

    A query naming none describes an indicator, so a role-bearing anchor recalled through a
    shared noun may not prune a cross-population shape.
    """
    return _FLOW_FRAMING_RE.search(variable) is not None


def _is_unnormalised_raw_count(shape: ShapeDraft) -> bool:
    """Whether a standard shape is the literal count reading a rate excludes (`measures_count`)."""
    return (
        not is_role_bearing(shape)
        and measures_count(shape)
        and shape.stat_type_dcid == "measuredValue"
        and shape.meas_denom_dcid is None
        and not shape.meas_denom_unresolved
    )


def _label_has_negated_occurrence(label: str, text: str) -> bool:
    """Whether any whole-word occurrence is negated, including mixed wording."""
    pattern = r"\b" + re.escape(label) + r"\b"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        prefix = text[: match.start()]
        prefix_words = list(re.finditer(r"[\w']+", prefix))
        context_start = prefix_words[-3].start() if len(prefix_words) >= 3 else 0
        context = prefix[context_start:]
        context = re.sub(pattern, "", context, flags=re.IGNORECASE)
        context = f"{context}{text[match.start() : match.end()]}"
        if _label_negated_in_text(label, context):
            return True
    return False


def _mentioned_raw_qualifier_props(shape: ShapeDraft, variable: str) -> tuple[set[str], bool]:
    """Return named varying properties. Any negated value label makes the evidence ambiguous."""
    mentioned: set[str] = set()
    value_labels = shape.value_labels or {}
    for property_dcid in _shape_disagreement_props(shape):
        for value_dcid in _unscoped_property_values(shape, property_dcid):
            label = value_labels.get(value_dcid)
            if label is None or not _label_named_in_text(label, variable):
                continue
            if _label_has_negated_occurrence(label, variable):
                return set(), True
            mentioned.add(property_dcid)
    return mentioned, False


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


def _same_head_normalised_shapes(shapes: list[ShapeDraft], head: str) -> list[ShapeDraft]:
    """Return the non-raw shapes whose labels share a rate-request head."""
    return [
        shape
        for shape in shapes
        if not _is_unnormalised_raw_count(shape) and _rate_head_compatible(shape, head)
    ]


def _raw_count_is_pending_joint_slice(
    raw_shape: ShapeDraft, shapes: list[ShapeDraft], variable: str
) -> bool:
    """From the pre-bind menu, check whether each same-head shape drops two named qualifiers."""
    if not _has_strong_rate_intent(variable) or not _is_unnormalised_raw_count(raw_shape):
        return False
    head = _rate_head(variable)
    if head is None or not _labels_share_rate_head(head, raw_shape.label):
        return False
    mentioned_props, contradictory = _mentioned_raw_qualifier_props(raw_shape, variable)
    if contradictory:
        return False
    normalized_shapes = _same_head_normalised_shapes(shapes, head)
    return bool(normalized_shapes) and all(
        len(mentioned_props - shape_vocabulary(candidate)) >= 2 for candidate in normalized_shapes
    )


def _prune_explicit_rate_raw_counts(shapes: list[ShapeDraft], variable: str) -> list[ShapeDraft]:
    """Prune covered raw counts before ranking and top-K selection.

    Keep an anchored family when no normalized sibling exists, wording conflicts, or every
    normalized shape loses at least two named qualifiers. Assess each candidate separately.
    """
    raw_counts = [shape for shape in shapes if _is_unnormalised_raw_count(shape)]
    if not raw_counts:
        return shapes
    non_raw_shapes = [shape for shape in shapes if not _is_unnormalised_raw_count(shape)]
    head = _rate_head(variable)
    retained_raw_ids: set[int] = set()
    for raw_shape in raw_counts:
        # Retain the exception only for a raw family whose label matches the rate head.
        if head is None or not _labels_share_rate_head(head, raw_shape.label):
            continue
        normalized_shapes = _same_head_normalised_shapes(non_raw_shapes, head)
        mentioned_props, contradictory = _mentioned_raw_qualifier_props(raw_shape, variable)
        # A bare count has no family qualifiers to preserve. Keeping it without a normalized
        # sibling would answer an explicit-rate request with a denominator-less count.
        if not normalized_shapes and not _shape_disagreement_props(raw_shape):
            continue
        if contradictory or not normalized_shapes:
            retained_raw_ids.add(id(raw_shape))
            continue
        if all(
            len(mentioned_props - shape_vocabulary(candidate)) >= 2
            for candidate in normalized_shapes
        ):
            retained_raw_ids.add(id(raw_shape))
    return [
        shape
        for shape in shapes
        if not _is_unnormalised_raw_count(shape) or id(shape) in retained_raw_ids
    ]


def _rank_key(shape: ShapeDraft) -> tuple[bool, float, int, str]:
    """Order by recall or topic-root rank, breadth, then ``shape_id``.

    Unscored shapes sort last. Breadth is the confirmed recall sample, capped at
    ``SEARCH_MAX_CONFIRM_CANDIDATES``. It breaks flat-score ties, broadest first. The same
    key selects the cross-population anchor.
    """
    return (
        shape.rank_score is None,
        -(shape.rank_score or 0.0),
        -(len(shape.sv_arc_facts) if shape.sv_arc_facts else 1),
        shape.shape_id,
    )


async def _read_denominator_kinds(state: ResolveState, shapes: list[ShapeDraft]) -> None:
    """Read the kinds of the shapes' StatVar denominators into ``state.denominator_kinds``,
    in one batch, when the leg names a per-person form (`quantity.PER_PERSON_FORMS`).

    Materialise and the alternate guard read the map: a series divided by
    `Count_Person_15OrMoreYears` is per person, one divided by GNI is not. Nothing is read
    for any other request. The menu keeps every shape: an off-topic per-capita series must
    not displace the shape the query names ("teacher count per capita" keeps the teacher
    count, which refuses ``denominator_not_available`` at materialise).
    """
    if not state.quantity_requirement.named & PER_PERSON_FORMS:
        return
    state.denominator_kinds = await asyncio.to_thread(
        denominator_kinds,
        [s.meas_denom_dcid for s in shapes if s.meas_denom_dcid is not None],
        graph=state.graph,
    )


async def _phase_prune_and_rank(state: ResolveState) -> RegionResult | None:
    """Build the ranked, capped candidate set. Return a terminal result if it is empty.

    Applies dataset candidacy (`_dataset_candidates`), the score floor, the measure-class
    check (`shape_measure_class`), the two-ended prune, the denominator-kind read
    (`_read_denominator_kinds`), availability and measure pruning,
    then reserves recalled shapes within the top-K menu. A part whose named dataset
    (`ResolveState.named_dataset`) publishes none of the recalled shapes refuses
    ``qualifier_not_bound``, naming the dataset, and so does one whose variable names a
    code no row of that dataset's candidate shapes holds, naming the term
    (`_undeclared_qualifiers`). Both refusals come before any role check. Falling through
    leaves ``_phase_bind`` to call the model.
    """
    candidates = _dataset_candidates(
        state.shapes_found, state.named_dataset, state.excluded_datasets
    )
    if not candidates and state.named_dataset is not None:
        await _record_shape_step(state)
        return state.no_data(
            "qualifier_not_bound",
            unbound_qualifiers=(_unbound_dataset(state.named_dataset),),
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
    if state.named_dataset is not None:
        undeclared = _undeclared_qualifiers(
            state.variable,
            state.variable_spans_in_every_system,
            candidates,
            state.variable_spans_by_vocabulary,
        )
        if undeclared:
            await _record_shape_step(state)
            return state.no_data("qualifier_not_bound", unbound_qualifiers=undeclared)
    # Retrieval score controls admission and order. The model chooses among the top-K.
    #
    # The floor filters candidates on their measured recall score, a topic expansion on its
    # root's. A role-bearing shape the index carries is exempt: its recall score describes
    # the flow's direction, and the index vouches for its members and offers the family's
    # full option sets to bind from. A role-bearing shape the index does not carry, a debt
    # series, holds only its recalled members, so a weak score is the whole of its claim to
    # the question and the floor applies. A shape with no score at all was not retrieved
    # against the query.
    eligible_shapes = [
        s
        for s in candidates
        if (is_role_bearing(s) and index_carries_shape(s))
        or s.measured_score is None
        or s.measured_score >= settings().unknown_score_threshold
    ]
    state.n_below_score_floor = len(candidates) - len(eligible_shapes)
    # A shape whose population type's name names another measure class than the query
    # (prevalence, incidence, mortality) measures something else, however well it scores. A
    # shape naming no class stays.
    named_class = measure_class(state.variable)
    if named_class is not None:
        kept = [s for s in eligible_shapes if shape_measure_class(s) in (None, named_class)]
        state.n_measure_class_refused = len(eligible_shapes) - len(kept)
        eligible_shapes = kept
    # When the query's places name both flow ends, `bind_place_roles` refuses every
    # subject-only shape `role_incompatible` (`subject_shapes_refused`). Once a role-bearing
    # shape survives the floor, they leave the election. With none surviving, bind elects a
    # subject-only shape and its refusal flags the identity it elected.
    if subject_shapes_refused(state) and any(is_role_bearing(s) for s in eligible_shapes):
        eligible_shapes = [s for s in eligible_shapes if is_role_bearing(s)]
    # A role-bearing shape that declares no role on an axis the query needs, a directed
    # place's or an operation's (`lacks_query_axis`), is no candidate once another shape
    # survives: "aid to Kenya" leaves a provider-only shape for one with a recipient. With
    # none surviving, bind elects it and its refusal flags the identity it elected.
    if any(not lacks_query_axis(state, s) for s in eligible_shapes):
        eligible_shapes = [s for s in eligible_shapes if not lacks_query_axis(state, s)]
    # An axis the query asks every member of is a preference among those shapes: a shape
    # the index carries that declares no role on it leaves once a role-bearing shape that
    # does survives (`without_open_axis_gaps`).
    eligible_shapes = without_open_axis_gaps(state, eligible_shapes)
    await _read_denominator_kinds(state, eligible_shapes)
    if _has_strong_rate_intent(state.variable):
        eligible_shapes = _prune_explicit_rate_raw_counts(eligible_shapes, state.variable)
    state.eligible_shapes = eligible_shapes
    # The availability prune, query-aware measure collapse and bind admission share the
    # eligible shapes' matches and ties.
    query_qualifiers = _match_query_qualifiers(
        sources,
        qualifiers=state.qualifiers,
        vocabularies=(shape_vocabulary(s) for s in eligible_shapes),
    )
    state.query_qualifier_matches = query_qualifiers.matches
    state.query_qualifier_ties = query_qualifiers.ties
    # Spans naming a code and a price basis ("PPP") are read per shape, so the prune, the
    # guard and bind admission each keep the reading their shape carries.
    query_qualifier_readings, reading_owned_phrases = match_query_readings(
        state.variable, state.dropped_qualifier_phrases, query_text=state.query_text
    )
    state.query_qualifier_readings = query_qualifier_readings
    # Measure values are read in every system together, so the measure matches come from
    # the union of every vocabulary's matches.
    query_qualifier_matches = state.all_qualifier_matches()
    # No-place and region-expanded queries need a supplemental availability probe. It uses
    # named places plus a bounded region sample, or representative places when none are
    # named. Probe failure leaves availability unknown and adds a warning.
    #
    # Measure-head pruning needs a named place. Denominator-resolution pruning applies to
    # named and representative-place probes.
    expanding_set = any(place_set.expands for place_set in state.place_sets.values())
    named_place = bool(state.recall_only_place_dcids) or expanding_set
    if not state.entities or expanding_set:
        eligible_shapes, _availability_probe_failed = await asyncio.to_thread(
            _resolve_sibling_availability,
            eligible_shapes,
            graph=state.graph,
            probe_places=_sibling_probe_places(state.recall_only_place_dcids, state.place_sets),
            named_place=named_place,
        )
        state.eligible_shapes = eligible_shapes
        if _availability_probe_failed:
            state.warnings.append(
                Warning(
                    code=AVAILABILITY_PROBE_FAILED,
                    severity="warn",
                    message=(
                        "the sibling-availability probe failed; denominator-"
                        "resolution siblings were left unprunable rather than "
                        "pruned on a stale or partial availability verdict."
                    ),
                )
            )
    eligible_shapes, state.measure_head_substitutions = _prune_availability_siblings(
        eligible_shapes,
        named_place=named_place,
        variable=state.variable,
        quantity=state.quantity_requirement,
        denominator_kinds=state.denominator_kinds,
        qualifier_matches=state.query_qualifier_matches,
        qualifier_readings=query_qualifier_readings,
    )
    state.eligible_shapes = eligible_shapes
    if not eligible_shapes:
        await _record_shape_step(state)
        return _unresolved_entity_terminal(state) or state.no_data("variable_not_resolved")

    # Families of one five-tuple and role class are rival readings, and the one reading the
    # most of the query reaches the election; of families reading alike, one with no series
    # at the named places leaves for one with a series there. Then a role-bearing shape
    # publishing on an axis the query does not ask for leaves for one reading as much
    # without it (`without_unasked_axes`).
    phrases_read = _phrases_reader(state)
    eligible_shapes = elect_five_tuple_family(
        eligible_shapes, phrases_read, series_at_places=_series_reader(state)
    )
    eligible_shapes = state.eligible_shapes = without_unasked_axes(
        state, eligible_shapes, phrases_read
    )
    # Dropped-qualifier warnings cover matches, ties and two-reading spans alike.
    state.variable_matched_qualifier_phrases = query_qualifiers.phrases | reading_owned_phrases
    measure_axes = get_measure_index().axes
    measure_query_matches, measure_qualifier_conflict_phrases = _measure_query_matches(
        query_qualifier_matches, measure_axes
    )
    state.measure_query_matches = measure_query_matches
    state.measure_qualifier_conflict_phrases = measure_qualifier_conflict_phrases
    # Only free-axis matches steer family election. Basis matches remain reportable only.
    role_free_axes = {axis.axis for axis in measure_axes if axis.role == "free"}
    role_free_matches = tuple(
        match for match in measure_query_matches if match.code_system in role_free_axes
    )

    # A scored role-bearing anchor can wrongly attract an unscored topic expansion through
    # a shared noun. Prune only when the query itself names aid or flow framing. Subject-only
    # expansions remain eligible. Use the top-ranked scored shape as anchor and keep the set
    # when pruning would remove every shape.
    scored_eligible = [s for s in eligible_shapes if s.representative_score is not None]
    if scored_eligible and _names_flow_framing(state.variable):
        anchor = min(scored_eligible, key=_rank_key)
        admitted = [
            s
            for s in eligible_shapes
            if s.representative_score is not None or not is_cross_population_false_friend(anchor, s)
        ]
        eligible_shapes = state.eligible_shapes = admitted or eligible_shapes

    # Preserve ranked order for capped shapes and the later cascade fan-out.
    ranked_shapes = sorted(eligible_shapes, key=_rank_key)
    # Collapse measure-only variants before top-K. Newly freed slots follow their own rank.
    # Shapes the index does not carry take the declared price-basis order, the query's named
    # readings first, unless the leg left a price-basis phrase unread.
    state.unread_price_basis = leaves_price_basis_unread(
        state.dropped_qualifier_phrases,
        operation_phrases=[operation.phrase for operation in state.operations],
    )
    collapsed = collapse_measure_families(
        ranked_shapes,
        get_index(),
        measure_axes,
        named=role_free_matches,
        reading=None if state.unread_price_basis else _default_reading(state, role_free_matches),
    )
    state.default_reading = _default_reading_trace(
        ranked_shapes, collapsed, applied=not state.unread_price_basis
    )
    ranked_shapes = collapsed
    state.ranked_shapes = list(ranked_shapes)
    # Reserve up to ``min_recall_bind_slots`` directly recalled shapes, then fill to
    # ``bind_top_k``. Selecting indices preserves rank order for near-neighbour checks and
    # cascade widening. The reserve adds no score tie-break.
    reserve = min(settings().min_recall_bind_slots, settings().bind_top_k)
    reserved_indices: set[int] = set()
    for i, s in enumerate(ranked_shapes):
        if len(reserved_indices) >= reserve:
            break
        if s.recalled:
            reserved_indices.add(i)
    selected_indices = set(reserved_indices)
    for i in range(len(ranked_shapes)):
        if len(selected_indices) >= settings().bind_top_k:
            break
        selected_indices.add(i)
    capped_shapes: list[ShapeDraft] = [
        _shape_with_taxonomy(s) for i, s in enumerate(ranked_shapes) if i in selected_indices
    ]
    state.capped_shapes = capped_shapes
    await _record_shape_step(state)
    return None


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
    shapes: Sequence[ShapeDraft], source: str | None, excluded: frozenset[str]
) -> list[ShapeDraft]:
    """The shapes a part may answer from, by the datasets that publish each
    (`shape_datasets`), each stamped with the datasets the part may answer it from and the
    dataset it names (`shape_for_part`).

    A part that names the dataset `source` keeps the shapes of the families that dataset
    publishes, so it never answers from another. A part that names none keeps a shape the
    index does not carry and a shape with an eligible dataset less `excluded` among the
    datasets publishing it (`part_datasets`, `code_index.members.eligible_among`). On a
    family a dataset that is not opt-in publishes, the opt-in publishers are not eligible,
    since they answer only a part that names them (`DatasetDeclaration.opt_in`). A family
    only opt-in datasets publish is kept, and when one of them is eligible the shape names
    it.
    """
    index = get_index()
    if source is not None:
        kept = [shape for shape in shapes if source in shape_datasets(shape, index)]
    else:
        kept = [
            shape
            for shape in shapes
            if not shape_datasets(shape, index)
            or part_datasets(shape, None, index, excluded=excluded)
        ]
    return [shape_for_part(shape, source, index, excluded=excluded) for shape in kept]


def _unbound_dataset(source: str) -> ResolvedQualifier:
    """The record of a named dataset that publishes no shape the part recalled, under its
    declared label."""
    label = get_index().datasets[source].label
    return ResolvedQualifier(
        text=label, named=[GraphRef(dcid=source, label=label)], slot=None, disposition="unbound"
    )


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
                    named=[GraphRef(dcid=dcid, label=label)],
                    slot=None,
                    disposition="unbound",
                )
            )
    return tuple(records)


def _unread_words(
    variable: str, span: QualifierSpan, read: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The character ranges of `span`'s words in `variable` that no range in `read` covers,
    in query order: the words of a term a shape's own reading leaves unread."""
    return [
        (span.start + word.start(), span.start + word.end())
        for word in _WORD_RE.finditer(variable[span.start : span.end])
        if not any(
            start <= span.start + word.start() and span.start + word.end() <= end
            for start, end in read
        )
    ]


def _unread_qualifier_phrases(state: ResolveState) -> tuple[str, ...]:
    """The words of the variable's code terms, read in every code system together, that the
    elected shape's own reading leaves unread where it reads others, each run of adjacent
    words as one phrase, in query order.

    "multilateral ODA" is DAC1 aid type 2000 in every system together; a CRS shape reads
    "ODA" alone, so "multilateral" is unread there. Every span of the shape's own reading
    counts as read, a measure value's included. A term the shape reads none of names no
    code the answer applies in part: "gross national income" is a CRS type of finance and
    the name of the series a GNI shape measures. A phrase overlapping a recorded
    qualifier, such as a phrase the bind model listed as unplaced, is left to that record.
    """
    variable = state.variable
    read = [
        (span.start, span.end)
        for span in for_shape(state.variable_spans_by_vocabulary, state.shape_draft)
    ]
    recorded = [
        found
        for qualifier in state.query_qualifiers
        for found in _all_normalized_text_ranges(qualifier.text, variable.casefold())
    ]
    phrases: list[str] = []
    for span in state.variable_spans_in_every_system:
        if not all(isinstance(row, CodeRow) for row in span.rows):
            continue
        unread = _unread_words(variable, span, read)
        if len(unread) == len(_unread_words(variable, span, ())):
            continue
        runs: list[tuple[int, int]] = []
        for start, end in unread:
            if runs and not variable[runs[-1][1] : start].strip():
                runs[-1] = (runs[-1][0], end)
            else:
                runs.append((start, end))
        phrases.extend(
            variable[start:end]
            for start, end in runs
            if not any(_ranges_overlap((start, end), found) for found in recorded)
        )
    return tuple(phrases)


def _undeclared_qualifiers(
    variable: str,
    spans: Sequence[QualifierSpan],
    shapes: Sequence[ShapeDraft],
    own_spans: Mapping[frozenset[str], tuple[QualifierSpan, ...]],
) -> tuple[ResolvedQualifier, ...]:
    """The variable's code terms, `spans` read in every code system, that no shape in
    `shapes` can hold, each as an unbound record.

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

    covered = [read_on(shape, codes) for shape, codes in readings]

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
        undeclared.append(
            ResolvedQualifier(
                text=span.matched_text,
                named=[GraphRef(dcid=row.dcid, label=row.label) for row in code_rows],
                slot=None,
                disposition="unbound",
            )
        )
    return tuple(undeclared)


def _unheld_bound_codes(
    bindings: Sequence[SlotBindingDraft],
    shape: ShapeDraft,
    *,
    excluded: frozenset[str] = frozenset(),
) -> tuple[ResolvedQualifier, ...]:
    """The codes `bindings` bind on `shape`'s what and how slots that no row of the
    datasets its part answers from, less the datasets it excludes, holds, itself or as a
    descendant (`_part_codes`), each as an unbound record under its label.

    A part whose source slot names a table elects from that table's rows alone, so a code
    the bind model chose from the unnamed part's menu that the table lacks can bind on none
    of them, as a term `_undeclared_qualifiers` refuses for a part naming a table. A
    shape the index does not carry has no rows to check.
    """
    index = get_index()
    if excluded and shape.datasets is not None:
        shape = replace(
            shape, datasets=tuple(dataset for dataset in shape.datasets if dataset not in excluded)
        )
    codes = _part_codes(shape, index)
    if codes is None:
        return ()
    unheld: list[ResolvedQualifier] = []
    for binding in bindings:
        if binding.axis not in ("what", "how") or binding.property_dcid is None:
            continue
        held = codes.get(binding.property_dcid, frozenset())
        for dcid in binding.value_dcids if binding.kind in ("value", "set") else ():
            if held & frozenset(index.descendants(dcid)):
                continue
            row = index.code_by_dcid(dcid)
            label = row.label if row is not None else dcid
            unheld.append(
                ResolvedQualifier(
                    text=label,
                    named=[GraphRef(dcid=dcid, label=label)],
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


def _default_reading(state: ResolveState, free_matches: Sequence[QualifierMatch]) -> DefaultReading:
    """The price-basis order and requested years the default-reading collapse applies.

    A price basis is named by a free-axis match, or by a two-reading span ("PPP") whose
    measure reading some eligible shape carries alone (`reading_for_shape`).
    `price_basis_order` ignores values of other axes.
    """
    named = [match.code_dcid for match in free_matches]
    named += [
        readings.measure.code_dcid
        for readings in state.query_qualifier_readings
        if any(reading_for_shape(readings, s) is readings.measure for s in state.eligible_shapes)
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


def _phrases_reader(state: ResolveState) -> Callable[[ShapeDraft], frozenset[str]]:
    """The query phrases each shape's family reads (`query_phrases_read`), in the shape's
    own vocabulary's matches and ties, each shape read once."""
    read: dict[int, frozenset[str]] = {}

    def phrases_read(shape: ShapeDraft) -> frozenset[str]:
        if id(shape) not in read:
            read[id(shape)] = query_phrases_read(
                state.held_codes_of(shape),
                for_shape(state.query_qualifier_matches, shape),
                for_shape(state.query_qualifier_ties, shape),
            )
        return read[id(shape)]

    return phrases_read


def _series_reader(state: ResolveState) -> Callable[[ShapeDraft], bool | None]:
    """Whether each role-bearing shape's family holds a series at the places the part
    names, on the roles they bind for the shape (`bind_place_roles`), read from the
    request's coverage snapshot. A set gives its node and its members. The family is the
    rows its part may answer from (`part_member_rows`), else its recalled members. `None`
    when the shape is subject-only, refuses the places, binds none, or has no member the
    coverage index holds."""
    coverage = state.coverage.index
    index = get_index()

    def series_at_places(shape: ShapeDraft) -> bool | None:
        if not is_role_bearing(shape):
            return None
        assignment = bind_place_roles(state, shape)
        if isinstance(assignment, str) or assignment.binds_no_place:
            return None
        bound = {
            _COVERAGE_ROLE_FOR_AXIS[axis]: frozenset(
                dcid
                for place_set in sets
                for dcid in (*place_set.dcids("default"), *place_set.dcids("members"))
            )
            for axis, sets in assignment.sets.items()
            if axis in _COVERAGE_ROLE_FOR_AXIS and sets
        }
        members = [row.dcid for row in part_member_rows(shape, index)] or list(
            shape.sv_arc_facts or ()
        )
        covered = [dcid for dcid in members if coverage.covers(dcid)]
        if not bound or not covered:
            return None
        return bool(coverage.filter_candidates(variables=covered, bound=bound))

    return series_at_places


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

        read_matches = tuple(match for reading in readings for match in reading.matches)
        seen_codes = {(m.code_system, m.code_dcid) for m in read_matches}
        joined_matches: list[QualifierMatch] = []
        for match in sidecar_matches:
            key = (match.code_system, match.code_dcid)
            if key in seen_codes or not _in_vocabulary(match.code_system, vocabulary):
                continue
            seen_codes.add(key)
            joined_matches.append(match)
        matches[vocabulary] = read_matches + tuple(joined_matches)

        read_ties = tuple(tie for reading in readings for tie in reading.ties)
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
    choice. A tie of a declared system whose codes the family holds none of is `unheld`, as
    "goods donations" ties two CRS cooperation modalities that no CRS flow row holds.
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


def _force_qualifier_bindings(
    bindings: list[SlotBindingDraft],
    matches: tuple[QualifierMatch, ...],
    shape_draft: ShapeDraft,
) -> list[SlotBindingDraft]:
    """Replace each admitted property's binding with its graph-sourced value.

    Two matches on one property that nest name one slot twice, and the descendant binds
    whatever their order: "refugee costs in donor countries ODA" binds DAC1's in-donor
    refugee costs, not ODA, its ancestor. Matches that do not nest still collide, and the
    last one binds.
    """
    axis_by_property = {
        slot.property_dcid: slot.axis for slot in shape_draft.slot_keys if slot.property_dcid
    }
    index = get_index()
    narrower = [
        match
        for match in matches
        if not any(
            other.code_system == match.code_system
            and other.code_dcid != match.code_dcid
            and other.code_dcid in index.descendants(match.code_dcid)
            for other in matches
        )
    ]
    forced = {
        match.code_system: SlotBindingDraft(
            axis=axis_by_property[match.code_system],
            property_dcid=match.code_system,
            kind="value",
            value_dcids=[match.code_dcid],
        )
        for match in narrower
        if match.code_system in axis_by_property
    }
    if not forced:
        return bindings
    kept = [b for b in bindings if b.property_dcid not in forced]
    return kept + list(forced.values())


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
) -> tuple[Literal["bound", "unbound"], str | None]:
    """Return bound when a final bound value names a descendant of a recorded code.

    Parent codes remain unbound. Spans and ties share this rule.
    """
    for code_system, code_dcid in rows:
        if set(bound_values.get(code_system, ())) & set(index.descendants(code_dcid)):
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
    answers from: the settled one (`ResolveState.settled_dataset`), else each dataset the
    elected shape answers from (`ShapeDraft.datasets`), since election may serve any of them.

    Election applied such a word by choosing the table: "imputed" in "imputed education aid"
    picks "ONE Imputed Multilateral Sectors", whose name describes every row it serves.
    """
    settled = state.settled_dataset()
    datasets = (settled,) if settled is not None else state.shape_draft.datasets or ()
    labels = [index.datasets[dataset].label for dataset in datasets if dataset in index.datasets]
    words = {word.casefold() for word in _WORD_RE.findall(phrase)}
    return (
        bool(words)
        and bool(labels)
        and all(words <= {word.casefold() for word in _WORD_RE.findall(label)} for label in labels)
    )


def _resolve_query_qualifiers(state: ResolveState) -> list[ResolvedQualifier]:
    """Build qualifier dispositions from final bindings without mutating state.

    The part's named dataset has no record here. Candidacy refused the part when that
    dataset published no recalled shape, and election bound the source to it otherwise.
    Five sources contribute records:

    1. Variable spans, as the elected shape's own vocabulary reads them
       (`ResolveState.variable_spans_by_vocabulary`), contribute their ``CodeRow`` values,
       all of systems the shape declares, so a CRS answer's "ODA" names the CRS flow type,
       not the DAC1 aid type of the same text. A word that reading leaves unread is no
       recognised span, so it can return as an ``unplaced`` phrase (source 3).
       Measure-value-only spans are handled by measure warnings. Disposition uses
       ``_qualifier_disposition``.
    2. ``state.admitted_ties`` records model-offered ambiguities unresolved by the final
       bindings. Keep ties with a row property declared by the elected shape, apply the
       shared disposition rule, and deduplicate normalized phrases against variable spans.
    3. An ``unplaced`` phrase must occur in ``variable`` outside recognised spans, must not
       be a per-person rate base the elected shape's denominator carries, and must not be, as
       a whole, the declared plain name of a slot the elected shape declares, singular or
       plural (`names_plain_name`): "sector" on a shape with `crsPurpose` names a distinction
       the data records, and its slot stays as bound. A longer phrase holding the noun ("space sector", "by sector") still refuses.
       Nor may every word of it occur in the label of a dataset the elected shape answers
       from (`_names_answering_dataset`): "imputed" on the imputed table chose that table.
       Surviving phrases get ``no_vocabulary`` with no named codes.
    4. Match ``non-concessional`` in variable text and optional query text only when the
       shape declares ``crsTypeOfFlow``. Record ``no_vocabulary`` and refuse the qualifier.
    5. Typed-sidecar spans use the same admission and disposition rules, then deduplicate
       against text spans and ties. Their offsets belong to each sidecar phrase and stay out
       of the overlap check on ``variable``.
    """
    index = get_index()
    declared = shape_vocabulary(state.shape_draft)
    foreign = other_families_properties(state.shape_draft, index)
    # Each property's final value or set binding.
    bound_values: dict[str, Sequence[str]] = {
        b.property_dcid: b.value_dcids
        for b in state.bindings
        if b.property_dcid is not None and b.kind in ("value", "set")
    }
    # Preserve source offsets by casefolding without stripping.
    normalized_variable = state.variable.casefold()

    variable_spans = for_shape(state.variable_spans_by_vocabulary, state.shape_draft)
    resolved: list[ResolvedQualifier] = []
    occupied_ranges: list[tuple[int, int]] = [(span.start, span.end) for span in variable_spans]
    for span in variable_spans:
        code_rows = [row for row in span.rows if isinstance(row, CodeRow)]
        if not code_rows:
            continue
        named = [GraphRef(dcid=row.dcid, label=row.label) for row in code_rows]
        disposition, slot = _qualifier_disposition(
            [(row.code_system, row.dcid) for row in code_rows], bound_values, index
        )
        resolved.append(
            ResolvedQualifier(
                text=span.matched_text, named=named, slot=slot, disposition=disposition
            )
        )

    already_recorded = frozenset(_normalize(q.text) for q in resolved)
    for tie in state.admitted_ties:
        if _normalize(tie.matched_text) in already_recorded:
            continue
        if not any(row.code_system in declared for row in tie.rows):
            continue
        named = [GraphRef(dcid=row.code_dcid, label=row.label) for row in tie.rows]
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
            named = [GraphRef(dcid=row.dcid, label=row.label) for row in code_rows]
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
        phrase_ranges = _all_normalized_text_ranges(phrase, normalized_variable)
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
    if "crsTypeOfFlow" in declared:
        for text in (state.variable, state.query_text):
            if text is None:
                continue
            for phrase in match_unsupported_qualifier_phrases(text):
                normalized_phrase = _normalize(phrase)
                if normalized_phrase in already_recorded:
                    continue
                resolved.append(
                    ResolvedQualifier(text=phrase, named=[], slot=None, disposition="no_vocabulary")
                )
                already_recorded.add(normalized_phrase)

    return resolved


def _validate_requested_all_bindings(
    bindings: list[SlotBindingDraft],
    dropped_qualifier_phrases: list[str],
    *,
    slot_properties: frozenset[str],
) -> tuple[list[SlotBindingDraft], frozenset[str]]:
    """Validate model ``requested_all`` claims before defaults or warning suppression use them.

    A claim must be unbound, name a property on the elected shape, and reproduce a dropped
    phrase exactly. Demote invalid claims to ``mentioned`` and clear their phrase. Return
    bindings in order plus normalised phrases from valid claims only.
    """
    applied: set[str] = set()
    validated: list[SlotBindingDraft] = []
    for binding in bindings:
        if binding.evidence == "requested_all":
            if (
                binding.kind == "unbound"
                and binding.property_dcid in slot_properties
                and binding.requested_all_phrase is not None
                and binding.requested_all_phrase in dropped_qualifier_phrases
            ):
                applied.add(_normalize(binding.requested_all_phrase))
                validated.append(binding)
                continue
            binding = binding.model_copy(
                update={"evidence": "mentioned", "requested_all_phrase": None}
            )
        validated.append(binding)
    return validated, frozenset(applied)


def _validate_sole_option_bindings(
    bindings: list[SlotBindingDraft], shape_draft: ShapeDraft
) -> list[SlotBindingDraft]:
    """Validate model ``sole_option`` claims against option sets the index supplies.

    A ``sole_option`` claim says the slot offers one value and the phrase says nothing
    about it. For a shape whose membership the index decides
    (`slot_taxonomy_index_sourced`), `slot_taxonomy` holds every value the index's family
    carries for the property, whatever the recalled sample holds. A claim on a slot that
    offers two or more values is false. The phrase did not name the value and the slot
    did not force it, so no binding rule produced the draft and the slot is left unbound.
    A claim on a slot with one value, or with no taxonomy entry to check against, stays.

    Claims on a shape with recall-sourced option sets pass through. That set is a capped
    sample, and there the model files a value the phrase implies under ``sole_option``
    ("child mortality" as the under-five age band), which this check would open.
    """
    if not shape_draft.slot_taxonomy_index_sourced:
        return bindings
    slot_taxonomy = shape_draft.slot_taxonomy or {}
    validated: list[SlotBindingDraft] = []
    for binding in bindings:
        if binding.kind == "value" and binding.evidence == "sole_option":
            options = slot_taxonomy.get(f"{binding.axis}:{binding.property_dcid}") or ()
            if len(set(options)) > 1:
                binding = binding.model_copy(
                    update={"kind": "unbound", "value_dcids": [], "evidence": "mentioned"}
                )
        validated.append(binding)
    return validated


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
    declarations = get_index().declarations
    plain_names = {declaration.plain_name for declaration in declarations.values()}
    opened: dict[str, SlotBindingDraft] = {}
    for operation in operations:
        if operation.over != "category":
            continue
        noun = slot_noun_named(operation.phrase, plain_names)
        if noun is None:
            continue
        for slot in shape_draft.slot_keys:
            prop = slot.property_dcid
            if prop is None or prop in opened or prop not in declarations:
                continue
            if declarations[prop].plain_name == noun:
                opened[prop] = SlotBindingDraft(
                    axis=slot.axis,
                    property_dcid=prop,
                    kind="unbound",
                    evidence="requested_all",
                    requested_all_phrase=operation.phrase,
                )
    if not opened:
        return bindings

    def is_open(binding: SlotBindingDraft) -> bool:
        return binding.kind in ("unbound", "absent") and not binding.value_dcids

    result = [
        opened[b.property_dcid] if b.property_dcid in opened and is_open(b) else b for b in bindings
    ]
    bound = {b.property_dcid for b in bindings}
    result.extend(binding for prop, binding in opened.items() if prop not in bound)
    return result


async def _phase_bind(state: ResolveState) -> RegionResult | None:
    """Run one choose-and-bind call. Return a terminal result for ask or an invalid index.

    The call elects a shape and binds its slots. Each shape's own admitted qualifiers become
    its fixed context (`_shape_qualifiers`). Ties with one row become matches, while larger
    ties remain closed alternatives. Declared
    constraint-measure axes are withheld from index-sourced shapes for deterministic repair.
    Election vetting checks ``requested_all`` claims against the chosen shape.
    """
    per_shape = [_shape_qualifiers(state, shape) for shape in state.capped_shapes]
    admitted_qualifiers = [matches for matches, _ties, _unheld in per_shape]
    admitted_ties = [ties for _matches, ties, _unheld in per_shape]
    state.admitted_qualifiers = merge_lists(admitted_qualifiers)
    state.admitted_ties = merge_lists(admitted_ties)
    withheld_measure_axes = frozenset(
        axis.axis for axis in get_measure_index().axes if axis.kind == "constraint"
    )
    t0 = now_ms()
    bind_result: _UniversalBindOutput
    _bind_usage: dict | None
    bind_task = asyncio.create_task(
        bind_universal(
            state.variable,
            state.capped_shapes,
            admitted_qualifiers=admitted_qualifiers,
            admitted_ties=admitted_ties,
            withheld_measure_axes=withheld_measure_axes,
            dropped_qualifier_phrases=tuple(state.dropped_qualifier_phrases),
            llm=state.llm,
        )
    )
    try:
        if state.emit is not None:
            # Emit the top-ranked candidate while the bind call runs. It may differ from the choice.
            shape_event = await _build_shape_event(
                state.capped_shapes[0], variable_index=state.variable_index, graph=state.graph
            )
            await state.emit(shape_event)
        bind_result, _bind_usage = await bind_task
        state.bind_result = bind_result
    finally:
        # Cancel and await an unfinished bind task so cancellation cannot detach it.
        if not bind_task.done():
            bind_task.cancel()
            await asyncio.gather(bind_task, return_exceptions=True)
    _var_usage = state.var_usage = _bind_usage
    state.timing["bind"] = now_ms() - t0
    await _emit_step(
        state.emit,
        state.variable_index,
        "bind",
        ms=state.timing["bind"],
    )

    # An ask or out-of-range index returns ``variable_not_resolved`` immediately.
    if bind_result.ask or not (0 <= bind_result.chosen_shape_index < len(state.capped_shapes)):
        return _unresolved_entity_terminal(state) or state.no_data(
            "variable_not_resolved", llm_usage=_var_usage
        )
    return None
