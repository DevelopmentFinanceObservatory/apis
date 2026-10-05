"""Rank shapes, admit query qualifiers, and run the universal choose-and-bind call."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from code_index.members import is_opt_in
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
from search.engine.dataset_mention import (
    DatasetMention,
    dataset_mentions,
    mention_measure,
    whole_mention,
    without_mention,
)
from search.engine.envelope import _record_step, now_ms
from search.engine.interpret import Recall
from search.engine.qualifier_match import (
    CodedQualifier,
    QualifierMatch,
    QualifierSpan,
    QualifierTie,
    _all_normalized_text_ranges,
    _match_source,
    _normalize,
    _ranges_overlap,
    _row_system,
    match_qualifier_ties,
    match_qualifiers,
    match_unsupported_qualifier_phrases,
    recognise_spans,
    slot_noun_named,
)
from search.engine.region.alternate_guard import (
    PER_HEAD_RATE_BASE_RE,
    PER_PERSON_DENOMINATORS,
    STRONG_RATE_REQUEST_RE,
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
from search.engine.region.place_roles import subject_shapes_refused
from search.engine.region.qualifier_readings import match_query_readings, reading_for_shape
from search.engine.region.rate_head import (
    _labels_share_rate_head,
    _rate_head,
    _rate_head_compatible,
)
from search.engine.region.result import RegionResult
from search.engine.region.sentence_events import _build_shape_event
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import (
    DefaultReading,
    collapse_measure_families,
    elect_five_tuple_family,
    index_carries_shape,
    part_datasets,
    part_member_rows,
    price_basis_order,
    shape_datasets,
    shape_for_part,
)
from search.engine.retrieval.representative import _shape_disagreement_props
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
    `rates` or a share term (`alternate_guard.STRONG_RATE_REQUEST_RE`), and no count term
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


def _constraint_property_dcids(shape: ShapeDraft) -> set[str]:
    """Return the constraint properties a candidate can carry into binding."""
    return {slot.property_dcid for slot in shape.slot_keys if slot.property_dcid is not None}


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
        len(mentioned_props - _constraint_property_dcids(candidate)) >= 2
        for candidate in normalized_shapes
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
            len(mentioned_props - _constraint_property_dcids(candidate)) >= 2
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


async def _phase_prune_and_rank(state: ResolveState) -> RegionResult | None:
    """Build the ranked, capped candidate set. Return a terminal result if it is empty.

    Applies dataset candidacy (`_dataset_candidates`), the score floor, the measure-class
    check (`shape_measure_class`), the two-ended prune, availability and measure pruning,
    then reserves recalled shapes within the top-K menu. A part whose bound mention's
    dataset publishes none of the recalled shapes refuses ``qualifier_not_bound``, naming
    the mention, and so does one whose variable names a code no row of that dataset's
    candidate shapes holds, naming the term (`_undeclared_qualifiers`). Both refusals
    come before any role check. Falling through leaves ``_phase_bind`` to call the model.
    """
    state.dataset_mention = _read_mention(state.qualifiers)
    candidates = _dataset_candidates(state.shapes_found, state.dataset_mention)
    if not candidates and state.dataset_mention is not None:
        await _record_shape_step(state)
        return state.no_data(
            "qualifier_not_bound",
            unbound_qualifiers=(_unbound_mention(state.dataset_mention),),
        )
    # Full spans retain ambiguous rows for final dispositions. Dropped phrases remain owned
    # by the raw-text path.
    state.variable_qualifier_spans = recognise_spans(
        state.variable, get_index(), get_measure_index()
    )
    if state.dataset_mention is not None:
        undeclared = _undeclared_qualifiers(state.variable_qualifier_spans, candidates)
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
    if _has_strong_rate_intent(state.variable):
        eligible_shapes = _prune_explicit_rate_raw_counts(eligible_shapes, state.variable)
    state.eligible_shapes = eligible_shapes
    # Share one recognition pass between the availability prune, query-aware measure
    # collapse and bind admission.
    query_qualifier_matches, variable_matched_qualifier_phrases = _match_query_qualifiers(
        state.variable,
        state.dropped_qualifier_phrases,
        query_text=state.query_text,
        qualifiers=state.qualifiers,
    )
    state.query_qualifier_matches = query_qualifier_matches
    # Spans naming a code and a price basis ("PPP") are read per shape, so the prune, the
    # guard and bind admission each keep the reading their shape carries.
    query_qualifier_readings, reading_owned_phrases = match_query_readings(
        state.variable, state.dropped_qualifier_phrases, query_text=state.query_text
    )
    state.query_qualifier_readings = query_qualifier_readings
    # No-place and region-expanded queries need a supplemental availability probe. It uses
    # named places plus a bounded region sample, or representative places when none are
    # named. Probe failure leaves availability unknown and adds a warning.
    #
    # Measure-head pruning needs a named place. Denominator-resolution pruning applies to
    # named and representative-place probes.
    named_place = bool(state.recall_only_place_dcids or state.place_children_by_name)
    if not state.entities or state.place_children_by_name:
        eligible_shapes, _availability_probe_failed = await asyncio.to_thread(
            _resolve_sibling_availability,
            eligible_shapes,
            graph=state.graph,
            probe_places=_sibling_probe_places(
                state.recall_only_place_dcids,
                state.place_children_by_name,
                cast(Recall, state.rcl).resolved_entity_names,
            ),
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
        qualifier_matches=query_qualifier_matches,
        qualifier_readings=query_qualifier_readings,
    )
    state.eligible_shapes = eligible_shapes
    if not eligible_shapes:
        await _record_shape_step(state)
        return _unresolved_entity_terminal(state) or state.no_data("variable_not_resolved")

    # The index's family and an unindexed family of one five-tuple are rival readings, and
    # one of them reaches the election.
    eligible_shapes = state.eligible_shapes = elect_five_tuple_family(
        eligible_shapes, query_qualifier_matches
    )
    # Merge tie-owned phrases so dropped-qualifier warnings cover matches and ties alike.
    query_qualifier_ties, tie_owned_phrases = _match_query_qualifier_ties(
        state.variable,
        state.dropped_qualifier_phrases,
        query_text=state.query_text,
        qualifiers=state.qualifiers,
    )
    state.query_qualifier_ties = query_qualifier_ties
    state.variable_matched_qualifier_phrases = (
        variable_matched_qualifier_phrases | tie_owned_phrases | reading_owned_phrases
    )
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
    """Record the shape step as run, timed from the start of shape recall."""
    state.timing["shape"] = now_ms() - state.shape_t0
    await _record_step(
        state.pipeline_steps,
        state.emit,
        state.variable_index,
        "shape",
        ran=True,
        ms=state.timing["shape"],
    )
    state.ran_steps.add("shape")


def _dataset_candidates(
    shapes: Sequence[ShapeDraft], mention: DatasetMention | None
) -> list[ShapeDraft]:
    """The shapes a part may answer from, by the datasets that publish each
    (`shape_datasets`), each stamped with the datasets the part may answer it from
    (`shape_for_part`).

    A part whose variable has a bound mention keeps the shapes whose five-tuple the
    mentioned dataset publishes. A part without one keeps a shape the index does not
    carry, and a shape whose five-tuple's eligible datasets
    (`code_index.members.eligible_datasets`) include one that is not opt-in, since an
    opt-in dataset answers only a request that names it (`DatasetDeclaration.opt_in`).
    Every door that reads a question elects from what is left, so the default answer
    never holds an opt-in dataset's rows and a named dataset's part never answers from
    another.
    """
    index = get_index()
    source = mention.provenance if mention is not None else None
    if source is not None:
        kept = [shape for shape in shapes if source in shape_datasets(shape, index)]
    else:
        kept = [
            shape
            for shape in shapes
            if not shape_datasets(shape, index)
            or any(not is_opt_in(index, dataset) for dataset in part_datasets(shape, None, index))
        ]
    return [shape_for_part(shape, source, index) for shape in kept]


def _unbound_mention(mention: DatasetMention) -> ResolvedQualifier:
    """The record of a mention whose dataset publishes no shape the part recalled."""
    return ResolvedQualifier(
        text=mention.text,
        named=[GraphRef(dcid=mention.provenance, label=mention.label(get_index()))],
        slot=None,
        disposition="unbound",
    )


def _undeclared_qualifiers(
    spans: Sequence[QualifierSpan], shapes: Sequence[ShapeDraft]
) -> tuple[ResolvedQualifier, ...]:
    """The variable's code terms that no row of a shape in `shapes` can hold, each as an
    unbound record.

    A part bound to a named dataset elects from that dataset's rows alone, so a code counts
    as held on a shape only when a row the dataset publishes of its five-tuple holds the
    code or one of its descendants (`_part_codes`), whatever other publishers of the
    five-tuple hold. Such a term can bind on none of them: "grants" on imputed multilateral
    aid, whose rows carry no type of finance, or a sector only another publisher of the
    five-tuple reports. A shape the index does not carry holds any code of a code system it
    declares a slot for. A span holding a measure value is the measure's to read and is
    left alone, as is a term the dataset's mention spans, which `recognise_spans` never
    reads.
    """
    index = get_index()
    readings = [(shape, _part_codes(shape, index)) for shape in shapes]

    def held(row: CodeRow) -> bool:
        descendants = frozenset(index.descendants(row.dcid))
        return any(
            row.code_system in _constraint_property_dcids(shape)
            if codes is None
            else bool(codes.get(row.code_system, frozenset()) & descendants)
            for shape, codes in readings
        )

    undeclared: list[ResolvedQualifier] = []
    for span in spans:
        code_rows = [row for row in span.rows if isinstance(row, CodeRow)]
        if len(code_rows) < len(span.rows) or any(held(row) for row in code_rows):
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


def _match_query_qualifiers(
    variable: str,
    dropped_qualifier_phrases: Sequence[str] = (),
    *,
    query_text: str | None = None,
    qualifiers: Sequence[CodedQualifier] | None = None,
) -> tuple[tuple[QualifierMatch, ...], frozenset[str]]:
    """Collect raw qualifier matches and the phrases owned by this region path.

    Match ``variable``, optional ``query_text``, and each dropped phrase separately so a
    match cannot span sources. Return matches before shape admission. The normalized second
    value contains phrases from ``variable``/``query_text`` and the typed sidecar, but omits
    dropped phrases because the raw-text path already owns their warnings.

    Sidecar matches deduplicate by ``(code_system, code_dcid)``. Their phrases remain owned
    even when a duplicate match is removed. Full qualifier spans are collected separately.
    """
    index = get_index()
    measure_index = get_measure_index()
    variable_matches = match_qualifiers(variable, index, measure_index)
    matches = variable_matches
    query_text_matches: tuple[QualifierMatch, ...] = ()
    if query_text:
        query_text_matches = match_qualifiers(query_text, index, measure_index)
        matches += query_text_matches
    for phrase in dropped_qualifier_phrases:
        matches += match_qualifiers(phrase, index, measure_index)
    owned_matches = variable_matches + query_text_matches
    variable_matched_phrases = frozenset(_normalize(m.matched_text) for m in owned_matches)
    if qualifiers:
        seen_codes = {(m.code_system, m.code_dcid) for m in matches}
        sidecar_matches: list[QualifierMatch] = []
        sidecar_phrases: set[str] = set()
        for coded in qualifiers:
            for match in coded.matches:
                sidecar_phrases.add(_normalize(match.matched_text))
                key = (match.code_system, match.code_dcid)
                if key in seen_codes:
                    continue
                seen_codes.add(key)
                sidecar_matches.append(match)
        matches += tuple(sidecar_matches)
        variable_matched_phrases |= frozenset(sidecar_phrases)
    return matches, variable_matched_phrases


def _match_query_qualifier_ties(
    variable: str,
    dropped_qualifier_phrases: Sequence[str] = (),
    *,
    query_text: str | None = None,
    qualifiers: Sequence[CodedQualifier] | None = None,
) -> tuple[tuple[QualifierTie, ...], frozenset[str]]:
    """Collect ambiguous qualifier ties from each text source and the typed sidecar.

    Match ``variable``, optional ``query_text``, and dropped phrases separately. Return
    normalized ownership phrases only for ``variable``/``query_text`` and sidecar ties.
    The raw-text path owns dropped phrases. Sidecar spans qualify only when they name at
    least two ``CodeRow`` values, and ties deduplicate by their row sets.
    """
    index = get_index()
    measure_index = get_measure_index()
    variable_ties = match_qualifier_ties(variable, index, measure_index)
    ties = variable_ties
    query_text_ties: tuple[QualifierTie, ...] = ()
    if query_text:
        query_text_ties = match_qualifier_ties(query_text, index, measure_index)
        ties += query_text_ties
    for phrase in dropped_qualifier_phrases:
        ties += match_qualifier_ties(phrase, index, measure_index)
    owned_ties = variable_ties + query_text_ties
    owned_phrases = frozenset(_normalize(t.matched_text) for t in owned_ties)
    if qualifiers:
        seen_row_sets = {
            frozenset((row.code_system, row.code_dcid) for row in tie.rows) for tie in ties
        }
        sidecar_ties: list[QualifierTie] = []
        sidecar_phrases: set[str] = set()
        for coded in qualifiers:
            for span in coded.spans:
                if len(span.rows) < 2 or not all(isinstance(row, CodeRow) for row in span.rows):
                    continue
                candidate_norm = _normalize(span.matched_text)
                sidecar_phrases.add(candidate_norm)
                tie_rows = tuple(
                    QualifierMatch(
                        code_system=_row_system(row),
                        code_dcid=row.dcid,
                        label=row.label,
                        source=_match_source(row, candidate_norm),
                        matched_text=span.matched_text,
                    )
                    for row in span.rows
                )
                key = frozenset((r.code_system, r.code_dcid) for r in tie_rows)
                if key in seen_row_sets:
                    continue
                seen_row_sets.add(key)
                sidecar_ties.append(QualifierTie(matched_text=span.matched_text, rows=tie_rows))
        ties += tuple(sidecar_ties)
        owned_phrases |= frozenset(sidecar_phrases)
    return ties, owned_phrases


def _admit_qualifier_matches(
    shapes: list[ShapeDraft],
    matches: tuple[QualifierMatch, ...],
) -> tuple[QualifierMatch, ...]:
    """Keep matches whose constraint property is declared by any candidate shape.

    Election vetting narrows them to the chosen shape. The caller supplies the shared raw
    matches so each variable's text is recognised once.
    """
    declared = {prop for shape in shapes for prop in _constraint_property_dcids(shape)}
    if not declared:
        return ()
    return _filter_admitted_matches(matches, declared)


def _admit_qualifier_ties(
    shapes: list[ShapeDraft],
    ties: tuple[QualifierTie, ...],
) -> tuple[tuple[QualifierMatch, ...], tuple[QualifierTie, ...]]:
    """Narrow ties to rows whose properties some candidate declares.

    Drop empty ties, promote a single surviving row to an ordinary match, and pass ties
    with two or more rows to the model for election. Vet that election after shape choice.
    """
    declared = {prop for shape in shapes for prop in _constraint_property_dcids(shape)}
    promoted: list[QualifierMatch] = []
    admitted: list[QualifierTie] = []
    for tie in ties:
        narrowed = tuple(row for row in tie.rows if row.code_system in declared)
        if not narrowed:
            continue
        if len(narrowed) == 1:
            promoted.append(narrowed[0])
            continue
        admitted.append(QualifierTie(matched_text=tie.matched_text, rows=narrowed))
    return tuple(promoted), tuple(admitted)


def _admit_qualifier_matches_against_elected_shape(
    matches: tuple[QualifierMatch, ...], shape_draft: ShapeDraft
) -> tuple[QualifierMatch, ...]:
    """Keep only matches whose constraint property the elected shape declares."""
    return _filter_admitted_matches(matches, _constraint_property_dcids(shape_draft))


def _force_qualifier_bindings(
    bindings: list[SlotBindingDraft],
    matches: tuple[QualifierMatch, ...],
    shape_draft: ShapeDraft,
) -> list[SlotBindingDraft]:
    """Replace each admitted property's binding with its graph-sourced value."""
    axis_by_property = {
        slot.property_dcid: slot.axis for slot in shape_draft.slot_keys if slot.property_dcid
    }
    forced = {
        match.code_system: SlotBindingDraft(
            axis=axis_by_property[match.code_system],
            property_dcid=match.code_system,
            kind="value",
            value_dcids=[match.code_dcid],
        )
        for match in matches
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


def _names_carried_per_head_rate(phrase: str, shape: ShapeDraft) -> bool:
    """Whether `phrase` is a per-person rate base that `shape`'s denominator carries.

    No constraint slot holds a denominator, so the bind model can list "per 100 people" as
    unplaced on a shape whose measurementDenominator is `Count_Person`.
    """
    return (
        shape.meas_denom_dcid in PER_PERSON_DENOMINATORS
        and PER_HEAD_RATE_BASE_RE.fullmatch(phrase.strip()) is not None
    )


def _read_mention(qualifiers: Sequence[CodedQualifier] | None) -> DatasetMention | None:
    """The dataset mention the request's reading bound to this variable, or `None`.

    The entry hands it over as a sidecar phrase that is a mention alone
    (`extract._with_mention_qualifiers`). No text scan of the variable, the query or a
    dropped phrase stands in for it, so a mention nobody read binds nothing.
    """
    index = get_index()
    return next(
        (
            mention
            for coded in qualifiers or ()
            if (mention := whole_mention(coded.phrase, index)) is not None
        ),
        None,
    )


def _resolve_query_qualifiers(state: ResolveState) -> list[ResolvedQualifier]:
    """Build qualifier dispositions from final bindings without mutating state.

    The dataset mention read for the variable has no record here. Candidacy refused the
    part when its dataset published no recalled shape, and election bound the source to it
    otherwise. Five sources contribute records:

    1. Variable spans contribute their ``CodeRow`` values only when the elected shape
       declares the property. Measure-value-only spans are handled by measure warnings.
       ``named`` keeps every CodeRow. Disposition uses ``_qualifier_disposition``.
    2. ``state.admitted_ties`` records model-offered ambiguities unresolved by the final
       bindings. Keep ties with a row property declared by the elected shape, apply the
       shared disposition rule, and deduplicate normalized phrases against variable spans.
    3. An ``unplaced`` phrase must occur in ``variable`` outside recognised spans and outside
       every dataset mention, and must not
       be a per-person rate base the elected shape's denominator carries. Surviving phrases
       get ``no_vocabulary`` with no named codes.
    4. Match ``non-concessional`` in variable text and optional query text only when the
       shape declares ``crsTypeOfFlow``. Record ``no_vocabulary`` and refuse the qualifier.
    5. Typed-sidecar spans use the same admission and disposition rules, then deduplicate
       against text spans and ties. Their offsets belong to each sidecar phrase and stay out
       of the overlap check on ``variable``.
    """
    index = get_index()
    declared = _constraint_property_dcids(state.shape_draft)
    # Each property's final value or set binding.
    bound_values: dict[str, Sequence[str]] = {
        b.property_dcid: b.value_dcids
        for b in state.bindings
        if b.property_dcid is not None and b.kind in ("value", "set")
    }
    # Preserve source offsets by casefolding without stripping.
    normalized_variable = state.variable.casefold()

    resolved: list[ResolvedQualifier] = []
    occupied_ranges: list[tuple[int, int]] = [
        (span.start, span.end) for span in state.variable_qualifier_spans
    ]
    # A dataset's name inside the variable is read by the request's mention reading, so
    # the bind model's echo of it is no qualifier without vocabulary.
    occupied_ranges.extend(
        (located.start, located.end) for located in dataset_mentions(state.variable, index)
    )
    for span in state.variable_qualifier_spans:
        code_rows = [row for row in span.rows if isinstance(row, CodeRow)]
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
            code_rows = [row for row in span.rows if isinstance(row, CodeRow)]
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

    for phrase in state.bind_result.unplaced:
        if _names_carried_per_head_rate(phrase, state.shape_draft):
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


def _bind_variable(state: ResolveState) -> str:
    """The part's variable less its bound mention. A variable that is the mention alone
    binds the measure the mention carries (`mention_measure`). "imputed multilateral aid"
    binds "aid", so the model reads the dataset's name as no qualifier. The variable stays
    whole only when no name of the dataset carries a measure word."""
    mention = state.dataset_mention
    if mention is None:
        return state.variable
    index = get_index()
    stripped = without_mention(state.variable, mention.provenance, index)
    if stripped:
        return stripped
    own = next(
        (m for m in dataset_mentions(state.variable, index) if m.provenance == mention.provenance),
        mention,
    )
    return mention_measure(own, index) or state.variable


async def _phase_bind(state: ResolveState) -> RegionResult | None:
    """Run one choose-and-bind call. Return a terminal result for ask or an invalid index.

    The call elects a shape and binds its slots. Admitted qualifiers become fixed context.
    Ties with one row become matches, while larger ties remain closed alternatives. Declared
    constraint-measure axes are withheld from index-sourced shapes for deterministic repair.
    The variable reaches the call without its bound mention (`without_mention`): candidacy
    has already used the mention up, and election binds its source. Election vetting checks
    ``requested_all`` claims against the chosen shape.
    """
    reading_codes = tuple(
        readings.code
        for readings in state.query_qualifier_readings
        if any(reading_for_shape(readings, shape) is readings.code for shape in state.capped_shapes)
    )
    admitted_qualifiers = _admit_qualifier_matches(
        state.capped_shapes, state.query_qualifier_matches + reading_codes
    )
    promoted_tie_matches, admitted_ties = _admit_qualifier_ties(
        state.capped_shapes, state.query_qualifier_ties
    )
    admitted_qualifiers = state.admitted_qualifiers = admitted_qualifiers + promoted_tie_matches
    state.admitted_ties = admitted_ties
    withheld_measure_axes = frozenset(
        axis.axis for axis in get_measure_index().axes if axis.kind == "constraint"
    )
    t0 = now_ms()
    bind_result: _UniversalBindOutput
    _bind_usage: dict | None
    bind_task = asyncio.create_task(
        bind_universal(
            _bind_variable(state),
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
    await _record_step(
        state.pipeline_steps,
        state.emit,
        state.variable_index,
        "bind",
        ran=True,
        ms=state.timing["bind"],
    )
    state.ran_steps.add("bind")

    # An ask or out-of-range index returns ``variable_not_resolved`` immediately.
    if bind_result.ask or not (0 <= bind_result.chosen_shape_index < len(state.capped_shapes)):
        return _unresolved_entity_terminal(state) or state.no_data(
            "variable_not_resolved", llm_usage=_var_usage
        )
    return None
