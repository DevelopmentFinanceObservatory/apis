"""Validate shape election, qualifiers, and bindings before materialisation."""

from __future__ import annotations

from code_index.shared import get_index, get_measure_index
from search.engine.bind import SlotBindingDraft
from search.engine.bind_repair import (
    apply_declared_defaults,
    apply_shape_slot_defaults,
    defaulted_slots_named,
    detect_degenerate_bindings,
    repair_draft_bindings,
    restore_starved_axis_binding,
)
from search.engine.config import settings
from search.engine.departures import measure_not_applied
from search.engine.extract import slot_default_answers
from search.engine.qualifier_match import QualifierMatch, QualifierTie, _normalize
from search.engine.region.availability import _shape_group_key
from search.engine.region.candidate_options import _elected_sibling_warning
from search.engine.region.codes import (
    DEGENERATE_BINDING_DETECTED,
    NEAR_NEIGHBOR_AMBIGUOUS,
    QUERY_QUALIFIER_DROPPED,
    REDUNDANT_BINDING_DROPPED,
    RETRIEVAL_SCORE_WEAK,
    SLOT_DEFAULTED,
    STARVED_AXIS_BINDING_RESTORED,
)
from search.engine.region.named_source import with_mention, without_dataset_drafts
from search.engine.region.near_neighbor import (
    _discriminating_token,
    _near_neighbor_check,
    _representative_score_is_weak,
)
from search.engine.region.phase_intake import _unresolved_entity_terminal
from search.engine.region.phase_rank_bind import (
    _admit_qualifier_matches_against_elected_shape,
    _force_qualifier_bindings,
    _log_admitted_qualifiers,
    _open_operation_slots,
    _resolve_query_qualifiers,
    _validate_requested_all_bindings,
    _validate_sole_option_bindings,
)
from search.engine.region.qualifier_readings import (
    bound_value_reads_measure,
    reading_for_shape,
    shape_carries_measure_match,
)
from search.engine.region.result import RegionResult
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import (
    bind_shared_values,
    elect_measure_family,
    shape_query,
)
from search.engine.shape import ShapeDraft
from search.models import Warning


async def _phase_vet_election(state: ResolveState) -> RegionResult | None:
    """Select the bound shape, validate claims, and record ambiguity warnings.

    A drafted binding on the source is discarded. The dataset mention read for the
    variable binds the source (`with_mention`), since shape candidacy left only that
    dataset's shapes to elect from. The near-neighbour gate supplements model alternates.
    Invalid ``requested_all`` claims are demoted, and on a shape whose option sets the index
    supplies a ``sole_option`` claim on a slot that offers two or more values leaves the
    slot unbound. Only validated phrases suppress defaults
    and dropped-qualifier warnings.
    An operation over a category slot opens that slot as a valid claim whether or not
    bind made one.
    """
    capped_shapes = state.capped_shapes
    bind_result = state.bind_result
    shape_draft: ShapeDraft = capped_shapes[bind_result.chosen_shape_index]
    state.shape_draft = shape_draft
    # The prune's substitution of a sibling with data for empty shapes is stated only when
    # bind elects that sibling. An empty shape beside another election was never chosen.
    substituted = [
        (dropped, survivor)
        for dropped, survivor in state.measure_head_substitutions
        if _shape_group_key(survivor) == _shape_group_key(shape_draft)
    ]
    if substituted:
        state.warnings.append(
            _elected_sibling_warning(
                no_data=[dropped.label for dropped, _ in substituted],
                elected=[survivor.label for _, survivor in substituted],
            )
        )
    bind_result.bindings = without_dataset_drafts(bind_result.bindings)
    bind_result.bindings, state.applied_dropped_qualifier_phrases = (
        _validate_requested_all_bindings(
            bind_result.bindings,
            state.dropped_qualifier_phrases,
            slot_properties=frozenset(
                slot.property_dcid
                for slot in shape_draft.slot_keys
                if slot.property_dcid is not None
            ),
        )
    )
    bind_result.bindings = _validate_sole_option_bindings(bind_result.bindings, shape_draft)
    bind_result.bindings = _open_operation_slots(
        bind_result.bindings, state.operations, shape_draft
    )

    # Share the weak-score predicate with the near-neighbour gate. A topic expansion is
    # judged on its root's recall score, as the floor judges it.
    if _representative_score_is_weak(shape_draft.measured_score):
        state.warnings.append(
            Warning(
                code=RETRIEVAL_SCORE_WEAK,
                severity="info",
                message=(
                    "the chosen reading's match to the query is weak; treat it "
                    "with less confidence than a strong match."
                ),
            )
        )

    # Track the fired index for candidate resolution and retain the Warning object so an
    # equivalent-reading diversion can retract it.
    near_neighbor_fired_index: int | None = None
    near_neighbor_warning: Warning | None = None
    state.near_neighbor_fired_index = near_neighbor_fired_index
    state.near_neighbor_warning = near_neighbor_warning
    neighbor_check = _near_neighbor_check(
        state.variable, shape_draft, bind_result.chosen_shape_index, capped_shapes
    )
    if neighbor_check is not None:
        neighbor_index, margin, similarity = neighbor_check
        neighbor_shape = capped_shapes[neighbor_index]
        # Require the configured similarity floor and a query token unique to the chosen
        # label. A neighbour-only token keeps the ambiguity live.
        explicitly_disambiguated = similarity >= settings().near_label_similarity and (
            _discriminating_token(state.variable, shape_draft.label, neighbor_shape.label)
            is not None
        )
        if margin < settings().near_margin and not explicitly_disambiguated:
            near_neighbor_fired_index = state.near_neighbor_fired_index = neighbor_index
            # Put the fired neighbour first so both candidate caps retain it, including
            # ``max_candidates=2`` and lists with 5 or more model alternates.
            _also_plausible_rest = [
                i for i in (bind_result.also_plausible or []) if i != neighbor_index
            ]
            bind_result.also_plausible = [neighbor_index, *_also_plausible_rest]
            # Assembly replaces this placeholder with served labels, or drops it if fewer
            # than two candidates survive.
            near_neighbor_warning = state.near_neighbor_warning = Warning(
                code=NEAR_NEIGHBOR_AMBIGUOUS,
                severity="info",
                message=(
                    "the chosen reading and a semantically distinct alternative "
                    "scored closely enough to be ambiguous; surfacing both as "
                    "candidates."
                ),
            )
            state.warnings.append(near_neighbor_warning)

    # Vet model tie choices against its original bindings before forced matches are added.
    # A row counts when a value binding names its code. One bound row satisfies the tie.
    # Zero or multiple rows leave it unresolved. Multiple bound properties reopen.
    bindings_by_property = {
        b.property_dcid: b for b in bind_result.bindings if b.property_dcid is not None
    }
    elected_tie_phrases: set[str] = set()
    tie_dropped: dict[str, QualifierTie] = {}
    properties_to_unbind: set[str] = set()
    for tie in state.admitted_ties:
        bound_rows = [
            row
            for row in tie.rows
            if (binding := bindings_by_property.get(row.code_system)) is not None
            and binding.kind == "value"
            and row.code_dcid in binding.value_dcids
        ]
        phrase = _normalize(tie.matched_text)
        if len(bound_rows) == 1:
            elected_tie_phrases.add(phrase)
        else:
            tie_dropped.setdefault(phrase, tie)
            properties_to_unbind.update(row.code_system for row in bound_rows)
    if properties_to_unbind:
        bind_result.bindings = [
            SlotBindingDraft(axis=b.axis, property_dcid=b.property_dcid, kind="unbound")
            if b.property_dcid in properties_to_unbind
            else b
            for b in bind_result.bindings
        ]
    state.elected_tie_phrases = frozenset(elected_tie_phrases)
    for phrase in sorted(tie_dropped.keys() & state.variable_matched_qualifier_phrases):
        tie = tie_dropped[phrase]
        systems = {row.code_system for row in tie.rows}
        state.depart(
            _qualifier_dropped_warning(phrase),
            measure_not_applied(
                asked=tie.matched_text,
                property_dcid=systems.pop() if len(systems) == 1 else None,
            ),
        )

    # Keep only qualifier matches admitted by the elected shape. A two-reading span admits
    # its code reading only when the elected shape carries that reading alone.
    readings = {r: reading_for_shape(r, shape_draft) for r in state.query_qualifier_readings}
    reading_codes = {r.code for r in readings}
    admitted_qualifiers = _admit_qualifier_matches_against_elected_shape(
        tuple(m for m in state.admitted_qualifiers if m not in reading_codes)
        + tuple(r.code for r, kept in readings.items() if kept is r.code),
        shape_draft,
    )
    state.admitted_qualifiers = admitted_qualifiers
    if admitted_qualifiers:
        bind_result.bindings = _force_qualifier_bindings(
            bind_result.bindings, admitted_qualifiers, shape_draft
        )
        _log_admitted_qualifiers(admitted_qualifiers)

    # The dataset the variable's read mention asks for.
    if state.dataset_mention is not None:
        bind_result.bindings = with_mention(bind_result.bindings, state.dataset_mention)

    # Record measure values the elected shape carries and warn on unsupported or conflicting
    # values. Raw-text warnings own dropped phrases. This path warns only for its own phrases.
    # A conflict phrase names two values on one axis, so it keeps no single match. A
    # two-reading span is satisfied when the shape reads it as the measure value, and was
    # admitted above when the shape reads it as the code. It is unapplied when the shape
    # carries both readings or neither. A measure value that a bound constraint value reads
    # is satisfied too (`bound_value_reads_measure`).
    unsatisfied: dict[str, QualifierMatch | None] = dict.fromkeys(
        state.measure_qualifier_conflict_phrases
    )
    satisfied_phrases: set[str] = set()
    for match in state.measure_query_matches:
        phrase = _normalize(match.matched_text)
        if shape_carries_measure_match(shape_draft, match) or bound_value_reads_measure(
            shape_draft, match, bind_result.bindings
        ):
            satisfied_phrases.add(phrase)
        else:
            unsatisfied.setdefault(phrase, match)
    for r, kept in readings.items():
        if kept is r.measure or (
            kept is None and bound_value_reads_measure(shape_draft, r.measure, bind_result.bindings)
        ):
            satisfied_phrases.add(_normalize(r.matched_text))
        elif kept is None:
            unsatisfied.setdefault(_normalize(r.matched_text), r.measure)
    state.satisfied_measure_phrases = frozenset(satisfied_phrases)
    for phrase in sorted(unsatisfied.keys() & state.variable_matched_qualifier_phrases):
        match = unsatisfied[phrase]
        state.depart(
            _qualifier_dropped_warning(phrase),
            measure_not_applied(
                asked=match.matched_text if match is not None else phrase,
                property_dcid=match.code_system if match is not None else None,
            ),
        )

    return None


def _qualifier_dropped_warning(phrase: str) -> Warning:
    """The `query_qualifier_dropped` warning for a qualifier phrase the answer does not
    apply."""
    return Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message=f'The requested qualifier "{phrase}" was not applied.',
    )


async def _phase_repair_bindings(state: ResolveState) -> RegionResult | None:
    """Repair bindings, apply measure elections and defaults, and refuse degenerate drafts.

    In order, detect degenerate values, repair drafts, restore starved axes, force open
    measure axes, then apply declared defaults. Place binding reads the result.
    """
    bindings: list[SlotBindingDraft] = list(state.bind_result.bindings)
    state.bindings = bindings

    # Empty value/set bindings assert no constraint and are refused before materialisation.
    bindings, _degenerate_bindings = detect_degenerate_bindings(bindings)
    state.bindings = bindings
    for _degenerate in _degenerate_bindings:
        state.warnings.append(
            Warning(
                code=DEGENERATE_BINDING_DETECTED,
                severity="degraded",
                message=(
                    f"{_degenerate.property_dcid!r} was drafted as kind={_degenerate.kind!r} "
                    "with no value named; refusing rather than resolving an "
                    "unconstrained axis the query never actually bound."
                ),
            )
        )
    if _degenerate_bindings:
        return _unresolved_entity_terminal(state) or state.no_data(
            "variable_not_resolved", llm_usage=state.var_usage
        )

    # Drop redundant single-member aliases so an answer using the family's other spelling
    # an answer using the family's other spelling.
    bindings, _dropped_bindings = repair_draft_bindings(
        bindings,
        list(state.shape_draft.sv_arc_facts or {}),
        state.shape_draft.sv_arc_facts or {},
        state.shape_draft.slot_taxonomy or {},
    )
    state.bindings = bindings
    for _dropped in _dropped_bindings:
        state.warnings.append(
            Warning(
                code=REDUNDANT_BINDING_DROPPED,
                severity="warn",
                message=(
                    f"{_dropped.property_dcid!r} was drafted as a redundant, "
                    "never-mentioned spelling of an already-answerable axis; "
                    "left unbound so the family's own answer under its other "
                    "spelling is not refused."
                ),
            )
        )

    # The place-availability floor can hide an axis. Restore a unique value named by the
    # query when that axis is otherwise closed in the sampled taxonomy.
    bindings, _restored_bindings = restore_starved_axis_binding(
        bindings, state.shape_draft, state.variable
    )
    state.bindings = bindings
    for _restored in _restored_bindings:
        state.warnings.append(
            Warning(
                code=STARVED_AXIS_BINDING_RESTORED,
                severity="warn",
                message=(
                    f"{_restored.property_dcid!r} was drafted as kind={_restored.kind!r} "
                    "because the availability floor emptied its slot_taxonomy at "
                    "the queried place; the variable phrase names one of its "
                    "unscoped values, so the binding is restored to that value."
                ),
            )
        )

    # A slot the five-tuple pins binds its one value, and a family the index does not
    # export on a five-tuple it carries binds the values its members share, so its identity
    # resolves apart from the exported family.
    bindings, _shared_bindings = bind_shared_values(state.shape_draft, bindings)
    state.bindings = bindings

    # Force measure axes left open after query matches and starved-axis restoration.
    # Run after draft repair, which judges realizability against a capped recall sample.
    # Keep dropped axes for slot metadata and the elected partition for the sentence event.
    # The final Spec derives its measure from served rows, which can differ from this vote.
    election = elect_measure_family(
        state.shape_draft,
        get_index(),
        get_measure_index().axes,
        shape_query(state.shape_draft, bindings),
    )
    forced_pairs = election.forced if election is not None else ()
    state.measure_dropped_axes = election.dropped if election is not None else ()
    state.measure_elected_partition = election.elected if election is not None else None
    if forced_pairs:
        axis_by_property = {
            slot.property_dcid: slot.axis
            for slot in state.shape_draft.slot_keys
            if slot.property_dcid
        }
        forced_bindings = []
        forced_props: set[str] = set()
        for prop, value in forced_pairs:
            axis = axis_by_property.get(prop)
            if axis is None:
                continue
            forced_bindings.append(
                SlotBindingDraft(axis=axis, property_dcid=prop, kind="value", value_dcids=[value])
            )
            forced_props.add(prop)
        if forced_bindings:
            bindings = [
                b
                for b in bindings
                if not (b.property_dcid in forced_props and b.kind in ("unbound", "absent"))
            ] + forced_bindings
            state.bindings = bindings
            state.measure_elected_properties = frozenset(forced_props)

    # Apply a declared default when the variable, raw query, or bindings trigger it. Raw
    # query text is available only for a single-variable request and participates in the
    # response cache key. Engine-authored defaults add no warning.
    bindings, _defaulted_bindings = apply_declared_defaults(
        bindings,
        state.shape_draft,
        index=get_index(),
        variable=state.variable,
        query_text=state.query_text,
        query_qualifier_matches=state.query_qualifier_matches,
        query_qualifier_ties=state.query_qualifier_ties,
        engine_authored_properties=state.measure_elected_properties,
    )
    if _defaulted_bindings:
        state.bindings = bindings
        state.defaulted_properties = frozenset(
            b.property_dcid for b in _defaulted_bindings if b.property_dcid is not None
        )
        # A default on a slot the query names by its plain name ("top 5 aid flows") warns,
        # and answers each dropped phrase that asked for no more than that slot.
        named = defaulted_slots_named(
            _defaulted_bindings,
            index=get_index(),
            texts=[
                text
                for text in (
                    state.variable,
                    state.query_text,
                    *state.dropped_qualifier_phrases,
                    *(operation.phrase for operation in state.operations),
                )
                if text is not None
            ],
        )
        for plain_name, label in named:
            state.warnings.append(
                Warning(
                    code=SLOT_DEFAULTED,
                    severity="warn",
                    message=(
                        f"No {plain_name} was named, so the answer covers {label}, "
                        f"the default {plain_name}."
                    ),
                )
            )
        state.slot_default_phrases = frozenset(
            _normalize(phrase)
            for phrase in state.dropped_qualifier_phrases
            if any(
                slot_default_answers(phrase, plain_name=plain_name, variable=state.variable)
                for plain_name, _label in named
            )
        )

    # Bind each slot the elected shape's population type declares a default for and the
    # query left open. This is the final binding mutation. Qualifier dispositions read the
    # resulting list.
    bindings, _shape_defaulted_bindings = apply_shape_slot_defaults(bindings, state.shape_draft)
    if _shape_defaulted_bindings:
        state.bindings = bindings
        state.defaulted_properties |= frozenset(
            b.property_dcid for b in _shape_defaulted_bindings if b.property_dcid is not None
        )

    # Resolve query qualifiers against the final bindings.
    state.query_qualifiers = _resolve_query_qualifiers(state)

    return None
