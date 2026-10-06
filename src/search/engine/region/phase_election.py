"""Validate shape election, qualifiers, and bindings before materialisation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress

from code_index.shared import get_index, get_measure_index
from search.engine.bind import (
    SlotBindingDraft,
    _SourceSlotBindOutput,
    _UniversalBindOutput,
    conflicting_sources,
    drafted_source,
)
from search.engine.bind_repair import (
    apply_declared_defaults,
    apply_shape_slot_defaults,
    defaulted_slot_notes,
    defaulted_slots_named,
    detect_degenerate_bindings,
    repair_draft_bindings,
    restore_starved_axis_binding,
)
from search.engine.config import settings
from search.engine.departures import measure_not_applied
from search.engine.extract import slot_default_answers
from search.engine.qualifier_match import (
    QualifierMatch,
    QualifierReadings,
    QualifierTie,
    _normalize,
)
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
from search.engine.region.named_source import with_named_source, without_dataset_drafts
from search.engine.region.near_neighbor import (
    _discriminating_token,
    _near_neighbor_check,
    _representative_score_is_weak,
)
from search.engine.region.phase_intake import _unresolved_entity_terminal
from search.engine.region.phase_rank_bind import (
    _admit_qualifier_matches_against_elected_shape,
    _bound_code_records,
    _force_qualifier_bindings,
    _log_admitted_qualifiers,
    _open_operation_slots,
    _resolve_query_qualifiers,
    _unbound_dataset,
    _unheld_bound_codes,
    _validate_requested_all_bindings,
    _validate_sole_option_bindings,
)
from search.engine.region.place_roles import bind_place_roles, capped_place_dcids
from search.engine.region.qualifier_readings import (
    bound_value_reads_measure,
    reading_for_shape,
    shape_carries_measure_match,
)
from search.engine.region.result import RegionResult
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import (
    PartNotServed,
    RoleKeys,
    bind_shared_values,
    canonical_pin_bindings,
    elect_measure_family,
    entity_role_keys,
    part_datasets,
    part_source,
    shape_datasets,
    shape_for_part,
    shape_invariants,
    shape_query,
)
from search.engine.shape import ShapeDraft
from search.heading.departure_fact import DepartureFact
from search.models import ResolvedQualifier, Warning


async def _phase_vet_election(state: ResolveState) -> RegionResult | None:
    """Select the bound shape, validate claims, and record ambiguity warnings.

    The dropped phrase an applied source slot consumed counts as applied
    (`_source_slot_phrases`), so it raises no dropped-qualifier warning.
    A source slot the model binds (`bind.drafted_source`) names or excludes datasets first
    (`ResolveState.named_dataset`, `ResolveState.excluded_datasets`), and the menu becomes
    the one shape candidacy offers a part naming or excluding them (`_apply_source_slot`):
    each capped shape is stamped again as such a part answers from it
    (`bound_values.shape_for_part`), and a shape left with no dataset leaves the menu, so
    the elected shape and every alternate answer as the parsed door's part would. The part
    refuses ``qualifier_not_bound`` when two source drafts contradict each other
    (`bind.conflicting_sources`), naming the tables they disagree on, when the tables the
    slot excludes leave the elected shape none (`_slot_shape`), or when the part's tables,
    less the excluded ones, hold no row with a code the bindings bind
    (`_unheld_bound_codes`). The elected shape then
    names the dataset the part is served from wherever its unnamed identity would resolve
    to another at the mint (`_name_served_dataset`), with or without a source slot, so
    every later read of the part's family names it. Binding repair names it again under
    the bindings it leaves, and refuses a part none of whose tables holds a row for them
    (`_settle_served_dataset`). Every one of these decisions
    reads the final bindings (`_settle_bindings`): the model's drafts, vetted, with the
    query's forced codes, settled against the shape the model bound from. This is the
    first point the elected shape and the slot are both known, and every reader of the
    part's dataset comes after it, so an opt-in table that candidacy withheld from an
    unnamed part is the part's dataset before its invariants, its source binding or its
    family are read.
    Every other drafted binding on the source is discarded. The dataset the elected shape
    names (`ShapeDraft.named_dataset`) becomes the part's named dataset and binds the source
    (`with_named_source`): the dataset the part or its source slot names, since shape
    candidacy left only that dataset's shapes to elect from, or the one opt-in dataset that
    alone may answer the shape. The near-neighbour gate supplements model alternates.
    Invalid ``requested_all`` claims are demoted, and on a shape whose option sets the index
    supplies a ``sole_option`` claim on a slot that offers two or more values leaves the
    slot unbound. Only validated phrases suppress defaults
    and dropped-qualifier warnings.
    An operation over a category slot opens that slot as a valid claim whether or not
    bind made one.
    """
    bind_result = state.bind_result
    state.shape_draft = state.capped_shapes[bind_result.chosen_shape_index]
    # The source slot's own draft is the one read of a drafted dataset. Every decision on
    # the part's dataset, and every refusal, reads the final bindings settled here: the
    # model's drafts, vetted, with the query's forced codes.
    conflicting = conflicting_sources(bind_result.bindings)
    if conflicting:
        return state.no_data(
            "qualifier_not_bound",
            unbound_qualifiers=tuple(_unbound_dataset(dataset) for dataset in conflicting),
        )
    named, excluded = drafted_source(bind_result.bindings)
    tie_departures, readings = _settle_bindings(state, state.shape_draft)
    if named is not None or excluded:
        refused = _apply_source_slot(state, named, excluded)
        if refused is not None:
            return refused
    # Binding repair can still change the codes, so a part no table serves under these
    # bindings is refused once the repair has run (`_settle_served_dataset`).
    with suppress(PartNotServed):
        _name_served_dataset(
            state, bind_result.bindings, lambda: _query_role_keys(state, state.shape_draft)
        )
    if named is not None or excluded:
        # The part's tables hold every code it binds, the model's and the forced query
        # matches alike, the excluded tables aside.
        unheld = _unheld_bound_codes(
            bind_result.bindings, state.shape_draft, excluded=state.excluded_datasets
        )
        if unheld:
            return state.no_data("qualifier_not_bound", unbound_qualifiers=unheld)
        state.applied_dropped_qualifier_phrases |= _source_slot_phrases(
            bind_result, state.dropped_qualifier_phrases
        )
    capped_shapes = state.capped_shapes
    shape_draft: ShapeDraft = state.shape_draft
    # Candidacy stamped every shape with the dataset it names, so the election decides the
    # part's, and every reader from here on reads the elected shape's.
    state.named_dataset = shape_draft.named_dataset
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

    for warning, fact in tie_departures:
        state.depart(warning, fact)

    # The dataset the part names.
    bind_result.bindings = with_named_source(bind_result.bindings, shape_draft)

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


def _settle_bindings(
    state: ResolveState, shape_draft: ShapeDraft
) -> tuple[list[tuple[Warning, DepartureFact]], dict[QualifierReadings, QualifierMatch | None]]:
    """Settle the elected shape's final bindings in `state.bind_result`, against the shape
    the model bound from. Return the departures of the tie phrases they leave unbound, for
    the caller to state in order, and the reading the shape keeps of each two-reading span
    (`reading_for_shape`), which admission read.

    The source drafts leave (the source slot is read before this), invalid
    ``requested_all`` claims are demoted, and on a shape whose option sets the index
    supplies a ``sole_option`` claim on a slot that offers two or more values leaves the
    slot unbound. An operation over a category slot opens that slot. Model tie choices are
    vetted against the model's own bindings, and the query's qualifier matches the elected
    shape admits are forced into the bindings (`_force_qualifier_bindings`).
    """
    bind_result = state.bind_result
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

    # Vet model tie choices against its original bindings before forced matches are added.
    # A row counts when a value binding names its code. One bound row satisfies the tie.
    # Zero or multiple rows leave it unresolved. Multiple bound properties reopen.
    bindings_by_property = {
        b.property_dcid: b for b in bind_result.bindings if b.property_dcid is not None
    }
    elected_tie_phrases: set[str] = set()
    departures: list[tuple[Warning, DepartureFact]] = []
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
        departures.append(
            (
                _qualifier_dropped_warning(phrase),
                measure_not_applied(
                    asked=tie.matched_text,
                    property_dcid=systems.pop() if len(systems) == 1 else None,
                ),
            )
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

    return departures, readings


def _qualifier_dropped_warning(phrase: str) -> Warning:
    """The `query_qualifier_dropped` warning for a qualifier phrase the answer does not
    apply."""
    return Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message=f'The requested qualifier "{phrase}" was not applied.',
    )


def _apply_source_slot(
    state: ResolveState, named: str | None, excluded: frozenset[str]
) -> RegionResult | None:
    """Make the part one naming `named`, or excluding `excluded`, from vetting on, or
    return the refusal of a slot that leaves the elected shape no dataset.

    The part's seam records the slot (`ResolveState.named_dataset`,
    `ResolveState.excluded_datasets`). Each capped shape becomes the one candidacy offers
    such a part (`_slot_shape`); a shape left with no dataset leaves the menu, and the
    elected index and the model's alternates follow the shapes that stay. An exclusion that
    leaves the elected shape no dataset refuses ``qualifier_not_bound``, naming each
    excluded table that publishes the shape under its declared label. The elected shape
    is named afterwards (`_name_served_dataset`).
    """
    if named is not None:
        state.named_dataset = named
    state.excluded_datasets |= excluded
    bind_result = state.bind_result
    chosen = bind_result.chosen_shape_index
    kept: dict[int, ShapeDraft] = {}
    for position, shape in enumerate(state.capped_shapes):
        stamped = _slot_shape(state, shape)
        if stamped is not None:
            kept[position] = stamped
    if chosen not in kept:
        shape = state.shape_draft
        unbound = (
            (named,)
            if named is not None
            else tuple(dataset for dataset in shape_datasets(shape) if dataset in excluded)
        )
        return state.no_data(
            "qualifier_not_bound",
            unbound_qualifiers=tuple(_unbound_dataset(dataset) for dataset in unbound),
        )
    renumbered = {position: number for number, position in enumerate(kept)}
    state.capped_shapes = list(kept.values())
    bind_result.chosen_shape_index = renumbered[chosen]
    if bind_result.also_plausible is not None:
        bind_result.also_plausible = [
            renumbered[i] for i in bind_result.also_plausible if i in renumbered
        ]
    state.shape_draft = state.capped_shapes[bind_result.chosen_shape_index]
    return None


def _slot_shape(state: ResolveState, shape: ShapeDraft) -> ShapeDraft | None:
    """`shape` as the part whose source slot named or excluded datasets answers from it,
    or `None` when the part may answer it from no dataset.

    A part naming a table keeps a shape that table publishes, stamped as naming it, as
    shape candidacy keeps it for a part naming the table at the seam. A part excluding
    tables keeps a shape the index does not carry, and a shape the index carries while a
    dataset it does not exclude may answer it, stamped as candidacy stamps it for a part
    excluding them at the seam: with the datasets that remain and the codes their rows
    hold, so no later binding takes a code only an excluded table holds.
    """
    index = get_index()
    publishers = shape_datasets(shape, index)
    if state.named_dataset is not None:
        if state.named_dataset not in publishers:
            return None
        return shape_for_part(shape, state.named_dataset, index)
    if publishers and not part_datasets(shape, None, index, excluded=state.excluded_datasets):
        return None
    return shape_for_part(shape, None, index, excluded=state.excluded_datasets)


def _name_served_dataset(
    state: ResolveState,
    bindings: Sequence[SlotBindingDraft],
    role_keys: Callable[[], RoleKeys],
) -> None:
    """Name the dataset the part is served from under `bindings` on the elected shape,
    where the shape names none and its unnamed identity would resolve to another dataset
    at the mint (`bound_values.part_source`): a part excluding the table that would serve,
    or a shape whose slot keys select another family than its own. `role_keys` gives the
    query's role keys, read where several tables could serve. Otherwise the shape stays as
    candidacy stamped it and the part mints the plain question's identity. An alternate is
    named the same way under its own bindings when it is walked
    (`candidates.walk_alternate`).

    Raises `bound_values.PartNotServed` when no table the part may answer the shape from,
    the one it names included, holds a row matching `bindings` together.
    """
    shape = state.shape_draft
    source = part_source(shape, bindings, role_keys, excluded=state.excluded_datasets)
    if source is None or source == shape.named_dataset:
        return
    named = shape_for_part(shape, source)
    state.capped_shapes[state.bind_result.chosen_shape_index] = named
    state.shape_draft = named


def _settle_served_dataset(state: ResolveState, role_keys: RoleKeys) -> RegionResult | None:
    """Settle the dataset the part is served from under its final bindings
    (`ResolveState.bindings`), or return the refusal of a part no table serves.

    Binding repair and the slot defaults can bind a code after vetting named the dataset,
    so the rule runs again here (`_name_served_dataset`), on the bindings every identity of
    the part is minted from. A shape it names takes the source binding and the canonical
    form of that dataset's invariants (`canonical_pin_bindings`). A part none of whose
    tables holds a row for the bindings together refuses ``qualifier_not_bound``, naming
    what it cannot bind (`_unserved_qualifiers`), so no table it excludes and no table of
    another family answers for it at any door. A part that names and excludes no table and
    binds no code has nothing to name, and refuses ``variable_not_resolved``.
    """
    unnamed = state.shape_draft.named_dataset is None
    try:
        _name_served_dataset(state, state.bindings, lambda: role_keys)
    except PartNotServed:
        unbound = _unserved_qualifiers(state)
        if not unbound:
            return _unresolved_entity_terminal(state) or state.no_data(
                "variable_not_resolved", llm_usage=state.var_usage
            )
        return state.no_data("qualifier_not_bound", unbound_qualifiers=unbound)
    shape = state.shape_draft
    if unnamed and shape.named_dataset is not None:
        state.named_dataset = shape.named_dataset
        state.bindings = canonical_pin_bindings(
            shape, with_named_source(state.bindings, shape), role_keys
        )
    return None


def _unserved_qualifiers(state: ResolveState) -> tuple[ResolvedQualifier, ...]:
    """What a part no table serves under its final bindings cannot bind, as unbound
    records: the dataset the elected shape names; else each dataset the part excludes that
    publishes the shape; else the codes the bindings bind (`_bound_code_records`)."""
    shape = state.shape_draft
    if shape.named_dataset is not None:
        return (_unbound_dataset(shape.named_dataset),)
    excluded = tuple(
        dataset for dataset in shape_datasets(shape) if dataset in state.excluded_datasets
    )
    if excluded:
        return tuple(_unbound_dataset(dataset) for dataset in excluded)
    return _bound_code_records(state.bindings)


def _source_slot_phrases(
    bind_result: _UniversalBindOutput, dropped_qualifier_phrases: list[str]
) -> frozenset[str]:
    """The normalised dropped phrase the applied source slot consumed
    (`_SourceSlotBindOutput.source_phrase`), when it is one of the part's dropped phrases
    verbatim; otherwise none, and that phrase's dropped-qualifier warning stays."""
    phrase = bind_result.source_phrase if isinstance(bind_result, _SourceSlotBindOutput) else None
    if phrase is None or phrase not in dropped_qualifier_phrases:
        return frozenset()
    return frozenset({_normalize(phrase)})


def _query_role_keys(state: ResolveState, shape: ShapeDraft) -> RoleKeys:
    """The role keys an identity minted for `shape` binds the query's places under and the
    axes it leaves open (`entity_role_keys`), read before place binding installs them: the
    binding `bind_place_roles` gives over `capped_place_dcids`. Places that refuse to bind
    give none, as the flagged identity of a role refusal binds no entity and leaves no axis
    open."""
    bound = bind_place_roles(state, shape, places=capped_place_dcids(state)[0])
    if isinstance(bound, str):
        return RoleKeys()
    return entity_role_keys(bound.resolved_entities, open_axes=bound.open_axes)


async def _phase_repair_bindings(state: ResolveState) -> RegionResult | None:
    """Repair bindings, apply measure elections and defaults, and refuse degenerate drafts.

    In order, detect degenerate values, repair drafts, restore starved axes, bind the
    values an unexported family shares, force open measure axes, apply declared defaults
    (none on a slot an invariant of the dataset the part names fixes), apply the shape's slot
    defaults, then put the bindings in canonical form: each invariant slot of the dataset
    the part names that they leave open bound to its value (`canonical_pin_bindings`).
    The dataset the part is served from is then settled under those bindings, and a part
    no table serves under them refuses ``qualifier_not_bound`` (`_settle_served_dataset`).
    Place binding reads the result.
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

    # A family the index does not export on a five-tuple it carries binds the values its
    # members share, so its identity resolves apart from the exported family.
    bindings, _shared_bindings = bind_shared_values(state.shape_draft, bindings)
    state.bindings = bindings

    # The role keys the minted identity binds its places under. With the bindings, they
    # elect the dataset the measure vote reads. A slot an invariant of the named dataset
    # fixes takes its value in canonical form, so no declared default binds it and no
    # default warning names it.
    role_keys = _query_role_keys(state, state.shape_draft)
    pinned = frozenset(shape_invariants(state.shape_draft))

    # Force measure axes left open after query matches and starved-axis restoration.
    # Run after draft repair, which judges realizability against a capped recall sample.
    # Keep dropped axes for slot metadata and the elected partition for the sentence event.
    # The final Spec derives its measure from served rows, which can differ from this vote.
    election = elect_measure_family(
        state.shape_draft,
        get_index(),
        get_measure_index().axes,
        shape_query(state.shape_draft, bindings, role_keys.bound, open_role_keys=role_keys.open),
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
        pinned_properties=pinned,
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
        # A default that declares a note states it whenever it fires.
        state.warnings.extend(
            Warning(code=SLOT_DEFAULTED, severity="warn", message=note)
            for note in defaulted_slot_notes(_defaulted_bindings, index=get_index())
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
    # query left open.
    bindings, _shape_defaulted_bindings = apply_shape_slot_defaults(bindings, state.shape_draft)
    if _shape_defaulted_bindings:
        state.bindings = bindings
        state.defaulted_properties |= frozenset(
            b.property_dcid for b in _shape_defaulted_bindings if b.property_dcid is not None
        )

    # Bind each invariant slot of the dataset the part names that the bindings leave open.
    # Every identity minted from here on, a refusal's flagged one included, is canonical,
    # and qualifier dispositions read the invariants as bound.
    state.bindings = canonical_pin_bindings(state.shape_draft, bindings, role_keys)

    # The bindings change no more, so the dataset the part is served from is settled on
    # them, and a part no table of its own serves is refused.
    refused = _settle_served_dataset(state, role_keys)
    if refused is not None:
        return refused

    # Resolve query qualifiers against the final bindings.
    state.query_qualifiers = _resolve_query_qualifiers(state)

    return None
