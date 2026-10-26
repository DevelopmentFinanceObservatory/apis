"""Settle the elected reading before place binding: vet the election, then repair and
default its bindings (`_phase_settle`)."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace

from code_index.scope_conventions import ODA_FLOW, ODA_FLOW_SYSTEM
from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureAxis
from search.engine.bind import SlotBindingDraft, validate_requested_all_bindings
from search.engine.bind_repair import (
    apply_broad_readings,
    apply_declared_defaults,
    apply_shape_slot_defaults,
    defaulted_slot_notes,
    defaulted_slots_named,
)
from search.engine.binders import StatedWordsRequest
from search.engine.binders.house_policy import slot_values
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read
from search.engine.envelope import _sum_usage
from search.engine.errors import LLMInfraError
from search.engine.extract import slot_default_answers
from search.engine.qualifier_match import (
    QualifierMatch,
    QualifierReadings,
    QualifierTie,
    _normalize,
)
from search.engine.region.codes import (
    RETRIEVAL_SCORE_WEAK,
    SLOT_DEFAULT_NOT_APPLIED,
    SLOT_DEFAULTED,
)
from search.engine.region.named_source import with_named_source, without_dataset_drafts
from search.engine.region.phase_rank_bind import (
    _admit_qualifier_matches_against_elected_shape,
    _bound_code_records,
    _coverage_reading,
    _coverage_records,
    _coverage_terms,
    _elected_index,
    _force_qualifier_bindings,
    _log_admitted_qualifiers,
    _menu_choices,
    _open_breakdown_slots,
    _open_operation_slots,
    _query_records,
    _resolve_query_qualifiers,
    _selected_members,
    _shape_qualifiers,
    _unchosen_matches,
    _unheld_bound_codes,
    _unrated_words,
    _unread_qualifier_phrases,
    _withheld_qualifiers,
    judge_value_requirements,
    operation_slot_phrases,
    served_member_names,
    value_requirements,
)
from search.engine.region.place_roles import (
    PlaceAssignment,
    bind_place_roles,
    holds_whole_total,
    whole_total_candidates,
)
from search.engine.region.qualifier_readings import (
    answer_reads_measure,
    bound_value_reads_measure,
    reading_for_shape,
)
from search.engine.region.requirements import (
    record_coverage,
    record_reading,
    settle_unanswered,
)
from search.engine.region.result import RegionResult
from search.engine.region.source_choice import source_record
from search.engine.region.state import (
    ResolveState,
    SlotDefaultDisclosure,
    for_shape,
    unjudged_requirement,
    with_requirement,
)
from search.engine.region.weak_score import _representative_score_is_weak
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    PartNotServed,
    RoleKeys,
    _collapse_bindings,
    bind_member_values,
    bind_shared_values,
    canonical_pin_bindings,
    elect_measure_family,
    identity_member_facts,
    named_scope,
    part_source,
    read_held_families,
    shape_datasets,
    shape_for_part,
    shape_invariants,
    shape_part_datasets,
    shape_query,
    sole_shared_value,
    without_values,
)
from search.engine.retrieval.member_facts import MemberFacts, asserted_constraint_properties
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.standard import (
    BreakdownDecision,
    Narrow,
    bind_named_member_values,
    bind_partition_values,
    breakdown_decision,
    family_totals,
    holds_broader_reading,
)
from search.engine.shape import ShapeDraft
from search.engine.written_phrase import written_phrase
from search.models import ResolvedQualifier, Warning
from search.scope.basis import (
    BASIS_RULE,
    CONVENTION_DATASETS,
    HEADLINE_OPTION,
    SERVED_DIFFERS_RULE,
    BasisElection,
    elect_basis,
    measure_of,
    oda_flows,
    part_shape,
    sector_systems,
    window_class,
)
from search.scope.sector import SECTOR_PROPERTIES, read_sector
from search.scope.state import AxisRecord
from search.scope.vocabulary import measure_basis_option_id

logger = logging.getLogger(__name__)


async def _phase_settle(state: ResolveState) -> RegionResult | None:
    """Settle the elected reading's bindings, or return the refusal of a part they cannot
    serve.

    Vetting runs first (`_vet_election`): source-slot settlement, unheld codes, undeclared
    qualifiers and tie departures. The domain rules then repair and default the bindings
    and settle the dataset the part is served from (`_repair_bindings`). Place binding
    reads the result. The held families of the bind menu, the elected shape's among them,
    are read first, off the event loop (`read_held_families`).
    """
    await read_held_families(state.capped_shapes)
    refused = _vet_election(state)
    if refused is not None:
        return refused
    return await _repair_bindings(state)


def _vet_election(state: ResolveState) -> RegionResult | None:
    """Select the bound shape, validate claims, and record ambiguity warnings.

    A part with a source scope refuses ``qualifier_not_bound`` when no row of its tables
    holds a code the bindings bind (`_unheld_bound_codes`), naming those codes and, since it
    judges no answer, each requirement only an answer can meet
    (`requirements.settle_unanswered`), as every refusal of this stage does. A code term
    the elected shape's own reading does not hold in full refuses once the bindings are
    final, with the elected shape as its near match (`phase_rank_bind._unavailable_qualifiers`).
    The elected shape then names the dataset the part is served from wherever its unnamed
    identity would resolve to another at the mint (`_name_served_dataset`), so every later
    read of the part's family names it, from the named places' role keys, which bind no
    total on a side the part leaves unnamed. Binding repair names it again under the
    bindings it leaves, and refuses a part none of whose tables holds a row for them
    (`_bind_query_places`). Every one of these decisions reads the final bindings
    (`_settle_bindings`): the model's drafts, vetted, with the query's forced codes, settled
    against the shape the model bound from. This is the first point the elected shape is
    known, and every reader of the part's dataset comes after it, so an opt-in table that
    candidacy withheld from an unnamed part is the part's dataset before its invariants, its
    source binding or its family are read.
    Every drafted binding on the source is discarded. The dataset the elected shape
    names (`ShapeDraft.named_dataset`) binds the source (`with_named_source`): the one
    dataset the part's scope leaves it, or the one opt-in dataset that alone may answer
    the shape. A weak retrieval score warns and sets no status (`status_rule`).
    Invalid ``requested_all`` claims are demoted. Only validated phrases suppress defaults
    and dropped-qualifier warnings.
    An operation over a category slot, or a breakdown phrase naming one, opens that slot
    as a valid claim whether or not bind made one.

    The elected shape is the highest-ranked of the menu shapes stated as the reading the
    model chose (`_elected_index`), so every later reader of the election sees one index.
    """
    bind_result = state.bind_result
    state.elected_index = _elected_index(state)
    state.shape_draft = state.capped_shapes[state.elected_index]
    # Every decision on the part's dataset, and every refusal, reads the final bindings
    # settled here: the model's drafts, vetted, with the query's forced codes.
    unresolved_ties, readings = _settle_bindings(state, state.shape_draft)
    # Binding repair can still change the codes, so a part no table serves under these
    # bindings is refused once the repair has run (`_bind_query_places`).
    with suppress(PartNotServed):
        _install_served_shape(
            state,
            _name_served_dataset(
                state,
                state.shape_draft,
                bind_result.bindings,
                lambda: _query_role_keys(state, state.shape_draft),
            ),
        )
    if state.scope is not None:
        # The part's tables hold every code it binds, the model's and the forced query
        # matches alike. A claim a recall sample alone backs binds none yet.
        unheld = _unheld_bound_codes(
            without_sampled_claims(bind_result.bindings), state.shape_draft
        )
        if unheld:
            return state.no_data(
                "qualifier_not_bound", unbound_qualifiers=(*unheld, *settle_unanswered(state))
            )
    shape_draft: ShapeDraft = state.shape_draft
    # A weak retrieval score is stated as a warning and sets no status. A topic expansion is
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

    # Each tie the bindings leave unresolved is a requirement every final answer is judged
    # against; it departs only if the final answer leaves it unapplied.
    for phrase, asked, prop in unresolved_ties:
        state.require_unapplied(phrase, asked=asked, property_dcid=prop, origin="tie")
    # A declined reading stands in to be flagged, and each code the question names that it
    # cannot bind is a requirement its flagged match leaves unmet, which its
    # `qualifier_not_bound` refusal names (`_declined_refusal`).
    if state.declined is not None:
        for match in state.declined.pinned_codes:
            state.require_unapplied(
                match.phrase,
                asked=match.matched_text,
                property_dcid=match.code_system,
                origin="bind",
            )

    # The dataset the part names.
    bind_result.bindings = with_named_source(bind_result.bindings, shape_draft)

    # Record measure values the elected shape carries and warn on unsupported or conflicting
    # values. Raw-text warnings own dropped phrases. This path warns only for its own phrases.
    # A conflict phrase names two values on one axis, so it keeps no single match. A
    # two-reading span is satisfied when the shape reads it as the measure value, and was
    # admitted above when the shape reads it as the code. It is unapplied when the shape
    # carries both readings or neither. A measure value that a bound constraint value reads
    # is satisfied too (`bound_value_reads_measure`). A measure value the shape does not
    # carry here is judged on each final answer (`ResolveState.require_measure`), since
    # binding repair can bind a value that reads it after this. One the raw text owns is
    # judged without departing, so the part admits it when the final answer meets it.
    unsatisfied: dict[str, QualifierMatch | None] = dict.fromkeys(
        state.measure_qualifier_conflict_phrases
    )
    satisfied_phrases: set[str] = set()
    held = without_sampled_claims(bind_result.bindings)
    for match in state.measure_query_matches:
        phrase = _normalize(match.matched_text)
        if answer_reads_measure(shape_draft, match, held):
            satisfied_phrases.add(phrase)
        else:
            unsatisfied.setdefault(phrase, match)
    for r, kept in readings.items():
        if kept is r.measure or (
            kept is None and bound_value_reads_measure(shape_draft, r.measure, held)
        ):
            satisfied_phrases.add(_normalize(r.matched_text))
        elif kept is None:
            unsatisfied.setdefault(_normalize(r.matched_text), r.measure)
    state.satisfied_measure_phrases = frozenset(satisfied_phrases)
    owned = state.variable_matched_qualifier_phrases
    for phrase in sorted(unsatisfied):
        match = unsatisfied[phrase]
        if match is not None and match in state.measure_query_matches:
            state.require_measure(phrase, match, discloses=phrase in owned)
            continue
        if phrase not in owned:
            continue
        state.require_unapplied(
            phrase,
            asked=match.matched_text if match is not None else phrase,
            property_dcid=match.code_system if match is not None else None,
            origin="bind",
        )

    return None


def _settle_bindings(
    state: ResolveState, shape_draft: ShapeDraft
) -> tuple[list[tuple[str, str, str | None]], dict[QualifierReadings, QualifierMatch | None]]:
    """Settle the elected shape's final bindings in `state.bind_result`, against the shape
    the model bound from. Return each tie phrase they leave unresolved, as its normalised
    phrase, its words and the one property its rows name (`None` for several), for the
    caller to record as a requirement in order (`ResolveState.require_unapplied`), and the
    reading the shape keeps of each two-reading span
    (`reading_for_shape`), which admission read.

    Each constraint slot of the shape the reply drafts nothing for gets an ``unbound``
    draft (`_with_every_slot_drafted`), so every later pass reads one draft per slot. The
    source drafts leave and invalid ``requested_all`` claims are demoted
    (`bind.validate_requested_all_bindings`). The binder applies it to its pick's drafts,
    and this pass keeps a claim only on a slot of the elected shape. Each slot
    whose taxonomy holds one value and that the reply left unbound is claimed as that
    ``sole_option`` value (`_claim_single_member_slots`). A
    ``sole_option`` claim on a value the shape's population shows the slot a reading
    without leaves the slot unbound (`_validate_sole_option_bindings`), as does a value the
    measure's headline series already means (`_unbind_implied_values`). A claim a recall
    sample alone backs stays bound, marked `SlotBindingDraft.sampled`, until the walk of the
    family confirms or refutes it (`phase_place_materialise._judge_sampled_claims`); every
    step after the implied values reads it open (`without_sampled_claims`). One value drafted
    on two properties no member holds together keeps the property a measure noun pins
    (`_keep_the_noun_pinned_property`). An operation
    over a category slot opens that slot, and so does a breakdown phrase naming it ("by
    wealth quintile", `_open_breakdown_slots`), ahead of `bind_partition_values` in
    `_settle_role_bindings`, which leaves an opened slot open. Model tie choices are
    vetted against the model's own bindings, and the query's qualifier matches the elected
    shape admits, with the codes the model chose from a phrase's candidate menu, are
    forced into the bindings (`_force_qualifier_bindings`).
    """
    bindings = _with_every_slot_drafted(
        without_dataset_drafts(state.bind_result.bindings), shape_draft
    )
    bindings, state.applied_dropped_qualifier_phrases = validate_requested_all_bindings(
        bindings,
        state.dropped_qualifier_phrases,
        slot_keys=shape_draft.slot_keys,
    )
    bindings = _settle_sole_options(bindings, shape_draft)
    bindings = _unbind_implied_values(bindings, shape_draft)
    # The rules from here on read each claim a recall sample alone backs open, and the claims
    # return to the slots they leave open (`with_sampled_claims`).
    claims = sampled_claim_drafts(bindings)
    bindings = without_values(bindings, claims)
    bindings = _keep_the_noun_pinned_property(bindings, shape_draft)
    bindings = _open_operation_slots(bindings, state.operations, shape_draft)
    bindings = _open_breakdown_slots(
        bindings,
        state.variable,
        state.dropped_qualifier_phrases,
        shape_draft,
        entities=state.entity_texts,
    )

    # The elected shape's own qualifiers: the matches its vocabulary reads and admits, and its
    # ties. Election vets only what that shape was shown, and reports its unheld ties, which
    # bind never offered, as dropped.
    own_matches, state.admitted_ties, unheld_ties = _shape_qualifiers(state, shape_draft)
    # Vet model tie choices against its original bindings before forced matches are added.
    # A row counts when a value binding names its code. One bound row satisfies the tie.
    # Zero or multiple rows leave it unresolved. Multiple bound properties reopen.
    bindings_by_property = {b.property_dcid: b for b in bindings if b.property_dcid is not None}
    elected_tie_phrases: set[str] = set()
    unresolved: list[tuple[str, str, str | None]] = []
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
        bindings = [
            SlotBindingDraft(axis=b.axis, property_dcid=b.property_dcid, kind="unbound")
            if b.property_dcid in properties_to_unbind
            else b
            for b in bindings
        ]
    state.elected_tie_phrases = frozenset(elected_tie_phrases)
    for tie in unheld_ties:
        if tie.phrase not in elected_tie_phrases:
            tie_dropped.setdefault(tie.phrase, tie)
    for phrase in sorted(tie_dropped.keys() & state.variable_matched_qualifier_phrases):
        tie = tie_dropped[phrase]
        systems = {row.code_system for row in tie.rows}
        unresolved.append((phrase, tie.matched_text, systems.pop() if len(systems) == 1 else None))

    # Keep only the qualifier matches the elected shape's own vocabulary reads and admits. A
    # two-reading span admits its code reading only when the elected shape carries that
    # reading alone.
    readings = {r: reading_for_shape(r, shape_draft) for r in state.query_qualifier_readings}
    # A code the model chose from a phrase's candidate menu that reads the phrase, a hit of
    # the menu or below one that nests under the terms inside the phrase on the code's
    # system (`_menu_choices`), binds in place of those terms. A phrase with no such
    # choice keeps its terms, which force as they would with no menu.
    terms = _admit_qualifier_matches_against_elected_shape(own_matches, shape_draft)
    if state.declined is not None:
        # A declined reading is flagged as it is: the codes it cannot bind stay off it.
        terms = tuple(m for m in terms if m not in state.declined.pinned_codes)
    choices = _menu_choices(state, shape_draft, bindings, terms=terms)
    state.menu_choices = choices
    admitted_qualifiers = _unchosen_matches(terms, choices)
    state.admitted_qualifiers = admitted_qualifiers
    state.text_recognized_properties = frozenset()
    state.grouped_properties = frozenset()
    if admitted_qualifiers or choices:
        bindings, forced = _force_qualifier_bindings(
            bindings,
            admitted_qualifiers,
            shape_draft,
            variable=state.variable,
            chosen=choices,
        )
        # A property whose admitted matches leave each other unresolved ("gross ODA, not
        # net") keeps the model's binding, so the text did not bind it and the measure
        # election may force it. Intersecting with the admitted matches' systems keeps a
        # property forced only by a menu choice at `binder` provenance, since the model
        # chose its code.
        state.text_recognized_properties = forced & {m.code_system for m in admitted_qualifiers}
        bindings = _settle_sector(state, bindings, admitted_qualifiers)
        _log_admitted_qualifiers(admitted_qualifiers)
    state.bind_result.bindings = with_sampled_claims(bindings, claims)

    return unresolved, readings


def _settle_sector(
    state: ResolveState,
    bindings: list[SlotBindingDraft],
    matches: tuple[QualifierMatch, ...],
) -> list[SlotBindingDraft]:
    """`bindings` with each text-recognised sector binding read as the sector axis reads it
    (`scope.sector.read_sector`), and the axis recorded on `ResolveState.scope_selection`.

    A sector concept the admitted `matches` name binds its declared grouping's codes, or
    the option an explicit or carried entry made for it chose, and its property joins
    `ResolveState.grouped_properties`: "health aid" binds CRS 120 and 130. A code the text
    named, "basic health" or "purpose code 120", keeps its binding and is recorded as an
    explicit entry of the words that named it."""
    settled: list[SlotBindingDraft] = []
    grouped: set[str] = set()
    for draft in bindings:
        prop = draft.property_dcid
        if (
            prop not in SECTOR_PROPERTIES
            or prop not in state.text_recognized_properties
            or not draft.value_dcids
        ):
            settled.append(draft)
            continue
        reading = read_sector(
            codes=draft.value_dcids,
            matches=[match for match in matches if match.code_system == prop],
            variable=state.variable,
            scope=state.scope_selection,
        )
        state.scope_selection.write(reading.record)
        if reading.grouped:
            grouped.add(prop)
            draft = SlotBindingDraft(
                axis=draft.axis,
                property_dcid=prop,
                kind="value" if len(reading.codes) == 1 else "set",
                value_dcids=list(reading.codes),
            )
        settled.append(draft)
    state.grouped_properties = frozenset(grouped)
    return settled


def _with_every_slot_drafted(
    bindings: list[SlotBindingDraft], shape_draft: ShapeDraft
) -> list[SlotBindingDraft]:
    """`bindings` with an ``unbound`` engine draft for each constraint slot of
    `shape_draft` they hold no draft for, in slot order after the reply's own.

    The bind prompt asks the reply for one binding per constraint slot, and a reply can
    leave one out. Every pass after it reads the slot through its draft: a single-member
    slot is claimed as its sole option (`_claim_single_member_slots`), and a declared
    default fires on an open slot's draft (`bind_repair.apply_declared_defaults`). A slot
    left out is open, so it reads as one drafted ``unbound``: "German aid to Kenya" binds
    DAC2a's ODA aid type by its default whether or not the reply lists the slot."""
    drafted = {draft.property_dcid for draft in bindings}
    missing = [
        SlotBindingDraft(axis=slot.axis, property_dcid=slot.property_dcid, kind="unbound")
        for slot in shape_draft.slot_keys
        if slot.property_dcid is not None and slot.property_dcid not in drafted
    ]
    return [*bindings, *missing] if missing else bindings


def _settle_sole_options(
    bindings: list[SlotBindingDraft], shape_draft: ShapeDraft
) -> list[SlotBindingDraft]:
    """`bindings` with each single-member slot the reply left unbound claimed
    (`_claim_single_member_slots`) and every ``sole_option`` claim validated
    (`_validate_sole_option_bindings`). A claim the validator demotes leaves its slot as
    the reply drafted it."""
    claimed = _claim_single_member_slots(bindings, shape_draft)
    validated = _validate_sole_option_bindings(claimed, shape_draft)
    return [
        draft if claim is not draft and kept.kind == "unbound" else kept
        for draft, claim, kept in zip(bindings, claimed, validated, strict=True)
    ]


def sampled_claim_drafts(bindings: Sequence[SlotBindingDraft]) -> dict[str, SlotBindingDraft]:
    """Each bound ``sole_option`` claim of the reply in `bindings` that rests on a recall
    sample (`SlotBindingDraft.sampled`), by its property."""
    return {
        b.property_dcid: b
        for b in bindings
        if b.sampled and b.kind == "value" and b.property_dcid is not None
    }


def sampled_claims(bindings: Sequence[SlotBindingDraft]) -> dict[str, str]:
    """The value each claim resting on a recall sample in `bindings`
    (`sampled_claim_drafts`) claims, by its property."""
    return {prop: b.value_dcids[0] for prop, b in sampled_claim_drafts(bindings).items()}


def without_sampled_claims(bindings: Sequence[SlotBindingDraft]) -> list[SlotBindingDraft]:
    """`bindings` with the slot of each claim resting on a recall sample
    (`sampled_claims`) left unbound: a sampled claim selects no member until the walk of
    the family confirms it.

    Every rule before the walk that asks whether a slot is bound reads the bindings so: a
    claim holds no slot against a rule that would bind, open or narrow it. Where such a
    rule writes the slot, its binding replaces the claim; the claim returns to each slot
    the rules leave open (`with_sampled_claims`), for the walk to judge."""
    sampled = sampled_claims(bindings)
    return without_values(bindings, sampled.keys()) if sampled else list(bindings)


def with_sampled_claims(
    bindings: Sequence[SlotBindingDraft], claims: Mapping[str, SlotBindingDraft]
) -> list[SlotBindingDraft]:
    """`bindings` with each of `claims` (`sampled_claim_drafts`) back on its slot where the
    rules that read it open (`without_sampled_claims`) left the slot unbound and asked for
    no more of it. A slot a rule bound keeps the rule's binding, and one opened as
    ``requested_all`` stays open."""
    return [
        claims[prop]
        if (prop := b.property_dcid) is not None
        and prop in claims
        and b.kind == "unbound"
        and b.evidence != "requested_all"
        else b
        for b in bindings
    ]


def _claim_single_member_slots(
    bindings: list[SlotBindingDraft], shape_draft: ShapeDraft
) -> list[SlotBindingDraft]:
    """`bindings` with each ``unbound`` draft on a slot whose taxonomy holds one value
    (`ShapeDraft.slot_taxonomy`) drafted as that value with evidence ``sole_option``.

    The bind prompt asks the reply to bind a single-member slot that way, and a reply can
    leave it unbound. The recorded reply to "infant mortality rate" leaves ``age`` unbound
    where the taxonomy holds only ``Years0``. The claim is the one the prompt asks of the
    reply, so it is stamped as the reply's (`SlotBindingDraft.drafted_by`) and every reader
    of the reply's claims judges it. A taxonomy read from a recall sample can show one value
    the family's total does not hold (a population shape whose sample holds one gender).
    `_validate_sole_option_bindings` refutes the claim where the sample shows it, and
    otherwise marks it `SlotBindingDraft.sampled`, so the walk of the family reads the slot
    open and refutes it on the total (`phase_place_materialise._judge_sampled_claims`). A
    ``requested_all`` draft asks for the slot in full and stays."""
    taxonomy = shape_draft.slot_taxonomy or {}
    claimed: list[SlotBindingDraft] = []
    for draft in bindings:
        values = taxonomy.get(f"{draft.axis}:{draft.property_dcid}", [])
        if (
            draft.kind == "unbound"
            and draft.evidence != "requested_all"
            and draft.property_dcid is not None
            and len(values) == 1
        ):
            draft = draft.model_copy(
                update={
                    "kind": "value",
                    "value_dcids": list(values),
                    "evidence": "sole_option",
                    "drafted_by": "reply",
                }
            )
        claimed.append(draft)
    return claimed


def _validate_sole_option_bindings(
    bindings: list[SlotBindingDraft], shape_draft: ShapeDraft
) -> list[SlotBindingDraft]:
    """Validate model ``sole_option`` claims against the shape's population.

    A ``sole_option`` claim says the slot forces one value and the phrase says nothing
    about it. The population is the family the shape's identity denotes
    (`identity_member_facts`): the rows the index carries for the dataset the part names,
    or for every dataset of the family when it names none, else the members of the family
    the catalogue holds for it beside its recalled ones, else its recalled members alone.
    A part excluding a dataset names none, so it mints the identity of the whole family,
    and a row of the excluded dataset holding another value keeps the slot open. The phrase did not name the value and a slot that
    the population leaves a reading without it does not force it, so no binding rule
    produced the draft and the slot is left unbound.

    On the index's family or a held family, which are complete, the slot forces a value
    only when every member asserts the property with that one value (`sole_shared_value`,
    the rule the engine binds a shared value by). A member asserting another value or
    none fails the claim: "life expectancy" offers ``Resident`` as the one value of
    ``residentStatus``, but members of the held family hold no resident status.

    A recalled sample is part of the family, so what it shows holds for the family too,
    but a member missing from it may assert the value. There a claim fails when a member
    asserts another value, when a total of the sample asserts none, or when a member
    asserting none states nothing a member asserting the value does not (`_sample_forces`).
    The last holds where the sample has no total: "net transfers on external debt" beside
    "net flows on external debt" splits on what the flow is net of, and the member stating
    no maturity band is the reading without one on its side of the split.
    The IDS "debt service" shape is such a sample, since the catalogue holds its family
    only beside the CRS rows of the same five-tuple: its members offer one value each of
    ``idsConcessionality``, ``includesIMFCredit`` and ``maturityBand``, but its
    by-creditor member, which asserts only the flow type every member shares, is the
    total and asserts none of them. A member that asserts none but narrows on another
    property is no reading without the value: "infant mortality rate" binds ``age=Years0``
    beside members that state an age group classification instead. A claim the sample
    does not refute is kept as `SlotBindingDraft.sampled`: a member outside the sample may
    still lack the value, and the walk of the family, which reads the slot open
    (`without_sampled_claims`), settles it (`phase_place_materialise._judge_sampled_claims`).
    A shape with no members passes every claim through.
    """
    population, index_sourced = identity_member_facts(shape_draft)
    if not population:
        return bindings
    complete = index_sourced or shape_draft.held_family is not None
    constraints = [member.read_constraints() for member in population.values()]
    # Other claims are under validation themselves, so only the rest state values.
    stated = _collapse_bindings([b for b in bindings if b.evidence != "sole_option"])
    validated: list[SlotBindingDraft] = []
    for binding in bindings:
        prop = binding.property_dcid
        if binding.kind == "value" and binding.evidence == "sole_option" and prop is not None:
            value = binding.value_dcids[0] if len(binding.value_dcids) == 1 else None
            others = {p: values for p, values in stated.items() if p != prop}
            if value is None or not (
                sole_shared_value(constraints, prop) == value
                if complete
                else _sample_forces(population, constraints, prop, value, others)
            ):
                binding = binding.model_copy(
                    update={"kind": "unbound", "value_dcids": [], "evidence": "mentioned"}
                )
            elif not complete:
                binding = binding.model_copy(update={"sampled": True})
        validated.append(binding)
    return validated


def _unbind_implied_values(
    bindings: list[SlotBindingDraft], shape_draft: ShapeDraft
) -> list[SlotBindingDraft]:
    """Unbind each value the bind reply drafted that the shape's measure already implies.

    The code index declares the value an open slot of the measure means
    (`CodeIndex.implied_values_of`): life expectancy is life expectancy at age 0. The
    headline series states nothing on the property, so a member stating the value is
    never the family's total and the binding elects a narrower series in its place. The
    binding is released where the family has totals under the other bindings
    (`family_totals`) and none of them asserts the property, so the total answers. A
    family with no total keeps the value, as does one whose totals assert the property or
    have unknown constraints: there the value may separate readings the family holds. The
    population is the one `_validate_sole_option_bindings` reads (`identity_member_facts`).

    The draft's evidence does not matter: the recorded bind reply to "life expectancy"
    drafts ``age=Years0`` as the slot's ``sole_option``, and a reply naming the age as
    ``mentioned`` is released the same way.
    """
    implied = get_index().implied_values_of(shape_draft.pop_type_dcid, shape_draft.meas_prop_dcid)
    if not implied:
        return bindings
    population, _ = identity_member_facts(shape_draft)
    settled = list(bindings)
    for i, binding in enumerate(settled):
        prop = binding.property_dcid
        if (
            binding.kind != "value"
            or binding.drafted_by != "reply"
            or prop is None
            or binding.value_dcids != [implied.get(prop)]
        ):
            continue
        totals = family_totals(population, _collapse_bindings(settled[:i] + settled[i + 1 :]))
        asserted = [asserted_constraint_properties(population[sv]) for sv in totals]
        if totals and all(props is not None and prop not in props for props in asserted):
            settled[i] = binding.model_copy(
                update={"kind": "unbound", "value_dcids": [], "evidence": "mentioned"}
            )
    return settled


def _keep_the_noun_pinned_property(
    bindings: list[SlotBindingDraft], shape_draft: ShapeDraft
) -> list[SlotBindingDraft]:
    """Keep one of two reply-drafted ``mentioned`` bindings of one value on two properties
    that no member of the shape's population holds together, and leave the other unbound.

    "External debt service" names ``ServicingPayment`` as the flow type, and the debt
    statistics also hold it as what net transfers are net of. The bind reply to an external
    debt service question drafts ``financialFlowType=ServicingPayment`` and
    ``netOf=ServicingPayment``, both ``mentioned``. No member holds both, and the answer
    resolved without the flow type refuses the term (`unrealisable.unbind_unrealisable`).
    A net transfers question drafts ``netOf`` alone, beside the flow type ``Disbursement``,
    and stands. The measure nouns of the code index (`MeasureIndex.nouns`) decide which
    property the value is read on: the one noun of the shape's measure that names the value
    on one of the two properties and whose every other constraint pair the drafts leave
    open or bind to its value. "Net transfers on external debt" names ``netOf`` with the
    flow type ``Disbursement``, which the draft contradicts, so only "external debt
    service" survives and the flow type keeps the value. With no such noun, or several,
    the bindings stand.
    """
    properties_by_value: dict[str, list[str]] = {}
    for b in bindings:
        if (
            b.kind == "value"
            and b.evidence == "mentioned"
            and b.drafted_by == "reply"
            and b.property_dcid is not None
            and len(b.value_dcids) == 1
        ):
            properties_by_value.setdefault(b.value_dcids[0], []).append(b.property_dcid)
    population, _ = identity_member_facts(shape_draft)
    stated = _collapse_bindings(bindings)
    released: set[str] = set()
    for value, drafted in properties_by_value.items():
        props = set(drafted)
        if len(drafted) != 2 or len(props) != 2 or not population:
            continue
        if any(
            all(member.read_constraints().get(prop) == value for prop in props)
            for member in population.values()
        ):
            continue
        pinned = _noun_pinned_properties(shape_draft, stated, props, value)
        if len(pinned) == 1:
            released |= props - pinned
    if not released:
        return bindings
    return [
        b.model_copy(update={"kind": "unbound", "value_dcids": []})
        if b.property_dcid in released
        else b
        for b in bindings
    ]


def _noun_pinned_properties(
    shape_draft: ShapeDraft, stated: Mapping[str, list[str]], props: set[str], value: str
) -> set[str]:
    """The properties among `props` a measure noun of the shape's measure names `value` on,
    of the nouns whose other constraint pairs `stated` leaves open or binds to the noun's
    value. A noun that names `value` on several of `props` pins none of them."""
    shape_axes = {
        "populationType": shape_draft.pop_type_dcid,
        "measuredProperty": shape_draft.meas_prop_dcid,
        "statType": shape_draft.stat_type_dcid,
        "measurementQualifier": shape_draft.meas_qual_dcid,
        "measurementDenominator": shape_draft.meas_denom_dcid,
    }
    pinned: set[str] = set()
    for noun in get_measure_index().nouns:
        named = {prop for prop, v in noun.axis_values if prop in props and v == value}
        if len(named) != 1:
            continue
        if all(
            shape_axes[axis] == v
            if axis in shape_axes
            else axis in named or axis not in stated or v in stated[axis]
            for axis, v in noun.axis_values
        ):
            pinned |= named
    return pinned


def _sample_forces(
    population: Mapping[str, MemberFacts],
    constraints: Sequence[Mapping[str, str]],
    prop: str,
    value: str,
    stated_values: Mapping[str, list[str]],
) -> bool:
    """Whether a recalled sample leaves `prop` no reading but `value`: no member asserts
    another value on it (`constraints` are the members' `read_constraints`), no member of
    the part asserting none is a reading without it, and every total of the sample under
    the part's other bindings (`family_totals`), the member a slot left open serves,
    asserts `value`.

    A member of the part asserting nothing on `prop` is one of three readings beside the
    members asserting `value`. Its constraints can be a subset of an asserting member's,
    so it is the broader reading an open `prop` serves (`holds_broader_reading`): in a
    sample that splits on another open slot, which holds no total, it is the broadest
    member of its side. It can differ from an asserting member on a property both state,
    so the two are siblings across a split and it is the reading of the part without
    `value` (`_splits_without_value`): gross domestic product beside the gross value added
    of agriculture. Or it can narrow on properties every asserting member leaves unstated,
    which forces nothing against the claim. The first two refute it; the third does not.

    The rule governs only answers the walk of the family cannot judge: where the walk
    reads the slot open, it confirms or refutes the claim on the whole family
    (`phase_place_materialise._judge_sampled_claims`)."""
    if any(member.get(prop) not in (None, value) for member in constraints):
        return False
    if holds_broader_reading(population, stated_values, prop, value):
        return False
    if _splits_without_value(constraints, prop, value, stated_values):
        return False
    return all(
        population[sv].read_constraints().get(prop) == value
        for sv in family_totals(population, stated_values)
    )


def _splits_without_value(
    constraints: Sequence[Mapping[str, str]],
    prop: str,
    value: str,
    stated_values: Mapping[str, list[str]],
) -> bool:
    """Whether a member of the part (the members stating one of the `stated_values` on
    each bound property) asserting nothing on `prop` asserts, on another property, a value
    other than the one a member asserting `value` there asserts. The two are the sides of
    a slot the part leaves open, and the side without `value` is a reading of the part
    without it: "GDP" recalls gross domestic product, which states no economic sector,
    beside the gross value added of agriculture, so the sector is not forced.
    `holds_broader_reading` misses such a pair, since neither member's constraints are a
    subset of the other's. A member asserting none that narrows on a property the members
    asserting `value` leave unstated splits nowhere with them: the age group
    classification members beside the age of zero of "infant mortality rate"."""
    part = [
        member
        for member in constraints
        if all(member.get(bound) in values for bound, values in stated_values.items())
    ]
    asserting = [member for member in part if member.get(prop) == value]
    return any(
        other.get(other_prop) not in (None, other_value)
        for other in part
        if prop not in other
        for member in asserting
        for other_prop, other_value in member.items()
        if other_prop != prop
    )


def _name_served_dataset(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    role_keys: Callable[[], RoleKeys],
) -> ShapeDraft:
    """`shape`, the elected shape, naming the dataset the part is served from under
    `bindings`, where it names none and its unnamed identity would resolve to another
    dataset at the mint (`bound_values.part_source`): a part excluding the table that would
    serve, or a shape whose slot keys select another family than its own. `role_keys` gives
    the query's role keys, read where several tables could serve. Otherwise `shape` as
    candidacy stamped it, and the part mints the plain question's identity. An alternate is
    named the same way under its own bindings when it is walked (`candidates.walk_alternate`).

    Raises `bound_values.PartNotServed` when no table the part may answer the shape from,
    the one it names included, holds a row matching `bindings` together.
    """
    source = part_source(shape, bindings, role_keys, scope=state.scope)
    if source is None or source == shape.named_dataset:
        return shape
    return shape_for_part(shape, named_scope(source))


def _install_served_shape(state: ResolveState, shape: ShapeDraft) -> None:
    """Make `shape` the elected shape, in play and on the menu at its index, where it is
    another shape than the one in play."""
    if shape is state.shape_draft:
        return
    state.capped_shapes[state.elected_index] = shape
    state.shape_draft = shape


def _served(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: list[SlotBindingDraft],
    role_keys: RoleKeys,
) -> tuple[ShapeDraft, list[SlotBindingDraft]]:
    """The elected shape naming the dataset the part is served from under `bindings`
    (`_name_served_dataset`), with `bindings` in that dataset's canonical form where the
    shape named none before (`canonical_pin_bindings`). Raises `bound_values.PartNotServed`
    as `_name_served_dataset` does."""
    served = _name_served_dataset(state, shape, bindings, lambda: role_keys)
    if shape.named_dataset is None and served.named_dataset is not None:
        bindings = canonical_pin_bindings(served, with_named_source(bindings, served), role_keys)
    return served, bindings


def _narrow_to_breakdown(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], role_keys: RoleKeys
) -> tuple[BreakdownDecision, ShapeDraft]:
    """Decide which what and how slots the identity of the elected `shape` carries under
    its settled `bindings` and the query's `role_keys` (`standard.breakdown_decision`), and
    return the decision with the shape it leaves.

    A `Narrow` decision leaves every other what or how slot out of the shape's slot keys
    (`ShapeDraft.omitted_slots`), so every reader after it mints, walks, admits and reads
    from the slots the decision keeps, as a resubmit of the identity does from its posted
    slots (`promote_shape._phase_build_shape`): a mint of the definition serves exactly the
    members the answer reads. Any other decision, a family with a total among them, leaves
    `shape` as it stands.
    """
    decision = breakdown_decision(shape, bindings, role_keys.bound)
    if not isinstance(decision, Narrow):
        return decision, shape
    return decision, narrowed_to_breakdown(shape, decision.properties)


def narrowed_to_breakdown(shape: ShapeDraft, properties: frozenset[str]) -> ShapeDraft:
    """`shape` with every what or how slot outside `properties` left out of its slot keys
    and recorded as omitted (`ShapeDraft.omitted_slots`), or `shape` itself when it has no
    such slot."""
    kept = tuple(
        slot
        for slot in shape.slot_keys
        if slot.axis not in ("what", "how")
        or slot.property_dcid is None
        or slot.property_dcid in properties
    )
    if len(kept) == len(shape.slot_keys):
        return shape
    return replace(
        shape,
        slot_keys=kept,
        omitted_slots=frozenset(
            slot.property_dcid
            for slot in shape.slot_keys
            if slot not in kept and slot.property_dcid is not None
        ),
    )


async def _settle_served_dataset(state: ResolveState, role_keys: RoleKeys) -> RegionResult | None:
    """Settle the dataset the part is served from under its final bindings
    (`ResolveState.bindings`), or return the refusal of a part no table serves.

    An unbinding at materialise can leave bindings that select another table than the one
    binding repair named, so the rule runs again there (`_served`), on the bindings every
    identity of the part is minted from. A part none of whose tables holds a row for the
    bindings together refuses (`_unserved_refusal`), with stated-word coverage judged first
    (`judge_coverage`). No read has served a member, so the judgement reads the shape's
    named members, and each term the judgement records is named or departs.
    """
    try:
        shape, state.bindings = _served(state, state.shape_draft, state.bindings, role_keys)
    except PartNotServed:
        await judge_coverage(state)
        return _unserved_refusal(state)
    _install_served_shape(state, shape)
    return None


def _unserved_refusal(state: ResolveState) -> RegionResult:
    """The refusal of a part none of whose tables holds a row for its bindings together:
    ``qualifier_not_bound`` naming the sources it cannot bind (`_unserved_sources`). A part
    under a scope that names sources names those alone, and any other part names the codes
    its bindings bind (`_bound_code_records`), so no table it excludes and no table of
    another family answers for it at any door. Each refusal also names each requirement
    only an answer can meet (`requirements.settle_unanswered`), such as a value bind drafted
    on a slot the elected shape lacks. A part that names and excludes no table and has none
    of these to name refuses ``variable_not_resolved``."""
    sources = _unserved_sources(state)
    terms = settle_unanswered(state)
    if sources or (state.scope is not None and state.scope.named):
        return state.no_data(
            "qualifier_not_bound", unbound_sources=sources, unbound_qualifiers=terms
        )
    unbound = (*_bound_code_records(state.bindings), *terms)
    if not unbound:
        return state.no_data("variable_not_resolved", llm_usage=state.var_usage)
    return state.no_data("qualifier_not_bound", unbound_qualifiers=unbound)


def _unserved_sources(state: ResolveState) -> tuple[ResolvedQualifier, ...]:
    """The sources a part no table serves under its final bindings cannot bind, as unbound
    records: the dataset the elected shape names; else, under a scope that excludes, each
    dataset publishing the shape that the scope leaves out; else none. A part under a scope
    that names its sources refuses naming those (`conjoin._unbound_terms`), and a part
    under no scope names the codes its bindings bind (`_bound_code_records`)."""
    shape = state.shape_draft
    if shape.named_dataset is not None:
        return (source_record(shape.named_dataset),)
    scope = state.scope
    if scope is None or scope.named:
        return ()
    return tuple(
        source_record(dataset) for dataset in shape_datasets(shape) if dataset not in scope.allowed
    )


def _query_role_keys(state: ResolveState, shape: ShapeDraft) -> RoleKeys:
    """The role keys an identity minted for `shape` binds the query's named places under and
    the axes it leaves open, read before binding repair settles the places, from the
    assignment `bind_place_roles` gives (`PlaceAssignment.role_keys`), which binds no total on
    a side the part leaves unnamed. Places that refuse to bind give none, as the flagged
    identity of a role refusal binds no entity and leaves no axis open."""
    bound = bind_place_roles(state, shape)
    return RoleKeys() if isinstance(bound, str) else bound.role_keys


async def _repair_bindings(state: ResolveState) -> RegionResult | None:
    """Repair bindings and apply measure elections and defaults.

    In order, bind the values an unexported family shares, bind the values a series code's
    StatVars share, apply declared defaults (none on a slot an invariant of the dataset the
    part names fixes), and read each bound value the code index reads more broadly as that
    value (`apply_broad_readings`). None of these reads the query's role keys. The query's
    places then bind to the elected shape, and the steps that read their role keys settle
    the bindings under them (`_bind_query_places`): the measure election, the shape's slot
    defaults, the named-member and partition binds, the canonical pins, the slots the
    identity carries (`standard.breakdown_decision`), and the dataset the part is served
    from, where a part no table serves refuses ``qualifier_not_bound``. The query records
    the coverage rule reads as recorded (`_query_records`) are then kept, so an unbinding
    reads the terms that name its values (`unrealisable._asked_terms`). Stated-word
    coverage is judged once the pass reaches its answer (`judge_coverage`).

    Every rule up to the slots the identity carries reads each claim a recall sample alone
    backs open (`without_sampled_claims`), so a default, a shared value, the measure election or a
    pin binds its slot over the claim, and its binding is the one the walk reads. "Birth
    rate" recalled as the crude birth rate alone claims ``medicalStatus=LiveBirth`` on that
    sample; the live-birth slot default binds it, where the claim read as bound kept the
    default off and the walk, finding the births total, refuted the claim and answered
    total births. The claims return to the slots the rules leave open
    (`with_sampled_claims`) once the slots the identity carries are decided, ahead of the
    dataset the part is served from, and `ResolveState.bindings` holds them throughout.
    """
    claims = sampled_claim_drafts(state.bind_result.bindings)
    state.bindings = list(state.bind_result.bindings)
    bindings = without_values(state.bindings, claims)

    # A family the index does not export on a five-tuple it carries binds the values its
    # members share, so its identity resolves apart from the exported family.
    bindings, _shared_bindings = bind_shared_values(state.shape_draft, bindings)
    state.bindings = with_sampled_claims(bindings, claims)

    # A part's series code names its StatVars, so each value they all share binds, and the
    # identity minted from the bindings elects those StatVars alone. A slot the query asked
    # to see in full that they hold one value on is a part the code cannot answer.
    if state.source_code is not None:
        bindings, _asked_in_full = bind_member_values(
            state.shape_draft, bindings, state.source_code.dcids
        )
        state.bindings = with_sampled_claims(bindings, claims)
        if _asked_in_full:
            return state.no_data("variable_not_resolved", llm_usage=state.var_usage)

    # A slot an invariant of the named dataset fixes takes its value in canonical form, so
    # no declared default binds it and no default warning names it.
    pinned = frozenset(shape_invariants(state.shape_draft))

    # Apply a declared default when the variable, raw query, or bindings trigger it. Raw
    # query text is available only for a single-variable request and participates in the
    # response cache key. Engine-authored defaults add no warning. Defaults bind before the
    # measure vote, so a default code published at one basis elects that basis.
    bindings, _defaulted_bindings = apply_declared_defaults(
        bindings,
        state.shape_draft,
        index=get_index(),
        variable=state.variable,
        query_text=state.query_text,
        query_qualifier_matches=for_shape(state.query_qualifier_matches, state.shape_draft),
        query_qualifier_ties=for_shape(state.query_qualifier_ties, state.shape_draft),
        engine_authored_properties=state.measure_elected_properties,
        pinned_properties=pinned,
    )
    if _defaulted_bindings:
        state.bindings = with_sampled_claims(bindings, claims)
        state.defaulted_properties = frozenset(
            b.property_dcid for b in _defaulted_bindings if b.property_dcid is not None
        )
        # A default on a slot the query names by its plain name ("top 5 aid flows") warns,
        # and answers each dropped phrase that asked for no more than that slot. A default
        # that declares a note states it whenever it fires. Each default's statements are
        # kept under its property, so unbinding it takes them back (`ResolveState.forget_bound`).
        texts = [
            text
            for text in (
                state.variable,
                state.query_text,
                *state.dropped_qualifier_phrases,
                *(operation.phrase for operation in state.operations),
            )
            if text is not None
        ]
        defaulted = [
            (prop, b) for b in _defaulted_bindings if (prop := b.property_dcid) is not None
        ]
        named = {
            prop: defaulted_slots_named([b], index=get_index(), texts=texts)
            for prop, b in defaulted
        }
        named_warnings = {
            prop: tuple(
                Warning(
                    code=SLOT_DEFAULTED,
                    severity="warn",
                    message=(
                        f"No {plain_name} was named, so the answer covers {label}, "
                        f"the default {plain_name}."
                    ),
                )
                for plain_name, label in pairs
            )
            for prop, pairs in named.items()
        }
        note_warnings = {
            prop: tuple(
                Warning(code=SLOT_DEFAULTED, severity="warn", message=note)
                for note in defaulted_slot_notes([b], index=get_index())
            )
            for prop, b in defaulted
        }
        state.warnings.extend(w for found in named_warnings.values() for w in found)
        state.warnings.extend(w for found in note_warnings.values() for w in found)
        state.slot_defaults = {
            prop: SlotDefaultDisclosure(
                warnings=named_warnings[prop] + note_warnings[prop],
                phrases=frozenset(
                    _normalize(phrase)
                    for phrase in state.dropped_qualifier_phrases
                    if any(
                        slot_default_answers(phrase, plain_name=plain_name, variable=state.variable)
                        for plain_name, _label in pairs
                    )
                ),
            )
            for prop, pairs in named.items()
        }

    # A bound value the code index reads more broadly unless the query names a value on
    # another slot binds the broader value ("publicly guaranteed" reads public and publicly
    # guaranteed debt), before the measure vote reads the bindings. Its note and the reading
    # itself, which applies the narrow value's words, are kept under its property, so
    # unbinding the slot takes both back.
    bindings, broad_readings = apply_broad_readings(bindings, state.shape_draft, index=get_index())
    if broad_readings:
        state.bindings = with_sampled_claims(bindings, claims)
        for reading in broad_readings:
            warning = Warning(code=SLOT_DEFAULTED, severity="warn", message=reading.note)
            state.warnings.append(warning)
            state.slot_defaults[reading.property_dcid] = SlotDefaultDisclosure(
                warnings=(warning,), reading=reading
            )

    # The query's places bind to the elected shape once, here, with a whole total where the
    # shape offers one and the bindings settled under its role keys hold its series
    # (`_bind_query_places`). Place binding reads the assignment.
    refused = _bind_query_places(state, bindings, claims)
    if refused is not None:
        return refused

    # The records no rating reads stand until the coverage judgement replaces them
    # (`judge_coverage`), so an unbinding before it reads the terms naming its values.
    record_reading(state, _query_records(state))
    return None


async def judge_coverage(state: ResolveState) -> None:
    """Judge stated-word coverage on the elected shape over the final bindings, once per
    pass, before its answer is first judged. That is once the read serves members, or on a
    refusal returned before it (`pipeline._run_phases`, `phase_shared._elected_flagged_match`).
    The coverage rule reads the members the read served (`served_member_names`), so the
    lexical readers, the binder's rating and the answer describe one list.

    The binder rates the words the rule leaves unrated (`_rate_unread_words`) and the query
    records over both replace those binding repair kept (`requirements.record_reading`).
    In "refuse" coverage mode a word of the variable no reader covers is a record and the
    part refuses on it. In log mode the records are kept for the line the pass logs when it
    ends (`log_would_refuse`), and a word of a qualifier term the elected shape's own
    reading leaves unread is a term no answer is judged on (`_unread_qualifier_phrases`).
    In refuse mode a record already covers that word. In log mode a term of a part naming
    no dataset that names only codes no shape can bind is withheld (`_withheld_qualifiers`),
    kept here alike, and is a term no answer is judged on, whatever its property. Each value
    the request names by its graph label is recorded once as a requirement
    (`value_requirements`) in refuse mode, and in log mode the elected answer's unmet ones
    are kept by the same rule (`judge_value_requirements`). Each value the bind drafted on
    a slot the elected shape lacks is a requirement judged by the same rule, which bind
    recorded (`phase_rank_bind._stray_requirements`), so its query record adds none. Every
    requirement departs only if the final answer leaves it unapplied
    (`requirements.with_requirement_departures`). Each candidate the part offers beside or
    in place of the elected answer is judged by the same rule on its own read
    (`judge_answer_coverage`)."""
    if state.coverage_judged:
        return
    state.coverage_judged = True
    await _rate_unread_words(state)
    record_reading(state, _resolve_query_qualifiers(state))
    withheld = _withheld_qualifiers(state)
    log_mode = settings().qualifier_coverage == "log"
    recorded = (*state.query_qualifiers, *withheld)
    coverage = _coverage_reading(state, _coverage_terms(state)) if log_mode else None
    values = value_requirements(state, (*recorded, *(coverage.records if coverage else ())))
    if coverage is not None:
        unmet = judge_value_requirements(values, state.shape_draft, state.bindings, state)
        state.would_refuse = (
            *withheld,
            *coverage.records,
            *(requirement.term for requirement in unmet),
        )
        state.word_verdicts = coverage.verdicts
    else:
        for requirement in values:
            state.requirements = with_requirement(state.requirements, requirement)
    index = get_index()
    for record in withheld:
        systems = {
            row.code_system
            for ref in record.named
            if (row := index.code_by_dcid(ref.ref)) is not None
        }
        state.requirements = with_requirement(
            state.requirements,
            unjudged_requirement(
                _normalize(record.text),
                asked=record.text,
                property_dcid=systems.pop() if len(systems) == 1 else None,
                origin="bind",
            ),
        )
    for phrase in _unread_qualifier_phrases(state, withheld):
        state.require_unapplied(phrase, asked=phrase, property_dcid=None, origin="bind")


async def judge_answer_coverage(
    state: ResolveState,
    answers: Sequence[tuple[ShapeDraft, Sequence[SlotBindingDraft], Materialised]],
) -> tuple[tuple[ResolvedQualifier, ...], ...]:
    """Judge stated-word coverage on each of `answers`, a candidate the part offers beside
    or in place of the elected answer: its shape, the bindings it is read under and its
    read. The rule is the elected answer's (`judge_coverage`), read on the members the
    candidate's own read served (`answer_view`), so "refugee" in "refugee assistance" is
    unread on "Assistance received" and read on "Refugee assistance given", whichever of
    the two was elected. The binder rates the words the lexical readers leave on each
    candidate, one call per candidate that leaves any, all at once (`_rated_words`).

    Each record no requirement of the part's ledger holds a phrase of is recorded as the
    elected reading's records are (`requirements.record_coverage`). Return each answer's
    records, in the order of `answers`, which the judgement reads it on and its trace names
    its phrases by, as two answers can name different codes for one phrase
    (`requirements.judge`). In log mode no record is a requirement, so no binder is asked
    and every answer holds no record."""
    if settings().qualifier_coverage != "refuse":
        return tuple(() for _answer in answers)
    views = [answer_view(state, shape, bindings, read) for shape, bindings, read in answers]
    ratings = await asyncio.gather(*(_rated_words(state, view) for view in views))
    uncovered: list[tuple[ResolvedQualifier, ...]] = []
    for view, rated in zip(views, ratings, strict=True):
        view.stated_word_ratings = rated
        records = tuple(
            record
            for record in _coverage_records(view, _query_records(view))
            if record.evidence == "uncovered"
        )
        record_coverage(state, records)
        uncovered.append(records)
    return tuple(uncovered)


def answer_view(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    read: Materialised,
) -> ResolveState:
    """A copy of `state` whose elected answer is `shape` read under `bindings` and served
    by `read`, with no word rated, for the coverage rule, which reads the elected answer
    from the state. Nothing writes to it but the ratings `judge_answer_coverage` keeps."""
    view = copy.copy(state)
    view.shape_draft = shape
    view.bindings = list(bindings)
    view.mat_result = read
    view.stated_word_ratings = {}
    return view


async def _rate_unread_words(state: ResolveState) -> None:
    """Keep the binder's rating of each word the coverage rule leaves unrated on the
    elected shape over the final bindings (`_rated_words`) as
    `ResolveState.stated_word_ratings`."""
    state.stated_word_ratings = await _rated_words(state, state)


async def _rated_words(state: ResolveState, answer: ResolveState) -> dict[str, float]:
    """The binder's rating of each word the coverage rule leaves unrated on the elected
    answer of `answer`, `state` or a view of one of its candidates (`answer_view`), over
    its final bindings (`_unrated_words`), with the read's usage added to `state`'s. The
    binder is asked only when such a word remains. A read that fails
    (`LLMInfraError`) rates no word, so each counts as stated, and is noted as lost
    (`note_degraded_read`), so the response cache stores no answer read without it. A read
    that leaves a word unrated is noted the same way, once: Jev leaving its question
    unanswered or holding none for a shape outside the menu, or Gemini answering no index
    for the word or answering it both ways. The request names every member the read served
    (`served_member_names`), and the binder rates the words on them wherever they differ
    from the shape's label (`stated_words_read.serves_beyond_label`), so a word one of them
    does not state reads as narrower than the answer. Where a served member has no name,
    the request names none and the bind's rating on the label stands, as the coverage rule
    then reads the named members' words (`phase_rank_bind._member_measure_words`). A read
    serving the one series the shape holds under the bindings serves the series the label
    names (`_serves_the_labelled_series`), so the request names none."""
    words = _unrated_words(answer)
    if not words:
        return {}
    shape = answer.shape_draft
    served_names = served_member_names(answer)
    names = list(served_names.values())
    named = [name for name in names if name is not None]
    served = (
        tuple(dict.fromkeys(named))
        if len(named) == len(names) and not _serves_the_labelled_series(answer, served_names)
        else ()
    )
    request = StatedWordsRequest(
        variable=state.variable,
        shape=shape,
        shape_index=next(
            (
                index
                for index, menu_shape in enumerate(state.capped_shapes)
                if menu_shape.shape_id == shape.shape_id
            ),
            None,
        ),
        words=words,
        bindings=tuple(answer.bindings),
        decision=state.bind_decision,
        served=served,
        part=state.variable_index,
        pool=state.pool,
    )
    try:
        ratings, usage = await state.binder.rate_words_stated(request, llm=state.llm)
    except LLMInfraError:
        logger.warning("stated words read failed; each word reads as stated", exc_info=True)
        note_degraded_read("stated words read")
        return {}
    rated = {word: ratings[word] for word in words if word in ratings}
    state.var_usage = _sum_usage(state.var_usage, usage)
    if len(rated) < len(words):
        note_degraded_read("stated words unrated")
    return rated


def _serves_the_labelled_series(answer: ResolveState, served: Mapping[str, str | None]) -> bool:
    """Whether the read of `answer`'s elected answer served, as `served`
    (`served_member_names`), the one member the elected shape holds under the final
    bindings (`phase_rank_bind._selected_members`). The label names that series however
    its own name reads: "net disbursements (interest received on OOF and export credits,
    non-grant (DAC type of finance))" is the series named "Interest received on OOF and
    export credits (net disbursements)", and naming it again drops what the label states.
    A shape holding another member under the bindings, as "Refugee population" beside the
    total population does, labels the series the bind elected, which the read may not
    serve."""
    return (
        len(served) == 1
        and served.keys() == _selected_members(answer.shape_draft, answer.bindings).keys()
    )


def log_would_refuse(state: ResolveState, result: RegionResult) -> None:
    """Log one ``qualifier_coverage would_refuse`` line, a JSON object, per term the pass's
    elected answer would refuse on in refuse mode (`ResolveState.would_refuse`): the part's
    index, the pass's pool, the variable, the elected shape, the term's phrase and the codes
    it names, the status `result`, the pass's answer, reached, the spec ids it answers with,
    and its flagged match's on a refusal, each word of the variable with its verdict
    (`ResolveState.word_verdicts`), and the binder's rating of each word it rated
    (`ResolveState.stated_word_ratings`)."""
    specs = result.specs or result.flagged_match or ()
    for record in state.would_refuse:
        line = {
            "part": state.variable_index,
            "pool": state.pool,
            "variable": state.variable,
            "shape": state.shape_draft.shape_id,
            "phrase": record.text,
            "codes": [ref.ref for ref in record.named],
            "status": result.status,
            "spec_ids": [spec.spec_id for spec in specs],
            "words": [list(verdict) for verdict in state.word_verdicts],
            "ratings": dict(state.stated_word_ratings),
        }
        logger.info("qualifier_coverage would_refuse %s", json.dumps(line, separators=(",", ":")))


@dataclass(frozen=True)
class _SettledBindings:
    """What the binding steps after the declared defaults settle under one set of role keys
    (`_settle_role_bindings`), for binding repair to install (`_install_settled`)."""

    # The final bindings, in canonical form.
    bindings: list[SlotBindingDraft]
    # The elected shape, naming the dataset the part is served from where it names one.
    shape: ShapeDraft
    # Whether a table of the part holds a row for `bindings` together.
    served: bool
    # The measure election (`bound_values.elect_measure_family`): the properties it forced,
    # the open axes the elected measure's members do not declare, and the partition.
    measure_forced: frozenset[str]
    measure_dropped_axes: tuple[str, ...]
    measure_elected_partition: MeasurePartition | None
    # The properties the shape's slot defaults bound, and the warning of each default no
    # member with data at the places reads.
    shape_defaulted: frozenset[str]
    warnings: tuple[Warning, ...]
    # Which what and how slots the identity carries (`_narrow_to_breakdown`), and the
    # elected shape it read, with every slot key.
    breakdown: BreakdownDecision
    breakdown_shape: ShapeDraft


def _bind_query_places(
    state: ResolveState,
    bindings: list[SlotBindingDraft],
    claims: Mapping[str, SlotBindingDraft],
) -> RegionResult | None:
    """Bind the query's places to the elected shape (`ResolveState.query_place_roles`) and
    settle the bindings under the role keys they give (`_settle_role_bindings`), or return
    the refusal of a part no table serves.

    The places bind as named (`place_roles.bind_place_roles`). Where the shape offers whole
    totals on sides they leave unnamed (`place_roles.whole_total_candidates`), the bindings
    are settled under each total's role keys in the order offered, and the first total
    binds whose bindings a table serves and, read without the claims, hold its series
    (`place_roles.holds_whole_total`). Where none does, they are settled under the named
    places' role keys, where no total remains to judge. With at most one total per side,
    the part settles at most three times: once per offered total, and once without. Each
    run starts from the same `bindings` and writes nothing to the state, and the bindings
    installed are those the bound total was judged under. Places that refuse to bind
    settle under no role keys, as the flagged identity of a role refusal binds no entity
    and leaves no axis open. `bindings` hold `claims`, the claims a recall sample alone
    backs, open, and each settling returns them (`_settle_role_bindings`)."""
    shape = state.shape_draft
    basis = _part_basis(state, shape)
    assignment = bind_place_roles(state, shape)
    if isinstance(assignment, PlaceAssignment):
        for whole in whole_total_candidates(state, shape, assignment):
            settled = _settle_role_bindings(
                state, shape, bindings, claims, whole.role_keys, basis=basis
            )
            if settled.served and holds_whole_total(
                state, settled.shape, whole, without_sampled_claims(settled.bindings)
            ):
                state.query_place_roles = whole
                return _install_settled(state, settled, basis=basis)
    state.query_place_roles = assignment
    role_keys = assignment.role_keys if isinstance(assignment, PlaceAssignment) else RoleKeys()
    settled = _settle_role_bindings(state, shape, bindings, claims, role_keys, basis=basis)
    return _install_settled(state, settled, basis=basis)


def _settle_role_bindings(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: list[SlotBindingDraft],
    claims: Mapping[str, SlotBindingDraft],
    role_keys: RoleKeys,
    *,
    basis: BasisElection | None = None,
) -> _SettledBindings:
    """Settle `bindings`, the elected `shape`'s after the declared defaults, under the
    query's `role_keys`, writing nothing to `state`. `basis` is the measure basis the part
    is counted on (`_part_basis`), None where the conventions do not read it. `bindings`
    hold `claims`, the claims a recall sample alone backs, open, and every step up to the
    slots the identity carries reads them so; the claims then return to the slots those
    steps leave open (`with_sampled_claims`).

    In order: force the measure axes the bindings leave open (`elect_measure_family`), the
    netting basis to `basis`'s first (`_basis_axes`), read an ODA flow binding as the flows
    the shape's basis counts (`_with_basis_flows`), bind
    the shape's slot defaults (`apply_shape_slot_defaults`), bind each open slot that
    partitions the served members to the value of the one member the variable phrase names
    (`bind_named_member_values`) and each that partitions a standard family to the one value
    its scored members state (`bind_partition_values`), then put the bindings in canonical
    form: each invariant slot of the dataset the part names that they leave open bound to
    its value (`canonical_pin_bindings`). The slots the identity carries are then decided
    over the family those bindings admit (`_narrow_to_breakdown`), and the dataset the part
    is served from is named under the bindings and those slots (`_served`)."""
    # Force measure axes left open after query matches and declared defaults. Run after draft
    # repair, which judges realizability against a capped recall sample.
    # Keep dropped axes for slot metadata and the elected partition for the sentence event.
    # The final Spec derives its measure from served rows, which can differ from this vote.
    election = elect_measure_family(
        shape,
        get_index(),
        _basis_axes(get_measure_index().axes, basis),
        shape_query(shape, bindings, role_keys.bound, open_role_keys=role_keys.open),
    )
    forced_pairs = election.forced if election is not None else ()
    axis_by_property = {
        slot.property_dcid: slot.axis for slot in shape.slot_keys if slot.property_dcid
    }
    forced_bindings = [
        SlotBindingDraft(axis=axis, property_dcid=prop, kind="value", value_dcids=[value])
        for prop, value in forced_pairs
        if (axis := axis_by_property.get(prop)) is not None
    ]
    forced_props = frozenset(b.property_dcid for b in forced_bindings if b.property_dcid)
    if forced_bindings:
        bindings = [
            b
            for b in bindings
            if not (b.property_dcid in forced_props and b.kind in ("unbound", "absent"))
        ] + forced_bindings

    # Bind each slot the elected shape's population type declares a default for and the
    # query left open. A default no member with data at the place reads stays unapplied, and
    # a warning says so.
    bindings, shape_defaulted, relaxed = apply_shape_slot_defaults(
        bindings, shape, index=get_index()
    )
    bindings = _with_basis_flows(shape, bindings)
    warnings: list[Warning] = []
    for default in relaxed:
        value = (shape.value_labels or {}).get(default.value_dcid, default.value_dcid)
        warnings.append(
            Warning(
                code=SLOT_DEFAULT_NOT_APPLIED,
                severity="info",
                message=(
                    f"No {value} series has data at the places asked, so the answer reads "
                    f"the total over {default.property_dcid} in place of the default, {value}."
                ),
            )
        )

    # A variable phrase equal to the name of one member the part would serve asks for that
    # member, so each open slot partitioning the served members binds to its value, and so
    # does the stretch of a single-variable request that writes a phrase extraction
    # reordered (`written_phrase`). The words outrank recall's evidence, which binds what
    # they leave open. A slot an operation reads is the reader's breakdown and stays as bound.
    bindings, _named_bindings = bind_named_member_values(
        shape,
        bindings,
        role_keys,
        variable=state.variable,
        written=written_phrase(state.variable, state.query_text),
        operation_slots=operation_slot_phrases(state.operations, shape),
    )

    # Bind each open slot that partitions a standard family to the one value its scored
    # members state, so no stand-in answers across it. The Spec's bound slot states it.
    bindings, _partition_bindings = bind_partition_values(shape, bindings, role_keys.bound)

    # Bind each invariant slot of the dataset the part names that the bindings leave open.
    # Every identity minted from here on, a refusal's flagged one included, is canonical,
    # and qualifier dispositions read the invariants as bound.
    bindings = canonical_pin_bindings(shape, bindings, role_keys)

    # The bindings change no more, so the slots the identity carries are decided on them,
    # over the family they admit, and a family with no total keeps only the slots its
    # answer reads. A claim selects no member until the walk confirms it, so the family
    # is the one the claim-free bindings admit.
    breakdown_shape = shape
    breakdown, shape = _narrow_to_breakdown(shape, bindings, role_keys)
    bindings = with_sampled_claims(bindings, claims)

    # The dataset the part is served from is settled on the bindings and the slots the
    # identity carries.
    served = True
    try:
        shape, bindings = _served(state, shape, bindings, role_keys)
    except PartNotServed:
        served = False
    return _SettledBindings(
        bindings=bindings,
        shape=shape,
        served=served,
        measure_forced=forced_props,
        measure_dropped_axes=election.dropped if election is not None else (),
        measure_elected_partition=election.elected if election is not None else None,
        shape_defaulted=frozenset(
            b.property_dcid for b in shape_defaulted if b.property_dcid is not None
        ),
        warnings=tuple(warnings),
        breakdown=breakdown,
        breakdown_shape=breakdown_shape,
    )


def _install_settled(
    state: ResolveState, settled: _SettledBindings, *, basis: BasisElection | None = None
) -> RegionResult | None:
    """Install `settled` on the state, or return the refusal of a part no table serves
    under its bindings (`_unserved_refusal`). A served part counted on `basis` records its
    measure-basis axis (`_record_basis`)."""
    state.bindings = settled.bindings
    _install_served_shape(state, settled.shape)
    state.measure_dropped_axes = settled.measure_dropped_axes
    state.measure_elected_partition = settled.measure_elected_partition
    if settled.measure_forced:
        state.measure_elected_properties = settled.measure_forced
    state.warnings.extend(settled.warnings)
    state.defaulted_properties |= settled.shape_defaulted
    state.breakdown = settled.breakdown
    state.breakdown_shape = settled.breakdown_shape
    if not settled.served:
        return _unserved_refusal(state)
    if basis is not None:
        _record_basis(state, settled, basis)
    return None


def _part_basis(state: ResolveState, shape: ShapeDraft) -> BasisElection | None:
    """The measure basis the part is counted on (`scope.basis.elect_basis`): read from the
    shape its places and admitted codes give it and its window, or from its explicit or
    carried measure-basis entry when `shape` measures the basis the entry asks for, else
    from the convention alone. None for a part the conventions read no shape of, and for a
    `shape` no dataset the conventions read publishes."""
    if not CONVENTION_DATASETS.intersection(shape_part_datasets(shape, get_index())):
        return None
    read = part_shape(
        directions=[place.direction for place in state.frame.places],
        admitted_systems={match.code_system for match in state.admitted_qualifiers},
        sector_systems=sector_systems(get_index()),
    )
    if read is None:
        return None
    election = elect_basis(
        shape=read,
        window=window_class(state.frame.window.value),
        entries=state.scope_selection.entries_on("measure_basis"),
    )
    netting = slot_values(shape.slot_taxonomy or {}, "nettingBasis")
    if election.entry is None or election.measured_by(
        shape.pop_type_dcid, shape.meas_prop_dcid, netting
    ):
        return election
    return election.by_convention()


def _basis_axes(
    axes: Sequence[MeasureAxis], basis: BasisElection | None
) -> tuple[MeasureAxis, ...]:
    """`axes` with the netting basis `basis` binds preferred first on its axis, so an open
    netting basis is elected as the basis counts it. `axes` as they are for no basis or a
    basis that binds no netting basis."""
    netting = basis.measure.netting_basis if basis is not None else None
    if netting is None:
        return tuple(axes)
    return tuple(
        replace(axis, prefer=(netting, *(v for v in axis.prefer if v != netting)))
        if axis.axis == "nettingBasis"
        else axis
        for axis in axes
    )


def _with_basis_flows(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> list[SlotBindingDraft]:
    """`bindings` with an ODA flow binding (CRS flow 10) read as the flows the basis
    `shape` measures counts (`BasisMeasure.oda_flows`) that its flow slot offers: grant
    equivalents count the private sector instruments of flow 60 beside it. Keyed on the
    served shape alone, however its basis was elected, so a part served on grant
    equivalents binds what a switch to them binds (`scope.edits`)."""
    measure = measure_of(shape.pop_type_dcid, shape.meas_prop_dcid, None)
    if measure is None:
        return list(bindings)
    offered = {
        code
        for key, codes in (shape.slot_taxonomy or {}).items()
        if key.endswith(f":{ODA_FLOW_SYSTEM}")
        for code in codes
    }
    flows = oda_flows(measure, offered)
    if len(flows) < 2:
        return list(bindings)
    return [
        draft.model_copy(update={"kind": "set", "value_dcids": list(flows)})
        if draft.property_dcid == ODA_FLOW_SYSTEM
        and draft.kind == "value"
        and draft.value_dcids == [ODA_FLOW]
        else draft
        for draft in bindings
    ]


def _record_basis(state: ResolveState, settled: _SettledBindings, basis: BasisElection) -> None:
    """Record the measure basis the part's served bindings hold
    (`ResolveState.scope_selection`), with the entry that settled it when they hold the
    option the entry asked for. A part whose convention is the OECD headline per year and
    which holds its net shape records the headline option, which the scope serves from the
    headline composite. A basis no entry settled that is not the convention's,
    served because no shape of the pool measures the convention's, records
    `SERVED_DIFFERS_RULE` in place of the convention's rule and is logged with the part's
    shape, so the choice does not claim the convention settled it. A shape whose measure no
    declared basis names records none."""
    shape = settled.shape
    netting = next(
        (
            draft.value_dcids[0]
            for draft in settled.bindings
            if draft.property_dcid == "nettingBasis"
            and draft.kind == "value"
            and len(draft.value_dcids) == 1
        ),
        None,
    )
    held = measure_of(shape.pop_type_dcid, shape.meas_prop_dcid, netting)
    if held is None:
        return
    active = measure_basis_option_id(held.basis)
    if basis.entry is None and basis.convention.basis == "oecd_headline" and held == basis.measure:
        # The headline's net shape stands for the headline per year, which the scope serves
        # from the headline composite (`scope.basis.HEADLINE_BINDING`).
        active = HEADLINE_OPTION
    entry = basis.entry if basis.entry is not None and basis.entry.option_id == active else None
    rule = BASIS_RULE
    if entry is None and held != basis.by_convention().measure:
        rule = SERVED_DIFFERS_RULE
        logger.info(
            "measure basis served differs from convention: %s",
            json.dumps(
                {
                    "part_shape": basis.convention.shape,
                    "window": basis.convention.window,
                    "convention": basis.by_convention().basis,
                    "served": held.basis,
                    "population_type": shape.pop_type_dcid,
                    "measured_property": shape.meas_prop_dcid,
                    "netting": netting,
                },
                separators=(",", ":"),
            ),
        )
    state.scope_selection.write(
        AxisRecord(
            axis="measure_basis",
            rule=rule,
            active=active,
            provenance="default" if entry is None else entry.provenance,
            entry=entry,
        )
    )
