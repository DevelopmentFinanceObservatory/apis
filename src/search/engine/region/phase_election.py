"""Settle the elected reading before place binding: vet the election, then repair and
default its bindings (`_phase_settle`)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress

from code_index.shared import get_index, get_measure_index
from search.engine.bind import SlotBindingDraft, validate_requested_all_bindings
from search.engine.bind_repair import (
    apply_broad_readings,
    apply_declared_defaults,
    apply_shape_slot_defaults,
    defaulted_slot_notes,
    defaulted_slots_named,
)
from search.engine.binders import StatedWordsRequest
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
    _coverage_terms,
    _elected_index,
    _force_qualifier_bindings,
    _log_admitted_qualifiers,
    _menu_choices,
    _open_operation_slots,
    _resolve_query_qualifiers,
    _shape_qualifiers,
    _unchosen_matches,
    _unheld_bound_codes,
    _unrated_words,
    _unread_qualifier_phrases,
    _withheld_qualifiers,
    judge_value_requirements,
    operation_slot_phrases,
    value_requirements,
)
from search.engine.region.place_roles import bind_place_roles
from search.engine.region.qualifier_readings import (
    answer_reads_measure,
    bound_value_reads_measure,
    reading_for_shape,
)
from search.engine.region.requirements import record_reading, settle_unanswered
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
    PartNotServed,
    RoleKeys,
    _collapse_bindings,
    bind_member_values,
    bind_shared_values,
    canonical_pin_bindings,
    elect_measure_family,
    five_tuple_member_facts,
    named_scope,
    part_source,
    read_held_families,
    shape_datasets,
    shape_for_part,
    shape_invariants,
    shape_query,
    sole_shared_value,
)
from search.engine.retrieval.member_facts import MemberFacts, asserted_constraint_properties
from search.engine.retrieval.standard import (
    bind_named_member_values,
    bind_partition_values,
    family_totals,
    holds_broader_reading,
)
from search.engine.shape import ShapeDraft
from search.engine.written_phrase import written_phrase
from search.models import ResolvedQualifier, Warning

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
    read of the part's family names it. Binding repair names it again under the bindings
    it leaves, and refuses a part none of whose tables holds a row for them
    (`_settle_served_dataset`). Every one of these decisions reads the final bindings
    (`_settle_bindings`): the model's drafts, vetted, with the query's forced codes,
    settled against the shape the model bound from. This is the first point the elected
    shape is known, and every reader of the part's dataset comes after it, so an opt-in
    table that candidacy withheld from an unnamed part is the part's dataset before its
    invariants, its source binding or its family are read.
    Every drafted binding on the source is discarded. The dataset the elected shape
    names (`ShapeDraft.named_dataset`) binds the source (`with_named_source`): the one
    dataset the part's scope leaves it, or the one opt-in dataset that alone may answer
    the shape. A weak retrieval score warns and sets no status (`status_rule`).
    Invalid ``requested_all`` claims are demoted. Only validated phrases suppress defaults
    and dropped-qualifier warnings.
    An operation over a category slot opens that slot as a valid claim whether or not
    bind made one.

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
    # bindings is refused once the repair has run (`_settle_served_dataset`).
    with suppress(PartNotServed):
        _name_served_dataset(
            state, bind_result.bindings, lambda: _query_role_keys(state, state.shape_draft)
        )
    if state.scope is not None:
        # The part's tables hold every code it binds, the model's and the forced query
        # matches alike.
        unheld = _unheld_bound_codes(bind_result.bindings, state.shape_draft)
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
    for match in state.measure_query_matches:
        phrase = _normalize(match.matched_text)
        if answer_reads_measure(shape_draft, match, bind_result.bindings):
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

    The source drafts leave and invalid ``requested_all`` claims are demoted
    (`bind.validate_requested_all_bindings`). The binder applies it to its pick's drafts,
    and this pass keeps a claim only on a slot of the elected shape. Each slot
    whose taxonomy holds one value and that the reply left unbound is claimed as that
    ``sole_option`` value (`_claim_single_member_slots`). A
    ``sole_option`` claim on a value the shape's population shows the slot a reading
    without leaves the slot unbound (`_validate_sole_option_bindings`), as does a value the
    measure's headline series already means (`_unbind_implied_values`). One value drafted
    on two properties no member holds together keeps the property a measure noun pins
    (`_keep_the_noun_pinned_property`). An operation
    over a category slot opens that slot. Model tie choices are
    vetted against the model's own bindings, and the query's qualifier matches the elected
    shape admits, with the codes the model chose from a phrase's candidate menu, are
    forced into the bindings (`_force_qualifier_bindings`).
    """
    bind_result = state.bind_result
    bind_result.bindings = without_dataset_drafts(bind_result.bindings)
    bind_result.bindings, state.applied_dropped_qualifier_phrases = validate_requested_all_bindings(
        bind_result.bindings,
        state.dropped_qualifier_phrases,
        slot_keys=shape_draft.slot_keys,
    )
    bind_result.bindings = _settle_sole_options(bind_result.bindings, shape_draft)
    bind_result.bindings = _unbind_implied_values(bind_result.bindings, shape_draft)
    bind_result.bindings = _keep_the_noun_pinned_property(bind_result.bindings, shape_draft)
    bind_result.bindings = _open_operation_slots(
        bind_result.bindings, state.operations, shape_draft
    )

    # The elected shape's own qualifiers: the matches its vocabulary reads and admits, and its
    # ties. Election vets only what that shape was shown, and reports its unheld ties, which
    # bind never offered, as dropped.
    own_matches, state.admitted_ties, unheld_ties = _shape_qualifiers(state, shape_draft)
    # Vet model tie choices against its original bindings before forced matches are added.
    # A row counts when a value binding names its code. One bound row satisfies the tie.
    # Zero or multiple rows leave it unresolved. Multiple bound properties reopen.
    bindings_by_property = {
        b.property_dcid: b for b in bind_result.bindings if b.property_dcid is not None
    }
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
        bind_result.bindings = [
            SlotBindingDraft(axis=b.axis, property_dcid=b.property_dcid, kind="unbound")
            if b.property_dcid in properties_to_unbind
            else b
            for b in bind_result.bindings
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
    choices = _menu_choices(state, shape_draft, bind_result.bindings, terms=terms)
    state.menu_choices = choices
    admitted_qualifiers = _unchosen_matches(terms, choices)
    state.admitted_qualifiers = admitted_qualifiers
    state.text_recognized_properties = frozenset()
    if admitted_qualifiers or choices:
        bind_result.bindings, forced = _force_qualifier_bindings(
            bind_result.bindings,
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
        _log_admitted_qualifiers(admitted_qualifiers)

    return unresolved, readings


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
    the family's total does not hold (a population shape whose sample holds one gender),
    which `_validate_sole_option_bindings` refutes. A ``requested_all`` draft asks for the
    slot in full and stays."""
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
    about it. The population is the members the shape's slot taxonomy describes
    (`five_tuple_member_facts`): the rows of the family the index carries for the shape,
    else the members of the family the catalogue holds for it beside its recalled ones,
    else its recalled members alone. The phrase did not name the value and a slot that
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
    beside members that state an age group classification instead. A shape with no
    members passes every claim through.
    """
    population, index_sourced = five_tuple_member_facts(shape_draft)
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
    have unknown constraints: there the value may separate readings the family holds. The population is the one
    `_validate_sole_option_bindings` reads (`five_tuple_member_facts`).

    The draft's evidence does not matter: the recorded bind reply to "life expectancy"
    drafts ``age=Years0`` as the slot's ``sole_option``, and a reply naming the age as
    ``mentioned`` is released the same way.
    """
    implied = get_index().implied_values_of(shape_draft.pop_type_dcid, shape_draft.meas_prop_dcid)
    if not implied:
        return bindings
    population, _ = five_tuple_member_facts(shape_draft)
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
    population, _ = five_tuple_member_facts(shape_draft)
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
    another value on it (`constraints` are the members' `read_constraints`), every total
    of the sample under the part's other bindings (`family_totals`), the member a slot
    left open serves, asserts `value`, and the sample holds no broader reading of the part
    without it (`holds_broader_reading`). A sample that splits on another open slot holds
    no total, and the broader reading is the member an open `prop` serves there."""
    if any(member.get(prop) not in (None, value) for member in constraints):
        return False
    if holds_broader_reading(population, stated_values, prop, value):
        return False
    return all(
        population[sv].read_constraints().get(prop) == value
        for sv in family_totals(population, stated_values)
    )


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
    source = part_source(shape, bindings, role_keys, scope=state.scope)
    if source is None or source == shape.named_dataset:
        return
    named = shape_for_part(shape, named_scope(source))
    state.capped_shapes[state.elected_index] = named
    state.shape_draft = named


def _settle_served_dataset(state: ResolveState, role_keys: RoleKeys) -> RegionResult | None:
    """Settle the dataset the part is served from under its final bindings
    (`ResolveState.bindings`), or return the refusal of a part no table serves.

    Binding repair and the slot defaults can bind a code after vetting named the dataset,
    so the rule runs again here (`_name_served_dataset`), on the bindings every identity of
    the part is minted from. A shape it names takes the source binding and the canonical
    form of that dataset's invariants (`canonical_pin_bindings`). A part none of whose
    tables holds a row for the bindings together refuses ``qualifier_not_bound``, naming
    the sources it cannot bind (`_unserved_sources`). A part under a scope that names
    sources names those alone, and any other part names the codes its bindings bind
    (`_bound_code_records`), so no table it excludes and no table of another family
    answers for it at any door. Each refusal also names each requirement only an answer
    can meet (`requirements.settle_unanswered`), such as a value bind drafted on a slot
    the elected shape lacks. A part that names and excludes no table and has none of these
    to name refuses ``variable_not_resolved``.
    """
    unnamed = state.shape_draft.named_dataset is None
    try:
        _name_served_dataset(state, state.bindings, lambda: role_keys)
    except PartNotServed:
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
    shape = state.shape_draft
    if unnamed and shape.named_dataset is not None:
        state.bindings = canonical_pin_bindings(
            shape, with_named_source(state.bindings, shape), role_keys
        )
    return None


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
    """The role keys an identity minted for `shape` binds the query's places under and the
    axes it leaves open. They are read before place binding installs them, from the assignment
    `bind_place_roles` gives (`PlaceAssignment.role_keys`). Places that refuse to bind give
    none, as the flagged identity of a role refusal binds no entity and leaves no axis
    open."""
    bound = bind_place_roles(state, shape)
    return RoleKeys() if isinstance(bound, str) else bound.role_keys


async def _repair_bindings(state: ResolveState) -> RegionResult | None:
    """Repair bindings and apply measure elections and defaults.

    In order, bind the values an unexported family shares, force open measure axes, apply
    declared defaults (none on a slot an invariant of the dataset the part names fixes), read
    each bound value the code index reads more broadly as that value (`apply_broad_readings`),
    apply the shape's slot defaults, bind each open slot that partitions the served members to
    the value of the one member the variable phrase names (`bind_named_member_values`) and
    each that partitions a standard family to the one value its scored members state
    (`bind_partition_values`), then put the bindings in
    canonical form: each invariant slot of the dataset the part names that they leave open
    bound to its value (`canonical_pin_bindings`). The dataset the part is served from is
    then settled under those bindings, and a part no table serves under them refuses
    ``qualifier_not_bound`` (`_settle_served_dataset`). The binder then rates the words the
    coverage rule leaves unrated on the elected shape (`_rate_unread_words`), and the query
    qualifiers resolve against the final bindings. Place binding reads the result.
    """
    bindings: list[SlotBindingDraft] = list(state.bind_result.bindings)
    state.bindings = bindings

    # A family the index does not export on a five-tuple it carries binds the values its
    # members share, so its identity resolves apart from the exported family.
    bindings, _shared_bindings = bind_shared_values(state.shape_draft, bindings)
    state.bindings = bindings

    # A part's series code names its StatVars, so each value they all share binds, and the
    # identity minted from the bindings elects those StatVars alone. A slot the query asked
    # to see in full that they hold one value on is a part the code cannot answer.
    if state.source_code is not None:
        bindings, _asked_in_full = bind_member_values(
            state.shape_draft, bindings, state.source_code.dcids
        )
        state.bindings = bindings
        if _asked_in_full:
            return state.no_data("variable_not_resolved", llm_usage=state.var_usage)

    # The role keys the minted identity binds its places under. With the bindings, they
    # elect the dataset the measure vote reads. A slot an invariant of the named dataset
    # fixes takes its value in canonical form, so no declared default binds it and no
    # default warning names it.
    role_keys = _query_role_keys(state, state.shape_draft)
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
        state.bindings = bindings
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
        state.bindings = bindings
        for reading in broad_readings:
            warning = Warning(code=SLOT_DEFAULTED, severity="warn", message=reading.note)
            state.warnings.append(warning)
            state.slot_defaults[reading.property_dcid] = SlotDefaultDisclosure(
                warnings=(warning,), reading=reading
            )

    # Force measure axes left open after query matches and declared defaults. Run after draft
    # repair, which judges realizability against a capped recall sample.
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

    # Bind each slot the elected shape's population type declares a default for and the
    # query left open. A default no member with data at the place reads stays unapplied, and
    # a warning says so.
    bindings, _shape_defaulted_bindings, relaxed = apply_shape_slot_defaults(
        bindings, state.shape_draft, index=get_index()
    )
    for default in relaxed:
        value = (state.shape_draft.value_labels or {}).get(default.value_dcid, default.value_dcid)
        state.warnings.append(
            Warning(
                code=SLOT_DEFAULT_NOT_APPLIED,
                severity="info",
                message=(
                    f"No {value} series has data at the places asked, so the answer reads "
                    f"the total over {default.property_dcid} in place of the default, {value}."
                ),
            )
        )
    if _shape_defaulted_bindings:
        state.bindings = bindings
        state.defaulted_properties |= frozenset(
            b.property_dcid for b in _shape_defaulted_bindings if b.property_dcid is not None
        )

    # A variable phrase equal to the name of one member the part would serve asks for that
    # member, so each open slot partitioning the served members binds to its value, and so
    # does the stretch of a single-variable request that writes a phrase extraction
    # reordered (`written_phrase`). The
    # words outrank recall's evidence, which binds what they leave open. A slot an operation
    # reads is the reader's breakdown and stays as bound.
    bindings, _named_bindings = bind_named_member_values(
        state.shape_draft,
        bindings,
        role_keys,
        variable=state.variable,
        written=written_phrase(state.variable, state.query_text),
        operation_slots=operation_slot_phrases(state.operations, state.shape_draft),
    )

    # Bind each open slot that partitions a standard family to the one value its scored
    # members state, so no stand-in answers across it. The Spec's bound slot states it.
    bindings, _partition_bindings = bind_partition_values(
        state.shape_draft, bindings, role_keys.bound
    )
    state.bindings = bindings

    # Bind each invariant slot of the dataset the part names that the bindings leave open.
    # Every identity minted from here on, a refusal's flagged one included, is canonical,
    # and qualifier dispositions read the invariants as bound.
    state.bindings = canonical_pin_bindings(state.shape_draft, bindings, role_keys)

    # The bindings change no more, so the dataset the part is served from is settled on
    # them, and a part no table of its own serves is refused.
    refused = _settle_served_dataset(state, role_keys)
    if refused is not None:
        return refused

    # Resolve query qualifiers against the final bindings, each one no binding covers a
    # requirement (`requirements.record_reading`). In "refuse" coverage mode a word of the
    # variable no reader covers is a record and the part refuses on it. In log mode the
    # record is kept once here, outside `unrealisable.unbind_unrealisable`'s re-read, for
    # the line the pass logs when it ends (`log_would_refuse`), and a word of a qualifier term the elected shape's own reading leaves unread is a term
    # no answer is judged on (`_unread_qualifier_phrases`). In refuse mode a record already
    # covers that word. In log mode a term of a part naming no dataset that names only codes
    # no shape can bind is withheld (`_withheld_qualifiers`), kept here alike, and is a
    # term no answer is judged on, whatever its property. Each value the request names by
    # its graph label is recorded once as a requirement (`value_requirements`) in refuse
    # mode, and in log mode the elected answer's unmet ones are kept by the same
    # rule (`judge_value_requirements`). Each value the bind drafted on a slot the elected
    # shape lacks is a requirement judged by the same rule, which bind recorded
    # (`phase_rank_bind._stray_requirements`), so its query record adds none. Every
    # requirement departs only if the final answer leaves it unapplied
    # (`requirements.with_requirement_departures`). The coverage rule reads a word beside
    # no code as unread only on the binder's rating, read first (`_rate_unread_words`).
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

    return None


async def _rate_unread_words(state: ResolveState) -> None:
    """Keep the binder's rating of each word the coverage rule leaves unrated on the
    elected shape over the final bindings (`_unrated_words`) as
    `ResolveState.stated_word_ratings`, with the read's usage added to the part's. The
    binder is asked once per pass, and only when such a word remains. A read that fails
    (`LLMInfraError`) rates no word, so each counts as stated, and is noted as lost
    (`note_degraded_read`), so the response cache stores no answer read without it. A read
    that leaves a word unrated is noted the same way, once: Jev leaving its question
    unanswered or holding none for a shape outside the menu, or Gemini answering no index
    for the word or answering it both ways."""
    words = _unrated_words(state)
    if not words:
        return
    shape = state.shape_draft
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
        bindings=tuple(state.bindings),
        decision=state.bind_decision,
        part=state.variable_index,
        pool=state.pool,
    )
    try:
        ratings, usage = await state.binder.rate_words_stated(request, llm=state.llm)
    except LLMInfraError:
        logger.warning("stated words read failed; each word reads as stated", exc_info=True)
        note_degraded_read("stated words read")
        return
    state.stated_word_ratings = {word: ratings[word] for word in words if word in ratings}
    state.var_usage = _sum_usage(state.var_usage, usage)
    if len(state.stated_word_ratings) < len(words):
        note_degraded_read("stated words unrated")


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
