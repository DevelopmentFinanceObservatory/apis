"""The request's requirements, judged against the part's final answers.

Each term the request names is recorded once in the part's ledger
(`state.ResolveState.requirements`): a code term the bindings leave unapplied, a measure
value the elected shape does not carry, a value the request names by its graph label, a query record of the elected reading no binding covers,
a value the bind drafted on a slot the elected reading lacks, a value unbinding drops and a
value a resubmit posts (`state.Requirement`). Nothing departs when a requirement is
recorded. Once the final answer is known, `judge` judges each final answer, the one answer,
a refusal's flagged or alternate match, the answer a refusal that judged none is about
(`phase_shared.settle_unjudged_refusal`) or every survivor of a candidate set, on its own shape and
bindings, and every disclosure derives from that one judgement: each answer's trace terms
(`answer_terms`), a refusal's terms, which equal its flagged match's trace
(`settle_answer`, `settle_flagged`), and the departures the part emits at its pipeline's
exit (`with_requirement_departures`). A refusal before the bindings are final names the
requirements only an answer can meet (`settle_unanswered`), and the rest depart.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from code_index.labels import PRICE_BASIS_AXIS, PRICES_BY_QUALIFIER
from code_index.shared import get_measure_index
from search.engine.bind import SlotBindingDraft
from search.engine.departures import (
    WarningKey,
    measure_not_applied,
    qualifier_property,
    warning_key,
)
from search.engine.qualifier_match import _normalize
from search.engine.region.materialise_step import Departed
from search.engine.region.phase_rank_bind import judge_value_requirements
from search.engine.region.qualifier_readings import answer_reads_measure
from search.engine.region.result import RegionResult
from search.engine.region.state import (
    CodeRule,
    MeasureRule,
    Requirement,
    RequirementRule,
    ResolveState,
    ValueRule,
    qualifier_dropped_warning,
    shape_vocabulary,
    with_requirement,
)
from search.engine.retrieval.bound_values import applies_value
from search.engine.shape import ShapeDraft
from search.engine.unit.conversion import PriceBasis
from search.models import GraphRef, ResolvedQualifier

if TYPE_CHECKING:
    from search.engine.region.promote_state import _PromoteState

# A final answer: its shape and the bindings it is read under.
type Answer = tuple[ShapeDraft, Sequence[SlotBindingDraft]]

# How one answer stands on a requirement. A requirement no answer is judged on, and a code
# on a property the answer's shape lacks, are outside it.
type Outcome = Literal["met", "unmet", "outside"]


@dataclass(frozen=True)
class Judgement:
    """The judgement of a part's final answers on its `requirements` (`judge`).
    `outcomes[i][j]` is how answer `i` stands on requirement `j`. `slot_terms[i][j]`, where
    set, is the term answer `i` names requirement `j` by in place of its own (`_value_term`).
    `refused` are the terms a `qualifier_not_bound` refusal names, empty when the part does
    not refuse.

    Requirements sharing a phrase are judged together: an answer leaves the phrase unmet
    when it leaves one of them unmet, meets it when it meets one and leaves none unmet, and
    is outside it otherwise. The phrase is named by its most specific requirement
    (`_specificity`), whatever order the ledger recorded them in, and departs as
    `departures` decides."""

    requirements: tuple[Requirement, ...]
    outcomes: tuple[tuple[Outcome, ...], ...]
    refused: tuple[ResolvedQualifier, ...] = ()
    slot_terms: tuple[Mapping[int, ResolvedQualifier], ...] = ()

    def _phrase_outcomes(self, answer: int) -> dict[str, Outcome]:
        by_phrase: dict[str, set[Outcome]] = {}
        for requirement, outcome in zip(self.requirements, self.outcomes[answer], strict=True):
            by_phrase.setdefault(requirement.phrase, set()).add(outcome)
        return {
            phrase: "unmet" if "unmet" in held else "met" if "met" in held else "outside"
            for phrase, held in by_phrase.items()
        }

    def met_phrases(self) -> frozenset[str]:
        """The phrases every judged answer meets, empty when no answer was judged."""
        if not self.outcomes:
            return frozenset()
        return frozenset.intersection(
            *(
                frozenset(
                    phrase
                    for phrase, outcome in self._phrase_outcomes(i).items()
                    if outcome == "met"
                )
                for i in range(len(self.outcomes))
            )
        )

    def terms(self, answer: int) -> tuple[ResolvedQualifier, ...]:
        """The phrases answer `answer` leaves unmet, in ledger order, each named by the term
        of its most specific requirement the answer leaves unmet."""
        slot_terms = self.slot_terms[answer] if self.slot_terms else {}
        terms: dict[str, int] = {}
        for j, (requirement, outcome) in enumerate(
            zip(self.requirements, self.outcomes[answer], strict=True)
        ):
            held = terms.get(requirement.phrase)
            if outcome == "unmet" and (
                held is None
                or _specificity(requirement.rule) > _specificity(self.requirements[held].rule)
            ):
                terms[requirement.phrase] = j
        return tuple(slot_terms.get(j, self.requirements[j].term) for j in terms.values())

    def departures(self) -> tuple[Departed, ...]:
        """The departures that disclose each phrase some answer does not meet and the
        refusal does not name, each such phrase once.

        A departure, a requirement's own or the one disclosing it `alone`, discloses every
        phrase whose requirement carries its warning (the values of one property unbinding
        drops together share one), so phrases sharing a departure are decided together
        (`_disclosing`): of the departures that disclose only phrases needing it, a group
        takes one joined departure when it discloses exactly the phrases needing it, else
        each such phrase's departure disclosing it alone. Of two departures disclosing the
        same phrases, the one whose requirement some answer leaves unmet wins, then the
        most specific (`_specificity`), then a requirement's own over its `alone`. A phrase
        every departure of which also discloses a met or refused phrase departs nothing.
        A `generic` departure names no phrase, so it discloses the phrases of its
        requirements that need it and no other departure discloses, whatever the others.
        The choice does not depend on the ledger's order; the departures follow it.

        With no answer, what the bindings leave unapplied departs: every requirement but a
        query record of the elected reading and a value that refuses, which are unmet only
        against an answer."""
        settled = {_normalize(term.text) for term in self.refused} | self.met_phrases()
        needing = {
            requirement.phrase
            for requirement in self.requirements
            if (requirement.departure or requirement.alone or requirement.generic) is not None
            and requirement.phrase not in settled
            and (
                self.outcomes
                or not requirement.refuses
                or not _met_only_by_an_answer(requirement.rule)
            )
        }
        # Each departure a requirement carries, its own (`departure`) and the one it
        # discloses alone (`alone`), with the departure's rank on that requirement: unmet
        # by some answer first, then the most specific, then its own departure.
        carriers: dict[WarningKey, list[tuple[tuple[bool, int, bool], int, Departed]]] = {}
        for j, requirement in enumerate(self.requirements):
            unmet = any(outcomes[j] == "unmet" for outcomes in self.outcomes)
            for departure, own in ((requirement.departure, True), (requirement.alone, False)):
                if departure is not None:
                    rank = (unmet, _specificity(requirement.rule), own)
                    carriers.setdefault(warning_key(departure[0]), []).append((rank, j, departure))
        covers = {
            key: frozenset(self.requirements[j].phrase for _rank, j, _departure in held)
            for key, held in carriers.items()
        }
        # A departure stands for the best ranked of its carriers, which within one rank
        # carry the same departure.
        lead = {key: max(held, key=lambda carrier: carrier[0]) for key, held in carriers.items()}
        chosen = {
            key: (min(j for _rank, j, _departure in carriers[key]), lead[key][2])
            for group in _disclosure_groups(covers)
            for key in _disclosing(group, covers, needing, rank=lambda key: lead[key][0])
        }
        undisclosed = needing.difference(*(covers[key] for key in chosen))
        for j, requirement in enumerate(self.requirements):
            if requirement.generic is not None and requirement.phrase in undisclosed:
                chosen.setdefault(warning_key(requirement.generic[0]), (j, requirement.generic))
        return tuple(departure for _j, departure in sorted(chosen.values(), key=lambda c: c[0]))


def _disclosure_groups(covers: Mapping[WarningKey, frozenset[str]]) -> list[set[WarningKey]]:
    """The departure keys of `covers`, grouped where they disclose a phrase in common,
    directly or through other keys."""
    groups: list[tuple[set[WarningKey], set[str]]] = []
    for key, phrases in covers.items():
        joined = [group for group in groups if not group[1].isdisjoint(phrases)]
        groups = [group for group in groups if group[1].isdisjoint(phrases)]
        groups.append(
            (
                {key}.union(*(keys for keys, _held in joined)),
                set(phrases).union(*(held for _keys, held in joined)),
            )
        )
    return [keys for keys, _phrases in groups]


def _disclosing(
    group: set[WarningKey],
    covers: Mapping[WarningKey, frozenset[str]],
    needing: set[str],
    *,
    rank: Callable[[WarningKey], tuple[bool, int, bool]],
) -> tuple[WarningKey, ...]:
    """The keys of `group` that disclose its phrases in `needing`, each once, from the keys
    disclosing only phrases in `needing`: the best ranked disclosing exactly them, else for
    each of them the best ranked disclosing it alone. For one unbinding, a key its values
    share plus each value's own, that is the fewest keys. Where joined keys of two
    properties overlap on a shared value label, it falls back to one key per phrase,
    which still discloses each phrase once."""
    eligible = sorted(key for key in group if covers[key] <= needing)
    wanted = frozenset().union(*(covers[key] for key in group)) & needing
    if whole := [key for key in eligible if covers[key] == wanted]:
        return (max(whole, key=rank),)
    return tuple(
        max(alone, key=rank)
        for phrase in sorted(wanted)
        if (alone := [key for key in eligible if covers[key] == {phrase}])
    )


def _met_only_by_an_answer(rule: RequirementRule) -> bool:
    """Whether a requirement of `rule` can be met only by an answer that applies it: a
    query record of the elected reading and a value the request names. With no answer
    judged, nothing met them, so a refusal names them (`settle_unanswered`), and every
    other requirement departs (`Judgement.departures`)."""
    return rule == "unbound" or isinstance(rule, ValueRule)


def _specificity(rule: RequirementRule) -> int:
    """How specifically a requirement of `rule` names its phrase: a code or measure value on
    a slot, then a query record of the elected reading, then a value named by its graph
    label, then a term no answer is judged on."""
    if isinstance(rule, CodeRule | MeasureRule):
        return 3
    if rule == "unbound":
        return 2
    return 1 if isinstance(rule, ValueRule) else 0


def judge(
    requirements: Sequence[Requirement],
    answers: Iterable[Answer],
    *,
    state: ResolveState | None = None,
    served: Sequence[PriceBasis | None] = (),
    uncovered: Sequence[tuple[ResolvedQualifier, ...] | None] = (),
) -> Judgement:
    """Judge each of `answers` on its own shape and bindings against every requirement.
    `served[i]` is the price basis of the unit request answer `i` is served under, None
    when it is served under none. An empty `served` serves every answer under none.
    `uncovered[i]` holds the records of the words answer `i` leaves unread, its coverage
    records read on the members its own read served (`phase_election.judge_answer_coverage`),
    None for the elected answer's, which are those of `state`'s query records. An empty
    `uncovered` reads every answer on the elected answer's.

    A `CodeRule` is met when the answer's bindings bind one of its codes or a descendant on
    its property, unmet when the answer's shape holds the property and binds none of them,
    and outside an answer whose shape lacks it. A `MeasureRule` is met when the answer's
    shape or a value its bindings bind carries the measure value (`answer_reads_measure`),
    else judged as its `CodeRule`. A price-basis `CodeRule`, alone or a `MeasureRule`'s, is
    also met by an answer served under a unit request whose price basis reads as one of its
    codes (`_serves_price_basis`). A `ValueRule` is met or unmet as
    `phase_rank_bind.judge_value_requirements` decides, and an answer leaving it unmet
    names it by its slot when its shape has one (`_value_term`). A value of a drafted set
    (origin ``set_value``) is met when the answer's bindings or some selected member hold
    it, or the answer leaves its set unmet, and is outside an answer that meets the set by
    membership without it, so it departs and refuses nothing. A query record of the elected
    reading no binding covers ("unbound") is unmet on every answer, but for a coverage
    record, which an answer names by its own record of the phrase (`_unbound_outcome`), and
    a term no answer is judged on (`None`) is outside
    every answer. A requirement that does not refuse (`Requirement.refuses`) is outside an
    answer that leaves it unmet, so it departs and refuses nothing."""
    requirements = tuple(requirements)
    answers = tuple(answers)
    served = tuple(served) or (None,) * len(answers)
    uncovered = tuple(uncovered) or (None,) * len(answers)
    elected_uncovered = (
        tuple(q for q in state.query_qualifiers if q.evidence == "uncovered")
        if state is not None
        else None
    )
    values = [r for r in requirements if isinstance(r.rule, ValueRule)]
    outcomes: list[tuple[Outcome, ...]] = []
    slot_terms: list[dict[int, ResolvedQualifier]] = []
    for (shape, bindings), prices, own in zip(answers, served, uncovered, strict=True):
        records = elected_uncovered if own is None else own
        by_phrase = None if records is None else {_normalize(r.text): r for r in records}
        held = shape_vocabulary(shape)
        bound: dict[str, set[str]] = {}
        for binding in bindings:
            if binding.property_dcid is not None:
                bound.setdefault(binding.property_dcid, set()).update(binding.value_dcids)
        unmet_values = {id(r) for r in judge_value_requirements(values, shape, bindings, state)}
        # The drafted sets this answer leaves unmet, whose own values it discloses nothing for.
        unmet_sets = [
            r.rule.values
            for r in values
            if r.origin == "stray"
            and r.refuses
            and id(r) in unmet_values
            and isinstance(r.rule, ValueRule)
        ]
        answer: list[Outcome] = []
        named: dict[int, ResolvedQualifier] = {}
        for j, requirement in enumerate(requirements):
            rule = requirement.rule
            if isinstance(rule, CodeRule):
                met_by_unit = prices is not None and _serves_price_basis(rule, prices)
                answer.append("met" if met_by_unit else _code_outcome(rule, held, bound))
            elif isinstance(rule, MeasureRule):
                met = answer_reads_measure(shape, rule.match, bindings) or (
                    prices is not None and _serves_price_basis(rule.code, prices)
                )
                answer.append("met" if met else _code_outcome(rule.code, held, bound))
            elif isinstance(rule, ValueRule) and requirement.origin == "set_value":
                # A value of a drafted set: met when the answer applies it or leaves the
                # whole set unmet, which the refusal names; else disclosed, never refused.
                set_unmet = any(rule.values <= drafted for drafted in unmet_sets)
                applies = id(requirement) not in unmet_values
                answer.append("met" if applies or set_unmet else "outside")
            elif isinstance(rule, ValueRule):
                if id(requirement) in unmet_values:
                    answer.append("unmet")
                    if (term := _value_term(requirement.term, rule, shape, held)) is not None:
                        named[j] = term
                else:
                    answer.append("met")
            elif rule == "unbound":
                outcome, record = _unbound_outcome(requirement, by_phrase)
                answer.append(outcome)
                if record is not None:
                    named[j] = record
            else:
                answer.append("outside")
            if not requirement.refuses and answer[-1] == "unmet":
                # A value only the bind model read is disclosed and refuses nothing.
                answer[-1] = "outside"
                named.pop(j, None)
        outcomes.append(tuple(answer))
        slot_terms.append(named)
    return Judgement(
        requirements=requirements, outcomes=tuple(outcomes), slot_terms=tuple(slot_terms)
    )


def _unbound_outcome(
    requirement: Requirement, records: Mapping[str, ResolvedQualifier] | None
) -> tuple[Outcome, ResolvedQualifier | None]:
    """How an answer stands on a query record of the elected reading no binding covers,
    and the answer's own record it names the phrase by, if any: unmet, but for a coverage
    record (evidence ``"uncovered"``), which an answer meets unless the coverage rule
    records its phrase on the answer's own read, `records` by phrase, and names by that
    record, so each answer names the codes of its own vocabulary. With no read judged
    (`records` None) it is unmet and named by the ledger's term."""
    if requirement.term.evidence != "uncovered" or records is None:
        return "unmet", None
    record = records.get(requirement.phrase)
    return ("met", None) if record is None else ("unmet", record)


def _value_term(
    term: ResolvedQualifier, rule: ValueRule, shape: ShapeDraft, held: frozenset[str]
) -> ResolvedQualifier | None:
    """The term an answer of `shape` that leaves the value requirement named `term` unmet
    names it by when the shape has a slot on a property of `rule`'s values: the words
    unbound on that slot, naming its values by their graph labels, as an unbound code is
    named. The age group slot of the infant mortality rate holds Infant, so "infant" is a
    category it leaves unbound. `None` when the shape has no such slot, where `term`, which
    names no code at all (``no_vocabulary``), stands."""
    slots = sorted({prop for prop, _value in rule.values} & held)
    if not slots:
        return None
    labels = shape.value_labels or {}
    return term.model_copy(
        update={
            "named": [
                GraphRef(ref=value, label=labels.get(value, value))
                for prop, value in sorted(rule.values)
                if prop == slots[0]
            ],
            "slot": slots[0],
            "disposition": "unbound",
        }
    )


def _serves_price_basis(rule: CodeRule, prices: PriceBasis) -> bool:
    """Whether an answer served under a unit request at `prices` meets the price-basis
    `rule`: the reading of one of its codes holds the measure value that states those
    prices (`PRICES_BY_QUALIFIER`). Constant prices meet `RealValue` and `InflationAdjusted`,
    current prices meet `Nominal`, and no unit request meets `PurchasingPowerParity`."""
    if rule.property_dcid != PRICE_BASIS_AXIS:
        return False
    index = get_measure_index()
    return any(
        PRICES_BY_QUALIFIER.get(value) == prices
        for code in rule.codes
        for value in index.reading_set(code)
    )


def _code_outcome(rule: CodeRule, held: frozenset[str], bound: Mapping[str, set[str]]) -> Outcome:
    if rule.property_dcid not in held:
        return "outside"
    values = bound.get(rule.property_dcid, set())
    applied = any(applies_value(code, value) for code in rule.codes for value in values)
    return "met" if applied else "unmet"


def _in_reading_order(
    reading: Sequence[ResolvedQualifier], unmet: Sequence[ResolvedQualifier]
) -> list[ResolvedQualifier]:
    """The elected reading's bound records and the terms `unmet`, each unmet phrase in the
    place of the reading's record of it, the rest after them in order."""
    pending = {_normalize(term.text): term for term in unmet}
    terms: list[ResolvedQualifier] = []
    for record in reading:
        if record.disposition == "bound":
            terms.append(record)
        elif (term := pending.pop(_normalize(record.text), None)) is not None:
            terms.append(term)
    return [*terms, *pending.values()]


def _reading(state: ResolveState, bindings: Sequence[SlotBindingDraft]) -> list[ResolvedQualifier]:
    """The elected reading's query records when `bindings` are its own (`state.bindings`),
    since an alternate or a measure sibling binds its values from its own representative."""
    return state.query_qualifiers if bindings is state.bindings else []


def answer_terms(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    *,
    served: PriceBasis | None = None,
    uncovered: tuple[ResolvedQualifier, ...] | None = None,
) -> list[ResolvedQualifier]:
    """The query terms the trace of an answer of `shape` read under `bindings`, served
    under a unit request at `served` prices (None for none) and whose coverage records hold
    `uncovered` (None for the elected answer's, `judge`) records: the elected reading's
    bound records when `bindings` are the elected ones, and the term of each requirement
    the answer leaves unmet (`judge`)."""
    unmet = judge(
        state.requirements,
        [(shape, bindings)],
        state=state,
        served=(served,),
        uncovered=(uncovered,),
    ).terms(0)
    return _in_reading_order(_reading(state, bindings), unmet)


def refusal_terms(
    state: ResolveState, shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> tuple[ResolvedQualifier, ...]:
    """The terms a `qualifier_not_bound` refusal on an answer of `shape` read under
    `bindings` names: every term of its trace (`answer_terms`) that is not bound."""
    return _unbound(answer_terms(state, shape, bindings))


def _unbound(terms: Iterable[ResolvedQualifier]) -> tuple[ResolvedQualifier, ...]:
    return tuple(q for q in terms if q.disposition != "bound")


def settle_answer(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    *,
    served: PriceBasis | None = None,
    uncovered: tuple[ResolvedQualifier, ...] | None = None,
) -> tuple[ResolvedQualifier, ...]:
    """Judge the part's one final answer, of `shape` read under `bindings`, served under
    a unit request at `served` prices (None for none, as a refusal serving no Spec) and
    whose coverage records hold `uncovered` (None for the elected answer's, `judge`), and
    keep the judgement for the part's departures (`ResolveState.judgement`). Return the
    terms its refusal names (`refusal_terms`), empty when the answer meets every requirement
    it can be judged on. A refused term departs nothing beside the refusal."""
    judgement = judge(
        state.requirements,
        [(shape, bindings)],
        state=state,
        served=(served,),
        uncovered=(uncovered,),
    )
    refused = _unbound(_in_reading_order(_reading(state, bindings), judgement.terms(0)))
    state.judgement = dataclasses.replace(judgement, refused=refused)
    return refused


def settle_flagged(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    *,
    uncovered: tuple[ResolvedQualifier, ...] | None = None,
) -> list[ResolvedQualifier]:
    """Judge the flagged or alternate match a refusal other than `qualifier_not_bound`
    carries, of `shape` read under `bindings` and whose coverage records hold `uncovered`
    (None for the elected answer's, `judge`), and keep the judgement for the part's
    departures. The refusal names no term, so each requirement the match leaves unmet
    departs. Return the terms its trace records (`answer_terms`)."""
    judgement = judge(state.requirements, [(shape, bindings)], state=state, uncovered=(uncovered,))
    state.judgement = judgement
    return _in_reading_order(_reading(state, bindings), judgement.terms(0))


def settle_unanswered(state: ResolveState) -> tuple[ResolvedQualifier, ...]:
    """Settle a `qualifier_not_bound` refusal that returns before any answer is judged, and
    keep its judgement for the part's departures. Return the terms it names beside its own
    records: each phrase of a requirement that refuses and only an answer can meet
    (`_met_only_by_an_answer`), named by its most specific such requirement, in ledger
    order. No answer met them, so the refusal names them, and every other requirement
    departs."""
    named: dict[str, Requirement] = {}
    for requirement in state.requirements:
        # A value of a drafted set is named by the set's own requirement.
        if (
            requirement.origin == "set_value"
            or not requirement.refuses
            or not _met_only_by_an_answer(requirement.rule)
        ):
            continue
        held = named.get(requirement.phrase)
        if held is None or _specificity(requirement.rule) > _specificity(held.rule):
            named[requirement.phrase] = requirement
    refused = tuple(requirement.term for requirement in named.values())
    state.judgement = Judgement(
        requirements=tuple(state.requirements), outcomes=(), refused=refused
    )
    return refused


def settle_candidate_set(
    state: ResolveState,
    answers: Sequence[Answer],
    *,
    served: Sequence[PriceBasis | None] = (),
    uncovered: Sequence[tuple[ResolvedQualifier, ...] | None] = (),
) -> None:
    """Judge a candidate set, each surviving answer on its own shape and bindings, served
    under a unit request at its `served` prices and on its own coverage records, `uncovered`
    (`judge`), and keep the judgement for the part's departures. The client chooses among
    the answers and can weigh an omission, so the set refuses nothing, and a requirement
    departs unless every answer meets it."""
    state.judgement = judge(
        state.requirements, answers, state=state, served=served, uncovered=uncovered
    )


def settle_posted(state: _PromoteState) -> tuple[ResolvedQualifier, ...]:
    """Judge a resubmit's answer on the values it posts and cannot apply
    (`unrealisable.drop_posted_values`), and keep the judgement for its departures. A posted
    identity's values are what the caller stated, so each is required: a posted set whose
    other values the answer applies still refuses on the one it filtered out. Return the
    terms its refusal names."""
    judgement = judge(state.requirements, [(state.shape_draft, state.bindings)])
    refused = judgement.terms(0)
    state.judgement = dataclasses.replace(judgement, refused=refused)
    return refused


def posted_refusal_terms(state: _PromoteState) -> tuple[ResolvedQualifier, ...]:
    """The terms a resubmit's refusal would name on its answer (`settle_posted`), without
    keeping the judgement."""
    return judge(state.requirements, [(state.shape_draft, state.bindings)]).terms(0)


def with_requirement_departures(
    result: RegionResult,
    requirements: Sequence[Requirement],
    judgement: Judgement | None,
) -> RegionResult:
    """`result` with the departure of each requirement its final answers leave unmet
    (`Judgement.departures`), the one place a requirement departs. A part whose final
    answer was never judged departs what its bindings leave unapplied. A warning the result
    already holds is not repeated, and a fact it already holds under that warning stays."""
    departed = (judgement or judge(requirements, ())).departures()
    if not departed:
        return result
    warnings = list(result.warnings)
    facts = dict(result.departure_facts)
    for warning, fact in departed:
        if warning not in warnings:
            warnings.append(warning)
        facts.setdefault(warning_key(warning), fact)
    return dataclasses.replace(result, warnings=tuple(warnings), departure_facts=facts)


def record_reading(state: ResolveState, records: Sequence[ResolvedQualifier]) -> None:
    """Set `records`, the elected shape's query records over the final bindings
    (`phase_rank_bind._resolve_query_qualifiers`), as `state.query_qualifiers`, and record
    each one no binding covers as a requirement (`_record_unbound`), replacing the previous
    reading's."""
    state.query_qualifiers = list(records)
    state.requirements = _record_unbound(
        tuple(r for r in state.requirements if r.rule != "unbound"), state.query_qualifiers
    )


def record_coverage(state: ResolveState, records: Sequence[ResolvedQualifier]) -> None:
    """Record each of `records`, the coverage records of a candidate the part offers
    beside or in place of the elected answer (`phase_election.judge_answer_coverage`), as
    the elected reading's are (`_record_unbound`), unless a requirement already holds its
    phrase, where that requirement judges the words on each answer (`judge`)."""
    held = {r.phrase for r in state.requirements}
    state.requirements = _record_unbound(
        state.requirements, [record for record in records if _normalize(record.text) not in held]
    )


def _record_unbound(
    requirements: tuple[Requirement, ...], records: Iterable[ResolvedQualifier]
) -> tuple[Requirement, ...]:
    """`requirements` with each of `records` no binding covers recorded as a requirement
    (rule "unbound", `judge`). A record whose phrase a code or value requirement holds adds
    nothing: that requirement judges the words on each answer. Each departs
    `measure_not_applied` cleared by the slot of the one property its rows name."""
    judged = {
        r.phrase for r in requirements if isinstance(r.rule, CodeRule | MeasureRule | ValueRule)
    }
    for record in records:
        if record.disposition == "bound" or _normalize(record.text) in judged:
            continue
        requirements = with_requirement(
            requirements,
            Requirement(
                term=record,
                origin="bind",
                rule="unbound",
                departure=(
                    qualifier_dropped_warning(record.text),
                    measure_not_applied(
                        asked=record.text, property_dcid=qualifier_property(record.named)
                    ),
                ),
            ),
        )
    return requirements


def _strays(requirements: Iterable[Requirement]) -> list[Requirement]:
    """The requirements of the values the bind drafted on slots the elected reading lacks
    and a recorded term supports (`phase_rank_bind._stray_requirements`), one per
    property."""
    return [
        r
        for r in requirements
        if r.origin == "stray" and r.refuses and isinstance(r.rule, ValueRule)
    ]


def stray_values(requirements: Iterable[Requirement]) -> frozenset[tuple[str, str]]:
    """The (property, value) pairs the bind drafted on slots the elected reading lacks and a
    recorded term supports (`phase_rank_bind._stray_requirements`), which the request
    names though no binding of the elected reading holds them."""
    return frozenset(
        pair
        for r in _strays(requirements)
        if isinstance(r.rule, ValueRule)
        for pair in r.rule.values
    )


def meets_stray_values(
    requirements: Iterable[Requirement], shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> bool:
    """Whether an answer of `shape` read under `bindings` meets every value the bind drafted
    on a slot the elected reading lacks (`stray_values`), judged as the ledger judges a
    value (`phase_rank_bind.judge_value_requirements`). The request names each such value,
    so it holds every answer, the elected reading included. The values drafted on one slot
    are one requirement, met by membership as a set bound on a slot reads: by an answer
    every member of which applies one of them (`bound_values.applies_one_of`), which is the
    alternate guard's rule for a member (`AlternateJudge._stray_refusal`)."""
    return not judge_value_requirements(_strays(requirements), shape, bindings)


def met_measure_phrases(judgement: Judgement | None) -> frozenset[str]:
    """The phrases of the measure values every final answer `judgement` judged carries
    (`MeasureRule`), which binding repair can bind after election. The part admits them, so
    no raw-text warning names a measure value its answer applies."""
    if judgement is None:
        return frozenset()
    measures = {r.phrase for r in judgement.requirements if isinstance(r.rule, MeasureRule)}
    return judgement.met_phrases() & measures


def unbinding_phrases(requirements: Iterable[Requirement]) -> frozenset[str]:
    """The normalised words of each departure unbinding records
    (`unrealisable.unbind_unrealisable`), the phrases the part's answer reads without."""
    return frozenset(
        _normalize(r.departure[1].asked)
        for r in requirements
        if r.origin == "unbinding" and r.departure is not None and r.departure[1].asked
    )
