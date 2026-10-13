"""The sentence each step record reads as, and the name of a computed version.

`describe_steps` serves each stored step as its JSON plus `text`, one sentence worded by its
operation's template and clipped to `LONG_TEXT_MAX_LENGTH`, since a template copies posted
strings such as a source step's description. `name_computed` names what a step list computes, such as "gross ODA
disbursements per gross national income". `left_out_of` words what one step left out, for a
caveat that points at it. `bound_entities` lists the entities a step's sides hold, whose
labels the vocabulary must hold for a sentence to name them. All are pure and read only the
steps and the vocabulary.

Every noun comes from data. An input is named by the vocabulary's label of its version, then
the label stored with the step, and an earlier step by the name of what it produced. The two
inputs of a comparison are told apart by the one role column of its `sides` (`{"left":
{"donor": "dcid:country/GBR"}, ...}`) whose entity differs between them, each named by the
vocabulary's label of its entity. A side the step itself narrows to one entity
(`left_entity`) is named with that entity in brackets. Parties
are named by the role wording of the roles a step's parameters group by (`numerator_by:
["recipient"]` reads "recipients"), else by what a count counts ("entities"). Periods are named
by the version's grain ("year"), else "period". Counts come from `kept`, `left_out` and the
coverage members only: "38 of 54" is the kept count plus the entities left out. A count of a
pooled role's parties stays out of that total and is named by the role's own noun ("left out
1 provider that totals other providers").

A reference no label names is served without one and named in `gaps`, as is an operation or a
reason the templates do not hold. Neither raises.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic.experimental.missing_sentinel import MISSING

from description.facts import ComposedVariable
from description.steps import (
    ComposedUnit,
    LeftOut,
    SimpleUnit,
    Step,
    StepInput,
    StepRef,
    Unit,
    VariableSide,
    VersionRef,
)
from description.stored import LONG_TEXT_MAX_LENGTH, clipped
from description.vocabulary import Vocabulary
from dfo_platform.roles import FlowFamily, RoleNoun, role_key_axis
from unit_words.currency_nouns import CURRENCY_NOUNS

type _Json = dict[str, Any]
"""A step record's JSON as `model_dump(mode="json")` writes it."""

type Join = Callable[[str, str, str], str]
"""Joins two sides of a name with the word between them: `(first, joiner, second)`, such as
`("gross ODA disbursements", "per", "population")`."""


def _spaced(first: str, joiner: str, second: str) -> str:
    """The two sides with the joiner between them, each separated by one space."""
    return f"{first} {joiner} {second}"


@dataclass(frozen=True, slots=True)
class DescribedSteps:
    """Steps as a description serves them.

    `served` holds each step record's JSON with its labels joined and its sentence in `text`,
    or `None` when the record holds no steps. `gaps` holds the dotted description paths of what
    could not be worded: `steps` when there are none, `steps.<id>.op` for an operation the
    templates do not hold, `steps.<id>.left_out.<n>.reason` for an unknown reason,
    `steps.<id>.left_out.<n>.role` for a pooled role no flow axis names,
    `steps.<id>.params.description` for a source pipeline step recorded without its
    description, and the `label` path of each reference no label names.
    """

    served: list[dict[str, object]] | None
    gaps: list[str]


def describe_steps(
    steps: Sequence[Step] | None,
    vocabulary: Vocabulary,
    *,
    grain: str | None = None,
    family: FlowFamily | None = None,
) -> DescribedSteps:
    """Word each step and join its labels.

    Args:
        steps: the record's steps in order, or `None` when it holds none.
        vocabulary: today's words.
        grain: the version's period grain ("year"), which names its periods. Without one
            they read "periods".
        family: the flow family whose role wording names the parties. Without one the
            generic wording ("providers", "recipients") applies.
    """
    if steps is None:
        return DescribedSteps(served=None, gaps=["steps"])
    gaps: list[str] = []
    served: list[_Json] = []
    for words in _walk(steps, vocabulary, grain=grain, family=family, gaps=gaps):
        step = words.step
        record = step.model_dump(mode="json")
        _join_labels(step, record, words)
        operation = _OPERATIONS.get(step.op)
        if operation is None:
            gaps.append(f"steps.{step.id}.op")
            text = f"Applied {step.op}, an operation this release does not describe."
        else:
            text = operation.template(words)
        record["text"] = clipped(text, limit=LONG_TEXT_MAX_LENGTH)
        served.append(record)
    return DescribedSteps(served=served, gaps=gaps)


def name_computed(steps: Sequence[Step], vocabulary: Vocabulary) -> str | None:
    """The name of what the last step produces, worded from its inputs and operations ("gross
    ODA disbursements per gross national income"), or `None` when a step on its path has no
    name: an unknown operation, a choice at mint, or an input no label names."""
    name: str | None = None
    for words in _walk(steps, vocabulary, grain=None, family=None, gaps=[]):
        name = _output_name(words)
    return name


@dataclass(frozen=True, slots=True)
class StepLeftOut:
    """What one step left out, in the words of its sentence.

    `operation` names the step ("the division"), `singular` and `plural` its parties
    ("recipient", "recipients"), and `reasons` each count of entities it left out with the
    phrase its sentence gives the reason ("with no gross national income figure").
    """

    operation: str
    singular: str
    plural: str
    reasons: list[tuple[int, str]]


def left_out_of(
    steps: Sequence[Step],
    step_id: str,
    vocabulary: Vocabulary,
    *,
    grain: str | None = None,
    family: FlowFamily | None = None,
) -> StepLeftOut | None:
    """What step `step_id` of `steps` left out, worded as `describe_steps` words it, or
    `None` when no step has that id."""
    for words in _walk(steps, vocabulary, grain=grain, family=family, gaps=[]):
        step = words.step
        if step.id != step_id:
            continue
        left_out = step.left_out if step.left_out is not MISSING else []
        return StepLeftOut(
            operation=operation.noun
            if (operation := _OPERATIONS.get(step.op)) is not None
            else f"step {step.id}",
            singular=words.entities(1),
            plural=words.entities(),
            reasons=[
                (gap.count, _reason_phrase(words, index, gap))
                for index, gap in enumerate(left_out)
                if gap.of == "entities"
            ],
        )
    return None


_DCID_PREFIX: Final = "dcid:"


def bound_entities(step: Step) -> list[str]:
    """The bare dcids of the entities `step`'s sides hold, in role order (`_sides`)."""
    return [dcid for side in _sides(step).values() for dcid in side.values()]


def _sides(step: Step) -> dict[str, dict[str, str]]:
    """`step`'s `sides` with each entity's bare dcid (`{"left": {"donor": "dcid:FRA"}}` reads
    `{"left": {"donor": "FRA"}}`). A step that records none has none."""
    if step.sides is MISSING:
        return {}
    return {
        role: {column: ref.removeprefix(_DCID_PREFIX) for column, ref in side.items()}
        for role, side in step.sides.items()
    }


def _walk(
    steps: Sequence[Step],
    vocabulary: Vocabulary,
    *,
    grain: str | None,
    family: FlowFamily | None,
    gaps: list[str],
    join: Join = _spaced,
) -> Iterator[_Words]:
    """The words of each step in order, each naming the earlier steps by what they produce,
    with two-sided names joined by `join`."""
    names: dict[str, str] = {}
    earlier: dict[str, Step] = {}
    for step in steps:
        words = _Words(
            step=step,
            vocabulary=vocabulary,
            names=names,
            steps=earlier,
            grain=grain,
            family=family,
            gaps=gaps,
            join=join,
        )
        yield words
        earlier[step.id] = step
        if (name := _output_name(words)) is not None:
            names[step.id] = name


# Words a template reads

_KEEPS_VARIABLE: Final = frozenset({"retrieve", "select", "filter", "rank", "aggregate"})
"""The operations whose output is the variable of their input, so it carries its name."""


@dataclass(slots=True)
class _Words:
    """One step and the words around it: earlier steps' names, the period and party nouns,
    and how a two-sided name joins its sides."""

    step: Step
    vocabulary: Vocabulary
    names: Mapping[str, str]
    steps: Mapping[str, Step]
    grain: str | None
    family: FlowFamily | None
    gaps: list[str]
    join: Join = _spaced

    def input_name(self, role: str | None = None) -> str | None:
        """The name of the step's input, or of its input in `role`."""
        step_input = self.step.input if role is None else _inputs(self.step).get(role)
        if step_input is None or step_input is MISSING:
            return None
        if isinstance(step_input, str):
            return self.names.get(step_input)
        label = version_label(step_input, self.vocabulary)
        return _mid_sentence(label) if label is not None else None

    def variable(self, role: str) -> str | None:
        """The variable the step's input in `role` carries: the version it traces back to
        through the steps that keep a variable, else the step that made it. `None` when the
        role has no input."""
        step_input = _inputs(self.step).get(role)
        while isinstance(step_input, str):
            earlier = self.steps.get(step_input)
            if earlier is None or earlier.op not in _KEEPS_VARIABLE:
                break
            if earlier.input is MISSING or earlier.input is None:
                break
            step_input = earlier.input
        if isinstance(step_input, VersionRef):
            return step_input.id
        return step_input if isinstance(step_input, str) else None

    def same_variable(self, role: str, other: str) -> bool:
        """Whether the step's inputs in two roles carry one variable."""
        variable = self.variable(role)
        return variable is not None and variable == self.variable(other)

    def entities_apart(self, role: str, other: str) -> tuple[str, str] | None:
        """The labels of the entities that tell the step's inputs in two roles apart. They are
        the entities in the one role column of the two sides (`_sides`) that differs, such as
        "donor" in `{"donor": GBR, "recipient": KEN}` against `{"donor": FRA, "recipient":
        KEN}`. `None` when no column, or more than one, differs, or a value has no label."""
        sides = _sides(self.step)
        first, second = sides.get(role, {}), sides.get(other, {})
        differing = {
            column
            for column in first.keys() | second.keys()
            if first.get(column) != second.get(column)
        }
        if len(differing) != 1:
            return None
        (column,) = differing
        if column not in first or column not in second:
            return None
        first_label = self.vocabulary.label(first[column])
        second_label = self.vocabulary.label(second[column])
        if first_label is None or second_label is None:
            return None
        return first_label, second_label

    def narrowed_to(self, role: str) -> str | None:
        """The label of the entity the step itself narrowed its input in `role` to, the one
        value of its `<role>_entity` parameter (`left_entity: {"donor": "dcid:FRA"}`). `None`
        when the parameter is absent, binds more than one column, or no label names it. An
        input retrieved for one entity holds it in `sides` but binds none here."""
        binding = _params(self.step).get(f"{role}_entity")
        if not isinstance(binding, dict) or len(binding) != 1:
            return None
        (ref,) = binding.values()
        return (
            self.vocabulary.label(ref.removeprefix(_DCID_PREFIX)) if isinstance(ref, str) else None
        )

    def input_words(self, role: str | None = None) -> str:
        """The input's name, else the id it is stored under."""
        if (name := self.input_name(role)) is not None:
            return name
        step_input = self.step.input if role is None else _inputs(self.step).get(role)
        if isinstance(step_input, VersionRef):
            return step_input.id
        if isinstance(step_input, str):
            return f"the result of step {step_input}"
        return "its input"

    def period(self, count: int = 1) -> str:
        noun = self.grain or "period"
        return noun if count == 1 else f"{noun}s"

    def entities(self, count: int = 2) -> str:
        """The noun for the step's parties: the plural of the one role its parameters group
        by, else "entities"."""
        params = _params(self.step)
        roles = {
            role
            for key, value in params.items()
            if (key == "by" or key.endswith("_by")) and isinstance(value, list)
            for role in value
            if isinstance(role, str)
        }
        axes = {role_key_axis(role) for role in roles}
        if len(axes) == 1 and (axis := axes.pop()) is not None:
            noun = self.vocabulary.roles(self.family).noun(axis)
            return noun.singular if count == 1 else noun.plural
        return "entity" if count == 1 else "entities"

    def role_noun(self, role: str) -> RoleNoun | None:
        """The wording of a graph role's parties ("providers"), or `None` for a role no flow
        axis names."""
        axis = role_key_axis(role)
        return self.vocabulary.roles(self.family).noun(axis) if axis is not None else None

    def pooled_parties(self, role: str) -> tuple[str, str]:
        """The singular and plural of a pooled role's parties: the role's noun, else
        "party"."""
        noun = self.role_noun(role)
        return (noun.singular, noun.plural) if noun is not None else ("party", "parties")


def version_label(version: VersionRef, vocabulary: Vocabulary) -> str | None:
    """A version's label: the vocabulary's, else the one stored with the step."""
    return vocabulary.label(version.id) or _held(version.label)


def _inputs(step: Step) -> Mapping[str, StepInput]:
    return step.inputs if step.inputs is not MISSING else {}


def _params(step: Step) -> Mapping[str, object]:
    return step.params if step.params is not MISSING else {}


def _held[T](value: T | MISSING) -> T | None:
    return None if value is MISSING else value


def _mid_sentence(label: str) -> str:
    """`label` as it reads inside a sentence: its first word lowered when it is plain Title
    Case ("Gross ODA disbursements" reads "gross ODA disbursements"). An acronym or a
    mixed-case word stays."""
    first, space, rest = label.partition(" ")
    if len(first) > 1 and first[0].isupper() and first[1:].islower():
        return f"{first.lower()}{space}{rest}"
    return label


def counted(count: int, singular: str, plural: str) -> str:
    """ "1 value", "2 values"."""
    return f"{count} {singular if count == 1 else plural}"


def series_text(items: Sequence[str]) -> str:
    """ "a", "a and b", "a, b and c"."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


# What a step kept and left out


def _kept_and_left(words: _Words) -> list[str]:
    """The clauses that say what a step kept and what data gaps it left out, by count."""
    step = words.step
    kept = step.kept if step.kept is not MISSING else {}
    left_out = step.left_out if step.left_out is not MISSING else []
    clauses: list[str] = []

    entities_kept = kept.get("entities")
    left_entities = sum(gap.count for gap in left_out if gap.of == "entities")
    if entities_kept is not None:
        total = entities_kept + left_entities
        noun = words.entities(total)
        clause = (
            f"{entities_kept} of {total} {noun} kept"
            if left_entities
            else f"{entities_kept} {noun} kept"
        )
        if (fewer := _fewer_periods(words, kept)) is not None:
            clause += f", {fewer}"
        clauses.append(clause)
    elif (coverage := _period_coverage(words)) is not None:
        clauses.append(coverage)

    for of in ("entities", "periods", "entity_periods"):
        gaps = [(index, gap) for index, gap in enumerate(left_out) if gap.of == of]
        if gaps:
            clauses.append(_left_out_clause(words, gaps, implicit=entities_kept is not None))
    pooled: dict[str, list[tuple[int, LeftOut]]] = {}
    for index, gap in enumerate(left_out):
        if gap.of == "pooled_entities" and gap.role is not MISSING:
            pooled.setdefault(gap.role, []).append((index, gap))
    clauses.extend(_pooled_clause(words, role, gaps) for role, gaps in pooled.items())
    return clauses


def _fewer_periods(words: _Words, kept: Mapping[str, int]) -> str | None:
    """How many kept entities hold fewer periods than the window: from `kept`, else from a
    rank's entity coverage. `None` when no count says so, or none falls short."""
    fewer = kept.get("entities_with_fewer_periods")
    periods = kept.get("periods")
    coverage = words.step.entity_coverage
    if fewer is None and coverage is not MISSING and coverage:
        periods = max(item.of for item in coverage)
        fewer = sum(1 for item in coverage if item.periods < item.of)
    if not fewer:
        return None
    if periods is None:
        return f"{fewer} of them missing some {words.period(2)}"
    return f"{fewer} of them with fewer than {periods} {words.period(periods)}"


def _period_coverage(words: _Words) -> str | None:
    """How many entities each period's total covers, when the step records it."""
    coverage = words.step.period_coverage
    if coverage is MISSING or not coverage:
        return None
    low = min(item.entities for item in coverage)
    high = max(item.entities for item in coverage)
    span = f"{low}" if low == high else f"{low} to {high}"
    return f"each {words.period()}'s total covers {span} {words.entities(high)}"


def _left_out_clause(words: _Words, gaps: Sequence[tuple[int, LeftOut]], *, implicit: bool) -> str:
    """One kind of count left out: "16 left out, 9 with no ..., 5 ... and 2 ...", or with one
    reason "25 left out with no value in 2015 or 2024"."""
    of = gaps[0][1].of
    total = sum(gap.count for _, gap in gaps)
    if of == "entities":
        subject = f"{total}" if implicit else f"{total} {words.entities(total)}"
    elif of == "periods":
        subject = f"{total} {words.period(total)}"
    else:
        subject = counted(total, "value", "values")
    phrases = [_reason_phrase(words, index, gap) for index, gap in gaps]
    if len(phrases) == 1:
        return f"{subject} left out {phrases[0]}"
    reasons = [f"{gap.count} {phrase}" for (_, gap), phrase in zip(gaps, phrases, strict=True)]
    return f"{subject} left out, {series_text(reasons)}"


def _pooled_clause(words: _Words, role: str, gaps: Sequence[tuple[int, LeftOut]]) -> str:
    """The parties of one pooled role left out, named by the role's noun: "left out 1 provider
    that totals other providers". A role no flow axis names reads "parties", with a gap."""
    if words.role_noun(role) is None:
        words.gaps.extend(f"steps.{words.step.id}.left_out.{index}.role" for index, _ in gaps)
    singular, plural = words.pooled_parties(role)
    total = sum(gap.count for _, gap in gaps)
    subject = counted(total, singular, plural)
    phrases = [_reason_phrase(words, index, gap) for index, gap in gaps]
    if len(phrases) == 1:
        return f"left out {subject} {phrases[0]}"
    reasons = [f"{gap.count} {phrase}" for (_, gap), phrase in zip(gaps, phrases, strict=True)]
    return f"left out {subject}, {series_text(reasons)}"


def _reason_phrase(words: _Words, index: int, gap: LeftOut) -> str:
    """Why a count was left out, as a phrase that follows the count."""
    other = words.input_name("denominator")
    period = words.period()
    periods = words.period(2)
    match gap.reason, gap.of:
        case "unmatched_key", "entities":
            return f"with no {other} figure" if other else "with no match on the other side"
        case "unmatched_key", _:
            return "with no match on the other side"
        case "period_unmatched", "entities":
            return f"sharing no {period} with {'it' if other else 'the other side'}"
        case "period_unmatched", "periods":
            return "that only one side reports"
        case "period_unmatched", _:
            return f"in {periods} the other side lacks"
        case "zero_denominator", "entities":
            return f"with zero {other}" if other else "with a zero denominator"
        case "zero_denominator", _:
            return f"where {other} is zero" if other else "where the denominator is zero"
        case "unit_unbound", _:
            return "that declare no unit"
        case "party_total", "pooled_entities" if gap.role is not MISSING:
            verb = "totals" if gap.count == 1 else "total"
            return f"that {verb} other {words.pooled_parties(gap.role)[1]}"
        case "party_total", _:
            return f"that total other {words.entities()}"
        case "endpoint_missing", _:
            start, end = _params(words.step).get("start"), _params(words.step).get("end")
            if isinstance(start, str) and isinstance(end, str):
                return f"with no value in {start} or {end}"
            return f"with no value in the first or last {period}"
        case "zero_or_sign_crossing_base", _:
            return "whose base is zero or changes sign"
        case "non_finite", _:
            return "whose change is not a finite number"
        case "currency_unmapped", _:
            return "that no single national currency names"
        case "deflator_missing", "entities":
            return "with no deflator in the base year"
        case "deflator_missing", _:
            return f"in {periods} the deflator table lacks"
        case _:
            words.gaps.append(f"steps.{words.step.id}.left_out.{index}.reason")
            return f"for {gap.reason}"


def _sentence(head: str, words: _Words) -> str:
    clauses = _kept_and_left(words)
    return f"{head}: {'; '.join(clauses)}." if clauses else f"{head}."


# One template per operation in dfo-spec-kit's `operations.json`


def _retrieve(words: _Words) -> str:
    return _sentence(f"Retrieved {words.input_words()}", words)


def _both_report(words: _Words) -> str:
    return f"for each {words.period()} both report"


def _sum(words: _Words) -> str:
    head = f"Added {words.input_words('left')} and {words.input_words('right')}"
    return _sentence(f"{head} {_both_report(words)}", words)


def _quotient(words: _Words) -> str:
    head = f"Divided {words.input_words('numerator')} by {words.input_words('denominator')}"
    return _sentence(f"{head} {_both_report(words)}", words)


def _share(words: _Words) -> str:
    numerator = words.input_words("numerator")
    if words.same_variable("numerator", "denominator"):
        head = f"Took each {words.entities(1)}'s share of the total of {numerator}"
        return _sentence(f"{head} for each {words.period()}", words)
    head = f"Took {numerator} as a share of {words.input_words('denominator')}"
    return _sentence(f"{head} {_both_report(words)}", words)


def _growth(words: _Words) -> str:
    params = _params(words.step)
    change = params.get("change")
    kind = f"{change} change" if isinstance(change, str) else "change"
    start, end = params.get("start"), params.get("end")
    window = (
        f" from {start} to {end}"
        if isinstance(start, str) and isinstance(end, str)
        else f" between the first and last {words.period()}"
    )
    return _sentence(f"Took the {kind} in {words.input_words()}{window}", words)


def _rank(words: _Words) -> str:
    params = _params(words.step)
    head = f"Ranked {words.entities()} by {words.input_words()}"
    limit = params.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool):
        end = "bottom" if params.get("order") == "ascending" else "top"
        head += f", keeping the {end} {limit}"
    return _sentence(head, words)


def _aggregate(words: _Words) -> str:
    return _sentence(f"Aggregated {words.input_words()}, skipping absent values", words)


_COMPARISONS: Final[Mapping[str, str]] = {
    "gt": "above",
    "ge": "at or above",
    "lt": "below",
    "le": "at or below",
    "eq": "equal to",
    "ne": "other than",
}


def _predicate(predicate: object) -> str | None:
    if not isinstance(predicate, dict):
        return None
    column, op, operand = predicate.get("column"), predicate.get("op"), predicate.get("operand")
    comparison = _COMPARISONS.get(op) if isinstance(op, str) else None
    if not isinstance(column, str) or comparison is None or isinstance(operand, (dict, list)):
        return None
    if column == "value":
        return f"{comparison} {operand}"
    return f"where {column} is {comparison} {operand}"


def _filter(words: _Words) -> str:
    predicates = _params(words.step).get("predicates")
    head = f"Kept the values of {words.input_words()}"
    if isinstance(predicates, list) and predicates:
        worded = [_predicate(predicate) for predicate in predicates]
        if all(phrase is not None for phrase in worded):
            head += " " + series_text([phrase for phrase in worded if phrase is not None])
        else:
            head += f" that meet {counted(len(predicates), 'condition', 'conditions')}"
    return _sentence(head, words)


def _compare(words: _Words) -> str:
    """A comparison reads as its name (`_difference`), from its inputs' names, else the ids
    they are stored under. One variable with nothing to tell its sides apart reads as the
    difference between two series of it."""
    left, right = words.input_words("left"), words.input_words("right")
    difference = _difference(words, left, right)
    head = (
        f"Calculated {difference}"
        if difference is not None
        else f"Calculated the difference between two series of {left}"
    )
    return _sentence(f"{head} {_both_report(words)}", words)


def _join(words: _Words) -> str:
    head = (
        f"Joined {words.input_words('left')} and {words.input_words('right')} "
        f"on the {words.entities()} and {words.period(2)} both hold"
    )
    return _sentence(head, words)


def _ref_words(ref: StepRef, vocabulary: Vocabulary) -> str:
    return ref_label(ref, vocabulary) or ref.ref


def _choose_source(words: _Words) -> str:
    choice = words.step.choice
    if choice is MISSING:
        return _sentence("Chose one source of several", words)
    head = f"Chose {_ref_words(choice.chosen, words.vocabulary)}"
    if choice.others:
        others = [_ref_words(ref, words.vocabulary) for ref in choice.others]
        head += f" over {series_text(others)}"
    ranking = ", then ".join(name.replace("_", " ") for name in choice.ranking)
    return _sentence(f"{head}, ranking by {ranking}", words)


_SUMMED_NAMED: Final = 3
"""How many summed variables a sentence names before counting the rest."""


def _sum_variables(words: _Words) -> str:
    summed = words.step.summed
    if summed is MISSING:
        return _sentence("Summed several variables into one series", words)
    head = f"Summed {counted(summed.count, 'variable', 'variables')}"
    named = [_ref_words(ref, words.vocabulary) for ref in summed.variables[:_SUMMED_NAMED]]
    if named:
        rest = summed.count - len(named)
        head += ": " + series_text([*named, f"{rest} more"] if rest > 0 else named)
    return _sentence(head, words)


def _convert(words: _Words) -> str:
    step = words.step
    into = _unit_words(_held(step.unit), words.vocabulary)
    source = _unit_words(_held(step.converted_from), words.vocabulary)
    if into is None:
        head = "Converted the values to another currency or price basis"
    elif source is None:
        head = f"Converted the values to {into}"
    else:
        head = f"Converted {source} to {into}"
    return _sentence(head, words)


def _continue_series(words: _Words) -> str:
    """The values read as estimated: the operation continues a series only with a method
    `materialise.method_status` classes as estimated, so the sentence takes the word from the
    operation and leaves the method's code in `params`."""
    periods = _params(words.step).get("periods")
    head = "Took the same source's estimated values"
    match periods:
        case [str(first), str(last)] if first == last:
            head += f" for {first}"
        case [str(first), str(last)]:
            head += f" for {first} to {last}"
        case _:
            head += f" for the later {words.period(2)}"
    return _sentence(f"{head}, after its reported series ends", words)


def _source_step(words: _Words) -> str:
    """A source pipeline step reads as its description as recorded. Without one it reads as
    its title, with a gap."""
    params = _params(words.step)
    description = params.get("description")
    if isinstance(description, str) and description.strip():
        return description
    words.gaps.append(f"steps.{words.step.id}.params.description")
    title = params.get("title")
    if isinstance(title, str) and title.strip():
        return f"{title}."
    return "Applied a step of the source's pipeline."


def _select(words: _Words) -> str:
    step = words.step
    kept = step.kept if step.kept is not MISSING else {}
    head = "Drew"
    if (series := kept.get("series")) is not None:
        head += f" {counted(series, 'series', 'series')} of"
    head += f" {words.input_words()}"
    window = _params(step).get("window")
    if isinstance(window, dict):
        start, end = window.get("start"), window.get("end")
        if isinstance(start, str) and isinstance(end, str):
            head += f", {start} to {end}"
    return _sentence(head, words)


_CHANGE_NOUNS: Final = {"difference": "difference", "percent_change": "percent change"}


def _change(words: _Words) -> str:
    """A stat card's change: its derivation, in what it drew, between its two basis
    periods."""
    params = _params(words.step)
    derivation = params.get("derivation")
    noun = _CHANGE_NOUNS.get(derivation, "change") if isinstance(derivation, str) else "change"
    head = f"Took the {noun} in {words.input_words()}"
    basis = params.get("basis")
    if isinstance(basis, list) and len(basis) == 2 and all(isinstance(p, str) for p in basis):
        head += f" from {basis[0]} to {basis[1]}"
    return _sentence(head, words)


@dataclass(frozen=True, slots=True)
class _Operation:
    """How a step of one operation reads: its sentence, and how a caveat names it."""

    template: Callable[[_Words], str]
    noun: str


_OPERATIONS: Final[Mapping[str, _Operation]] = {
    "retrieve": _Operation(_retrieve, "the retrieval"),
    "sum": _Operation(_sum, "the sum"),
    "quotient": _Operation(_quotient, "the division"),
    "share": _Operation(_share, "the share"),
    "growth": _Operation(_growth, "the change"),
    "rank": _Operation(_rank, "the ranking"),
    "aggregate": _Operation(_aggregate, "the aggregate"),
    "filter": _Operation(_filter, "the filter"),
    "compare": _Operation(_compare, "the comparison"),
    "join": _Operation(_join, "the join"),
    "source_step": _Operation(_source_step, "the source's pipeline"),
    "choose_source": _Operation(_choose_source, "the choice of source"),
    "sum_variables": _Operation(_sum_variables, "the sum of variables"),
    "convert": _Operation(_convert, "the conversion"),
    "continue_series": _Operation(_continue_series, "the continuation"),
    "select": _Operation(_select, "the selection"),
    "change": _Operation(_change, "the change"),
}
"""The sentence template and noun of each operation dfo-spec-kit lists."""


def templated_operations() -> frozenset[str]:
    """The operations a sentence template words."""
    return frozenset(_OPERATIONS)


# What each step produces


def _output_name(words: _Words) -> str | None:
    """The name of what a step produces, from its inputs' names and its operation."""
    match words.step.op:
        case op if op in _KEEPS_VARIABLE:
            return words.input_name()
        case "growth":
            name = words.input_name()
            return f"change in {name}" if name else None
        case "quotient" | "share":
            numerator = words.input_name("numerator")
            denominator = words.input_name("denominator")
            if numerator is None or denominator is None:
                return None
            if words.step.op == "share":
                if words.same_variable("numerator", "denominator"):
                    return f"share of {numerator}"
                return words.join(numerator, "as a share of", denominator)
            return words.join(numerator, "per", denominator)
        case "compare":
            return _difference_name(words)
        case "sum" | "join":
            left, right = words.input_name("left"), words.input_name("right")
            if left is None or right is None:
                return None
            joiner = {"sum": "plus", "join": "with"}[words.step.op]
            return words.join(left, joiner, right)
        case _:
            return None


def _difference_name(words: _Words) -> str | None:
    """A comparison's output (`_difference`), else "difference in <variable>". `None` when an
    input has no name."""
    left, right = words.input_name("left"), words.input_name("right")
    if left is None or right is None:
        return None
    return _difference(words, left, right) or f"difference in {left}"


def _difference(words: _Words, left: str, right: str) -> str | None:
    """A comparison's output in the order it subtracts, from the words `left` and `right`
    for its inputs. Each side reads as its words, followed by the entity the step itself
    narrowed it to (`narrowed_to`) in brackets: "gross ODA disbursements (France)". Two
    versions whose sides read differently give "<right> minus <left>". One version, or two
    whose sides read alike, give "<variable>, <right entity> minus <left entity>" when one role
    column tells the sides apart (`entities_apart`). `None` when nothing tells them apart."""
    left_side, right_side = _side_words(words, "left", left), _side_words(words, "right", right)
    if not words.same_variable("left", "right") and left_side != right_side:
        return words.join(right_side, "minus", left_side)
    if (entities := words.entities_apart("left", "right")) is None:
        return None
    left_entity, right_entity = entities
    return f"{left}, {right_entity} minus {left_entity}"


def _side_words(words: _Words, role: str, name: str) -> str:
    entity = words.narrowed_to(role)
    return f"{name} ({entity})" if entity is not None else name


# Labels


def ref_label(ref: StepRef, vocabulary: Vocabulary) -> str | None:
    """A reference's label: the vocabulary's, else the one stored with the step."""
    return vocabulary.label(ref.ref) or _held(ref.label)


def _current_unit_label(side: SimpleUnit | VariableSide, vocabulary: Vocabulary) -> str | None:
    """Today's words for a unit side: a variable's label; for a unit, the currency noun of its
    dcid, then the vocabulary's label of its dcid, then the noun of its currency code."""
    if isinstance(side, VariableSide):
        return vocabulary.label(side.stat_var)
    ref, currency = _held(side.ref), _held(side.currency)
    if ref is not None and (word := vocabulary.unit_noun(ref) or vocabulary.label(ref)):
        return word
    return CURRENCY_NOUNS.get(currency) if currency is not None else None


def _unit_label(side: SimpleUnit | VariableSide, vocabulary: Vocabulary) -> str | None:
    """A unit side's words: today's, else the label stored with the step."""
    return _current_unit_label(side, vocabulary) or _held(side.label)


def _unit_words(unit: Unit | None, vocabulary: Vocabulary) -> str | None:
    """A unit as a sentence names it: "US dollars in constant 2023 prices"."""
    if isinstance(unit, ComposedUnit):
        numerator = _unit_label(unit.numerator, vocabulary)
        denominator = _unit_label(unit.denominator, vocabulary)
        return f"{numerator} per {denominator}" if numerator and denominator else None
    if unit is None or (label := _unit_label(unit, vocabulary)) is None:
        return None
    prices = _held(unit.prices)
    if prices is None:
        return label
    year = _held(unit.base_year)
    basis = f"{prices} {year}" if year is not None else prices
    return f"{label} in {basis} prices"


def _set_label(served: _Json, label: str | None, *, current: bool) -> bool:
    """Write `label` into a served reference. A current label drops `at_mint`; a stored one
    keeps it. Returns whether a label was found."""
    if label is None:
        return False
    served["label"] = label
    if current:
        served.pop("at_mint", None)
    return True


def _join_ref(ref: StepRef | VersionRef, served: _Json, words: _Words, path: str) -> None:
    key = ref.ref if isinstance(ref, StepRef) else ref.id
    current = words.vocabulary.label(key)
    stored = _held(ref.label)
    if not _set_label(served, current or stored, current=current is not None):
        words.gaps.append(f"{path}.label")


def _join_unit(unit: Unit, served: _Json | None, words: _Words, path: str) -> None:
    if unit is None or served is None:
        return
    if isinstance(unit, ComposedUnit):
        for side_name, side in (("numerator", unit.numerator), ("denominator", unit.denominator)):
            _join_unit_side(side, served[side_name], words, f"{path}.{side_name}")
        return
    _join_unit_side(unit, served, words, path)


def _join_unit_side(
    side: SimpleUnit | VariableSide, served: _Json, words: _Words, path: str
) -> None:
    current = _current_unit_label(side, words.vocabulary)
    if not _set_label(served, current or _held(side.label), current=current is not None):
        words.gaps.append(f"{path}.label")


def _join_labels(step: Step, served: _Json, words: _Words) -> None:
    """Join today's label, else the stored one, into every reference a served step carries,
    and name each reference left without one in `gaps`."""
    path = f"steps.{step.id}"
    if isinstance(step.input, VersionRef):
        _join_ref(step.input, served["input"], words, f"{path}.input")
    for role, step_input in _inputs(step).items():
        if isinstance(step_input, VersionRef):
            _join_ref(step_input, served["inputs"][role], words, f"{path}.inputs.{role}")
    if step.choice is not MISSING:
        choice = served["choice"]
        _join_ref(step.choice.chosen, choice["chosen"], words, f"{path}.choice.chosen")
        for index, ref in enumerate(step.choice.others):
            _join_ref(ref, choice["others"][index], words, f"{path}.choice.others.{index}")
    if step.summed is not MISSING:
        variables = served["summed"]["variables"]
        for index, ref in enumerate(step.summed.variables):
            _join_ref(ref, variables[index], words, f"{path}.summed.variables.{index}")
    if step.entity_coverage is not MISSING:
        for index, item in enumerate(step.entity_coverage):
            entity = served["entity_coverage"][index]["entity"]
            _join_ref(item.entity, entity, words, f"{path}.entity_coverage.{index}.entity")
    if step.unit is not MISSING:
        _join_unit(step.unit, served["unit"], words, f"{path}.unit")
    if step.converted_from is not MISSING:
        _join_unit(step.converted_from, served["converted_from"], words, f"{path}.converted_from")


# Names of the variables a computed version writes


def step_output_names(
    steps: Sequence[Step], vocabulary: Vocabulary, *, join: Join = _spaced
) -> dict[str, str]:
    """The name of what each step produces, by step id, worded mid-sentence from its inputs
    and operation as `name_computed` words the last one ("gross ODA disbursements per gross
    national income"), each two-sided name joined by `join`, which a later step's name reads.
    A step whose output has no name is left out."""
    names: dict[str, str] = {}
    for words in _walk(steps, vocabulary, grain=None, family=None, gaps=[], join=join):
        if (name := _output_name(words)) is not None:
            names[words.step.id] = name
    return names


_COMPOSED_JOINERS: Final = {
    "quotient": "per",
    "sum": "plus",
    "difference": "minus",
    "share": "as a share of",
}
_EQUAL_SIDES_PREFIXES: Final = {
    "quotient": "Ratio of ",
    "difference": "Difference in ",
    "share": "Share of ",
}


def composed_names(
    composed: Sequence[ComposedVariable], labels: Mapping[str, str], *, join: Join = _spaced
) -> dict[str, str]:
    """The name of each composed variable an `ops_` body stored, by bare dcid, worded from its
    sides: `<first> per <second>` for a quotient, `<first> plus <second>` for a sum,
    `<first> minus <second>` for a difference, and `<first> as a share of <second>` for a
    share. With equal sides a quotient reads `Ratio of <side>`, a difference
    `Difference in <side>`, a share `Share of <side>` and a sum `<side>`.

    A side reads as its variables' names joined by " and ": an earlier entry's name, else its
    label in `labels` (by bare dcid), else its dcid. `join` joins two unequal sides, and a
    later entry reads an earlier entry's name as joined. A relation outside those four has no
    name."""
    names: dict[str, str] = {}
    for entry in composed:
        joiner = _COMPOSED_JOINERS.get(entry.relation)
        if joiner is None:
            continue
        first = _side_name(entry.first, names=names, labels=labels)
        if entry.first == entry.second:
            names[entry.variable] = _EQUAL_SIDES_PREFIXES.get(entry.relation, "") + first
        else:
            second = _side_name(entry.second, names=names, labels=labels)
            names[entry.variable] = join(first, joiner, second)
    return names


def _side_name(
    variables: Sequence[str], *, names: Mapping[str, str], labels: Mapping[str, str]
) -> str:
    return " and ".join(names.get(ref) or labels.get(ref) or ref for ref in variables)
