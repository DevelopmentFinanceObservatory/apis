"""The sentence each caveat reads as.

`caveat_text` words a served caveat, one template per kind:

- `estimated` and `provisional` name the count of values and what they cover: "7 values of
  infectious disease control in 2024 are provisional."
- `break_in_series` names where the series break: "A break in series of infectious disease
  control in 2015."
- `coverage_gap` that points at a step words what that step left out, with the party noun and
  reason phrases of the step's own sentence: "16 recipients, among them South Sudan and
  Somalia, left out of the division: 9 with no gross national income figure, 5 sharing no year
  with it and 2 with zero gross national income." Series it names lead: "Tuberculosis grant
  disbursements for Kenya left out of the selection in 2022."
- `coverage_gap` without a step names who holds no data: "South Sudan and Somalia have no data
  in 2024."

Every noun comes from data: series and entities by the labels the caveat is served with,
periods by the version's grain, and parties by the role wording of the step's parameters, as
`steps_text` words them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from description.steps import Step
from description.steps_text import counted, left_out_of, series_text
from description.vocabulary import Vocabulary
from dfo_platform.roles import FlowFamily

NAMED: Final = 3
"""How many series, entities or periods a sentence names before counting the rest."""


def caveat_text(
    caveat: Mapping[str, Any],
    *,
    steps: Sequence[Step],
    vocabulary: Vocabulary,
    grain: str | None,
    family: FlowFamily | None,
) -> str:
    """The sentence of a served caveat.

    Args:
        caveat: the caveat as served, its references labelled.
        steps: the steps the caveat's `step` may point at.
        vocabulary: today's words.
        grain: the version's period grain ("year"), which names a count of periods.
        family: the flow family whose role wording names a step's parties.
    """
    kind = caveat["kind"]
    series = [ref["label"] for ref in caveat.get("series", [])]
    entities = [ref["label"] for ref in caveat.get("entities", [])]
    periods = _periods(caveat.get("periods", []), grain)
    count = caveat.get("count")
    if kind in ("estimated", "provisional"):
        values = "Values" if count is None else counted(count, "value", "values")
        verb = "is" if count == 1 else "are"
        scope = _of(series) + _for(entities, "entities") + periods
        return f"{values}{scope} {verb} {kind}."
    if kind == "break_in_series":
        affecting = "" if count is None else f", affecting {counted(count, 'value', 'values')}"
        return f"A break in series{_of(series)}{_for(entities, 'entities')}{periods}{affecting}."
    if (step_id := caveat.get("step")) is not None:
        return _left_out(
            step_id,
            series,
            entities,
            count,
            periods,
            steps=steps,
            vocabulary=vocabulary,
            grain=grain,
            family=family,
        )
    scope = _for(series, "series") + periods
    if entities:
        total = _total(entities, count)
        verb = "has" if total == 1 else "have"
        return _capitalised(f"{_parties(entities, total, 'entities')} {verb} no data{scope}.")
    if count is not None:
        return f"{counted(count, 'value', 'values')} missing{scope}."
    return f"No data{scope}."


def _left_out(
    step_id: str,
    series: Sequence[str],
    entities: Sequence[str],
    count: int | None,
    periods: str,
    *,
    steps: Sequence[Step],
    vocabulary: Vocabulary,
    grain: str | None,
    family: FlowFamily | None,
) -> str:
    """What the step left out: the series, for the entities. The step's reasons follow when
    their counts add up to the caveat's; a step no id matches is named by its id."""
    step = left_out_of(steps, step_id, vocabulary, grain=grain, family=family)
    operation = step.operation if step is not None else f"step {step_id}"
    reasons = step.reasons if step is not None else []
    total = _total(entities, count) or sum(reason_count for reason_count, _ in reasons)
    named = _named(series, "series") if series else ""
    if not total:
        return _capitalised(f"{named or 'Data'} left out of {operation}{periods}.")
    if step is None:
        noun = "entity" if total == 1 else "entities"
    else:
        noun = step.singular if total == 1 else step.plural
    subject = _parties(entities, total, noun)
    if named:
        subject = f"{named} for {subject}"
    if sum(reason_count for reason_count, _ in reasons) != total:
        why = ""
    elif len(reasons) == 1:
        why = f" {reasons[0][1]}"
    else:
        why = ": " + series_text([f"{reason_count} {phrase}" for reason_count, phrase in reasons])
    return _capitalised(f"{subject} left out of {operation}{periods}{why}.")


def _total(entities: Sequence[str], count: int | None) -> int:
    """How many entities a caveat applies to: its count, or the entities it names."""
    return max(count or 0, len(entities))


def _parties(entities: Sequence[str], total: int, noun: str) -> str:
    """The entities by name when the caveat names them all, up to `NAMED`; else counted,
    with the first named: "16 recipients, among them South Sudan and Somalia,"."""
    if entities and total == len(entities) and total <= NAMED:
        return series_text(entities)
    parties = f"{total} {noun}"
    if not entities:
        return parties
    return f"{parties}, among them {series_text(entities[:NAMED])},"


def _of(series: Sequence[str]) -> str:
    return f" of {_named(series, 'series')}" if series else ""


def _for(labels: Sequence[str], noun: str) -> str:
    return f" for {_named(labels, noun)}" if labels else ""


def _periods(periods: Sequence[str], grain: str | None) -> str:
    noun = f"{grain or 'period'}s"
    return f" in {_named(periods, noun)}" if periods else ""


def _named(labels: Sequence[str], noun: str) -> str:
    """`NAMED` labels, then how many more: "2010, 2011, 2012 and 12 more years". One label past
    `NAMED` is named itself."""
    if len(labels) <= NAMED + 1:
        return series_text(labels)
    return series_text([*labels[:NAMED], f"{len(labels) - NAMED} more {noun}"])


def _capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]
