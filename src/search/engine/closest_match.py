"""Closest match: an answer that shows a measure other than the one the request named.

`closest_match_fact` raises the departure at any retrieval score. A publisher series code
the words hold that stands for the answer (`SourceCode.exact`) clears it.

When the request's words for the variable name declared measures
(`qualifier_match.declared_measures_named`: a label, `also_called` entry or declared acronym,
whole, the longest declared name winning its words), it raises it when some name's verdict
(`measure_verdict.measure_verdicts`) is neither holds nor base. The reading's shape and bound
values holding the measure show it ("gross disbursements" answered with gross ODA
disbursements). The name of the StatVar it divides by names its base, so "GDP" answered
with any share of GDP raises none. Its noun or every StatVar name show a measure only where
its pairs hold no declared measure, so "GDP" answered with GDP growth raises it.

When the words name no declared measure, it raises it unless a content word
(`search.heading.words`) of them appears in the heading's measure noun, among the words
every StatVar name holds, or in a name of a code the answer binds (its label, synonyms and
`DeclaredDefault.when_named`), or the words hold a bound code's `CodeRow.also_called` name
whole ("aid" for ODA). A declared acronym whose expansion names no declared measure ("PPG")
adds the expansion's content words. "cost of living" answered with healthcare expenditure
per person raises it.

An answer the engine substituted, its elected measure holding no data, raises it without
the test (`substituted_measure_fact`, `materialise_step.family_total_absent_departure`). A
candidates part whose readings stand in for such a measure states the plural one at
assembly unless one reading passes the test (`substituted_readings_fact`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from code_index.shared import get_index, get_measure_index
from search.engine.measure_verdict import ReadingEvidence, measure_verdicts, shape_pairs
from search.engine.qualifier_match import declared_measures_named, holds_words
from search.heading.departure_fact import DepartureFact
from search.heading.words import content_words
from search.models import Binding, BindingSet, BindingValue, SlotValue, Spec


def closest_match_fact(
    *,
    variable_text: str,
    evidence: ReadingEvidence,
    bound_codes: Iterable[str] = (),
    series_codes: Iterable[str] = (),
) -> DepartureFact | None:
    """The `closest_match` departure for one reading, or `None` when it shows the
    measure asked for.

    Args:
        variable_text: the request's words for the variable.
        evidence: what the reading states of its measure (`reading_inputs`), its `noun`
            the measure noun the heading shows.
        bound_codes: the code dcids the reading binds on its what and how slots
            (`reading_inputs`). Each adds its names
            (`bound_code_names`) and its `CodeRow.also_called` names.
        series_codes: the publisher series codes that stand for the reading
            (`SourceCode.exact`), each a whole name of its measure.

    Returns:
        A fact with `asked` set to `variable_text` and `shown` to the evidence's noun. No
        slot edit clears it, so its `binding_ref` is `None`.
    """
    asked = content_words(variable_text)
    if not asked or any(holds_words(variable_text, code) for code in series_codes):
        return None
    noun = evidence.noun or ""
    fact = DepartureFact(code="closest_match", binding_ref=None, asked=variable_text, shown=noun)
    bound_codes = tuple(bound_codes)
    index = get_index()
    named = declared_measures_named(variable_text, get_measure_index(), index.variable_aliases)
    if named:
        verdicts = measure_verdicts(named, evidence)
        return None if all(verdict in ("holds", "base") for verdict in verdicts) else fact
    asked |= {
        word
        for acronym, expansion in index.variable_aliases.items()
        if holds_words(variable_text, acronym)
        for word in content_words(expansion)
    }
    if (
        any(asked & content_words(name) for name in (noun, *bound_code_names(bound_codes)))
        or asked & _shared_words(evidence.member_names)
        or _names_bound_code(variable_text, bound_codes)
    ):
        return None
    return fact


def _shared_words(names: Iterable[str]) -> frozenset[str]:
    """The content words every one of `names` holds, none when there are no names."""
    words = [content_words(name) for name in names]
    return frozenset.intersection(*words) if words else frozenset()


def _names_bound_code(variable_text: str, bound_codes: Sequence[str]) -> bool:
    """Whether `variable_text` holds an `also_called` name of a bound code whole."""
    index = get_index()
    return any(
        holds_words(variable_text, name)
        for dcid in bound_codes
        if (row := index.code_by_dcid(dcid)) is not None
        for name in row.also_called
    )


def reading_inputs(
    spec: Spec, *, measure_noun: str
) -> tuple[ReadingEvidence, tuple[str, ...], tuple[str, ...]]:
    """The rule's inputs for the reading `spec` answers with, whose heading shows
    `measure_noun`, from what a Spec states whether assembly built it or a recorded response
    rebuilt it with its sentence facts: its evidence, its bound codes and its series codes.

    Its member names are the served members' labels (`ResolvedSpec.sentence_facts`). Its
    bound codes are the values bound on its what and how slots, in slot order: every
    constraint value a member asserts on a bound slot is one of them
    (`spec_build.build_stat_vars`). Its pairs are its shape's (`measure_verdict.shape_pairs`)
    and each bound value under its slot's constraint property. Its series codes are the
    publisher codes that stand for it (`ResolvedStatVars.source_codes`, `exact`). A Spec
    states no member constraints and no denominator pairs, only the denominator's name."""
    resolved = spec.resolved
    bound = [
        (slot.key.property.ref if slot.key.property is not None else None, value.ref.ref)
        for slot in resolved.slots
        if slot.key.axis in ("what", "how")
        for value in _bound_values(slot.binding)
        if value.ref is not None
    ]
    shape = resolved.shape
    denominator = shape.measurement_denominator
    evidence = ReadingEvidence(
        pairs=frozenset(
            [
                *shape_pairs(
                    *(
                        ref.ref if ref is not None else None
                        for ref in (shape.population_type, shape.measured_property, shape.stat_type)
                    )
                ),
                *((prop, code) for prop, code in bound if prop is not None),
            ]
        ),
        noun=measure_noun,
        member_names=tuple(m.label for m in resolved.sentence_facts.members if m.label),
        denominator_name=denominator.label if denominator is not None else None,
    )
    bound_codes = tuple(dict.fromkeys(code for _property, code in bound))
    series_codes = tuple(code.code for code in resolved.stat_vars.source_codes if code.exact)
    return evidence, bound_codes, series_codes


def spec_closest_match_fact(
    spec: Spec, *, variable_text: str, measure_noun: str
) -> DepartureFact | None:
    """`closest_match_fact` for the reading `spec` answers with, on its `reading_inputs`."""
    evidence, bound_codes, series_codes = reading_inputs(spec, measure_noun=measure_noun)
    return closest_match_fact(
        variable_text=variable_text,
        evidence=evidence,
        bound_codes=bound_codes,
        series_codes=series_codes,
    )


def _bound_values(binding: Binding) -> list[SlotValue]:
    """The values a slot binding holds: one for a value, each for a set, none otherwise."""
    if isinstance(binding, BindingValue):
        return [binding.value]
    if isinstance(binding, BindingSet):
        return list(binding.values)
    return []


def substituted_measure_fact(*, variable_text: str | None, measure_noun: str) -> DepartureFact:
    """The `closest_match` departure of an answer on a measure other than the one elected,
    because the elected measure holds no data: its measure sibling, or a member of a family
    whose total holds none. `measure_noun` is the measure noun the answer's heading shows,
    and `variable_text` the request's words for the measure, `None` on a door that holds
    none (a resubmit). The engine made the substitution, so it needs no lexical test."""
    return DepartureFact(
        code="closest_match", binding_ref=None, asked=variable_text, shown=measure_noun
    )


def substituted_readings_fact(
    *, variable_text: str, readings: Iterable[tuple[Spec, str]]
) -> DepartureFact | None:
    """The `closest_match` departure of a candidates answer offered because the elected
    measure holds no data, or `None` when one of its readings shows the measure asked.

    Args:
        variable_text: the request's words for the variable.
        readings: each reading the part shows, with the measure noun its heading shows.
            A reading names the measure asked when `spec_closest_match_fact` finds the
            request's words among its names.

    Returns:
        A fact with `asked` set to `variable_text` and `readings` set, stating that the
        readings stand in for the measure asked.
    """
    if any(
        spec_closest_match_fact(spec, variable_text=variable_text, measure_noun=noun) is None
        for spec, noun in readings
    ):
        return None
    return DepartureFact(code="closest_match", binding_ref=None, asked=variable_text, readings=True)


def bound_code_names(bound_codes: Iterable[str]) -> Iterator[str]:
    """The names of each code in `bound_codes`: its code-index label and synonyms, then,
    when it is its code system's declared default, the words that name that default
    (`DeclaredDefault.when_named`). Codes the code index does not hold add no names."""
    index = get_index()
    for dcid in bound_codes:
        row = index.code_by_dcid(dcid)
        if row is None:
            continue
        yield row.label
        yield from row.synonyms
        declaration = index.declarations.get(row.code_system)
        default = declaration.default if declaration is not None else None
        if default is not None and default.code == row.dcid:
            yield from default.when_named
