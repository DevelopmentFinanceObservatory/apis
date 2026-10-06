"""Closest match: an answer that shows a measure other than the one the request named.

A weak retrieval score alone says only that the match is loose. `closest_match_fact`
raises the departure when the score is weak and the words the request used for the
variable name none of the measure's names: no content word (`search.heading.words`) of
`variable_text` appears in the measure noun the heading shows, among the words every
elected StatVar's name holds, or in the names of each code the answer binds (its
code-index label and synonyms, and the words that name it when it is its system's declared
default, `DeclaredDefault.when_named`, "aid" for ODA). "cost of living" answered with
healthcare expenditure per person raises it. "debt service" answered with debt service on
external debt, by creditor, does not. A word only one member of a family holds names that
member alone. "climate finance" answered with ODA gross disbursements over 356
sectors raises it, though one sector is public finance management.
An answer on the elected shape's measure sibling, read because the elected measure holds
no data, raises it without the test (`substituted_measure_fact`), as does an answer showing
one member of a family whose total holds no data
(`materialise_step.family_total_absent_departure`). A candidates part warned
`elected_shape_no_data`, its readings elected in place of a measure without data, states
the plural one at assembly (`assemble._with_readings_substituted`) when no reading names
the measure asked, each reading judged by the same test (`substituted_readings_fact`).
"GDP of Kenya in current US dollars" answered with Kenya's GDP and its GDP per capita
raises none.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from code_index.shared import get_index
from search.heading.departure_fact import DepartureFact
from search.heading.words import content_words
from search.models import Binding, BindingSet, BindingValue, SlotValue, Spec


def closest_match_fact(
    *,
    variable_text: str,
    retrieval_score_weak: bool,
    measure_noun: str,
    stat_var_names: Iterable[str],
    bound_codes: Iterable[str],
) -> DepartureFact | None:
    """The `closest_match` departure for one reading, or `None` when it shows the
    measure asked for.

    Args:
        variable_text: the request's words for the variable.
        retrieval_score_weak: whether the reading carries `retrieval_score_weak`.
        measure_noun: the measure noun the heading shows.
        stat_var_names: the name of each elected StatVar. A word names the measure when
            every one of them holds it.
        bound_codes: the code dcids the reading binds: its bound how-slot values and
            the elected StatVars' constraint values. Each adds its names
            (`bound_code_names`).

    Returns:
        A fact with `asked` set to `variable_text` and `shown` to `measure_noun`. No slot
        edit clears it, so its `binding_ref` is `None`.
    """
    if not retrieval_score_weak:
        return None
    asked = content_words(variable_text)
    if not asked:
        return None
    names = [measure_noun, *bound_code_names(bound_codes)]
    if any(asked & content_words(name) for name in names) or asked & _shared_words(stat_var_names):
        return None
    return DepartureFact(
        code="closest_match", binding_ref=None, asked=variable_text, shown=measure_noun
    )


def _shared_words(names: Iterable[str]) -> frozenset[str]:
    """The content words every one of `names` holds, none when there are no names."""
    words = [content_words(name) for name in names]
    return frozenset.intersection(*words) if words else frozenset()


def spec_closest_match_fact(
    spec: Spec, *, variable_text: str, retrieval_score_weak: bool, measure_noun: str
) -> DepartureFact | None:
    """`closest_match_fact` for the reading `spec` answers with. Its StatVar names are the
    elected StatVars' labels, the names the Spec holds (`alternateName` is not read at
    assembly), and its bound codes are its how-slot values and the StatVars' constraint
    values, in that order."""
    stat_vars = spec.resolved.members
    codes = [
        value.ref.dcid
        for slot in spec.resolved.slots
        if slot.key.axis == "how"
        for value in _bound_values(slot.binding)
        if value.ref is not None
    ]
    codes.extend(
        slot_value.value.ref.dcid
        for stat_var in stat_vars
        for slot_value in stat_var.slot_values
        if slot_value.value.ref is not None
    )
    return closest_match_fact(
        variable_text=variable_text,
        retrieval_score_weak=retrieval_score_weak,
        measure_noun=measure_noun,
        stat_var_names=[stat_var.ref.label for stat_var in stat_vars if stat_var.ref.label],
        bound_codes=dict.fromkeys(codes),
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
        spec_closest_match_fact(
            spec, variable_text=variable_text, retrieval_score_weak=True, measure_noun=noun
        )
        is None
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
