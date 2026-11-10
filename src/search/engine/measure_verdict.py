"""What a reading shows of each declared measure a request names: one verdict per name.

`qualifier_match.declared_measures_named` reads the request's names of declared measures
(measure nouns and partition labels), each with its alternatives, the declarations that
share the name. `measure_verdicts` judges each name on a reading's `ReadingEvidence`,
built by `reading_evidence` from a Spec (`closest_match.reading_inputs`) or from a shape
draft (`admissibility.draft_evidence`), in this order:

1. ``holds``: the reading's pairs hold an alternative's `axis_values`, each five-tuple key
   equal and each constraint value equal to the alternative's or related to it by ancestry
   in the code index, and hold no declared noun outside the name that refines it
   (potential GDP refines GDP);
2. ``base``: the pairs of the StatVar the reading divides by hold one;
3. ``contradicts``: some alternative shares the reading's population type, and for every
   such alternative the reading's members each carry a value on a constraint property it
   keys on, none of the values the reading holds there related to its own. An alternative
   of another population type says nothing of the reading, so "GDP", which names WEO's
   GDP (`EconomicProduction`) and WDI's (`EconomicActivity`), contradicts a WDI series of
   gross national income;
4. ``holds`` from names: the reading's pairs hold no declared measure at all, and its
   measure noun, or every one of its member names, names an alternative
   (`declared_measures_named` again, so "GDP growth (annual %)" names "GDP growth" and
   not "GDP");
5. ``base`` from names: its denominator's pairs hold no declared measure, and the
   denominator's name names an alternative;
6. ``unknown`` otherwise.

Pairs decide wherever they identify a declared measure, so "GDP" answered with a reading
whose pairs hold the noun "GDP growth" is ``unknown`` whatever its names hold, while a
series the graph identifies by name alone (a UN or WHO ratio) is judged on its names.
Admissibility rule 4 refuses a shape on which some name's verdict is ``contradicts``,
before bind; closest match departs a reading on which some name's verdict is neither
``holds`` nor ``base``, at assembly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureIndex, MeasurePartitionLabel
from search.engine.qualifier_match import NamedMeasure, declared_measures_named

type Verdict = Literal["holds", "base", "contradicts", "unknown"]

type Pairs = frozenset[tuple[str, str]]

# The five-tuple keys of a declared measure's `axis_values`; its other pairs are
# constraint values.
_FIVE_TUPLE_KEYS = frozenset(
    {
        "populationType",
        "measuredProperty",
        "statType",
        "measurementQualifier",
        "measurementDenominator",
    }
)


@dataclass(frozen=True, slots=True)
class ReadingEvidence:
    """What one reading states of the measure it shows.

    `pairs` are its five-tuple pairs and the constraint values its members hold;
    `member_properties` the constraint properties each member states, empty where the
    members' constraints are not known; `noun` its measure noun and `member_names` its
    members' names; `denominator_pairs` the five-tuple pairs and constraint values of the
    StatVar it divides by, empty where unknown, and `denominator_name` that StatVar's
    name."""

    pairs: Pairs = frozenset()
    member_properties: tuple[frozenset[str], ...] = ()
    noun: str | None = None
    member_names: tuple[str, ...] = ()
    denominator_pairs: Pairs = frozenset()
    denominator_name: str | None = None


def reading_evidence(
    *,
    population_type: str | None,
    measured_property: str | None,
    stat_type: str | None,
    constraint_values: Iterable[tuple[str, str]],
    noun: str | None,
    member_names: Sequence[str] = (),
    member_properties: Sequence[frozenset[str]] = (),
    denominator_pairs: Iterable[tuple[str, str]] = (),
    denominator_name: str | None = None,
) -> ReadingEvidence:
    """The evidence of a reading of the five-tuple `population_type`, `measured_property`
    and `stat_type` whose members hold `constraint_values`, as (property, value) pairs: its
    pairs are the five-tuple's (`shape_pairs`) and those values. The one builder both
    judges read, so a value the members hold reaches rule 4 and closest match alike."""
    return ReadingEvidence(
        pairs=frozenset(
            [*shape_pairs(population_type, measured_property, stat_type), *constraint_values]
        ),
        member_properties=tuple(member_properties),
        noun=noun,
        member_names=tuple(member_names),
        denominator_pairs=frozenset(denominator_pairs),
        denominator_name=denominator_name,
    )


def shape_pairs(
    population_type: str | None, measured_property: str | None, stat_type: str | None
) -> tuple[tuple[str, str], ...]:
    """The five-tuple pairs a shape or StatVar states, as a declared measure's
    `axis_values` key them, leaving out a missing value."""
    return tuple(
        (key, value)
        for key, value in (
            ("populationType", population_type),
            ("measuredProperty", measured_property),
            ("statType", stat_type),
        )
        if value
    )


def measure_verdicts(names: Sequence[NamedMeasure], evidence: ReadingEvidence) -> list[Verdict]:
    """The verdict on each of `names`, in order, for the reading `evidence` states. A name
    with several alternatives is contradicted only when every alternative of the reading's
    population type is."""
    index = get_index()
    measures = get_measure_index()
    verdicts = [_by_pairs(name.alternatives, evidence, measures.nouns, index) for name in names]
    if "unknown" not in verdicts:
        return verdicts

    def declared(text: str | None) -> set[MeasurePartitionLabel]:
        named = declared_measures_named(text or "")
        return {alternative for name in named for alternative in name.alternatives}

    shown: set[MeasurePartitionLabel] = set()
    if not _identifies(evidence.pairs, measures, index):
        members = [declared(name) for name in evidence.member_names]
        shown = declared(evidence.noun) | (set.intersection(*members) if members else set())
    based = (
        set()
        if _identifies(evidence.denominator_pairs, measures, index)
        else declared(evidence.denominator_name)
    )
    return [
        verdict if verdict != "unknown" else _by_names(name.alternatives, shown, based)
        for verdict, name in zip(verdicts, names, strict=True)
    ]


def _by_names(
    alternatives: Sequence[MeasurePartitionLabel],
    shown: set[MeasurePartitionLabel],
    based: set[MeasurePartitionLabel],
) -> Verdict:
    """Steps 4 to 6: ``holds`` when the reading's names name an alternative (`shown`),
    ``base`` when its denominator's name does (`based`), ``unknown`` otherwise."""
    if shown.intersection(alternatives):
        return "holds"
    if based.intersection(alternatives):
        return "base"
    return "unknown"


def _by_pairs(
    alternatives: Sequence[MeasurePartitionLabel],
    evidence: ReadingEvidence,
    nouns: Sequence[MeasurePartitionLabel],
    index: CodeIndex,
) -> Verdict:
    """The verdict the reading's pairs and its denominator's pairs give: steps 1 to 3."""
    if _held(alternatives, evidence.pairs, nouns, index):
        return "holds"
    if _held(alternatives, evidence.denominator_pairs, nouns, index):
        return "base"
    comparable = [
        alt
        for alt in alternatives
        if ("populationType", dict(alt.axis_values).get("populationType")) in evidence.pairs
    ]
    if comparable and all(_contradicts(alt.axis_values, evidence, index) for alt in comparable):
        return "contradicts"
    return "unknown"


def _held(
    alternatives: Sequence[MeasurePartitionLabel],
    pairs: Pairs,
    nouns: Sequence[MeasurePartitionLabel],
    index: CodeIndex,
) -> bool:
    """Whether `pairs` hold one of `alternatives` (`_holds`) and no declared noun outside
    them that refines it, one whose `axis_values` strictly contain the alternative's: a
    reading of potential GDP holds no "GDP". Partition labels refine nothing here, so
    "disbursements" answered with gross disbursements holds."""
    return any(
        _holds(alt.axis_values, pairs, index)
        and not any(
            noun not in alternatives
            and set(alt.axis_values) < set(noun.axis_values)
            and _holds(noun.axis_values, pairs, index)
            for noun in nouns
        )
        for alt in alternatives
    )


def _holds(axis_values: Iterable[tuple[str, str]], pairs: Pairs, index: CodeIndex) -> bool:
    """Whether `pairs` hold every one of `axis_values`: a five-tuple key's value equal, a
    constraint value equal or related by ancestry (`_related`)."""
    return bool(pairs) and all(
        (axis, value) in pairs
        if axis in _FIVE_TUPLE_KEYS
        else any(held_axis == axis and _related(held, value, index) for held_axis, held in pairs)
        for axis, value in axis_values
    )


def _contradicts(
    axis_values: Iterable[tuple[str, str]], evidence: ReadingEvidence, index: CodeIndex
) -> bool:
    """Whether the reading, of the population type of the declared measure keyed on
    `axis_values`, measures something else: on some constraint property the measure keys
    on every member carries a value and none of the values the reading holds there is
    related to the measure's. A member silent on the property contradicts nothing, and a
    reading whose members are not known contradicts nothing."""
    keyed = dict(axis_values)
    if not evidence.member_properties:
        return False
    return any(
        all(prop in member for member in evidence.member_properties)
        and not any(
            held_axis == prop and _related(held, value, index) for held_axis, held in evidence.pairs
        )
        for prop, value in keyed.items()
        if prop not in _FIVE_TUPLE_KEYS
    )


def _identifies(pairs: Pairs, measures: MeasureIndex, index: CodeIndex) -> bool:
    """Whether `pairs` hold some declared measure's `axis_values`."""
    return any(
        _holds(measure.axis_values, pairs, index)
        for measure in (*measures.partition_labels, *measures.nouns)
    )


def _related(a: str, b: str, index: CodeIndex) -> bool:
    """Whether codes `a` and `b` are one code, or one descends from the other in `index`."""
    return a in index.descendants(b) or b in index.descendants(a)
