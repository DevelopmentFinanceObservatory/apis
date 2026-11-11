"""What a reading shows of each declared measure a request names: one verdict per name.

`qualifier_match.declared_measures_named` reads the request's names of declared measures
(measure nouns and partition labels), each with its alternatives, the declarations that
share the name. `measure_verdicts` judges each name on a reading's `ReadingEvidence`,
built by `reading_evidence` from a Spec (`closest_match.reading_inputs`) or from a shape
draft (`admissibility.draft_evidence`). The evidence states the reading's five-tuple and,
for each constraint property, the values its members declare and whether every member
declares one (`PropertyEvidence`). A declared value is related to an alternative's value
when the two are equal or one descends from the other in the code index. The verdicts, in
this order:

1. ``holds``: the reading's five-tuple holds an alternative's five-tuple keys, and on each
   constraint property the alternative keys on every member declares a value and every
   declared value is related to the alternative's; and the reading relates to no declared
   noun outside the name that refines it (potential GDP refines GDP);
2. ``some``: as ``holds``, except that on some keyed property only some declared value is
   related: the members split between related and unrelated values, or some declare none;
3. ``base``: the StatVar the reading divides by holds one;
4. ``contradicts``: some alternative shares the reading's population type, and for every
   such alternative some constraint property it keys on has every member declaring a value
   and no declared value related to its own. An alternative of another population type
   says nothing of the reading, so "GDP", which names WEO's GDP (`EconomicProduction`) and
   WDI's (`EconomicActivity`), contradicts a WDI series of gross national income. A name
   whose alternative the denominator's name names (step 6) is ``base`` instead, so a
   share of GDP whose numerator contradicts "GDP" divides by it;
5. ``holds`` from names: the reading relates to no declared measure at all, and its measure
   noun, or every one of its member names, names an alternative (`declared_measures_named`
   again, so "GDP growth (annual %)" names "GDP growth" and not "GDP");
6. ``base`` from names: its denominator relates to no declared measure, and the
   denominator's name names an alternative;
7. ``unknown`` otherwise.

The evidence decides wherever it relates to a declared measure, so "GDP" answered with a
reading whose values hold the noun "GDP growth" is ``unknown`` whatever its names hold,
while a series the graph identifies by name alone (a UN or WHO ratio) is judged on its
names. Admissibility rule 4 refuses a shape on which some name's verdict is
``contradicts``, before bind. Closest match departs a reading on which some name's verdict
is neither ``holds`` nor ``base``, at assembly, so a reading whose members split departs on
``some``. Per-property evidence cannot see an alternative keyed on two properties each
declared by different members: where every member declares a related value on each
property, such a reading reads ``holds``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureIndex, MeasurePartitionLabel
from search.engine.qualifier_match import NamedMeasure, declared_measures_named

type Verdict = Literal["holds", "some", "base", "contradicts", "unknown"]

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
class PropertyEvidence:
    """What a reading's members declare on one constraint property: the `values` some
    member declares, and whether `every_member` declares one. A value known only from a
    summary of the members, which does not say which members declare it, adds to `values`
    and never sets `every_member`."""

    values: frozenset[str]
    every_member: bool


@dataclass(frozen=True, slots=True)
class ReadingEvidence:
    """What one reading states of the measure it shows.

    `five_tuple` are its five-tuple pairs (`shape_pairs`); `properties` what its members
    declare on each constraint property (`PropertyEvidence`); `noun` its measure noun and
    `member_names` its members' names; `denominator_pairs` the five-tuple pairs and
    constraint values of the StatVar it divides by, empty where unknown, and
    `denominator_name` that StatVar's name."""

    five_tuple: Pairs = frozenset()
    properties: Mapping[str, PropertyEvidence] = field(default_factory=dict)
    noun: str | None = None
    member_names: tuple[str, ...] = ()
    denominator_pairs: Pairs = frozenset()
    denominator_name: str | None = None


def reading_evidence(
    *,
    population_type: str | None,
    measured_property: str | None,
    stat_type: str | None,
    properties: Mapping[str, PropertyEvidence],
    noun: str | None,
    member_names: Sequence[str] = (),
    denominator_pairs: Iterable[tuple[str, str]] = (),
    denominator_name: str | None = None,
) -> ReadingEvidence:
    """The evidence of a reading of the five-tuple `population_type`, `measured_property`
    and `stat_type` whose members declare `properties`. The one builder both judges read,
    so rule 4 and closest match judge the same evidence alike."""
    return ReadingEvidence(
        five_tuple=frozenset(shape_pairs(population_type, measured_property, stat_type)),
        properties=dict(properties),
        noun=noun,
        member_names=tuple(member_names),
        denominator_pairs=frozenset(denominator_pairs),
        denominator_name=denominator_name,
    )


def property_evidence(
    members: Sequence[Mapping[str, str]], *, summary: Iterable[tuple[str, str]] = ()
) -> dict[str, PropertyEvidence]:
    """The evidence on each constraint property of a reading whose members declare
    `members`, one `{property: value}` mapping per member, beside the (property, value)
    pairs a `summary` of its members holds. A property every one of `members` declares has
    `every_member`; a reading with no known members has it on no property, and a summary
    value never sets it."""
    values: dict[str, set[str]] = {}
    for prop, value in [*(pair for member in members for pair in member.items()), *summary]:
        values.setdefault(prop, set()).add(value)
    return {
        prop: PropertyEvidence(
            values=frozenset(found),
            every_member=bool(members) and all(prop in member for member in members),
        )
        for prop, found in values.items()
    }


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
    verdicts = [_by_evidence(name.alternatives, evidence, measures.nouns, index) for name in names]
    if "unknown" not in verdicts and "contradicts" not in verdicts:
        return verdicts

    def declared(text: str | None) -> set[MeasurePartitionLabel]:
        named = declared_measures_named(text or "")
        return {alternative for name in named for alternative in name.alternatives}

    shown: set[MeasurePartitionLabel] = set()
    if not _identifies(_Stated(evidence.five_tuple, evidence.properties), measures, index):
        members = [declared(name) for name in evidence.member_names]
        shown = declared(evidence.noun) | (set.intersection(*members) if members else set())
    based = (
        set()
        if _identifies(_statvar(evidence.denominator_pairs), measures, index)
        else declared(evidence.denominator_name)
    )
    return [
        _by_names(verdict, name.alternatives, shown, based)
        for verdict, name in zip(verdicts, names, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class _Stated:
    """A reading's five-tuple and property evidence, or a StatVar's (`_statvar`)."""

    five_tuple: Pairs
    properties: Mapping[str, PropertyEvidence]


def _statvar(pairs: Pairs) -> _Stated:
    """One StatVar whose pairs are `pairs`: its one member declares every constraint value
    it holds."""
    five_tuple = frozenset(pair for pair in pairs if pair[0] in _FIVE_TUPLE_KEYS)
    values: dict[str, set[str]] = {}
    for prop, value in pairs - five_tuple:
        values.setdefault(prop, set()).add(value)
    return _Stated(
        five_tuple,
        {
            prop: PropertyEvidence(values=frozenset(found), every_member=True)
            for prop, found in values.items()
        },
    )


def _by_names(
    verdict: Verdict,
    alternatives: Sequence[MeasurePartitionLabel],
    shown: set[MeasurePartitionLabel],
    based: set[MeasurePartitionLabel],
) -> Verdict:
    """The verdict `verdict` from the evidence with the names read: a ``contradicts`` is
    ``base`` where the denominator's name names an alternative (`based`); an ``unknown`` is
    ``holds`` where the reading's names name one (`shown`), ``base`` where the denominator's
    name does, and ``unknown`` otherwise."""
    if verdict == "contradicts":
        return "base" if based.intersection(alternatives) else verdict
    if verdict != "unknown":
        return verdict
    if shown.intersection(alternatives):
        return "holds"
    if based.intersection(alternatives):
        return "base"
    return "unknown"


def _by_evidence(
    alternatives: Sequence[MeasurePartitionLabel],
    evidence: ReadingEvidence,
    nouns: Sequence[MeasurePartitionLabel],
    index: CodeIndex,
) -> Verdict:
    """The verdict the reading's evidence and its denominator's pairs give: steps 1 to 4."""
    stated = _Stated(evidence.five_tuple, evidence.properties)
    if _held(alternatives, stated, nouns, index, every=True):
        return "holds"
    if _held(alternatives, stated, nouns, index, every=False):
        return "some"
    if _held(alternatives, _statvar(evidence.denominator_pairs), nouns, index, every=True):
        return "base"
    comparable = [
        alt
        for alt in alternatives
        if ("populationType", dict(alt.axis_values).get("populationType")) in evidence.five_tuple
    ]
    if comparable and all(_contradicts(alt.axis_values, stated, index) for alt in comparable):
        return "contradicts"
    return "unknown"


def _held(
    alternatives: Sequence[MeasurePartitionLabel],
    stated: _Stated,
    nouns: Sequence[MeasurePartitionLabel],
    index: CodeIndex,
    *,
    every: bool,
) -> bool:
    """Whether `stated` holds one of `alternatives` (`_holds`, on every member where `every`
    is true, on some otherwise) and relates to no declared noun outside them that refines
    it, one whose `axis_values` strictly contain the alternative's: a reading of potential
    GDP holds no "GDP". Partition labels refine nothing here, so "disbursements" answered
    with gross disbursements holds."""
    return any(
        _holds(alt.axis_values, stated, index, every=every)
        and not any(
            noun not in alternatives
            and set(alt.axis_values) < set(noun.axis_values)
            and _holds(noun.axis_values, stated, index, every=False)
            for noun in nouns
        )
        for alt in alternatives
    )


def _holds(
    axis_values: Iterable[tuple[str, str]], stated: _Stated, index: CodeIndex, *, every: bool
) -> bool:
    """Whether `stated` holds every one of `axis_values`: a five-tuple key's value equal, and
    on a constraint property, with `every`, every member declaring a value and every declared
    value related (`_related`), else some declared value related."""
    if not (stated.five_tuple or stated.properties):
        return False
    for axis, value in axis_values:
        if axis in _FIVE_TUPLE_KEYS:
            if (axis, value) not in stated.five_tuple:
                return False
            continue
        found = stated.properties.get(axis)
        if found is None:
            return False
        related = [_related(held, value, index) for held in found.values]
        if not (found.every_member and all(related) if every else any(related)):
            return False
    return True


def _contradicts(axis_values: Iterable[tuple[str, str]], stated: _Stated, index: CodeIndex) -> bool:
    """Whether the reading, of the population type of the declared measure keyed on
    `axis_values`, measures something else: on some constraint property the measure keys
    on every member declares a value and none of the values declared there is related to
    the measure's. A member silent on the property contradicts nothing, and a reading whose
    members are not known contradicts nothing."""
    return any(
        (found := stated.properties.get(prop)) is not None
        and found.every_member
        and not any(_related(held, value, index) for held in found.values)
        for prop, value in dict(axis_values).items()
        if prop not in _FIVE_TUPLE_KEYS
    )


def _identifies(stated: _Stated, measures: MeasureIndex, index: CodeIndex) -> bool:
    """Whether `stated` relates to some declared measure: holds it on some member."""
    return any(
        _holds(measure.axis_values, stated, index, every=False)
        for measure in (*measures.partition_labels, *measures.nouns)
    )


def _related(a: str, b: str, index: CodeIndex) -> bool:
    """Whether codes `a` and `b` are one code, or one descends from the other in `index`."""
    return a in index.descendants(b) or b in index.descendants(a)
