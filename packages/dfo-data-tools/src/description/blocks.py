"""The blocks of a description, worded from a vocabulary over a record's facts and evidence.

Each builder returns JSON-ready wire data. `Words` carries the vocabulary and the labels the
record stored as evidence, and collects the dotted path of every member a builder could not
fill. A member left `null` with its path in the gaps is unknown; one left `null` without a
gap is one the data declares none of.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final
from urllib.parse import quote

from dfo_frame_grammar import STAT_VAR_COLUMN
from pydantic.experimental.missing_sentinel import MISSING

from code_index.query import CodeIndex
from description.caveats_text import caveat_text
from description.facts import (
    CaveatFact,
    ComposedVariable,
    DescriptionRecord,
    DimensionFacts,
    DrawnSeries,
    InputVersion,
    SelectionFacts,
    SeriesFacts,
    SourceEvidence,
    Span,
    StatusCounts,
    VariableFacts,
)
from description.steps import ComposedUnit, SimpleUnit, Step, Unit, VariableSide, VersionRef
from description.vocabulary import MeasureWords, Vocabulary
from description.world_bank import world_bank_definition
from dfo_platform.roles import FlowFamily, axis_for_role
from unit_words.currency_nouns import CURRENCY_NOUNS

type Wire = dict[str, Any]
"""A JSON object as served."""

TOP_VALUES = 2
"""How many values an open dimension leads with."""

CAVEAT_ENTITIES_MAX = 50
"""The most entities a served caveat names, as the description schema bounds them."""
CAVEATS_MAX = 50
"""The most caveats a description serves, the description schema's bound."""
DRAWN_VALUES_MAX = 50
"""The most values a drawn dimension lists, the description schema's bound."""
SOURCE_CODES_MAX = 50
"""The most publisher codes a `source_codes` list holds, the description schema's bound."""
DRAWN_VALUES_BUDGET_BYTES = 256 * 1024
"""The most bytes the drawn dimensions' `values` take together, as compact UTF-8 JSON. A
card's envelope holds at most `assembler.limits.MAX_ENVELOPE_BYTES` (2 MiB). The schema lets
the values reach about 6.5 million definition characters, while a CRS card drawing 24
series across five dimensions, each value defined by the code index's longest definition
(940 characters), takes about 130 KB. The budget doubles that and leaves 1.75 MiB for the
component, whose largest table takes about 1.24 MB, and the rest of the description."""

_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_PARTIAL_DATE = re.compile(r"^[0-9]{4}(-[0-9]{2}(-[0-9]{2})?)?$")


@dataclass(slots=True)
class Words:
    """The words of one description, and the paths of the members it could not fill.

    A label is today's, from `vocabulary`; else the one stored with the fact or as evidence,
    marked `at_mint`; else the dcid itself, with a gap naming the label. `family` words the
    two parties of a flow.
    """

    vocabulary: Vocabulary
    evidence: Mapping[str, str]
    family: FlowFamily | None = None
    gaps: list[str] = field(default_factory=list)

    @classmethod
    def for_record(cls, record: DescriptionRecord, vocabulary: Vocabulary) -> Words:
        """The words of `record`'s description: its evidence labels and flow family."""
        return cls(
            vocabulary=vocabulary,
            evidence={label.ref: label.label for label in record.evidence.labels or []},
            family=record.variables.family if record.variables is not None else None,
        )

    def gap(self, path: str) -> None:
        """Name `path` as a member that could not be filled."""
        if path not in self.gaps:
            self.gaps.append(path)

    def known[T](self, value: T | None, path: str) -> T | None:
        """`value`, with a gap at `path` when it is `None`."""
        if value is None:
            self.gap(path)
        return value

    def name(self, dcid: str, stored: str | None = None) -> str | None:
        """Today's label of `dcid`, else `stored` or its evidence label, else `None`. Records
        no gap."""
        return self.vocabulary.label(dcid) or stored or self.evidence.get(dcid)

    def label(self, dcid: str, path: str, stored: str | None = None) -> Wire:
        """The `label` member, and `at_mint` where it is not today's, of the object at `path`
        that names `dcid`. `stored` is a label stored with the fact itself."""
        if (today := self.vocabulary.label(dcid)) is not None:
            return {"label": today}
        if (kept := stored or self.evidence.get(dcid)) is not None:
            return {"label": kept, "at_mint": True}
        self.gap(f"{path}.label")
        return {"label": dcid}

    def input_label(self, version: InputVersion, path: str) -> Wire:
        """The `label` member of the object at `path` that names a computed version's input.
        An input without an id has no label in the vocabulary, so it takes the label its step
        stored, else its content version with a gap naming the label."""
        stored = version.model_dump().get("label")
        if version.id is not None:
            return self.label(version.id, path, stored)
        if stored is not None:
            return {"label": stored, "at_mint": True}
        self.gap(f"{path}.label")
        return {"label": version.content_version}

    def input_ref(self, version: InputVersion, path: str) -> Wire:
        """A served reference to a computed version's input at `path`: its id, content
        version and label. An input whose id is unknown serves `id: null` with a gap."""
        if version.id is None:
            self.gap(f"{path}.id")
        return {
            "id": version.id,
            "content_version": version.content_version,
            **self.input_label(version, path),
        }

    def ref(self, dcid: str, path: str, stored: str | None = None) -> Wire:
        """A served reference to `dcid` at `path`, with its label."""
        return {"ref": dcid, **self.label(dcid, path, stored)}

    def defined_ref(self, dcid: str, path: str) -> Wire:
        """A served reference to `dcid` at `path`, with its label and, where the vocabulary
        holds one, its definition (`Vocabulary.definition`)."""
        wire = self.ref(dcid, path)
        if (text := self.vocabulary.definition(dcid)) is not None:
            wire["definition"] = text
        return wire

    def property(self, prop: str, path: str) -> Wire:
        """A served reference to a property: a flow role by its family's noun, a constraint
        or measure axis by its noun, else by its label."""
        if (axis := axis_for_role(prop)) is not None:
            return {"ref": prop, "label": self.vocabulary.roles(self.family).noun(axis).singular}
        if (noun := self.vocabulary.axis_name(prop)) is not None:
            return {"ref": prop, "label": noun}
        return self.ref(prop, path)

    def unit(self, unit: Unit, path: str) -> Wire | None:
        """A served unit, with a label on each side; `None` for a unit the data declares
        none of."""
        if unit is None:
            return None
        if isinstance(unit, ComposedUnit):
            return {
                "numerator": self._unit_side(unit.numerator, f"{path}.numerator"),
                "denominator": self._unit_side(unit.denominator, f"{path}.denominator"),
            }
        return self._simple_unit(unit, path)

    def _unit_side(self, side: SimpleUnit | VariableSide, path: str) -> Wire:
        if isinstance(side, VariableSide):
            stored = side.model_dump().get("label")
            return {"stat_var": side.stat_var, **self.label(side.stat_var, path, stored)}
        return self._simple_unit(side, path)

    def _simple_unit(self, unit: SimpleUnit, path: str) -> Wire:
        held = unit.model_dump()
        ref, currency = held.get("ref"), held.get("currency")
        today = (
            (self.vocabulary.unit_noun(ref) if ref is not None else None)
            or (CURRENCY_NOUNS.get(currency) if currency is not None else None)
            or (self.vocabulary.label(ref) if ref is not None else None)
        )
        if today is not None:
            words: Wire = {"label": today}
        elif (stored := held.get("label")) is not None:
            words = {"label": stored, "at_mint": True}
        else:
            self.gap(f"{path}.label")
            words = {"label": ref or currency}
        members = {
            key: held[key] for key in ("ref", "currency", "prices", "base_year") if key in held
        }
        return {**members, **words}


# about


def measure(variables: VariableFacts, words: Words, *, named: MeasureWords | None) -> Wire | None:
    """The measure the definition's shape and fixed values name, with its definition.

    The measure is `named`, the one the vocabulary names (`defined_measure`), with the
    definition the vocabulary authors for it. Where no measure label or noun covers the
    definition and it resolved one variable, the measure is that variable, named by its
    label. Otherwise the measure is unknown, with a gap.

    Where the vocabulary defines no measure and the definition resolved one World Development
    Indicators variable, that variable's graph description defines it
    (`world_bank_definition`), read from the vocabulary's definitions. With none of these,
    `definition` is `None` with gap `about.measure.definition`."""
    if named is not None:
        wire: Wire = {"label": named.label}
        definition = named.definition
    elif variables.variable is not None:
        wire = words.ref(variables.variable, "about.measure")
        definition = None
    else:
        words.gap("about.measure")
        return None
    if definition is None and (variable := variables.variable) is not None:
        definition = world_bank_definition(variable, words.vocabulary.definitions.get(variable))
    if definition is None:
        words.gap("about.measure.definition")
    return {**wire, "definition": definition}


def defined_measure(variables: VariableFacts, vocabulary: Vocabulary) -> MeasureWords | None:
    """The measure the definition's shape and fixed values name (`Vocabulary.measure`), else
    `None`."""
    return vocabulary.measure(
        shape={item.property: item.value for item in variables.shape},
        constraints={item.property: item.value for item in variables.fixed},
    )


def fixed_values(
    variables: VariableFacts, words: Words, *, named: Collection[str] = ()
) -> list[Wire]:
    """The properties the definition fixes, in its order, each with its value and the value's
    definition where the vocabulary holds one. Properties in `named` are left out."""
    fixed = [item for item in variables.fixed if item.property not in named]
    return [
        {
            "property": words.property(item.property, f"about.fixed.{n}.property"),
            "value": words.defined_ref(item.value, f"about.fixed.{n}.value"),
        }
        for n, item in enumerate(fixed)
    ]


def named_by_measure(named: MeasureWords | None) -> frozenset[str]:
    """The properties whose fixed values the measure `named` names, which are those its label
    or noun states (`MeasureWords.stated`). A partition label states every measure axis the
    constraints hold ("gross disbursements" states `financialFlowType` and `nettingBasis`).
    A noun states the constraints its key covers ("gross capital formation" states
    `finalExpenditureType`), so a measure axis it leaves uncovered keeps its fixed value
    ("external debt service" leaves `nettingBasis`). With no measure named, the result is
    empty. A measure that falls back to the version's one variable is named by that
    variable's label, which need not state its flow type or netting basis, so the fixed
    values keep them."""
    return named.stated if named is not None else frozenset()


def open_dimensions(
    record: DescriptionRecord, variables: VariableFacts, words: Words
) -> list[Wire]:
    """The dimensions the definition leaves open, in its order. A dimension holding one value
    stays open, since another version of the same definition may hold more. One that no
    variable with data holds a value of varies over nothing and is left out, unless it is
    `partial`, when a variable that left the index may have held one."""
    held = [
        dimension
        for dimension in variables.dimensions or []
        if dimension.values != [] or dimension.partial
    ]
    return [
        _open_dimension(record, dimension, words, f"about.varies.{n}")
        for n, dimension in enumerate(held)
    ]


def ranked_values(
    counts: Iterable[tuple[str, int]], *, labels: Mapping[str, str]
) -> list[tuple[str, int]]:
    """An open dimension's `(value, count)` pairs, highest count first, ties ordered by the
    value's label in `labels`, then by the value itself."""
    return sorted(counts, key=lambda pair: (-pair[1], labels[pair[0]], pair[0]))


def _open_dimension(
    record: DescriptionRecord, dimension: DimensionFacts, words: Words, path: str
) -> Wire:
    """`count` values hold data. `top` leads with up to `TOP_VALUES` of them in
    `ranked_values` order, and `more` counts the rest. A dimension whose values are
    unknown serves `count: null` with a gap, and no `top`, `more` or values link."""
    wire: Wire = {"property": words.property(dimension.property, f"{path}.property")}
    if dimension.within is not None:
        wire["within"] = words.ref(dimension.within, f"{path}.within")
    if dimension.values is None:
        wire["count"] = words.known(None, f"{path}.count")
        return wire
    ranked = ranked_values(
        [(value.value, value.variables) for value in dimension.values],
        labels={value.value: words.name(value.value) or value.value for value in dimension.values},
    )
    top = [
        {**words.ref(dcid, f"{path}.top.{n}"), "variables": variables}
        for n, (dcid, variables) in enumerate(ranked[:TOP_VALUES])
    ]
    wire["count"] = len(dimension.values)
    if dimension.partial:
        wire["partial"] = True
    if dimension.undeclared_variables:
        wire["undeclared_variables"] = dimension.undeclared_variables
    wire["top"] = top
    wire["more"] = len(dimension.values) - len(top)
    wire["values"] = values_link(record, dimension.property)
    return wire


def variable_counts(variables: VariableFacts, words: Words) -> Wire:
    """How many variables the definition admitted at mint, and how many hold data."""
    return {
        "admitted_at_mint": variables.admitted_at_mint,
        "with_data": words.known(variables.with_data, "about.variables.with_data"),
    }


def period(series: SeriesFacts, words: Words) -> Wire | None:
    """The observed span, with its grain where known."""
    if words.known(series.span, "about.period") is None:
        return None
    return span(series.span, series.grain)


def span(window: Span | None, grain: str | None) -> Wire | None:
    """A span as served."""
    if window is None:
        return None
    served: Wire = {"start": window.start, "end": window.end}
    if grain is not None:
        served["grain"] = grain
    return served


def series_unit(series: SeriesFacts, words: Words, path: str = "about.unit") -> Wire | None:
    """The unit every series holds. Unknown when the series per unit are, or when the series
    hold several units; `None` without a gap when they declare none."""
    if series.by_source_unit is None:
        words.gap(path)
        return None
    return _one_unit([count.unit for count in series.by_source_unit], words, path)


def _one_unit(units: Sequence[Unit], words: Words, path: str) -> Wire | None:
    distinct: list[Unit] = []
    for unit in units:
        if unit not in distinct:
            distinct.append(unit)
    if len(distinct) > 1:
        words.gap(path)
        return None
    return words.unit(distinct[0], path) if distinct else None


# Cards


def drawn(record: DescriptionRecord, selection: SelectionFacts, words: Words) -> Wire:
    """What a card draws against what its version holds: per open dimension, how many of the
    version's values the drawn series hold, and the card's window.

    A drawn series names a code through a coordinate on the dimension's own column, or
    through a variable the code index lists with that code. A dimension whose values are
    unknown is left out. A dimension lists the codes drawn on it in `values`, in the order the
    series first draw them, each with its definition where the vocabulary holds one, unless it
    draws more than `DRAWN_VALUES_MAX`, so a list that is served is complete. A dimension
    with a drawn code that neither the vocabulary nor the evidence labels serves no `values`,
    with one gap naming the list, so a served list labels every code. A value the drawn
    variables declare on their own index rows carries the publisher codes of the series
    holding it (`source_codes`). The lists are held to `DRAWN_VALUES_BUDGET_BYTES`
    (`_within_budget`)."""
    held = {
        dimension.property: {value.value for value in dimension.values}
        for dimension in (record.variables.dimensions or [] if record.variables else [])
        if dimension.values is not None
    }
    taken: dict[str, dict[str, None]] = {prop: {} for prop in held}
    for series in selection.series:
        for prop, code in _series_codes(series, words.vocabulary):
            if prop in held and code in held[prop]:
                taken[prop][code] = None
    declaring: dict[tuple[str, str], list[DrawnSeries]] = {}
    for series in selection.series:
        variable = _drawn_variable(series)
        row = words.vocabulary.index.statvar_by_dcid(variable) if variable is not None else None
        for pair in row.constraint_properties.items() if row is not None else ():
            declaring.setdefault(pair, []).append(series)
    dimensions = [prop for prop in held if taken[prop]]
    wire: Wire = {
        "dimensions": _within_budget(
            [
                _drawn_dimension(
                    prop,
                    list(taken[prop]),
                    of=len(held[prop]),
                    words=words,
                    path=f"about.drawn.dimensions.{n}",
                    coded={
                        code: codes
                        for code in taken[prop]
                        if (prop, code) in declaring
                        and (codes := source_codes(record, declaring[prop, code], words.vocabulary))
                    },
                )
                for n, prop in enumerate(dimensions)
            ],
            words,
        ),
        "series": len(selection.series),
    }
    wire["period"] = span(selection.window, record.series.grain)
    return wire


def _drawn_dimension(
    prop: str,
    codes: Sequence[str],
    *,
    of: int,
    words: Words,
    path: str,
    coded: Mapping[str, list[Wire]],
) -> Wire:
    """A drawn dimension at `path`, its values each with its definition and the publisher
    codes `coded` gives it."""
    dimension: Wire = {
        "property": words.property(prop, f"{path}.property"),
        "count": len(codes),
        "of": of,
    }
    if len(codes) > DRAWN_VALUES_MAX:
        return dimension
    if any(words.name(code) is None for code in codes):
        words.gap(f"{path}.values")
        return dimension
    dimension["values"] = [
        words.defined_ref(code, f"{path}.values.{n}")
        | ({"source_codes": coded[code]} if code in coded else {})
        for n, code in enumerate(codes)
    ]
    return dimension


def _within_budget(dimensions: list[Wire], words: Words) -> list[Wire]:
    """`dimensions` with their `values` held to `DRAWN_VALUES_BUDGET_BYTES` together.

    Over the budget, the dimensions give up their values' definitions, keeping the labels,
    one dimension at a time, the largest list first, until the lists fit. Still over, they
    give up their `values`, leaving the counts, in the same order, each with a gap naming the
    list. Ties keep the dimensions' order, so the outcome depends on the words alone."""
    sizes = [_values_bytes(dimension) for dimension in dimensions]
    for strip in (_without_definitions, _without_values):
        for n in sorted(range(len(dimensions)), key=lambda n: -sizes[n]):
            if sum(sizes) <= DRAWN_VALUES_BUDGET_BYTES:
                return dimensions
            if "values" in dimensions[n]:
                dimensions[n] = strip(dimensions[n], words, path=f"about.drawn.dimensions.{n}")
                sizes[n] = _values_bytes(dimensions[n])
    return dimensions


def _values_bytes(dimension: Wire) -> int:
    values = dimension.get("values")
    if values is None:
        return 0
    return len(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode())


def _without_definitions(dimension: Wire, words: Words, *, path: str) -> Wire:
    values = [
        {key: item for key, item in value.items() if key != "definition"}
        for value in dimension["values"]
    ]
    return dimension | {"values": values}


def _without_values(dimension: Wire, words: Words, *, path: str) -> Wire:
    words.gap(f"{path}.values")
    return {key: item for key, item in dimension.items() if key != "values"}


def _series_codes(series: DrawnSeries, vocabulary: Vocabulary) -> Iterator[tuple[str, str]]:
    for coordinate in series.coordinates:
        if coordinate.value is None:
            continue
        yield coordinate.column, coordinate.value
        if (row := vocabulary.index.statvar_by_dcid(coordinate.value)) is not None:
            yield from row.constraint_properties.items()


def drawn_unit(selection: SelectionFacts, words: Words) -> Wire | None:
    """The unit every drawn series holds; unknown when they hold several."""
    return _one_unit([series.unit for series in selection.series], words, "about.unit")


# Source codes

type _Coded = tuple[frozenset[tuple[str, str]], str | None]
"""The `(provenance, code)` pairs naming one series' variable, and the source it is read
from."""


def source_codes(
    record: DescriptionRecord, series: Sequence[DrawnSeries], vocabulary: Vocabulary
) -> list[Wire] | None:
    """The publisher codes of the variables `series` hold (`served_codes`). A series of a
    computed version whose variable the code index gives no code takes the codes of the
    variables it was composed from (`_variable_codes`)."""
    composed = {
        variable.variable: variable
        for variable in (record.computed.composed or [] if record.computed else [])
    }
    index = vocabulary.index
    return served_codes(
        [(_variable_codes(_drawn_variable(one), index, composed), one.source) for one in series],
        index,
        computed=record.computed is not None,
    )


def version_source_codes(record: DescriptionRecord, vocabulary: Vocabulary) -> list[Wire] | None:
    """The publisher codes of the one variable a dataset version's definition resolved to,
    read from each source its series name (`served_codes`). `None` for a version of several
    variables, which its record does not list."""
    variables = record.variables
    if variables is None or variables.variable is None:
        return None
    codes = frozenset(vocabulary.index.source_codes_of(variables.variable))
    counts = source_counts(record.series)
    sources = (list(counts) if counts is not None else []) or [None]
    return served_codes([(codes, source) for source in sources], vocabulary.index, computed=False)


def served_codes(held: Sequence[_Coded], index: CodeIndex, *, computed: bool) -> list[Wire] | None:
    """The `(provenance, code)` pairs `held` names, each once, sorted by provenance and then
    code, as Search serves `stat_vars.source_codes`. `None` when there are none, or more
    than `SOURCE_CODES_MAX`, so a list that is served is complete.

    A code is `exact`, as Search's is when its Spec is the series the provenance files
    under it (`search.engine.source_code.stands_for_code`), when every series of `held` is
    read from its provenance and named by it, and the provenance gives none of the code's
    variables another code. A computed version's series were made from the variables the
    codes name, so none of their codes is exact."""
    pairs = sorted({pair for codes, _ in held for pair in codes})
    if not pairs or len(pairs) > SOURCE_CODES_MAX:
        return None
    return [
        {"provenance": provenance, "code": code}
        | ({"exact": True} if not computed and _names_all(provenance, code, held, index) else {})
        for provenance, code in pairs
    ]


def _names_all(provenance: str, code: str, held: Sequence[_Coded], index: CodeIndex) -> bool:
    return not index.shared_codes(provenance, code) and all(
        source == provenance and (provenance, code) in codes for codes, source in held
    )


def _variable_codes(
    variable: str | None,
    index: CodeIndex,
    composed: Mapping[str, ComposedVariable],
    seen: frozenset[str] = frozenset(),
) -> frozenset[tuple[str, str]]:
    """The `(provenance, code)` pairs naming `variable` in the code index, else those of the
    variables a computed version composed it from, followed back through each composed
    variable. Empty for a variable that neither names."""
    if variable is None:
        return frozenset()
    if own := index.source_codes_of(variable):
        return frozenset(own)
    parts = composed.get(variable)
    if parts is None or variable in seen:
        return frozenset()
    return frozenset().union(
        *(
            _variable_codes(part, index, composed, seen | {variable})
            for part in [*parts.first, *parts.second]
        )
    )


# sources


def dataset_sources(record: DescriptionRecord, words: Words) -> list[Wire]:
    """Each source the version used, with its series count."""
    return [
        source_entry(
            source,
            series=_series_of(record.series, source.ref),
            words=words,
            path=f"sources.{n}",
        )
        for n, source in enumerate(record.evidence.sources)
    ]


MAX_SOURCES: Final = 64
"""The most entries a description's `sources` may hold, the bound the kit's description schema
sets on `sources`."""


@dataclass(frozen=True, slots=True)
class Credit:
    """A source a computed version's inputs credit, and how many described series draw on
    it, `None` where unknown."""

    source: SourceEvidence
    series: int | None


@dataclass(frozen=True, slots=True)
class ServedSources:
    """A description's `sources` entries, and the sources among them its citation line
    names, in their order."""

    entries: list[Wire]
    cited: list[SourceEvidence]


def drawn_sources(
    record: DescriptionRecord,
    selection: SelectionFacts,
    words: Words,
    *,
    inputs: Sequence[DescriptionRecord] = (),
) -> ServedSources:
    """The sources of the series a card draws, each with how many it draws.

    A drawn series of a computed version that names no source of its own was computed from
    the inputs its variable depends on (`drawn_credits`). The sources those inputs' records
    (`inputs`) credit follow the named sources, then those inputs (`computed_sources`). A
    computed version's card cites the sources it serves; any other card cites none here, and
    its citation line names its version's."""
    counts: dict[str | None, int] = {}
    for series in selection.series:
        counts[series.source] = counts.get(series.source, 0) + 1
    credits: list[Credit] = []
    drawn: list[InputVersion] = []
    if record.computed is not None and None in counts:
        credits, drawn = drawn_credits(record, selection, inputs)
    shown = {credit.source.ref for credit in credits}
    named = recorded_sources(
        record, [ref for ref in counts if ref is not None and ref not in shown]
    )
    entries = [
        source_entry(source, series=counts[source.ref], words=words, path=f"sources.{n}")
        for n, source in enumerate(named)
    ]
    if not drawn and not credits:
        return ServedSources(entries=entries, cited=named if record.computed else [])
    served = computed_sources(credits, drawn, words, start=len(entries))
    return ServedSources(entries=[*entries, *served.entries], cited=[*named, *served.cited])


def drawn_credits(
    record: DescriptionRecord, selection: SelectionFacts, inputs: Sequence[DescriptionRecord]
) -> tuple[list[Credit], list[InputVersion]]:
    """The sources a card's drawn series that name no source draw on, and the inputs of the
    computed version they were computed from, in the version's input order.

    Each such series reads the inputs its variable depends on (`_read_inputs`), and draws on
    every source the records of those inputs (`inputs`) credit. A credit counts the series
    drawing on it, and the drawn series naming it. An input whose record is not held credits
    nothing."""
    computed = record.computed
    assert computed is not None, "a computed version's record holds what it was computed from"
    held = {record.content_version: record for record in inputs}
    named: dict[str, int] = {}
    drawing: dict[str, int] = {}
    for series in selection.series:
        if series.source is not None:
            named[series.source] = named.get(series.source, 0) + 1
            continue
        versions = _read_inputs(record, _drawn_variable(series))
        refs = {
            source.ref
            for version in versions
            if version in held
            for source in held[version].evidence.sources
        }
        for ref in refs:
            drawing[ref] = drawing.get(ref, 0) + 1
    drawn = drawn_inputs(record, selection)
    evidence = input_evidence([held[v.content_version] for v in drawn if v.content_version in held])
    credits = [
        Credit(source=source, series=drawing[source.ref] + named.get(source.ref, 0))
        for source in evidence
        if source.ref in drawing
    ]
    return credits, drawn


def drawn_inputs(record: DescriptionRecord, selection: SelectionFacts) -> list[InputVersion]:
    """The inputs of a computed version that the drawn series naming no source were computed
    from (`_read_inputs`), in the version's input order."""
    computed = record.computed
    assert computed is not None, "a computed version's record holds what it was computed from"
    read: set[str] = set()
    for series in selection.series:
        if series.source is None:
            read |= _read_inputs(record, _drawn_variable(series))
    return [version for version in computed.inputs if version.content_version in read]


def _drawn_variable(series: DrawnSeries) -> str | None:
    """The bare dcid of the variable a drawn series holds, `None` where it names none."""
    return next((one.value for one in series.coordinates if one.column == STAT_VAR_COLUMN), None)


def _read_inputs(record: DescriptionRecord, variable: str | None) -> set[str]:
    """The content versions of the inputs a computed version's `variable` was computed from:
    the versions the `retrieve` steps behind each step writing it read, followed back through
    each step's inputs. A variable no step writes, such as one retrieved as published, steps
    that are unknown, and a step input naming no recorded step read every input."""
    computed = record.computed
    assert computed is not None, "a computed version's record holds what it was computed from"
    every = {version.content_version for version in computed.inputs}
    steps = {step.id: step for step in record.steps or ()}
    pending = [
        step for step in steps.values() if variable is not None and step.stat_var == variable
    ]
    if not pending:
        return every
    read: set[str] = set()
    seen: set[str] = set()
    while pending:
        step = pending.pop()
        if step.id in seen:
            continue
        seen.add(step.id)
        given = [] if step.input is MISSING else [step.input]
        if step.inputs is not MISSING:
            given.extend(step.inputs.values())
        for one in given:
            if isinstance(one, VersionRef):
                read.add(one.content_version)
            elif one in steps:
                pending.append(steps[one])
            else:
                return every
    return read & every or every


def input_source(version: InputVersion, words: Words, *, path: str) -> Wire:
    """A computed version's input as a source at `path`: its id as `ref`, its label, its
    content version and the link to its description. An input whose id is unknown has
    `ref: null` with a gap and no link."""
    source: Wire = {
        "ref": version.id,
        **words.input_label(version, path),
        "content_version": version.content_version,
    }
    if version.id is None:
        words.gap(f"{path}.ref")
    else:
        source["description"] = f"{version_link(version.id, version.content_version)}/description"
    return source


def input_evidence(inputs: Sequence[DescriptionRecord]) -> list[SourceEvidence]:
    """The sources the records of a computed version's inputs credit, each once by ref, in
    input order and then each input's own order. An input that credits no source, such as a
    computed version, adds none."""
    credited: dict[str, SourceEvidence] = {}
    for record in inputs:
        for source in record.evidence.sources:
            credited.setdefault(source.ref, source)
    return list(credited.values())


def version_credits(record: DescriptionRecord, inputs: Sequence[DescriptionRecord]) -> list[Credit]:
    """The sources the records of a computed version's inputs credit (`input_evidence`), each
    counting the version's series naming it and those naming none, which were computed from
    the inputs. The version's counts by source hold no variable to trace a series to the inputs
    it read, so a series naming no source counts toward every credit. With its counts by
    source unknown, each count is unknown. The credit reaches one level, since an input that
    is itself a computed version credits no source of its own and adds none."""
    counts = source_counts(record.series)
    return [
        Credit(
            source=source,
            series=None if counts is None else counts.get(source.ref, 0) + counts.get(None, 0),
        )
        for source in input_evidence(inputs)
    ]


def computed_sources(
    credits: Sequence[Credit],
    inputs: Sequence[InputVersion],
    words: Words,
    *,
    start: int = 0,
) -> ServedSources:
    """A computed version's sources from path `sources.{start}` on: its credits, then each
    input with the link to its own description (`input_source`). The credits served, which
    the citation line names, are those that fit in `MAX_SOURCES` beside the inputs
    (`_most_drawn`)."""
    kept = _most_drawn(credits, room=MAX_SOURCES - start - len(inputs))
    entries = [
        source_entry(credit.source, series=credit.series, words=words, path=f"sources.{n}")
        for n, credit in enumerate(kept, start=start)
    ]
    entries.extend(
        input_source(version, words, path=f"sources.{n}")
        for n, version in enumerate(inputs, start=start + len(entries))
    )
    return ServedSources(entries=entries, cited=[credit.source for credit in kept])


def _most_drawn(credits: Sequence[Credit], *, room: int) -> list[Credit]:
    """`credits`, or past `room` the `room` that most series draw on, an unknown count
    last and ties to the earlier, in their order. The inputs are always served, since each
    links to the description that credits what is left out."""
    if len(credits) <= room:
        return list(credits)
    ranked = sorted(
        range(len(credits)),
        key=lambda n: -1 if credits[n].series is None else credits[n].series,
        reverse=True,
    )
    return [credits[n] for n in sorted(ranked[: max(room, 0)])]


def source_counts(series: SeriesFacts) -> dict[str | None, int] | None:
    """A version's series by the source each names, `None` for those naming none, or `None`
    where its counts by source are unknown."""
    if series.by_source_unit is None:
        return None
    counts: dict[str | None, int] = {}
    for count in series.by_source_unit:
        counts[count.source] = counts.get(count.source, 0) + count.series
    return counts


def recorded_sources(record: DescriptionRecord, refs: Sequence[str]) -> list[SourceEvidence]:
    """The evidence `record` holds for each source in `refs`, in their order; a source it
    holds none for is its bare ref."""
    recorded = {source.ref: source for source in record.evidence.sources}
    return [
        recorded.get(ref)
        or SourceEvidence(ref=ref, name=None, licence=None, downloaded=None, pipeline_steps=None)
        for ref in refs
    ]


def card_sources(
    record: DescriptionRecord, sources: Sequence[SourceEvidence], words: Words
) -> list[Wire]:
    """A source card's sources, each with the series the version holds from it."""
    return [
        source_entry(
            source, series=_series_of(record.series, source.ref), words=words, path=f"sources.{n}"
        )
        for n, source in enumerate(sources)
    ]


def source_entry(source: SourceEvidence, *, series: int | None, words: Words, path: str) -> Wire:
    """One source: today's name, else the name stored at mint unless that was the bare ref;
    its publisher's short name, else the publisher's label; the licence recorded at mint;
    and the link to its document."""
    stored = source.name if source.name != source.ref else None
    return {
        "ref": source.ref,
        **words.label(source.ref, path, stored),
        "publisher": words.known(_publisher(source.ref, words.vocabulary), f"{path}.publisher"),
        "licence": words.known(source.licence, f"{path}.licence"),
        "series": words.known(series, f"{path}.series"),
        "documentation": documentation_link(source.ref),
    }


def documentation_link(ref: str) -> str:
    """The Graph API route of the document of the source `ref` names."""
    return f"/api/graph/documentation/{ref}"


def source_name(source: SourceEvidence, words: Words) -> str:
    """A source's name for a sentence: as `source_entry` labels it, else its ref."""
    stored = source.name if source.name != source.ref else None
    return words.name(source.ref, stored) or source.ref


def _publisher(source: str, vocabulary: Vocabulary) -> str | None:
    if (short := vocabulary.publisher(source)) is not None:
        return short
    facts = vocabulary.sources.get(source)
    if facts is None or facts.publisher is None:
        return None
    return vocabulary.label(facts.publisher)


def _series_of(series: SeriesFacts, source: str) -> int | None:
    if series.by_source_unit is None:
        return None
    return sum(count.series for count in series.by_source_unit if count.source == source)


# freshness


def dataset_freshness(record: DescriptionRecord, words: Words) -> Wire:
    """How current the version is: when it was prepared, its latest period, the source
    release and downloads behind it, the next release, and its status counts.

    Over several sources, a date is the earliest of theirs, and unknown when any is or when
    a document writes one in another form."""
    downloads = source_downloads(record.evidence.sources, words)
    return {
        "prepared": record.evidence.prepared.isoformat(),
        **latest(record.series, words),
        "source_release": words.known(record.evidence.source_release, "freshness.source_release"),
        **downloads.served(words),
        **status(record.series.status, words),
    }


@dataclass(frozen=True, slots=True)
class SourceDownloads:
    """When sources were downloaded for a version, the later download their documents
    record, and their next release."""

    at_mint: str | None
    since: str | None
    next_release: str | None

    def served(self, words: Words) -> Wire:
        """The three as `freshness` members, an unknown date named in the gaps."""
        return {
            "source_downloaded_at_mint": words.known(
                self.at_mint, "freshness.source_downloaded_at_mint"
            ),
            "source_downloaded_since": self.since,
            "next_release": words.known(self.next_release, "freshness.next_release"),
        }


def source_downloads(sources: Sequence[SourceEvidence], words: Words) -> SourceDownloads:
    """When `sources` were downloaded for the version, the latest download their documents
    record where it is later, and their next release. Over several sources, a date is the
    earliest of theirs, and unknown when any is or when a document writes one in another
    form. An unknown latest download is named in the gaps."""
    documents = [words.vocabulary.sources.get(source.ref) for source in sources]
    at_mint = _earliest(
        [source.downloaded.isoformat() if source.downloaded else None for source in sources], _DATE
    )
    today = _earliest([doc.last_downloaded if doc else None for doc in documents], _DATE)
    if today is None:
        words.gap("freshness.source_downloaded_since")
    return SourceDownloads(
        at_mint=at_mint,
        since=today if today is not None and (at_mint is None or today > at_mint) else None,
        next_release=_earliest(
            [doc.next_release if doc else None for doc in documents], _PARTIAL_DATE
        ),
    )


def _earliest(dates: Sequence[str | None], form: re.Pattern[str]) -> str | None:
    known = [date for date in dates if date is not None and form.match(date)]
    if not known or len(known) < len(dates):
        return None
    return min(known)


def latest(series: SeriesFacts, words: Words) -> Wire:
    """The latest period and how many series reach it."""
    return {
        "latest_period": words.known(series.latest_period, "freshness.latest_period"),
        "series_reaching_it": words.known(
            series.series_reaching_latest, "freshness.series_reaching_it"
        ),
    }


def status(counts: StatusCounts, words: Words) -> Wire:
    """How many values are estimated and how many provisional; a known 0 stays 0."""
    return {
        "estimated_values": words.known(counts.estimated, "freshness.estimated_values"),
        "provisional_values": words.known(counts.provisional, "freshness.provisional_values"),
    }


def drawn_freshness(selection: SelectionFacts, words: Words) -> Wire:
    """The latest period of the drawn series inside the card's window, how many reach it, and
    their status counts summed. One series with an unknown count makes the sum unknown."""
    ends = [series.last_period for series in selection.series if series.last_period is not None]
    newest = max(ends) if ends else None
    return {
        "latest_period": words.known(newest, "freshness.latest_period"),
        "series_reaching_it": words.known(
            sum(end == newest for end in ends) if newest is not None else None,
            "freshness.series_reaching_it",
        ),
        **status(
            StatusCounts(
                estimated=_sum([series.status.estimated for series in selection.series]),
                provisional=_sum([series.status.provisional for series in selection.series]),
            ),
            words,
        ),
    }


def _sum(counts: Sequence[int | None]) -> int | None:
    if any(count is None for count in counts):
        return None
    return sum(count for count in counts if count is not None)


# caveats


def caveats(
    facts: Sequence[CaveatFact],
    words: Words,
    *,
    steps: Sequence[Step],
    grain: str | None,
    drawn_entities: Collection[str] = (),
) -> list[Wire]:
    """Each caveat with labelled references and its sentence in `text`, worded from the
    `steps` a caveat may point at and the version's `grain`. At most `CAVEATS_MAX` are
    served. A caveat naming more than `CAVEAT_ENTITIES_MAX` entities names that many, those in
    `drawn_entities` first and the rest in stored order, and counts them all."""
    served: list[Wire] = []
    for fact in facts:
        if len(served) == CAVEATS_MAX:
            break
        entities, count = fact.entities, fact.count
        if len(entities) > CAVEAT_ENTITIES_MAX:
            count = max(count or 0, len(entities))
            entities = sorted(entities, key=lambda ref: ref not in drawn_entities)
            entities = entities[:CAVEAT_ENTITIES_MAX]
        path = f"caveats.{len(served)}"
        caveat: Wire = {"kind": fact.kind}
        if fact.series:
            caveat["series"] = [
                words.ref(ref, f"{path}.series.{n}") for n, ref in enumerate(fact.series)
            ]
        if entities:
            caveat["entities"] = [
                words.ref(ref, f"{path}.entities.{n}") for n, ref in enumerate(entities)
            ]
        if fact.periods:
            caveat["periods"] = list(fact.periods)
        if fact.step is not None:
            caveat["step"] = fact.step
        if count is not None:
            caveat["count"] = count
        caveat["text"] = caveat_text(
            caveat, steps=steps, vocabulary=words.vocabulary, grain=grain, family=words.family
        )
        served.append(caveat)
    return served


# Links


def version_link(id: str, content_version: str) -> str:
    """The route of a version's data."""
    return f"/api/datasets/{id}/{content_version}"


def data_link(record: DescriptionRecord) -> str:
    """The route of a version's data: the Datasets snapshot for a dataset version, the
    Presentation frame for a computed version, which no Datasets route serves."""
    if record.computed is None:
        return version_link(record.id, record.content_version)
    return f"/api/presentation/frames/{record.id}/{record.content_version}"


def values_link(record: DescriptionRecord, prop: str) -> str:
    """The route that pages one dimension's values."""
    base = version_link(record.id, record.content_version)
    return f"{base}/description/values?property={quote(prop, safe='')}"
