"""The variable card's `definition`: what a variable measures, part by part.

A definition names the population type, measured property and statistical type, the measure,
the roles a variable's observations carry, the publisher's indicator codes, and its constraints,
each constraint as a `{property, value}` pair of references, ordered by property dcid. Its
`definitions` list the prose the graph holds for each part, in order: the variable's own, then
the measure's, then each constraint value's. A World Development Indicators variable's own prose
leaves out the series code its import appends (`description.world_bank`). Each part stays its
own entry.

A variable the code index holds takes its shape and constraints from its row; its indicator
codes come from the node door's arc read, as every variable's do. Any other
variable, base or UN, takes them from its graph arcs: the node door's arc read gives
`populationType`, `measuredProperty`, `constraintProperties` and `observationProperties`, and
one batched read gives `statType`, `measurementQualifier` and each constraint property's value.

The measure is the code index's partition label over the measure axes
(`code_index.labels.compose_measure_label`), else its measure noun (`MeasureIndex.nouns`). A
constraint the measure states is left out of `constraints`. No graph node holds a measure's
definition yet, so a measure always carries the gap `stat_var.definition.measure.definition`.

Labels come from the code index first: a code's row, a code system's `plain_name`, a measure
axis's `MeasureAxis.label` and a measure value's row. A target the index cannot label takes its
graph `name`. One batched read gives those names and every constraint value's `description`.

A value the builder cannot know is `null`, or a missing `definitions` entry, and its member path
is in `gaps`. A failed graph read marks the definition `transient`, so the door answers
`no-store`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import ConfigDict, Field

from code_index.labels import compose_measure_label
from code_index.query import CodeIndex
from code_index.types import MeasureIndex, label_for_pairs
from dc_graph import display_name
from description.world_bank import WORLD_BANK_NAMESPACE, world_bank_definition
from dfo_platform.wire import ResponseModel
from graph_api.reader import GraphNodeReader, NodeArcsByDcid
from graph_api.refs import NodeRef, first_value
from materialise.graph import GraphReadError
from materialise.roles import DegenerateObservationProperties, declared_roles

logger = logging.getLogger(__name__)

_PATH = "stat_var.definition"

MEASURE_DEFINITION_GAP = f"{_PATH}.measure.definition"
"""The gap every variable with a measure carries, since no graph node holds a measure's
definition."""
_SHAPE_READ: tuple[str, ...] = ("statType", "measurementQualifier")
_WORDS_READ: tuple[str, ...] = ("name", "description")
_SHAPE_MEMBERS = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "statType": "stat_type",
}

type Arcs = Mapping[str, tuple[str, ...]]


class Constraint(ResponseModel):
    """One constraint: the property and the value the variable fixes it at."""

    model_config = ConfigDict(title="GraphDefinitionConstraint")

    property: NodeRef
    value: NodeRef


class Measure(ResponseModel):
    """What the variable counts or sums, as the code index names it."""

    model_config = ConfigDict(title="GraphDefinitionMeasure")

    label: str = Field(description="the measure's noun phrase, such as `net disbursements`")


class DefinitionText(ResponseModel):
    """The graph's prose for one part of a definition."""

    model_config = ConfigDict(title="GraphDefinitionText")

    of: NodeRef = Field(description="the part the text defines")
    text: str


class VariableDefinition(ResponseModel):
    """What a variable measures, part by part.

    `constraints` are ordered by property dcid: the code index holds a variable's constraints
    as a mapping with no declared order, and the graph's `constraintProperties` arc has none.
    """

    model_config = ConfigDict(title="GraphVariableDefinition")

    population_type: NodeRef | None = Field(
        description="the populationType; null when the variable declares none"
    )
    measured_property: NodeRef | None = Field(
        description="the measuredProperty; null when the variable declares none"
    )
    stat_type: NodeRef | None = Field(
        description="the statType; null when the variable declares none or its read failed"
    )
    measure: Measure | None = Field(
        description="the measure the code index names; null when it names none"
    )
    roles: list[str] = Field(
        description=(
            "the entity roles the variable's observations carry, from its declared "
            "observationProperties, defaulting to observationAbout when it declares none"
        )
    )
    source_indicator_codes: list[str] = Field(
        description=(
            "the sourceIndicatorCode arc's values: the codes naming the source series, as the "
            "publisher codes it (IMF WEO GDP carries NGDP and NGDPD); empty when it has none"
        )
    )
    constraints: list[Constraint] | None = Field(
        description=(
            "each constraint the measure does not state, ordered by property dcid; null when "
            "the "
            "read of a variable's constraint values failed"
        )
    )
    definitions: list[DefinitionText] = Field(
        description=(
            "the graph's prose per part: the variable's own, then the measure's, then each "
            "constraint value's, in constraint order; a part with none is left out"
        )
    )


@dataclass(frozen=True, slots=True)
class CardDefinition:
    """One variable's definition, the member paths it cannot fill, and whether any of them
    comes from a failed read."""

    definition: VariableDefinition
    gaps: tuple[str, ...]
    transient: bool


@dataclass(frozen=True, slots=True)
class _Shape:
    """A variable's shape, roles and constraint values, from its index row or its graph arcs.
    `constraints` is `None` when the read of its constraint values failed."""

    population_type: str | None
    measured_property: str | None
    stat_type: str | None
    measurement_qualifier: str | None
    observation_properties: tuple[str, ...]
    constraints: Mapping[str, str] | None


@dataclass(frozen=True, slots=True)
class _MeasureMatch:
    label: str
    stated: frozenset[str]
    """The constraint properties the measure's label states."""


@dataclass(slots=True)
class _Gaps:
    paths: list[str] = field(default_factory=list)
    transient: bool = False


def build_definitions(
    variables: Mapping[str, Arcs],
    *,
    graph: GraphNodeReader,
    code_index: CodeIndex,
    measure_index: MeasureIndex,
    deadline: float,
) -> dict[str, CardDefinition]:
    """Build the definition of every variable in `variables`.

    Makes at most two batched graph reads under `deadline`: the shape arcs and constraint
    values of the variables the code index does not hold, then the names and descriptions of
    every constraint value and of each property or shape value the indexes cannot label. Each read is skipped
    when it has no target. Runs synchronously, so the caller runs it on the DCP-read executor.

    Args:
        variables: each StatVar dcid mapped to the arcs the node door read for it. `name`,
            `description`, `populationType`, `measuredProperty`, `observationProperties`,
            `constraintProperties` and `sourceIndicatorCode` are read.
        graph: the reader both reads go through.
        code_index: the index whose rows, declarations and code labels come first.
        measure_index: the measure axes, values, partition labels and nouns.
        deadline: the absolute monotonic deadline both reads share.

    Returns:
        Each variable's `CardDefinition`, keyed by dcid.
    """
    gaps = {dcid: _Gaps() for dcid in variables}
    shapes = _shapes(variables, graph=graph, code_index=code_index, gaps=gaps, deadline=deadline)
    labels = _IndexLabels(code_index=code_index, measure_index=measure_index)
    measures: dict[str, _MeasureMatch | None] = {}
    constraints: dict[str, list[tuple[str, str]] | None] = {}
    targets: dict[str, list[str]] = {}
    for dcid, shape in shapes.items():
        measures[dcid] = _measure(shape, measure_index) if shape.constraints is not None else None
        constraints[dcid] = _open_constraints(shape, measures[dcid])
        targets[dcid] = _word_targets(shape, constraints[dcid], labels)

    words: NodeArcsByDcid = {}
    unique = list(dict.fromkeys(target for found in targets.values() for target in found))
    if unique:
        try:
            words = graph.node_arcs_many(unique, properties=_WORDS_READ, deadline=deadline)
        except GraphReadError:
            logger.warning("Variable card's definition word read failed", exc_info=True)
            for dcid, found in targets.items():
                gaps[dcid].transient = gaps[dcid].transient or bool(found)

    return {
        dcid: _assemble(
            dcid,
            variables[dcid],
            shape=shapes[dcid],
            measure=measures[dcid],
            constraints=constraints[dcid],
            labels=labels,
            words=words,
            gaps=gaps[dcid],
        )
        for dcid in variables
    }


def _shapes(
    variables: Mapping[str, Arcs],
    *,
    graph: GraphNodeReader,
    code_index: CodeIndex,
    gaps: Mapping[str, _Gaps],
    deadline: float,
) -> dict[str, _Shape]:
    """Each variable's shape: its index row's, else its graph arcs' and one batched read of
    their `statType`, `measurementQualifier` and constraint values."""
    shapes: dict[str, _Shape] = {}
    from_graph: list[str] = []
    for dcid in variables:
        row = code_index.statvar_by_dcid(dcid)
        if row is None:
            from_graph.append(dcid)
            continue
        shapes[dcid] = _Shape(
            population_type=row.population_type,
            measured_property=row.measured_property,
            stat_type=row.stat_type,
            measurement_qualifier=row.measurement_qualifier,
            observation_properties=row.observation_properties,
            constraints=row.constraint_properties,
        )
    if not from_graph:
        return shapes

    properties = sorted(
        {prop for dcid in from_graph for prop in variables[dcid].get("constraintProperties", ())}
    )
    read: NodeArcsByDcid | None
    try:
        read = graph.node_arcs_many(
            from_graph, properties=(*_SHAPE_READ, *properties), deadline=deadline
        )
    except GraphReadError:
        logger.warning("Variable card's constraint read failed", exc_info=True)
        read = None

    for dcid in from_graph:
        arcs = variables[dcid]
        held = (read.get(dcid) or {}) if read is not None else {}
        values: dict[str, str] | None = None
        if read is None:
            gaps[dcid].paths += [f"{_PATH}.measure", f"{_PATH}.constraints"]
            gaps[dcid].transient = True
        else:
            values = {}
            for prop in arcs.get("constraintProperties", ()):
                if (value := first_value(held, prop)) is None:
                    gaps[dcid].paths.append(f"{_PATH}.constraints.{prop}")
                else:
                    values[prop] = value
        shapes[dcid] = _Shape(
            population_type=first_value(arcs, "populationType"),
            measured_property=first_value(arcs, "measuredProperty"),
            stat_type=first_value(held, "statType"),
            measurement_qualifier=first_value(held, "measurementQualifier"),
            observation_properties=arcs.get("observationProperties", ()),
            constraints=values,
        )
    return shapes


def _measure(shape: _Shape, measure_index: MeasureIndex) -> _MeasureMatch | None:
    """The partition label over the measure axes, else the measure noun, or `None`."""
    constraints = shape.constraints or {}
    shape_values = {
        "populationType": shape.population_type,
        "measuredProperty": shape.measured_property,
        "measurementQualifier": shape.measurement_qualifier,
    }
    pairs = [
        (
            axis.axis,
            shape_values.get(axis.axis) if axis.kind == "shape" else constraints.get(axis.axis),
        )
        for axis in measure_index.axes
    ]
    match = label_for_pairs(pairs, measure_index.partition_labels)
    if match is not None:
        # `compose_measure_label` folds every uncovered measure-axis value into the label.
        return _MeasureMatch(
            label=compose_measure_label(match, measure_index.values),
            stated=frozenset(
                axis.axis
                for axis in measure_index.axes
                if axis.kind == "constraint" and axis.axis in constraints
            ),
        )

    noun = label_for_pairs(
        [
            ("populationType", shape.population_type),
            ("measuredProperty", shape.measured_property),
            ("statType", shape.stat_type),
            *constraints.items(),
        ],
        measure_index.nouns,
    )
    if noun is None:
        return None
    uncovered = frozenset(noun.uncovered)
    return _MeasureMatch(
        label=noun.label,
        stated=frozenset(pair[0] for pair in constraints.items() if pair not in uncovered),
    )


def _open_constraints(shape: _Shape, measure: _MeasureMatch | None) -> list[tuple[str, str]] | None:
    """The constraints the measure does not state, ordered by property."""
    if shape.constraints is None:
        return None
    stated = measure.stated if measure is not None else frozenset()
    return sorted((prop, value) for prop, value in shape.constraints.items() if prop not in stated)


@dataclass(frozen=True, slots=True)
class _IndexLabels:
    """The labels the code index and measure index give properties and values."""

    code_index: CodeIndex
    measure_index: MeasureIndex

    def property(self, prop: str) -> str | None:
        """A code system's `plain_name`, else a measure axis's `label`."""
        declaration = self.code_index.declarations.get(prop)
        if declaration is not None:
            return declaration.plain_name
        return next((axis.label for axis in self.measure_index.axes if axis.axis == prop), None)

    def value(self, prop: str, value: str) -> str | None:
        """A code row's label, else a measure value row's on the property's axis."""
        row = self.code_index.code_by_dcid(value)
        if row is not None:
            return row.label
        return next(
            (
                measured.label
                for measured in self.measure_index.values
                if measured.axis == prop and measured.dcid == value
            ),
            None,
        )


def _word_targets(
    shape: _Shape, constraints: list[tuple[str, str]] | None, labels: _IndexLabels
) -> list[str]:
    """The dcids whose graph names or descriptions a definition shows: every constraint value,
    and each property or shape value the indexes cannot label."""
    targets = [
        ref
        for axis, ref in _shape_refs(shape)
        if ref is not None and labels.value(axis, ref) is None
    ]
    for prop, value in constraints or ():
        if labels.property(prop) is None:
            targets.append(prop)
        targets.append(value)
    return targets


def _assemble(
    dcid: str,
    arcs: Arcs,
    *,
    shape: _Shape,
    measure: _MeasureMatch | None,
    constraints: list[tuple[str, str]] | None,
    labels: _IndexLabels,
    words: NodeArcsByDcid,
    gaps: _Gaps,
) -> CardDefinition:
    def graph_name(target: str) -> str | None:
        return _name(words.get(target) or {}, target)

    shape_refs: dict[str, NodeRef | None] = {}
    for axis, ref in _shape_refs(shape):
        member = _SHAPE_MEMBERS[axis]
        shape_refs[member] = None
        if ref is None:
            gaps.paths.append(f"{_PATH}.{member}")
            continue
        label = labels.value(axis, ref) or graph_name(ref)
        if label is None:
            gaps.paths.append(f"{_PATH}.{member}.label")
        shape_refs[member] = NodeRef(ref=ref, label=label)

    definitions: list[DefinitionText] = []
    if (prose := _variable_prose(dcid, first_value(arcs, "description"))) is not None:
        definitions.append(
            DefinitionText(of=NodeRef(ref=dcid, label=_name(arcs, dcid)), text=prose)
        )
    if measure is not None:
        gaps.paths.append(MEASURE_DEFINITION_GAP)
    elif shape.constraints is not None:
        gaps.paths.append(f"{_PATH}.measure")

    built: list[Constraint] | None = None
    if constraints is not None:
        built = []
        for prop, value in constraints:
            path = f"{_PATH}.constraints.{prop}"
            prop_label = labels.property(prop) or graph_name(prop)
            value_label = labels.value(prop, value) or graph_name(value)
            if prop_label is None:
                gaps.paths.append(f"{path}.property.label")
            if value_label is None:
                gaps.paths.append(f"{path}.value.label")
            value_ref = NodeRef(ref=value, label=value_label)
            built.append(Constraint(property=NodeRef(ref=prop, label=prop_label), value=value_ref))
            if (text := first_value(words.get(value) or {}, "description")) is None:
                gaps.paths.append(f"{path}.definition")
            else:
                definitions.append(DefinitionText(of=value_ref, text=text))

    return CardDefinition(
        definition=VariableDefinition(
            population_type=shape_refs["population_type"],
            measured_property=shape_refs["measured_property"],
            stat_type=shape_refs["stat_type"],
            measure=Measure(label=measure.label) if measure is not None else None,
            roles=_roles(shape.observation_properties),
            source_indicator_codes=list(arcs.get("sourceIndicatorCode", ())),
            constraints=built,
            definitions=definitions,
        ),
        gaps=tuple(gaps.paths),
        transient=gaps.transient,
    )


def _shape_refs(shape: _Shape) -> list[tuple[str, str | None]]:
    """The shape values a definition names, each beside the measure axis it would sit on."""
    return [
        ("populationType", shape.population_type),
        ("measuredProperty", shape.measured_property),
        ("statType", shape.stat_type),
    ]


def _roles(observation_properties: tuple[str, ...]) -> list[str]:
    try:
        return list(declared_roles(observation_properties))
    except DegenerateObservationProperties:
        # Report the graph declaration as-is, as the node door does. The mint path rejects it.
        return list(observation_properties)


def _variable_prose(dcid: str, description: str | None) -> str | None:
    """A variable's own prose: a World Development Indicators variable's description without
    the series code its import appends (`world_bank_definition`), any other's as the graph
    holds it."""
    if dcid.startswith(WORLD_BANK_NAMESPACE):
        return world_bank_definition(dcid, description)
    return description


def _name(arcs: Arcs, dcid: str) -> str | None:
    return display_name(arcs.get("name", ()), dcid)
