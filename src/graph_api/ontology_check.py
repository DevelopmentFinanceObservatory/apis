"""Check candidate StatisticalVariable definitions against the instance's ontology.

`OntologyLookup` holds the ontology document's nodes as the check reads them, built once per
document. `check_rows` answers each node a row names from it and reads the nodes the document
does not hold. A read is `node_arcs_batch`, every out-arc of each node, one batch per level of
`subClassOf` ancestry; a node a batch of several answers with no arc is read again alone. A
node held with any out-arc is held, its types its `typeOf` values, so a held node with no
`typeOf` is untyped; a node read alone with no out-arc is not held. A read error propagates, so
no node is reported absent because a read failed.

Some constraint values need no read. A value that is a number, a boolean, or a string outside
dcid syntax names no node: `node_not_held`. A value outside the document, on an instance
property whose every `rangeIncludes` class is an instance-declared enumeration the document
lists in full, is no member of any of them: `value_outside_range`. Every other id outside the
document is read, and a request whose rows name more than `CHECK_MAX_READ_IDS` distinct such
ids raises `OntologyCheckTooManyReads` before any read.

Every row gets existence and kind checks. A constraint value must name a resolved node, by
`code_index.constraint_values.constraint_value_fault`, the test the code index export refuses
a release on. Range and domain rules apply only to a property the document declares
`declared_by == "instance"`: a value is inside the range when one of its types reaches a
`rangeIncludes` class by `subClassOf`, and a property is inside its domain when the row's
population type reaches a `domainIncludes` class the same way. A property with no
`rangeIncludes` or no `domainIncludes` admits anything. The domain rule covers the measured
property and the constraint properties, and is skipped when the row names no population type.
A row whose `dcid` the code index serves is compared with the served definition field by field.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from code_index import CodeIndex, StatVarRow
from code_index.constraint_values import constraint_value_fault
from dc_graph import UNRESOLVED_NODE, parse_arcs
from graph_api.ontology_models import (
    CHECK_MAX_READ_IDS,
    ConstraintValue,
    FindingCode,
    OntologyCheckRow,
    OntologyDocument,
    OntologyFinding,
)
from graph_api.reader import GraphNodeReader
from graph_api.routing import DcidConvertor

_CLASS: Final[str] = "Class"
_PROPERTY: Final[str] = "Property"
_READ_PROPERTIES: Final[tuple[str, ...]] = ("typeOf", "subClassOf")
_DCID: Final[re.Pattern[str]] = re.compile(rf"{DcidConvertor.regex}")
_SHAPE_FIELDS: Final[tuple[str, ...]] = (
    "population_type",
    "measured_property",
    "stat_type",
    "measurement_qualifier",
    "measurement_denominator",
)
_CONSTRAINT: Final[str] = "constraint_properties"
_OBSERVATION: Final[str] = "observation_properties"


class OntologyCheckTooManyReads(Exception):
    """The rows name more ids to read from the graph than `CHECK_MAX_READ_IDS`."""

    def __init__(self, count: int) -> None:
        super().__init__(
            f"the rows name {count} distinct ids the ontology document does not hold, past "
            f"the cap of {CHECK_MAX_READ_IDS} one check reads from the graph; send fewer rows"
        )
        self.count = count


@dataclass(frozen=True)
class _Node:
    types: frozenset[str]
    parents: tuple[str, ...]
    instance: bool = False
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()


@dataclass(frozen=True)
class OntologyLookup:
    """The ontology document's nodes as the check reads them, and the instance-declared
    enumerations the document lists in full."""

    nodes: Mapping[str, _Node]
    full_enumerations: frozenset[str]

    @classmethod
    def of(cls, document: OntologyDocument) -> OntologyLookup:
        nodes = {
            node.dcid: _Node(
                types=frozenset(node.types),
                parents=tuple(node.sub_class_of),
                instance=node.declared_by == "instance",
                domain=tuple(node.domain_includes),
                range=tuple(node.range_includes),
            )
            for node in document.nodes
        }
        full = frozenset(
            node.dcid
            for node in document.nodes
            if node.declared_by == "instance"
            and node.enumeration is not None
            and not node.enumeration.truncated
        )
        return cls(nodes=MappingProxyType(nodes), full_enumerations=full)

    def decides(self, prop: str) -> bool:
        """Whether a value outside the document is outside `prop`'s range with no read: `prop`
        is an instance property whose every `rangeIncludes` class is an enumeration the
        document lists in full."""
        node = self.nodes.get(prop)
        return (
            node is not None
            and _ranged(node)
            and all(target in self.full_enumerations for target in node.range)
        )


class _Nodes:
    """The nodes the rows name: document nodes free, the rest read once each."""

    def __init__(self, lookup: OntologyLookup, *, reader: GraphNodeReader, deadline: float) -> None:
        self._lookup = lookup
        self._read: dict[str, _Node | None] = {}
        self._reader = reader
        self._deadline = deadline

    def known(self, dcid: str) -> bool:
        return dcid in self._lookup.nodes or dcid in self._read

    def read(self, dcids: Iterable[str]) -> None:
        """Read in one batch every one of `dcids` not yet known, and again alone each one a
        batch of several answers with no arc."""
        unknown = sorted({dcid for dcid in dcids if not self.known(dcid)})
        if not unknown:
            return
        arcs = dict(self._reader.node_arcs_batch(unknown, deadline=self._deadline))
        if len(unknown) > 1:
            for dcid in [dcid for dcid in unknown if arcs.get(dcid) is None]:
                arcs.update(self._reader.node_arcs_batch([dcid], deadline=self._deadline))
        for dcid in unknown:
            self._read[dcid] = _held(arcs.get(dcid))

    def read_ancestry(self, roots: Collection[str]) -> None:
        """Read `roots` and their `subClassOf` ancestors, one batch per level."""
        seen: set[str] = set()
        level = set(roots)
        while level:
            self.read(level)
            seen |= level
            level = {
                parent
                for dcid in level
                if (node := self.get(dcid)) is not None
                for parent in node.parents
            } - seen

    def get(self, dcid: str) -> _Node | None:
        if (node := self._lookup.nodes.get(dcid)) is not None:
            return node
        return self._read[dcid]

    def ancestry(self, dcids: Iterable[str]) -> set[str]:
        """`dcids` and every `subClassOf` ancestor of them, over the nodes known."""
        seen: set[str] = set()
        stack = list(dcids)
        while stack:
            dcid = stack.pop()
            if dcid in seen:
                continue
            seen.add(dcid)
            if self.known(dcid) and (node := self.get(dcid)) is not None:
                stack.extend(node.parents)
        return seen


def _held(arcs: Mapping[str, object] | None) -> _Node | None:
    """The node `node_arcs_batch` answered with `arcs`, or `None` for no out-arc."""
    if not arcs:
        return None
    values = parse_arcs(dict(arcs), _READ_PROPERTIES, mark_unresolved=True)
    return _Node(
        types=frozenset(value for value in values["typeOf"] if value != UNRESOLVED_NODE),
        parents=tuple(value for value in values["subClassOf"] if value != UNRESOLVED_NODE),
    )


def check_rows(
    rows: Sequence[OntologyCheckRow],
    *,
    lookup: OntologyLookup,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    deadline: float,
) -> list[OntologyFinding]:
    """Find the faults in candidate StatVar definitions.

    Args:
        rows: the candidate definitions.
        lookup: the ontology document's lookup, which answers most ids with no read.
        reader: the graph reads for nodes the document does not hold.
        code_index: the code index whose StatVar rows the rows' dcids are compared with.
        deadline: the absolute monotonic deadline passed to every read.

    Returns:
        One finding per fault, in row order. Within a row: the shape fields in the order
        `population_type`, `measured_property`, `stat_type`, `measurement_qualifier`,
        `measurement_denominator`; the constraint properties sorted, each property's finding
        before its value's; the observation properties in row order; then `definition_differs`
        findings in the same field order.

    Raises:
        OntologyCheckTooManyReads: the rows name more than `CHECK_MAX_READ_IDS` ids to read,
            raised before any read.
        GraphReadError: a read failed.
    """
    nodes = _Nodes(lookup, reader=reader, deadline=deadline)
    to_read = {dcid for row in rows for dcid in _named(row, lookup) if not nodes.known(dcid)}
    if len(to_read) > CHECK_MAX_READ_IDS:
        raise OntologyCheckTooManyReads(len(to_read))
    nodes.read(to_read)
    roots: set[str] = set()
    for row in rows:
        if row.population_type is not None:
            roots.add(row.population_type)
        for prop, value in row.constraint_properties.items():
            read = _read_value(prop, value, lookup)
            if read is not None and _ranged(nodes.get(prop)):
                roots.update(_types(nodes.get(read)))
    nodes.read_ancestry(roots)

    findings: list[OntologyFinding] = []
    for row in rows:
        findings.extend(_check_row(row, nodes, lookup))
        served = code_index.statvar_by_dcid(row.dcid) if row.dcid is not None else None
        if served is not None:
            findings.extend(_differences(row, served))
    return findings


def _is_dcid(value: ConstraintValue) -> bool:
    return isinstance(value, str) and _DCID.fullmatch(value) is not None


def _read_value(prop: str, value: ConstraintValue, lookup: OntologyLookup) -> str | None:
    """`value` when the check answers it from a node, `None` when the value alone decides."""
    if not isinstance(value, str) or not _is_dcid(value):
        return None
    if value not in lookup.nodes and lookup.decides(prop):
        return None
    return value


def _named(row: OntologyCheckRow, lookup: OntologyLookup) -> list[str]:
    """Every node id the check answers the row from."""
    named = [value for field in _SHAPE_FIELDS if (value := getattr(row, field)) is not None]
    for prop, value in row.constraint_properties.items():
        named.append(prop)
        if (read := _read_value(prop, value, lookup)) is not None:
            named.append(read)
    named.extend(row.observation_properties)
    return named


def _types(node: _Node | None) -> frozenset[str]:
    return frozenset() if node is None else node.types


def _ranged(node: _Node | None) -> bool:
    return node is not None and node.instance and _PROPERTY in node.types and bool(node.range)


def _check_row(
    row: OntologyCheckRow, nodes: _Nodes, lookup: OntologyLookup
) -> list[OntologyFinding]:
    findings: list[OntologyFinding] = []

    def find(field: str, value: ConstraintValue, code: FindingCode) -> None:
        severity = "warning" if code == "property_outside_domain" else "error"
        findings.append(
            OntologyFinding(ref=row.ref, field=field, value=value, code=code, severity=severity)
        )

    population: set[str] | None = None
    if row.population_type is not None:
        held = nodes.get(row.population_type)
        if held is None:
            find("population_type", row.population_type, "node_not_held")
        elif _CLASS not in held.types:
            find("population_type", row.population_type, "not_a_class")
        else:
            population = nodes.ancestry([row.population_type])

    def check_property(field: str, prop: str) -> None:
        held = nodes.get(prop)
        if held is None:
            find(field, prop, "node_not_held")
        elif _PROPERTY not in held.types:
            find(field, prop, "not_a_property")
        elif (
            population is not None
            and held.instance
            and held.domain
            and population.isdisjoint(held.domain)
        ):
            find(field, prop, "property_outside_domain")

    check_property("measured_property", row.measured_property)
    for field in ("stat_type", "measurement_qualifier", "measurement_denominator"):
        value = getattr(row, field)
        if value is not None and nodes.get(value) is None:
            find(field, value, "node_not_held")

    for prop, value in sorted(row.constraint_properties.items()):
        field = f"{_CONSTRAINT}.{prop}"
        check_property(field, prop)
        if value is None:
            find(field, value, "value_missing")
            continue
        if not _is_dcid(value):
            find(field, value, "node_not_held")
            continue
        read = _read_value(prop, value, lookup)
        if read is None:
            find(field, value, "value_outside_range")
            continue
        held_value = nodes.get(read)
        fault = constraint_value_fault(None if held_value is None else held_value.types)
        if fault == "not_held":
            find(field, value, "node_not_held")
        elif fault == "untyped":
            find(field, value, "node_untyped")
        elif fault == "provisional":
            find(field, value, "node_provisional")
        elif (
            (prop_node := nodes.get(prop)) is not None
            and _ranged(prop_node)
            and nodes.ancestry(_types(held_value)).isdisjoint(prop_node.range)
        ):
            find(field, value, "value_outside_range")

    for prop in row.observation_properties:
        held = nodes.get(prop)
        if held is None:
            find(_OBSERVATION, prop, "node_not_held")
        elif _PROPERTY not in held.types:
            find(_OBSERVATION, prop, "not_a_property")
    return findings


def _differences(row: OntologyCheckRow, served: StatVarRow) -> list[OntologyFinding]:
    """One `definition_differs` warning per field where the row departs from `served`."""
    differs: list[tuple[str, ConstraintValue | list[str]]] = [
        (field, getattr(served, field))
        for field in _SHAPE_FIELDS
        if getattr(row, field) != getattr(served, field)
    ]
    differs.extend(_constraint_differences(row.constraint_properties, served.constraint_properties))
    if set(row.observation_properties) != set(served.observation_properties):
        differs.append((_OBSERVATION, sorted(served.observation_properties)))
    return [
        OntologyFinding(
            ref=row.ref, field=field, value=value, code="definition_differs", severity="warning"
        )
        for field, value in differs
    ]


def _constraint_differences(
    sent: Mapping[str, ConstraintValue], served: Mapping[str, str]
) -> list[tuple[str, ConstraintValue | list[str]]]:
    return [
        (f"{_CONSTRAINT}.{prop}", served.get(prop))
        for prop in sorted(sent.keys() | served.keys())
        if prop not in sent or prop not in served or sent[prop] != served[prop]
    ]
