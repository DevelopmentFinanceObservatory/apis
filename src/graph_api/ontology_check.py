"""Check candidate StatisticalVariable definitions against the instance's ontology.

`OntologyLookup` holds the ontology document's nodes as the check reads them, built once per
document. `check_rows` answers each node a row names from it and reads every other named id
once, through `node_arcs_batch` with `confirm_empty`, which reads each id a batch answers empty
again alone, so an empty answer means the graph does not hold it. A node held with any out-arc
is held, its types its `typeOf` values, so a held node with no `typeOf` is untyped. A read
error propagates, so no node is reported absent because a read failed. The check reads no
ancestry.

The reads come in two rounds. The first reads the shape fields, the constraint properties and
the observation properties the document does not hold, and the values of constraint properties
the document holds as Properties. The second reads the values of constraint properties the
first round found held as Properties. A constraint value whose property is not a held Property
is never read and gets no finding of its own: the property's finding marks the cell. Each
round is one `node_arcs_batch` call, so a request makes at most two; inside each, the client
sends one batch, one group retry of the empty answers and one read alone per id still empty.

Some constraint values need no read. A value that is a number, a boolean, or a string outside
dcid syntax names no node: `node_not_held`. A value outside the document, on an instance
property whose range the document decides (`OntologyLookup.decides`), is no member of the
range: `value_outside_range`. Every other id outside the document is read. The ids a request
could read are counted before any read, every value under a property outside the document
included, and a request naming more than `CHECK_MAX_READ_IDS` raises
`OntologyCheckTooManyReads`. One row names at most `CHECK_MAX_READ_IDS` ids, so a single row
always fits.

Every row gets existence and kind checks. A constraint value must name a resolved node, by
`code_index.constraint_values.constraint_value_fault`, the test the code index export refuses
a release on. Range and domain rules apply only to a property the document declares
`declared_by == "instance"`, and judge only through the document, which holds every
`subClassOf` ancestor of each class it holds. A value is outside the range when every one of
its types is a document node and none reaches a `rangeIncludes` class by `subClassOf`; a value
with any type outside the document gets no range finding. A property is outside its domain
when the row's population type is a document node that reaches no `domainIncludes` class by
`subClassOf`; a population type outside the document gets no domain finding. A property with
no `rangeIncludes` or no `domainIncludes` admits anything. The domain rule covers the measured
property and the constraint properties, and is skipped when the row names no population type.
A row whose `dcid` the code index serves is compared with the served definition field by field.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
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
    """The ontology document's nodes as the check reads them, and the classes whose instances
    the document lists in full: instance-declared enumerations listed in full whose every
    enumeration below them in the document by `subClassOf` is one too."""

    nodes: Mapping[str, _Node]
    listed_in_full: frozenset[str]

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
        enumerations = [node for node in document.nodes if node.enumeration is not None]
        full = {
            node.dcid
            for node in enumerations
            if node.declared_by == "instance"
            and node.enumeration is not None
            and not node.enumeration.truncated
        }
        for node in enumerations:
            if node.dcid not in full:
                full -= _ancestry(nodes, [node.dcid])
        return cls(nodes=MappingProxyType(nodes), listed_in_full=frozenset(full))

    def decides(self, prop: str) -> bool:
        """Whether a value outside the document is outside `prop`'s range with no read: `prop`
        is an instance property whose every `rangeIncludes` class the document lists in full,
        with every enumeration below it."""
        node = self.nodes.get(prop)
        return (
            node is not None
            and _ranged(node)
            and all(target in self.listed_in_full for target in node.range)
        )

    def ancestry(self, dcids: Iterable[str]) -> set[str]:
        """`dcids` and every `subClassOf` ancestor of them the document holds."""
        return _ancestry(self.nodes, dcids)


def _ancestry(nodes: Mapping[str, _Node], dcids: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(dcids)
    while stack:
        dcid = stack.pop()
        if dcid in seen:
            continue
        seen.add(dcid)
        if (node := nodes.get(dcid)) is not None:
            stack.extend(node.parents)
    return seen


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
        """Read every one of `dcids` not yet known in one `node_arcs_batch` call that confirms
        each empty answer alone."""
        unknown = sorted({dcid for dcid in dcids if not self.known(dcid)})
        if not unknown:
            return
        arcs = self._reader.node_arcs_batch(unknown, confirm_empty=True, deadline=self._deadline)
        for dcid in unknown:
            self._read[dcid] = _held(arcs.get(dcid))

    def get(self, dcid: str) -> _Node | None:
        if (node := self._lookup.nodes.get(dcid)) is not None:
            return node
        return self._read[dcid]


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
    first: set[str] = set()
    pending: dict[str, set[str]] = {}
    for row in rows:
        first.update(_named(row))
        for prop, value in row.constraint_properties.items():
            if (read := _read_value(prop, value, lookup)) is None:
                continue
            if prop in lookup.nodes:
                first.add(read)
            else:
                pending.setdefault(prop, set()).add(read)
    to_read = {dcid for dcid in first.union(*pending.values()) if not nodes.known(dcid)}
    if len(to_read) > CHECK_MAX_READ_IDS:
        raise OntologyCheckTooManyReads(len(to_read))
    nodes.read(first)
    nodes.read(
        value
        for prop, values in pending.items()
        if _is_property(nodes.get(prop))
        for value in values
    )

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
    """`value` when the check answers it from a node, `None` when the value or the property
    alone decides: a literal, a value under a document node that is no Property, or a value
    outside the document whose range the document decides."""
    if not isinstance(value, str) or not _is_dcid(value):
        return None
    if prop in lookup.nodes and not _is_property(lookup.nodes[prop]):
        return None
    if value not in lookup.nodes and lookup.decides(prop):
        return None
    return value


def _named(row: OntologyCheckRow) -> list[str]:
    """Every node id the row names other than its constraint values."""
    named = [value for field in _SHAPE_FIELDS if (value := getattr(row, field)) is not None]
    named.extend(row.constraint_properties)
    named.extend(row.observation_properties)
    return named


def _is_property(node: _Node | None) -> bool:
    return node is not None and _PROPERTY in node.types


def _ranged(node: _Node | None) -> bool:
    return node is not None and node.instance and _is_property(node) and bool(node.range)


def _outside_range(value: _Node, prop: _Node, lookup: OntologyLookup) -> bool:
    """Every type of `value` is a document node and none reaches `prop`'s range."""
    if not value.types or any(dcid not in lookup.nodes for dcid in value.types):
        return False
    return lookup.ancestry(value.types).isdisjoint(prop.range)


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
        elif row.population_type in lookup.nodes:
            population = lookup.ancestry([row.population_type])

    def check_property(field: str, prop: str) -> bool:
        """Find the property's faults; whether it is a held Property."""
        held = nodes.get(prop)
        if held is None:
            find(field, prop, "node_not_held")
            return False
        if not _is_property(held):
            find(field, prop, "not_a_property")
            return False
        if (
            population is not None
            and held.instance
            and held.domain
            and population.isdisjoint(held.domain)
        ):
            find(field, prop, "property_outside_domain")
        return True

    check_property("measured_property", row.measured_property)
    for field in ("stat_type", "measurement_qualifier", "measurement_denominator"):
        value = getattr(row, field)
        if value is not None and nodes.get(value) is None:
            find(field, value, "node_not_held")

    for prop, value in sorted(row.constraint_properties.items()):
        field = f"{_CONSTRAINT}.{prop}"
        if not check_property(field, prop):
            continue
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
            held_value is not None
            and _ranged(prop_node := nodes.get(prop))
            and prop_node is not None
            and _outside_range(held_value, prop_node, lookup)
        ):
            find(field, value, "value_outside_range")

    for prop in row.observation_properties:
        held = nodes.get(prop)
        if held is None:
            find(_OBSERVATION, prop, "node_not_held")
        elif not _is_property(held):
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
