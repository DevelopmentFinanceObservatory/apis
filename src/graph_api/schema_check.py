"""Check candidate StatisticalVariable definitions against the deployment's schema.

`SchemaLookup` holds the schema document's nodes as the check reads them, built once per
document. `plan_reads` decides, before any read, how many rows from the first the check takes
and which ids outside the document they read: it adds rows while their distinct ids stay within
`CHECK_MAX_READ_IDS`. `check_rows` answers each node a row names from the document and asks
every other named id once of the reader's `node_arcs_many`, for its `typeOf`, `name` and
`subClassOf`, with no probe budget. The served reader,
`origin_catalogue.graph.OriginGraphClient`, answers an id the origin catalogue's arc tables hold
from those tables, a node they hold with no arc as absent, and reads every other id from the
graph. There an id a batch answers empty is read again alone, so `None` is the graph's answer
for that id read alone. A node with neither a type nor a name reads as absent, and a node with a
name and no `typeOf` is untyped. A read error propagates, so no node is reported absent because
a read failed.

The check reads every planned id in one `node_arcs_many` call: the shape fields, constraint
properties, observation properties and constraint values the document does not hold, less the
cells no read can decide and the values under a document node that is no Property. A
constraint value whose property is not a held Property gets no finding of its own, because the
property's finding marks the cell, so a value under a property outside dcid syntax or outside
the served origins is not read. It then reads, one `node_arcs_many` call a level, the
`subClassOf` parents of each type the range rule judges that lies outside the document, then
of each parent outside it, until every chain reaches the document or ends. Each id is read
once, and an id the first call read needs no second read, since that call asked its
`subClassOf` too. These reads share the budget of `CHECK_MAX_READ_IDS`, taken in the order of
the first row that needs each id, and when a level would pass it the check stops before the
first row whose chain it leaves unread (`_read_ancestry`).

A check reads at most `CHECK_MAX_READ_IDS` (200) ids, each once. It reads fewer from the
graph when the arc tables answer some. Each `node_arcs_many` call sends, for each backend its
ids go to (the mixer, and the UN deployment where the deployment serves the UN origin), one
POST per chunk of 50 of its ids, then one POST per id answered empty, 16 at a time. A call of k
ids therefore sends at most k + ceil(k / 50) POSTs with one backend, 204 for 200 ids, and one
more with two, 205, and each level of ancestry is one more call. A front-door refusal splits a
POST of several ids into two halves, so a chunk of j ids costs at most 2j - 1 POSTs, and a
check of k ids over all its calls at most 3k - 1, 599 for 200 ids, on either kind of
deployment. A continuation page adds one POST. Each POST is tried up to 3 times on a 5xx
answer or a transport error. Every read carries the check's deadline,
`GRAPH_SCHEMA_CHECK_TIMEOUT_SECONDS` (30 s) from the request's start.

No read can decide three kinds of cell, so each gets its finding with no read: a string outside
dcid syntax, a number or a boolean gets `not_a_node`, and an id of the UN origin on a
deployment that does not serve it (`origin_catalogue.graph.is_un_dcid`) gets
`origin_not_served`. Every other id outside the document is read, whatever the range of its
property: the document holds the ancestors of the classes it holds and none of their
descendants, so no value list proves that a value it lacks lies outside a range. One row
names at most `CHECK_MAX_ROW_READ_IDS` (41) ids, fewer than `CHECK_MAX_READ_IDS`, so the
first row always fits.

`check_rows` stops before the row whose findings would take the total past
`CHECK_MAX_FINDINGS`, or the encoded response past `CHECK_MAX_BODY_BYTES`. It measures each
finding as `SchemaCheckResponse` encodes it, plus a separating comma, over the encoded response
with no finding and the widest `rows_checked`. One row yields at most 79 findings
(`CHECK_MAX_FINDINGS` counts them), so the first row fits the findings cap, and the first row is
taken whatever its size.

Every row gets existence and kind checks. A row with no measured property gets `value_missing`.
A constraint value, `stat_type`, `measurement_qualifier` and `measurement_denominator` must each
name a resolved node, by `code_index.constraint_values.constraint_value_fault`, the test the
code index export refuses a release on for a constraint value. Range and domain rules apply
only to a property the document holds with `declared_by == "dfo"`. The range rule judges
through the document, which holds every `subClassOf` ancestor of each class it holds, and
through the parents read for the types outside it. A value is outside the range when it has a
type, the ancestry of every type is known, and none reaches a `rangeIncludes` class by
`subClassOf`. A type whose chain reaches an id neither the document nor the reads hold, left
unread by the budget or an id of an origin not served, gives no range verdict. A property is
outside its domain when the row's population type is a document
node that reaches no `domainIncludes` class by `subClassOf`; a population type outside the
document gets no domain finding. A property with no `rangeIncludes` or no `domainIncludes`
admits anything. The domain rule covers the measured property and the constraint properties,
and is skipped when the row names no population type. A row's `dcid` outside dcid syntax gets
`not_a_node`. A row whose `dcid` the code index serves is compared with the served
definition cell by cell.
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from code_index import CodeIndex, StatVarRow
from code_index.constraint_values import ConstraintValueFault, constraint_value_fault
from dc_graph import UNRESOLVED_NODE
from graph_api.dataset_usage import SHAPE_FIELDS
from graph_api.reader import GraphNodeReader
from graph_api.routing import DcidConvertor
from graph_api.schema_models import (
    CHECK_MAX_BODY_BYTES,
    CHECK_MAX_FINDINGS,
    CHECK_MAX_READ_IDS,
    CHECK_MAX_ROWS,
    ConstraintValue,
    FindingCode,
    FindingField,
    SchemaCheckResponse,
    SchemaCheckRow,
    SchemaDocument,
    SchemaFinding,
)
from origin_catalogue.graph import is_un_dcid

_CLASS: Final[str] = "Class"
_PROPERTY: Final[str] = "Property"
_SUB_CLASS_OF: Final[str] = "subClassOf"
_READ_PROPERTIES: Final[tuple[str, ...]] = ("typeOf", "name", _SUB_CLASS_OF)
_ANCESTRY_PROPERTIES: Final[tuple[str, ...]] = (_SUB_CLASS_OF,)
_DCID: Final[re.Pattern[str]] = re.compile(rf"{DcidConvertor.regex}")
_CONSTRAINT: Final = "constraint_properties"
_OBSERVATION: Final = "observation_properties"


@dataclass(frozen=True)
class _Node:
    types: frozenset[str]
    parents: tuple[str, ...] = ()
    dfo: bool = False
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaLookup:
    """The schema document's nodes as the check reads them."""

    nodes: Mapping[str, _Node]

    @classmethod
    def of(cls, document: SchemaDocument) -> SchemaLookup:
        nodes = {
            node.ref: _Node(
                types=frozenset(node.types),
                parents=tuple(node.sub_class_of),
                dfo=node.declared_by == "dfo",
                domain=tuple(node.domain_includes),
                range=tuple(node.range_includes),
            )
            for node in document.nodes
        }
        return cls(nodes=MappingProxyType(nodes))

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


@dataclass(frozen=True)
class _Unread:
    """Decides, with no read, the cells no read can answer."""

    lookup: SchemaLookup
    un_served: bool

    def fault(self, dcid: str) -> FindingCode | None:
        """`not_a_node` for `dcid` outside dcid syntax, `origin_not_served` for an id of the UN
        origin outside the document while the deployment does not serve it, else `None`."""
        if _DCID.fullmatch(dcid) is None:
            return "not_a_node"
        if not self.un_served and dcid not in self.lookup.nodes and is_un_dcid(dcid):
            return "origin_not_served"
        return None


@dataclass(frozen=True)
class ReadPlan:
    """The rows from the first that one check takes, and the ids outside the document they
    read."""

    row_count: int
    ids: frozenset[str]


def plan_reads(
    rows: Sequence[SchemaCheckRow], *, lookup: SchemaLookup, un_served: bool
) -> ReadPlan:
    """Take rows from the first while the distinct ids they read stay within
    `CHECK_MAX_READ_IDS`, decided with no read. A row reads every shape field, constraint
    property and observation property outside the document, and every constraint value outside
    it that is a dcid, less the cells no read can decide and the values under a property no read
    can decide or under a document node that is no Property. The first row is always taken.

    Args:
        rows: the candidate definitions, at least one.
        lookup: the schema document's lookup.
        un_served: whether the deployment serves the UN origin, whose ids
            `origin_catalogue.graph.is_un_dcid` names.
    """
    unread = _Unread(lookup, un_served)
    ids: set[str] = set()
    for count, row in enumerate(rows):
        taken = ids | _row_reads(row, unread)
        if count and len(taken) > CHECK_MAX_READ_IDS:
            return ReadPlan(row_count=count, ids=frozenset(ids))
        ids = taken
    return ReadPlan(row_count=len(rows), ids=frozenset(ids))


def _row_reads(row: SchemaCheckRow, unread: _Unread) -> set[str]:
    named = [value for field in SHAPE_FIELDS if (value := getattr(row, field)) is not None]
    named.extend(row.observation_properties)
    for prop, value in row.constraint_properties.items():
        named.append(prop)
        if unread.fault(prop) is None and (read := _read_value(prop, value, unread)) is not None:
            named.append(read)
    return {
        dcid for dcid in named if dcid not in unread.lookup.nodes and unread.fault(dcid) is None
    }


class _Nodes:
    """The nodes the rows name: document nodes free, the rest read once each, and the
    `subClassOf` parents of every id read."""

    def __init__(self, lookup: SchemaLookup, *, reader: GraphNodeReader, deadline: float) -> None:
        self.lookup = lookup
        self._read: dict[str, _Node | None] = {}
        self.parents: dict[str, tuple[str, ...]] = {}
        self._reader = reader
        self._deadline = deadline

    def read(self, dcids: Iterable[str]) -> None:
        """Read `dcids` in one `node_arcs_many` call that confirms each empty answer alone."""
        for dcid, arcs in self._arcs(dcids, _READ_PROPERTIES).items():
            self._read[dcid] = _held(arcs)

    def read_parents(self, dcids: Iterable[str]) -> None:
        """Read the `subClassOf` parents of `dcids` in one `node_arcs_many` call that confirms
        each empty answer alone."""
        self._arcs(dcids, _ANCESTRY_PROPERTIES)

    def _arcs(
        self, dcids: Iterable[str], properties: tuple[str, ...]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        unknown = sorted(set(dcids))
        if not unknown:
            return {}
        arcs = self._reader.node_arcs_many(
            unknown, properties=properties, individual_probe_budget=None, deadline=self._deadline
        )
        answers = {dcid: arcs.get(dcid) for dcid in unknown}
        for dcid, answer in answers.items():
            self.parents[dcid] = _resolved(answer, _SUB_CLASS_OF)
        return answers

    def get(self, dcid: str) -> _Node | None:
        if (node := self.lookup.nodes.get(dcid)) is not None:
            return node
        return self._read[dcid]

    def ancestry(self, dcids: Iterable[str]) -> set[str] | None:
        """`dcids` and every `subClassOf` ancestor of them, over the document and the parents
        read, or `None` when the chain of one reaches an id neither holds. A parent a document
        node names that the document lacks ends its chain, since the document holds every
        ancestor the graph holds of each class it holds."""
        seen: set[str] = set()
        stack = [(dcid, False) for dcid in dcids]
        while stack:
            dcid, named_by_document = stack.pop()
            if dcid in seen:
                continue
            seen.add(dcid)
            if (node := self.lookup.nodes.get(dcid)) is not None:
                stack.extend((parent, True) for parent in node.parents)
            elif dcid in self.parents:
                stack.extend((parent, False) for parent in self.parents[dcid])
            elif not named_by_document:
                return None
        return seen


def _resolved(arcs: Mapping[str, tuple[str, ...]] | None, prop: str) -> tuple[str, ...]:
    """The values of `prop` in `arcs`, less the mark of an unresolved node."""
    if arcs is None:
        return ()
    return tuple(value for value in arcs[prop] if value != UNRESOLVED_NODE)


def _held(arcs: Mapping[str, tuple[str, ...]] | None) -> _Node | None:
    """The node `node_arcs_many` answered with `arcs`, or `None` for a node with neither a
    type nor a name."""
    if arcs is None:
        return None
    types = frozenset(_resolved(arcs, "typeOf"))
    if not types and all(value == UNRESOLVED_NODE for value in arcs["name"]):
        return None
    return _Node(types=types)


@dataclass(frozen=True)
class CheckedRows:
    """The rows one check took, from the first, and their findings."""

    rows_checked: int
    findings: list[SchemaFinding]


def check_rows(
    rows: Sequence[SchemaCheckRow],
    *,
    reads: frozenset[str],
    lookup: SchemaLookup,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    un_served: bool,
    generation: str | None,
    deadline: float,
) -> CheckedRows:
    """Find the faults in candidate StatVar definitions, from the first row, stopping before
    the row whose findings would take the total past `CHECK_MAX_FINDINGS`, or the encoded
    response past `CHECK_MAX_BODY_BYTES`. The first row is always taken.

    Args:
        rows: the rows `plan_reads` took.
        reads: the ids `plan_reads` planned for `rows`.
        lookup: the schema document's lookup, which answers most ids with no read.
        reader: the graph reads for nodes the document does not hold, called only when
            `reads` names an id.
        code_index: the code index whose StatVar rows the rows' dcids are compared with.
        un_served: whether the deployment serves the UN origin, as `plan_reads` was told.
        generation: the generation the response carries, which counts toward its size.
        deadline: the absolute monotonic deadline passed to every read.

    Returns:
        The count of rows checked, at least one, and one finding per fault, in row order.
        Within a row: `dcid`; the shape fields in the order `population_type`,
        `measured_property`, `stat_type`, `measurement_qualifier`, `measurement_denominator`;
        the constraint properties sorted, each property's finding before its value's; the
        observation properties in row order; then `definition_differs` findings in the same
        field order.

    Raises:
        GraphReadError: a read failed.
    """
    nodes = _Nodes(lookup, reader=reader, deadline=deadline)
    nodes.read(reads)
    unread = _Unread(lookup, un_served)
    rows = rows[: _read_ancestry(rows, nodes, unread, budget=CHECK_MAX_READ_IDS - len(reads))]

    findings: list[SchemaFinding] = []
    size = _envelope_bytes(generation)
    for index, row in enumerate(rows):
        found = _check_row(index, row, nodes, unread)
        if row.dcid is not None and (served := code_index.statvar_by_dcid(row.dcid)) is not None:
            found.extend(_differences(index, row, served))
        found_size = sum(len(finding.model_dump_json().encode()) + 1 for finding in found)
        if index and (
            len(findings) + len(found) > CHECK_MAX_FINDINGS
            or size + found_size > CHECK_MAX_BODY_BYTES
        ):
            return CheckedRows(rows_checked=index, findings=findings)
        findings.extend(found)
        size += found_size
    return CheckedRows(rows_checked=len(rows), findings=findings)


def _envelope_bytes(generation: str | None) -> int:
    """The encoded response carrying `generation` and no finding, at its widest
    `rows_checked`."""
    envelope = SchemaCheckResponse(generation=generation, rows_checked=CHECK_MAX_ROWS, findings=[])
    return len(envelope.model_dump_json().encode())


def _is_dcid(value: ConstraintValue) -> bool:
    return isinstance(value, str) and _DCID.fullmatch(value) is not None


def _read_value(prop: str, value: ConstraintValue, unread: _Unread) -> str | None:
    """`value` when the check answers it from a node, `None` when the value or the property
    alone decides: a literal, or a value under a document node that is no Property."""
    if not isinstance(value, str) or not _is_dcid(value):
        return None
    nodes = unread.lookup.nodes
    if prop in nodes and not _is_property(nodes[prop]):
        return None
    return value


def _is_property(node: _Node | None) -> bool:
    return node is not None and _PROPERTY in node.types


def _ranged(node: _Node | None) -> bool:
    return node is not None and node.dfo and _is_property(node) and bool(node.range)


def _range_judged(
    prop: str, value: ConstraintValue, nodes: _Nodes, unread: _Unread
) -> tuple[_Node, _Node] | None:
    """The nodes of `value` and `prop` when the range rule judges the cell: `prop` is a
    ranged DFO Property and `value` a resolved node. `None` otherwise."""
    if unread.fault(prop) or not isinstance(value, str) or unread.fault(value):
        return None
    if not _ranged(prop_node := nodes.get(prop)) or prop_node is None:
        return None
    held = nodes.get(value)
    if held is None or _node_fault(held) is not None:
        return None
    return held, prop_node


def _outside_range(value: _Node, prop: _Node, nodes: _Nodes) -> bool:
    """`value` has a type, the ancestry of every one is known, and none reaches `prop`'s
    range."""
    if not value.types or (ancestry := nodes.ancestry(value.types)) is None:
        return False
    return ancestry.isdisjoint(prop.range)


def _read_ancestry(
    rows: Sequence[SchemaCheckRow], nodes: _Nodes, unread: _Unread, *, budget: int
) -> int:
    """Read the `subClassOf` chains of the types the range rule judges that lie outside the
    document, one `node_arcs_many` call per level of the chains, within `budget` ids, and
    return how many rows from the first to take.

    Each id to read carries the first row that needs it, and a level reads its ids in that
    order. When a level's ids pass `budget`, the rows from the first row of an id left unread
    are not taken, but the first row is always taken, and a type of it whose chain is left
    unread gets no range verdict.
    """
    queue: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        for prop, value in row.constraint_properties.items():
            if (judged := _range_judged(prop, value, nodes, unread)) is not None:
                queue.extend((index, dcid) for dcid in judged[0].types)
    heapq.heapify(queue)
    taken = len(rows)
    first_row: dict[str, int] = {}
    while queue:
        level: list[tuple[int, str]] = []
        while queue:
            index, dcid = heapq.heappop(queue)
            if index >= taken or first_row.get(dcid, taken) <= index:
                continue
            first_row[dcid] = index
            if dcid in nodes.lookup.nodes or unread.fault(dcid) is not None:
                continue
            if dcid in nodes.parents:
                queue.extend((index, parent) for parent in nodes.parents[dcid])
            else:
                level.append((index, dcid))
        if len(level) > budget:
            taken = max(1, level[budget][0])
            level = level[:budget]
        nodes.read_parents(dcid for _, dcid in level)
        budget -= len(level)
        queue = [(index, parent) for index, dcid in level for parent in nodes.parents[dcid]]
        heapq.heapify(queue)
    return taken


_NODE_FAULTS: Final[Mapping[ConstraintValueFault, FindingCode]] = MappingProxyType(
    {"not_held": "node_not_held", "untyped": "node_untyped", "provisional": "node_provisional"}
)


def _node_fault(node: _Node | None) -> FindingCode | None:
    """Why `node` is no resolved node, by `constraint_value_fault`, or `None` when it is."""
    fault = constraint_value_fault(None if node is None else node.types)
    return None if fault is None else _NODE_FAULTS[fault]


_WARNINGS: Final[frozenset[FindingCode]] = frozenset(
    {"property_outside_domain", "definition_differs"}
)


def _finding(
    index: int,
    field: FindingField,
    value: ConstraintValue | list[str],
    code: FindingCode,
    *,
    prop: str | None = None,
    subject: Literal["property", "value"] = "value",
    served: ConstraintValue | list[str] = None,
) -> SchemaFinding:
    return SchemaFinding(
        row=index,
        field=field,
        property=prop,
        subject=subject,
        value=value,
        served=served,
        code=code,
        severity="warning" if code in _WARNINGS else "error",
    )


def _check_row(
    index: int, row: SchemaCheckRow, nodes: _Nodes, unread: _Unread
) -> list[SchemaFinding]:
    findings: list[SchemaFinding] = []
    lookup = unread.lookup

    def find(
        field: FindingField,
        value: ConstraintValue,
        code: FindingCode,
        *,
        prop: str | None = None,
        subject: Literal["property", "value"] = "value",
    ) -> None:
        findings.append(_finding(index, field, value, code, prop=prop, subject=subject))

    if row.dcid is not None and _DCID.fullmatch(row.dcid) is None:
        find("dcid", row.dcid, "not_a_node")

    population: set[str] | None = None
    if row.population_type is not None:
        if fault := unread.fault(row.population_type):
            find("population_type", row.population_type, fault)
        elif (held := nodes.get(row.population_type)) is None:
            find("population_type", row.population_type, "node_not_held")
        elif _CLASS not in held.types:
            find("population_type", row.population_type, "not_a_class")
        elif row.population_type in lookup.nodes:
            population = lookup.ancestry([row.population_type])

    def check_property(field: FindingField, prop: str, *, cell: bool) -> bool:
        """Find the property's faults and return whether it is a held Property. `cell` is true
        when `prop` is a constraint property, which the finding then names as its subject."""

        def find_property(code: FindingCode) -> None:
            if cell:
                find(field, prop, code, prop=prop, subject="property")
            else:
                find(field, prop, code)

        if fault := unread.fault(prop):
            find_property(fault)
            return False
        held = nodes.get(prop)
        if held is None:
            find_property("node_not_held")
            return False
        if not _is_property(held):
            find_property("not_a_property")
            return False
        if (
            population is not None
            and held.dfo
            and held.domain
            and population.isdisjoint(held.domain)
        ):
            find_property("property_outside_domain")
        return True

    if row.measured_property is None:
        find("measured_property", None, "value_missing")
    else:
        check_property("measured_property", row.measured_property, cell=False)
    for field in ("stat_type", "measurement_qualifier", "measurement_denominator"):
        value = getattr(row, field)
        if value is None:
            continue
        if fault := unread.fault(value) or _node_fault(nodes.get(value)):
            find(field, value, fault)

    for prop, value in sorted(row.constraint_properties.items()):
        if not check_property(_CONSTRAINT, prop, cell=True):
            continue
        if value is None:
            find(_CONSTRAINT, value, "value_missing", prop=prop)
            continue
        if not isinstance(value, str):
            find(_CONSTRAINT, value, "not_a_node", prop=prop)
            continue
        if fault := unread.fault(value):
            find(_CONSTRAINT, value, fault, prop=prop)
            continue
        if fault := _node_fault(nodes.get(value)):
            find(_CONSTRAINT, value, fault, prop=prop)
        elif (judged := _range_judged(prop, value, nodes, unread)) is not None and _outside_range(
            *judged, nodes
        ):
            find(_CONSTRAINT, value, "value_outside_range", prop=prop)

    for prop in row.observation_properties:
        if (fault := unread.fault(prop)) is None:
            held = nodes.get(prop)
            if held is None:
                fault = "node_not_held"
            elif not _is_property(held):
                fault = "not_a_property"
        if fault is not None:
            find(_OBSERVATION, prop, fault, prop=prop, subject="property")
    return findings


def _differences(index: int, row: SchemaCheckRow, served: StatVarRow) -> list[SchemaFinding]:
    """One `definition_differs` warning per cell where the row departs from `served`, carrying
    what the row sent and what the code index serves."""
    differs = [
        _finding(index, field, sent, "definition_differs", served=getattr(served, field))
        for field in SHAPE_FIELDS
        if (sent := getattr(row, field)) != getattr(served, field)
    ]
    sent_constraints = row.constraint_properties
    served_constraints = served.constraint_properties
    differs.extend(
        _finding(
            index,
            _CONSTRAINT,
            sent_constraints.get(prop),
            "definition_differs",
            prop=prop,
            served=served_constraints.get(prop),
        )
        for prop in sorted(sent_constraints.keys() | served_constraints.keys())
        if prop not in sent_constraints
        or prop not in served_constraints
        or sent_constraints[prop] != served_constraints[prop]
    )
    if set(row.observation_properties) != set(served.observation_properties):
        differs.append(
            _finding(
                index,
                _OBSERVATION,
                sorted(row.observation_properties),
                "definition_differs",
                served=sorted(served.observation_properties),
            )
        )
    return differs
