"""Build one node's view of the graph from live reads: the node and its schema neighbours.

`build_node_view` serves any node the graph holds, base, UN or DFO, and raises
`NodeNotHeldError` for a dcid the graph answers with no arc. Every arc in the view is one the
graph answered.

Every read runs on the executor `build_node_view` is given, under a copy of the caller's
context, so a reader that needs the bound origin catalogue, as the UN deployment's client does
for its cache generation, finds it on the worker. Each read carries the view's deadline and
caps each decoded response at `NODE_VIEW_MAX_BYTES`. A read the graph answers with HTTP 429 is
sent again after each of `RATE_LIMIT_WAITS_SECONDS`, one budget for the whole view, while the
wait ends before the deadline. A read with no dcid to read is not sent, and a view that ends,
by an answer or a failure, cancels every read it submitted that has not started.

Round one runs three reads at once: `->*` on the centre (`node_all_arc_targets_many`), the
centre's in-arcs under `subClassOf`, `subPropertyOf`, `specializationOf`, `rangeIncludes` and
`domainIncludes` (`node_in_arcs`, each listing capped at `IN_ARC_CAP`), and the centre's
`->subClassOf+` closure (`node_closure`). The centre is a full `SchemaNode` built as the schema
document builds its nodes, with `declared_by` from `schema_document.declared_by`. Each node
target of the centre's arcs is a neighbour: those of `typeOf` and the five schema arcs, and
those `other_properties` keeps, such as a UN series' `unConcept`. Each in-arc subject is a
neighbour carrying the one arc that ties it to the centre.

A Class whose closure holds `Enumeration` is a value list. Its round two lists its `<-typeOf`
members, capped at `MEMBER_CAP`, and each member joins as a neighbour. When `Enumeration` is
not a direct parent it also reads the direct `subClassOf` edges of the closure, so the chain
from the centre to `Enumeration` in the view is the graph's. The centre's `instances` holds the
members and is complete only when the listing read every one and the centre has no narrower
class, since a narrower class's members are not listed. Each narrower class and each chain
parent that reaches `Enumeration` gets `instances: {refs: [], complete: false}`. No other node
has its `<-typeOf` read, so a class such as `Place` lists no members.

A Property reads in round two the `->subClassOf+` closure of each of its first
`RANGE_CLASS_CAP` range classes other than `Thing`, sorted, one read each, and a range class
whose closure holds `Enumeration` gets `instances: {refs: [], complete: false}`, so a range
class the schema document lacks still reads as a value list.

A value, a node typed neither Class nor Property, reads in round two the `<-rangeIncludes`
subjects of its first `VALUE_TYPE_CAP` type classes other than `Thing`, sorted, in one read,
and the `->subClassOf+` closure of each, one read each. Each `rangeIncludes` subject joins as a
neighbour. A type class whose closure holds `Enumeration` is the value's list: round three
reads the direct `subClassOf` edges of every such class and its closure in one batch, and the
class gets `instances: {refs: [centre], complete: false}`.

A neighbour's `provenances` are those of the arcs the view read from it: a member's `typeOf`,
an in-arc subject's arc to the centre or to a type class, and a chain class's `subClassOf`
edges. Its `declared_by` follows from them by `schema_document.declared_by`, and is null for a
neighbour the view read no arc from, such as an out-arc target of the centre, since the arc
the centre holds to it names the centre's declarer. `capped` names each in-arc listing that
stopped at its cap by the node listed and the property: the centre's schema arcs and members,
and a value's type classes under `rangeIncludes`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from typing import Final, Protocol

from dc_graph import (
    UNRESOLVED_NODE,
    ArcTarget,
    GraphReadError,
    InArcs,
    InArcSubject,
    display_name,
)
from dfo_platform.executor import submit_with_context
from graph_api.node_view_models import CappedListing, SchemaNodeView
from graph_api.schema_document import (
    NODE_PROPERTIES,
    declared_by,
    node_other_properties,
    resolved_entry,
    schema_node,
)
from graph_api.schema_models import SchemaInstances, SchemaNode, SchemaNodeReference
from origin_catalogue import Catalogue

logger = logging.getLogger(__name__)

IN_ARC_CAP: Final[int] = 1_000
"""The most subjects one in-arc listing reads for one property."""

MEMBER_CAP: Final[int] = 1_000
"""The most `<-typeOf` members a value list's view lists."""

VALUE_TYPE_CAP: Final[int] = 3
"""The most type classes a value's view reads its list for, the first sorted, so a value's view
makes at most `3 + 1 + VALUE_TYPE_CAP + 1` reads."""

RANGE_CLASS_CAP: Final[int] = 3
"""The most range classes a property's view reads the closure of, the first sorted, so a
property's view makes at most `3 + RANGE_CLASS_CAP` reads."""

READS_AT_ONCE: Final[int] = 3
"""The reads a view's first round runs at once, the most any later round runs but a value's
second, which runs `1 + VALUE_TYPE_CAP`."""

NODE_VIEW_MAX_BYTES: Final[int] = 2 * 1024 * 1024
"""The most decoded bytes one response of a view's reads may carry. A larger one fails the
read."""

RATE_LIMIT_WAITS_SECONDS: Final[tuple[float, ...]] = (1.0, 2.0)
"""The waits before each further request for a read the graph answered with HTTP 429, one
budget for the whole view, so a view waits 3 s at most for 429s."""

IN_ARCS: Final[tuple[str, ...]] = (
    "subClassOf",
    "subPropertyOf",
    "specializationOf",
    "rangeIncludes",
    "domainIncludes",
)
"""The schema arcs whose subjects join a centre's view."""

_FIELDS: Final[Mapping[str, str]] = {
    "subClassOf": "sub_class_of",
    "subPropertyOf": "sub_property_of",
    "specializationOf": "specialization_of",
    "rangeIncludes": "range_includes",
    "domainIncludes": "domain_includes",
}
_TYPE_OF: Final[str] = "typeOf"
_SUB_CLASS_OF: Final[str] = "subClassOf"
_RANGE_INCLUDES: Final[str] = "rangeIncludes"
_CLASS: Final[str] = "Class"
_PROPERTY: Final[str] = "Property"
_ENUMERATION: Final[str] = "Enumeration"
_THING: Final[str] = "Thing"
_RATE_LIMITED: Final[int] = 429
_NO_IN_ARCS: Final[InArcs] = InArcs(subjects=(), capped=frozenset())

type _Targets = dict[str, dict[str, tuple[ArcTarget, ...]] | None]


class NodeViewReader(Protocol):
    """The graph reads `build_node_view` makes. `dc_graph.GraphClient` satisfies it."""

    def node_all_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> _Targets:
        """Every outgoing property's targets (`->*`), `None` for a dcid with no arc."""
        ...

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> _Targets:
        """The named properties' targets, `None` for an absent dcid."""
        ...

    def node_in_arcs(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        max_results: int,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> Mapping[str, InArcs]:
        """The subjects of each node's in-arcs under `properties`, each with its name and
        types, capped at `max_results` per node and property."""
        ...

    def node_closure(
        self,
        dcid: str,
        *,
        property: str,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> Sequence[ArcTarget]:
        """The nodes `->{property}+` reaches from `dcid`."""
        ...


class NodeNotHeldError(LookupError):
    """The graph answers the dcid with no arc, so it holds no such node."""

    def __init__(self, dcid: str) -> None:
        super().__init__(f"the graph holds no node {dcid!r}")
        self.dcid = dcid


def _done[R](value: R) -> Future[R]:
    future: Future[R] = Future()
    future.set_result(value)
    return future


class _Reads:
    """The reads of one view, each submitted to `executor` under a copy of the caller's
    context with the view's deadline and byte cap, sharing one 429 budget."""

    def __init__(self, reader: NodeViewReader, *, executor: Executor, deadline: float) -> None:
        self._reader = reader
        self._executor = executor
        self._deadline = deadline
        self._cancels: list[Callable[[], bool]] = []
        self._waits: Iterator[float] = iter(RATE_LIMIT_WAITS_SECONDS)
        self._waits_lock = threading.Lock()

    def out_arcs(self, dcid: str) -> Future[_Targets]:
        return self._submit(
            lambda: self._reader.node_all_arc_targets_many(
                [dcid], max_bytes=NODE_VIEW_MAX_BYTES, deadline=self._deadline
            )
        )

    def in_arcs(
        self, dcids: Sequence[str], *, properties: Sequence[str], cap: int
    ) -> Future[Mapping[str, InArcs]]:
        if not dcids:
            return _done({})
        return self._submit(
            lambda: self._reader.node_in_arcs(
                dcids,
                properties=properties,
                max_results=cap,
                max_bytes=NODE_VIEW_MAX_BYTES,
                deadline=self._deadline,
            )
        )

    def closure(self, dcid: str) -> Future[frozenset[str]]:
        """The dcids `->subClassOf+` reaches from `dcid`."""
        return self._submit(
            lambda: frozenset(
                target.value
                for target in self._reader.node_closure(
                    dcid,
                    property=_SUB_CLASS_OF,
                    max_bytes=NODE_VIEW_MAX_BYTES,
                    deadline=self._deadline,
                )
            )
        )

    def edges(self, dcids: Sequence[str]) -> Future[_Targets]:
        """The direct `subClassOf` targets of each of `dcids`."""
        if not dcids:
            return _done({})
        return self._submit(
            lambda: self._reader.node_arc_targets_many(
                dcids,
                properties=(_SUB_CLASS_OF,),
                max_bytes=NODE_VIEW_MAX_BYTES,
                deadline=self._deadline,
            )
        )

    def cancel(self) -> None:
        """Cancel every read submitted that has not started."""
        for cancel in self._cancels:
            cancel()

    def _submit[R](self, read: Callable[[], R]) -> Future[R]:
        def run() -> R:
            return self._waiting_out_rate_limits(read)

        future = submit_with_context(self._executor, run)
        self._cancels.append(future.cancel)
        return future

    def _waiting_out_rate_limits[R](self, read: Callable[[], R]) -> R:
        while True:
            try:
                return read()
            except GraphReadError as exc:
                if exc.upstream_status != _RATE_LIMITED:
                    raise
                with self._waits_lock:
                    wait = next(self._waits, None)
                if wait is None or time.monotonic() + wait >= self._deadline:
                    raise
                logger.info("node view: a read answered 429, waiting %.0f s", wait)
                time.sleep(wait)


@dataclass
class _Neighbour:
    names: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    arcs: dict[str, set[str]] = field(default_factory=dict)
    provenances: set[str] = field(default_factory=set)
    instances: SchemaInstances | None = None


class _Neighbours:
    """The view's neighbours by dcid, each merged over every answer that names it. The centre
    is left out."""

    def __init__(self, centre: str) -> None:
        self._centre = centre
        self.by_ref: dict[str, _Neighbour] = {}

    def add(
        self,
        ref: str,
        *,
        names: Iterable[str] = (),
        types: Iterable[str] = (),
        provenances: Iterable[str] = (),
    ) -> None:
        """Add `ref` with `provenances`, those of arcs read from `ref` itself."""
        if ref == self._centre:
            return
        neighbour = self.by_ref.setdefault(ref, _Neighbour())
        neighbour.names.update(names)
        neighbour.types.update(types)
        neighbour.provenances.update(provenances)

    def add_target(self, target: ArcTarget) -> None:
        """Add a node target with the name and types the answer carries, and none of its
        provenances, which name the declarer of the arc to it. A literal and an unresolved
        target are left out."""
        if target.literal or target.value == UNRESOLVED_NODE:
            return
        self.add(
            target.value, names=() if target.name is None else (target.name,), types=target.types
        )

    def add_subject(self, subject: InArcSubject) -> None:
        self.add(
            subject.dcid,
            names=() if subject.name is None else (subject.name,),
            types=subject.types,
            provenances=subject.provenances,
        )

    def link(self, ref: str, prop: str, target: str) -> None:
        """Record the arc `ref -prop-> target` on neighbour `ref`."""
        if (neighbour := self.by_ref.get(ref)) is not None:
            neighbour.arcs.setdefault(_FIELDS[prop], set()).add(target)

    def mark_list(self, ref: str, instances: SchemaInstances) -> None:
        if (neighbour := self.by_ref.get(ref)) is not None:
            neighbour.instances = instances

    def nodes(self, catalogue: Catalogue) -> list[SchemaNode]:
        return [
            _neighbour_node(ref, neighbour, catalogue)
            for ref, neighbour in sorted(self.by_ref.items())
        ]


def build_node_view(
    dcid: str,
    *,
    reader: NodeViewReader,
    catalogue: Catalogue,
    generation: str | None,
    executor: Executor,
    deadline: float,
) -> SchemaNodeView:
    """The view of node `dcid`: the node and its schema neighbours, read from `reader` on
    `executor`, the reads of a round at once.

    Raises:
        NodeNotHeldError: the graph answers `dcid` with no arc.
        GraphReadError: a read failed.
    """
    reads = _Reads(reader, executor=executor, deadline=deadline)
    try:
        return _build(dcid, reads=reads, catalogue=catalogue, generation=generation)
    finally:
        reads.cancel()


def _build(
    dcid: str, *, reads: _Reads, catalogue: Catalogue, generation: str | None
) -> SchemaNodeView:
    out_read = reads.out_arcs(dcid)
    in_read = reads.in_arcs([dcid], properties=IN_ARCS, cap=IN_ARC_CAP)
    closure_read = reads.closure(dcid)
    entry = resolved_entry(out_read.result().get(dcid))
    if entry is None:
        raise NodeNotHeldError(dcid)
    in_arcs = in_read.result().get(dcid, _NO_IN_ARCS)
    closure = closure_read.result()

    arcs = {
        prop: tuple(dict.fromkeys(target.value for target in entry.get(prop, ())))
        for prop in NODE_PROPERTIES
    }
    other_properties = node_other_properties(dcid, entry, {})
    types = set(arcs[_TYPE_OF])
    neighbours = _Neighbours(dcid)
    capped: set[tuple[str, str]] = set()
    for prop in (_TYPE_OF, *IN_ARCS):
        for target in entry.get(prop, ()):
            neighbours.add_target(target)
    for prop, values in other_properties.items():
        kept = {value.ref for value in values if isinstance(value, SchemaNodeReference)}
        for target in entry.get(prop, ()):
            if target.value in kept:
                neighbours.add_target(target)
    capped.update((dcid, prop) for prop in in_arcs.capped & set(_FIELDS))
    for subject in in_arcs.subjects:
        if subject.property in _FIELDS:
            neighbours.add_subject(subject)
            neighbours.link(subject.dcid, subject.property, dcid)

    instances: SchemaInstances | None = None
    if _CLASS in types and _ENUMERATION in closure:
        instances = _read_value_list(
            dcid,
            parents=arcs[_SUB_CLASS_OF],
            closure=closure,
            in_arcs=in_arcs,
            reads=reads,
            neighbours=neighbours,
            capped=capped,
        )
    elif _PROPERTY in types:
        _read_range_lists(
            sorted(set(arcs[_RANGE_INCLUDES]) - {_THING})[:RANGE_CLASS_CAP],
            reads=reads,
            neighbours=neighbours,
        )
    elif _CLASS not in types:
        _read_value(
            dcid,
            type_classes=sorted(types - {_THING})[:VALUE_TYPE_CAP],
            reads=reads,
            neighbours=neighbours,
            capped=capped,
        )

    provenances = {
        provenance for target in entry.get(_TYPE_OF, ()) for provenance in target.provenances
    }
    centre = schema_node(dcid, arcs, other_properties, provenances, instances, catalogue)
    return SchemaNodeView(
        generation=generation,
        centre=centre,
        nodes=neighbours.nodes(catalogue),
        capped=[CappedListing(ref=ref, property=prop) for ref, prop in sorted(capped)],
    )


def _read_value_list(
    dcid: str,
    *,
    parents: Collection[str],
    closure: Collection[str],
    in_arcs: InArcs,
    reads: _Reads,
    neighbours: _Neighbours,
    capped: set[tuple[str, str]],
) -> SchemaInstances:
    """Round two of value list `dcid`: its members, and the chain to `Enumeration` when that is
    not a direct parent. Returns the centre's `instances`."""
    members_read = reads.in_arcs([dcid], properties=(_TYPE_OF,), cap=MEMBER_CAP)
    edges_read = (
        None if _ENUMERATION in parents else reads.edges(sorted(set(closure) - {_ENUMERATION}))
    )
    members = members_read.result().get(dcid, _NO_IN_ARCS)
    if edges_read is not None:
        edges = _join_chain(edges_read.result(), neighbours)
        edges[dcid] = set(parents)
        _mark_chain(edges, exclude=dcid, neighbours=neighbours)

    member_subjects = [subject for subject in members.subjects if subject.property == _TYPE_OF]
    for subject in member_subjects:
        neighbours.add_subject(subject)
    if _TYPE_OF in members.capped:
        capped.add((dcid, _TYPE_OF))

    narrower = {subject.dcid for subject in in_arcs.subjects if subject.property == _SUB_CLASS_OF}
    for ref in narrower:
        neighbours.mark_list(ref, SchemaInstances(refs=[], complete=False))
    return SchemaInstances(
        refs=sorted({subject.dcid for subject in member_subjects}),
        complete=not members.capped and not narrower,
    )


def _read_range_lists(
    range_classes: Sequence[str], *, reads: _Reads, neighbours: _Neighbours
) -> None:
    """Round two of a property: the closure of each of `range_classes`, marking each that
    descends from `Enumeration` as a value list."""
    closure_reads = {range_class: reads.closure(range_class) for range_class in range_classes}
    for range_class, closure_read in closure_reads.items():
        if _ENUMERATION in closure_read.result():
            neighbours.mark_list(range_class, SchemaInstances(refs=[], complete=False))


def _read_value(
    dcid: str,
    *,
    type_classes: Sequence[str],
    reads: _Reads,
    neighbours: _Neighbours,
    capped: set[tuple[str, str]],
) -> None:
    """Rounds two and three of value `dcid`: each type class's `<-rangeIncludes` subjects and
    closure, then the chain of each type class that descends from `Enumeration`, which becomes
    the value's list."""
    range_read = reads.in_arcs(type_classes, properties=(_RANGE_INCLUDES,), cap=IN_ARC_CAP)
    closure_reads = [reads.closure(type_class) for type_class in type_classes]
    lists: list[str] = []
    chain: set[str] = set()
    ranges = range_read.result()
    for type_class, closure_read in zip(type_classes, closure_reads, strict=True):
        listing = ranges.get(type_class, _NO_IN_ARCS)
        if _RANGE_INCLUDES in listing.capped:
            capped.add((type_class, _RANGE_INCLUDES))
        for subject in listing.subjects:
            if subject.property == _RANGE_INCLUDES:
                neighbours.add_subject(subject)
                neighbours.link(subject.dcid, _RANGE_INCLUDES, type_class)
        closure = closure_read.result()
        if _ENUMERATION in closure:
            lists.append(type_class)
            chain |= {type_class, *closure} - {_ENUMERATION}
    if not lists:
        return

    edges = _join_chain(reads.edges(sorted(chain)).result(), neighbours)
    _mark_chain(edges, exclude=dcid, neighbours=neighbours)
    for type_class in lists:
        neighbours.mark_list(type_class, SchemaInstances(refs=[dcid], complete=False))


def _join_chain(
    answer: Mapping[str, Mapping[str, Sequence[ArcTarget]] | None], neighbours: _Neighbours
) -> dict[str, set[str]]:
    """Add each class the edges `answer` read and each `subClassOf` parent it names as
    neighbours, link each edge, and return the parents by class."""
    edges: dict[str, set[str]] = {}
    for ref, read in answer.items():
        targets = [
            target
            for target in (read or {}).get(_SUB_CLASS_OF, ())
            if not target.literal and target.value != UNRESOLVED_NODE
        ]
        edges[ref] = {target.value for target in targets}
        neighbours.add(
            ref, provenances=(provenance for target in targets for provenance in target.provenances)
        )
        for target in targets:
            neighbours.add_target(target)
            neighbours.link(ref, _SUB_CLASS_OF, target.value)
    return edges


def _mark_chain(
    edges: Mapping[str, Collection[str]], *, exclude: str, neighbours: _Neighbours
) -> None:
    """Mark each class of `edges` whose read `subClassOf` edges reach `Enumeration` as listing
    instances it has not listed, `exclude` left alone."""
    reaches: dict[str, bool] = {}

    def reach(ref: str, path: frozenset[str]) -> bool:
        if ref == _ENUMERATION:
            return True
        if ref in reaches:
            return reaches[ref]
        if ref in path:
            return False
        reaches[ref] = any(reach(parent, path | {ref}) for parent in edges.get(ref, ()))
        return reaches[ref]

    for ref in edges:
        if ref != exclude and reach(ref, frozenset()):
            neighbours.mark_list(ref, SchemaInstances(refs=[], complete=False))


def _neighbour_node(ref: str, neighbour: _Neighbour, catalogue: Catalogue) -> SchemaNode:
    arcs = {name: sorted(neighbour.arcs.get(name, ())) for name in _FIELDS.values()}
    return SchemaNode(
        ref=ref,
        label=display_name(neighbour.names, ref),
        description=None,
        types=sorted(neighbour.types),
        alternate_name=[],
        sub_class_of=arcs["sub_class_of"],
        sub_property_of=arcs["sub_property_of"],
        specialization_of=arcs["specialization_of"],
        domain_includes=arcs["domain_includes"],
        range_includes=arcs["range_includes"],
        classification_code=[],
        other_properties={},
        provenances=sorted(neighbour.provenances),
        declared_by=(
            declared_by(neighbour.provenances, catalogue=catalogue)
            if neighbour.provenances
            else None
        ),
        instances=neighbour.instances,
    )
