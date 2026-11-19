"""Build one node's view of the graph from live reads: the node and its schema neighbours.

`build_node_view` serves any node the graph holds, base, UN or DFO, and raises
`NodeNotHeldError` for a dcid the graph answers with no arc. Every arc in the view is one the
graph answered.

Round one runs three reads at once: `->*` on the centre (`node_all_arc_targets_many`), the
centre's in-arcs under `subClassOf`, `subPropertyOf`, `specializationOf`, `rangeIncludes` and
`domainIncludes` (`node_in_arcs`, each listing capped at `IN_ARC_CAP`), and the centre's
`->subClassOf+` closure (`node_closure`). The centre is a full `SchemaNode` built as the schema
document builds its nodes, with `declared_by` from `schema_document.declared_by`. Each node
target of the centre's `typeOf` and five schema arcs is a neighbour, and each in-arc subject is a neighbour carrying the one arc that ties it to the
centre.

A Class whose closure holds `Enumeration` is a value list. Its round two lists its `<-typeOf`
members, capped at `MEMBER_CAP`, and each member joins as a neighbour; when `Enumeration` is
not a direct parent it also reads the direct `subClassOf` edges of the closure, so the chain
from the centre to `Enumeration` in the view is the graph's. The centre's `instances` holds the
members and is complete only when the listing read every one and the centre has no narrower
class, since a narrower class's members are not listed. Each narrower class and each chain
parent that reaches `Enumeration` gets `instances: {refs: [], complete: false}`. No other node
has its `<-typeOf` read, so a class such as `Place` never requests its members.

A value, a node typed neither Class nor Property, reads in round two the `<-rangeIncludes`
subjects of its first `VALUE_TYPE_CAP` type classes other than `Thing`, sorted, in one read, and
the `->subClassOf+` closure of each. Each `rangeIncludes` subject joins as a neighbour. A type class
whose closure holds `Enumeration` is the value's list: round three reads the direct
`subClassOf` edges of every such class and its closure in one batch, and the class gets
`instances: {refs: [centre], complete: false}`.

A neighbour's `provenances` are those of the arcs the view read from it: a member's `typeOf`,
an in-arc subject's arc to the centre or to a type class, and a chain class's `subClassOf`
edges. Its `declared_by` follows from them by `schema_document.declared_by`, and is null for a
neighbour the view read no arc from, such as an out-arc target of the centre, since the arc
the centre holds to it names the centre's declarer. `capped` names each in-arc property whose
listing stopped at its cap.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Final, Protocol

from dc_graph import UNRESOLVED_NODE, ArcTarget, InArcs, InArcSubject, display_name
from graph_api.node_view_models import SchemaNodeView
from graph_api.schema_document import (
    NODE_PROPERTIES,
    declared_by,
    node_other_properties,
    resolved_entry,
    schema_node,
)
from graph_api.schema_models import SchemaInstances, SchemaNode
from origin_catalogue import Catalogue

IN_ARC_CAP: Final[int] = 1_000
"""The most subjects one in-arc listing reads for one property."""

MEMBER_CAP: Final[int] = 1_000
"""The most `<-typeOf` members a value list's view lists."""

VALUE_TYPE_CAP: Final[int] = 3
"""The most type classes a value's view reads its list for, the first sorted, so a value's view
makes at most `3 + 1 + VALUE_TYPE_CAP + 1` reads."""

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
_NO_IN_ARCS: Final[InArcs] = InArcs(subjects=(), capped=frozenset())


class NodeViewReader(Protocol):
    """The graph reads `build_node_view` makes. `dc_graph.GraphClient` satisfies it."""

    def node_all_arc_targets_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        """Every outgoing property's targets (`->*`), `None` for a dcid with no arc."""
        ...

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        """The named properties' targets, `None` for an absent dcid."""
        ...

    def node_in_arcs(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        max_results: int,
        deadline: float | None = None,
    ) -> Mapping[str, InArcs]:
        """The subjects of each node's in-arcs under `properties` (`<-[p1,...]`), each with its
        name and types, capped at `max_results` per node and property."""
        ...

    def node_closure(
        self, dcid: str, *, property: str, deadline: float | None = None
    ) -> Sequence[ArcTarget]:
        """The nodes `->{property}+` reaches from `dcid`."""
        ...


class NodeNotHeldError(LookupError):
    """The graph answers the dcid with no arc, so it holds no such node."""

    def __init__(self, dcid: str) -> None:
        super().__init__(f"the graph holds no node {dcid!r}")
        self.dcid = dcid


@dataclass
class _Neighbour:
    names: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    arcs: dict[str, set[str]] = field(default_factory=dict)
    provenances: set[str] = field(default_factory=set)
    instances: SchemaInstances | None = None


class _Neighbours:
    """The view's neighbours by dcid, each merged over every answer that names it. The centre
    is never one."""

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
        provenances, which name the declarer of the arc to it; a literal and an unresolved
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
    out_read = executor.submit(reader.node_all_arc_targets_many, [dcid], deadline=deadline)
    in_read = executor.submit(
        reader.node_in_arcs,
        [dcid],
        properties=IN_ARCS,
        max_results=IN_ARC_CAP,
        deadline=deadline,
    )
    closure_read = executor.submit(
        reader.node_closure, dcid, property=_SUB_CLASS_OF, deadline=deadline
    )
    entry = resolved_entry(out_read.result().get(dcid))
    if entry is None:
        raise NodeNotHeldError(dcid)
    in_arcs = in_read.result().get(dcid, _NO_IN_ARCS)
    closure = {target.value for target in closure_read.result()}

    arcs = {
        prop: tuple(dict.fromkeys(target.value for target in entry.get(prop, ())))
        for prop in NODE_PROPERTIES
    }
    types = set(arcs[_TYPE_OF])
    neighbours = _Neighbours(dcid)
    capped: set[str] = set()
    for prop in (_TYPE_OF, *IN_ARCS):
        for target in entry.get(prop, ()):
            neighbours.add_target(target)
    capped.update(in_arcs.capped & set(_FIELDS))
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
            reader=reader,
            executor=executor,
            deadline=deadline,
            neighbours=neighbours,
            capped=capped,
        )
    elif _CLASS not in types and _PROPERTY not in types:
        _read_value(
            dcid,
            type_classes=sorted(types - {_THING})[:VALUE_TYPE_CAP],
            reader=reader,
            executor=executor,
            deadline=deadline,
            neighbours=neighbours,
            capped=capped,
        )

    provenances = {
        provenance for target in entry.get(_TYPE_OF, ()) for provenance in target.provenances
    }
    centre = schema_node(
        dcid,
        arcs,
        node_other_properties(dcid, entry, {}),
        provenances,
        instances,
        catalogue,
    )
    return SchemaNodeView(
        generation=generation,
        centre=centre,
        nodes=neighbours.nodes(catalogue),
        capped=sorted(capped),
    )


def _read_value_list(
    dcid: str,
    *,
    parents: Collection[str],
    closure: Collection[str],
    in_arcs: InArcs,
    reader: NodeViewReader,
    executor: Executor,
    deadline: float,
    neighbours: _Neighbours,
    capped: set[str],
) -> SchemaInstances:
    """Round two of value list `dcid`: its members, and the chain to `Enumeration` when that is
    not a direct parent. Returns the centre's `instances`."""
    members_read = executor.submit(
        reader.node_in_arcs,
        [dcid],
        properties=(_TYPE_OF,),
        max_results=MEMBER_CAP,
        deadline=deadline,
    )
    edges_read = (
        None
        if _ENUMERATION in parents
        else executor.submit(
            reader.node_arc_targets_many,
            sorted(set(closure) - {_ENUMERATION}),
            properties=(_SUB_CLASS_OF,),
            deadline=deadline,
        )
    )
    members = members_read.result().get(dcid, _NO_IN_ARCS)
    if edges_read is not None:
        edges = _join_chain(edges_read.result(), neighbours)
        edges[dcid] = set(parents)
        _mark_chain(edges, exclude=dcid, neighbours=neighbours)

    member_subjects = [subject for subject in members.subjects if subject.property == _TYPE_OF]
    for subject in member_subjects:
        neighbours.add_subject(subject)
    capped.update(members.capped & {_TYPE_OF})

    narrower = {
        subject.dcid
        for subject in in_arcs.subjects
        if subject.property == _SUB_CLASS_OF and _CLASS in subject.types
    }
    for ref in narrower:
        neighbours.mark_list(ref, SchemaInstances(refs=[], complete=False))
    complete = not members.capped and not narrower and _SUB_CLASS_OF not in in_arcs.capped
    return SchemaInstances(
        refs=sorted({subject.dcid for subject in member_subjects}), complete=complete
    )


def _read_value(
    dcid: str,
    *,
    type_classes: Sequence[str],
    reader: NodeViewReader,
    executor: Executor,
    deadline: float,
    neighbours: _Neighbours,
    capped: set[str],
) -> None:
    """Rounds two and three of value `dcid`: each type class's `<-rangeIncludes` subjects and
    closure, then the chain of each type class that descends from `Enumeration`, which becomes
    the value's list."""
    range_read = executor.submit(
        reader.node_in_arcs,
        type_classes,
        properties=(_RANGE_INCLUDES,),
        max_results=IN_ARC_CAP,
        deadline=deadline,
    )
    closure_reads = [
        executor.submit(reader.node_closure, type_class, property=_SUB_CLASS_OF, deadline=deadline)
        for type_class in type_classes
    ]
    lists: list[str] = []
    chain: set[str] = set()
    ranges = range_read.result()
    for type_class, closure_read in zip(type_classes, closure_reads, strict=True):
        listing = ranges.get(type_class, _NO_IN_ARCS)
        capped.update(listing.capped & {_RANGE_INCLUDES})
        for subject in listing.subjects:
            if subject.property == _RANGE_INCLUDES:
                neighbours.add_subject(subject)
                neighbours.link(subject.dcid, _RANGE_INCLUDES, type_class)
        closure = {target.value for target in closure_read.result()}
        if _ENUMERATION in closure:
            lists.append(type_class)
            chain |= {type_class, *closure} - {_ENUMERATION}
    if not lists:
        return

    edges = _join_chain(
        reader.node_arc_targets_many(sorted(chain), properties=(_SUB_CLASS_OF,), deadline=deadline),
        neighbours,
    )
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
