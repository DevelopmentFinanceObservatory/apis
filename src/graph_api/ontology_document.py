"""Build the instance's ontology document and the terms list from live graph reads.

`build_ontology` reads `Class <-typeOf` and `Property <-typeOf` once each; every subject is a
term. The document's seed is every term the instance declares plus every node a StatVar row
of the code index names. One `node_arcs_many` reads the seed's arcs, and each further level
reads the nodes not yet read that the instance's terms name in `subClassOf`,
`subPropertyOf`, `domainIncludes` or `rangeIncludes`, the `typeOf` targets of every node
read that is neither a Class nor a Property, and the `subClassOf` and `subPropertyOf`
parents, so the class of every value is present and every ancestor is too. One
`node_arc_subjects_many` lists the members of every instance-declared enumeration, and the
members not yet read join the document with the same closure over their types and ancestors.
One `node_arc_targets_many` reads the declaring provenances of every node outside the
listings, and each node it answers with none is read again alone. The number of reads grows
with the depth of the class and property hierarchy, never with the number of nodes.

Every node and term carries at least one declaring provenance. A listing subject with none,
or a node that two reads leave with none, refuses the build with
`OntologyProvenanceUnreadError`. A value `node_arcs_many` marks `UNRESOLVED_NODE` is dropped
where the arcs are read, so no member, description or followed dcid carries it.

Every node, member and list of values is sorted, so two builds from the same graph serialise
to the same bytes.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from code_index import CodeIndex
from dc_graph import UNRESOLVED_NODE, GraphReadError, display_name
from graph_api.ontology_models import (
    DeclaredBy,
    OntologyDocument,
    OntologyEnumeration,
    OntologyNode,
    OntologyTerm,
    OntologyTerms,
)
from graph_api.reader import GraphNodeReader

TERM_LISTING_CAP: Final[int] = 20_000
"""The most subjects one `<-typeOf` listing of `Class` or `Property` reads. A listing that
reaches it raises `OntologyListingTruncatedError`."""

ENUMERATION_MEMBER_CAP: Final[int] = 5_000
"""The most members read for one instance-declared enumeration. An enumeration with more is
served with the members read and `truncated: true`."""

_BASE_PREFIX: Final[str] = "dc/base/"
_CLASS: Final = "Class"
_PROPERTY: Final = "Property"
_ENUMERATION: Final[str] = "Enumeration"

_TYPE_OF: Final[str] = "typeOf"
_ANCESTOR_ARCS: Final[tuple[str, ...]] = ("subClassOf", "subPropertyOf")
_SCHEMA_ARCS: Final[tuple[str, ...]] = (*_ANCESTOR_ARCS, "domainIncludes", "rangeIncludes")
_NODE_PROPERTIES: Final[tuple[str, ...]] = (
    _TYPE_OF,
    "name",
    "description",
    "alternateName",
    "subClassOf",
    "subPropertyOf",
    "specializationOf",
    "domainIncludes",
    "rangeIncludes",
    "classificationCode",
)

type _Arcs = dict[str, tuple[str, ...]]
type _TermKind = Literal["Class", "Property"]


class OntologyListingTruncatedError(GraphReadError):
    """A `Class` or `Property` listing reached `TERM_LISTING_CAP`, so the terms are incomplete."""


class OntologyProvenanceUnreadError(GraphReadError):
    """A term or node came back with no declaring provenance, so it cannot be labelled base or
    instance."""


@dataclass(frozen=True)
class Ontology:
    """The ontology document and the terms list from one build."""

    document: OntologyDocument
    terms: OntologyTerms


@dataclass
class _ListedTerm:
    names: set[str] = field(default_factory=set)
    kinds: set[_TermKind] = field(default_factory=set)
    provenances: set[str] = field(default_factory=set)


def declared_by(provenances: Collection[str]) -> DeclaredBy:
    """`"base"` when every declaring provenance starts `dc/base/`, `"instance"` otherwise.

    Raises:
        ValueError: `provenances` is empty, which labels nothing.
    """
    if not provenances:
        raise ValueError("a term with no declaring provenance is neither base nor instance")
    if all(provenance.startswith(_BASE_PREFIX) for provenance in provenances):
        return "base"
    return "instance"


def build_ontology(
    *,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    generation: str | None,
    deadline: float,
) -> Ontology:
    """Build the ontology document and the terms list.

    Args:
        reader: the graph reads.
        code_index: the bound code index, whose StatVar rows name nodes the document holds.
        generation: the graph generation token both bodies carry, `None` where it is unknown.
        deadline: the absolute monotonic deadline passed to every read.

    Raises:
        OntologyListingTruncatedError: a `Class` or `Property` listing reached its cap.
        OntologyProvenanceUnreadError: a term or node has no declaring provenance.
        GraphReadError: a read failed.
    """
    listed = _read_terms(reader, deadline=deadline)
    instance_terms = {
        dcid for dcid, term in listed.items() if declared_by(term.provenances) == "instance"
    }
    index_named = _index_named(code_index)

    arcs: dict[str, _Arcs] = {}
    absent: set[str] = set()
    _read_nodes(reader, instance_terms | index_named, arcs=arcs, absent=absent, deadline=deadline)
    _close(
        reader,
        _named(arcs, instance_terms, _SCHEMA_ARCS) | _next_level(arcs, arcs.keys(), listed),
        arcs=arcs,
        absent=absent,
        listed=listed,
        deadline=deadline,
    )

    instance_enumerations = sorted(
        dcid
        for dcid in instance_terms
        if dcid in arcs and _CLASS in listed[dcid].kinds and _reaches_enumeration(dcid, arcs)
    )
    members: dict[str, OntologyEnumeration] = {}
    member_provenances: dict[str, set[str]] = {}
    if instance_enumerations:
        listings = reader.node_arc_subjects_many(
            instance_enumerations,
            property=_TYPE_OF,
            max_results=ENUMERATION_MEMBER_CAP,
            deadline=deadline,
        )
        for enumeration in instance_enumerations:
            subjects, coverage = listings[enumeration]
            for subject in subjects:
                member_provenances.setdefault(subject.dcid, set()).update(subject.provenances)
            members[enumeration] = OntologyEnumeration(
                members=sorted({subject.dcid for subject in subjects}),
                truncated=coverage != "complete",
            )
    if unread_members := member_provenances.keys() - arcs.keys() - absent:
        read = _read_nodes(reader, unread_members, arcs=arcs, absent=absent, deadline=deadline)
        _close(
            reader,
            _next_level(arcs, read, listed),
            arcs=arcs,
            absent=absent,
            listed=listed,
            deadline=deadline,
        )

    enumerations = (
        dcid
        for dcid, term in listed.items()
        if dcid in arcs and _CLASS in term.kinds and _reaches_enumeration(dcid, arcs)
    )
    for enumeration in enumerations:
        if enumeration not in members:
            members[enumeration] = OntologyEnumeration(
                members=sorted(
                    dcid
                    for dcid in index_named
                    if dcid in arcs and enumeration in arcs[dcid][_TYPE_OF]
                ),
                truncated=False,
            )

    provenances = {dcid: term.provenances for dcid, term in listed.items()} | {
        dcid: found for dcid, found in member_provenances.items() if found
    }
    if unsourced := sorted(arcs.keys() - provenances.keys()):
        provenances |= _read_provenances(reader, unsourced, deadline=deadline)

    nodes = [_node(dcid, arcs[dcid], provenances[dcid], members.get(dcid)) for dcid in sorted(arcs)]
    terms = [
        OntologyTerm(
            dcid=dcid,
            name=display_name(term.names, dcid),
            types=sorted(term.kinds),
            provenances=sorted(term.provenances),
            declared_by=declared_by(term.provenances),
        )
        for dcid, term in sorted(listed.items())
    ]
    return Ontology(
        document=OntologyDocument(generation=generation, nodes=nodes),
        terms=OntologyTerms(generation=generation, terms=terms),
    )


def _read_terms(reader: GraphNodeReader, *, deadline: float) -> dict[str, _ListedTerm]:
    """Every subject of `Class <-typeOf` and `Property <-typeOf`, merged by dcid."""
    listed: dict[str, _ListedTerm] = {}
    kinds: tuple[_TermKind, ...] = (_CLASS, _PROPERTY)
    for kind in kinds:
        page = reader.node_arc_subjects(
            kind, property=_TYPE_OF, max_results=TERM_LISTING_CAP, deadline=deadline
        )
        if page.truncated:
            raise OntologyListingTruncatedError(
                f"{kind} <-typeOf lists more than {TERM_LISTING_CAP} subjects"
            )
        for subject in page.subjects:
            term = listed.setdefault(subject.dcid, _ListedTerm())
            term.names.update(subject.names)
            term.kinds.add(kind)
            term.provenances.update(subject.provenances)
    if unsourced := sorted(dcid for dcid, term in listed.items() if not term.provenances):
        raise OntologyProvenanceUnreadError(
            f"{len(unsourced)} listed terms carry no declaring provenance, first {unsourced[0]!r}"
        )
    return listed


def _index_named(code_index: CodeIndex) -> set[str]:
    """Every node a StatVar row names in its five fields, constraints or observation roles."""
    named: set[str] = set()
    for row in (*code_index.statvars, *code_index.listed_statvars):
        named.update(
            value
            for value in (
                row.population_type,
                row.measured_property,
                row.stat_type,
                row.measurement_qualifier,
                row.measurement_denominator,
            )
            if value
        )
        named.update(row.constraint_properties.keys())
        named.update(row.constraint_properties.values())
        named.update(row.observation_properties)
    return named


def _read_nodes(
    reader: GraphNodeReader,
    dcids: Collection[str],
    *,
    arcs: dict[str, _Arcs],
    absent: set[str],
    deadline: float,
) -> set[str]:
    """Read `dcids`' node properties in one batch into `arcs`, or `absent` where the graph
    holds no node, dropping every `UNRESOLVED_NODE` value. Returns the dcids held."""
    read = reader.node_arcs_many(sorted(dcids), properties=_NODE_PROPERTIES, deadline=deadline)
    held: set[str] = set()
    for dcid in dcids:
        entry = read.get(dcid)
        if entry is None:
            absent.add(dcid)
            continue
        arcs[dcid] = {
            prop: tuple(value for value in entry.get(prop, ()) if value != UNRESOLVED_NODE)
            for prop in _NODE_PROPERTIES
        }
        held.add(dcid)
    return held


def _close(
    reader: GraphNodeReader,
    frontier: set[str],
    *,
    arcs: dict[str, _Arcs],
    absent: set[str],
    listed: Collection[str],
    deadline: float,
) -> None:
    """Read `frontier` and each `_next_level` after it into `arcs` until no node is left
    unread."""
    while frontier := frontier - arcs.keys() - absent:
        read = _read_nodes(reader, frontier, arcs=arcs, absent=absent, deadline=deadline)
        frontier = _next_level(arcs, read, listed)


def _read_provenances(
    reader: GraphNodeReader, dcids: Sequence[str], *, deadline: float
) -> dict[str, set[str]]:
    """The provenances declaring each of `dcids`' `typeOf` arcs. One batch reads them all and
    each dcid it answers with none is read alone once more, since a batch can empty an entry
    depending on which other dcids share it.

    Raises:
        OntologyProvenanceUnreadError: a dcid still has no declaring provenance.
    """
    found = _typeof_provenances(reader, dcids, deadline=deadline)
    for dcid in [dcid for dcid in dcids if not found[dcid]]:
        found |= _typeof_provenances(reader, [dcid], deadline=deadline)
        if not found[dcid]:
            raise OntologyProvenanceUnreadError(
                f"no declaring provenance read for {dcid!r} after a second read alone"
            )
    return found


def _typeof_provenances(
    reader: GraphNodeReader, dcids: Sequence[str], *, deadline: float
) -> dict[str, set[str]]:
    targets = reader.node_arc_targets_many(dcids, properties=(_TYPE_OF,), deadline=deadline)
    return {
        dcid: {
            provenance
            for target in (targets.get(dcid) or {}).get(_TYPE_OF, ())
            for provenance in target.provenances
        }
        for dcid in dcids
    }


def _named(arcs: Mapping[str, _Arcs], dcids: Iterable[str], properties: Iterable[str]) -> set[str]:
    """The values `dcids` hold in `properties`, over the dcids read."""
    properties = tuple(properties)
    return {
        value for dcid in dcids if dcid in arcs for prop in properties for value in arcs[dcid][prop]
    }


def _next_level(
    arcs: Mapping[str, _Arcs], read: Collection[str], listed: Collection[str]
) -> set[str]:
    """The nodes the closure reads after `read`: their `subClassOf` and `subPropertyOf`
    parents, and the `typeOf` targets of those that are neither a Class nor a Property."""
    values = [dcid for dcid in read if dcid not in listed]
    return _named(arcs, read, _ANCESTOR_ARCS) | _named(arcs, values, (_TYPE_OF,))


def _reaches_enumeration(dcid: str, arcs: Mapping[str, _Arcs]) -> bool:
    """Whether `dcid`'s `subClassOf` chain, over the nodes read, reaches `Enumeration`."""
    seen: set[str] = set()
    stack = [dcid]
    while stack:
        current = stack.pop()
        if current in seen or current not in arcs:
            continue
        seen.add(current)
        parents = arcs[current]["subClassOf"]
        if _ENUMERATION in parents:
            return True
        stack.extend(parents)
    return False


def _node(
    dcid: str,
    arcs: _Arcs,
    provenances: Collection[str],
    enumeration: OntologyEnumeration | None,
) -> OntologyNode:
    return OntologyNode(
        dcid=dcid,
        name=display_name(arcs["name"], dcid),
        description=min(arcs["description"], default=None),
        types=sorted(arcs[_TYPE_OF]),
        alternate_name=sorted(arcs["alternateName"]),
        sub_class_of=sorted(arcs["subClassOf"]),
        sub_property_of=sorted(arcs["subPropertyOf"]),
        specialization_of=sorted(arcs["specializationOf"]),
        domain_includes=sorted(arcs["domainIncludes"]),
        range_includes=sorted(arcs["rangeIncludes"]),
        classification_code=sorted(arcs["classificationCode"]),
        provenances=sorted(provenances),
        declared_by=declared_by(provenances),
        enumeration=enumeration,
    )
