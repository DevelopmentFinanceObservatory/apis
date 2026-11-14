"""Build the deployment's schema document and the terms list from live graph reads.

`build_schema` reads `Class <-typeOf` and `Property <-typeOf` once each, and every subject is a
term. The document's seed is every term DFO declares plus every node a StatVar row of the code
index names. `node_all_arc_targets_many` reads every outgoing property of each node with the
provenances that declare each value, and the closure reads the nodes not yet read that DFO's
terms name in `subClassOf`, `subPropertyOf`, `domainIncludes` or `rangeIncludes`, the `typeOf`
targets of every node read that is neither a Class nor a Property, and the `subClassOf` and
`subPropertyOf` parents, so the class of every value is present and every ancestor is too. One
`node_arc_subjects` listing per DFO-declared Class reads its members, and the members of each
that is an enumeration join the document with the same closure over their types and ancestors.

The reads run as a pipeline, at most `BUILD_CONCURRENCY` at once: the index-named nodes are read
while the term listings run, each node read of at most 12 dcids is expanded the moment it lands,
every DFO Class is listed as soon as the terms are known, and an enumeration's members are read
once its `subClassOf` chain reaches `Enumeration`. The nodes read follow from the graph alone,
so the order the answers arrive in changes nothing in the document. Each node read probes every
empty answer: it reads each empty dcid again alone, and a dcid empty after that is absent from
the document. The node reads are uncached, since no other
caller reads every property of these nodes. A read that fails ends the build with its error. One
log line per build gives each phase's read count and when its first read started and last read
ended: the term listings, the seed, each closure level, the member listings and the members.

Each node's values and provenances come from that one read of the node. The ten properties
`SchemaNode` names are its named members, and every other property the read answers is in
`other_properties`, each target a node reference or a literal, less any literal past
`OTHER_LITERAL_MAX_CHARS` and the values past the first `OTHER_VALUES_MAX` of a property, which
one warning per build counts. Its provenances are
those its `typeOf` targets carry, and `declared_by` follows from them by the origin each
belongs to: the origin catalogue's provenance table names it where it lists the provenance, and
the id's prefix decides otherwise, `dc/base/` for base Data Commons, `undata/` for the UN
origin and any other for DFO. The table lists the data provenances the catalogue admits and none
of the ontology's, such as `provenance/DFOOntology`, so those take the prefix rule. A node
declared by any DFO provenance is `"dfo"`, one whose provenances are all base is `"base"`, and
any other is `"un_system"`. A node with no `typeOf` arc, or whose `typeOf` targets carry no
provenance, is served with none and `declared_by` null. A term's provenances come from its listing entries by the same rule. One warning per
build counts the terms served with none and the nodes with a `typeOf` target that carries none,
and names the first ten of each. A build whose document holds no node DFO declares refuses with
`SchemaNoDfoDeclarationError`.

A DFO-declared enumeration's members are its `<-typeOf` listing, `complete` when the
listing read every subject. A base enumeration lists the members StatVar rows use and is
always `complete: false`.

Every node, member and list of values is sorted, so two builds from the same graph serialise
to the same bytes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass, field
from functools import partial
from typing import Final, Literal

from code_index import CodeIndex
from dc_graph import ArcSubject, ArcSubjectPage, ArcTarget, GraphReadError, display_name
from dfo_platform.executor import ContextCarryingExecutor
from graph_api.dataset_usage import SHAPE_FIELDS
from graph_api.reader import GraphNodeReader
from graph_api.schema_models import (
    DeclaredBy,
    SchemaDocument,
    SchemaEnumeration,
    SchemaLiteral,
    SchemaNode,
    SchemaNodeReference,
    SchemaTerm,
    SchemaTerms,
)
from origin_catalogue import Catalogue, OriginId
from origin_catalogue.origins import UN_ID_PREFIX

logger = logging.getLogger(__name__)

TERM_LISTING_CAP: Final[int] = 20_000
"""The most subjects one `<-typeOf` listing of `Class` or `Property` reads. A listing that
reaches it raises `SchemaListingTruncatedError`."""

ENUMERATION_MEMBER_CAP: Final[int] = 5_000
"""The most members read for one DFO-declared enumeration. An enumeration with more is
served with the members read and `complete: false`."""

OTHER_LITERAL_MAX_CHARS: Final[int] = 2_000
"""The longest literal `other_properties` holds. A longer one, such as a geometry, is left out."""

OTHER_VALUES_MAX: Final[int] = 100
"""The most values `other_properties` holds for one property: the first of them sorted."""

BUILD_CONCURRENCY: Final[int] = 24
"""The most reads one build runs at once. Each node read is one request, or one at a time when
the mixer answers a continuation, and each listing is one, so the build has at most this many
requests in flight beside the confirmation probes, which the client's `->*` pool bounds."""

_NODE_READ_CHUNK: Final[int] = 12
"""The most dcids one node read names: `dc_graph.GraphClient`'s default `->*` chunk, so each
read is one request. A chunk of 12 schema nodes carries 65 to 124 arc values, well inside the
mixer's page of about 500."""

_TERM_LISTINGS: Final[str] = "term listings"
_SEED: Final[str] = "seed"
_MEMBER_LISTINGS: Final[str] = "member listings"
_MEMBERS: Final[str] = "members"

_LOGGED_DCIDS: Final[int] = 10
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
type _OtherProperties = dict[str, list[SchemaNodeReference | SchemaLiteral]]
type _TermKind = Literal["Class", "Property"]


class SchemaListingTruncatedError(GraphReadError):
    """A `Class` or `Property` listing reached `TERM_LISTING_CAP`, so the terms are incomplete."""


class SchemaNoDfoDeclarationError(GraphReadError):
    """No node of the built document is declared by DFO, so the graph named no DFO provenance
    for any of them and the check would judge no range or domain."""

    def __init__(self, node_count: int) -> None:
        super().__init__(
            f"none of the {node_count} nodes of the schema document is declared by DFO"
        )
        self.node_count = node_count


@dataclass(frozen=True)
class Schema:
    """The schema document and the terms list from one build."""

    document: SchemaDocument
    terms: SchemaTerms


@dataclass
class _ListedTerm:
    names: set[str] = field(default_factory=set)
    kinds: set[_TermKind] = field(default_factory=set)
    provenances: set[str] = field(default_factory=set)


def provenance_origin(provenance: str, catalogue: Catalogue) -> OriginId:
    """The origin `provenance` belongs to: the one `catalogue`'s provenance table names, else
    `"base"` for an id under `dc/base/`, `"un_system"` for one under `undata/` and `"dfo"` for
    any other."""
    if (row := catalogue.provenance(provenance)) is not None:
        return row.data_commons.id
    if provenance.startswith(_BASE_PREFIX):
        return "base"
    if provenance.startswith(UN_ID_PREFIX):
        return "un_system"
    return "dfo"


def declared_by(provenances: Collection[str], *, catalogue: Catalogue) -> DeclaredBy:
    """`"dfo"` when any declaring provenance is DFO's, `"base"` when every one is base Data
    Commons', `"un_system"` otherwise, each by `provenance_origin`.

    Raises:
        ValueError: `provenances` is empty, which names no origin.
    """
    if not provenances:
        raise ValueError("a term with no declaring provenance has no origin")
    origins = {provenance_origin(provenance, catalogue) for provenance in provenances}
    if "dfo" in origins:
        return "dfo"
    if origins == {"base"}:
        return "base"
    return "un_system"


def build_schema(
    *,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    catalogue: Catalogue,
    generation: str | None,
    deadline: float,
) -> Schema:
    """Build the schema document and the terms list.

    Args:
        reader: the graph reads.
        code_index: the bound code index, whose StatVar rows name nodes the document holds.
        catalogue: the origin catalogue whose provenance table names the origin of the
            provenances it lists.
        generation: the graph generation token both bodies carry, `None` where it is unknown.
        deadline: the absolute monotonic deadline passed to every read.

    Raises:
        SchemaListingTruncatedError: a `Class` or `Property` listing reached its cap.
        SchemaNoDfoDeclarationError: no node of the document is declared by DFO.
        GraphReadError: a read failed.
    """
    started = time.monotonic()
    index_named = _index_named(code_index)
    with ContextCarryingExecutor(
        max_workers=BUILD_CONCURRENCY, thread_name_prefix="schema-build"
    ) as executor:
        reads = _Reads(reader, executor=executor, catalogue=catalogue, deadline=deadline)
        reads.run(index_named)
    read_at = time.monotonic()
    listed, arcs = reads.listed, reads.arcs

    members: dict[str, SchemaEnumeration] = {}
    for enumeration in sorted(reads.dfo_terms):
        if (
            enumeration in arcs
            and _CLASS in listed[enumeration].kinds
            and _reaches_enumeration(enumeration, arcs)
        ):
            subjects, complete = reads.member_pages[enumeration]
            members[enumeration] = SchemaEnumeration(
                members=sorted({subject.dcid for subject in subjects}), complete=complete
            )

    index_named_by_type: dict[str, list[str]] = defaultdict(list)
    for dcid in index_named:
        for type_ in arcs.get(dcid, {}).get(_TYPE_OF, ()):
            index_named_by_type[type_].append(dcid)
    enumerations = (
        dcid
        for dcid, term in listed.items()
        if dcid in arcs and _CLASS in term.kinds and _reaches_enumeration(dcid, arcs)
    )
    for enumeration in enumerations:
        if enumeration not in members:
            members[enumeration] = SchemaEnumeration(
                members=sorted(index_named_by_type.get(enumeration, ())), complete=False
            )

    unsourced_terms = sorted(dcid for dcid, term in listed.items() if not term.provenances)
    if unsourced_terms or reads.unsourced:
        logger.warning(
            "schema build read %d terms with no listed provenance, served with declared_by null, "
            "among them %s, and %d nodes with a typeOf target that carries no provenance, among "
            "them %s",
            len(unsourced_terms),
            unsourced_terms[:_LOGGED_DCIDS],
            len(reads.unsourced),
            sorted(reads.unsourced)[:_LOGGED_DCIDS],
        )
    if reads.left_out:
        logger.warning(
            "schema build left %d values out of other_properties, literals past %d characters "
            "or values past %d a property, among them on %s",
            sum(reads.left_out.values()),
            OTHER_LITERAL_MAX_CHARS,
            OTHER_VALUES_MAX,
            sorted(reads.left_out)[:_LOGGED_DCIDS],
        )
    nodes = [
        _node(
            dcid,
            arcs[dcid],
            reads.others[dcid],
            reads.provenances[dcid],
            members.get(dcid),
            catalogue,
        )
        for dcid in sorted(arcs)
    ]
    if not any(node.declared_by == "dfo" for node in nodes):
        raise SchemaNoDfoDeclarationError(len(nodes))
    terms = [
        SchemaTerm(
            ref=dcid,
            label=display_name(term.names, dcid),
            types=sorted(term.kinds),
            provenances=sorted(term.provenances),
            declared_by=(
                declared_by(term.provenances, catalogue=catalogue) if term.provenances else None
            ),
        )
        for dcid, term in sorted(listed.items())
    ]
    schema = Schema(
        document=SchemaDocument(generation=generation, nodes=nodes),
        terms=SchemaTerms(generation=generation, terms=terms),
    )
    reads.log_timings(started=started, read_at=read_at, ended=time.monotonic())
    return schema


def _index_named(code_index: CodeIndex) -> set[str]:
    """Every node a StatVar row names in its five fields, constraints or observation roles."""
    named: set[str] = set()
    for row in (*code_index.statvars, *code_index.listed_statvars):
        named.update(value for name in SHAPE_FIELDS if (value := getattr(row, name)))
        named.update(row.constraint_properties.keys())
        named.update(row.constraint_properties.values())
        named.update(row.observation_properties)
    return named


type _NodeEntries = Mapping[str, Mapping[str, Sequence[ArcTarget]] | None]
type _Landing = Callable[[], None]


@dataclass(slots=True)
class _PhaseTiming:
    reads: int = 0
    first_start: float = float("inf")
    last_end: float = 0.0
    longest: float = 0.0

    def add(self, started: float, ended: float) -> None:
        self.reads += 1
        self.first_start = min(self.first_start, started)
        self.last_end = max(self.last_end, ended)
        self.longest = max(self.longest, ended - started)


class _Reads:
    """The build's graph reads, each started as soon as what it depends on has landed, at most
    `BUILD_CONCURRENCY` at once on `executor`.

    The `Class` and `Property` listings and the reads of the index-named nodes start together.
    Once both listings land, the DFO terms are read and every DFO Class's members are listed.
    Each node read is expanded as it lands: its `subClassOf` and `subPropertyOf` parents, the
    `typeOf` targets of a node that is no listed term, and the four schema arcs of a DFO term
    are read next, unless already read or under way. A node read that lands before the listings
    is expanded when they land. A DFO Class's members are read, and expanded the same way, once
    its listing has landed and its `subClassOf` chain over the nodes read reaches `Enumeration`.
    Each node is read once, in reads of at most `_NODE_READ_CHUNK` dcids, and the reads end when
    none is left to start. The successor rule depends only on a node's own read and the
    listings, so the nodes read are the same whatever order the answers arrive in. The first
    read that raises cancels the reads not yet started and its error propagates.
    """

    def __init__(
        self,
        reader: GraphNodeReader,
        *,
        executor: Executor,
        catalogue: Catalogue,
        deadline: float,
    ) -> None:
        self._reader = reader
        self._executor = executor
        self._catalogue = catalogue
        self._deadline = deadline
        self._term_pages: dict[str, ArcSubjectPage] = {}
        self._listed: dict[str, _ListedTerm] | None = None
        self.dfo_terms: set[str] = set()
        self.arcs: dict[str, _Arcs] = {}
        self.others: dict[str, _OtherProperties] = {}
        # The values left out of `others`, by node and property.
        self.left_out: dict[tuple[str, str], int] = {}
        # Each node's provenances, read with its arcs.
        self.provenances: dict[str, frozenset[str]] = {}
        self.absent: set[str] = set()
        self.unsourced: set[str] = set()
        # Each DFO Class's `<-typeOf` subjects, and whether the listing read every one.
        self.member_pages: dict[str, tuple[tuple[ArcSubject, ...], bool]] = {}
        self._members_read: set[str] = set()
        self._submitted: set[str] = set()
        self._unexpanded: list[tuple[set[str], str]] = []
        # Each read under way, whose answer is the landing the build runs on its own thread.
        self._pending: set[Future[_Landing]] = set()
        self._phases: dict[str, _PhaseTiming] = {}
        self._phases_lock = threading.Lock()

    @property
    def listed(self) -> dict[str, _ListedTerm]:
        """Every listed term, once both listings have landed."""
        if self._listed is None:
            raise RuntimeError("the term listings have not landed")
        return self._listed

    def run(self, index_named: Collection[str]) -> None:
        """Start the listings and the index-named reads, and land every read until none is left.

        Raises:
            SchemaListingTruncatedError: a `Class` or `Property` listing reached its cap.
            GraphReadError: a read failed.
        """
        for kind in (_CLASS, _PROPERTY):
            self._list(kind, cap=TERM_LISTING_CAP, phase=_TERM_LISTINGS)
        self._read_nodes(index_named, phase=_SEED)
        try:
            while self._pending:
                done, self._pending = wait(self._pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()()
                if self._listed is not None:
                    self._read_enumeration_members()
        except BaseException:
            for future in self._pending:
                future.cancel()
            raise

    def log_timings(self, *, started: float, read_at: float, ended: float) -> None:
        """Log how long each phase's reads took, by when its first read started and its last
        ended, in milliseconds from the start of the build."""

        def ms(at: float) -> int:
            return round((at - started) * 1_000)

        phases = sorted(self._phases.items(), key=lambda item: item[1].first_start)
        logger.info(
            "schema build took %d ms, reads ended at %d ms: %s; assembly %d ms",
            ms(ended),
            ms(read_at),
            "; ".join(
                f"{name} {timing.reads} reads {ms(timing.first_start)}-{ms(timing.last_end)} ms "
                f"longest {round(timing.longest * 1_000)} ms"
                for name, timing in phases
            ),
            round((ended - read_at) * 1_000),
        )

    def _list(self, dcid: str, *, cap: int, phase: str) -> None:
        def read() -> _Landing:
            started = time.monotonic()
            page = self._reader.node_arc_subjects(
                dcid, property=_TYPE_OF, max_results=cap, deadline=self._deadline
            )
            self._record(phase, started)
            return partial(self._land_listing, page, phase=phase, dcid=dcid)

        self._pending.add(self._executor.submit(read))

    def _read_nodes(self, dcids: Iterable[str], *, phase: str) -> None:
        fresh = sorted(set(dcids) - self._submitted)
        self._submitted.update(fresh)
        for start in range(0, len(fresh), _NODE_READ_CHUNK):
            chunk = tuple(fresh[start : start + _NODE_READ_CHUNK])

            def read(chunk: tuple[str, ...] = chunk) -> _Landing:
                started = time.monotonic()
                entries = self._reader.node_all_arc_targets_many(chunk, deadline=self._deadline)
                self._record(phase, started)
                return partial(self._land_nodes, entries, dcids=chunk, phase=phase)

            self._pending.add(self._executor.submit(read))

    def _record(self, phase: str, started: float) -> None:
        with self._phases_lock:
            self._phases.setdefault(phase, _PhaseTiming()).add(started, time.monotonic())

    def _land_listing(self, page: ArcSubjectPage, *, phase: str, dcid: str) -> None:
        if phase == _MEMBER_LISTINGS:
            self.member_pages[dcid] = (page.subjects, not page.truncated)
            return
        if page.truncated:
            raise SchemaListingTruncatedError(
                f"{dcid} <-typeOf lists more than {TERM_LISTING_CAP} subjects"
            )
        self._term_pages[dcid] = page
        if len(self._term_pages) == 2:
            self._land_terms()

    def _land_terms(self) -> None:
        listed: dict[str, _ListedTerm] = {}
        for kind in (_CLASS, _PROPERTY):
            for subject in self._term_pages[kind].subjects:
                term = listed.setdefault(subject.dcid, _ListedTerm())
                term.names.update(subject.names)
                term.kinds.add(kind)
                term.provenances.update(subject.provenances)
        self._listed = listed
        self.dfo_terms = {
            dcid
            for dcid, term in listed.items()
            if term.provenances
            and declared_by(term.provenances, catalogue=self._catalogue) == "dfo"
        }
        self._read_nodes(self.dfo_terms, phase=_SEED)
        for dcid in sorted(self.dfo_terms):
            if _CLASS in listed[dcid].kinds:
                self._list(dcid, cap=ENUMERATION_MEMBER_CAP, phase=_MEMBER_LISTINGS)
        for held, phase in self._unexpanded:
            self._expand(held, phase=phase)
        self._unexpanded.clear()

    def _land_nodes(self, entries: _NodeEntries, *, dcids: tuple[str, ...], phase: str) -> None:
        """Keep each node's arcs, other properties and provenances, or mark it absent: the read
        confirms every empty answer alone, so an answer of none is the graph's answer for the
        dcid read alone. A node's provenances are those its `typeOf` targets carry, and a node
        with a target that carries none joins `unsourced`."""
        held: set[str] = set()
        for dcid in dcids:
            entry = entries.get(dcid)
            if entry is None:
                self.absent.add(dcid)
                continue
            types = entry.get(_TYPE_OF, ())
            self.arcs[dcid] = {
                prop: tuple(target.value for target in entry.get(prop, ()))
                for prop in _NODE_PROPERTIES
            }
            self.others[dcid] = _other_properties(dcid, entry, self.left_out)
            if any(not target.provenances for target in types):
                self.unsourced.add(dcid)
            self.provenances[dcid] = frozenset(
                provenance for target in types for provenance in target.provenances
            )
            held.add(dcid)
        if self._listed is None:
            self._unexpanded.append((held, phase))
        else:
            self._expand(held, phase=phase)

    def _expand(self, held: set[str], *, phase: str) -> None:
        successors = _next_level(self.arcs, held, self.listed) | _named(
            self.arcs, held & self.dfo_terms, _SCHEMA_ARCS
        )
        self._read_nodes(successors, phase=_next_phase(phase))

    def _read_enumeration_members(self) -> None:
        """Read the members of each listed DFO Class whose chain now reaches `Enumeration`."""
        for enumeration, (subjects, _) in sorted(self.member_pages.items()):
            if enumeration in self._members_read or enumeration not in self.arcs:
                continue
            if _reaches_enumeration(enumeration, self.arcs):
                self._members_read.add(enumeration)
                self._read_nodes((subject.dcid for subject in subjects), phase=_MEMBERS)


def _next_phase(phase: str) -> str:
    """The phase a read's successors are timed under: the seed's are level 2, a level's the
    next level's, and the members' the members'."""
    if phase == _MEMBERS:
        return _MEMBERS
    if phase == _SEED:
        return "level 2"
    return f"level {int(phase.removeprefix('level ')) + 1}"


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


def _other_properties(
    dcid: str, entry: Mapping[str, Sequence[ArcTarget]], left_out: dict[tuple[str, str], int]
) -> _OtherProperties:
    """Every property of `entry` outside `_NODE_PROPERTIES`, each target a node reference or a
    literal, properties and targets sorted. A literal past `OTHER_LITERAL_MAX_CHARS` and the
    values past the first `OTHER_VALUES_MAX` are left out and counted in `left_out` under
    `(dcid, property)`, and a property left with no value is left out."""
    others: _OtherProperties = {}
    for prop, targets in sorted(entry.items()):
        if prop in _NODE_PROPERTIES:
            continue
        kept = sorted(
            (
                target
                for target in targets
                if not target.literal or len(target.value) <= OTHER_LITERAL_MAX_CHARS
            ),
            key=lambda target: (target.value, target.literal),
        )
        if dropped := len(targets) - min(len(kept), OTHER_VALUES_MAX):
            left_out[(dcid, prop)] = dropped
        if kept:
            others[prop] = [
                SchemaLiteral(value=target.value)
                if target.literal
                else SchemaNodeReference(ref=target.value)
                for target in kept[:OTHER_VALUES_MAX]
            ]
    return others


def _node(
    dcid: str,
    arcs: _Arcs,
    other_properties: _OtherProperties,
    provenances: Collection[str],
    enumeration: SchemaEnumeration | None,
    catalogue: Catalogue,
) -> SchemaNode:
    return SchemaNode(
        ref=dcid,
        label=display_name(arcs["name"], dcid),
        description=min(arcs["description"], default=None),
        types=sorted(arcs[_TYPE_OF]),
        alternate_name=sorted(arcs["alternateName"]),
        sub_class_of=sorted(arcs["subClassOf"]),
        sub_property_of=sorted(arcs["subPropertyOf"]),
        specialization_of=sorted(arcs["specializationOf"]),
        domain_includes=sorted(arcs["domainIncludes"]),
        range_includes=sorted(arcs["rangeIncludes"]),
        classification_code=sorted(arcs["classificationCode"]),
        other_properties=other_properties,
        provenances=sorted(provenances),
        declared_by=declared_by(provenances, catalogue=catalogue) if provenances else None,
        enumeration=enumeration,
    )
