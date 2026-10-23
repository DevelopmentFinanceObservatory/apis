"""Reads the `->*` arcs a build's arc tables (`origin_catalogue.arc_tables`) hold.

- DFO arcs come from Spanner's `Edge` and `Node` tables through BigQuery
  `EXTERNAL_QUERY` (`read_dfo_arcs`), for each variable Spanner holds
  observations of and declares `typeOf StatisticalVariable`. A variable
  Spanner holds observations of and declares no node for, such as
  `Count_Person`, is a DFO variable whose node is base Data Commons', and
  its arcs come from the base read. An `Edge` target with no `Node` row is
  a node of base Data Commons too, and takes its name and types from our
  mixer.
- Base arcs come from our mixer's `/v2/node` `->*` read
  (`node_reads.iter_node_arcs`). A base variable the mixer answers with no
  arcs in a read naming it alone is held empty. One a read naming several
  leaves out or answers empty is read again alone, and one left out of
  that read fails the build.
- Schema arcs come from the same read, over the nodes a held node's arcs
  depend on (`schema_closure`).
- The resolver indexes decide which other nodes the tables hold, so that
  every candidate our `/v2/resolve` can answer reads from them
  (`ResolverIndex`). Our side of the resolve searches Spanner's
  `NodeEmbedding` table (`read_index_subjects`); a subject Spanner declares
  no DFO variable for and our mixer holds no node for, its `->typeOf` read
  answering no type (`held_by_mixer`), is left over from an import that
  dropped it, and the `dfo` table holds it empty, a confirmed absence. A
  subject Spanner holds `Edge` rows of reads absent only from a read naming
  it alone; one Spanner holds no `Edge` row of reads absent from a read
  naming 500, since Spanner and the batch agree. One our mixer holds is read
  as a base variable. A subject the
  published build holds absent stays absent without the read when the
  caller carries that build's absences forward (`carried_absent`). The base side
  searches the public base Data Commons embeddings index
  (`iter_base_index`); every dcid of it, with the leaves of each topic it
  names, is read as a base variable (`resolver_base_variables`). A subject
  of either index that admission refuses, a withheld variable Spanner holds
  no observations of, is left out of every table, since admission refuses
  it before its arcs are read and its arcs would otherwise reach the family
  summaries (`origin_catalogue.families`).
- Labels come from our mixer's `->name` read of every node an arc names
  (`node_reads.iter_node_labels`), which picks the label
  `GraphClient.node_label` picks.

`read_catalogue_arcs` puts the three together, taking the base arcs an
earlier build read where the caller carries them forward. Every answer
becomes rows of one `arc_tables.ArcTablesBuilder` as it lands, so the
stage holds the rows and one read's answer, and never every answer at once.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol
from urllib.parse import urlsplit

import httpx

from artifact_publication.jobs import BigQueryReader
from dc_graph import GraphReadError
from origin_catalogue.arc_tables import (
    DENOMINATOR_PROPERTY,
    SCHEMA_CLOSED_PROPERTIES,
    ArcTables,
    ArcTablesBuilder,
    SpannerNode,
    WireArcs,
    spanner_wire_arcs,
)
from origin_catalogue.node_reads import iter_node_arcs, iter_node_labels
from origin_catalogue.origins import UN_ID_PREFIX
from origin_catalogue.schema import IndexPresence
from origin_catalogue.un_mirror import UnArc

DFO_VARIABLE_TYPE: Final[str] = "StatisticalVariable"
"""The type a Spanner node carries, as a `typeOf` edge, to be a DFO variable."""

ROLE_LIST_PROPERTY: Final[str] = "observationProperties"
"""The arc naming a variable's role properties, whose ranges role admission reads."""

SUBJECT_ROLE_PROPERTY: Final[str] = "observationAbout"
"""The role admission reads for a variable that names no role property."""

_TARGET_PROPERTIES: Final[tuple[str, ...]] = ("name", "typeOf")
"""What the mixer supplies of an `Edge` target Spanner has no `Node` row for."""

_TYPE_NODES_PER_REQUEST: Final[int] = 500
"""Nodes a `->typeOf` read names, as `iter_node_labels` names them a `->name` read."""

_BASE_INDEX_DCID_COLUMN: Final[str] = "dcid"
"""The column of the base embeddings index naming a sentence's dcids, comma-separated."""


class ArcGraph(Protocol):
    """Our mixer's reads the arc tables are built from: the `/v2/node` POST
    `iter_node_arcs` and `iter_node_labels` send and the named-property read
    of the targets Spanner has no node for, which `dc_graph.GraphClient`
    provides."""

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict: ...

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]: ...


# --- DFO ---------------------------------------------------------------------


def dfo_edges_sql(connection: str) -> str:
    """Every distinct `Edge` row of a DFO variable, a subject Spanner declares
    a `DFO_VARIABLE_TYPE`, as `subject_id`, `predicate` and `object_id`, with
    its target's `Node` row as `node_id`, `value`, `name` and `types`, all
    null for a target with no `Node` row. Spanner filters `Edge` to the
    variables' rows, and BigQuery joins each to its target's `Node` row, so
    the build reads `Edge` once. Each `EXTERNAL_QUERY` text is root-partitionable:
    Spanner refuses one with `DISTINCT`, `LIMIT`, `ORDER BY` or a join."""
    declared = (
        "SELECT e.subject_id, e.predicate, e.object_id FROM Edge AS e "
        "WHERE EXISTS (SELECT 1 FROM Edge AS t WHERE t.subject_id = e.subject_id "
        f"AND t.predicate = 'typeOf' AND t.object_id = '{DFO_VARIABLE_TYPE}')"
    )
    return (
        "WITH variable_edges AS (\n"
        "  SELECT DISTINCT subject_id, predicate, object_id\n"
        f'  FROM EXTERNAL_QUERY("{connection}", "{declared}")\n'
        ")\n"
        "SELECT e.subject_id, e.predicate, e.object_id, "
        "n.subject_id AS node_id, n.value, n.name, n.types\n"
        "FROM variable_edges AS e\n"
        f'LEFT JOIN EXTERNAL_QUERY("{connection}", '
        '"SELECT subject_id, value, name, types FROM Node") AS n\n'
        "ON n.subject_id = e.object_id"
    )


@dataclass(frozen=True, slots=True)
class DfoArcs:
    """The DFO variables whose arcs `read_dfo_arcs` held, and `without_node`,
    the variables Spanner holds observations of and declares no DFO variable
    node for, whose node, and so whose arcs, are base Data Commons'.
    `base_targets` lists the targets the mixer named."""

    variables: frozenset[str]
    without_node: frozenset[str]
    base_targets: frozenset[str]


def read_dfo_arcs(
    bq: BigQueryReader,
    *,
    connection: str,
    observed: Iterable[str],
    graph: ArcGraph,
    into: ArcTablesBuilder,
) -> DfoArcs:
    """Holds in `into`'s `dfo` table the arcs of each variable in
    `observed`, the variables Spanner holds observations of, that Spanner
    declares a DFO variable, from one BigQuery read (`dfo_edges_sql`) and
    one mixer read naming the targets with no `Node` row. Each target is an entity, a
    literal or a base Data Commons node by `arc_tables.spanner_wire_target`,
    which raises `ArcTablesError` for a row that is none of them.

    BigQuery streams rows in no set order, so each property's targets are
    sorted by `object_id`, the order our mixer's `->*` read gives them.
    `materialise.roles.declared_roles` reads `observationProperties` in that
    order, so a variable's roles come in one order from the tables and from
    a live read."""
    observed_set = frozenset(observed)
    string = into.string
    # Each variable's targets by property, as shared strings, until every
    # target's `Node` row is known.
    edges: dict[str, dict[str, list[str]]] = {}
    nodes: dict[str, SpannerNode] = {}
    for row in bq.stream(dfo_edges_sql(connection)):
        subject = str(row["subject_id"])
        if subject not in observed_set:
            continue
        object_id = string(str(row["object_id"]))
        edges.setdefault(string(subject), {}).setdefault(string(str(row["predicate"])), []).append(
            object_id
        )
        if row["node_id"] is not None and object_id not in nodes:
            nodes[object_id] = SpannerNode(
                value=_text(row["value"]),
                name=_text(row["name"]),
                types=tuple(string(str(kind)) for kind in row["types"] or ()),
            )
    named = {
        object_id
        for properties in edges.values()
        for objects in properties.values()
        for object_id in objects
    }
    unrowed = sorted(named - nodes.keys())
    if unrowed:
        read = graph.node_arcs_many(unrowed, properties=_TARGET_PROPERTIES)
        for dcid in unrowed:
            arcs = read.get(dcid) or {}
            types = tuple(arcs.get("typeOf", ()))
            # A node with no types is answered by its dcid alone, as `->*` answers it.
            if types:
                nodes[dcid] = SpannerNode(value=dcid, name=_first(arcs.get("name")), types=types)
    variables = frozenset(edges)
    while edges:
        variable, properties = edges.popitem()
        into.add("dfo", variable, spanner_wire_arcs(properties, nodes))
    return DfoArcs(
        variables=variables,
        without_node=observed_set - variables,
        base_targets=frozenset(unrowed),
    )


def _text(value: object) -> str | None:
    return str(value) if value else None


def _first(values: Iterable[str] | None) -> str | None:
    return next((value for value in values or () if value), None)


# --- Schema ------------------------------------------------------------------


def _targets(arcs: WireArcs | None, properties: Iterable[str]) -> Iterable[str]:
    for property in properties:
        for target in ((arcs or {}).get(property) or {}).get("nodes") or ():
            if dcid := target.get("dcid"):
                yield dcid


def schema_closure(
    variables: Iterable[Iterable[UnArc]],
    *,
    held: Iterable[str],
    read: Callable[[Sequence[str]], Mapping[str, WireArcs]],
) -> dict[str, WireArcs]:
    """The `->*` arcs of the nodes outside `held` that the arc tables must
    hold for the variables whose rows are `variables`: each
    `DENOMINATOR_PROPERTY` target, each `ROLE_LIST_PROPERTY` target and
    `SUBJECT_ROLE_PROPERTY`, and, of every node so found, each
    `DENOMINATOR_PROPERTY` and `SCHEMA_CLOSED_PROPERTIES` target, until none
    is new. `read` answers a batch of nodes, `{}` for a node the graph
    answers with no arcs."""
    held = set(held)
    seeds = {SUBJECT_ROLE_PROPERTY}
    for rows in variables:
        seeds.update(
            arc.dcid
            for arc in rows
            if arc.dcid is not None and arc.property in (DENOMINATOR_PROPERTY, ROLE_LIST_PROPERTY)
        )
    schema: dict[str, WireArcs] = {}
    frontier = sorted(seeds - held)
    while frontier:
        found = read(frontier)
        schema.update({node: found.get(node) or {} for node in frontier})
        frontier = sorted(
            {
                dcid
                for node in frontier
                for dcid in _targets(
                    schema[node], (DENOMINATOR_PROPERTY, *SCHEMA_CLOSED_PROPERTIES)
                )
            }
            - held
            - schema.keys()
        )
    return schema


# --- Resolver indexes --------------------------------------------------------


def index_subjects_sql(connection: str) -> str:
    """Every subject of Spanner's `NodeEmbedding` table, as `subject_id`, and
    `has_edges`, whether Spanner holds an `Edge` row of it. Spanner answers
    `has_edges` by key from `Edge`, so the read exports no `Edge` row; the
    inner query is a plain projection and BigQuery aggregates, as
    `dfo_edges_sql` does."""
    inner = (
        "SELECT n.subject_id, EXISTS (SELECT 1 FROM Edge AS e WHERE e.subject_id = n.subject_id) "
        "AS has_edges FROM NodeEmbedding AS n"
    )
    return f'SELECT DISTINCT subject_id, has_edges\nFROM EXTERNAL_QUERY("{connection}", "{inner}")'


@dataclass(frozen=True, slots=True)
class IndexSubjects:
    """The subjects of our resolver index (`read_index_subjects`), and
    `unedged`, those Spanner holds no `Edge` row of, which an
    import that dropped them left behind (on dev 45,876 of 69,847)."""

    subjects: frozenset[str]
    unedged: frozenset[str]


def read_index_subjects(bq: BigQueryReader, *, connection: str) -> IndexSubjects:
    """Every node our side of `/v2/resolve` can answer, the subjects of
    Spanner's `NodeEmbedding` table, which our mixer searches for
    `resolver: indicator`, and those of them Spanner holds no `Edge` row of."""
    subjects: set[str] = set()
    unedged: set[str] = set()
    for row in bq.stream(index_subjects_sql(connection)):
        subject = str(row["subject_id"])
        subjects.add(subject)
        if not row["has_edges"]:
            unedged.add(subject)
    return IndexSubjects(subjects=frozenset(subjects), unedged=frozenset(unedged))


def iter_base_index(uri: str) -> Iterator[str]:
    """Each dcid the base Data Commons embeddings index at `uri` names, an
    `embeddings.csv` holding a row per sentence whose `dcid` column lists the
    sentence's dcids, comma-separated. Base Data Commons answers the base side
    of `/v2/resolve` from it. `uri` is `https://` or `file://`; the file is
    read a line at a time, so the read holds one row. Raises `GraphReadError`
    when the HTTP read fails."""
    if urlsplit(uri).scheme == "file":
        with Path(urlsplit(uri).path).open(newline="") as lines:
            yield from _index_dcids(lines)
        return
    try:
        with httpx.stream("GET", uri, timeout=120.0, follow_redirects=True) as response:
            response.raise_for_status()
            yield from _index_dcids(response.iter_lines())
    except httpx.HTTPError as exc:
        raise GraphReadError(f"base resolver index read {uri!r} failed: {exc}") from exc


def _index_dcids(lines: Iterable[str]) -> Iterator[str]:
    rows = csv.reader(lines)
    header = next(rows, None)
    if header is None or _BASE_INDEX_DCID_COLUMN not in header:
        raise GraphReadError(f"the base resolver index has no {_BASE_INDEX_DCID_COLUMN!r} column")
    column = header.index(_BASE_INDEX_DCID_COLUMN)
    for row in rows:
        for dcid in row[column].split(","):
            if dcid := dcid.strip():
                yield dcid


def resolver_base_variables(
    index: Iterable[str],
    *,
    topics: Mapping[str, Sequence[str]],
) -> frozenset[str]:
    """The base nodes whose arcs the tables hold so that every base candidate
    the base resolve can answer reads from them: each dcid of `index`, and in
    place of a topic of `topics` its every leaf, the leaves search's topic
    expansion reads. A UN dcid is left out, since the UN tables answer it;
    `read_catalogue_arcs` leaves out the ones admission refuses."""
    found: set[str] = set()
    for dcid in index:
        found.update(topics.get(dcid, (dcid,)))
    return frozenset(dcid for dcid in found if not dcid.startswith(UN_ID_PREFIX))


@dataclass(frozen=True, slots=True)
class ResolverIndex:
    """What `/v2/resolve` can answer beyond the variables the build lists:
    `ours`, every subject our mixer's index holds (`read_index_subjects`),
    `unedged`, those of `ours` Spanner holds no `Edge` row of, and `base`,
    the base nodes `resolver_base_variables` draws from the base index."""

    ours: frozenset[str]
    base: frozenset[str]
    unedged: frozenset[str] = frozenset()


EMPTY_RESOLVER_INDEX: Final = ResolverIndex(ours=frozenset(), base=frozenset())
"""The resolver index of a build that holds no node beyond the variables it lists."""


def held_by_mixer(
    graph: ArcGraph,
    nodes: Sequence[str],
    *,
    reread_empty: bool = True,
    deadline: float | None = None,
) -> frozenset[str]:
    """The nodes of `nodes` our mixer holds, those its `->typeOf` read answers
    with a type, since every node has one. Reads `_TYPE_NODES_PER_REQUEST`
    nodes a request by `iter_node_arcs`' rules: a batch can leave out or empty
    a node depending on which others share it, so a node a read naming
    several leaves out is read again alone, as is one it answers with no type
    unless `reread_empty` is false. Raises `GraphReadError` when a read fails
    or one naming a node alone leaves it out."""
    return frozenset(
        node
        for node, arcs in iter_node_arcs(
            graph,
            nodes,
            deadline=deadline,
            arc="->typeOf",
            per_request=_TYPE_NODES_PER_REQUEST,
            reread_empty=reread_empty,
        )
        if arcs.get("typeOf", {}).get("nodes")
    )


# --- Tables ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CatalogueArcs:
    """A build's arc tables, the DFO variables they hold (`dfo`), how many
    base variables the build read from the mixer and how many it took from
    an earlier build, how many arc targets it read labels for, and what the
    resolver indexes added: `index_absent`, the subjects of our index held
    empty, `index_held_by_mixer`, those read as base nodes, `index_base`,
    the base nodes drawn from the base index that the build does not list
    as variables, and `index_presence`, how many subjects of our index the
    build took as absent from an earlier build and how many it read
    `->typeOf` for."""

    tables: ArcTables
    dfo: DfoArcs
    base_read: int
    base_carried: int
    labels_read: int
    index_absent: int
    index_held_by_mixer: int
    index_base: int
    index_presence: IndexPresence


def read_catalogue_arcs(
    bq: BigQueryReader,
    *,
    connection: str,
    observed: Iterable[str],
    base_variables: Iterable[str],
    resolver_index: ResolverIndex,
    carried_base: Mapping[str, tuple[UnArc, ...]],
    carried_absent: frozenset[str] = frozenset(),
    withheld: frozenset[str] = frozenset(),
    graph: ArcGraph,
    deadline: float | None = None,
) -> CatalogueArcs:
    """The arc tables of the variables of `observed` (`read_dfo_arcs`), each
    one Spanner declares no node for in the `base` table, every variable of
    `base_variables` and every base node
    of `resolver_index` they do not hold, and the schema nodes their arcs
    depend on (`schema_closure`), with the label of every node their arcs
    name (`iter_node_labels`). A subject of our index that neither list
    holds is read as a base node when our mixer holds it (`held_by_mixer`)
    and held empty in the `dfo` table when it does not, and one of
    `carried_absent`, which an earlier build held empty, is held empty
    without the read. A subject of either index that admission refuses, one
    of `withheld` outside `observed`, is in no table. A base node
    `carried_base` holds keeps the rows an earlier build read, and the mixer
    is read for the rest. Labels are read afresh each build. `deadline` is
    an absolute `time.monotonic()` value bounding every mixer read. Raises
    `GraphReadError` when a read fails and `ArcTablesError` for tables
    breaking `ArcTables`' invariants."""
    builder = ArcTablesBuilder()
    dfo = read_dfo_arcs(bq, connection=connection, observed=observed, graph=graph, into=builder)
    # A DFO variable with no Spanner node reads as the base node it is.
    base_variables = set(base_variables) | dfo.without_node
    # Admission refuses a withheld variable DFO holds no observations of.
    refused = withheld - dfo.variables - dfo.without_node
    unlisted = sorted(resolver_index.ours - refused - dfo.variables - base_variables)
    unknown = [subject for subject in unlisted if subject not in carried_absent]
    # Spanner holding no `Edge` row of a subject agrees with a batch answering it
    # untyped, so that answer stands without a read naming it alone.
    mixer_held = held_by_mixer(
        graph,
        [subject for subject in unknown if subject not in resolver_index.unedged],
        deadline=deadline,
    ) | held_by_mixer(
        graph,
        [subject for subject in unknown if subject in resolver_index.unedged],
        reread_empty=False,
        deadline=deadline,
    )
    absent = [subject for subject in unlisted if subject not in mixer_held]
    for subject in absent:
        builder.add("dfo", subject, None)
    index_base = (
        ((resolver_index.base - refused) | mixer_held)
        - base_variables
        - dfo.variables
        - set(absent)
    )
    listed = sorted((base_variables | index_base) - dfo.variables)
    carried = [variable for variable in listed if variable in carried_base]
    for variable in carried:
        builder.add_rows("base", variable, carried_base[variable])
    fresh = [variable for variable in listed if variable not in carried_base]
    for variable, arcs in iter_node_arcs(graph, fresh, deadline=deadline):
        builder.add("base", variable, arcs)
    # Schema nodes are read afresh each build, a carried variable's from its rows.
    schema = schema_closure(
        [*builder.table("dfo").values(), *builder.table("base").values()],
        held=[*dfo.variables, *absent, *listed],
        read=lambda nodes: dict(iter_node_arcs(graph, nodes, deadline=deadline)),
    )
    for node, arcs in schema.items():
        builder.add("schema", node, arcs)
    targets = builder.targets()
    labels = dict(iter_node_labels(graph, targets, deadline=deadline))
    return CatalogueArcs(
        tables=builder.build(labels=labels),
        dfo=dfo,
        base_read=len(fresh),
        base_carried=len(carried),
        labels_read=len(targets),
        index_absent=len(absent),
        index_held_by_mixer=len(mixer_held),
        index_base=len(index_base),
        index_presence=IndexPresence(carried=len(unlisted) - len(unknown), read=len(unknown)),
    )


__all__ = [
    "DFO_VARIABLE_TYPE",
    "EMPTY_RESOLVER_INDEX",
    "ROLE_LIST_PROPERTY",
    "SUBJECT_ROLE_PROPERTY",
    "ArcGraph",
    "CatalogueArcs",
    "DfoArcs",
    "IndexSubjects",
    "ResolverIndex",
    "dfo_edges_sql",
    "held_by_mixer",
    "index_subjects_sql",
    "iter_base_index",
    "read_catalogue_arcs",
    "read_dfo_arcs",
    "read_index_subjects",
    "resolver_base_variables",
    "schema_closure",
]
