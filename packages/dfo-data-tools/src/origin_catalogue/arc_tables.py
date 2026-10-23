"""The `->*` arcs of DFO variables, base variables and the schema nodes they name.

A build holds three tables of arcs, each mapping a node to its arcs as shared
`UnArc` rows, the shape the UN tables use:

- `dfo`: every DFO variable the build lists in `dfo_variables` whose node
  is DFO's, and, held empty, every other subject of our resolver index the
  graph has no node for;
- `base`: every base variable it lists in `base_variables`, every DFO
  variable whose node is base Data Commons', such as `Count_Person`, and
  every base node a resolver index names that admission can admit;
- `schema`: the nodes outside both lists that a held node's arcs depend on:
  a `measurementDenominator` StatVar neither list holds, each role property
  with its `rangeIncludes`, and each type they range over with its
  `subClassOf` chain.

A held node with no arcs is held and empty: the graph has no node for it, a
confirmed absence. A node none of the tables holds is out of their scope and
reads from the graph.

Each node an arc names also has a label, the answer `GraphClient.node_label`
gives for it, read apart from its arcs. An arc carries the name the graph's
`->*` read gives the node, which is its first name, or its dcid when it has
none, and `node_label` gives its last name, or none. Search reads both.

`ArcTablesBuilder` builds the tables a node at a time from `->*` answers in
the shape `dc_graph.GraphClient.node_arcs_batch` returns. `spanner_wire_target` turns the
target of a Spanner `Edge` row of DFO's graph into that shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from origin_catalogue.un_mirror import UnArc

type WireArcs = Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]
"""One node's `->*` arcs as `node_arcs_batch` returns them: each property's
`{"nodes": [...]}`, each target carrying `dcid` or `value`, and `name`,
`types` and `provenanceId` where the graph has them."""

DENOMINATOR_PROPERTY: Final[str] = "measurementDenominator"
"""A property whose every node target, on any held node, the tables hold,
so a denominator's arcs read locally."""

SCHEMA_CLOSED_PROPERTIES: Final[tuple[str, ...]] = ("rangeIncludes", "subClassOf")
"""Properties whose every node target, on a schema node, the tables hold, so
a role property's range and a type's superclasses read locally. Variables
are not held to them: 470 WHO base variables carry `rangeIncludes Number`."""


class ArcTablesError(ValueError):
    """Arc tables or the rows they are built from break an invariant.
    `subject` is the node at fault."""

    def __init__(self, message: str, *, subject: str) -> None:
        super().__init__(message)
        self.subject = subject


@dataclass(frozen=True, slots=True)
class ArcTables:
    """One build's arc tables: `dfo`, `base` and `schema` each map a node to
    its arcs, ordered by property, one target per `dcid` or `value` within a
    property.

    `labels` maps each node some arc names to its label, `None` for a node
    with no name. `targets` maps each such node to one arc row naming it,
    which carries the node's name and types as the graph gave them on that
    arc: a row with a name where one exists, else the first. It is derived
    on construction and counted in the tables' resident size.

    Construction refuses a node held by two tables, a `DENOMINATOR_PROPERTY`
    target no table holds, a `SCHEMA_CLOSED_PROPERTIES` target of a schema
    node no table holds, and `labels` that leave out a node an arc names or
    label one no arc names.
    """

    dfo: Mapping[str, tuple[UnArc, ...]]
    base: Mapping[str, tuple[UnArc, ...]]
    schema: Mapping[str, tuple[UnArc, ...]]
    labels: Mapping[str, str | None]
    targets: Mapping[str, UnArc] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        for first, second in (
            (self.dfo, self.base),
            (self.dfo, self.schema),
            (self.base, self.schema),
        ):
            if both := sorted(first.keys() & second.keys()):
                raise ArcTablesError(
                    f"{len(both)} node(s) are held by two arc tables, e.g. {both[:3]!r}",
                    subject=both[0],
                )
        variable_closed = (DENOMINATOR_PROPERTY,)
        schema_closed = (DENOMINATOR_PROPERTY, *SCHEMA_CLOSED_PROPERTIES)
        targets: dict[str, UnArc] = {}
        for table, closed in (
            (self.dfo, variable_closed),
            (self.base, variable_closed),
            (self.schema, schema_closed),
        ):
            # Nodes share arc rows, so each row is checked once per table.
            seen: set[int] = set()
            for node, arcs in table.items():
                for arc in arcs:
                    if id(arc) in seen:
                        continue
                    seen.add(id(arc))
                    if arc.dcid is not None:
                        known = targets.get(arc.dcid)
                        if known is None or (known.name is None and arc.name is not None):
                            targets[arc.dcid] = arc
                    if (
                        arc.property in closed
                        and arc.dcid is not None
                        and self.arcs(arc.dcid) is None
                    ):
                        raise ArcTablesError(
                            f"{node!r} has {arc.property} {arc.dcid!r}, which no arc table holds",
                            subject=node,
                        )
        if self.labels.keys() != targets.keys():
            unlabelled = sorted(targets.keys() - self.labels.keys())
            extra = sorted(self.labels.keys() - targets.keys())
            raise ArcTablesError(
                f"{len(unlabelled)} arc target(s) have no label, e.g. {unlabelled[:3]!r}, and "
                f"{len(extra)} label(s) name no arc target, e.g. {extra[:3]!r}",
                subject=(unlabelled or extra)[0],
            )
        object.__setattr__(self, "targets", targets)

    def arcs(self, dcid: str) -> tuple[UnArc, ...] | None:
        """`dcid`'s arcs, empty for a node held and empty, `None` for one no
        table holds."""
        for table in (self.dfo, self.base, self.schema):
            arcs = table.get(dcid)
            if arcs is not None:
                return arcs
        return None


EMPTY_ARC_TABLES: Final = ArcTables(dfo={}, base={}, schema={}, labels={})
"""The arc tables of a build that holds none."""


type ArcTableName = Literal["dfo", "base", "schema"]
"""One of the three tables `ArcTables` holds."""


class ArcTablesBuilder:
    """Builds `ArcTables` a node at a time.

    Each node's `->*` answer becomes rows as it is added, with one string
    per distinct value and one `UnArc` per distinct row across the tables,
    so a build holds the rows of the nodes added so far and drops each
    answer once it is added. `build` orders each table by node, so the
    tables are the same whatever order the nodes came in.
    """

    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._arcs: dict[UnArc, UnArc] = {}
        self._tables: dict[ArcTableName, dict[str, tuple[UnArc, ...]]] = {
            "dfo": {},
            "base": {},
            "schema": {},
        }

    def string(self, value: str) -> str:
        """`value`, shared with every equal string the tables hold."""
        return self._strings.setdefault(value, value)

    def add(self, table: ArcTableName, node: str, arcs: WireArcs | None) -> None:
        """Holds `node` in `table` with `arcs`, rows ordered by property. A
        target another provenance repeats is kept once, by its `dcid` or
        `value`, as `dc_graph` keeps it. `None` or no arcs holds the node
        empty, as `node_arcs_batch` answers a node the graph does not have."""
        arcs = arcs or {}
        rows: dict[UnArc, None] = {}
        for property in sorted(arcs):
            seen: set[str | None] = set()
            for target in arcs[property].get("nodes") or ():
                key = target.get("dcid") or target.get("value")
                if key not in seen:
                    seen.add(key)
                    rows[self._arc(property, target)] = None
        self._tables[table][self.string(node)] = tuple(rows)

    def add_rows(self, table: ArcTableName, node: str, rows: Iterable[UnArc]) -> None:
        """Holds `node` in `table` with `rows`, rows an earlier build holds,
        taken as they are and shared with equal rows."""
        self._tables[table][self.string(node)] = tuple(
            self._arcs.setdefault(arc, arc) for arc in rows
        )

    def table(self, table: ArcTableName) -> Mapping[str, tuple[UnArc, ...]]:
        """The nodes `table` holds so far, with their rows."""
        return self._tables[table]

    def targets(self) -> list[str]:
        """Every node an arc held so far names, sorted."""
        return sorted(
            {
                arc.dcid
                for table in self._tables.values()
                for rows in table.values()
                for arc in rows
                if arc.dcid is not None
            }
        )

    def build(self, *, labels: Mapping[str, str | None]) -> ArcTables:
        """The tables, with `labels`, each node `targets` lists mapped to its
        `node_label` answer. Raises `ArcTablesError` for tables breaking
        `ArcTables`' invariants."""

        def ordered(name: ArcTableName) -> dict[str, tuple[UnArc, ...]]:
            table = self._tables[name]
            return {node: table[node] for node in sorted(table)}

        return ArcTables(
            dfo=ordered("dfo"),
            base=ordered("base"),
            schema=ordered("schema"),
            labels={
                self.string(node): None if label is None else self.string(label)
                for node, label in sorted(labels.items())
            },
        )

    def _arc(self, property: str, target: Mapping[str, Any]) -> UnArc:
        dcid, value, name = target.get("dcid"), target.get("value"), target.get("name")
        arc = UnArc(
            property=self.string(property),
            dcid=self.string(dcid) if dcid else None,
            value=self.string(value) if value else None,
            name=self.string(name) if name else None,
            types=tuple(self.string(kind) for kind in target.get("types") or ()),
        )
        return self._arcs.setdefault(arc, arc)


# --- Spanner -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpannerNode:
    """One Spanner `Node` row: `value`, `name` and `types`."""

    value: str | None
    name: str | None
    types: tuple[str, ...]


def spanner_wire_target(object_id: str, node: SpannerNode | None) -> dict[str, Any]:
    """The `->*` target an `Edge` row's `object_id` names, as the graph
    answers it.

    An entity's `Node` row has types and `value` equal to its id, and the
    target carries its `dcid`, `name` and `types`. A literal's row, with an
    id of the form `<prefix>:<hash>`, has no types and holds the literal in
    `value`, and the target carries the `value`. A target with no `Node` row
    is a node of base Data Commons and carries its `dcid` alone. Raises
    `ArcTablesError` for a row that is neither."""
    if node is None:
        return {"dcid": object_id}
    if node.types and node.value == object_id:
        target: dict[str, Any] = {"dcid": object_id, "types": list(node.types)}
        if node.name:
            target["name"] = node.name
        return target
    if not node.types and node.value is not None and node.value != object_id:
        return {"value": node.value}
    raise ArcTablesError(
        f"Spanner node {object_id!r} is neither an entity (types, value equal to its id) nor a "
        f"literal (no types, another value): types {node.types!r}",
        subject=object_id,
    )


def spanner_wire_arcs(
    edges: Mapping[str, Iterable[str]], nodes: Mapping[str, SpannerNode]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """A DFO variable's `->*` answer from its `Edge` rows, `edges` mapping each predicate to
    its `object_id`s, and the `Node` rows of their targets, each target as
    `spanner_wire_target` gives it. Each property's targets are sorted by `object_id`, the
    order our mixer's `->*` read gives them, so the answer is the same whatever order the rows
    came in."""
    return {
        property: {
            "nodes": [
                spanner_wire_target(object_id, nodes.get(object_id))
                for object_id in sorted(objects)
            ]
        }
        for property, objects in edges.items()
    }


__all__ = [
    "DENOMINATOR_PROPERTY",
    "EMPTY_ARC_TABLES",
    "SCHEMA_CLOSED_PROPERTIES",
    "ArcTableName",
    "ArcTables",
    "ArcTablesBuilder",
    "ArcTablesError",
    "SpannerNode",
    "WireArcs",
    "spanner_wire_arcs",
    "spanner_wire_target",
]
