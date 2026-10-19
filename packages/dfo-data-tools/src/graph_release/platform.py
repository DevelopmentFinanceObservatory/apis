"""`RawInputs` to the five platform parts, and their staging as part files.

The platform parts copy Spanner, less what a live mixer read cannot see, and carry no DFO rule.
`build_platform_parts` is pure: it applies stage one of `dc_graph.visible_arcs`, the edge
closure and the authored rows, and refuses a snapshot the release format cannot hold.
`stage_platform_parts` writes the parts through `graph_release.format.write_part`.

The edge closure has one rule. An edge whose subject is a live variable is kept. A second-hop
edge, one whose subject is not a live variable, is kept only when its object has a Spanner
`Node` row or a kept live variable's edge names that object. Every kept edge's subject and
object must then have a `Node` row, from Spanner or `Authored`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.compute as pc

from dc_graph.visible_arcs import mixer_visible_mask
from graph_release.authored import AUTHORED_PROVENANCE, Authored
from graph_release.format import MASK_PERIODS, PLATFORM, StagedPart, conform_part, write_part
from graph_release.reads import RawInputs

_EXAMPLES: Final[int] = 5


class PlatformRefusal(Exception):
    """A snapshot the builder must not release. `code` names the fault with the gate's code for
    it."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class PlatformParts:
    """One table per platform part, each as `graph_release.format.conform_part` gives it: the
    part's schema, sorted dictionaries, rows in key order."""

    series: pa.Table
    facets: pa.Table
    edges: pa.Table
    nodes: pa.Table
    imports: pa.Table

    def tables(self) -> Mapping[str, pa.Table]:
        return {
            "series": self.series,
            "facets": self.facets,
            "edges": self.edges,
            "nodes": self.nodes,
            "imports": self.imports,
        }


def build_platform_parts(raw: RawInputs, *, authored: Authored) -> PlatformParts:
    """The platform parts of `raw`, with `authored`'s rows added to `nodes` and `edges`.

    Raises `PlatformRefusal` with code `unsupported_observation_period` for a facet whose period
    the mask does not count, `inconsistent_facet` for a facet whose series carry different facet
    JSON, `unknown_facet` for a series whose facet was not read, `authored_mismatch` for an
    authored node Spanner now holds, and `dangling_edge` for a kept edge whose subject or object
    has no `Node` row.
    """
    facets = _facets(raw.facets)
    series = _series(raw.series_summary)
    unknown = pc.unique(_dictionary(series["facet_id"]).dictionary_decode())
    unknown = unknown.filter(pc.invert(pc.is_in(unknown, value_set=facets["facet_id"])))
    if len(unknown):
        raise PlatformRefusal(
            "unknown_facet", f"series name facets the facet read lacks: {_examples(unknown)}"
        )

    spanner_ids = raw.nodes_raw["subject_id"]
    authored_nodes = authored.nodes_table()
    restated = authored_nodes["subject_id"].filter(
        pc.is_in(authored_nodes["subject_id"], value_set=spanner_ids)
    )
    if len(restated):
        raise PlatformRefusal(
            "authored_mismatch", f"Spanner now holds authored nodes: {_examples(restated)}"
        )

    edges = pa.concat_tables([_closure(raw), authored.edges_table()])
    known = pa.chunked_array([*spanner_ids.chunks, *authored_nodes["subject_id"].chunks])
    ends = pc.unique(pa.chunked_array([*edges["subject_id"].chunks, *edges["object_id"].chunks]))
    dangling = ends.filter(pc.invert(pc.is_in(ends, value_set=known)))
    if len(dangling):
        raise PlatformRefusal(
            "dangling_edge", f"edges name nodes with no Node row: {_examples(dangling)}"
        )

    return PlatformParts(
        series=conform_part(series, name="series"),
        facets=conform_part(facets, name="facets"),
        edges=conform_part(edges, name="edges"),
        nodes=conform_part(_nodes(raw, edges, authored_nodes), name="nodes"),
        imports=conform_part(_imports(raw.imports), name="imports"),
    )


def stage_platform_parts(
    parts: PlatformParts, *, staging_dir: Path, execution_id: str
) -> list[StagedPart]:
    """Writes every part to `staging_dir`: `series` one file set per facet provenance, `edges`
    one per edge provenance."""

    def write(table: pa.Table, name: str, source: str, provenance: str | None = None) -> None:
        staged.extend(
            write_part(
                table,
                staging_dir=staging_dir,
                name=name,
                layer=PLATFORM,
                source=source,
                produced_by_execution=execution_id,
                provenance=provenance,
            )
        )

    staged: list[StagedPart] = []
    write(parts.facets, "facets", "spanner.timeseries.facet")

    facet_ids = _dictionary(parts.series["facet_id"])
    facet_provenance = pc.take(
        _dictionary(parts.facets["provenance"]).dictionary_decode(),
        pc.index_in(facet_ids.dictionary, value_set=parts.facets["facet_id"]),
    )
    for provenance, mask in _groups(facet_ids.indices, facet_provenance):
        write(parts.series.filter(mask), "series", "spanner.observation.aggregate", provenance)

    edge_provenance = _dictionary(parts.edges["provenance"])
    for provenance, mask in _groups(edge_provenance.indices, edge_provenance.dictionary):
        source = "graph_release.authored" if provenance == AUTHORED_PROVENANCE else "spanner.edge"
        write(parts.edges.filter(mask), "edges", source, provenance)

    write(parts.nodes, "nodes", "spanner.node")
    write(parts.imports, "imports", "spanner.importstatus")
    return staged


def _facets(raw: pa.Table) -> pa.Table:
    columns: dict[str, list[object]] = {
        name: []
        for name in (
            "facet_id",
            "provenance",
            "import_name",
            "measurement_method",
            "observation_period",
            "unit",
            "scaling_factor",
            "is_dc_aggregate",
        )
    }
    inconsistent, unsupported = [], []
    for facet_id, low, high in zip(
        cast("list[str]", raw["facet_id"].to_pylist()),
        cast("list[str]", raw["facet_min"].to_pylist()),
        cast("list[str]", raw["facet_max"].to_pylist()),
        strict=True,
    ):
        if low != high:
            inconsistent.append(facet_id)
            continue
        facet = json.loads(low)
        if facet.get("observationPeriod") not in MASK_PERIODS:
            unsupported.append(f"{facet_id} ({facet.get('observationPeriod')})")
        scaling = facet.get("scalingFactor")
        columns["facet_id"].append(facet_id)
        columns["provenance"].append(facet.get("provenance"))
        columns["import_name"].append(facet.get("importName"))
        columns["measurement_method"].append(facet.get("measurementMethod"))
        columns["observation_period"].append(facet.get("observationPeriod"))
        columns["unit"].append(facet.get("unit"))
        columns["scaling_factor"].append(None if scaling is None else str(scaling))
        columns["is_dc_aggregate"].append(bool(facet.get("isDcAggregate", False)))
    if inconsistent:
        raise PlatformRefusal(
            "inconsistent_facet",
            f"series of one facet carry different facet JSON: {_examples(inconsistent)}",
        )
    if unsupported:
        raise PlatformRefusal(
            "unsupported_observation_period",
            f"the mask counts {list(MASK_PERIODS)}; facets {_examples(unsupported)}",
        )
    return pa.table(
        {
            name: pa.array(values, pa.bool_() if name == "is_dc_aggregate" else pa.string())
            for name, values in columns.items()
        }
    )


def _series(raw: pa.Table) -> pa.Table:
    """The summary in the `series` columns: dates parsed, the count unsigned, `entity3` left out
    when no series has one."""
    columns = {
        name: raw[name]
        for name in ("variable_measured", "entity1", "extra_entities_id", "facet_id", "entity2")
    }
    if raw["entity3"].null_count < raw.num_rows:
        columns["entity3"] = raw["entity3"]
    columns["observation_count"] = raw["observation_count"].cast(pa.uint32())
    columns["earliest_date"] = _dates(raw["earliest_date"])
    columns["latest_date"] = _dates(raw["latest_date"])
    for name in raw.column_names:
        if name.startswith("period_mask_"):
            columns[name] = raw[name]
    return pa.table(columns)


def _dates(column: pa.ChunkedArray) -> pa.ChunkedArray:
    """`Observation.date` strings as dates: `YYYY` is 1 January, `YYYY-MM` the first of the
    month, `YYYY-MM-DD` itself. Raises `ValueError` for any other form."""
    parsed = pc.coalesce(
        *(
            pc.strptime(column, format=form, unit="s", error_is_null=True)
            for form in ("%Y", "%Y-%m", "%Y-%m-%d")
        )
    )
    unparsed = column.filter(pc.and_(pc.is_null(parsed), pc.is_valid(column)))
    if len(unparsed):
        raise ValueError(f"observation dates of no known form: {_examples(unparsed)}")
    return cast("pa.ChunkedArray", parsed.cast(pa.date32()))


def _closure(raw: RawInputs) -> pa.Table:
    """The Spanner edges the parts keep: stage one of the visibility rule, then the closure."""
    edges = raw.edges_raw
    edges = edges.filter(mixer_visible_mask(edges["predicate"].combine_chunks()))
    objects = edges["object_id"].combine_chunks()
    live = raw.live_variables["variable_measured"].combine_chunks()
    first_hop = pc.is_in(edges["subject_id"].combine_chunks(), value_set=live)
    named = pc.unique(objects.filter(first_hop))
    nodes = raw.nodes_raw["subject_id"].combine_chunks()
    second_hop = pc.is_in(objects, value_set=pa.concat_arrays([nodes, named]))
    return edges.filter(cast("pa.BooleanArray", pc.or_(first_hop, second_hop)))


def _nodes(raw: RawInputs, edges: pa.Table, authored_nodes: pa.Table) -> pa.Table:
    """Spanner's rows for every subject and object of `edges` and every live variable, then the
    authored rows."""
    referenced = pa.chunked_array(
        [
            *edges["subject_id"].chunks,
            *edges["object_id"].chunks,
            *raw.live_variables["variable_measured"].chunks,
        ]
    )
    spanner = raw.nodes_raw.filter(pc.is_in(raw.nodes_raw["subject_id"], value_set=referenced))
    spanner = spanner.select(authored_nodes.column_names[:-1]).append_column(
        "authored", pa.array([False] * spanner.num_rows, pa.bool_())
    )
    return pa.concat_tables([spanner, authored_nodes])


def _imports(raw: pa.Table) -> pa.Table:
    return pa.table(
        {
            "import_name": raw["import_name"],
            "provenance": pa.nulls(raw.num_rows, pa.string()),
            "version": raw["version"],
            "state": raw["state"],
            "data_import_timestamp": raw["data_import_timestamp"],
        }
    )


def _dictionary(column: pa.ChunkedArray) -> pa.DictionaryArray:
    return cast("pa.DictionaryArray", column.combine_chunks())


def _groups(indices: pa.Array, labels: pa.Array) -> list[tuple[str, pa.BooleanArray]]:
    """For each distinct label, sorted, the mask of rows whose `indices` point at it."""
    groups = []
    for label in sorted(set(cast("list[str]", labels.to_pylist()))):
        hits = pc.equal(labels, pa.scalar(label, pa.string()))
        groups.append((label, pc.take(hits, indices)))
    return groups


def _examples(values: pa.Array | pa.ChunkedArray | list[str]) -> str:
    items = sorted(cast("list[str]", values if isinstance(values, list) else values.to_pylist()))
    more = f" and {len(items) - _EXAMPLES} more" if len(items) > _EXAMPLES else ""
    return f"{items[:_EXAMPLES]}{more}"


__all__ = [
    "PlatformParts",
    "PlatformRefusal",
    "build_platform_parts",
    "stage_platform_parts",
]
