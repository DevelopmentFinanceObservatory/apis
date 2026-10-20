"""The origin catalogue's DFO half, read from a release: each DFO variable's arcs and the
family summaries of the DFO variables.

The catalogue job reads both from Spanner (`origin_catalogue.arc_reads.read_dfo_arcs`): a DFO
variable is one Spanner holds observations of and declares `typeOf StatisticalVariable`, and
its arcs are its `Edge` rows with each target as its `Node` row gives it
(`origin_catalogue.arc_tables.spanner_wire_arcs`). A release copies those rows. `series` holds
the variables with observations, `edges` every edge of each (less the predicates the mixer
never returns, which no served read shows), and `nodes` the row of every target. A target
DFO's Spanner holds no row for is a base Data Commons node, which the job names from our mixer
and the release carries as an `authored` row recorded from the same read
(`graph_release.authored`). So the DFO table and summaries read here are the job's for the
same import.

`ReleaseService` builds the view as a release opens when `GRAPH_RELEASE_ORIGIN` is on
(`graph_release.context.origin_release_enabled`), and it lives on the `Release`.
`origin_catalogue.catalogue.CatalogueSnapshot` reads it through `served_origin` for the DFO
variables' arcs and joins its summaries to the catalogue's base ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from graph_release.format import LoadedRelease
from graph_release.views import ViewRefused
from origin_catalogue.arc_reads import DFO_VARIABLE_TYPE
from origin_catalogue.arc_tables import ArcTablesBuilder, SpannerNode, spanner_wire_arcs
from origin_catalogue.families import FamilyKey, FamilySummary, summarise_table
from origin_catalogue.un_mirror import UnArc

_TYPE_OF: Final[str] = "typeOf"


@dataclass(frozen=True, eq=False)
class ReleaseOrigin:
    """A release's DFO variables, each with its arc rows in dcid order (`arcs`), and the
    summary of each family of them (`families`), as the catalogue job's `dfo` arc table and
    `origin_catalogue.families.summarise_table` over it give them."""

    arcs: Mapping[str, tuple[UnArc, ...]]
    families: Mapping[FamilyKey, FamilySummary]


def origin_view(*, series: pa.Table, edges: pa.Table, nodes: pa.Table) -> ReleaseOrigin:
    """The `ReleaseOrigin` of the `series`, `edges` and `nodes` parts. Raises `ViewRefused`
    for a target whose `Node` row is neither an entity nor a literal
    (`origin_catalogue.arc_tables.spanner_wire_target`)."""
    # Unique before decoding: the series column has a row per series, millions of them.
    observed = set(cast("list[str]", series["variable_measured"].unique().to_pylist()))
    subjects, predicates, objects = (
        _Column.of(edges[name]) for name in ("subject_id", "predicate", "object_id")
    )
    typed = (predicates.codes == predicates.code(_TYPE_OF)) & (
        objects.codes == objects.code(DFO_VARIABLE_TYPE)
    )
    variables = [
        code for code in np.unique(subjects.codes[typed]) if subjects.values[code] in observed
    ]
    held = np.isin(subjects.codes, variables)

    by_variable: dict[str, dict[str, list[str]]] = {}
    for subject, predicate, target in zip(
        subjects.codes[held].tolist(),
        predicates.codes[held].tolist(),
        objects.codes[held].tolist(),
        strict=True,
    ):
        by_variable.setdefault(subjects.values[subject], {}).setdefault(
            predicates.values[predicate], []
        ).append(objects.values[target])
    targets = _target_nodes(
        nodes,
        pa.array([objects.values[code] for code in np.unique(objects.codes[held])], pa.string()),
    )

    builder = ArcTablesBuilder()
    try:
        for variable in sorted(by_variable):
            builder.add("dfo", variable, spanner_wire_arcs(by_variable.pop(variable), targets))
    except ValueError as exc:
        raise ViewRefused(f"the origin view does not build: {exc}") from exc
    table = builder.table("dfo")
    return ReleaseOrigin(arcs=table, families=summarise_table(table))


@dataclass(frozen=True)
class _Column:
    """An `edges` column as each row's code into `values`, its distinct values, so a value
    is decoded once however many rows hold it."""

    codes: np.ndarray
    values: list[str]

    @staticmethod
    def of(column: pa.ChunkedArray) -> _Column:
        array = column.combine_chunks()
        encoded = cast(
            "pa.DictionaryArray",
            array if pa.types.is_dictionary(array.type) else array.dictionary_encode(),
        )
        return _Column(
            codes=encoded.indices.to_numpy(zero_copy_only=False),
            values=cast("list[str]", encoded.dictionary.to_pylist()),
        )

    def code(self, value: str) -> int:
        """`value`'s code, or -1, which no row holds, for a value no row holds."""
        try:
            return self.values.index(value)
        except ValueError:
            return -1


def _target_nodes(nodes: pa.Table, wanted: pa.Array) -> dict[str, SpannerNode]:
    """The `SpannerNode` of each `wanted` id, as the job reads it. A Spanner row is taken as
    it is. An `authored` row stands for the mixer's answer the job reads for a target with no
    Spanner row: an entity, its dcid as its value, when it has types, else a literal when it
    has a value. One with neither is left out, so it answers by its dcid alone, as `->*`
    answers an untyped node."""
    rows = nodes.filter(pc.is_in(nodes["subject_id"], value_set=wanted))
    found: dict[str, SpannerNode] = {}
    for subject, name, types, value, authored in zip(
        *(rows[column].to_pylist() for column in ("subject_id", "name", "types", "value")),
        rows["authored"].to_pylist(),
        strict=True,
    ):
        kinds = tuple(types or ())
        if not authored:
            found[subject] = SpannerNode(value=value or None, name=name or None, types=kinds)
        elif kinds:
            found[subject] = SpannerNode(value=subject, name=name or None, types=kinds)
        elif value:
            found[subject] = SpannerNode(value=value, name=None, types=())
    return found


def release_origin(loaded: LoadedRelease) -> ReleaseOrigin:
    """`origin_view` over a published release. Raises `ViewRefused` for a release that lacks
    a part the view reads."""
    try:
        series, edges, nodes = (loaded.tables[name] for name in ("series", "edges", "nodes"))
    except KeyError as exc:
        raise ViewRefused(f"the release holds no {exc.args[0]} part") from exc
    return origin_view(series=series, edges=edges, nodes=nodes)


__all__ = ["ReleaseOrigin", "origin_view", "release_origin"]
