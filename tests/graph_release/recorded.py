"""The recorded `RawInputs` of `tests/fixtures/graph_release/raw_inputs/raw_inputs.json`, the
platform parts built from them, and the inputs that build the slice's DFO parts.

`SLICE_DECLARATIONS` cuts the declarations to what the twenty-variable slice holds: the CRS
dataset through `crsTypeOfFlow`, the only code enum the slice read, and no listed dataset. The
slice reads no party nodes, so `slice_platform_parts` adds its CRS rows' parties as dev's
Spanner holds them: two by `typeOf` and `name` edges, one country as a placeholder `Node` row
an edge names, and one country not at all. `SLICE_BASE_PARTIES` answers the two countries as
the mixer does.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Final

import pyarrow as pa

from artifact_publication.jobs import ImportKey
from graph_release import reads
from graph_release.authored import AUTHORED, literal_id
from graph_release.declarations import DECLARATIONS
from graph_release.dfo import read_base_parties
from graph_release.format import conform_part
from graph_release.platform import PlatformParts, build_platform_parts
from graph_release.reads import RawInputs

FIXTURES = Path(__file__).parents[1] / "fixtures/graph_release"

_NOW = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
_HISTORY_ROW = {
    "WorkflowExecutionID": "wf-1",
    "Status": "SUCCESS",
    "IngestionFailure": False,
    "CreationTimestamp": _NOW,
    "CompletionTimestamp": _NOW,
}


def _table(recorded: dict, name: str, schema: pa.Schema) -> pa.Table:
    table = recorded["tables"][name]
    columns = list(zip(*table["rows"], strict=True))
    arrays = []
    for field, values in zip(schema, columns, strict=True):
        if pa.types.is_timestamp(field.type):
            values = [None if v is None else datetime.fromisoformat(v) for v in values]
        arrays.append(pa.array(list(values), field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


@cache
def raw_inputs() -> RawInputs:
    recorded = json.loads((FIXTURES / "raw_inputs/raw_inputs.json").read_text())
    return RawInputs(
        import_key=ImportKey.from_rows(None, _HISTORY_ROW, _HISTORY_ROW),
        series_summary=_table(recorded, "series_summary", reads.RAW_SERIES),
        facets=_table(recorded, "facets", reads.RAW_FACETS),
        live_variables=_table(recorded, "live_variables", reads.RAW_LIVE_VARIABLES),
        edges_raw=_table(recorded, "edges_raw", reads.RAW_EDGES),
        nodes_raw=_table(recorded, "nodes_raw", reads.RAW_NODES),
        imports=_table(recorded, "imports", reads.RAW_IMPORTS),
    )


@cache
def platform_parts() -> PlatformParts:
    return build_platform_parts(raw_inputs(), authored=AUTHORED)


_CRS: Final[str] = "provenance/OECDCreditorReportingSystem"

SLICE_DECLARATIONS = dataclasses.replace(
    DECLARATIONS,
    code_systems=tuple(
        dataclasses.replace(config, default=None)
        for config in DECLARATIONS.code_systems
        if config.code_system == "crsTypeOfFlow"
    ),
    datasets=tuple(dataset for dataset in DECLARATIONS.datasets if dataset.provenance == _CRS),
    listed_datasets=(),
    party_kind_overrides={},
    total_groups={},
    measure_axes=(),
    measure_partition_labels=(),
    measure_nouns=(),
    shape_slot_defaults=(),
    partly_stated_values=(),
)

RELEASE_PARTIES: Final[Mapping[str, tuple[str, str]]] = {
    "AsianDevelopmentBank": ("Asian Development Bank", "Organization"),
    "DACAsiaUnallocated": ("Asia, regional", "DACRecipientUnallocatedByCountry"),
}
"""The slice's parties the release holds, by `typeOf` and `name` edges: name and type."""

PLACEHOLDER_PARTY: Final[str] = "country/KGZ"
"""A country DFO's Spanner holds only as a placeholder `Node` row (its dcid as name, typed
`Thing`), there because an edge names it."""

UNHELD_PARTY: Final[str] = "country/TJK"
"""A country with no `Node` row, which therefore no kept edge names."""

MIXER_PARTIES: Final[Mapping[str, dict[str, tuple[str, ...]]]] = {
    "country/KGZ": {"typeOf": ("Country",), "name": ("Kyrgyzstan",)},
    "country/TJK": {"typeOf": ("Country",), "name": ("Tajikistan",)},
}
"""The slice's parties only the mixer answers."""


class RecordedMixer:
    """Answers `node_arcs_many` from `MIXER_PARTIES`, `None` for any other dcid."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self.calls.append(tuple(dcids))
        return {
            dcid: None
            if (found := MIXER_PARTIES.get(dcid)) is None
            else {prop: found.get(prop, ()) for prop in properties}
            for dcid in dcids
        }


SLICE_BASE_PARTIES: Final[bytes] = read_base_parties(sorted(MIXER_PARTIES), graph=RecordedMixer())


_PROVENANCE: Final[str] = "provenance/DFOOntology"


@cache
def slice_platform_parts() -> PlatformParts:
    """`platform_parts` with `RELEASE_PARTIES` held by `typeOf` and `name` edges, and
    `PLACEHOLDER_PARTY` as a placeholder `Node` row the first of them names."""
    parts = platform_parts()
    nodes: list[tuple[str, str | None, list[str], str | None]] = []
    edges: list[tuple[str, str, str, str]] = []
    for party, (name, kind) in RELEASE_PARTIES.items():
        nodes += [
            (party, name, [kind], None),
            (kind, kind, ["Class"], None),
            (literal_id(name), "", [], name),
        ]
        edges += [
            (party, "typeOf", kind, _PROVENANCE),
            (party, "name", literal_id(name), _PROVENANCE),
        ]
    nodes.append((PLACEHOLDER_PARTY, PLACEHOLDER_PARTY, ["Thing"], None))
    edges.append((next(iter(RELEASE_PARTIES)), "memberOf", PLACEHOLDER_PARTY, _PROVENANCE))

    node_rows = pa.table(
        {
            "subject_id": [row[0] for row in nodes],
            "name": [row[1] for row in nodes],
            "types": pa.array([row[2] for row in nodes], pa.list_(pa.string())),
            "value": pa.array([row[3] for row in nodes], pa.string()),
            "authored": [False] * len(nodes),
        }
    )
    edge_rows = pa.table(
        {
            column: [row[index] for row in edges]
            for index, column in enumerate(("subject_id", "predicate", "object_id", "provenance"))
        }
    )
    return dataclasses.replace(
        parts,
        nodes=conform_part(
            pa.concat_tables([parts.nodes.cast(node_rows.schema), node_rows]), name="nodes"
        ),
        edges=conform_part(
            pa.concat_tables([parts.edges.cast(edge_rows.schema), edge_rows]), name="edges"
        ),
    )
