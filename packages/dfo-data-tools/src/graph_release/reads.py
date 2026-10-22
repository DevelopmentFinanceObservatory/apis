"""The builder's Spanner statements and `read_inputs`, which runs them in one snapshot.

Every statement runs in the snapshot the caller opens, so the release reads one database state.
The first re-reads the import key. The rest read what the platform parts copy. Rows stream into
Arrow record batches of `CHUNK_ROWS`, so no read holds a whole table as Python rows. Each
statement carries a request tag, `app=graph-release,action=<read>`, so Spanner's query
statistics name it.

The statements copy Spanner and apply no rule. `graph_release.platform` decides what the parts
keep. `google-cloud-spanner` (the `graph-release` extra) is imported only to type array
parameters, so a test can pass any object with `execute_sql`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import pyarrow as pa
import pyarrow.compute as pc

from artifact_publication.jobs import ImportKey, read_import_key
from code_index.declarations import CODE_SYSTEMS
from graph_release.format import MASK_EPOCH, MASK_WORDS

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

CHUNK_ROWS = 200_000
"""Rows per record batch while a statement streams. A module variable so a test can lower it."""

_TAG: Final[str] = "app=graph-release"

_KEY = pa.dictionary(pa.int32(), pa.string())


def _mask_columns() -> str:
    """`period_mask_<w>`: bit `b` of word `w` set when the series has an observation in year
    `epoch + 64w + b`. Annual periods only. `graph_release.platform` refuses a facet with another
    `observation_period`. The shift reaches bit 63, so a word is a signed INT64 as Spanner stores
    it.

    `mask_years` counts the distinct years inside the mask, which the words' popcount must
    equal. `outside_mask` is whether any observation's year falls outside the mask or does not
    parse."""
    epoch = date.fromisoformat(MASK_EPOCH).year
    last = epoch + 64 * MASK_WORDS - 1
    year = "SAFE_CAST(SUBSTR(o.date, 1, 4) AS INT64)"
    columns = []
    for word in range(MASK_WORDS):
        first = epoch + 64 * word
        columns.append(
            f"BIT_OR(IF({year} BETWEEN {first} AND {first + 63}, "
            f"CAST(1 AS INT64) << ({year} - {first}), 0)) AS period_mask_{word}"
        )
    columns.append(
        f"COUNT(DISTINCT IF({year} BETWEEN {epoch} AND {last}, {year}, NULL)) AS mask_years"
    )
    columns.append(
        f"LOGICAL_OR(IFNULL({year} NOT BETWEEN {epoch} AND {last}, TRUE)) AS outside_mask"
    )
    return ",\n  ".join(columns)


# One row per `TimeSeries` row with observations: its key, `entity2` and `entity3`, and the
# summary of its `Observation` rows. `Observation` is interleaved in `TimeSeries` on the series
# key, so the join runs within each split and the plan holds no hash join.
SERIES_SQL: Final[str] = f"""SELECT t.variable_measured, t.entity1, t.extra_entities_id, t.facet_id,
  t.entity2, t.entity3,
  COUNT(*) AS observation_count, MIN(o.date) AS earliest_date, MAX(o.date) AS latest_date,
  {_mask_columns()}
FROM TimeSeries t
JOIN Observation o
  ON o.variable_measured = t.variable_measured AND o.entity1 = t.entity1
  AND o.extra_entities_id = t.extra_entities_id AND o.facet_id = t.facet_id
GROUP BY t.variable_measured, t.entity1, t.extra_entities_id, t.facet_id, t.entity2, t.entity3"""

# Each facet's JSON twice, so `graph_release.platform` can refuse a facet whose series disagree.
FACETS_SQL: Final[str] = """SELECT facet_id, MIN(TO_JSON_STRING(facet)) AS facet_min,
  MAX(TO_JSON_STRING(facet)) AS facet_max
FROM TimeSeries
GROUP BY facet_id"""

LIVE_VARIABLES_SQL: Final[str] = "SELECT DISTINCT variable_measured FROM TimeSeries"

# The live variables' edges, each with whether its target has a typed `Node` row. The typed
# targets (measure values, enum members, properties, other variables) are read next.
VARIABLE_EDGES_SQL: Final[str] = """SELECT e.subject_id, e.predicate, e.object_id, e.provenance,
  ARRAY_LENGTH(n.types) > 0 AS target_typed
FROM UNNEST(@subjects) AS v
JOIN@{JOIN_METHOD=APPLY_JOIN} Edge@{FORCE_INDEX=_BASE_TABLE} e ON e.subject_id = v
LEFT JOIN@{JOIN_METHOD=APPLY_JOIN} Node n ON n.subject_id = e.object_id"""

# Every edge of every member of the code enums.
CODE_EDGES_SQL: Final[str] = """SELECT e.subject_id, e.predicate, e.object_id, e.provenance
FROM Edge@{FORCE_INDEX=InEdge} t
JOIN@{JOIN_METHOD=APPLY_JOIN} Edge@{FORCE_INDEX=_BASE_TABLE} e ON e.subject_id = t.subject_id
WHERE t.object_id IN UNNEST(@enums) AND t.predicate = 'typeOf'"""

# The `specializationOf` parents of `@codes`, one level up. `read_inputs` runs it again on the
# parents it has not read yet, until a level has none.
PARENTS_SQL: Final[str] = """SELECT e.subject_id, e.object_id
FROM UNNEST(@codes) AS v
JOIN@{JOIN_METHOD=APPLY_JOIN} Edge@{FORCE_INDEX=_BASE_TABLE} e
  ON e.subject_id = v AND e.predicate = 'specializationOf'"""

# The `AgentSkill` nodes' edges the skills index reads.
SKILL_EDGES_SQL: Final[str] = """SELECT e.subject_id, e.predicate, e.object_id, e.provenance
FROM Edge@{FORCE_INDEX=InEdge} t
JOIN@{JOIN_METHOD=APPLY_JOIN} Edge@{FORCE_INDEX=_BASE_TABLE} e ON e.subject_id = t.subject_id
WHERE t.object_id = 'AgentSkill' AND t.predicate = 'typeOf'
  AND e.predicate IN ('typeOf', 'skillAppliesTo', 'skillAppliesToProvenance',
    'agentSkillReferences', 'url')"""

TARGET_EDGES_SQL: Final[str] = """SELECT e.subject_id, e.predicate, e.object_id, e.provenance
FROM UNNEST(@subjects) AS v
JOIN@{JOIN_METHOD=APPLY_JOIN} Edge@{FORCE_INDEX=_BASE_TABLE} e ON e.subject_id = v"""

NODES_SQL: Final[str] = """SELECT subject_id, value, name, types FROM Node
WHERE subject_id IN UNNEST(@ids)"""

IMPORTS_SQL: Final[str] = """SELECT ImportName, LatestVersion, State, DataImportTimestamp
FROM ImportStatus"""

# The imports one ingestion run wrote, for the manifest's `ingested_imports`.
INGESTED_IMPORTS_SQL: Final[str] = """SELECT IngestedImports FROM IngestionHistory
WHERE WorkflowExecutionID = @execution"""

CODE_ENUMS: Final[tuple[str, ...]] = tuple(system.enum_dcid for system in CODE_SYSTEMS)

RAW_SERIES = pa.schema(
    [
        ("variable_measured", _KEY),
        ("entity1", _KEY),
        ("extra_entities_id", _KEY),
        ("facet_id", _KEY),
        ("entity2", _KEY),
        ("entity3", _KEY),
        ("observation_count", pa.int64()),
        # `Observation.date` as stored: `YYYY`, `YYYY-MM` or `YYYY-MM-DD`.
        ("earliest_date", pa.string()),
        ("latest_date", pa.string()),
        *((f"period_mask_{word}", pa.int64()) for word in range(MASK_WORDS)),
        # Read for `graph_release.platform`'s checks; the `series` part does not carry them.
        ("mask_years", pa.int64()),
        ("outside_mask", pa.bool_()),
    ]
)
RAW_FACETS = pa.schema(
    [("facet_id", pa.string()), ("facet_min", pa.string()), ("facet_max", pa.string())]
)
RAW_LIVE_VARIABLES = pa.schema([("variable_measured", pa.string())])
RAW_EDGES = pa.schema(
    [
        ("subject_id", pa.string()),
        ("predicate", pa.string()),
        ("object_id", pa.string()),
        ("provenance", pa.string()),
    ]
)
RAW_NODES = pa.schema(
    [
        ("subject_id", pa.string()),
        ("value", pa.string()),
        ("name", pa.string()),
        ("types", pa.list_(pa.string())),
    ]
)
RAW_IMPORTS = pa.schema(
    [
        ("import_name", pa.string()),
        ("version", pa.string()),
        ("state", pa.string()),
        ("data_import_timestamp", pa.timestamp("us", tz="UTC")),
    ]
)
_VARIABLE_EDGES = RAW_EDGES.append(pa.field("target_typed", pa.bool_()))
_PARENTS = pa.schema([("child", pa.string()), ("parent", pa.string())])


@dataclass(frozen=True)
class RawInputs:
    """What one snapshot read, one table per read in the `RAW_*` schemas.

    `import_key` is the key the snapshot re-read first, `None` when `IngestionHistory` is
    empty. `edges_raw` is every edge read, deduplicated: the live variables', the code enum
    members', the `AgentSkill` nodes', and those of the typed targets and constraint-value
    ancestors the variables reach. `nodes_raw` holds the `Node` rows of every subject and object
    in `edges_raw` and of every live variable.
    """

    import_key: ImportKey | None
    series_summary: pa.Table
    facets: pa.Table
    live_variables: pa.Table
    edges_raw: pa.Table
    nodes_raw: pa.Table
    imports: pa.Table


class Snapshot(Protocol):
    """The part of a `google.cloud.spanner` snapshot the reads use. `execute_sql` yields each row
    as a sequence in the statement's column order."""

    def execute_sql(
        self,
        sql: str,
        params: Mapping[str, Any] | None = None,
        param_types: Mapping[str, Any] | None = None,
        *,
        request_options: Mapping[str, str] | None = None,
    ) -> Iterable[Sequence[object]]: ...


def snapshot_at(database: Any, *, read_timestamp: datetime) -> AbstractContextManager[Snapshot]:
    """A multi-use read-only snapshot of `database` (a `google.cloud.spanner` `Database`) at
    `read_timestamp`. Raises `ValueError` for a naive timestamp, which the client would send
    as-is."""
    if read_timestamp.tzinfo is None:
        raise ValueError(f"read_timestamp must be timezone-aware: {read_timestamp!r}")
    return database.snapshot(read_timestamp=read_timestamp, multi_use=True)


def read_inputs(snapshot: Snapshot) -> RawInputs:
    """Runs every statement in `snapshot`, which must be multi-use: the import key's three, then
    the live variables, the edge reads, the nodes, facets, series and imports."""
    import_key = read_import_key(_Tagged(snapshot, action="import_key"))

    live = _stream(snapshot, LIVE_VARIABLES_SQL, action="live_variables", schema=RAW_LIVE_VARIABLES)
    live_ids = cast("list[str]", live["variable_measured"].to_pylist())

    variable_edges = _stream(
        snapshot,
        VARIABLE_EDGES_SQL,
        action="variable_edges",
        schema=_VARIABLE_EDGES,
        params={"subjects": live_ids},
    )
    code_edges = _stream(
        snapshot,
        CODE_EDGES_SQL,
        action="code_edges",
        schema=RAW_EDGES,
        params={"enums": CODE_ENUMS},
    )
    skill_edges = _stream(snapshot, SKILL_EDGES_SQL, action="skill_edges", schema=RAW_EDGES)
    children = _specialized(snapshot, _constraint_values(variable_edges))
    read_subjects = pa.chunked_array(
        [
            *variable_edges["subject_id"].chunks,
            *code_edges["subject_id"].chunks,
            *skill_edges["subject_id"].chunks,
        ],
        pa.string(),
    )
    typed = cast("pa.ChunkedArray", pc.fill_null(variable_edges["target_typed"], pa.scalar(False)))
    typed_targets = variable_edges.filter(typed)
    targets = _unique([typed_targets["object_id"], children], excluding=read_subjects)
    target_edges = _stream(
        snapshot,
        TARGET_EDGES_SQL,
        action="target_edges",
        schema=RAW_EDGES,
        params={"subjects": targets},
    )
    edges = _distinct(
        pa.concat_tables(
            [variable_edges.drop_columns(["target_typed"]), code_edges, skill_edges, target_edges]
        )
    )

    node_ids = _unique([edges["subject_id"], edges["object_id"], live_ids])
    nodes = _stream(snapshot, NODES_SQL, action="nodes", schema=RAW_NODES, params={"ids": node_ids})
    facets = _stream(snapshot, FACETS_SQL, action="facets", schema=RAW_FACETS)
    series = _stream(snapshot, SERIES_SQL, action="series", schema=RAW_SERIES)
    imports = _stream(snapshot, IMPORTS_SQL, action="imports", schema=RAW_IMPORTS)
    return RawInputs(
        import_key=import_key,
        series_summary=series,
        facets=facets,
        live_variables=live,
        edges_raw=edges,
        nodes_raw=nodes,
        imports=imports,
    )


def read_ingested_imports(snapshot: Snapshot, *, execution_id: str) -> tuple[str, ...]:
    """`IngestionHistory.IngestedImports` of the run `execution_id`, in stored order, empty when
    the row holds none. Raises `LookupError` when no row has that id."""
    from google.cloud.spanner_v1 import param_types as types

    rows = snapshot.execute_sql(
        INGESTED_IMPORTS_SQL,
        params={"execution": execution_id},
        param_types={"execution": types.STRING},
        request_options=_request_options("ingested_imports"),
    )
    for (imports,) in rows:
        return tuple(str(name) for name in cast("Sequence[object]", imports or ()))
    raise LookupError(f"IngestionHistory has no row {execution_id!r}")


def _constraint_values(variable_edges: pa.Table) -> list[str]:
    """The objects of every property a variable names in its `constraintProperties`, sorted."""
    named = variable_edges.filter(
        pc.equal(variable_edges["predicate"], pa.scalar("constraintProperties"))
    )
    constrained = named.select(["subject_id", "object_id"]).rename_columns(
        ["subject_id", "predicate"]
    )
    values = variable_edges.select(["subject_id", "predicate", "object_id"]).join(
        constrained, keys=["subject_id", "predicate"], join_type="inner"
    )
    return _unique([values["object_id"]])


def _specialized(snapshot: Snapshot, values: list[str]) -> list[str]:
    """Every code in `values`, or above one in the `specializationOf` chain, that has a parent,
    sorted. Reads one level per statement, each a level's parents not read before, so the walk
    reaches any depth and ends at a cycle."""
    children: set[str] = set()
    read = set(values)
    level = values
    while level:
        edges = _stream(
            snapshot,
            PARENTS_SQL,
            action="specialization_parents",
            schema=_PARENTS,
            params={"codes": level},
        )
        children.update(cast("list[str]", edges["child"].to_pylist()))
        level = sorted(set(cast("list[str]", edges["parent"].to_pylist())) - read)
        read.update(level)
    return sorted(children)


class _Tagged:
    """A snapshot whose `execute_sql(sql)` carries the request tag of `action`, for
    `read_import_key`."""

    def __init__(self, snapshot: Snapshot, *, action: str) -> None:
        self._snapshot = snapshot
        self._action = action

    def execute_sql(self, sql: str) -> Iterable[Sequence[object]]:
        return self._snapshot.execute_sql(sql, request_options=_request_options(self._action))


def _request_options(action: str) -> dict[str, str]:
    return {"request_tag": f"{_TAG},action={action}"}


def _stream(
    snapshot: Snapshot,
    sql: str,
    *,
    action: str,
    schema: pa.Schema,
    params: Mapping[str, Sequence[str]] | None = None,
) -> pa.Table:
    """Runs `sql` and collects its rows, in `schema`'s column order, into record batches of
    `CHUNK_ROWS`. Each parameter is an array of strings."""
    spanner_params = param_types = None
    if params is not None:
        from google.cloud.spanner_v1 import param_types as types

        spanner_params = {name: list(values) for name, values in params.items()}
        param_types = {name: types.Array(types.STRING) for name in params}
    rows = snapshot.execute_sql(
        sql,
        params=spanner_params,
        param_types=param_types,
        request_options=_request_options(action),
    )

    batches = []
    columns: list[list[object]] = [[] for _ in schema]
    for row in rows:
        for column, value in zip(columns, row, strict=True):
            column.append(value)
        if len(columns[0]) >= CHUNK_ROWS:
            batches.append(_batch(columns, schema))
    if columns[0] or not batches:
        batches.append(_batch(columns, schema))
    return pa.Table.from_batches(batches, schema=schema)


def _batch(columns: list[list[object]], schema: pa.Schema) -> pa.RecordBatch:
    batch = pa.RecordBatch.from_arrays(
        [pa.array(values, type=field.type) for values, field in zip(columns, schema, strict=True)],
        schema=schema,
    )
    for values in columns:
        values.clear()
    return batch


def _unique(
    columns: Sequence[pa.ChunkedArray | pa.Array | Sequence[str]],
    *,
    excluding: pa.ChunkedArray | None = None,
) -> list[str]:
    """The distinct non-null strings of `columns`, sorted, less those in `excluding`."""
    chunks = []
    for column in columns:
        if isinstance(column, pa.ChunkedArray):
            chunks.extend(column.chunks)
        elif isinstance(column, pa.Array):
            chunks.append(column)
        else:
            chunks.append(pa.array(column, pa.string()))
    values = pc.unique(pa.chunked_array(chunks, pa.string())).drop_null()
    if excluding is not None:
        values = values.filter(pc.invert(pc.is_in(values, value_set=pc.unique(excluding))))
    return cast("list[str]", values.take(pc.sort_indices(values)).to_pylist())


def _distinct(edges: pa.Table) -> pa.Table:
    return edges.group_by(edges.column_names, use_threads=False).aggregate([])


__all__ = [
    "CHUNK_ROWS",
    "CODE_EDGES_SQL",
    "CODE_ENUMS",
    "FACETS_SQL",
    "IMPORTS_SQL",
    "INGESTED_IMPORTS_SQL",
    "LIVE_VARIABLES_SQL",
    "NODES_SQL",
    "PARENTS_SQL",
    "RAW_EDGES",
    "RAW_FACETS",
    "RAW_IMPORTS",
    "RAW_LIVE_VARIABLES",
    "RAW_NODES",
    "RAW_SERIES",
    "SERIES_SQL",
    "SKILL_EDGES_SQL",
    "TARGET_EDGES_SQL",
    "VARIABLE_EDGES_SQL",
    "RawInputs",
    "Snapshot",
    "read_ingested_imports",
    "read_inputs",
    "snapshot_at",
]
